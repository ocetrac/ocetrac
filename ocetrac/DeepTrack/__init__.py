# ============================================================
# DeepTrack/__init__.py
# ============================================================
from __future__ import annotations

import numpy as np
import xarray as xr

from .tracker import (
    build_3d_objects,
    filter_area_2d_global_depth,
    filter_preserve_labels_global,
    label_2d_stack,
    relabel_2d,
    track_objects_with_splitting,
)
from .grid import make_anisotropic_struct
from ..preprocessing.preprocessing import clean_binary

__all__ = ["DeepTracker"]


def _checkpoint(name: str, arr) -> None:
    a    = arr.values if hasattr(arr, "values") else arr
    a    = np.asarray(a, dtype=float)
    flat = a.ravel()
    pos  = flat[~np.isnan(flat) & (flat > 0)]
    print(f"    [{name}]  shape={a.shape}  "
          f"non-zero voxels={pos.size:,}  "
          f"unique IDs={len(np.unique(pos))}")


def _wrap_longitude(arr: np.ndarray) -> np.ndarray:
    """
    Merge labels that straddle the date line per (t, z) slice.
 
    For each (t, z), checks nlon=0 and nlon=-1 at the same nlat position.
    If two different labels appear at the same y on opposite edges they are
    the same physical object split by the periodic boundary — the larger ID
    is replaced by the smaller ID everywhere in that (t, z) slice.
 
    Parameters
    ----------
    arr : int ndarray (time, z, nlat, nlon)
 
    Returns
    -------
    out : int ndarray — same shape, date-line objects merged
    """
    out = arr.copy()
    for t in range(out.shape[0]):
        for z in range(out.shape[1]):
            left_col  = out[t, z, :,  0].copy()
            right_col = out[t, z, :, -1].copy()
            for y in range(out.shape[2]):
                l = int(left_col[y])
                r = int(right_col[y])
                if l > 0 and r > 0 and l != r:
                    old_id = max(l, r)
                    new_id = min(l, r)
                    out[t, z][out[t, z] == old_id] = new_id
    return out

class DeepTracker:
    """
    4-D event tracker

    Parameters
    ----------
    da             : DataArray (time, z_t, nlat, nlon)
    radius         : int   — disk radius for morphological close→open
    min_area_cells : int   — absolute minimum 2-D blob area in grid cells
    min_quantile   : float — relative area-filter percentile (combined with
                     min_area_cells via max)
    contain_thresh : float — minimum containment score to link objects across time
    alpha          : float — voxel vs volume containment weight (0=volume, 1=voxel)
    frac_filter    : float — drop bottom fraction of 3-D objects by voxel count
                     before tracking
    connect_z      : bool  — vertical connectivity in 3-D labelling
    positive       : bool  — True → warm anomalies, False → cold
    n_z            : int   — number of depth levels to use
    wrap_lon       : bool  - True -> merge objects that straddle the dateline
                     Should be True for global grids (nlon spans 360°) and
                     False for regional domains. Default False
    """

    def __init__(
        self,
        da:             xr.DataArray,
        *,
        radius:         int   = 3,
        min_area_cells: int   = 200,
        min_quantile:   float = 0.25,
        contain_thresh: float = 0.3,
        alpha:          float = 0.5,
        frac_filter:    float = 0.25,
        connect_z:      bool  = True,
        positive:       bool  = True,
        n_z:            int   = 20,
        wrap_lon:       bool  = False,
    ) -> None:
        self.da             = da
        self.radius         = radius
        self.min_area_cells = min_area_cells
        self.min_quantile   = min_quantile
        self.contain_thresh = contain_thresh
        self.alpha          = alpha
        self.frac_filter    = frac_filter
        self.connect_z      = connect_z
        self.positive       = positive
        self.n_z            = n_z
        self.wrap_lon       = wrap_lon

        self.binary_clean:    xr.DataArray | None = None
        self.labeled_2d:      xr.DataArray | None = None
        self.labeled_3d:      np.ndarray   | None = None
        self.filtered_labels: np.ndarray   | None = None
        self.tracked:         np.ndarray   | None = None
        self.origin_map:      dict         | None = None
        self.result:          xr.DataArray | None = None

    # ── Step 1 ───────────────────────────────────────────────────────────────
    def clean(self) -> "DeepTracker":
        """Binarise and morphologically clean (close→open) the input field."""
        print("Step 1 · morphological cleaning …")
        self.binary_clean = clean_binary(
            self.da, radius=self.radius, positive=self.positive
        ).compute()
        frac = float(self.binary_clean.values.mean())
        print(f"    fraction flagged warm = {frac:.4f}  "
              f"({'OK' if frac > 0 else 'WARNING: nothing flagged'})")
        return self

    # ── Step 2 ───────────────────────────────────────────────────────────────
    def label(self) -> "DeepTracker":
        """
        Label each (t, z) slice with 2-D connected components, then
        apply longitude wrapping so objects that straddle the date line
        are merged before area filtering
        """
        if self.binary_clean is None:
            self.clean()
        print("Step 2 · 2-D connected-component labelling …")
        self.labeled_2d = label_2d_stack(self.binary_clean).compute()

        if self.wrap_lon:
            print("Step 2b · 2-D longitude wrap")
            n_before = len(np.unique(self.labeled_2d.values[self.labeled_2d.values > 0]))
            wrapped  = _wrap_longitude(self.labeled_2d.values)
            n_after  = len(np.unique(wrapped[wrapped > 0]))
            print(f"    merged {n_before - n_after} date-line objects  "
                  f"({n_before} → {n_after})")
            self.labeled_2d = xr.DataArray(
                wrapped,
                dims   = self.labeled_2d.dims,
                coords = self.labeled_2d.coords,
            )
            
        n_surf = [len(np.unique(self.labeled_2d.values[t, 0])) - 1
                  for t in range(self.labeled_2d.shape[0])]
        print(f"    surface blobs — mean={np.mean(n_surf):.1f}  "
              f"min={min(n_surf)}  max={max(n_surf)}")
        return self

    # ── Step 3 ───────────────────────────────────────────────────────────────
    def connect_depth(self) -> "DeepTracker":
        """
        Area filter → relabel → 3-D depth connectivity. Then longitude wrap.
        Longitude wrap is applied after 3-D labelling so that any objects
        reconnected through depth are also correctly merged across the dateline 
        before the global volume filter runs.
        """
        if self.labeled_2d is None:
            self.label()

        print("Step 3a · area filtering …")
        filtered_np = filter_area_2d_global_depth(
            self.labeled_2d.values,
            min_quantile   = self.min_quantile,
            min_area_cells = self.min_area_cells,
        )
        n_before = sum(len(np.unique(self.labeled_2d.values[t, z])) - 1
                       for t in range(self.labeled_2d.shape[0])
                       for z in range(self.labeled_2d.shape[1]))
        n_after  = sum(len(np.unique(filtered_np[t, z][filtered_np[t, z] > 0]))
                       for t in range(filtered_np.shape[0])
                       for z in range(filtered_np.shape[1]))
        print(f"    2-D blobs: {n_before:,} → {n_after:,}  "
              f"(removed {n_before - n_after:,})")
        if n_after == 0:
            print("    WARNING: all blobs removed — lower min_area_cells or min_quantile")

        print("Step 3b · 3-D depth connectivity …")
        relabeled_np = np.zeros_like(filtered_np, dtype=int)
        for t in range(filtered_np.shape[0]):
            for z in range(filtered_np.shape[1]):
                relabeled_np[t, z] = relabel_2d(filtered_np[t, z])

        struct = make_anisotropic_struct(connect_xy=True, connect_z=self.connect_z)
        self.labeled_3d = build_3d_objects(relabeled_np, struct)

        if self.wrap_lon:
            print("Step 3c · 3-D longitude wrap")
            n_before_wrap = sum(
                len(np.unique(self.labeled_3d[t][self.labeled_3d[t] > 0]))
                for t in range(self.labeled_3d.shape[0])
            )
            self.labeled_3d = _wrap_longitude(self.labeled_3d)
            n_after_wrap = sum(
                len(np.unique(self.labeled_3d[t][self.labeled_3d[t] > 0]))
                for t in range(self.labeled_3d.shape[0])
            )
            print(f"    merged {n_before_wrap - n_after_wrap} date-line objects  "
                  f"({n_before_wrap} → {n_after_wrap})")
            
        n_3d = [len(np.unique(self.labeled_3d[t][self.labeled_3d[t] > 0]))
                for t in range(self.labeled_3d.shape[0])]
        print(f"    3-D objects/timestep — mean={np.mean(n_3d):.1f}  "
              f"min={min(n_3d)}  max={max(n_3d)}  total={sum(n_3d):,}")
        return self

    # ── Step 4 ───────────────────────────────────────────────────────────────
    def prefilter(self) -> "DeepTracker":
        """Drop globally smallest 3-D objects by voxel count."""
        if self.labeled_3d is None:
            self.connect_depth()
        print("Step 4 · global volume filter …")
        n_before = sum(len(np.unique(self.labeled_3d[t][self.labeled_3d[t] > 0]))
                       for t in range(self.labeled_3d.shape[0]))
        self.filtered_labels = filter_preserve_labels_global(
            self.labeled_3d, frac=self.frac_filter
        )
        n_after = sum(len(np.unique(self.filtered_labels[t][self.filtered_labels[t] > 0]))
                      for t in range(self.filtered_labels.shape[0]))
        print(f"    3-D objects: {n_before:,} → {n_after:,}  "
              f"(removed {n_before - n_after:,})")
        return self

    # ── Step 5 ───────────────────────────────────────────────────────────────
    def track(self, cell_volume: np.ndarray | None = None) -> "DeepTracker":
        """Containment-based temporal tracking with lineage preservation."""
        if self.filtered_labels is None:
            self.prefilter()
        print("Step 5 · containment tracking …")
        self.tracked, self.origin_map = track_objects_with_splitting(
            self.filtered_labels,
            volume_weights = cell_volume,
            contain_thresh = self.contain_thresh,
            alpha          = self.alpha,
            plot_results   = False,
        )
        n_ids = len(np.unique(self.tracked[self.tracked > 0]))
        print(f"    unique event IDs assigned: {n_ids}")
        if n_ids == 0:
            print("    WARNING: 0 tracks — check filtered_labels has positive values")
        return self

    # ── Step 6 ───────────────────────────────────────────────────────────────
    def postprocess(self) -> "DeepTracker":
        """Wrap tracked array as an xr.DataArray result."""
        if self.tracked is None:
            raise RuntimeError("Call .track() before .postprocess()")
        print("Step 6 · wrapping result …")
        self.result = xr.DataArray(
            self.tracked.astype(float),
            dims   = self.da.dims,
            coords = self.da.coords,
            name   = "events_tracked",
        )
        # Replace 0 (background) with NaN
        self.result = self.result.where(self.result > 0)
        n = self.n_events()
        print(f"    final events: {n}")
        return self

    # ── Full pipeline ────────────────────────────────────────────────────────
    def run(self, cell_volume: np.ndarray | None = None) -> xr.DataArray:
        """
        Execute the full pipeline:
        clean → label → connect_depth → prefilter → track → postprocess

        Parameters
        ----------
        cell_volume : ndarray (z_t, nlat, nlon) — physical cell volume in m³,
                      used for volume-weighted containment in tracking

        Returns
        -------
        result : DataArray (time, z_t, nlat, nlon) with persistent event IDs,
                 NaN = background
        """
        return (
            self.clean()
                .label()
                .connect_depth()
                .prefilter()
                .track(cell_volume=cell_volume)
                .postprocess()
                .result
        )

    # ── Diagnostics ──────────────────────────────────────────────────────────
    def n_events(self) -> int:
        """Number of unique tracked events in the final result."""
        if self.result is None:
            raise RuntimeError("Run the tracker first.")
        arr  = self.result.values if hasattr(self.result, "values") else self.result
        flat = arr.ravel().astype(float)
        flat = flat[~np.isnan(flat) & (flat > 0)]
        return int(len(np.unique(flat)))

    def event_duration(self) -> dict[int, int]:
        """Return {event_id: n_timesteps_present} from the final result."""
        if self.result is None:
            raise RuntimeError("Run the tracker first.")
        arr = self.result.values if hasattr(self.result, "values") else self.result
        arr = arr.astype(float)
        duration: dict[int, int] = {}
        for t in range(arr.shape[0]):
            plane = arr[t].ravel()
            for fid in np.unique(plane[~np.isnan(plane)]):
                if fid > 0:
                    duration[int(fid)] = duration.get(int(fid), 0) + 1
        return duration

    def summary(self) -> None:
        """Print event count, duration distribution, and parameters."""
        if self.result is None:
            print("Not run yet — call .run() first.")
            return
        durations = self.event_duration()
        n         = len(durations)
        print("=" * 55)
        print("DeepTracker — Result Summary")
        print("=" * 55)
        print(f"  Input shape    : {tuple(self.da.shape)}")
        print(f"  Tracked events : {n}")
        if n > 0:
            durs = np.array(list(durations.values()))
            print(f"  Duration  min/median/max : "
                  f"{durs.min()} / {int(np.median(durs))} / {durs.max()}")
            for thr in [1, 3, 6, 12]:
                print(f"    >= {thr:2d} ts : {(durs >= thr).sum()}")
        print()
        print("  Parameters:")
        for k in ["radius", "min_area_cells", "min_quantile",
                  "frac_filter", "contain_thresh", "alpha", "connect_z", "wrap_lon"]:
            print(f"    {k:<16} = {getattr(self, k)}")
        print("=" * 55)

    def __repr__(self) -> str:
        n = self.n_events() if self.result is not None else "(not run yet)"
        return (
            f"DeepTracker(shape={tuple(self.da.shape)}, "
            f"radius={self.radius}, "
            f"contain_thresh={self.contain_thresh}, "
            f"wrap_lon={self.wrap_lon}, "
            f"events={n})"
        )
