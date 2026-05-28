# ============================================================
# SurfTrack/__init__.py
# ============================================================
"""
SurfTrack — one layer (2-D + time) tracker.

Tracks geospatial features in 3-D data (time, lat, lon) using
morphological operations, connected-component labelling, and
area filtering.

Public API
----------
    from ocetrac.SurfTrack import SurfTracker
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from .tracker import (
    apply_mask,
    filter_area,
    label_3d,
    label_temporal_neighbor,
    morphological_operations,
    wrap_labels,
)

from ..preprocessing.preprocessing import clean_binary

__all__ = ["SurfTracker"]


class SurfTracker:
    """
    End-to-end surface marine heatwave tracker.

    Operates on 3-D data (time, ydim, xdim).  Each step stores its
    output as an attribute so intermediate results can be inspected.

    Pipeline
    --------
    clean → filter_area → track → postprocess

    Parameters
    ----------
    da                : DataArray (time, ydim, xdim)
        Anomaly or thresholded feature field.
    mask              : DataArray (ydim, xdim)
        Binary ocean mask — 1 = valid ocean cell, 0 = land/ignored.
    radius            : int
        Disk radius for morphological close→open.  ``radius=1`` is a
        no-op.  Larger values fill wider gaps but risk bridging separate
        events.  Default 2.
    min_size_quartile : float
        Relative area percentile threshold (0–1).  Combined with
        ``min_area_cells`` via ``max()``.  Default 0.25.
    min_area_cells    : int
        Absolute minimum object area in grid cells.  Objects smaller
        than this are always removed regardless of the percentile.
        Default 100.
    timedim           : str
        Name of the time dimension.  Default ``"time"``.
    xdim              : str
        Name of the x (longitude) dimension.  Default ``"nlon"``.
    ydim              : str
        Name of the y (latitude) dimension.  Default ``"nlat"``.
    positive          : bool
        ``True`` → track warm anomalies (>0).
        ``False`` → track cold anomalies (<0).
        Default ``True``.
    method            : str
        Labelling method. ``"3d"`` (default) runs connected-component
        labelling on the full (time, lat, lon) volume. 
        ``"temporal_neighbor"`` labels each 2-D frame independently
        and only links blobs between adjacent frames if they spatially
        overlap. A gap of even one timestep starts a new event.
    contain_thresh    : float
        Only used when ``method="temporal_neighbor"``. Containment
        threshold for merging two blobs using
        max(|A∩B|/|A|, |A∩B|/|B|) >= contain_thresh.
        0.0 means any spatial overlap is sufficient. Default 0.0.
    """

    def __init__(
        self,
        da:                xr.DataArray,
        mask:              xr.DataArray,
        *,
        radius:            int   = 2,
        min_size_quartile: float = 0.25,
        min_area_cells:    int   = 100,
        timedim:           str   = "time",
        xdim:              str   = "nlon",
        ydim:              str   = "nlat",
        positive:          bool  = True,
        method:            str         = "3d",
        contain_thresh:    float = 0.0,
    ) -> None:
        _valid_methods = {"3d", "temporal_neighbor"}
        if method not in _valid_methods:
            raise ValueError(
                f"method={method!r} not recognised. "
                f"Choose from {_valid_methods}."
            )
        self.da                = da
        self.mask              = mask
        self.radius            = radius
        self.min_size_quartile = min_size_quartile
        self.min_area_cells    = min_area_cells
        self.timedim           = timedim
        self.xdim              = xdim
        self.ydim              = ydim
        self.positive          = positive
        self.method            = method
        self.contain_thresh = contain_thresh

        # Intermediate state
        self.binary_clean:    xr.DataArray | None = None
        self.binary_filtered: xr.DataArray | None = None
        self.area:            xr.DataArray | None = None
        self.min_area:        float | None        = None
        self.N_initial:       int | None          = None
        self.labels_raw:      np.ndarray | None   = None
        self.result:          xr.DataArray | None = None

    # ── Step 1 ───────────────────────────────────────────────────────────────
    def clean(self) -> "SurfTracker":
        """
        Binarise the input field, apply morphological close→open per
        (lat, lon) slice, then apply the ocean mask.

        Outputs
        -------
        self.binary_clean : DataArray — binary field after cleaning and masking
        """
        if (self.mask == 0).all():
            raise ValueError(
                "Found only zeros in `mask`. "
                "Mask should have 1 = valid ocean cells."
            )

        print("Step 1 · morphological cleaning …")
        binary = morphological_operations(
            self.da, self.radius, self.xdim, self.ydim, self.positive
        )
        self.binary_clean = apply_mask(binary, self.mask)
        frac = float(self.binary_clean.values.mean())
        print(f"    fraction flagged = {frac:.4f}  "
              f"({'OK' if frac > 0 else 'WARNING: nothing flagged'})")
        return self

    # ── Step 2 ───────────────────────────────────────────────────────────────
    def filter(self) -> "SurfTracker":
        """
        Label 2-D slices, make IDs consecutive across time, wrap across
        the date line, then filter by area using:

            max(min_area_cells, percentile(areas, min_size_quartile))

        Outputs
        -------
        self.binary_filtered : DataArray — binary field with small objects removed
        self.area            : DataArray — area of every detected object
        self.min_area        : float     — effective threshold applied
        self.N_initial       : int       — object count before filtering
        """
        if self.binary_clean is None:
            self.clean()
        print("Step 2 · area filtering …")
        self.area, self.min_area, self.binary_filtered, self.N_initial = filter_area(
            self.binary_clean,
            xdim              = self.xdim,
            ydim              = self.ydim,
            min_size_quartile = self.min_size_quartile,
            min_area_cells    = self.min_area_cells,
        )
        return self

    # ── Step 3 ───────────────────────────────────────────────────────────────
    def track(self) -> "SurfTracker":
        """
        Label connected objects and wrap across the date line.

        Uses method='3d' (full volume CCL) or method='temporal_neighbor'
        (strict adjacent-timestep linking). See class parameters for details.

        Outputs
        -------
        self.labels_raw : ndarray — integer labels before postprocessing
        """
        if self.binary_filtered is None:
            self.filter()
        print(f"Step 3 · labelling (method={self.method!r}) …")

        if self.method == "temporal_neighbor":
            labels, num = label_temporal_neighbor(
                self.binary_filtered,
                contain_thresh=self.contain_thresh,
            )
        else:
            labels, num = label_3d(self.binary_filtered, connectivity=3)

        grid_res = abs(self.da[self.xdim][1] - self.da[self.xdim][0])
        if self.da[self.xdim][-1] - self.da[self.xdim][0] >= 360 - grid_res:
            self.labels_raw, N_final = wrap_labels(labels)
        else:
            self.labels_raw = labels
            N_final         = int(np.max(labels))

        print(f"    initial objects : {self.N_initial}")
        print(f"    final objects   : {N_final}")
        return self

    # ── Step 4 ───────────────────────────────────────────────────────────────
    def postprocess(self) -> "SurfTracker":
        """
        Wrap the tracked array as an ``xr.DataArray`` with NaN background
        and attach tracking diagnostics as attributes.

        Outputs
        -------
        self.result : DataArray — final event labels, NaN = background
        """
        if self.labels_raw is None:
            raise RuntimeError("Call .track() before .postprocess()")
        print("Step 4 · wrapping result …")

        result = xr.DataArray(
            self.labels_raw, dims=self.da.dims, coords=self.da.coords
        )
        result = result.where(result != 0, drop=False, other=np.nan)

        # Diagnostics
        N_final      = self.n_events()
        sum_tot      = int(np.sum(self.area.values))
        pct_reject   = (int(np.sum(
            self.area.where(self.area <= self.min_area, drop=True).values
        )) / sum_tot)
        pct_accept   = 1.0 - pct_reject

        result = result.rename("labels")
        result.attrs["initial objects identified"] = self.N_initial
        result.attrs["final objects tracked"]      = N_final
        result.attrs["radius"]                     = self.radius
        result.attrs["size quantile threshold"]    = self.min_size_quartile
        result.attrs["min area cells"]             = self.min_area_cells
        result.attrs["min area (effective)"]       = self.min_area
        result.attrs["percent area reject"]        = pct_reject
        result.attrs["percent area accept"]        = pct_accept
        result.attrs["method"]          = self.method
        result.attrs["contain_thresh"]  = self.contain_thresh

        self.result = result
        return self

    # ── Full pipeline ─────────────────────────────────────────────────────────
    def run(self) -> xr.DataArray:
        """
        Execute the full pipeline:
        clean → filter → track → postprocess

        Returns
        -------
        result : DataArray (time, ydim, xdim) — event labels, NaN = background
        """
        return (
            self.clean()
                .filter()
                .track()
                .postprocess()
                .result
        )

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def n_events(self) -> int:
        """Number of unique tracked events in the final result."""
        arr = self.labels_raw if self.result is None else self.result.values
        if arr is None:
            raise RuntimeError("Run the tracker first.")
        flat = arr.ravel().astype(float)
        flat = flat[~np.isnan(flat) & (flat > 0)]
        return int(len(np.unique(flat)))

    def event_duration(self) -> dict[int, int]:
        """Return ``{event_id: n_timesteps_present}`` from the final result."""
        if self.result is None:
            raise RuntimeError("Run the tracker first.")
        arr = self.result.values.astype(float)
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
        print("SurfTracker — Result Summary")
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
        for k in ["radius", "min_area_cells", "min_size_quartile",
                  "positive", "method", "contain_thresh"]:
            print(f"    {k:<20} = {getattr(self, k)}")

    def __repr__(self) -> str:
        n = self.n_events() if self.result is not None else "(not run yet)"
        return (
            f"SurfTracker(shape={tuple(self.da.shape)}, "
            f"radius={self.radius}, "
            f"method={self.method!r}, "
            f"events={n})"
        )
