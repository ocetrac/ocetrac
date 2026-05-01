# ============================================================
# DeepTrack/tracker.py
# ============================================================
"""
Core algorithms:

  2-D labelling      label_2d_stack, filter_area_2d_global_depth, relabel_2d
  3-D connectivity   build_3d_objects
  Volume filter      filter_preserve_labels_global
  Tracking           track_objects_with_splitting

---------------
TODO: _wrap method to handle longitude periodicity properly
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import ndimage


# ──────────────────────────────────────────────────────────────────────────────
# 2-D labelling
# ──────────────────────────────────────────────────────────────────────────────

def label_2d_stack(binary: xr.DataArray) -> xr.DataArray:
    """
    Connected-component labelling on each (t, z) slice independently.

    Dask-parallel via xr.apply_ufunc with vectorize=True.

    Parameters
    ----------
    binary : DataArray (time, z_t, nlat, nlon) — bool

    Returns
    -------
    labels : DataArray float, same shape (background = 0)
    """
    def _label_slice(slice_2d: np.ndarray) -> np.ndarray:
        lab, _ = ndimage.label(slice_2d.astype(bool))
        return lab.astype(float)

    return xr.apply_ufunc(
        _label_slice, binary,
        input_core_dims=[["nlat", "nlon"]],
        output_core_dims=[["nlat", "nlon"]],
        vectorize=True, dask="parallelized", output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )


def filter_area_2d_global_depth(
    arr_4d:         np.ndarray,
    min_quantile:   float = 0.25,
    min_area_cells: int   = 100,
) -> np.ndarray:
    """
    Remove small 2-D objects per depth level using an area distribution
    computed across ALL timesteps.

    Threshold = max(min_area_cells, percentile(all_areas_at_z, min_quantile×100))

    Using max() prevents the percentile from collapsing near zero when all
    blobs are similarly sized, ensuring tiny blobs are always removed.

    Parameters
    ----------
    arr_4d         : (time, z, nlat, nlon) — labelled integer array
    min_quantile   : float — relative percentile (0–1)
    min_area_cells : int   — absolute minimum area in grid cells

    Returns
    -------
    out : (time, z, nlat, nlon) ndarray — same dtype, small objects zeroed
    """
    T, Z, Y, X = arr_4d.shape
    out = arr_4d.copy()

    for z in range(Z):
        all_areas = []
        for t in range(T):
            labels, n = ndimage.label(arr_4d[t, z].astype(bool))
            if n == 0:
                continue
            areas = np.bincount(labels.ravel())[1:]
            if areas.size > 0:
                all_areas.append(areas)
        if not all_areas:
            continue

        all_areas_cat = np.concatenate(all_areas)
        pct_threshold = np.percentile(all_areas_cat, min_quantile * 100)
        threshold     = max(min_area_cells, pct_threshold)

        for t in range(T):
            labels, n = ndimage.label(arr_4d[t, z].astype(bool))
            if n == 0:
                out[t, z] = 0
                continue
            areas    = np.bincount(labels.ravel())[1:]
            keep_ids = np.where(areas >= threshold)[0] + 1
            out[t, z] = arr_4d[t, z] * np.isin(labels, keep_ids)

    return out


def relabel_2d(arr: np.ndarray) -> np.ndarray:
    """Re-run connected-component labelling on a binary 2-D slice."""
    labels, _ = ndimage.label(arr.astype(bool))
    return labels


# ──────────────────────────────────────────────────────────────────────────────
# 3-D depth connectivity
# ──────────────────────────────────────────────────────────────────────────────

def build_3d_objects(
    filtered_np: np.ndarray,
    structure:   np.ndarray,
) -> np.ndarray:
    """
    Apply 3-D connected-component labelling across (z, nlat, nlon) for every
    timestep independently.

    Parameters
    ----------
    filtered_np : (time, z_t, nlat, nlon) int array
    structure   : bool ndarray (3, 3, 3) — anisotropic structuring element

    Returns
    -------
    out : int ndarray (time, z_t, nlat, nlon)
          Labels reset to 1..N at each timestep independently.
    """
    out = np.zeros_like(filtered_np, dtype=int)
    for t in range(filtered_np.shape[0]):
        out[t], _ = ndimage.label(filtered_np[t].astype(bool), structure=structure)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Global volume filter
# ──────────────────────────────────────────────────────────────────────────────

def filter_preserve_labels_global(
    tracks: np.ndarray,
    frac:   float = 0.25,
) -> np.ndarray:
    """
    Remove the smallest frac fraction of 3-D objects by total voxel count
    summed across ALL timesteps. Original label IDs are preserved.

    Parameters
    ----------
    tracks : (time, z_t, nlat, nlon) int
    frac   : bottom fraction to discard (e.g. 0.25 → drop bottom 25 %)

    Returns
    -------
    filtered : same shape as tracks
    """
    ids = np.unique(tracks)
    ids = ids[ids != 0]
    if len(ids) == 0:
        return np.zeros_like(tracks)

    volumes    = {fid: int(np.sum(tracks == fid)) for fid in ids}
    sorted_ids = sorted(volumes, key=volumes.get)
    n_remove   = int(len(sorted_ids) * frac)
    keep_ids   = set(sorted_ids[n_remove:])

    filtered = np.zeros_like(tracks)
    for fid in keep_ids:
        filtered[tracks == fid] = fid
    return filtered


# ──────────────────────────────────────────────────────────────────────────────
# Temporal tracker — containment-based with lineage preservation
# ──────────────────────────────────────────────────────────────────────────────

def track_objects_with_splitting(
    depth_connected: xr.DataArray | np.ndarray,
    volume_weights:  np.ndarray | None = None,
    contain_thresh:  float = 0.3,
    alpha:           float = 0.5,
    plot_results:    bool  = False,
    depth_idx:       int   = 0,
) -> tuple[np.ndarray, dict]:
    """
    Track objects across time using containment with lineage preservation.

    Containment metric
    ------------------
    score = max(|A∩B|/|A|, |A∩B|/|B|)

    Dividing by the SMALLER object means a small fragment that is fully
    contained within a large parent scores 1.0, correctly linking split
    children back to their parent event regardless of size change.

    When volume_weights is provided:
        score = alpha × voxel_containment + (1−alpha) × volume_containment

    Lineage rules
    -------------
    - Each current object independently finds all parents above contain_thresh.
    - It inherits the SMALLEST original ID among all overlapping parents.
    - Multiple current objects can share the same parent (splitting).
    - When two lineages merge, the smaller original ID wins.
    - Unmatched current objects become new events with new IDs.

    Parameters
    ----------
    depth_connected : DataArray or ndarray (time, z, nlat, nlon)
                      3-D labelled objects, labels reset per timestep.
    volume_weights  : ndarray (z, nlat, nlon) — physical volume per voxel (m³).
                      When provided, containment is volume-weighted.
    contain_thresh  : float — minimum containment score to link objects.
                      Lower = more permissive. Default 0.3.
    alpha           : float — voxel vs volume containment weight. Default 0.5.
    plot_results    : bool  — if True, plot original vs tracked per timestep.
    depth_idx       : int   — depth slice to use for plotting.

    Returns
    -------
    tracked    : int ndarray (time, z, nlat, nlon) — persistent event IDs
    origin_map : dict {id: original_id} — maps every ID to its first ancestor
    """
    arr = depth_connected.values if hasattr(depth_connected, "values") \
          else depth_connected
    T, Z, Y, X = arr.shape

    tracked = np.zeros_like(arr, dtype=int)
    next_id = 1
    origin_map: dict[int, int] = {}

    # ── helpers ──────────────────────────────────────────────────────────────

    def _extract(vol: np.ndarray):
        labels = np.unique(vol)
        labels = labels[labels > 0]
        return labels, [(vol == lbl) for lbl in labels]

    def _containment(maskA: np.ndarray, maskB: np.ndarray) -> float:
        """max(|A∩B|/|A|, |A∩B|/|B|), optionally volume-weighted."""
        inter = maskA & maskB
        if not inter.any():
            return 0.0
        sizeA = maskA.sum()
        sizeB = maskB.sum()
        if volume_weights is not None:
            iw  = volume_weights[inter].sum()
            wA  = volume_weights[maskA].sum()
            wB  = volume_weights[maskB].sum()
            vox = max(inter.sum() / sizeA, inter.sum() / sizeB)
            vol = max(iw / wA, iw / wB)
            return float(alpha * vox + (1.0 - alpha) * vol)
        return float(max(inter.sum() / sizeA, inter.sum() / sizeB))

    # ── t = 0: assign local labels as global IDs ─────────────────────────────
    labels0, masks0 = _extract(arr[0])
    prev_objects: dict[int, np.ndarray] = {}

    for lbl, mask in zip(labels0, masks0):
        lid = int(lbl)
        tracked[0][mask] = lid
        prev_objects[lid] = mask
        origin_map[lid]   = lid
        next_id = max(next_id, lid + 1)

    # ── main time loop ────────────────────────────────────────────────────────
    for t in range(1, T):
        curr_labels, curr_masks = _extract(arr[t])

        if len(curr_masks) == 0:
            prev_objects = {}
            continue

        if len(prev_objects) == 0:
            for mask in curr_masks:
                tracked[t][mask] = next_id
                prev_objects[next_id] = mask.copy()
                origin_map[next_id]   = next_id
                next_id += 1
            continue

        prev_ids   = list(prev_objects.keys())
        prev_masks = list(prev_objects.values())

        # Containment matrix (n_curr × n_prev)
        contain_matrix = np.zeros((len(curr_masks), len(prev_ids)))
        for i, cm in enumerate(curr_masks):
            for j, pm in enumerate(prev_masks):
                contain_matrix[i, j] = _containment(cm, pm)

        # Each current object independently picks the parent with the
        # smallest original ID among all parents above contain_thresh
        curr_to_original: dict[int, int] = {}
        assigned_curr: set[int] = set()

        for i in range(len(curr_masks)):
            best_original = None
            best_parent   = None
            best_score    = 0.0

            for j in range(len(prev_ids)):
                if contain_matrix[i, j] > contain_thresh:
                    parent_id   = prev_ids[j]
                    original_id = origin_map.get(parent_id, parent_id)

                    if best_original is None or original_id < best_original:
                        best_original = original_id
                        best_parent   = parent_id
                        best_score    = contain_matrix[i, j]
                    elif original_id == best_original and \
                         contain_matrix[i, j] > best_score:
                        best_parent = parent_id
                        best_score  = contain_matrix[i, j]

            if best_parent is not None:
                curr_to_original[i] = best_original   # type: ignore[assignment]
                assigned_curr.add(i)

        # Build new_objects dict for next timestep
        new_objects: dict[int, np.ndarray] = {}

        for i, original_id in curr_to_original.items():
            curr_mask = curr_masks[i]
            tracked[t][curr_mask] = original_id
            if original_id in new_objects:
                new_objects[original_id] |= curr_mask
            else:
                new_objects[original_id] = curr_mask.copy()
            origin_map[original_id] = original_id

        for i, curr_mask in enumerate(curr_masks):
            if i not in assigned_curr:
                tracked[t][curr_mask] = next_id
                new_objects[next_id]    = curr_mask.copy()
                origin_map[next_id]     = next_id
                next_id += 1

        prev_objects = new_objects

    return tracked, origin_map