# ============================================================
# SurfTrack/tracker.py
# ============================================================
"""
Surface (2-D + time) connected-component labelling and tracking.

Changes from the original version
-------------------
1. `min_area_cells` parameter added — absolute floor for object area so the
   percentile threshold does not collapse near zero when all objects are
   similarly sized.  Threshold = max(min_area_cells, percentile(areas, q)).
 
2. Relabelling loop fixed — the original code added max_id to ALL label
   values including NaN cells, silently corrupting them.  Now the offset is
   applied only to positive (valid) label cells.
 
3. `dask_gufunc_kwargs={"allow_rechunk": True}` added to the apply_ufunc
   call in _morphological_operations — without this, spatially chunked dask
   arrays raise a ValueError because nlat/nlon are core dims.

Core algorithms:
 
  Morphological cleaning   morphological_operations
  Masking                  apply_mask
  Area filtering           filter_area
  3-D labelling            label_3d
  Date-line wrapping       wrap_labels
"""
from __future__ import annotations

import dask.array as dsa
import numpy as np
import scipy.ndimage
import xarray as xr

from skimage.measure import label as label_np
from skimage.measure import regionprops

# ──────────────────────────────────────────────────────────────────────────────
# Masking
# ──────────────────────────────────────────────────────────────────────────────

def apply_mask(
    binary_images: xr.DataArray, 
    mask:          xr.DataArray, 
) -> xr.DataArray:
    """
    Zero out grid cells that fall outside the ocean mask.
 
    Parameters
    ----------
    binary_images : DataArray
        Binary (0/1) field to be masked.
    mask : DataArray
        Binary mask — 1 = valid ocean cell, 0 = land or ignored region.
 
    Returns
    -------
    DataArray
        ``binary_images`` with land/ignored cells set to 0.
    """
    return binary_images.where(mask == 1, drop=False, other=0)

# ──────────────────────────────────────────────────────────────────────────────
# Morphological cleaning
# ──────────────────────────────────────────────────────────────────────────────

def morphological_operations(
    da:       xr.DataArray,
    radius:   int,
    xdim:     str,
    ydim:     str,
    positive: bool = True,
) -> xr.DataArray:
    """
    Convert the input field to binary and apply morphological close→open
    on each (ydim, xdim) slice independently using Dask.
 
    Closing fills small interior holes and bridges narrow gaps within a
    feature.  Opening then removes any tiny isolated patches left by
    closing.  Padding is applied in ``wrap`` mode to handle the periodic
    longitude boundary before each operation.
 
    Parameters
    ----------
    da       : DataArray (time, ydim, xdim)
    radius   : int — disk radius for the structuring element.
               ``radius=1`` is a no-op (single-pixel element).
    xdim     : str — name of the x (longitude) dimension
    ydim     : str — name of the y (latitude) dimension
    positive : bool — True → warm anomalies (>0); False → cold (<0)
 
    Returns
    -------
    DataArray
        Binary (0/1) field, same shape as ``da``.
    """
    if positive:
        bitmap_binary = da.where(da > 0, drop=False, other=0)
    else:
        bitmap_binary = da.where(da < 0, drop=False, other=0)
    bitmap_binary = bitmap_binary.where(bitmap_binary == 0, drop=False, other=1)
 
    diameter = radius * 2
    x        = np.arange(-radius, radius + 1)
    x, y     = np.meshgrid(x, x)
    se       = (x ** 2 + y ** 2) < radius ** 2
 
    def _binary_open_close(slice_2d: np.ndarray) -> np.ndarray:
        """Apply wrap-padded close→open to a single 2-D slice."""
        padded = np.pad(
            slice_2d,
            ((diameter, diameter), (diameter, diameter)),
            mode="wrap",
        )
        if radius == 1:
            s2 = padded
        elif radius > 1:
            s1 = scipy.ndimage.binary_closing(padded, se, iterations=1)
            s2 = scipy.ndimage.binary_opening(s1,     se, iterations=1)
        else:
            raise ValueError("radius must be an integer >= 1")
        return s2[diameter:-diameter, diameter:-diameter]
 
    return xr.apply_ufunc(
        _binary_open_close,
        bitmap_binary,
        input_core_dims=[[ydim, xdim]],
        output_core_dims=[[ydim, xdim]],
        output_dtypes=[bitmap_binary.dtype],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Area filtering
# ──────────────────────────────────────────────────────────────────────────────
 
def filter_area(
    binary_images:    xr.DataArray,
    xdim:             str,
    ydim:             str,
    min_size_quartile: float = 0.25,
    min_area_cells:   int   = 100,
) -> tuple[xr.DataArray, float, xr.DataArray, int]:
    """
    Label 2-D slices per timestep, make IDs consecutive across time,
    wrap across the date line, then filter by object area.
 
    Effective area threshold = max(min_area_cells, percentile(areas, q)).
 
    Parameters
    ----------
    binary_images     : DataArray (time, ydim, xdim) — binary after masking
    xdim              : str — name of the x dimension
    ydim              : str — name of the y dimension
    min_size_quartile : float — relative area percentile (0–1)
    min_area_cells    : int  — absolute minimum area in grid cells
 
    Returns
    -------
    area          : DataArray — voxel count per detected object
    min_area      : float    — effective threshold applied
    binary_labels : DataArray — binary field with only kept objects
    N_initial     : int      — object count before area filtering
    """
    def _get_labels(slice_2d: np.ndarray) -> np.ndarray:
        return _label_either(slice_2d, background=0)
 
    labels = xr.apply_ufunc(
        _get_labels,
        binary_images,
        input_core_dims=[[ydim, xdim]],
        output_core_dims=[[ydim, xdim]],
        output_dtypes=[binary_images.dtype],
        vectorize=True,
        dask="parallelized",
    )
    labels = xr.DataArray(
        labels, dims=binary_images.dims, coords=binary_images.coords
    )
    labels = labels.where(labels > 0, drop=False, other=np.nan)
 
    # Make labels consecutive across timesteps — apply offset to positive
    # cells only; NaN cells are left untouched.
    max_id = 0
    for i in range(1, labels.shape[0]):
        max_id = int(np.nanmax([max_id, labels[i - 1, :, :].max().values]))
        vals   = labels[i, :, :].values
        valid  = vals > 0
        vals[valid] += max_id
        labels[i, :, :] = vals
 
    labels = labels.where(labels > 0, drop=False, other=0)
    labels_wrapped, N_initial = wrap_labels(np.array(labels))
 
    props      = regionprops(labels_wrapped.astype("int"))
    label_ids  = [p.label for p in props]
    labelprops = xr.DataArray(label_ids, dims=["label"],
                               coords={"label": label_ids})
    area = xr.DataArray(
        [p.area for p in props], dims=["label"],
        coords={"label": labelprops},
    )
 
    if area.size == 0:
        raise ValueError(
            "No objects detected. "
            "Try lowering radius or min_size_quartile."
        )
 
    pct_threshold = float(np.percentile(area, min_size_quartile * 100))
    min_area      = float(max(min_area_cells, pct_threshold))
    print(f"area threshold : {min_area:.0f} cells  "
          f"(floor={min_area_cells}, "
          f"percentile={pct_threshold:.1f})")
 
    keep_labels   = labelprops.where(area >= min_area, drop=True)
    keep_where    = np.isin(labels_wrapped, keep_labels)
    out_labels    = xr.DataArray(
        np.where(keep_where, labels_wrapped, 0),
        dims=binary_images.dims,
        coords=binary_images.coords,
    )
    binary_labels = out_labels.where(out_labels == 0, drop=False, other=1)
 
    return area, min_area, binary_labels, N_initial
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 3-D connected-component labelling
# ──────────────────────────────────────────────────────────────────────────────
 
def label_3d(
    binary_labels: xr.DataArray,
    connectivity:  int = 3,
) -> tuple[np.ndarray, int]:
    """
    Apply 3-D connected-component labelling across (time, lat, lon)
    simultaneously.
 
    Objects that are spatially adjacent at adjacent timesteps are merged
    into a single event.
 
    Parameters
    ----------
    binary_labels : DataArray (time, ydim, xdim) — binary field
    connectivity  : int — passed to skimage/dask_image label (default 3)
 
    Returns
    -------
    labels : ndarray (time, ydim, xdim)
    n      : int — number of unique labels
    """
    return _label_either(binary_labels, return_num=True, connectivity=connectivity)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Date-line wrapping
# ──────────────────────────────────────────────────────────────────────────────
 
def wrap_labels(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Merge labels that straddle the periodic (date-line) longitude boundary.
 
    Compares the first and last longitude columns.  Any label in the last
    column that coincides spatially with a label in the first column is
    reassigned to the first-column label, joining features that cross the
    date line into a single event.
 
    Parameters
    ----------
    labels : ndarray (time, ydim, xdim) — integer label array
 
    Returns
    -------
    labels_wrapped : ndarray — relabelled with consecutive IDs after wrapping
    N              : int     — number of unique labels after wrapping
    """
    first_column = labels[..., 0]
    last_column  = labels[..., -1]
    unique_first = np.unique(first_column[first_column > 0])
 
    for val in unique_first:
        first      = np.where(first_column == val)
        last       = last_column[first[0], first[1]]
        bad_labels = np.unique(last[last > 0])
        labels[np.isin(labels, bad_labels)] = val
 
    labels_wrapped = (np.unique(labels, return_inverse=True)[1]
                      .reshape(labels.shape))
    return labels_wrapped, int(np.max(labels_wrapped))
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────
 
def _label_either(data: np.ndarray | dsa.Array, **kwargs):
    """
    Apply connected-component labelling to a NumPy or Dask array.
 
    Uses ``skimage.measure.label`` for NumPy arrays and
    ``dask_image.ndmeasure.label`` for Dask arrays.
 
    Parameters
    ----------
    data     : ndarray or dask Array
    **kwargs : passed to the underlying label function
 
    Returns
    -------
    Labelled array (and number of labels if ``return_num=True``).
 
    Raises
    ------
    ImportError
        If ``data`` is a Dask array and ``dask_image`` is not installed.
    """
    if isinstance(data, dsa.Array):
        try:
            from dask_image.ndmeasure import label as label_dask
            def label_func(a, **kwargs):
                ids, _ = label_dask(a, **kwargs)
                return ids
        except ImportError:
            raise ImportError(
                "dask_image is required for Dask arrays. "
                "Install it with `pip install dask_image` or call "
                ".load() on your data before tracking."
            )
    else:
        label_func = label_np
    return label_func(data, **kwargs)
