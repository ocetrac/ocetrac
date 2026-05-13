# ============================================================
# preprocessing.py
# ============================================================
"""
Preprocessing pipeline:
  1. compute_anomalies  — remove climatological mean + seasonal cycle
  2. clean_binary       — morphological close→open per (t, z) slice  [DASK]
  3. threshold_features — 90th-percentile exceedance mask             [DASK]
  4. calculate_anomalies_trend_features - one surface level
"""
from __future__ import annotations

import functools

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from numpy.linalg import pinv
from scipy import ndimage

from .utils import compute_dask_quantile


def compute_anomalies(ds: xr.DataArray) -> xr.DataArray:
    """
    Remove climatological mean and seasonal cycle from a 4-D temperature
    field using a 6-coefficient harmonic model:

        T(t) = β0 + β1·t' + β2·sin(2π·t') + β3·cos(2π·t')
                          + β4·sin(4π·t') + β5·cos(4π·t')

    where t' = year + month/12 (fractional year).

    Parameters
    ----------
    ds : DataArray (time, z_t, nlat, nlon)

    Returns
    -------
    anom : DataArray — same shape, seasonally detrended anomalies
    """
    dyr = ds.time.dt.year + ds.time.dt.month / 12.0
    model = np.array([
        np.ones(len(dyr)),
        dyr - float(np.mean(dyr)),
        np.sin(2 * np.pi * dyr), np.cos(2 * np.pi * dyr),
        np.sin(4 * np.pi * dyr), np.cos(4 * np.pi * dyr),
    ])
    model_da = xr.DataArray(
        model.T, dims=["time", "coeff"],
        coords={"time": ds.time, "coeff": np.arange(1, 7)},
    )
    pmodel = xr.apply_ufunc(
        pinv, model_da,
        input_core_dims=[["coeff", "time"]],
        output_core_dims=[["time", "coeff"]],
        dask="parallelized", output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"time": len(ds.time), "coeff": 6}},
    )
    coeffs     = xr.dot(pmodel, ds, dims=["time"])
    full_model = model_da.dot(coeffs)
    return (ds - full_model).chunk({"time": -1, "z_t": 5, "nlat": 50, "nlon": 50})


def _clean_binary_2d(slice_2d: np.ndarray, radius: int, positive: bool) -> np.ndarray:
    """
    Morphological close→open on a single (nlat, nlon) slice.

    Closing fills interior holes and bridges small gaps within a warm region.
    Opening then removes any tiny isolated blobs left by closing.
    Wraps with periodic padding in longitude to prevent edge artefacts.

    Parameters
    ----------
    slice_2d : (nlat, nlon) float array
    radius   : int  — disk radius for the structuring element
    positive : bool — True → warm events (>0), False → cold (<0)

    Returns
    -------
    bool ndarray (nlat, nlon)
    """
    binary = (slice_2d > 0) if positive else (slice_2d < 0)
    y, x   = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    se     = (x ** 2 + y ** 2) <= radius ** 2
    padded = np.pad(binary, radius, mode="wrap")
    closed = ndimage.binary_closing(padded, structure=se)
    opened = ndimage.binary_opening(closed,  structure=se)
    return opened[radius:-radius, radius:-radius].astype(bool)


def clean_binary(
    da:       xr.DataArray,
    radius:   int  = 3,
    positive: bool = True,
) -> xr.DataArray:
    """
    Morphological close→open on each (t, z) horizontal slice.

    Dask-parallel via xr.apply_ufunc with vectorize=True.

    Parameters
    ----------
    da       : DataArray (time, z_t, nlat, nlon) — may be dask-backed
    radius   : int  — disk radius for the structuring element.
               Larger = more infilling and gap-bridging, but also more
               risk of merging nearby separate events.
    positive : bool — True → warm anomalies (>0), False → cold (<0)

    Returns
    -------
    cleaned : DataArray bool, same shape as da
    """
    _fn = functools.partial(_clean_binary_2d, radius=radius, positive=positive)
    return xr.apply_ufunc(
        _fn, da,
        input_core_dims=[["nlat", "nlon"]],
        output_core_dims=[["nlat", "nlon"]],
        vectorize=True, dask="parallelized", output_dtypes=[bool],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )


def threshold_features(
    anom: xr.DataArray,
    q:    float = 0.9,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute a spatial quantile threshold and return the binary exceedance mask.

    Uses compute_dask_quantile for a dask-parallel percentile over the time
    axis; both outputs are computed eagerly.

    Parameters
    ----------
    anom : DataArray (time, z_t, nlat, nlon)
    q    : float — quantile level (default 0.9 → 90th percentile)

    Returns
    -------
    features      : DataArray — anomaly where >= threshold, else NaN
    threshold_map : DataArray (z_t, nlat, nlon)
    """
    threshold_map = compute_dask_quantile(anom, q=q)
    with ProgressBar():
        threshold_map = threshold_map.compute()
    features = anom.where(anom >= threshold_map)
    with ProgressBar():
        features = features.compute()
    return features, threshold_map


def calculate_anomalies_trend_features(
    ds:             xr.DataArray,
    threshold_perc: float = 0.9,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Calculate anomalies and extreme features by removing seasonal cycles
    and linear trends using a 6-coefficient harmonic model:

        T(t) = β₀ + β₁·t' + β₂·sin(2π·t') + β₃·cos(2π·t')
                          + β₄·sin(4π·t') + β₅·cos(4π·t')

    where t' = year + month/12 (fractional year).

    Parameters
    ----------
    ds              : DataArray (time, lat, lon)
                      Input climate data — must have a ``time`` coordinate
                      with dt accessors (year, month).
    threshold_perc  : float — percentile threshold for extreme event
                      detection, e.g. 0.9 for the 90th percentile.
                      Default 0.9.

    Returns
    -------
    mean            : DataArray (lat, lon)          — long-term mean
    trend           : DataArray (time, lat, lon)    — linear trend component
    seas            : DataArray (time, lat, lon)    — annual + semiannual cycles
    features_notrend: DataArray (time, lat, lon)    — anomalies >= threshold,
                                                       NaN elsewhere
    var_anom_notrend: DataArray (time, lat, lon)    — full residual anomalies
    """
    # Convert time to decimal years (e.g. 2000.5 for July 2000)
    dyr = ds.time.dt.year + ds.time.dt.month / 12

    # 6-component harmonic model matrix (6, T)
    model = np.array([
        np.ones(len(dyr)),               # β₀ — mean
        dyr - np.mean(dyr),              # β₁ — linear trend
        np.sin(2 * np.pi * dyr),         # β₂ — annual sine
        np.cos(2 * np.pi * dyr),         # β₃ — annual cosine
        np.sin(4 * np.pi * dyr),         # β₄ — semiannual sine
        np.cos(4 * np.pi * dyr),         # β₅ — semiannual cosine
    ])

    # Solve least squares via pseudo-inverse (T, 6)
    pmodel = np.linalg.pinv(model)

    # Convert to DataArrays for xarray dot-product operations
    model_da = xr.DataArray(
        model.T,
        dims=["time", "coeff"],
        coords={"time": ds.time.values, "coeff": np.arange(1, 7)},
    )
    pmodel_da = xr.DataArray(
        pmodel.T,
        dims=["coeff", "time"],
        coords={"coeff": np.arange(1, 7), "time": ds.time.values},
    )

    # Fit coefficients per grid cell: (coeff, lat, lon)
    var_mod = xr.DataArray(
        pmodel_da.dot(ds),
        dims=["coeff", "lat", "lon"],
        coords={
            "coeff": np.arange(1, 7),
            "lat":   ds.lat.values,
            "lon":   ds.lon.values,
        },
    )

    # Reconstruct components
    mean  = model_da[:, 0].dot(var_mod[0, :, :])   # long-term mean (lat, lon)
    trend = model_da[:, 1].dot(var_mod[1, :, :])   # linear trend (time, lat, lon)
    seas  = model_da[:, 2:].dot(var_mod[2:, :, :]) # seasonal cycles (time, lat, lon)

    # Full-field anomaly = observed − fitted model
    var_anom_notrend = ds - model_da.dot(var_mod)

    # Rechunk time to -1 for quantile computation
    if var_anom_notrend.chunks:
        var_anom_notrend = var_anom_notrend.chunk({"time": -1})

    # Threshold and extreme features
    threshold        = var_anom_notrend.quantile(threshold_perc, dim="time")
    features_notrend = var_anom_notrend.where(
        var_anom_notrend >= threshold, other=np.nan
    )

    return mean, trend, seas, features_notrend, var_anom_notrend
    