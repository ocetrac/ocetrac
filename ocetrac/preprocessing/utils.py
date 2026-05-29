# ============================================================
# DeepTrack/utils.py
# ============================================================
"""
Utility helpers: dask-parallel quantile and memory inspection.
"""
from __future__ import annotations

import inspect

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr


def compute_dask_quantile(anom_data: xr.DataArray, q: float = 0.9) -> xr.DataArray:
    """
    Compute a per-grid-cell quantile over the time axis.

    Parameters
    ----------
    anom_data : DataArray (time, ...)
    q         : float — quantile level, e.g. 0.9 for the 90th percentile

    Returns
    -------
    DataArray with the time dimension removed.
    """
    return anom_data.quantile(q, dim="time").drop_vars("quantile")



def get_xarray_memory_usage() -> pd.DataFrame:
    """
    Walk the caller's globals and return a DataFrame with the memory
    footprint of every xarray object found there.

    Returned columns: Variable, Type, Size (MB).
    """
    frame     = inspect.currentframe().f_back
    mem_usage = []
    for var_name, var_value in frame.f_globals.items():
        if isinstance(var_value, (xr.DataArray, xr.Dataset)):
            size_mb = var_value.nbytes / 1024 ** 2
            mem_usage.append({
                "Variable":  var_name,
                "Type":      type(var_value).__name__,
                "Size (MB)": round(size_mb, 2),
            })
    return pd.DataFrame(mem_usage).sort_values("Size (MB)", ascending=False)
