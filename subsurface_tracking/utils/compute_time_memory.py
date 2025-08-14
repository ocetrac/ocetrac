import xarray as xr
import dask.array as da  # Changed from 'import dask as da'
import pandas as pd

def compute_dask_quantile(anom_data, q=0.9):
    """Compute quantile using Dask's parallelized algorithm"""
    # Convert to dask array (preserving chunks)
    dask_array = anom_data.data
    
    # Compute quantile along time axis (axis=0 for xarray's time dimension)
    quantile_array = da.quantile(
        dask_array,
        q=q,
        axis=0,  # Corresponds to time dimension
        method='linear',
        keepdims=False
    )
    
    # Convert back to xarray with original coordinates
    return xr.DataArray(
        quantile_array,
        dims=anom_data.dims[1:],  # Remove time dimension
        coords={k: v for k, v in anom_data.coords.items() if k != 'time'}
    )

def get_xarray_memory_usage():
    """Returns memory usage of xarray objects."""
    mem_usage = []
    for var_name, var_value in globals().items():
        if isinstance(var_value, (xr.DataArray, xr.Dataset)):
            size_mb = var_value.nbytes / (1024 ** 2)  # MB
            mem_usage.append({
                'Variable': var_name,
                'Type': type(var_value).__name__,
                'Size (MB)': round(size_mb, 2)
            })
    return pd.DataFrame(mem_usage).sort_values('Size (MB)', ascending=False)
