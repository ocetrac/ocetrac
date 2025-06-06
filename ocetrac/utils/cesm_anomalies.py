""""
cesm_anomalies.py - Calculate anomalies and features from CESM model data
"""

import numpy as np
import xarray as xr

def calculate_anomalies_trend_features(ds, threshold_perc):
    """
    Calculate anomalies and extreme features by removing seasonal cycles and trends.
    
    Parameters
    ----------
    ds : xarray.DataArray
        Input climate data with dimensions (time, lat, lon)
    threshold_perc : float
        Percentile threshold (0-1) for extreme event detection (e.g., 0.95 for 95th percentile)
    
    Returns
    -------
    tuple
        (mean, trend, seasonal_cycle, extreme_features, anomalies)
        - mean: Long-term mean (lat, lon)
        - trend: Linear trend component (time, lat, lon)
        - seasonal_cycle: Annual+semiannual cycles (time, lat, lon)
        - extreme_features: Values exceeding threshold (time, lat, lon)
        - anomalies: Residual anomalies (time, lat, lon)
    """

    # Convert time to decimal years (e.g., 2000.5 for July 2000)
    dyr = ds.time.dt.year + ds.time.dt.month/12
    
    # 6-component harmonic model matrix:
    model = np.array([np.ones(len(dyr))] +          # Mean
                    [dyr-np.mean(dyr)] +           # Trend
                    [np.sin(2*np.pi*dyr)] +        # Annual sine
                    [np.cos(2*np.pi*dyr)] +        # Annual cosine
                    [np.sin(4*np.pi*dyr)] +        # Semiannual sine
                    [np.cos(4*np.pi*dyr)])        # Semiannual cosine
    
    # Solve least squares problem using pseudo-inverse
    pmodel = np.linalg.pinv(model)
    
    # Convert to xarray DataArrays for dimensional operations
    model_da = xr.DataArray(model.T, 
                          dims=['time','coeff'], 
                          coords={'time':ds.time.values, 'coeff':np.arange(1,7,1)}) 
    
    pmodel_da = xr.DataArray(pmodel.T, 
                           dims=['coeff','time'], 
                           coords={'coeff':np.arange(1,7,1), 'time':ds.time.values})
    
    # Calculate model coefficients per grid cell
    var_mod = xr.DataArray(pmodel_da.dot(ds), 
                         dims=['coeff','lat','lon'], 
                         coords={'coeff':np.arange(1,7,1), 'lat':ds.lat.values, 'lon':ds.lon.values})
    
    # Reconstruct components
    mean = model_da[:,0].dot(var_mod[0,:,:])          # Long-term mean
    trend = model_da[:,1].dot(var_mod[1,:,:])         # Linear trend
    seas = model_da[:,2:].dot(var_mod[2:,:,:])        # Seasonal cycles
    
    # Calculate full-field anomalies (observed - model)
    var_anom_notrend = ds-model_da.dot(var_mod)
    
    # Ensure optimal chunking for parallel operations
    if var_anom_notrend.chunks:
        var_anom_notrend = var_anom_notrend.chunk({'time': -1})
    
    # Calculate threshold and extreme features
    threshold = var_anom_notrend.quantile(threshold_perc, dim=('time'))
    features_notrend = var_anom_notrend.where(var_anom_notrend>=threshold, other=np.nan)
    
    return mean, trend, seas, features_notrend, var_anom_notrend