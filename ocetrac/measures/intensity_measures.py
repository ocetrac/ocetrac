import xarray as xr 

def calculate_intensity_metrics(anomalies: xr.DataArray, 
                              quantile_threshold: float = 0.9) -> dict:
    """
    Calculate comprehensive intensity metrics from anomaly data.
    
    Parameters
    ----------
    anomalies : xr.DataArray
        Input anomaly data with dimensions (time, lat, lon)
    quantile_threshold : float, optional
        Quantile threshold for calculating extreme intensities (0-1), default 0.9
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'cumulative_intensity': Spatial sum time series
        - 'mean_intensity_timeseries': Spatial mean time series
        - 'mean_intensity': Global mean value
        - 'max_intensity_timeseries': Spatial max time series  
        - 'max_intensity': Global maximum value
        - 'std_intensity_timeseries': Spatial std time series
        - 'std_intensity': Global std value
        - f'percentile_{int(quantile_threshold*100)}_intensity_timeseries': Spatial quantile time series
        - f'percentile_{int(quantile_threshold*100)}_intensity': Global quantile value
        - 'quantile_threshold_used': The quantile threshold applied
    """
    # Calculate all metrics
    results = {
        'cumulative_intensity': anomalies.sum(dim=('lat', 'lon')),
        'mean_intensity_timeseries': anomalies.mean(dim=('lat', 'lon')),
        'mean_intensity': float(anomalies.mean().item()),
        'max_intensity_timeseries': anomalies.max(dim=('lat', 'lon')),
        'max_intensity': float(anomalies.max().item()),
        'std_intensity_timeseries': anomalies.std(dim=('lat', 'lon')),
        'std_intensity': float(anomalies.std().item()),
        f'percentile_{int(quantile_threshold*100)}_intensity_timeseries': anomalies.quantile(quantile_threshold, dim=('lat', 'lon')),
        f'percentile_{int(quantile_threshold*100)}_intensity': float(anomalies.quantile(quantile_threshold).item()),
        'quantile_threshold_used': quantile_threshold
    }
    
    return results