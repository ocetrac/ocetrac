import xarray as xr
import numpy as np
import pandas as pd

def to_dataframe(event, ssta):
    '''
    Creates a Pandas DataFrame of event attributes.
    
    Parameters
    ----------
      event : xarray.DataArray   
              Image set containing only objects corresponding to the event of interest. 
              Dimensions should be ('time', 'lat', 'lon')
              
      ssta  : xarray.DataArray
              Temperature vector [1D numpy array of length T]
    
    Returns
    -------
    
    mhw : pandas.DataFrame
          Marine heat wave event attributes. The keys listed below are 
          are contained within the dataset.
 
        'id'                     Unique label given to the MHW [int]
        'date'                   Dates corresponding to the event [datetime format]
        'coords'                 Latitude and longitude of all points contained in the event [tuple(lat,lon)]
        'centroid'               Center of each object contained in the event [tuple(lat,lon)]
        'duration'               Duration of event [months]
        'intensity_max'          Maximum intensity at each time interval [degC]
        'intensity_mean'         Mean intensity at each time interval [degC]
        'intensity_min'          Minimum intensity at each time interval [degC]
        'intensity_cumulative'   Cumulated intensity over the entire event [degC months]
        'area'                   Area of the event at each time interval [km2]
        
    '''
    
    # Initialize dictionary 
    metrics = {}
    metrics['id'] = [] # event label
    metrics['date'] = [] # datetime format
    metrics['coords'] = [] # (lat, lon)
    metrics['centroid'] = []  # (lat, lon)
    metrics['duration'] = [] # [months]
    metrics['intensity_max'] = [] # [deg C]
    metrics['intensity_mean'] = [] # [deg C]
    metrics['intensity_min'] = [] # [deg C]
    metrics['intensity_cumulative'] = [] # [deg C]
    metrics['area'] = [] # [km2]

    # TO ADD:
    # mhw['rate_onset'] = [] # [deg C / month]
    # mhw['rate_decline'] = [] # [deg C / month]

    metrics['id'].append(int(np.nanmedian(event.values)))
    metrics['date'].append(event.time.values.astype('datetime64[M]'))
    metrics['duration'].append(event.time.shape[0])

    # Turn images into binary
    binary_event = event.where(event>=0, other=0)
    binary_event = binary_event.where(binary_event==0, other=1)
      
    sub_labels = xr.apply_ufunc(_get_labels, binary_event,
                                input_core_dims=[['lat', 'lon']],
                                output_core_dims=[['lat', 'lon']],
                                output_dtypes=[binary_event.dtype],
                                vectorize=True,
                                dask='parallelized')
    
    # Turn background to NaNs
    sub_labels = xr.DataArray(sub_labels, dims=binary_event.dims, coords=binary_event.coords)
    sub_labels = sub_labels.where(sub_labels>0, drop=False, other=np.nan) 

    # The labels are repeated each time step, therefore we relabel them to be consecutive
    for p in range(1, sub_labels.shape[0]):
        sub_labels[p,:,:] = sub_labels[p,:,:].values + sub_labels[p-1,:,:].max().values
    
    sub_labels_wrapped = _wrap(sub_labels)
    
    mhw = _get_intensity_area(event, ssta, mhw)
    
    centroid = []
    for s in np.arange(0, sub_labels_wrapped.shape[0]):
        lx = sub_labels_wrapped.isel(time=s)
        east = lx.where(lx.lon < 180, drop=True)
        east['lon'] = np.arange(360.125, 540.125, .25)
        append_east = xr.concat([lx.where(lx.lon >= 180, drop=True), east], dim="lon")
        centroid.append(_get_centroids(append_east))
    metrics['centroid'].append(centroid)
    
    metrics = pd.DataFrame(dict([(name, pd.Series(data)) for name,data in metrics.items()]))

    return mhw
