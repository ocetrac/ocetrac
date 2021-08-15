##############
Examples
##############


Here are some quick examples of how to use :py:class:`Tracker.track`. 

Start by importing numpy, xarray, and ocetrac using their standard abbreviations:

.. ipython:: python

    import numpy as np
    import xarray as xr
    import ocetrac 
    

Import NCEP Surface Air Temperatures over North America
------------------------------------------------------------------------


.. ipython:: python

    ds = xr.tutorial.load_dataset('air_temperature')
    da = ds.air
    da
    
    
Define Anomalies & Extremes
------------------------------------------------------------------------
    
Let's simply define anomalies by subtracting the monthly climatology.    

.. ipython:: python
    
    climatology = da.groupby(da.time.dt.month).mean()
    anomaly = da.groupby(da.time.dt.month) - climatology

    percentile = .9
    threshold = da.groupby(da.time.dt.month).quantile(percentile, dim='time', keep_attrs=True, skipna=True)
    
    hot_air = anomaly.groupby(da.time.dt.month).where(da.groupby(da.time.dt.month)>threshold)
    hot_air.isel(time=0).plot(cmap='Reds', vmin=0)
    
    
Label & Track Hot Air Events with Ocetrac
------------------------------------------------------------------------    
    

.. ipython:: python

    %%time 
    mask = xr.ones_like(anomaly.isel(time=0)) 
    Tracker =  ocetrac.Tracker(hot_air, mask, radius=2, min_size_quartile=0, timedim = 'time', xdim = 'lon', ydim='lat', positive=True)
    blobs = Tracker.track()
    
.. ipython:: python

    from matplotlib.colors import ListedColormap
    maxl = int(np.nanmax(blobs.values))
    cm = ListedColormap(np.random.random(size=(maxl, 3)).tolist())
    blobs.isel(time=0).plot(cmap= cm)
