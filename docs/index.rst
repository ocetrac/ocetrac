Ocetrac
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


.. image:: https://github.com/ocetrac/ocetrac/workflows/Tests/badge.svg
   :target: https://github.com/ocetrac/ocetrac/actions

.. image:: https://codecov.io/gh/ocetrac/ocetrac/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/ocetrac/ocetrac

.. image:: https://img.shields.io/badge/License-MIT-lightgray.svg?style=flat-square
   :target: https://opensource.org/licenses/MIT

.. image:: https://img.shields.io/pypi/v/ocetrac.svg
   :target: https://pypi.org/project/ocetrac

.. image:: https://img.shields.io/conda/dn/conda-forge/ocetrac?label=conda-forge
   :target: https://anaconda.org/conda-forge/ocetrac

.. image:: https://readthedocs.org/projects/ocetrac/badge/?version=latest
   :target: https://ocetrac.readthedocs.io/en/latest/?badge=latest

|


Label and track the evolution of geospatial features in gridded datasets.
Ocetrac operates lazily with Dask for memory-efficient, parallelised execution and provides two tracking algorithms:

- **SurfTrack** — surface events ``(time, lat, lon)``
- **DeepTrack** — subsurface volumetric events ``(time, depth, lat, lon)``
 
----
 
Quick start
-----------

**SurfTrack**

.. code-block:: python
 
   from ocetrac.preprocessing import compute_anomalies, threshold_features
   from ocetrac.SurfTrack import SurfTracker
 
   mean, trend, seas, features, anom = calculate_anomalies_trend_features(
    ds, 0.9)
 
   mask    = xr.where(~ds.isel(time=0).isnull(), 1, 0)
   tracker = SurfTracker(
       features,
       mask,
       radius            = 2,
       min_size_quartile = 0.25,
       min_area_cells    = 100,
       timedim           = 'time',
       xdim              = 'lon',
       ydim              = 'lat',
       positive          = True,
   )
   result = tracker.run()
   tracker.summary()

**DeepTrack**

.. code-block:: python
 
   from ocetrac.preprocessing import compute_anomalies, threshold_features
   from ocetrac.DeepTrack import DeepTracker
   from ocetrac.DeepTrack.grid import build_cell_volume
 
   # Compute anomalies and threshold at the 90th percentile
   anom = compute_anomalies(ds)
   features, _ = threshold_features(anom, q=0.9)
 
   # Build cell volume for weighted tracking
   cell_volume = build_cell_volume(TAREA, z_t, n_z=20).compute()
   cell_vol_np = cell_volume.values
 
   # Run the tracker
   tracker = DeepTracker(
        features,
        radius         = 3,
        min_area_cells = 200,
        min_quantile   = 0.25, 
        contain_thresh = 0.3,
        alpha          = 0.5,
        frac_filter    = 0.25,
        connect_z      = True,
        positive       = True,
        n_z            = 20,
    )

    result_full = tracker.run(cell_volume=cell_vol_np)
    tracker.summary()
 
----

For recommendations or bug reports, please visit: https://github.com/ocetrac/ocetrac/issues/new

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   dependencies
   examples

.. toctree::
   :maxdepth: 2
   :caption: Trackers

    SurfTrack
    DeepTrack

.. toctree::
   :maxdepth: 2
   :caption: Help & Reference

   api
   contributing
   wishlist
   whats-new
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
