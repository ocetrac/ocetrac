API reference
=============

The API reference is automatically generated from the function docstrings
in the ocetrac package. Refer to the examples in the sidebar for reference
on how to use the functions.

----

Preprocessing
-------------

.. currentmodule:: ocetrac.preprocessing

.. autosummary::
   :toctree: ./_generated/

   preprocessing.compute_anomalies
   preprocessing.clean_binary
   preprocessing.threshold_features
   utils.compute_dask_quantile
   utils.get_xarray_memory_usage

----

SurfTrack
---------

Tracker
^^^^^^^

.. currentmodule:: ocetrac.SurfTrack

.. autosummary::
   :toctree: ./_generated/

   SurfTracker
   SurfTracker.run
   SurfTracker.clean
   SurfTracker.filter
   SurfTracker.track
   SurfTracker.postprocess
   SurfTracker.n_events
   SurfTracker.event_duration
   SurfTracker.summary

Measures — motion
^^^^^^^^^^^^^^^^^

.. currentmodule:: ocetrac.SurfTrack.measures

.. autosummary::
   :toctree: ./_generated/
   :recursive:

   MotionMeasures

Measures — shape
^^^^^^^^^^^^^^^^

.. currentmodule:: ocetrac.SurfTrack.measures

.. autosummary::
   :toctree: ./_generated/
   :recursive:

   ShapeMeasures

Measures — intensity
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: ocetrac.SurfTrack.measures

.. autosummary::
   :toctree: ./_generated/
   :recursive:

   calculate_intensity_metrics

Measures — temporal
^^^^^^^^^^^^^^^^^^^

.. currentmodule:: ocetrac.SurfTrack.measures

.. autosummary::
   :toctree: ./_generated/
   :recursive:

   get_duration
   get_initial_detection_time

Measures — plotting
^^^^^^^^^^^^^^^^^^^

.. currentmodule:: ocetrac.SurfTrack.measures

.. autosummary::
   :toctree: ./_generated/
   :recursive:

   plot_displacement

----

DeepTrack
---------

Tracker
^^^^^^^

.. currentmodule:: ocetrac.DeepTrack

.. autosummary::
   :toctree: ./_generated/

   DeepTracker
   DeepTracker.run
   DeepTracker.clean
   DeepTracker.label
   DeepTracker.connect_depth
   DeepTracker.prefilter
   DeepTracker.track
   DeepTracker.postprocess
   DeepTracker.n_events
   DeepTracker.event_duration
   DeepTracker.summary

Grid
^^^^

.. currentmodule:: ocetrac.DeepTrack.grid

.. autosummary::
   :toctree: ./_generated/

   compute_dz
   build_cell_volume
   make_anisotropic_struct

Core algorithms
^^^^^^^^^^^^^^^

.. currentmodule:: ocetrac.DeepTrack.tracker

.. autosummary::
   :toctree: ./_generated/

   label_2d_stack
   filter_area_2d_global_depth
   build_3d_objects
   filter_preserve_labels_global
   track_objects_with_splitting
