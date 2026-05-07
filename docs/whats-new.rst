.. currentmodule:: ocetrac

What's New
===========

This page summarises the major changes and additions to Ocetrac. For a full list of 
commits and pull requests, see the `GitHub changelog <https://github.com/ocetrac/ocetrac/commits/main>`_.


.. Template (do not remove)
    ------------------------

    Breaking changes
    ~~~~~~~~~~~~~~~~
    Description. (:pull:`ii`, :issue:`ii`). By `Name <https://github.com/github_username>`_.

    New Features
    ~~~~~~~~~~~~

    Documentation
    ~~~~~~~~~~~~~

    Internal Changes
    ~~~~~~~~~~~~~~~~

    Bug fixes
    ~~~~~~~~~
New Features
~~~~~~~~~~~~
 
- Extended Ocetrac to support subsurface tracking via the new
  :class:`ocetrac.DeepTrack.DeepTracker` class, which operates on fields with dimensions
  ``(time, depth, lat, lon)``.
 
- Restructured the package so that the original tracking algorithm is now formalised as
  :mod:`ocetrac.SurfTrack`, giving each tracker a namespace and making it
  easier to add new trackers in the future.
 
- Added :meth:`~ocetrac.DeepTrack.DeepTracker.connect_depth` for linking blobs across
  depth layers, anisotropic structuring element support via
  :func:`~ocetrac.DeepTrack.grid.make_anisotropic_struct`, and cell volume computation
  via :func:`~ocetrac.DeepTrack.grid.build_cell_volume`.
 
Documentation
~~~~~~~~~~~~~
 
- Rebuilt the documentation using the
  `pydata-sphinx-theme <https://pydata-sphinx-theme.readthedocs.io>`_, including a
  redesigned homepage with an animated feature showcase.
 
- Added a new :doc:`Ocetrac Structure Overview <examples/ocetrac_overview>` page
  describing the package architecture and data flow pipeline.
 
- Added a new :doc:`DeepTrack tutorial <examples/DeepTrack_tutorial>` notebook
  demonstrating end-to-end subsurface event tracking on CESM2 Large Ensemble data.
 
- Restructured sidebar navigation with separate Overview and Tutorial sections and
  improved API reference with cleaner class signatures.
 
Internal Changes
~~~~~~~~~~~~~~~~
 
- Reorganised the package into :mod:`ocetrac.SurfTrack` and :mod:`ocetrac.DeepTrack`
  submodules. The original algorithm is preserved and the :mod:`ocetrac.SurfTrack`
  interface is unchanged.
 