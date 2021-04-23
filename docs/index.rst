.. gcm-filters documentation master file, created by
   sphinx-quickstart on Tue Jan 12 09:24:23 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Ocetrac
=======================================
Label and track unique geospatial features from gridded datasets
~~~~~~
.. image:: https://github.com/ocetrac/ocetrac/workflows/Tests/badge.svg
   :target: https://github.com/ocetrac/ocetrac/actions

.. image:: https://codecov.io/gh/ocetrac/ocetrac/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/ocetrac/ocetrac
   
.. image:: https://img.shields.io/badge/License-MIT-lightgray.svg?style=flt-square
   :target: https://opensource.org/licenses/MIT

.. image:: https://img.shields.io/pypi/v/ocetrac.svg
   :target: https://pypi.org/project/ocetrac
   
.. image:: https://readthedocs.org/projects/ocetrac/badge/?version=latest
   :target: https://ocetrac.readthedocs.io/en/latest/?badge=latest
   
.. image:: https://img.shields.io/conda/dn/conda-forge/ocetrac?label=conda-forge
   :target: https://anaconda.org/conda-forge/ocetrac


| **Welcome to the documentation page for Ocetrac!** Ocetrac is a python package specifically designed to identify and track the evolution of extreme climatic events in gridded data. Here you will find instructions on how to install Ocetrac, use it's API, and contribute to future releases.  

.. toctree::
   :maxdepth: 1

   install
   api
   dev

| **How does it work?** Extreme values are first identified in the dataset by the user using a criteria appropriate for the data and application. Some examples of *oceanographic* use cases could include marine heatwaves (example below), salinity anomalies, hypoxia, or high acidity events. Ocetrac treats these data as stacks of still frame images to identify and label coherent objects. It then stitches these fames together to track the evolution of these objects in both time and space.

.. image:: _static/ocetrac_flowchart.png
   :width: 600 px
   :align: center

.. toctree::
   :maxdepth: 1

   marine_heatwave_example


