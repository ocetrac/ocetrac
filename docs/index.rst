.. gcm-filters documentation master file, created by
   sphinx-quickstart on Tue Jan 12 09:24:23 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Ocetrac
=======================================

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
  
Welcome to the documentation page for Ocetrac! Here you will find instructions on how to install and use the API. Ocetrac is a python package specifically designed to identify and track the evolution of extreme climatic events from gridded data. Some oceanograpohic extremes include marine heatwaves, salinity anomalies, hypoxia, and high acidity. Extreme values are first identified by the user using a criteria appropriate for the data and application. Ocetrac then treats this data as a stack of still frame images to identify and label coherent object. It then stitches these fames together to track the evolution of these objects in time and space. 

.. toctree::
   :maxdepth: 3

   install
   new_section
   new_section_notebook
   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

