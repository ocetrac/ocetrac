ocetrac
==============================
[![Build Status](https://github.com/ocetrac/ocetrac/workflows/Tests/badge.svg)](https://github.com/ocetrac/ocetrac/actions)
[![codecov](https://codecov.io/gh/ocetrac/ocetrac/branch/main/graph/badge.svg)](https://codecov.io/gh/ocetrac/ocetrac)
[![conda-forge](https://img.shields.io/conda/dn/conda-forge/ocetrac?label=conda-forge)](https://anaconda.org/conda-forge/ocetrac)
[![pypi](https://img.shields.io/pypi/v/ocetrac.svg)](https://pypi.org/project/ocetrac)
[![downloads](https://pepy.tech/badge/ocetrac)](https://pepy.tech/project/ocetrac)
[![Documentation Status](https://readthedocs.org/projects/ocetrac/badge/?version=latest)](https://ocetrac.readthedocs.io/en/latest/?badge=latest)
[![License:MIT](https://img.shields.io/badge/License-MIT-lightgray.svg?style=flt-square)](https://opensource.org/licenses/MIT)

Ocetrac is a Python 3.7+ packaged designed to label and track unique geospatial features from gridded datasets. The package is designed to accept data that have already been preprocessed, meaning that the data only contain values the user is interested in tracking. Ocetrac operates lazily with Dask so that it is memory uninhibited and fast through parallelized execution.


Installation
------------
Conda
.....
To install the core package from conda-forge run: ``conda install -c conda-forge ocetrac``

PyPI
....
To install the core package run: ``pip install ocetrac``.

GitHub
......
1. Clone ocetrac to your local machine: ``git clone https://github.com/ocetrac/ocetrac.git``
2. Change to the parent directory of ocetrac
3. Install ocetrac with ``pip install -e ./ocetrac``. This will allow
   changes you make locally, to be reflected when you import the package in Python.
   
 
