ocetrac
==============================
[![Build Status](https://github.com/ocetrac/ocetrac/workflows/Tests/badge.svg)](https://github.com/ocetrac/ocetrac/actions)
[![codecov](https://codecov.io/gh/ocetrac/ocetrac/branch/main/graph/badge.svg)](https://codecov.io/gh/ocetrac/ocetrac)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/ocetrac.svg)](https://anaconda.org/conda-forge/ocetrac)
[![pypi](https://img.shields.io/pypi/v/ocetrac.svg)](https://pypi.org/project/ocetrac)
[![downloads](https://pepy.tech/badge/ocetrac)](https://pepy.tech/project/ocetrac)
[![Documentation Status](https://readthedocs.org/projects/ocetrac/badge/?version=latest)](https://ocetrac.readthedocs.io/en/latest/?badge=latest)
[![License:MIT](https://img.shields.io/badge/License-MIT-lightgray.svg?style=flt-square)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5102928.svg)](https://doi.org/10.5281/zenodo.5102928)
[![All platform](https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/ocetrac-feedstock?branchName=master)](https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=13414&branchName=maste)


**ocetrac** is a Python package for labelling and tracking geospatial features in gridded datasets. It provides two tracking algorithms:

- **DeepTrack** — 4-D connected-component labelling and temporal tracking across `(time, depth, lat, lon)`, designed for subsurface features such as volumetric marine heatwaves.
- **SurfTrack** — 3-D connected-component labelling and temporal tracking across `(time, lat, lon)`, designed for surface features. SurfTrack supports two temporal connectivity modes:
  - **Permissive** — any spatial overlap between consecutive timesteps links two features as the same event.
  - **Restrictive** — a user-defined overlap threshold (e.g. 0.45) must be exceeded to link two features.

Both trackers operate lazily with [Dask](https://www.dask.org) for memory-efficient, parallelised execution on large datasets.

---
 
## Installation

**Mamba (Recommended)**

We recommend [mamba](https://mamba.readthedocs.io/en/latest/) for installation. Mamba is a fast drop-in replacement for conda. If you don't have mamba, install [Miniforge](https://github.com/conda-forge/miniforge) which ships with mamba by default, or install mamba into an existing conda environment:

```bash
conda install -n base -c conda-forge mamba
```

Then install ocetrac:

```bash
mamba install -c conda-forge ocetrac
```

```
 
**PyPI**
 
```bash
pip install ocetrac
```
 
**From source**
 
```bash
git clone https://github.com/ocetrac/ocetrac.git
cd ocetrac
pip install -e .
```
 
**Development environment**
 
```bash
mamba env create -f environment.yml
mamba activate ocetrac
python -m pytest tests/ -v
```

## Package structure
 
```
ocetrac/
├── preprocessing/       — anomaly computation, morphological cleaning, thresholding
├── DeepTrack/           — 4-D tracker (time, depth, lat, lon)
└── SurfTrack/           — 3-D tracker (time, lat, lon)
```
---
   
## Contributing

Issues and pull requests are welcome on [GitHub](https://github.com/ocetrac/ocetrac). Please file a [bug report](https://github.com/ocetrac/ocetrac/issues) if you find a problem, or open a [pull request](https://github.com/ocetrac/ocetrac/pulls) if you make an improvement.

---

## Citation

**SurfTrack** — When using the SurfTrack module, please cite the original [software](https://doi.org/10.5281/zenodo.5102928) and [Spatiotemporal Evolution of Marine Heatwaves Globally](https://journals.ametsoc.org/view/journals/atot/aop/JTECH-D-23-0126.1/JTECH-D-23-0126.1.xml).

**DeepTrack** — The DeepTrack module is currently being prepared for publication. Citation details will be added here upon publication.

---

## Acknowledgements

- We rely heavily on [scikit-image](https://peerj.com/articles/453/) and its community of contributors.
- This work is currently supported through a collaboration with the [UW eScience Institute](https://escience.washington.edu/).
- This work originally grew from a collaboration with NCAR during the ASP Graduate Visitor Program attended by Hillary Scannell. This project received support from the Leonardo DiCaprio Foundation, Microsoft, and the Gordon and Betty Moore Foundation.