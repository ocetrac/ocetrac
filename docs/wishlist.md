Wishlist
========

By adopting open-source best practices, we hope Ocetrac will grow into a widely used, community-driven and community-owned project. We anticipate Ocetrac will have broad applications in geoscience and are excited to see it used in other domains besides oceanography.

If you have an idea, open an issue on the [Ocetrac GitHub repository](https://github.com/ocetrac/ocetrac/issues) and let us know! 
See our Contribution Guide to get involved.

Planned features
---------------------------------

**Tracking**
- Add support for additional grid types
- Improve splitting and merging logic for complex event topologies
- Extend overlap tracking to support user-defined matching criteria

**Performance**
- Optimize Dask integration for larger-than-memory datasets, including chunked
  temporal (t−1 → t) connectivity via boundary patching in `SurfTrack.utils` and
  3-D (time × depth × lat × lon) chunking strategies unique to DeepTrack
- Reduce memory footprint during connected component labelling
- Add benchmarking capabilities to track performance across releases, with CI
- validation of memory behavior under realistic Dask configurations

**CI/CD and packaging**
- Add `release.yaml` GitHub Actions workflow for automated PyPI publishing on version
  tags, including cross-platform install testing (Ubuntu, macOS, Windows) across
  Python 3.10–3.12, automated changelog generation from git log, and a ReadTheDocs
  webhook trigger on publish
- Add a `CITATION.cff` file for standardised software citation
- Update contributing guide to reflect current Python version support (≥ 3.10)
  
**Validation**
- Implement feature validation metrics and skill scores
- Add uncertainty quantification tools for tracked events
- Provide comparison utilities for evaluating against observational datasets

**Documentation and examples**
- Add tutorial: temporal-neighbour (t−1 → t) connectivity with boundary patching for `SurfTrack`
- Add tutorial: boundary patching on ERA5 data
- Add tutorial: `DeepTrack` measures (shape, motion, intensity) to complement the existing `SurfTrack` Measures tutorial
- Improve API reference with richer docstrings and usage examples
- Add example on isopycnal surfaces

----
 
Community goals
---------------
 
Ocetrac is built to be extended. Beyond the planned features above, we hope the 
community will contribute new tracking algorithms, grid adapters, and diagnostic 
tools. The long-term vision is a modular framework where SurfTrack and DeepTrack 
are two of many available trackers.