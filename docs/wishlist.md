Wishlist
========

By adopting open-source best practices, we hope Ocetrac will grow into a widely used, community-driven project. We anticipate Ocetrac will have broad applications in geoscience and are excited to see it used in other domains besides oceanography.

If you have an idea, open an issue on the [Ocetrac GitHub repository](https://github.com/ocetrac/ocetrac/issues) and let us know! See our Contribution Guide to get involved.

Planned features
---------------------------------

**Tracking**
- Support periodic boundaries in DeepTrack
- Add support for additional grid types
- Improve splitting and merging logic for complex event topologies
- Extend overlap tracking to support user-defined matching criteria

**Performance**
- Optimize Dask integration for larger-than-memory datasets
- Reduce memory footprint during connected component labelling
- Add benchmarking capabilities to track performance across releases

**Validation**
- Implement feature validation metrics and skill scores
- Add uncertainty quantification tools for tracked events
- Provide comparison utilities for evaluating against observational datasets

**Documentation and examples**
- Expand tutorial notebooks
- Improve API reference with richer docstrings and usage examples
- Add example on isopycnal surfaces

----
 
Community goals
---------------
 
Ocetrac is built to be extended. Beyond the planned features above, we hope the 
community will contribute new tracking algorithms, grid adapters, and diagnostic 
tools. The long-term vision is a modular framework where SurfTrack and DeepTrack 
are two of many available trackers.