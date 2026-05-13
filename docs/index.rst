Ocetrac documentation
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

.. raw:: html
 
   <div style="margin-top: 0.8em;"></div>

Welcome to the documentation of Ocetrac! **Ocetrac** is a Python toolkit for labelling and
tracking the evolution of geospatial features in gridded datasets. It is designed to operate 
lazily using Dask, enabling memory-efficient and parallelised compuation on large datasets. 
Ocetrac provides two core tracking algorithms:

- **SurfTrack** — surface features with dimensions ``(time, lat, lon)``
- **DeepTrack** — subsurface volumetric features with dimensions ``(time, depth, lat, lon)``

.. raw:: html
 
   <div style="margin-top: -1.5em;"></div>

.. raw:: html
 
   <div style="text-align: center; margin: -1.5em 0 0 0;">
     <img id="ocetrac-hero-gif" src="" alt="Ocetrac feature animation"
          style="max-width: 100%; height: auto; background: transparent; display: block; margin: 0 auto;" />
     <p id="ocetrac-hero-caption"
        style="text-align: left; font-style: italic; font-size: 1em;
               max-width: 100%; margin: 0em auto 0 auto; padding: -0.5; color: inherit;"></p>
   </div>
   <script>
     (function () {
       var figs = [
         { src: "_static/feature1.gif",  caption: "Animation of an event tracked using Ocetrac using data from an ensemble member of the CESM2 Large Ensemble." },
         { src: "_static/feature38.gif",  caption: "Animation of an event tracked using Ocetrac using data from an ensemble member of the CESM2 Large Ensemble." },
       ];
       var chosen = figs[Math.floor(Math.random() * figs.length)];
       document.getElementById("ocetrac-hero-gif").src = chosen.src;
       document.getElementById("ocetrac-hero-caption").textContent = chosen.caption;
     })();
   </script>
.. raw:: html
 
   <div style="margin-top: 1.5em;"></div>

With Ocetrac, users can track a wide range of phenomena, including marine heatwaves, cold spells, 
phytoplankton blooms, and other time-evolving features in gridded data. The documentation includes
everything needed to get started, from installation instructions to detailed API references and 
usage examples.

For recommendations or bug reports, please file an issue on GitHub: 
https://github.com/ocetrac/ocetrac/issues/new. If you are interested in contributing to 
the codebase, please see the `contributing guidelines <https://ocetrac.readthedocs.io/en/latest/contributing.html>`_.
If you need more help with Parcels, try the `Discussions page on GitHub <https://github.com/ocetrac/ocetrac/discussions>`_. 

----

.. toctree::
   :hidden:

   Home <self>
   Installation <installation>
   Examples and Documentation <examples>
   API Reference <api>