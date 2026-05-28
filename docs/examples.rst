Examples and Documentation
==========================

Ocetrac has several tutorial Jupyter notebooks available to help you get started. Static 
examples are provided to demonstrate different use cases of Ocetrac. We also provide a set of documentation pages 
that explain the implementation details of each tracking method.

Overview
--------

- `Ocetrac Overview <examples/ocetrac_overview.html>`_ — Architecture, data flow, and key concepts behind SurfTrack and DeepTrack.
- `Quickstart <examples/quickstart.html>`_ — Get up and running with SurfTrack and DeepTrack in a few lines of code.
- `Working with output <examples/working_with_output.html>`_ — Read, analyse, plot, and animate tracked event output.
 
Tutorials
--------

.. raw:: html

   <div class="gallery-grid">
     <div class="gallery-card">
       <a href="examples/SurfTrack_tutorial.html">
         <div class="gallery-thumb">
           <img src="_static/surftrack_preview.png" alt="SurfTrack Tutorial" />
         </div>
         <div class="gallery-body">
           <div class="gallery-title">SurfTrack Tutorial</div>
           <div class="gallery-desc">Step-by-step walkthrough of the SurfTrack tracker from preprocessing to labelled events.</div>
         </div>
       </a>
     </div>

     <div class="gallery-card">
       <a href="examples/SurfTrack_measures_tutorial.html">
        <div class="gallery-thumb">
           <img src="_static/SurfTrack_measures_preview.png" alt="SurfTrack Measures" />
         </div>
         <div class="gallery-body">
           <div class="gallery-title">SurfTrack Measures</div>
           <div class="gallery-desc">Compute shape, motion, temporal, and intensity characteristics of tracked surface events.</div>
         </div>
       </a>
     </div>

     <div class="gallery-card">
       <a href="examples/SurfTrack_tutorial_restricting_time_connectivity.html">
        <div class="gallery-thumb">
           <img src="_static/surftrack_preview_method2.png" alt="SurfTrack Time Restrictive Connectivity" />
         </div>
         <div class="gallery-body">
           <div class="gallery-title">SurfTrack Time Restrictive Connectivity</div>
           <div class="gallery-desc">Step-by-step walkthrough of the SurfTrack tracker from preprocessing to labelled events using temporal neighbor tracking.</div>
         </div> 
       </a>
     </div>

     <div class="gallery-card">
       <a href="examples/DeepTrack_tutorial.html">
         <div class="gallery-thumb">
           <img src="_static/deeptrack_preview.png" alt="DeepTrack Tutorial" />
         </div>
         <div class="gallery-body">
           <div class="gallery-title">DeepTrack Tutorial</div>
           <div class="gallery-desc">Track subsurface marine heatwaves in 4-D (time, depth, lat, lon) CESM2 data.</div>
         </div>
       </a>
     </div>

   </div>

.. raw:: html

   <style>
   .gallery-grid {
     display: grid;
     grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
     gap: 1.25rem;
     margin: 1.5rem 0 2rem;
   }
   .gallery-card {
     border: 1px solid var(--pst-color-border, #e0e0e0);
     border-radius: 8px;
     overflow: hidden;
     transition: box-shadow 0.2s ease, transform 0.2s ease;
   }
   .gallery-card:hover {
     box-shadow: 0 4px 16px rgba(42,181,200,0.18);
     transform: translateY(-2px);
   }
   .gallery-card a {
     text-decoration: none;
     color: inherit;
     display: block;
   }
   .gallery-thumb {
     height: 100px;
     display: flex;
     align-items: center;
     justify-content: center;
   }
   .gallery-icon {
     font-size: 2.5rem;
   }
   .gallery-body {
     padding: 0.75rem 1rem 1rem;
   }
   .gallery-title {
     font-size: 14px;
     font-weight: 600;
     color: var(--pst-color-text-base, #222);
     margin-bottom: 0.4rem;
   }
   .gallery-desc {
     font-size: 12px;
     color: var(--pst-color-text-muted, #666);
     line-height: 1.5;
   }
  .gallery-thumb {
    height: 120px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    background: '#f0f0f0';
  }
  .gallery-thumb img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
   </style>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Overview

   examples/ocetrac_overview
   examples/quickstart
   examples/working_with_output

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorials

   examples/SurfTrack_tutorial
   examples/SurfTrack_measures_tutorial
   examples/DeepTrack_tutorial