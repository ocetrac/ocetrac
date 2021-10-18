---
title: '`Ocetrac`: A Python package to track the spatiotemporal evolution of marine heatwaves'
tags:
  - Python
  - image processing
  - oceanography
  - extremes
authors:
  - name: Hillary Scannell
    orcid: 0000-0002-6604-1695
    affiliation: 1
  - name: Julius Busecke
    orcid: 0000-0001-8571-865X
    affiliation: 1
  - name: Ryan Abernathey
    orcid: 0000-0001-5999-4917
    affiliation: 1
  - name: David John Gagne
    orcid: 0000-0002-0469-2740
    affiliation: 2
  - name: Daniel Whitt
    affiliation: 3
  - name: LuAnne Thompson
    orcid: 0000-0001-8295-0533
    affiliation: 4
affiliations:
  - name: Lamont-Doherty Earth Observatory, Columbia University, Palisades, New York, USA
    index: 1
  - name: National Center for Atmospheric Research, Computation and Information Systems Laboratory, Boulder, Colorado, USA
    index: 2
  - name: National Aeronautics and Space Administration, Ames Research Center, California, USA
    index: 3
  - name: University of Washingtion, School of Oceanography, Seattle, Washington, USA
    index: 4
    
date: 24 September 2021
bibliography: paper.bib

---

# Background

Marine heatwaves (MHWs) are a type of extreme warming event in the field of oceanography that have gained increased attention since first being described off the coast of Western Australia in 2011 [@Pearce2013]. Qualitatively, MHWs are defined as prolonged and discrete periods of extremely warm sea surface temperatures (SSTs) that cause thermal stress for marine ecosystems [@Hobday2016; @Smale2019]. The consequences range from coral bleaching to mass die-offs of critical marine species, harmful algal blooms, unproductive fisheries, economic loss, and even cross-border tensions [@Cheung2020; @McCabe2016; @Mills2013]. The overall long-lived persistence and large spatial scales of MHWs present several management challenges to protect living marine resources and secure economies [@Pershing2019]. These challenges are further complicated by the anticipation of stronger, longer lasting, and more frequent MHWs occurring as the ocean warms due to increased anthropogenic greenhouse gas emissions [@Frolicher2018; @IPCC2021; @Oliver2019a]. To fully understand the threat of MHWs evolving under future global warming scenarios, a careful inspection of the spatiotemporal connectivity of MHWs is an important prerequisite to link the known physical drivers of MHWs with their time evolving patterns [@Holbrook2020]. 

Detection methods for MHWs are typically based on point-wise thresholding, where the time series of SST at each spatial grid point is seperately analyzed [@Oliver2021]. A MHW occurs when SST at a particular locatoin exceeds the local 90th percentile climatology for at least 5 consecutive days with no more than a 2-day gap [@Hobday2016]. This definition has been applied to globally gridded historial and projected SSTs to assess trends in MHW metrics [@Frolicher2018; @Oliver2018; @Oliver2019a], drivers [@Holbrook2019; @SenGupt2020], and variability [@Oliver2019b]. Once MHWs are detected, their properties (e.g., intensity, duration, and frequency) are often averaged over time and evaluated as composite maps. While these approaches are useful in detecting MHWs locally, they are limited by their inability to characterize the full evolution of spatially connected discrete events. Knowlege of the spatiotemporal connectivity of MHWs is critical to advance state-of-the-art systems for MHW prediction and forecasting.

# Summary

To overcome the complex spatiotemporal challenges of characterizing MHWs, we present the first known global detection and tracking algorithm for MHWs. By leveraging mathematical morphological operations from multidimensional image processing, `Ocetrac` provides new spatiotemporal metrics that can be probed to explore how past events evolved. Applied to over 40 years of observed global SST data, @Scannell20XX use `Ocetrac` to reveal new pathways that connect geographically distant MHWs. The tropical Pacific Ocean acts as a major conduit for extremely persist and long-lived MHWs ushered by oceanic and atmospheric teleconnections from the tropics to mid-latitudes. The detection algorithm can be tuned using an object boundary smoothing parameter that is dependent on the horizontal resolution of the dataset. Additionally, a MHW size thresholding parameter is imposed to filter out small area MHWs according to the full area distribution of detected objects. These criteria ensure that only substantially sized MHWs are identified and tracked. @Scannell20XX provide a systematic parameter sensitivity analysis and recommendations for different use cases.  

The `Ocetrac` package contains a global class called *Tracker* that includes a collection of functions designed to detect, label, and track MHW objects. The primary algorithms are implemented using the Python programming language as a wrapper around many well-established numeric packages including numpy, xarray, dask, scipy, and scikit-image. A collection of [Jupyter notebooks](https://github.com/ocetrac/ocetrac/tree/main/notebooks) are provided to assist users through the entire workflow.

# Statement of Need

`Ocetrac` is intended to be used as a data processing tool for the extraction of MHW trajectories past, present, and future. It provides many useful metrics such as  event size, location, intensity, and duration. We anticipate that these new metrics will be incorporated into machine learning forecasts to predict when and where MHWs are likely to occur, with the intent for operational use in warning vulnerable coastal communities of physical risk.  

# Example use case

`Ocetrac` is an analysis routine with several data analysis steps. For the purpose of detecting and tracking marine heatwaves, SST anomalies (SSTAs) are calculated by removing the mean, trend, and seasonal cycle at each grid point. During this data preprocessing step, the 90th percentile of SSTAs is computed, again at each location, and used as a threshold to identify candidate MHW grid points. The dataset is transformed into a 3-dimensional binary image where 1's signify candidate MHW points and 0's signify the background image. Object detection is performed using a set of mathematical morphological operations in unique sucession (closing then opening). These operations pass a structuring element across the image and manipulate the image to remove small features, smooth the borders of the larger features, and fill small holes within large feature clusters. The resulting objects form the inital set of detected MHWs and are further eliminated based on their area using a percentile threshold defined from the collective size distribution.

![Example workflow of identifying and track marine heatwaves using Ocetrac.\label{fig:thr}](fig1.png)



<!-- 
![Ocetrac boundaries compared to initial sea surface temperature anomalies.\label{fig:thr}](fig2.png)

![Multiple Object Tracking; merging and splitting.\label{fig:thr}](fig3.png)
 -->
# External libraries used

Data manipulatoin is conveniently handled using `numpy` [@numpy] and `xarray` [@xarray], and are integrated with `dask` [@dask] when possible for efficient parallelized computing. The detection algorithm uses a set of morphological operations contained within the `scipy` [@scipy] multidimensional image processing package. Detected objects are labeled and analyzed using the `scikit-image` [@scikit-image] measure module.

# Acknowledgements

This work is supported by the Gordon and Betty Moore Foundation grant No. XXXXX. This is a collaborative project between Columbia University, the National Center for Atmospheric Research (NCAR), National Aeronautics and Space Administration (NASA) Ames Research Center, and the University of Washingtion. We also recieved support from the Leonardo DiCaprio Foundatoin AI for Earth Innovation Grant co-sponsored by Microsoft. 

# References
