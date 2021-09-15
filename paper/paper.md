---
title: 'Ocetrac: A Python package to track the spatiotemporal evolution of marine heatwaves'
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
    affiliation: University of Washington
affiliations:
  - name: Lamont-Doherty Earth Observatory, Columbia University, Palisades, New York, USA
    index: 1
  - name: National Center for Atmospheric Research, Computation and Information Systems Laboratory, Boulder, Colorado, USA
    index: 2
  - name: National Aeronautics and Space Administration, Ames Research Center, California, USA
    index: 3
  - name: University of Washingtion, School of Oceanography, Seattle, Washington, USA
    index: 4
    
date: 15 September 2021
bibliography: paper.bib

---

# Background

Marine heatwaves (MHWs) are a type of extreme warming event in the field of oceanography that have gained increased attention since first being described off the coast of Western Australia in 2011 [@Pearce2013]. Qualitatively, MHWs are defined as prolonged and discrete periods of extremely warm sea surface temperatures that cause damage to marine habitats and disrupt ecological functioning [@Hobday2016; @Smale2019]. As a result, MHWs have resulted in declines in fishery productivity and even economic loss [@Cheung2020; @McCabe2016; @Mills2013]. The overall long persistence and large spatial scales of MHWs present several management challenges to protect living marine resources and secure economies [@Pershing2019]. These challenges are further complicated by the anticipation of stronger, longer lasting, and more frequent MHWs occurring as the ocean warms due to the continued emission of anthropogenic greenhouse gases [@Frolicher201; @IPCC2021; @Oliver2019a]. To fully understand the threat of MHWs evolving under future global warming scenarios, a careful inspection of the spatiotemporal connectivity of MHWs is an important prerequisite to link the known physical drivers of MHWs with their time evolving patterns [@Holbrook2020]. 

Detection methods for MHWs are typically based on point-wise thresholding, where the time series of sea surface temperature (SST) at each spatial grid point is seperately analyzed [@Oliver2021]. A MHW occurs when the local SST exceeds the 90th percentile climatology for at least 5 consecutive days with no more than a 2-day gap [@Hobday2016]. This definition has been applied to globally gridded historial [@Oliver2018] and projected [@Frolicher201; @Oliver2019] SST to assess trends in MHW metrics, drivers [@Holbrook2019; @SenGupt2020], and variability [@Oliver2019b]. Once MHWs are detected, their properties (e.g., intensity, duration, and frequency) are often averaged as composite maps. While these approaches are useful in detecting MHWs locally, they are limited by their inability to characterize the evolution of spatially connected discrete events. 


# Summary



# Example use case

# External libraries used
    
# Acknowledgements

# References
