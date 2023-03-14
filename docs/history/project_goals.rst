
.. _goals_ref: 

============================
Mission & Goals 
============================

Here are some details about the mission and the goals of the software. 

Mission
=========

`WATex` stands for *WAT*-er-*ex*-ploration and its mission is : 

- to provide “smart” algorithms for sustainable solutions faced with different challenges in the :term:`groundwater` 
  exploration field (:term:`GWE`):
  
  - For instance, geophysical methods such as DC, EM, and Logging are mostly used in association  
    with pure hydrogeological methods to propose the right location for drilling operations and 
    determine the permeability coefficient parameter k. Unfortunately, despite this combination, 
    unsuccessful, unsustainable boreholes are persisting and the k parameter detection remains costly 
    and difficult to collect thereby creating a huge loss for funders, geophysical and drilling ventures.
    
    :code:`watex` henceforth works around these issues by bringing  efficient algorithms and smart approaches 
    such as the recovery EM tensors, the automatic location detection for drilling operations, 
    the prediction of flow rate, and the :term:`mixture learning strategy` using :term:`machine learning` ...

- to facilitate the reproducibility of some published papers results related to :term:`GWE` which codes are 
  not available in the literature, to help the academicians of geosciences community, and non-dedicated users who have no-more in-depth 
  skills in programming.
  
  - Indeed, one of the major problems encountered in the scientific research world is the code availability for reproducing works. 
    Some journals tried their best by recommending that authors must share their codes to make their work reproducible by third-party. 
    Unfortunately, some of the resource codes, once the paper is published are no more available. Sometimes the link referring 
    to codes source does no more exist or is broken. This habit seems not worthy thereby making it difficult for 
    the user and academicians to achieve the same results, especially for new-comers in programming. :code:`watex` tries to remedy 
    this issue, especially related to :term:`GWE` by reproducing the results from the paper himself and making the code available for all. 
 
    It invites geosciences developers to adhere to the project by sharing their codes or reproducing published papers related 
    to :term:`groundwater exploration` in this platform to make together the world better and scientific progress. 

    See the :doc:`development guide <../development>` for sharing your code or reproduced published papers and making it available for the geosciences 
    community following the software API.  
 
	
Goals
========
Being open-source, the :code:`watex` expects to help academicians and companies working in the :term:`GWE` field to achieve the 
following goals:

* reducing the unsuccessful and unsustainable boreholes during the drinking water supply campaign (:term:`DWSC`) by predicting  the :term:`flow` rate (FR )
  before any :term:`drilling` operations and be more accurate in minimizing losses for future :term:`hydro-geophysical` engineering projects.
* minimizing the failure of pumping tests by predicting in advance the permeability coefficient k. Furthermore, it 
  should be an alternative way for geosciences engineers to accurately define the depth of the existing underground aquifer, 
  to find the depth to start and end the pumping test.
* improving the traditional geophysical methods by detecting the appropriate location for drilling operations; 
* proposing a fast and efficient solution in the processing of short-period electromagnetic data especially the Natural Source Audio-frequency 
  Magnetotellurics (:term:`NSAMT`) to locate the conductive zone for the :term:`drilling` operations after the :term:`EM` survey. 
* contributing in the `SDG-n6`_  and the Africa-Union `Agenda-2063-n1`_  achievements for the rural and urban population welfare. 
* globally providing a better future by putting a little smile on children's faces of families from the poorest regions of the world and Africa in particular.
* reproducing the published papers related to :term:`GWE` field to help academicians and non-dedicated user for achieving the results of 
  published papers which codes are not available in the litterature. 


.. _SDG-n6: https://unric.org/en/sdg-6/
.. _Agenda-2063-n1: https://au.int/en/agenda2063/flagship-projects


Benefits
===========

:code:`watex` has been used to solve a real-engineering problem, such as the FR prediction during DWSC. It aims to compute some 
geoelectrical parameters using the DC-resistivity methods (Resistivity Profiling and vertical sounding) and used the Support vector 
machines for the FR prediction with a success rate greater than 77%. The case history is published in `Water Resources Research`_ journal. 

.. _Water Resources Research: https://doi.org/10.1029/2021wr031623


.. note::

    Although many novel approaches are focused on :term:`hydro-geophysical` issues resolution, it is not limited to that way, It also implements 
    :term:`geology` methods through the :mod:`~watex.geology` sub-packages. Any other fields in geosciences where new approaches are 
    discovered and useful to address a problem in the :term:`GWE` field are welcome.




