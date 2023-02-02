
.. _about: 

==================
Mission & Goals 
==================

Here are some introductory notes about the mission and the goals of the package. 

Mission
=========

`WATex` stands for *WAT*-er-*ex*-ploration. Its mission is to provide “smart” algorithms 
for sustainable solutions faced with different challenges in the groundwater exploration field (GWE). 
 
For instance, geophysical methods such as DC, EM, and Logging are mostly used in association  
with pure hydrogeological methods to propose the right location for drilling operations and 
determine the permeability coefficient parameter k. Unfortunately, despite this combination, 
unsuccessful, unsustainable boreholes are persisting and the k parameter detection remains costly 
and difficult to collect thereby creating a huge loss for funders, geophysical and drilling ventures.
 
:code:`watex` henceforth works around these issues by bringing  efficient algorithms and smart approaches 
to solve these issues such as the recovery EM tensors, the automatic location detection for drilling operations, 
the prediction of flow rate, and the mixture learning strategy using machine learning. 

Via some tangible real-world examples implemented in case-history areas, :code:`watex` is an alternative Python-based package to:

* `reduce human effort`; 
* `improve the traditional geophysical methods by detecting the appropriate location for drilling operations; 
* `be more accurate in minimizing losses for future hydro-geophysical engineering projects`... 

.. note:: 
	Although the primary mission is focused on hydro-geophysical issues resolution, it is not limited to that way, It also implements 
	geology methods through the :mod:`~watex.geology` sub-packages. Any other fields in geosciences where new approaches are 
	discovered and useful to address a problem in the GWE field are welcome. 

	
Goals
========
Being open-source, the :code:`watex` expects to help academicians and companies working in the GWE field to achieve the following goals:

* reducing the unsuccessful and unsustainable boreholes during the drinking water supply campaign (DWSC) by predicting FR 
  before any drilling operations. 
* minimizing the failure of pumping tests by predicting in advance the permeability coefficient k. Furthermore, it 
  should be an alternative way for geosciences engineers to accurately define the depth of the existing underground aquifer, 
  to find the depth to start and end the pumping test.
* proposing a fast and efficient solution in the processing of short-period electromagnetic data especially the Natural Source Audio-frequency 
  Magnetotellurics NSAMT to locate the conductive zone for the drilling operations after the EM survey. 
* contributing in the `SDG-n6`_  and the Africa-Union `Agenda-2063-n1`_  achievements for the rural and urban population welfare. 
* globally providing a better future by putting a little smile on children's faces of families from the poorest regions of the world and Africa in particular.

.. _SDG-n6: https://unric.org/en/sdg-6/
.. _Agenda-2063-n1: https://au.int/en/agenda2063/flagship-projects


Benefits
===========

:code:`watex` has been used to solve a real-engineering problem, such as the FR prediction during DWSC. It aims to compute some 
geoelectrical parameters using the DC-resistivity methods (Resistivity Profiling and vertical sounding) and used the Support vector 
machines for the FR prediction with a success rate greater than 77%. The case history is published in `Water Resources Research`_ journal. 

.. _Water Resources Research: https://doi.org/10.1029/2021wr031623



