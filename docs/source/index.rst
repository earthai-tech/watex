.. WATex documentation master file, created by
   sphinx-quickstart on Wed Jun 15 16:21:53 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

💧 `WATex's` documentation!
=================================

`A machine learning Research library for Hydrogeophysic` 
---------------------------------------------------------

`WATex`_ is an open-source package entirely written in Python to bring a piece of solution 
in the field of groundwater exploration (GWE) via the use of ML learning methods. 
Currently, it deals with the geophysical methods (Electrical and Electromagnetic
methods). And, Modules and packages are written to solve real-engineering 
problems in the field of GWE. Later, it expects to add other methods such as the
induced polarisation and the near surface refraction-seismic for environmental
purposes (especially, for cavities detection to preserve the productive aquifers) 
as well as including pure Hydrogeology methods. 

.. _WATex: https://github.com/WEgeophysics/watex/


Development progress of `WATex`  
-------------------------------

Currently, `WATex`_ works with the learning methods enumerated below:

* Support vector machines
* Neighbors: KNN
* Trees: DTC
* Ensemble methods (RandomForests, Bagging and Pasting, Boosting)
* Apriori
* Kernel Principal Component Analysis k-PCA
* t-distributed Stochastic Neighbor Embedding t-SNE
* Randomized PCA
* Locally Linear Embedding (LLE)

and implements the geophysical methods below:

* Electrical Resistivity Profiling
* Vertical Electrical Sounding
* Natural and Controlled Source Audio-frequency Magnetotelluric


`WATex` API
------------


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   analysis
   bases
   datasets 
   geology 
   methods
   models
   property
   tools
   view

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
