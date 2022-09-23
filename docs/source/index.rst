.. WATex documentation master file, created by
   sphinx-quickstart on Wed Jun 15 16:21:53 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

💧 `WATex's` documentation
================================

`A machine learning research for hydrogeophysic` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

< *Life is much better with potable water* >

`WATex`_ is an open-source package entirely written in Python to bring a piece of solution 
in the field of groundwater exploration (GWE). It uses the  ML learning methods to compute electrical and logging features to predict 
to predict the flow rate and also the hydrogeological parameters such as water content (water inflow), the medium permeability. 
Currently, it deals with the geophysical methods (DC resistivity profiling & sounding, Electromagnetic methods for short-periods (Audio-freqency magnetotellurics)). 
All modules and packages are written to intend solving real-engineering problems in the field of GWE.  

.. _WATex: https://github.com/WEgeophysics/watex/


`WATex's` development 
^^^^^^^^^^^^^^^^^^^^^^^^^^

`WATex`_ works with the learning methods enumerated below:

* `Support vector machines`
* `Neighbors: KNN`
* `Trees: DTC, Extratrees` 
* `Ensemble methods (RandomForests, Bagging and Pasting, Boosting and Stacking)`
* `Apriori`
* `Kernel Principal Component Analysis k-PCA`
* `t-distributed Stochastic Neighbor Embedding t-SNE`
* `Randomized PCA`
* `Locally Linear Embedding (LLE)`

and the geophysical methods:

* `DC- Electrical Resistivity Profiling`
* `DC- Vertical Electrical Sounding`
* `Audio-frequency Magnetotellurics`
* `Logging`


`WATex` API
^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   getting_started
   tutorials
   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
