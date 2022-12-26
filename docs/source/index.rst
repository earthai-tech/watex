.. WATex documentation master file, created by
   sphinx-quickstart on Wed Jun 15 16:21:53 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

💧 `WATex's` documentation 
==============================

`A machine learning research for hydrogeophysic` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

< *Life is much better with potable water* >

`WATex`_ is an open-source library entirely written in Python to bring a piece of solution 
in the field of groundwater exploration (GWE). It computes some electrical, logging and hydrogeology features  
and uses the machine learning methods to predict some water content target such as the flow rate and 
the hydrogeological parameters(e.g. the permeabilty coefficient related to the water inrush). 
Indeed, the package deals with the geophysical (DC resistivity profiling & sounding, Electromagnetic for short-periods (Audio-freqency magnetotellurics)), 
logging and hydrogeology methods. The modules are written to intend solving real-engineering problems and will 
grow in the future release as the new methods are discovered especially in the GWE field.   

.. _WATex: https://github.com/WEgeophysics/watex/
.. _scikit-learn: http://scikit-learn.org/stable/

`WATex's` development 
^^^^^^^^^^^^^^^^^^^^^^^^^^

:code:`watex` works with methods enumerated below: 

* learning:

    * `Support vector machines`
    * `Neighbors: KNN`
    * `Trees: Decision Tree (DTC), Extratrees` 
    * `Ensemble methods (RandomForests, Bagging and Pasting, Boosting and Stacking)`
    * `Apriori, KMeans and Hierachical Agglomerative Trees`
    * `Kernel -Incremental- Principal Component Analysis k-PCA, i-PCA, nPCA`
    * `t-distributed Stochastic Neighbor Embedding t-SNE`
    * `Randomized PCA`
    * `Locally Linear Embedding (LLE)`
    * `more...`
    
    Furthermore, :code:`watex` implements an additional learning methods which are not implement in `scikit-learn`_ yet. These 
    are: 
    
        * `SequentialBackwardSelection`
        * `MajorityVoteClassifier`
        * `AdelineStochasticGradientDescent`
        * `AdelineGradientDescent`
        * `Perceptron`
        
* geophysical:

    * `DC- Electrical Resistivity Profiling`
    * `DC- Vertical Electrical Sounding`
    * `Base Electromagnetic short-period methods such as Audio-frequency Magnetotellurics`
    * `Logging`
    
* hydrogeology: 

    * `geology structures` 
    * `geostrata model conception`  
    * `borehole, drill & Hydro-parameters calculation`

`WATex's` benefits
^^^^^^^^^^^^^^^^^^^^

:code:`watex` has been used to solve real-engineering problem, such as the FR prediction 
during the campaigns for drinking water supply. It aims computing some geoelectrical parameters 
using the DC-resistivity method (Resistivity Profiling and vertical sounding) and used the 
`Support vector machines` for the FR prediction with a success rate greater than 77% . The case history is published in 
`Water Resources Research`_ journal known as one of the most popular journal in the GWE field research.  

.. _Water Resources Research: https://doi.org/10.1029/2021wr031623


`WATex` User Guide 
^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   getting_started
   demo/tutorials	
   scripts


`WATex` API
^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1
   :caption: Reference API

   api/watex
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
