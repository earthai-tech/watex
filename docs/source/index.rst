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
    * `Base Electromagnetic short-period methods such as Natural Source Audio-frequency Magnetotellurics`
    * `Logging`
    
* hydrogeology: 

    * `geology structures` 
    * `geostrata model conception`  
    * `borehole, drill & Hydro-parameters calculation`
	

.. _WATex: https://github.com/WEgeophysics/watex/
.. _scikit-learn: http://scikit-learn.org/stable/


`WATex` User Guide 
^^^^^^^^^^^^^^^^^^^^^^^^^

The user guide is composed of the `installation-guide` and the codes snippets implementations. Note that :code:`watex`
is not available on PyPI or conda-forge yet. For installation , use the repo installation instead. The code snippets 
for pratical examples are not exhaustive. It is just made as field guide for users. Throughout the design of 
:code:`watex`, many examples support all functions, methods and classes in docstrings.  

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   demo/tutorials	

`WATex` API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The package follows the modular approach of existing software like `scikit-learn`_ and a bit more GMT (Wessel and Smith, 1998).  It mostly uses scikit-learn  classes as a top-level module for the predictions. The :code:`watex` API follows the scheme below: 
	
	* choose the class of model by importing the appropriate module, class estimator, or assessor. The assessor is the class of the module designed for solving a specific task. 
	* choose model hyperparameters by instantiating this class with desired values. 
	* arrange data into a feature matrix and target vector following the discussion from before. 
	* fit the model to your data by calling the `fit()` method of the instantiated model even the plotting modules. 
	* apply the method to a new data. For supervising learning, often labels are predicted for unknown data using the prediction methods whereas for unsupervised learning, the data are often transformed or inferred properties using the `transform ()` of `predict ()` methods. 

.. toctree::
   :maxdepth: 1
   :caption: Reference API

   api/watex
   api_references
   
`WATex's` benefits
^^^^^^^^^^^^^^^^^^^^

:code:`watex` has been used to solve real-engineering problem, such as the FR prediction 
during the campaigns for drinking water supply. It aims computing some geoelectrical parameters 
using the DC-resistivity method (Resistivity Profiling and vertical sounding) and used the 
`Support vector machines` for the FR prediction with a success rate greater than 77% . The case history is published in 
`Water Resources Research`_ journal known as one of the most popular journal in the GWE field research.  

.. _Water Resources Research: https://doi.org/10.1029/2021wr031623


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
