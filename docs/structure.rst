.. _structure:

=================
Structure
=================

The struture pages gives a synopsis about the data formats and the methods implemented by the package. 

Data formats 
===============

The data space domain of :code:`watex` is composed of four different data types referring to the implemented methods such as:

ERP data type
------------------

It can be arranged into several formats such as `*.csv`, `*. xlsx`, `*.xml`, `*.html`, or 
simple in `Pandas <https://pandas.pydata.org/>`_  data frame. The columns of ERP must be composed of station positions, the resistivity data, 
and the coordinates such as longitude/latitude or easting/ northing. 

VES data type
---------------

It expects the same format as ERP. However, the columns of DC- sounding must be the AB/2 depth measurements at each 
time the current electrodes are moved apart and the resistivity values collected at each sounding depth. The MN/2 values 
of the potential electrodes are not compulsory. 


EM data type
--------------

`watex` deals only with the SEG-Electrical Data Interchange format(.edi). However, the EDI - object 
created from external software like `pycsamt <https://github.com/WEgeophysics/pycsamt>`_ and `MTpy <https://github.com/MTgeophysics/mtpy>`_ 
can also be read. Indeed, the watex EDI module API is designed to work with both. In addition, attributes and methods 
from EDI objects are constructed following both software structures. Boreholes and geology data type: Both can be collected 
in `. yaml`, `.json` or `.csv formats`. An example of data arrangement can be found in the `data/boreholes` directory of the package. 


Implemented Methods
====================

:code:`watex` works with methods enumerated below: 

Learning
----------

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
* `AdalineStochasticGradientDescent`
* `AdalineGradientDescent`
* `GreedyPerceptron`
        
Geophysical
--------------

* `DC- Electrical Resistivity Profiling`
* `DC- Vertical Electrical Sounding`
* `Short-period Electromagnetic methods such as Natural Source Audio-frequency Magnetotellurics`
* `Logging`
    
Hydrogeology
-------------

* `geology structures` 
* `geostrata model conception`  
* `borehole, drill & Hydro-parameters calculation`
	

.. _scikit-learn: http://scikit-learn.org/stable/
 

.. topic:: References 

	.. [1] Wessel, D.E., Smith, W.., 1998. New, improved version of generic mapping tools realeased. Eos Trans. Am. Geophys. 
		Union 79, 579.
	.. [2] Ferri, F.J., Pudil, P., Hatef, M., Kittler, J., 1994. Comparative study of techniques for large-scale feature 
		selection.This work was suported by a SERC grant GR/E 97549. The first author was also supported by a FPI grant from the 
		Spanish MEC, PF92 73546684, in: GELSEMA, E.S., KANAL, L.S. (Eds.), Pattern Recognition in Practice IV, Machine Intelligence 
		and Pattern Recognition. North-Holland, pp. 403–413. https://doi.org/https://doi.org/10.1016/B978-0-444-81892-8.50040-7