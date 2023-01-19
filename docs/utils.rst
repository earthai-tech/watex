.. _utils

===============
Utilities
===============

.. currentmodule:: watex.utils

:mod:`~watex.utils` is a module composed of set of tricks, hints to quick performs operations and 
get systematically results. The tools are composed of functions to handle specifics task. The functions 
can be used in a third party software. For clarity, we will group utilities into three categories: 

* core utilities: :mod:`~watex.utils.coreutils` 
  Is a set of tools to deal with the software acceptable formats. It reads, controls, validates and parses the data 
  and distribute to the different modules that composed the package. It is also able to transform the data to a 
  recommended data-type if there is a minimum informations in the data that is recognized by the software. 
* mathematical extension utilities: :mod:`~watex.utils.exmath` 
  Is a set of tools mainly dedicated for mathematical algebras and geometry calculus. It is the core maths 
  tricks implemented throughout the software. 
* plot utilities: :mod:`~watex.utils.plotutils` 
  Is a set of plot tricks to quick visualize the data. It composed the additional utilities to support the 
  :mod:`~watex.view` module. 
* functions utilities: :mod:`~watex.utils.funcutils` 
  Is a set of pieces of components that is used by other functions to handle a specific task. It also contains many 
  functions that can be used by third-party software  to built its own packages. 
* geology utilities: :mod:`~watex.utils.geotools` 
  Is a set of functions maily focused on managing the layer properties, stratigraphic plots, everything related to the 
  geology and stratigraphic. 
* hydrogeology utilities: :mod:`~watex.utils.hydroutils`
  Is a set of tools for hydrogeology parameters computation, aquifer layer management, the boreholes depth section computing. 
  It is also composed of math concepts that is related to hydrogeology and logging. 
* GIS utilities: :mod:`~watex.utils.gistools` 
  Is a set of tools for managing the coordinates, everything refering to the geographic information system. It is used in 
  the package to assert the coordinates, transform or recomputed the  given values based of the referential system. 
  
In addition to the tools listed above, there are other modules that can help the user to construct its own Python Package. 

.. note::
   In this guide, we could not list all the module of these utilities. We will give some tools that expect to be useful for 
   the user to fast achieve some expected task. However for more-in depth, we recommend the user to clone the repository
   in `github <https://github.com/WEgeophysics/watex>`_ to discover other function tasks. 

.. _coreutils: 

Core Utilities: :mod:`~watex.utils.coreutils` 
================================================= 
:mod:`~watex.utils.coreutils` is composed of utilities that represent the core of the package. It manage the data, controls the data and 
implements basics task like parsing and transforming data to be recognized by other module of the package. We will list some below: 


.. note::
	
	Full Utility Guide will be available soon. 
	
.. _exmathutils: 

Math Extension Utilities: :mod:`~watex.utils.exmath` 
==================================================== 