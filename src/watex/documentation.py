# -*- coding: utf-8 -*-
"""
.. |ohmS| replace:: Pseudo-area of fractured zone 
.. |sfi| replace:: Pseudo-fracturing index 
.. |VES| replace:: Vertical Electrical Sounding 
.. |ERP| replace:: Electrical Resistivity Profiling 

.. _Bagoue region: https://en.wikipedia.org/wiki/Bagou%C3%A9

.. _Dieng et al: http://documents.irevues.inist.fr/bitstream/handle/2042/36362/2IE_2004_12_21.pdf?sequence=1
.. _Kouadio et al: https://doi.org/10.1029/2021WR031623
.. _FlowRatePredictionUsingSVMs: https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021WR031623

.. _GeekforGeeks: https://www.geeksforgeeks.org/style-plots-using-matplotlib/#:~:text=Matplotlib%20is%20the%20most%20popular,without%20using%20any%20other%20GUIs

.. _IUPAC nommenclature: https://en.wikipedia.org/wiki/IUPAC_nomenclature_of_inorganic_chemistry

.. _Matplotlib scatter: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.scatter.html
.. _Matplotlib plot: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.plot.html
.. _Matplotlib pyplot: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.plot.html
.. _Matplotlib figure: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.figure.html
.. _Matplotlib figsuptitle: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.suptitle.html

.. _Properties of water: https://en.wikipedia.org/wiki/Properties_of_water#Electrical_conductivity 
.. _pandas DataFrame: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
.. _pandas Series: https://pandas.pydata.org/docs/reference/api/pandas.Series.html

.. _scipy.optimize.curve_fit: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

.. _Water concept: https://en.wikipedia.org/wiki/Water
.. _Water triple point: https://en.wikipedia.org/wiki/Properties_of_water#/media/File:Phase_diagram_of_water.svg
.. _WATex: https://github.com/WEgeophysics/watex/

.. _pycsamt: https://github.com/WEgeophysics/pycsamt
.. |MT| replace:: Magnetetolluric 
.. |AMT| replace:: Audio-Magnetotellurics 
.. |CSAMT| replace:: Controlled Source |AMT| 
.. |NSAMT| replace:: Natural Source |AMT| 
.. |EM| replace:: electromagnetic
.. |EMAP| replace:: |EM| array profiling

"""

from collections import namedtuple 


_core_params= dict ( 
    data ="""
data: str, filepath_or_buffer or :class:`pandas.core.DataFrame`
    Path -like object or Dataframe. If data is given as path-like object,
    data is read, asserted and validated. Any valid string path is acceptable. 
    The string could be a URL. Valid URL schemes include http, ftp, s3, gs, and
    file. For file URLs, a host is expected. A local file could be a
    file://localhost/path/to/table.csv. If you want to pass in a path object, 
    pandas accepts any :code:`os.PathLike`. By file-like object, we refer to 
    objects with a `read()` method, such as a file handle e.g. via builtin 
    `open` function or `StringIO`.
    """
    
) 
_core_returns = dict ( 
    ax="""
:class:`matplotlib.axes.Axes`
    The matplotlib axes containing the plot.
    """,
    facetgrid="""
:class:`FacetGrid`
    An object managing one or more subplots that correspond to conditional data
    subsets with convenient methods for batch-setting of axes attributes.
    """,
    jointgrid="""
:class:`JointGrid`
    An object managing multiple subplots that correspond to joint and marginal axes
    for plotting a bivariate relationship or distribution.
    """,
    pairgrid="""
:class:`PairGrid`
    An object managing multiple subplots that correspond to joint and marginal axes
    for pairwise combinations of multiple variables in a dataset.
    """, 
 )


_ReturnDocs = namedtuple('_ReturnDocs', tuple (_core_returns.keys()))
returnDocs = _ReturnDocs( * [ v  for v in _core_returns.values()] )


_Docstrings = namedtuple('_Docstrings', tuple (_core_params.keys()))
paramDocs = _Docstrings( * [ v  for v in _core_params.values()] )


class sklearn: 
    """ 
    Machine Learning in Python
    
    Scikit-learn (Sklearn) is the most useful and robust library for machine 
    learning in Python. It provides a selection of efficient tools for machine 
    learning and statistical modeling including classification, regression, 
    clustering and dimensionality reduction via a consistence interface in Python. 
    This library, which is largely written in Python, is built upon NumPy, SciPy 
    and Matplotlib.
    
    It was originally called scikits.learn and was initially developed by David 
    Cournapeau as a Google summer of code project in 2007. Later, in 2010, 
    Fabian Pedregosa, Gael Varoquaux, Alexandre Gramfort, and Vincent Michel,
    from FIRCA (French Institute for Research in Computer Science and Automation), 
    took this project at another level and made the first public release 
    (v0.1 beta) on 1st Feb. 2010. At this time, itts version history is listed above 

        * May 2019: scikit-learn 0.21.0
        * March 2019: scikit-learn 0.20.3
        * December 2018: scikit-learn 0.20.2        
        * November 2018: scikit-learn 0.20.1        
        * September 2018: scikit-learn 0.20.0        
        * July 2018: scikit-learn 0.19.2        
        * July 2017: scikit-learn 0.19.0        
        * September 2016. scikit-learn 0.18.0        
        * November 2015. scikit-learn 0.17.0        
        * March 2015. scikit-learn 0.16.0        
        * July 2014. scikit-learn 0.15.0        
        * August 2013. scikit-learn 0.14
    
    Installation
    -------------
    If you already installed NumPy and Scipy, following are the two easiest 
    ways to install scikit-learn. Following command can be used to install 
    scikit-learn via::
        
        * Using pip
        
        :code:`pip install -U scikit-learn`
        
        * Using conda
        
        :conda:`install scikit-learn`
        
    On the other hand, if NumPy and Scipy is not yet installed on your Python 
    workstation then, you can install them by using either pip or conda.
    
    Another option to use scikit-learn is to use Python distributions like 
    Canopy and Anaconda because they both ship the latest version of scikit-learn.
    
    References 
    ----------
    .. https://scikit-learn.org/stable/index.html
    
    """
    
class xgboost: 
    """
    Extreme Gradient Boosting
    
    XGBoost XgBoost stands for Extreme Gradient Boosting, is an open-source 
    software library that implements optimized distributed gradient boosting 
    machine learning algorithms under the Gradient Boosting framework.
    
    XgBoost, which was proposed by the researchers at the University of 
    Washington. It is a library written in C++ which optimizes the training for 
    Gradient  Boosting. Before understanding the XGBoost, we first need to 
    understand the trees especially the decision tree. 
    
    Indeed , a Decision tree(DT) is a flowchart-like tree structure, where 
    each internal node denotes a test on an attribute, each branch represents 
    an outcome of the test, and each leaf node (terminal node) holds a class 
    label. A tree can be 'learned' by splitting the source set into subsets 
    based on an attribute value test. This process is repeated on each derived 
    subset in a recursive manner called recursive partitioning. The recursion 
    is completed when the subset at a node all has the same value of the target 
    variable, or when splitting no longer adds value to the predictions.
    
    References 
    -----------
    ..[1] https://www.geeksforgeeks.org/xgboost/
    ..[2] https://www.nvidia.com/en-us/glossary/data-science/xgboost/
    
    """
    

