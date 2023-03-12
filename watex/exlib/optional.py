# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

import warnings 
from ..utils._dependency import import_optional_dependency 

xgboostdoc = type ('xgboostdoc', (), dict (
    __doc__= """\
    Extreme Gradient Boosting
    
    XGBoost stands for Extreme Gradient Boosting, is an open-source 
    software library that implements optimized distributed gradient boosting 
    machine learning algorithms under the Gradient Boosting framework.
    
    XgBoost, which was proposed by the researchers at the University of 
    Washington. It is a library written in C++ which optimizes the training for 
    Gradient  Boosting [1]_. Before understanding the XGBoost, we first need to 
    understand the trees especially the decision tree. 
    
    Indeed , a Decision tree(DT) is a flowchart-like tree structure, where 
    each internal node denotes a test on an attribute, each branch represents 
    an outcome of the test, and each leaf node (terminal node) holds a class 
    label. A tree can be 'learned' by splitting the source set into subsets 
    based on an attribute value test. This process is repeated on each derived 
    subset in a recursive manner called recursive partitioning. The recursion 
    is completed when the subset at a node all has the same value of the target 
    variable, or when splitting no longer adds value to the predictions [2]_.
    
    References 
    -----------
    ..[1] https://www.geeksforgeeks.org/xgboost/
    ..[2] https://www.nvidia.com/en-us/glossary/data-science/xgboost/
    
    """
    )
)
    
IS_GBM = False
#XXX TODO replace the doi by the hydrology paper doi
extra =("'xgboost' is one the pretrained models stored in the watex package"
        " especially for flow rate (FR) prediction by implementing a new"
        " paradigm for boosting the FR. It is needed to fetch the package"
        " premodels. Refer to :doi:`https://doi.org/10.1029/2021wr031623`"
        " for further details."
        )
import_optional_dependency ("xgboost", extra=extra, min_version="1.2.0" ) 
try : 
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category =FutureWarning)
        import xgboost 
except ( ImportError, ModuleNotFoundError ): 
    warnings.warn("Missing Gradient Boosting Machine <'xgboost'> module"
         " Use pip or conda for its installation.")
else :IS_GBM =True 
 
if IS_GBM: 
    from xgboost import XGBClassifier 
    
    
__all__= [ 'xgboost', 'xgboostdoc', 'XGBClassifier'] 