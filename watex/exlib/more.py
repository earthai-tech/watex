# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:27:57 2022

@author: Daniel
"""
from warnings import warn 
from ..base import is_installing 

xgboostdoc = type ('xgboostdoc', (), dict (
    __doc__= """\
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
    )
)
    
IS_GBM = False 
try : 
    import xgboost 
except : 
    warn("Gradient Boosting Machine is installing."
                  " Please wait... ")
    print('!-> Please wait for Gradient Boosting Machines to be installed...')
    IS_GBM = is_installing('xgboost')
    if IS_GBM : 
        print("!---> 'xgboost' installation is complete")
    else : 
        warn ("'xgoost' installation failed.")
        print("Fail to install 'xgboost', please install it mannualy.")
else :IS_GBM =True 
 
if IS_GBM: 
    from xgboost import XGBClassifier 
    
    
__all__= [ 'xgboost', 'xgboostdoc', 'XGBClassifier'] 