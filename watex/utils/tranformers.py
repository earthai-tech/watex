# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Wed Jul 14 20:00:26 2021
# This module is a set of transformers for data preparing. It is  part of 
# the WATex preprocessing module which is released under a MIT- licence.
"""
Created on Mon Sep  6 17:53:06 2021

@author: @Daniel03
"""
# import numpy as np 
# import pandas as pd 

from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.model_selection import train_test_split 
from sklearn.base import BaseEstimator, TransformerMixin 

import  watex.utils.ml_utils as mlfunc

class StratifiedSamplingAdderCategory( BaseEstimator, TransformerMixin ): 
    """Stratified sampling transformer based on new generated category   
    from :func:`~mlfunc.DiscretizeCategoriesforStratification` and 
    :func:~mlfunc.stratifiedUsingDiscretedCategories"""
    
    def __init__(self, apply_to=None, added_category =True, operator = 1.5, 
                 max_category=5, n_splits=1, test_size=0.2, random_state=42):
        
        self.apply_to = apply_to 
        self.added_category = added_category 
        self.operator = operator 
        self.max_category = max_category 
        self.n_splits = n_splits 
        self.test_size = test_size 
        self.random_state = random_state 
        
        #create inspection attributes
        self.inner_category_=None 
        self.statistics_=None 
        
        
    def fit(self, X, y=None): 
        """ Fit method and populated isnpections attributes 
        from hyperparameters."""
        if not self.added_category and self.apply_to is None: 
            return self 
        if self.added_category:
           self.inner_category_ = 'tem'
        X= mlfunc.DiscretizeCategoriesforStratification(
            data=X, in_cat=self.apply_to,  new_cat =self.inner_category_, 
             divby =self.operator, higherclass = self.max_category)
        
        return self
    
    def transform(self, X, y=None):
        if self.added_category is False or self.apply_to is None: 
            strat_train_set, strat_test_set = train_test_split(
                X, test_size = self.test_size ,random_state= self.random_state )
            return strat_train_set, strat_test_set 
        
        split = StratifiedShuffleSplit(self.n_splits, self.test_size, 
                                       self.random_state)
        
        for train_index, test_index in split.split(X, X[self.inner_category_]): 
            strat_train_set = X.loc[train_index]
            strat_test_set = X.loc[test_index] 
        # remove the add category into the set 
        for set in(strat_train_set, strat_test_set): 
            set.drop([self.inner_category_], axis=1, inplace =True)
            
        return strat_train_set, strat_test_set 
    

