# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Wed Jul 14 20:00:26 2021
# This module is a set of transformers for data preparing. It is  part of 
# the WATex preprocessing module which is released under a MIT- licence.
"""
Created on Mon Sep  6 17:53:06 2021

@author: @Daniel03
"""
import warnings 
import numpy as np 
import pandas as pd 

from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.model_selection import train_test_split 
from sklearn.base import BaseEstimator, TransformerMixin 

import watex.utils.exceptions as Wex 
from watex.utils._watexlog import watexlog 

import  watex.utils.ml_utils as mlfunc

_logger = watexlog().get_watex_logger(__name__)

class StratifiedWithCatogoryAdder( BaseEstimator, TransformerMixin ): 
    """
    Stratified sampling transformer based on new generated category 
    from numerical attributes and return stratified trainset and test set.
    
    Arguments: 
    ---------- 
        *base_num_feature*:str, 
            Numerical features to categorize. 
        *threshold_operator*: float, 
            The coefficient to divised the numerical features value to 
            normalize the data 
        *max_category*: Maximum value fits a max category to gather all 
            value greather than. 
            
    Another way to stratify dataset is to get insights from the dataset and 
    to add a new category as additional mileage. From this new attributes,
    data could be stratified after categorizing numerical features. 
    Once data is tratified, the new category will be drop and return the 
    train set and testset stratified. For instance::  
        
        >>> from watex.utils.transformers import StratifiedWithCatogoryAdder
        >>> stratifiedNumObj= StratifiedWithCatogoryAdder('flow')
        >>> stratifiedNumObj.fit_transform(X=df)
        >>> stats2 = stratifiedNumObj.statistics_
        
    Usage::
        
        In this example, we firstly categorize the `flow` attribute using 
        the ceilvalue (see :func:`~discretizeCategoriesforStratification`) 
        and groupby other values greater than the ``max_category`` value to the 
        ``max_category`` andput in the temporary features. From this features 
        the categorization is performed and stratified the trainset and 
        the test set.
        
    Note::
        
        If `base_num_feature` is not given, dataset will be stratified using 
        prurely random sampling.
    """
    
    def __init__(self, base_num_feature=None, threshold_operator = 1., 
                 max_category=3, n_splits=1, test_size=0.2, random_state=42):

        self.base_num_feature= base_num_feature
        self.threshold_operator=  threshold_operator
        self.max_category = max_category 
        self.n_splits = n_splits 
        self.test_size = test_size 
        self.random_state = random_state 
        self.base_items_ =None 
        self.statistics_=None 
        
    def fit(self, X, y=None): 
        """ Fit method and populated inspections attributes 
        from hyperparameters."""
        return self
    
    def transform(self, X, y=None):
        
        if self.base_num_feature is None: 
            return train_test_split( X, test_size = self.test_size ,
                                    random_state= self.random_state )
        if self.base_num_feature is not None:
            in_c= 'temf_'
            # discretize new added category from threshold
            X = mlfunc.discretizeCategoriesforStratification(
                                             X,
                                            in_cat=self.base_num_feature, 
                                             new_cat=in_c, 
                                             divby =self.threshold_operator,
                                             higherclass = self.max_category
                 )

            self.base_items_ = list(
            X[in_c].value_counts().index.values)
        
            split = StratifiedShuffleSplit(self.n_splits, self.test_size, 
                                           self.random_state)
            
            for train_index, test_index in split.split(X, X[in_c]): 
                strat_train_set = X.loc[train_index]
                strat_test_set = X.loc[test_index] 
                
        train_set, test_set = train_test_split( X, test_size = self.test_size,
                                   random_state= self.random_state)
        
        if self.base_num_feature is None: 
            return train_set, test_set
        
        # get statistic from `in_c` category proportions into the 
        # the overall dataset, 
        o_ =X[in_c].value_counts() /len(X)
        r_ = test_set[in_c].value_counts()\
            /len(test_set)
        s_ = strat_test_set[in_c].value_counts()\
            /len( strat_test_set)
        r_error , s_error = ((r_/ o_)-1)*100, ((s_/ o_)-1)*100
        
        self.statistics_ = np.c_[np.array(self.base_items_), 
                                 o_,
                                 r_,
                                 s_, 
                                 r_error,
                                 s_error
                                 ]
  
        self.statistics_ = pd.DataFrame(data = self.statistics_,
                            columns =[in_c, 
                                      'Overall',
                                      'Random', 
                                      'Stratified', 
                                      'Rand. %error',
                                      'strat. %error'
                                      ])
        
        # set a pandas dataframe for inspections attributes `statistics`.
        self.statistics_.set_index(in_c, inplace=True)
        
        # remove the add category into the set 
        for set in(strat_train_set, strat_test_set): 
            set.drop([in_c], axis=1, inplace =True)
            
        return strat_train_set, strat_test_set 

    
class StratifiedUsingBaseCategory( BaseEstimator, TransformerMixin ): 
    """
    Transformer to stratified dataset to have data more representativce into 
    the trainset and the test set especially when data is not large enough.
    
    Arguments: 
    ----------
        *base_column*: str or int, 
            Hyperparameters and can be index of the base mileage(category)
            for stratifications. If `base_column` is None, will return 
            the purely random sampling.
        *test_size*: float 
            Size to put in the test set 
        *random_state*: shuffled number of instance in the overall dataset. 
            default is ``42``.
    
    Usage::
        
        If data is  not large enough especially relative number of attributes
        if much possible to run therisk of introducing a significant sampling 
        biais.Therefore strafied sampling is a better way to avoid 
         a significant biais of sampling survey. For instance:: 
            
            >>> from watex.utils.ml_utils import load_data 
            >>> df = load_data('data/geo_fdata')
            >>>stratifiedObj = StratifiedUsingBaseCategory(base_column='geol')
            >>> stratifiedObj.fit_transform(X=df)
            >>> stats= stratifiedObj.statistics_

    Note::
        An :attr:`~statictics_` inspection attributes is good way to observe 
        thetest set generated using purely random sampling and using the 
        stratified sampling. The stratified sampling has category 
        ``base_column``proportions almost indentical to those in the full 
        dataset whereas the testset generated using purely random sampling 
        is quite skewed. 
    """
    
    def __init__(self, base_column =None,test_size=0.2, random_state=42):
        self._logging= watexlog().get_watex_logger(self.__class__.__name__)
        
        self.base_column = base_column 
        self.test_size = test_size  
        self.random_state = random_state
        
        #create inspection attributes
        self.statistics_=None 
        self.base_flag_ =False 
        self.base_items_= None 
        
    def fit(self, X, y=None): 
        """ Fit method and populated isnpections attributes 
        from hyperparameters."""
        return self
    
    def transform(self, X, y=None):
        """ return dataset `trainset` and `testset` using stratified sampling. 
        
        If `base_column` not given will return the `trainset` and `testset` 
        using purely random sampling.
        """
        if self.base_column is None: 
            self.stratified = False 
            self._logging.debug(
                f'Base column is not given``{self.base_column}``.Test set'
                ' will be generated using purely random sampling.')
            
        train_set, test_set = train_test_split(
                X, test_size = self.test_size ,random_state= self.random_state )
        
        if self.base_column  is not None: 
            
            if isinstance(self.base_column, (int, float)):  # use index to find 
            # base colum name. 
                self.base_column = X.columns[int(self.base_column)]
                self.base_flag_ =True 
            elif isinstance(self.base_column, str):
                # check wether the column exist into dataframe
                for elm in X.columns:
                    if elm.find(self.base_column.lower())>=0 : 
                        self.base_column = elm
                        self.base_flag_=True
                        break 
                    
        if not self.base_flag_: 
            self._logging.debug(
                f'Base column ``{self.base_column}`` not found '
                f'in `{X.columns}`')
            warnings.warn(
                f'Base column ``{self.base_column}`` not found in '
                f'{X.columns}.Test set is generated using purely '
                'random sampling')
            
        self.base_items_ = list(
            X[self.base_column].value_counts().index.values)
        
        if self.base_flag_: 
            strat_train_set, strat_test_set = \
                mlfunc.stratifiedUsingDiscretedCategories(X, self.base_column)
                
            # get statistic from `basecolumn category proportions into the 
            # the whole dataset, in the testset generated using purely random 
            # sampling and the test set generated using the stratified sampling.
            
            o_ =X[self.base_column].value_counts() /len(X)
            r_ = test_set [self.base_column].value_counts()/len(test_set)
            s_ = strat_test_set[self.base_column].value_counts()/len( strat_test_set)
            r_error , s_error = ((r_/ o_)-1)*100, ((s_/ o_)-1)*100
            
            self.statistics_ = np.c_[np.array(self.base_items_), o_,r_, s_, 
                                     r_error, s_error]
      
            self.statistics_ = pd.DataFrame(data = self.statistics_,
                                columns =[self.base_column, 'Overall', 'Random', 
                                          'Stratified', 'Rand. %error',
                                          'strat. %error'])
            
            # set a pandas dataframe for inspections attributes `statistics`.
            self.statistics_.set_index(self.base_column, inplace=True)
            
            return strat_train_set, strat_test_set 
        
        if not self.base_flag_: 
            return train_set, test_set 
        

if __name__=='__main__': 
    
    # df =pd.read_csv('data/geo_fdata/_bagoue_civ_loc_ves&erpdata.csv')
    # print(df)
    df = mlfunc.load_data('data/geo_fdata')
    
    # df.hist(bins=50, figsize=(20, 15))

    stratifiedNumObj= StratifiedWithCatogoryAdder('flow')
    stratifiedNumObj.fit_transform(X=df)
    stats2 = stratifiedNumObj.statistics_
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
