# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Wed Jul  7 22:23:02 2021 hz
# This module is part of the WATex modeling package, which is released under a
# MIT- licence.

from __future__ import print_function, division 

import os 
import numpy as np 
import pandas as pd 

from typing import TypeVar, Generic, Iterable , Callable


T= TypeVar('T', float, int)

from sklearn.model_selection import validation_curve, cross_val_score
from sklearn.model_selection import RandomizedSearchCV,  GridSearchCV
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GroupKFold 
from sklearn.model_selection import learning_curve 

from sklearn.metrics import confusion_matrix, f1_score, classification_report 

from watex.processing.sl import Processing , d_estimators__ 

import  watex.utils.exceptions as Wex 
import  watex.utils.decorator as deco
from watex.utils._watexlog import watexlog 

_logger =watexlog().get_watex_logger(__name__)


class Modeling: 
    """
    Modeling class 
    
    """
    def __init__(self, data_fn =None, df=None , **kwargs)->None: 
        self._logging = watexlog().get_watex_logger(self.__class__.__name__)
        
        self._data_fn = data_fn 
        self._df =df 
        
        self.pipelines = kwargs.pop('pipelines', None) 
        self.estimators =kwargs.pop('estimators', None) 
        self.auto= kwargs.pop('auto', False)


        self.Processing = Processing(self._data_fn, df =self._df, 
                        pipelines = self.pipelines,auto = self.auto,
                        estimators = self.estimators,
                                    ) 
       
   
        for key in list(kwargs.keys()): 
            setattr(self, key, kwargs[key])
            

if __name__=='__main__': 
    modelObj = Modeling(data_fn ='data/geo_fdata/BagoueDataset2.xlsx', 
                           auto =True)
    
    # print(modelObj.Processing.df_cache)