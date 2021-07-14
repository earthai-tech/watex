# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Wed Jul 14 20:00:26 2021
# This module is part of the WATex viewer package, which is released under a
# MIT- licence.

"""
Created on Wed Jul 14 20:00:26 2021

@author: @Daniel03

"""

from typing import TypeVar, Generic, Iterable 

T=TypeVar('T', list , tuple, dict)

import os 
import numpy as np 
import pandas as pd

from watex.utils.exceptions import WATexError_hints as WexH
        
def cfexist(features_to: Iterable[T], 
            features:Iterable[T] )-> bool:      
    """
    Desciption: 
        
        Control features existence into another list . List or array 
        can be a dataframe columns for pratical examples.  
        
    Usage:
        
        todo: write usage
            
    :param features_to :list of array to be controlled .
    :param features: list of whole features located on array of 
                `pd.DataFrame.columns` 
    
    :returns: 
        -``True``:If the provided list exist in the features colnames 
        - ``False``: if not 

    """
    if isinstance(features_to, str): 
        features_to =[features_to]
    if isinstance(features, str): features =[features]
    
    if sorted(list(features_to))== sorted(list(
            set(features_to).intersection(set(features)))): 
        return True
    else: return False 