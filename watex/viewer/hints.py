# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Wed Jul 14 20:00:26 2021
# This module is part of the WATex viewer package, which is released under a
# MIT- licence.

"""
Created on Wed Jul 14 20:00:26 2021

@author: @Daniel03

"""

from typing import TypeVar, Generic, Iterable 

T=TypeVar('T', list , tuple, dict, float, int)
K= TypeVar('K', complex, float)

import os 
import numpy as np 
import pandas as pd

import watex.utils.exceptions as WexH
 
from watex.utils._watexlog import watexlog 

__logging =watexlog().get_watex_logger(__name__) 
      
def cfexist(features_to: Iterable[T], 
            features:Iterable[T] )-> bool:      
    """
    Desciption: 
        
        Control features existence into another list . List or array 
        can be a dataframe columns for pratical examples.  
        
    Usage:
        
        todo: test usage
            
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

def format_generic_obj(generic_obj :Generic [T])-> T: 
    """
    Desciption: 
        
        Format a generic object using the number of items composed. 
    
    Usage:
        
        todo: write usage
    :param generic_obj: Can be a ``list``, ``dict`` or other `TypeVar` 
    classified objects.
    
    :Example: 
        
        >>> from watex.hints import format_generic_obj 
        >>> format_generic_obj ({'ohmS', 'lwi', 'power', 'id', 
        ...                         'sfi', 'magnitude'})
        
    """
    
    return ['{0}{1}{2}'.format('`{', ii, '}`') for ii in range(
                    len(generic_obj))]


def findIntersectionGenObject(gen_obj1: Generic[T], gen_obj2: Generic[T]
                              )-> Generic[T]: 
    """
     Desciption: 
         
        Find the intersection of generic object and keep the shortest len 
        object `type` at the be beginning: 
     
    Usage:

        todo: write usage
        
    :param gen_obj1: Can be a ``list``, ``dict`` or other `TypeVar` 
        classified objects.
    :param gen_obj2: Idem for `gen_obj1`.
    
    :Example: 
        
        >>> from watex.hints import findIntersectionGenObject
        >>> findIntersectionGenObject(
        ...    ['ohmS', 'lwi', 'power', 'id', 'sfi', 'magnitude'], 
        ...    {'ohmS', 'lwi', 'power'})
        [out]:
        ...  {'ohmS', 'lwi', 'power'}
    
    """
    if len(gen_obj1) <= len(gen_obj2):
        objType = type(gen_obj1)
    else: objType = type(gen_obj2)

    return objType(set(gen_obj1).intersection(set(gen_obj2)))
    
def featureExistError(superv_features: Iterable[T], 
                      features:Iterable[T]) -> None:
    """
    Description:
        Catching feature existence errors.
        
    Usage: 
        
        to check error. If nothing occurs  then pass 
    
    :param superv_features: 
        list of features presuming to be controlled or supervised
        
    :param features: 
        List of all features composed of pd.core.DataFrame. 
    
    """
    for ii, supff in enumerate([superv_features, features ]): 
        if isinstance(supff, str): 
            if ii==0 : superv_features=[superv_features]
            if ii==1 :features =[superv_features]
            
    try : 
        resH= cfexist(features_to= superv_features,
                           features = features)
    except TypeError: 
        
        print(' Features can not be a NoneType value.'
              'Please set a right features.')
        __logging.error('NoneType can not be a features!')
    except :
        raise WexH.WATexError_parameter_number(
           f'Parameters number of {features} is  not found in the '
           ' dataframe columns ={0}'.format(list(features)))
    
    else: 
        if not resH:  raise WexH.WATexError_parameter_number(
            f'Parameters number is ``{features}``. NoneType object is'
            ' not allowed in  dataframe columns ={0}'.
            format(list(features)))
        

if __name__=='__main__': 
    
    #ss= format_generic_obj({'ohmS', 'lwi', 'power', 'id', 'sfi', 'magnitude'})
    obj1 = ['ohmS', 'lwi', 'power', 'id', 'sfi', 'magnitude']
    obj2= {'ohmS', 'lwi', 'power'}
    
    op= findIntersectionGenObject(gen_obj1=obj1, gen_obj2=obj2)
    print(op)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
