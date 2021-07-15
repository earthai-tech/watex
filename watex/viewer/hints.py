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
    
    

if __name__=='__main__': 
    
    #ss= format_generic_obj({'ohmS', 'lwi', 'power', 'id', 'sfi', 'magnitude'})
    obj1 = ['ohmS', 'lwi', 'power', 'id', 'sfi', 'magnitude']
    obj2= {'ohmS', 'lwi', 'power'}
    
    op= findIntersectionGenObject(gen_obj1=obj1, gen_obj2=obj2)
    print(op)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
