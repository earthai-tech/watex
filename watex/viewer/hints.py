# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Wed Jul 14 20:00:26 2021
# This module is part of the WATex viewer package, which is released under 
# the MIT- licence.

"""
Created on Wed Jul 14 20:00:26 2021

@author: @Daniel03

"""

from typing import TypeVar, Generic, Iterable 

T=TypeVar('T', list , tuple, dict, float, int)
K= TypeVar('K', complex, float)

import os
import re
import warnings 
import inspect 

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
        
        Format a generic object using the number of composed items. 
    
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

def findDifferenceGenObject(gen_obj1: Generic[T], gen_obj2: Generic[T]
                              )-> Generic[T]: 
    """
     Desciption: 
         
        Find the difference of generic object and keep the shortest len 
        object `type` at the be beginning: 
     
    Usage:

        todo: write usage
        
    :param gen_obj1: Can be a ``list``, ``dict`` or other `TypeVar` 
        classified objects.
    :param gen_obj2: Idem for `gen_obj1`.
    
    :Example: 
        
        >>> from watex.viewer.hints import findDifferenceGenObject
        >>> findDifferenceGenObject(
        ...    ['ohmS', 'lwi', 'power', 'id', 'sfi', 'magnitude'], 
        ...    {'ohmS', 'lwi', 'power'})
        [out]:
        ...  {'ohmS', 'lwi', 'power'}
    
    """
    if len(gen_obj1) < len(gen_obj2):
        objType = type(gen_obj1)
        return objType(set(gen_obj2).difference(set(gen_obj1)))
    elif len(gen_obj1) > len(gen_obj2):
        objType = type(gen_obj2)
        return objType(set(gen_obj1).difference(set(gen_obj2)))
    else: return 
   
        
    
    return set(gen_obj1).difference(set(gen_obj2))
    
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
        
def controlExistingEstimator(estimator_name: T= str ) -> T: 
    """ 
    Description: 
        When estimator name is provided by user , will chech the prefix 
        corresponding
        
    Usage: 
        
        Catching estimator name and find the corresponding prefix 
        
    :param estimator_name: Name of given estimator 
    
    :Example: 
        
        >>> from watex.viewer.hints import controlExistingEstimator 
        >>> test_est =controlExistingEstimator('svm')
        ('svc', 'SupportVectorClassifier')
        
    """
    estimator_name = estimator_name.lower()
    _estimator ={
            'dtc': ['DecisionTreeClassifier', 'dtc', 'dec'],
            'svc': ['SupportVectorClassifier', 'svc', 'sup', 'svm'],
            'sdg': ['SGDClassifier','sdg', 'sdg'],
            'knn': ['KNeighborsClassifier','knn''kne'],
            'rdf': ['RandomForestClassifier', 'rdf', 'ran', 'rfc'],
            'ada': ['AdaBoostClassifier','ada', 'adc'],
            'vtc': ['VotingClassifier','vtc', 'vot'],
            'bag': ['BaggingClassifier', 'bag', 'bag'],
            'stc': ['StackingClassifier','stc', 'sta'],
            }
    estfull = [ e_key[0] for e_key in _estimator.values()]
    
    full_estimator_name =None 
    
    for estim_key, estim_val in _estimator.items(): 
        if estimator_name == estim_key : 
            full_estimator_name = estim_val[0]
            return estim_key , full_estimator_name 
        
        elif estimator_name != estim_key : 
            for s_estim in estim_val : 
                if re.match(r'^{}+'.format(estimator_name),
                            s_estim.lower()): 
                    full_estimator_name = estim_val[0]
                    return estim_key , full_estimator_name 
    
    if full_estimator_name is None : 
        __logging.error(
            f'Estimator `{estimator_name}` not found in the default '
            ' list {}'.format(format_generic_obj(estfull)).format(*estfull))
        warnings.warn(
            f'Estimator `{estimator_name}` not found in the default estimators'
            ' list {}'.format(format_generic_obj(estfull)).format(*estfull))
        return 
    
def formatModelScore(model_score: T=None, select_estimator:str=None )   : 
    """
    Description : 
        Format the result of `model_score`
        
    :param model_score: Can be float of dict of float where key is 
                        the estimator name 
    :param select_estimator: Estimator name 
    
    :Example: 
        
        >>> from watex.viewer.hints import formatModelScore 
        >>>  formatModelScore({'DecisionTreeClassifier':0.26, 
                      'BaggingClassifier':0.13}
        )
    """ 
    print('-'*77)
    if isinstance(model_score, dict): 
        for key, val in model_score.items(): 
            print('> {0:<30}:{1:^10}= {2:^10} %'.format( key,' Score', round(
                val *100,3 )))
    else : 
        if select_estimator is None : 
            select_estimator ='___'
        if inspect.isclass(select_estimator): 
            select_estimator =select_estimator.__class__.__name__
        
        try : 
            _, select_estimator = controlExistingEstimator(select_estimator)
        
        except : 
            if select_estimator is None :
                select_estimator =str(select_estimator)
            else: select_estimator = '__'
            
        print('> {0:<30}:{1:^10}= {2:^10} %'.format(select_estimator,
                     ' Score', round(
            model_score *100,3 )))
        
    print('-'*77)
    
  
if __name__=='__main__': 
    
    #ss= format_generic_obj({'ohmS', 'lwi', 'power', 'id', 'sfi', 'magnitude'})
    # obj1 = ['ohmS', 'lwi', 'power', 'id', 'sfi', 'magnitude']
    # obj2= ['ohmS', 'lwi', 'power']
    # op= findDifferenceGenObject(gen_obj1=obj1, gen_obj2=obj2)
    # print(op)
    
    # sop_est, otp =controlExistingEstimator('SVC')
    # print(sop_est)

    # print(len(char))
    
    # formatModelScore({'DecisionTreeClassifier':0.26, 
    #                   'BaggingClassifier':0.13}
    #     )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
