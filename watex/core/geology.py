# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Wed Jul  7 22:23:02 2021 hz
# This module is part of the WATex core package, which is released under a
# MIT- licence.
"""
Created on Wed Jul  7 22:23:02 2021

@author: @Daniel03
"""
import os 
import numpy as np 
import pandas as pd

from typing import TypeVar, Iterable, Tuple, Callable

from watex.utils._watexlog import watexlog  

_logger =watexlog().get_watex_logger(__name__)

class Geology: 
    pass
 
class Borehole: 
    """
    Focused on Wells and `Borehole` offered to the population. To use the data
     for prediction purpose, each `Borehole` provided must be referenced on 
     coordinates values or provided the same as the one used on `ves` or `erp` 
     file. 
    
    """
    def __init__(self, boreh_fn:str  =None, easting: float  = None,
                 northing:float =None, flow:float =None,
                 **kwargs) : 
        self._logging =watexlog().get_watex_logger(self.__class__.__name__)
        
        self._easting =easting 
        self._northing = northing 
        
        self.boreh_fn =boreh_fn 
        self.flow = flow
        
        self.dptm = kwargs.pop('department', None)
        self.sp= kwargs.pop('s/p', None)
        self.nameOflocation =kwargs.pop('nameOflocation', None)
        self._latitude = kwargs.pop('latitude', None) 
        self._longitude = kwargs.pop('longitude ', None)
        
        self.borehStatus = kwargs.pop('borehStatus',None  )
        self.depth = kwargs.pop('borehDepth', None)
        self.basementdepth =kwargs.pop('basementDepth', None)
        self.geol = kwargs.pop('geology', None) 
        self.staticLevel =kwargs.pop('staticLevel', None)
        self.airLiftflow =kwargs.pop('AirliftFlow', None)
        self.wellID =kwargs.pop('wellID', None)
        self.qmax=kwargs.pop('Qmax', None) 
        
        for key in list(kwargs.keys()): 
            setattr(self, key, kwargs[key])

def test( fak, key:list , reverse: int ): 
     return key[reverse] 
 
        
if __name__=='__main__':
    none =None 
        


