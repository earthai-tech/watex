# -*- coding: utf-8 -*-
# Copyright (c) 2022 LKouadio a.k.a Daniel <etanoyau@gmail.com>
# Created on Thu Sep 29 08:30:12 2022 
# Licence: MIT
from __future__ import ( 
    print_function , annotations)

from ..typing import ( 
    Optional , 
    ArrayLike , 
    NDArray, 
    DataFrame, 
    Series 
)
from .geology import ( 
    Geology 
    )

class Borehole(Geology): 
    """
    Focused on Wells and `Borehole` offered to the population. To use the data
    for prediction purpose, each `Borehole` provided must be referenced on 
    coordinates values or provided the same as the one used on `ves` or `erp` 
    file. 
    
    """
    def __init__(self,
                 lat:float = None, 
                 lon:float = None, 
                 area:str = None, 
                 status:str =None, 
                 depth:float = None, 
                 base_depth:float =None, 
                 geol:str=None, 
                 staticlevel:float =None, 
                 airlift:float =None, 
                 id=None, 
                 qmax =None, 
                 **kwds) : 
       super().__init__(**kwds)
        
       self.lat=lat 
       self.lon=lon 
       self.area=area 
       self.status=status 
       self.depth=depth 
       self.base_depth=base_depth 
       self.geol=geol 
       self.staticlevel=staticlevel 
       self.airlift =airlift 
       self.id=id 
       self.qmax=qmax 
       
       for key in list(kwds.keys()): 
           setattr (self, key, kwds[key])

    
    def fit(self,
            data: str |DataFrame | NDArray 
        )-> object: 
        """ Fit Borehole data and populate the corrsponding attributes"""
        
        self._logging.info ("fit {self.__class__.__name__!r} for corresponding"
                            "attributes. ")
        
        # self._easting =easting 
        # self._northing = northing 
        
        # self.boreh_fn =boreh_fn 
        # self.flow = flow
        
        # self.dptm = kwargs.pop('department', None)
        # self.sp= kwargs.pop('s/p', None)
        # self.nameOflocation =kwargs.pop('nameOflocation', None)
        # self._latitude = kwargs.pop('latitude', None) 
        # self._longitude = kwargs.pop('longitude ', None)
        
        # self.borehStatus = kwargs.pop('borehStatus',None  )
        # self.depth = kwargs.pop('borehDepth', None)
        # self.basementdepth =kwargs.pop('basementDepth', None)
        # self.geol = kwargs.pop('geology', None) 
        # self.staticLevel =kwargs.pop('staticLevel', None)
        # self.airLiftflow =kwargs.pop('AirliftFlow', None)
        # self.wellID =kwargs.pop('wellID', None)
        # self.qmax=kwargs.pop('Qmax', None) 
        
        # for key in list(kwargs.keys()): 
        #     setattr(self, key, kwargs[key])
            
        return self