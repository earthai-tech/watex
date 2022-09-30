# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Wed Jul  7 22:23:02 2021 hz
# This module is part of the WATex core package, which is released under a
# MIT- licence.
"""
Created on Wed Jul  7 22:23:02 2021

@author: @Daniel03
"""
import os
import warnings 
from typing import TypeVar, Iterable, Tuple, Callable
import numpy as np 
import pandas as pd

from ..utils._watexlog import watexlog  
import watex.utils.exceptions as Wex

_logger =watexlog().get_watex_logger(__name__)


class Geology: 
    """ Geology class deals with all concer ns the structures during 
    investigation sites about the boreholes or """
    
    def __init__(self, geofn = None,  **kwargs)-> None:
        self._logging =watexlog().get_watex_logger(self.__class__.__name__)
        pass
 
    
class Borehole(Geology): 
    """
    Focused on Wells and `Borehole` offered to the population. To use the data
     for prediction purpose, each `Borehole` provided must be referenced on 
     coordinates values or provided the same as the one used on `ves` or `erp` 
     file. 
    
    """
    def __init__(self,
                 geofn =None,
                 boreh_fn:str =None,
                 easting: float = None,
                 northing:float=None,
                 flow:float =None,
                 **kwargs) : 
        Geology.__init__(self, geo_fn = geofn , **kwargs)
        

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
            
 
class geo_pattern: 
    """
    Singleton class to deal with geopattern  with other modules.
    It is and exhaustive pattern dict, can be add and change.
     This pattern  will be depreacted later , to create for pyCSAMT,
    its owwn geological pattern in coformity with the conventional 
    geological swatches .Deal with USGS(US Geological Survey ) swatches
     - references and FGDC (Digital cartographic Standard for Geological
      Map Symbolisation-FGDCgeostdTM11A2_A-37-01cs2.eps).
         
    make _pattern:{'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
            /   - diagonal hatching
            \   - back diagonal
            |   - vertical
            -   - horizontal
            +   - crossed
            x   - crossed diagonal
            o   - small circle
            O   - large circle
            .   - dots
            *   - stars
    """
    pattern={
             "basement rocks" :      ['.+++++.', (.25, .5, .5)],
             "igneous rocks":        ['.o.o.', (1., 1., 1.)], 
             "duricrust"   :         ['+.+',(1., .2, .36)],
             "gravel" :              ['oO',(.75,.86,.12)],
             "sand":                 ['....',(.23, .36, .45)],
             "conglomerate"    :     ['.O.', (.55, 0., .36)],
             "dolomite" :            ['.-.', (0., .75, .23)],
             "limestone" :           ['//.',(.52, .23, .125)],
            "permafrost"  :          ['o.', (.2, .26, .75)],
             "metamorphic rocks" :   ['*o.', (.2, .2, .3)],
             "tills"  :              ['-.', (.7, .6, .9)],
             "standstone ":          ['..', (.5, .6, .9)],
             "lignite coal":         ['+/.',(.5, .5, .4)],
             "coal":                 ['*.', (.8, .9, 0.)],
             "shale"   :             ['=', (0., 0., 0.7)],
             "clay"   :              ['=.',(.9, .8, 0.8)],
             "saprolite" :           ['*/',(.3, 1.2, .4)],
             "sedimentary rocks":    ['...',(.25, 0., .25)],
             "fresh water"  :        ['.-.',(0., 1.,.2)],
             "salt water"   :        ['o.-',(.2, 1., .2)],
             "massive sulphide" :     ['.+O',(1.,.5, .5 )],
             "sea water"     :       ['.--',(.0, 1., 0.)],
             "ore minerals"  :       ['--|',(.8, .2, .2)],
             "graphite"    :         ['.++.',(.2, .7, .7)],
                        }
 
def get_color_palette (RGB_color_palette): 
    """
    Convert RGB color into matplotlib color palette. In the RGB color 
    system two bits of data are used for each color, red, green, and blue. 
    That means that each color runson a scale from 0 to 255. Black  would be
     00,00,00, while white would be 255,255,255. Matplotlib has lots of
     pre-defined colormaps for us . They are all normalized to 255,
    so they run from 0 to 1. So you need only normalize data, 
    then we can manually  select colors from a color map  

    :param RGB_color_palette: str value of RGB value 
    :type RGB_color_palette: str 
        
    :returns: rgba, tuple of (R, G, B)
    :rtype: tuple
     
    :Example: 
        
        >>> from watex.core.geology import get_color_palette 
        >>> get_color_palette (RGB_color_palette ='R128B128')
    """   
    def ascertain_cp (cp): 
        if cp >255. : 
            warnings.warn(
                ' !RGB value is range 0 to 255 pixels , '
                'not beyond !. Your input values is = {0}.'.format(cp))
            raise Wex.WATexError_parameter_number(
                'Error color RGBA value ! '
                'RGB value  provided is = {0}.'
                ' It is larger than 255 pixels.'.format(cp))
        return cp
    if isinstance(RGB_color_palette,(float, int, str)): 
        try : 
            float(RGB_color_palette)
        except : 
              RGB_color_palette= RGB_color_palette.lower()
             
        else : return ascertain_cp(float(RGB_color_palette))/255.
    
    rgba = np.zeros((3,))
    
    if 'r' in RGB_color_palette : 
        knae = RGB_color_palette .replace('r', '').replace(
            'g', '/').replace('b', '/').split('/')
        try :
            _knae = ascertain_cp(float(knae[0]))
        except : 
            rgba[0]=1.
        else : rgba [0] = _knae /255.
        
    if 'g' in RGB_color_palette : 
        knae = RGB_color_palette .replace('g', '/').replace(
            'b', '/').split('/')
        try : 
            _knae =ascertain_cp(float(knae[1]))
        except : 
            rgba [1]=1.
            
        else :rgba[1]= _knae /255.
    if 'b' in RGB_color_palette : 
        knae = knae = RGB_color_palette .replace('g', '/').split('/')
        try : 
            _knae =ascertain_cp(float(knae[1]))
        except :
            rgba[2]=1.
        else :rgba[2]= _knae /255.
        
    return tuple(rgba)       
if __name__=='__main__':
    none =None 
        


