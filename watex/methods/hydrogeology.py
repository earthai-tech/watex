# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Created date: Sep 19 09:04:21 2022

"""
Hydrogeological module 
========================
Hydrogeological parameters of aquifer are the essential and crucial basic data 
in the designing and construction progress of geotechnical engineering and 
groundwater dewatering, which are directly related to the reliability of these 
parameters.

Created on Mon Sep 19 09:04:21 2022

"""

from __future__ import ( 
    division, 
    annotations 
    )

from .._watexlog import watexlog 


class Hydrogeology :
    """ 
    A branch of geology concerned with the occurrence, use, and functions of 
    surface water and groundwater. 
    
    Hydrogeology is the study of groundwater – it is sometimes referred to as
    geohydrology or groundwater hydrology. Hydrogeology deals with how water 
    gets into the ground (recharge), how it flows in the subsurface 
    (through aquifers) and how groundwater interacts with the surrounding soil 
    and rock (the geology).
    
    
    see also
    ---------

    Hydrogeologists apply this knowledge to many practical uses. They might:
        
        * Design and construct water wells for drinking water supply, irrigation 
            schemes and other purposes;
        * Try to discover how much water is available to sustain water supplies 
            so that these do not adversely affect the environment – for example, 
            by depleting natural baseflows to rivers and important wetland 
            ecosystems;
        * Investigate the quality of the water to ensure that it is fit for its 
            intended use; 
        * Where the groundwater is polluted, they design schemes to try and 
            clean up this pollution;
            Design construction dewatering schemes and deal with groundwater 
            problems associated with mining; Help to harness geothermal energy
            through groundwater-based heat pumps.
    """
    
    def __init__(self, **kwds): 
        self._logging = watexlog.get_watex_logger(self.__class__.__name__)

