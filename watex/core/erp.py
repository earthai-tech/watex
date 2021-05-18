# -*- coding: utf-8 -*-
"""
===============================================================================
Copyright (c) 2021 Kouadio K. Laurent

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
===============================================================================

.. synopsis:: 'watex.core.erp'
            Module to deal with Electrical resistivity profile (ERP)
            exploration tools 


Created on Tue May 18 12:33:15 2021

@author: @Daniel03
"""
import os 
import numpy as np 


class ERP: 
    """
    Special module wich deal with Electrical Resistivity profile 
    
    """
    def __init__(self, erpData=None, horizonDis=None , rhoApp=None,**kwargs):
        self.horizonDis =horizonDis 
        self.rhoApp =rhoApp 
        
        
        self.utmX =kwargs.pop('utmX', None)
        self.utmY =kwargs.pop('utmY', None)
        
        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])
            
    @classmethod 
    def from_csv(cls, erp_fn):
        """
        Method essentially created to read file from csv , collected 
        horizontal distance value and apparent resitivy values. 
        then send to the class for computation purposes. 
        
        :param erp_fn: path_like string of CSV file 
        :type erp_fn: str 
        
        :return: horizontal distance im meters 
        :rtype: np.array of all data
        """
        
        
        
        
        
        