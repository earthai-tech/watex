# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Wed Jul  7 22:23:02 2021 hz
# This module is part of the WATex viewer package, which is released under a
# MIT- licence.

"""
 .. Synopsis:: Module ``plot`` collects essentailly deals with all plots into  
     the whole packages. 

Created on Tue Jul  13 15:48:14 2021

@author: ~alias @Daniel03

"""
import os ,re, warnings
import functools 
import numpy as np 
import matplotlib as mpl 
import  matplotlib.pyplot  as plt

import matplotlib.cm as cm 
import matplotlib.colorbar as mplcb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, NullLocator
import matplotlib.gridspec as gspec


from watex.utils._watexlog import watexlog
_logger=watexlog.get_watex_logger(__name__)




class Quick_plot : 
    """
    Special class deals with analusis modules. To quick plot diagrams, 
    histograms and bar plots.
    
    ==================  =======================================================
    Key Words           Description        
    ==================  =======================================================
    fig_dpi             dots-per-inch resolution of the figure
                        *default* is 300
    fig_num             number of the figure instance
                        *default* is 'Mesh'
    fig_size            size of figure in inches (width, height)
                        *default* is [5, 5]
    fs                  size of font of axis tick labels, axis labels are
                        fs+2. *default* is 6 
    ls                  [ '-' | '.' | ':' ] line style of mesh lines
                        *default* is '-'
    marker              marker of stations 
                        *default* is r"$\blacktriangledown$"
    ms                  size of marker in points. *default* is 5
    ==================  =======================================================
    
    """
    def __init__(self, data =None , **kwargs): 
        
        self._logging =watexlog().get_watex_logger(self.__class__.__name__)
        
        self.fig_num= kwargs.pop('fig_num', 1)
        self.fig_size = kwargs.pop('fig_size', [12,8])
        self.fig_dpi =kwargs.pop('fig_dpi', 300)
        
        self.fig_title =kwargs.pop('title', None)
        
        self.x_minorticks=kwargs.pop('xminorticks', 1)
        self.y_minorticks =kwargs.pop('yminorticks', 1)
        
        self.font_size =kwargs.pop('font_size',3.)
        self.font_style=kwargs.pop('font_style', 'italic')
        self.fs =kwargs.pop('fs', 2.)
        
        
        self.mstyle =kwargs.pop('maker_style', 'o')
        self.ms =kwargs.pop('ms', 3)
        self.mplfont =kwargs.pop('font','cursive')
        self.markerfacecolor=kwargs.pop('markefacecolor', 'r')
        self.markeredgecolor=kwargs.pop('markeredgecolor', 'gray')
        
        self.fontweight =kwargs.pop('font_weight', 'bold')
        self.ls= kwargs.pop('ls', '-')
        self.lw =kwargs.pop('lw', 1.5)
        self.alpha = kwargs.pop('alpha', 0.5)
        
        self.xlim =None 
        self.ylim=None 
        
        for key in kwargs.keys(): 
            setattr(self, key, kwargs[key])
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        