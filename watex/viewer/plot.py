# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Wed Jul  7 22:23:02 2021 hz
# This module is part of the WATex viewer package, which is released under a
# MIT- licence.

"""
 .. Synopsis:: Module ``plot`` collects essentially data to other sources  
          and deals with all plots into the whole packages. 
          ... 

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

import seaborn as sns 

from  watex.analysis.features import sl_analysis 

import watex.utils.exceptions as Wex
from watex.utils._watexlog import watexlog

_logger=watexlog.get_watex_logger(__name__)


class QuickPlot : 
    """
    Special class deals with analusis modules. To quick plot diagrams, 
    histograms and bar plots.
    
    Arguments: 
    ----------
            *df*: pd.core.DataFrame
                Dataframe for quick plotting. Commonly `QuickPlot` deals 
                with `:mod:`watex.analysis` package. 
            *data_fn*: str 
                Raw data for plotting. `QuickPlot` doesnt straighforwadly  
                read   the raw datafile. It calls  the module from 
                :mod:`watex.analysis.features.sl_analysis` module 
                for data reading and sanitizing data before plotting. 

    Hold others optionnal informations: 
        
    ==================  =======================================================
    Key Words           Description        
    ==================  =======================================================
    fig_dpi             dots-per-inch resolution of the figure
                        *default* is 300
    fig_num             number of the figure instance
                        *default* is 'Mesh'
    fig_size            size of figure in inches (width, height)
                        *default* is [5, 5]
    savefig             savefigure's name, *default* is ``None``
    fig_orientation     figure orientation. *default* is ``landscape``
    fig_title           figure title. *default* is ``None``
    fs                  size of font of axis tick labels, axis labels are
                        fs+2. *default* is 6 
    ls                  [ '-' | '.' | ':' ] line style of mesh lines
                        *default* is '-'
    lc                  line color of the plot, *default* is ``k``
    lw                  line weight of the plot, *default* is ``1.5``
    alpha               transparency number, *default* is ``0.5``  
    font_weight         weight of the font , *default* is ``bold``.        
    marker              marker of stations 
                        *default* is r"$\blacktriangledown$"
    ms                  size of marker in points. *default* is 5
    mstyle              style  of marker in points. *default* is ``o``.
    x_minorticks        minortick according to x-axis size and *default* is 1.
    y_minorticks        minortick according to y-axis size and *default* is 1.
    font_size           size of font in inches (width, height)
                        *default* is 3.
    font_style          style of font. *default* is ``italic``
                        
    bins                histograms element separation between two bar. 
                         *default* is ``10``. 
    xlim                limit of x-axis in plot. *default* is None 
    ylim                limit of y-axis in plot. *default* is None 
    ==================  =======================================================
    
    """
    def __init__(self, df=None, data_fn = None , **kwargs): 
        
        self._logging =watexlog().get_watex_logger(self.__class__.__name__)
        
        self._df =df 
        self._data_fn = data_fn 
        
        self.fig_num= kwargs.pop('fig_num', 1)
        self.fig_size = kwargs.pop('fig_size', [12,8])
        self.fig_dpi =kwargs.pop('fig_dpi', 300)
        self.savefig = kwargs.pop('savefig', None)
        
        self.fig_orientation =kwargs.pop('fig_orientation','landscape')
        self.fig_title =kwargs.pop('title', None)
        
        self.x_minorticks=kwargs.pop('xminorticks', 1)
        self.y_minorticks =kwargs.pop('yminorticks', 1)
        
        self.font_size =kwargs.pop('font_size',3.)
        self.font_style=kwargs.pop('font_style', 'italic')
        self.fs =kwargs.pop('fs', 2.)
        
        self.mstyle =kwargs.pop('maker_style', 'o')
        self.ms =kwargs.pop('ms', 3)
        # self.mplfont =kwargs.pop('font','cursive')
        self.markerfacecolor=kwargs.pop('markefacecolor', 'r')
        self.markeredgecolor=kwargs.pop('markeredgecolor', 'gray')
        
        self.lc = kwargs.pop('color', 'k')
        self.font_weight =kwargs.pop('font_weight', 'bold')
        self.ls= kwargs.pop('ls', '-')
        self.lw =kwargs.pop('lw', 1.5)
        self.alpha = kwargs.pop('alpha', 0.5)
        
        self.stacked = kwargs.pop('stacked', False)
        self.bins = kwargs.pop('bins', 10)
        
        self.xlim =kwargs.pop('xlim', None )
        self.ylim=kwargs.pop('y_lim', None) 
        
        for key in kwargs.keys(): 
            setattr(self, key, kwargs[key])
             
    @property 
    def df (self): 
        """ DataFrame of analysis. """
        return self._df 
    
    @df.setter 
    def df (self, dff): 
        """ Ressetting dataframe when comming from raw file. """
        if dff is not None : 
            self._df = dff

    @property 
    def data_fn(self): 
        return self._data_fn 
    
    @data_fn.setter 
    def data_fn (self, datafn):
        """ Can read the data file provided  and set the data into 
        pd.DataFrame by calling :class:~analysis.features.sl_analysis` to 
          populate convenient attributes. """
        if datafn is not None : 
            self._data_fn = datafn 

        slObj= sl_analysis(data_fn=self._data_fn, set_index=True)
        self.df= slObj.df 


    def hist_cat_distribution(self, df=None, data_fn =None, 
                              target_name : str =None,   **kws): 
        """
        Quick plot a distributions of categorized classes accoring to the 
        percentage of occurence. 
        
        :param df: Dataframe to quickplotted
        :param data_fn: 
                Datafile to be read. `QuickPlot`  will call `sl_analysis`
                module to read file and sanitizing data before plotting. 
        :param target_name: 
                Specify the `target_name` for histogram plot. If not given 
                an `UnboundLocalError` will raise.
                
        :param xlabel: Optional, x-axis label name. 
                If ``none`` will generated a defaut value ::
                    
                    xlabel= 'Flow classes in m3/h'
                    
        :param ylabel: Optional,   y-axis label name. 
                If ``none`` will generated a defaut value::
                    
                    ylabel = 'Number of  occurence (%)' 
                    
        :param fig_title: Optional, figure title `str` name. 
                If ``none`` will generated a defaut value:
                    
                    fig_title ='Distribution of flow classes(FR)'
        :Example: 
            
            >>> from watex.viewer.plot import Quick_plot 
            >>> qplotObj = QuickPlot(
            ...    data_fn ='data/geo_fdata/BagoueDataset2.xlsx' , lc='b')
            >>> qplotObj.hist_cat_distribution(target_name='flow')
        
        """
        savefig =kws.pop('savefig', None)
        xlabel =kws.pop('xlabel', None)
        ylabel =kws.pop('xlabel', None)
        fig_title =kws.pop('fig_title', None)
        
        if savefig is not None : self.savefig = savefig
        
        if df is not None : self.df = df 
        if data_fn is not None : self._data_fn = data_fn 

        if self._data_fn is not None :
            self.data_fn = self._data_fn  
            
        for attr, valattr, optval in zip(
                ['target_name', 'xlabel', 'ylabel', 'fig_title'], 
                [target_name, xlabel, ylabel, fig_title],
                ['flow', 'Flow classes in m3/h','Number of  occurence (%)',
                'Distribution of flow classes(FR)' ]
                ): 
            if not hasattr(self, attr): 
                if attr =='target_name' and attr is None: 
                    self._logging.error(
                        'No `target_name` is known. Please specify the target'
                        " column's name for `cat_distribution` plot.")
                    raise Wex.WATexError_plot_featuresinputargument(
                        'No `target_name` is detected.Please specify the target'
                        " column's name for `cat_distribution` plot.") 
                elif valattr is None : 
                   valattr = optval
                   
                setattr(self, attr, valattr)
                
        # reset index 

        df_= self.df.copy(deep=True)  #make a copy for safety 
        df_.reset_index(inplace =True)
        plt.hist(df_[target_name], bins=self.bins ,
                  stacked = self.stacked , color= self.lc)

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.fig_title)

        if self.savefig is not None :
            plt.savefig(self.savefig,
                        dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
        

if __name__=='__main__': 
    qplotObj = QuickPlot(data_fn ='data/geo_fdata/BagoueDataset2.xlsx' , lc='b')
    qplotObj.hist_cat_distribution(target_name='flow')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        