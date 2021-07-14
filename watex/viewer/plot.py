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
import pandas as pd
import matplotlib as mpl 
import  matplotlib.pyplot  as plt

import watex.viewer.hints as hints

from typing import Generic, TypeVar 

T=TypeVar('T', dict, list, tuple)

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
        self.fig_size = kwargs.pop('fig_size', [12,6])
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
        
        self.sns_orient =kwargs.pop('orient', 'v')
        self.sns_height =kwargs.pop ('sns_height', 4.)
        self.sns_aspect =kwargs.pop ('sns_aspect', .7)
        
        
        for key in kwargs.keys(): 
            setattr(self, key, kwargs[key])

        if self._data_fn is not None: 
            self.data_fn = self._data_fn 
        
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
        
        plt.figure(figsize =self.fig_size)
        plt.hist(df_[target_name], bins=self.bins ,
                  stacked = self.stacked , color= self.lc)

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.fig_title)

        if self.savefig is not None :
            plt.savefig(self.savefig,
                        dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
        
    def bar_cat_distribution(self, df=None, data_fn =None, 
                              target_name : str ='flow',
                              basic_plot: bool = True,
                              groupby: Generic[T]=None, **kws):
        """
        Bar plot distribution. Can plot a distribution according to 
        the occurence of the `target` in the data and other parameters 
        
        :param df: DataFrame of data containers 
        :param target_name: Name of the target 
        :param data_fn:  see :doc:`watex.viewer.plot.QuickPlot` documentation.
        
        :param basic_pot: Plot only the occurence of targetted columns. 
        :param specific_plot: 
            
            Plot others features located in the df columns. The plot features
            can be on ``list`` and use default plot properties. To customize 
            plot provide the features on ``dict`` with convenients properties 
            like::

                - `groupby`= ['shape', 'type'] #{'type':{'color':'b',
                                             'width':0.25 , 'sep': 0.}
                                     'shape':{'color':'g', 'width':0.25, 
                                             'sep':0.25}}
        :Example: 
            
            >>> from watex.viewer.plot import QuickPlot
            >>> qplotObj = QuickPlot(
            ...    data_fn ='data/geo_fdata/BagoueDataset2.xlsx' , lc='b', 
            ...             target_name = 'flow', set_theme ='darkgrid')
            >>> qplotObj.bar_cat_distribution(basic_plot =False, 
            ...                                  groupby=['shape' ],
            ...                          xlabel ='Anomaly type ',
            ...                          ylabel='Number of  occurence (%)' )
        """

        if df is not None : self.df = df 
        if target_name is not None : self.target_name =target_name
        if data_fn is not None : 
            self._data_fn = data_fn 
      
        if self._data_fn is not None :
            self.data_fn = self._data_fn   
            
        for plst, plsval in zip(['basic_plot', 'groupby'], 
                 [basic_plot, groupby]): 
            if not hasattr(self, plst): 
                setattr(self, plst, plsval)
    
        for key in kws.keys(): 
            setattr(self, key , kws[key])

        fig, ax = plt.subplots(figsize = self.fig_size)
        
        df_= self.df.copy(deep=True)  #make a copy for safety 
        df_.reset_index(inplace =True)
        
        if self.groupby is None:
            mess= ''.join([
                'Basic plot is turn to``False`` but no specific plot is', 
                "  detected. Please provide a specific column's into "
                " a `specific_plot` argument."])
            self._logging.debug(mess)
            warnings.warn(mess)
            self.basic_plot =True
            
        if self.basic_plot : 
            ax.bar(list(set(df_[self.target_name])), 
                        df_[self.target_name].value_counts(normalize =True),
                        label= self.fig_title, color = self.lc)  
    
        if self.groupby is not None : 
            if hasattr(self, 'set_theme'): 
                sns.set_style(self.set_theme)
            if isinstance(self.groupby, str): 
                self.groupby =[self.groupby]
            if isinstance(self.groupby , dict):
                self.groupby =list(self.groupby.keys())
            for sll in self.groupby :
                ax= sns.countplot(x= sll,  hue=self.target_name, 
                                  data = df_, 
                              orient = self.sns_orient, ax=ax )

        ax.set_xlabel(self. xlabel)
        ax.set_ylabel (self.ylabel)
        ax.set_title(self.fig_title)
        ax.legend() 
        plt.show()
        
    def multi_cat_distribution(self, df =None, data_fn =None,  x_features=None ,
                               targets=None, y_features=None, kind:str='count',
                               **kws): 
        """
        Multiple categorials plots  from tragetted pd.series. 
        
        `x_features` , `y_features` as well as `targets` must be among the 
        dataframe. 
        
        :param df: refer to :doc:`watex.viewer.plot.QuickPlot`
        :param data_fn: see :doc:`watex.viewer.plot.QuickPlot`
        
        :param x_features: 
            x-axis features. More than 01, put the x_features 
            on a list. 
        :param y_features: 
            y_axis features matchs the columns name for `sns.catplot`.
            If number of feature is more than one, create a list to hold 
            all features. 
        :param targets: 
            corresponds to `sns.catplot` argument ``hue``. If more than one 
            set the targets on a list. 
        
        :Example: 
            
            >>> from watex.viewer.plot import QuickPlot 
            >>> qplotObj = QuickPlot(
            ...    data_fn ='data/geo_fdata/BagoueDataset2.xlsx' , lc='b', 
            ...             target_name = 'flow', set_theme ='darkgrid')
            >>> fdict={
            ...            'x_features':['shape', 'type', 'type'], 
            ...            'y_features':['type', 'geol', 'shape'], 
            ...            'targets':['flow', 'flow', 'geol'],
            ...            } 
            >>>    qplotObj.multi_cat_distribution(**fdict)  
            
        """
        features_dict=kws.pop('features_dict',None )
        
        for key in kws.keys(): 
            setattr(self, key, kws[key])
        
        if data_fn is not None : 
            self.data_fn = data_fn
            
        minlen=9999999
        if features_dict is None : 
            features_dict ={ feature: featvalue  for
                            feature, featvalue in zip(
                                ['x_features', 'y_features', 'targets'],
                            [x_features, y_features, targets] )}
        
        if features_dict is not None : 
            for ffn, ffval in features_dict.items():

                if ffval is None: 
                    warnings.warn(f'Need `{ffn}` value for multiple '
                                  'categorial plotting.')
                    raise Wex.WATexError_plot_featuresinputargument(
                        f'Need `{ffn}` value for multiple categorial plots.')
                if isinstance(ffval, str): 
                    ffval=[ffval]
                    
                if minlen > len(ffval): 
                    minlen= len(ffval)
            features_dict ={ feature: featvalue[:minlen] for
                            feature, featvalue in zip(
                                ['x_features', 'y_features', 'targets'],
                            [x_features, y_features, targets] )}
        
        df_= self.df.copy(deep=True)
        df_.reset_index(inplace=True )
         
        if not hasattr(self, 'ylabel'): 
            self.ylabel= 'Number of  occurence (%)'
        for ii in range(minlen): 
            sns.catplot( data = df_, kind= kind, 
                    x=features_dict ['x_features'][ii], 
                    col= features_dict['y_features'][ii], 
                    hue=features_dict['targets'][ii], linewidth = self.lw, 
                    height = self.sns_height, aspect = self.sns_aspect
                    ).set_ylabels(self.ylabel)

            plt.show()
            
    def plot_correlation_matrix(self, df=None, data_fn =None, 
                                feature_names=None, plot_params:str ='qual',
                                 target:str =None, **sns_kws): 
        """
        Method to quick plot the qualitatif and quantitatives parameters. 
        
        Set `quant_feature_names` by providing the quantitative features' name
        as well as the `qual_feature_names`. If none value is provided, It 
        assumes to considers on groundwater exploration. If not a case , an 
        errors will raise. 
        
        :param df: refer to :doc:`watex.viewer.plot.QuickPlot`
        :param data_fn: see :doc:`watex.viewer.plot.QuickPlot`
        
        :param feature_names: List of features to plot correlations. 
        
        :param plot_params: The typle of parameters plot when `feature_names`
                is set to ``None``. `plot_params` argument must be `quan` and
                `qual` for quantitative and qualitative features respectively."
        
        :Note: One than one features, see `qual_feature_names` and 
                `quant_feature_names` on list. 
                
        :params sns_kws: Other seabon heatmap arguments. Refer to 
                https://seaborn.pydata.org/generated/seaborn.heatmap.html
                
        :Example: 
            
            >>> from watex.viewer.plot import QuickPlot 
            >>> qplotObj = QuickPlot(
            ...    data_fn ='data/geo_fdata/BagoueDataset2.xlsx' , lc='b', 
            ...             target_name = 'flow', set_theme ='darkgrid', 
            ...             fig_title='Qualitative features correlation')
            >>> sns_kwargs ={'annot': False, 
            ...            'linewidth': .5, 
            ...            'center':0 , 
            ...            # 'cmap':'jet_r', 
            ...            'cbar':True}
            >>> qplotObj.plot_correlation_matrix(
            ...    plot_params='quan', **sns_kwargs,)   
        """

        
        if data_fn is not None : 
            self.data_fn = data_fn
        
        df_= self.df.copy(deep=True)
        df_.reset_index(inplace=True )

        if feature_names is None: 
            if plot_params.lower().find('qual')>=0 :
                feature_names =['shape', 'type', 'geol', 'flow']
                plot_params='qual'
            elif plot_params.lower().find('quan')>=0 : 
                feature_names= list(set.difference(
                    set(df_.columns), set(['shape', 'type', 'geol', 'flow'])))
                tem=['{0}{1}{2}'.format('`{', ii, '}`') for ii in range(
                    len(feature_names))]
                
                self._logging.debug(
                    "We will consider the default quantitatives "
                    " features name {}.".format(*tem).format(
                        *feature_names))
                plot_params='quan'
            else: 
                raise Wex.WATexError_inputarguments(
                    f"Feature's name is set to ``{feature_names}``."
                    "Please provided the right `plot_params` argument "
                    "not {plot_params}."
                        )
                
        # Control the existenceof providing features into the pd.dataFramename:
        try : 
            if  hints.cfexist(features_to= feature_names,
                               features = df_.columns) is False: 
                raise Wex.WATexError_parameter_number(
                f'Parameters number of {feature_names} is  not found in the '
                ' dataframe columns ={0}'.format(list(df_.columns)))
        except: pass 

        if plot_params =='qual': 
            for ftn in feature_names: 
                df_[ftn] = df_[ftn].astype('category').cat.codes 
            ax= sns.heatmap(data = df_[list(feature_names)].corr(),
                 **sns_kws)
            ax.set_title(self.fig_title)
            ax.set_title(self.fig_title)

        elif plot_params =='quan': 
            if target is None: 
                if 'flow' in df_.columns: target ='flow'
                try: 
                    df_[target] 
                except: 
                    self._logging.error(
                        f"A given target's name `{target}`is wrong")
                else: 
                     feature_names.extend(['flow'])   
        
            sns.heatmap(data = df_[list(feature_names)].corr(),
                        **sns_kws)
 
        plt.show()
                
 
        
if __name__=='__main__': 
    qplotObj = QuickPlot(data_fn ='data/geo_fdata/BagoueDataset2.xlsx' , lc='b', 
                         target_name = 'flow', set_theme ='darkgrid', 
                         fig_title='Qualitative features correlation')

    sns_kwargs ={'annot': False, 
                'linewidth': .5, 
                'center':0 , 
                # 'cmap':'jet_r', 
                'cbar':True}
    qplotObj.plot_correlation_matrix( plot_params='quan', **sns_kwargs,)   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        