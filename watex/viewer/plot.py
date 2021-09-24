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
import os 
import re 
import warnings
import functools 
from typing import Generic, TypeVar, Iterable, Callable 

import numpy as np 
import pandas as pd
import matplotlib as mpl 
import  matplotlib.pyplot  as plt
import matplotlib.cm as cm 
import matplotlib.colorbar as mplcb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, NullLocator
import matplotlib.gridspec as gspec
import seaborn as sns 

import watex.viewer.hints as hints
from  watex.analysis.features import sl_analysis 
import watex.utils.exceptions as Wex
from watex.utils._watexlog import watexlog

T=TypeVar('T', dict, list, tuple)

_logger=watexlog.get_watex_logger(__name__)


class QuickPlot : 
    """
    Special class deals with analysis modules. To quick plot diagrams, 
    histograms and bar plots.
    
    Arguments 
    ----------
            *df*: pd.core.DataFrame
                Dataframe for quick plotting. Commonly `QuickPlot` deals 
                with `:mod:`watex.analysis` package. 
                
            *data_fn*: str 
                Raw data for plotting. `QuickPlot` doesnt straighforwadly  
                read   the raw datafile. It calls  the module from 
                :mod:`watex.analysis.features.sl_analysis` module 
                for data reading and sanitizing data before plotting.
                
            *flow_classes*: list of classes values to convert the categorical 
                    features. default is [0., 1., 3.] which means: 
                        - 0 m3/h  --> FR0
                        - > 0 to 1 m3/h --> FR1
                        - > 1 to 3 m3/h --> FR2
                        - > 3 m3/h  --> FR3

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
        self._flow_classes = kwargs.pop('flow_classes',[0., 1., 3.])
        self.target_name= kwargs.pop('target_name', 'flow')

        self.fig_num= kwargs.pop('fig_num', 1)
        self.fig_size = kwargs.pop('fig_size', [12,6])
        self.fig_dpi =kwargs.pop('fig_dpi', 300)
        self.savefig = kwargs.pop('savefig', None)
        self.fig_legend= kwargs.pop('fig_legend_kws', None)
        
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
        self.sns_style =kwargs.pop('sns_style', None)
        self.sns_palette = kwargs.pop('sns_palette', None)
        self.sns_height =kwargs.pop ('sns_height', 4.)
        self.sns_aspect =kwargs.pop ('sns_aspect', .7)
        self.sns_theme_kws = kwargs.pop('set_theme', 
                                        {'style':self.sns_style, 
                                         'palette':self.sns_palette, 
                                                      }
                                        )

        self.xlabel=kwargs.pop('xlabel', None)
        self.ylabel=kwargs.pop('ylabel', None)
        
        
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

        slObj= sl_analysis(data_fn=self._data_fn, set_index=True, 
                           flow_classes = self._flow_classes , 
                           target = self.target_name)
        self.df= slObj.df 


    def hist_cat_distribution(self, df=None, data_fn =None, 
                              target_name : str =None,   **kws): 
        """
        Quick plot a distributions of categorized classes according to the 
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
            
            >>> from watex.viewer.plot import QuickPlot 
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
        if target_name is not None: 
            self.target_name =target_name 
        
        for attr, valattr, optval in zip(
                ['target_name', 'xlabel', 'ylabel', 'fig_title'], 
                [target_name, xlabel, ylabel, fig_title],
                ['flow', 'Flow classes in m3/h','Number of  occurence (%)',
                'Distribution of flow classes(FR)' ]
                ): 

            if getattr(self, attr) is None: 
                if attr =='target_name': 
                    self._logging.error(
                        'No `target_name` is known. Please specify the target'
                        " column's name for `cat_distribution` plot.")
                    raise Wex.WATexError_plot_featuresinputargument(
                        'No `target_name` is detected.Please specify the target'
                        " column's name for `cat_distribution` plot.") 
                
                if valattr is None : 
                   valattr = optval
                   
                setattr(self, attr, valattr)
                
        self._logging('Quick plot a distributions of categorized classes.'
                      f'Target name is ={self.target_name}')      
        # reset index 

        df_= self.df.copy(deep=True)  #make a copy for safety 
        df_.reset_index(inplace =True)
        
        plt.figure(figsize =self.fig_size)
        plt.hist(df_[self.target_name], bins=self.bins ,
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
        if target_name is not None : 
            self.target_name =target_name
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
            if hasattr(self, 'sns_style'): 
                sns.set_style(self.sns_style)
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
        
        self._logging.info(
            'Multiple bar plot distribution grouped by  `{0}`.'.format(
                hints.format_generic_obj(self.groupby)).format(*self.groupby))
        
        if self.savefig is not None :
            plt.savefig(self.savefig,
                        dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
        plt.show()
        
    def multi_cat_distribution(self, df =None, data_fn =None,  x_features=None ,
                               targets=None, y_features=None, kind:str='count',
                               sns_style: str =None, **kws): 
        """
        Multiple categorials plots  from targetted pd.series. 
        
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
            corresponds to `sns.catplot` argument ``hue``. If more than one, 
            set the targets on a list. 
        
        :Example: 
            
            >>> from watex.viewer.plot import QuickPlot 
            >>> qplotObj = QuickPlot(
            ...    data_fn ='data/geo_fdata/BagoueDataset2.xlsx' , lc='b', 
            ...             target_name = 'flow', set_style ='darkgrid')
            >>> fdict={
            ...            'x_features':['shape', 'type', 'type'], 
            ...            'y_features':['type', 'geol', 'shape'], 
            ...            'targets':['flow', 'flow', 'geol'],
            ...            } 
            >>>    qplotObj.multi_cat_distribution(**fdict)  
            
        """
        
        features_dict=kws.pop('features_dict',None )
        
        if sns_style is not None: 
            self.sns_style = sns_style

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
            
        self._logging.info(
            'Multiple categorials plots  from targetted `{0}`.'.format(
                hints.format_generic_obj(targets)).format(*targets))
        
        for ii in range(minlen): 
            
            sns.catplot( data = df_, kind= kind, 
                    x=features_dict ['x_features'][ii], 
                    col= features_dict['y_features'][ii], 
                    hue=features_dict['targets'][ii],
                    linewidth = self.lw, 
                    height = self.sns_height,
                    aspect = self.sns_aspect, 
                    ).set_ylabels(self.ylabel)
            
        
            plt.show()
       
        if self.sns_style is not None: 
            sns.set_style(self.sns_style)
            
        print('--> Multiple categorials plots sucessfully done!')    
        
        
    def plot_correlation_matrix(self, df=None, data_fn =None, 
                                feature_names=None, plot_params:str ='qual',
                                 target:str =None, corr_method: str ='pearson',
                                 min_periods=1, **sns_kws) -> None: 
        """
        Method to quick plot the qualitatif and quantitatives parameters. 
        
        Set `feature_names` by providing the quantitative features as well
         as the qualitative feature names. If ``None`` value is provided, It 
        assumes to consider on groundwater exploration therefore the 
        `target` is set to ``flow``. If not the case and ``feature_names`` are 
        still ``None``, an errors raises. 

        :param df: refer to :doc:`watex.viewer.plot.QuickPlot`
        :param data_fn: see :doc:`watex.viewer.plot.QuickPlot`
        
        :param target: 
            
                Assuming for prediction purposes , `target` is useful 
                to comparae its correlation with others parameters. If ``None`` 
                will assume the survey is intended for groundwater exploration
                and will check whether `target` is among the dataFrame columns.
                If exists will set or *default* is ``flow``. 
                
        :param feature_names: List of features to plot correlations. 
        
        :param plot_params: 
            
                The typle of parameters plot when `feature_names`
                is set to ``None``. `plot_params` argument must be `quan` and
                `qual` for quantitative and qualitative features respectively."
        
        :param corr_method: correlation methods.*Default is ``pearson``
        :param min_periods: 
                Minimum number of observations required per pair of columns
                to have a valid result. Currently only available for 
                ``pearson`` and ``spearman`` correlation. For more details 
                refer to https://www.geeksforgeeks.org/python-pandas-dataframe-corr/
        
        :Note: One than one features, see `feature_names` on list. 
                
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
        # df_.reset_index(inplace=True )

        if feature_names is None: 
            if plot_params.lower().find('qual')>=0  or \
                plot_params.lower().find('cat')>=0 :
                feature_names =['shape', 'type', 'geol', 'flow']
                plot_params='qual'
            elif plot_params.lower().find('quan')>=0 or\
                plot_params.lower().find('num')>=0: 
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
                
        # Control the existence of providing features into the pd.dataFramename:
        try : 
            reH=  hints.cfexist(features_to= feature_names,
                               features = df_.columns) 
                
        except: 
            raise Wex.WATexError_parameter_number(
                f'Parameters number of {feature_names} is  not found in the '
                ' dataframe columns ={0}'.format(list(df_.columns)))
        else : 
            if reH is False: 
                raise Wex.WATexError_parameter_number(
                f'Parameters number `{feature_names}` is  not found in the '
                ' dataframe columns ={0}'.format(list(df_.columns)))

        if plot_params =='qual': 
            for ftn in feature_names: 
            
                df_[ftn] = df_[ftn].astype('category').cat.codes 
        
            ax= sns.heatmap(data = df_[list(feature_names)].corr(
                method= corr_method,min_periods=min_periods),
                 **sns_kws)
            
        elif plot_params =='quan': 
            if target is None: 
                if 'flow' in df_.columns: target ='flow'
                try: 
                    df_[target] 
                except: 
                    self._logging.error(
                        f"A given target's name `{target}`is wrong")
                else: 
                    pass
                    feature_names.extend(['flow']) 
                    df_['flow']= df_['flow'].astype('category').cat.codes
            if 'id' in feature_names: 
                feature_names.remove('id')
                df_= df_.drop('id', axis=1)

            ax= sns.heatmap(data =df_[list(feature_names)].corr(
                method= corr_method, min_periods=min_periods), 
                    **sns_kws
                    )

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(self.fig_title)
        if self.savefig is not None :
            plt.savefig(self.savefig,
                        dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
            
        fmObj = hints.format_generic_obj(feature_names)
        
        print(" --> Successfully plot of matrix correlation between "
              f"{'categorial' if plot_params =='qual' else 'numerical'}"
              " features {0}.".format(fmObj).format(*feature_names))
              
        plt.show()
                
    def plot_numerical_features(self, df=None, data_fn =None , target= None,
                                  numerical_features:Iterable[T]=None, 
                                  trigger_map_lower_kws: bool =False, 
                                  map_lower_kws: Generic[T]=None, **sns_kws): 
        """
        Plot qualitative features distribution using correlative aspect. Be 
        sure to provided numerical features arguments. 
        
        :param df: refer to :doc:`watex.viewer.plot.QuickPlot`
        :param data_fn: see :doc:`watex.viewer.plot.QuickPlot`
        
        :param numerical features: 
            List of numerical features to plot for  correlating analyses. 
        
        :param sns_kws: 
            Keywords word arguments of seabon pairplots. Refer to 
            http://seaborn.pydata.org/generated/seaborn.pairplot.html for 
            further details.             
        
        :param trigger_map_lower_kws: 
            trigger the `map_lower_kws` arguments to customize plot.
                                
        :param map_lower_kws: 
            Dictionnary of sns.pairplot map_lower kwargs arguments.
            If he diagram `kind` is ``kde`` and `trigger_map_lower_kws` is
            ``True``, plot is customized with the provided kwargs 
            `map_lower_kws` arguments. if ``None``, will check whether the 
            `diag_kind` argument on `sns_kws` is ``kde`` before applicating on 
            plot map. 
            
        :param target: Goal of prediction purposes match the `hue` argument of 
            seaborn pairplot. Please refer to the `target`  argument of 
            :func:~.viewer.plot.QuickPlot.plot_correlation_matrix` for more 
            details. If ``None``, *default* is ``flow``.
   
        :Example: 
            
            >>> from watex.viewer.plot import QuickPlot 
            >>> qkObj = QuickPlot(
            ...         data_fn ='data/geo_fdata/BagoueDataset2.xlsx', lc='b', 
            ...             target_name = 'flow', set_theme ='darkgrid', 
            ...             fig_title='Quantitative features correlation'
            ...             )  
            >>> sns_pkws={'aspect':2 , 
            ...          "height": 2, 
            ...          'markers':['o', 'x', 'D', 'H', 's'], 
            ...          'diag_kind':'kde', 
            ...          'corner':False,
            ...          }
            >>> marklow = {'level':4, 
            ...          'color':".2"}
            >>> qkObj.plot_numerical_features(trigger_map_lower_kws=True, 
            ...                                    map_lower_kws=marklow, 
            ...                                    **sns_pkws)
                                                
        """
        if data_fn is not None : 
            self.data_fn = data_fn
            
        if df is not None : self.ddf = df 
        
        df_= self.df.copy(deep=True)
        
        tem=[]
        
        if target is None : 
            if 'flow' in [it.lower() for it in df_.columns]:
                target ='flow'
                self._logging.info(
                    ' Target is  ``None``, `flow` is set instead.')
            else: 
                warnings.warn(
                    ' No target  is detected and your purpose'
                    'is not for water exploration. Could not plot numerical'
                    " features' distribution.")
                raise Wex.WATexError_geoFeatures(
                    'Target feature is missing. Could not plot numerical'
                    '  features. Please provide the right target``hue`` name.'
                    )
        elif target is not None : 
            if not target in df_.columns: 
                raise Wex.WATexError_inputarguments(
                    f"The given target {target} is wrong. Please provide the "
                    " the right target (hue)instead.")
        
        
        if target =='flow': 
            if sorted(hints.findIntersectionGenObject(
                    {'ohmS', 'power', 'sfi', 'magnitude'}, df_.columns
                    ))== sorted({'ohmS', 'power', 'sfi', 'magnitude'}):
                numerical_features= sorted({'ohmS', 'power', 'sfi', 'magnitude'})
                
            if target =='flow': 
                numerical_features.append('flow')
                # df_['flow']=df_['flow'].astype('category').cat.codes
        try : 
            resH= hints.cfexist(features_to= numerical_features,
                               features = df_.columns)
        except:
             raise Wex.WATexError_parameter_number(
                f'Parameters number of {numerical_features} is  not found in the '
                ' dataframe columns ={0}'.format(list(df_.columns)))
        
        else: 
            if not resH:  raise Wex.WATexError_parameter_number(
                f'Parameters number is ``{numerical_features}``. NoneType'
                '  object is not allowed in  dataframe columns ={0}'.
                format(list(df_.columns)))

            for ff in numerical_features:
                if ff== target: continue 
                try : 
                    df_=df_.astype({
                             ff:np.float})
                except:
                    rem=[]
                    self._logging.error(
                        f" Feature `{ff}` is not quantitive. It should be "
                        " for numerical analysis.")
                    warnings.warn(f'The given feature `{ff}` will be remove for'
                                  " numerical analysis.")
                    rem.append(ff)
                    
                else: 
                    
                    tem.append(ff)
            if len(tem)==0 : 
                raise Wex.WATexError_parameter_number(
                    " No parameter number is found. Plot is cancelled."
                    'Provide a right numerical features different'
                    ' from `{}`'.format(rem))

            numerical_features= [cc for cc in tem ] + [target] 

        ax =sns.pairplot(data =df_[numerical_features], hue=target,**sns_kws)
        
        if trigger_map_lower_kws : 
            try : 
                sns_kws['diag_kind']
         
            except: 
                self._logging.info('Impossible to set `map_lower_kws`.')
                warnings.warn(
                    '``kde|sns.kdeplot``is not found for seaborn pairplot.'
                    "Impossible to lowering the distribution map.")
            else: 
                if sns_kws['diag_kind']=='kde' : 
                    ax.map_lower(sns.kdeplot, **map_lower_kws)
                    
        plt.show()
        
        if self.savefig is not None :
            plt.savefig(self.savefig,
                        dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
   
    def joint2features(self,*, data_fn =None, df=None, 
                      features: Iterable[T]=['ohmS', 'lwi'], 
                      join_kws=None, marginals_kws=None, 
                      **sns_kwargs)-> None:
        """
        Joint methods allow to visualize correlation of two features. 
        
        Draw a plot of two features with bivariate and univariate graphs. 
        
        :param df: refer to :doc:`watex.viewer.plot.QuickPlot`
        :param data_fn: see :doc:`watex.viewer.plot.QuickPlot`
        
        :param features: 
            List of quantitative features to plot for correlating analyses.
            Can change the *default* value for your convenient data features.
            
        :param join_kws: 
            Additional keyword arguments are passed to the function used 
            to draw the plot on the joint Axes, superseding items in the 
            `joint_kws` dictionary.
            
        :param marginals_kws: 
            Additional keyword arguments are passed to the function used 
            to draw the plot on the marginals Axes. 
            
        :param sns_kwargs: 
            keywords arguments of seaborn joinplot methods. Refer to 
            :ref:`<http://seaborn.pydata.org/generated/seaborn.jointplot.html>` 
            for more details about usefull kwargs to customize plots. 
            
        :Example: 
            
            >>> from watex.viewer.plot.QuickPlot import joint2features
            >>> qkObj = QuickPlot(
            ...        data_fn ='data/geo_fdata/BagoueDataset2.xlsx', lc='b', 
            ...             target_name = 'flow', set_theme ='darkgrid', 
            ...             fig_title='Quantitative features correlation'
            ...             )  
            >>> sns_pkws={
            ...            'kind':'reg' , #'kde', 'hex'
            ...            # "hue": 'flow', 
            ...               }
            >>> joinpl_kws={"color": "r", 
                            'zorder':0, 'levels':6}
            >>> plmarg_kws={'color':"r", 'height':-.15, 'clip_on':False}           
            >>> qkObj.joint2features(features=['ohmS', 'lwi'], 
            ...            join_kws=joinpl_kws, marginals_kws=plmarg_kws, 
            ...            **sns_pkws, 
            ...            ) 
        """
        if data_fn is not None : 
            self.data_fn = data_fn
        if df is not None: self.df = df 
        
        df_= self.df.copy(deep=True)
        
        try : 
            resH= hints.cfexist(features_to= features,
                               features = df_.columns)
        except TypeError: 
            
            print(' Features can not be a NoneType value.'
                  'Please set a right features.')
            self._logging.error('NoneType can not be a features!')
        except :
            raise Wex.WATexError_parameter_number(
               f'Parameters number of {features} is  not found in the '
               ' dataframe columns ={0}'.format(list(df_.columns)))
        
        else: 
            if not resH:  raise Wex.WATexError_parameter_number(
                f'Parameters number is ``{features}``. NoneType object is'
                ' not allowed in  dataframe columns ={0}'.
                format(list(df_.columns)))
        
        if isinstance(features, str): 
            features=[features]
        # checker whether features is quantitative features 
        for ff in features: 
            try: 
                df_=df_.astype({ff:np.float})
            except ValueError: 
                raise  Wex.WATexError_geoFeatures(
                    f" Feature `{ff}` is qualitative parameter."
                    ' Could not convert string values to float')
                
        if len(features)>2: 
            self._logging.debug(
                'Features length provided is = {0}. The first two '
                'features `{1}` is used for joinplot.'.format(
                    len(features), features[:2]))
            features=list(features)[:2]
        elif len(features)<=1: 
            self._logging.error(
                'Could not jointplotted. Need two features. Only {0} '
                'is given.'.format(len(features)))
            raise Wex.WATexError_parameter_number(
                'Only {0} is feature number is given. Need two '
                'features!'.format(len(features)))
            
        ax= sns.jointplot(features[0], features[1], data=df_,  **sns_kwargs)

        if join_kws is not None:
            ax.plot_joint(sns.kdeplot, **join_kws)
        if marginals_kws is not None: 
            ax.plot_marginals(sns.rugplot, **marginals_kws)
            
        plt.show()
        
        if self.savefig is not None :
            plt.savefig(self.savefig,
                        dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
            
    def scatteringFeatures(self,data_fn=None, df=None, 
                           features:Iterable[T] =['lwi', 'flow'],
                           relplot_kws:Generic[T] = None, 
                           **sns_kwargs )->None: 
        """
        Draw a scatter plot with possibility of several semantic features 
        groupings.
        
        Indeed `scatteringFeatures` analysis is a process of understanding 
        how features in a dataset relate to each other and how those
        relationships depend on other features. Visualization can be a core 
        component of this process because, when data are visualized properly,
        the human visual system can see trends and patterns
        that indicate a relationship. 
        
        :param df: refer to :doc:`watex.viewer.plot.QuickPlot`
        :param data_fn: see :doc:`watex.viewer.plot.QuickPlot`
        
        :param features: 
            List of features to plot for scattering analyses.
            Can change the *default* value for your convenient data features.
        
        :param relplot_kws: 
            Extra keyword arguments to show the relationship between 
            two features with semantic mappings of subsets.
            refer to :ref:`<http://seaborn.pydata.org/generated/seaborn.relplot.html#seaborn.relplot>`
            for more details. 
            
        :param sns_kwargs:
            kwywords arguments to control what visual semantics are used 
            to identify the different subsets. For more details, please consult
            :ref:`<http://seaborn.pydata.org/generated/seaborn.scatterplot.html>`. 
            
        :Example: 
            
            >>> from watex.viewer.plot.QuickPlot import  scatteringFeatures
            >>> qkObj = QuickPlot(
            ...    data_fn ='data/geo_fdata/BagoueDataset2.xlsx' , lc='b', 
            ...             target_name = 'flow', set_theme ='darkgrid', 
            ...             fig_title='geol vs lewel of water inflow',
            ...             xlabel='Level of water inflow (lwi)', 
            ...             ylabel='Flow rate in m3/h'
            ...            )  
            >>> marker_list= ['o','s','P', 'H']
            >>> markers_dict = {key:mv 
            ...               for key, mv in zip( list (
            ...                       dict(qkObj.df ['geol'].value_counts(
            ...                           normalize=True)).keys()), 
            ...                            marker_list)}
            >>> sns_pkws={'markers':markers_dict, 
            ...          'sizes':(20, 200),
            ...          "hue":'geol', 
            ...          'style':'geol',
            ...         "palette":'deep',
            ...          'legend':'full',
            ...          # "hue_norm":(0,7)
            ...            }
            >>> regpl_kws = {'col':'flow', 
            ...             'hue':'lwi', 
            ...             'style':'geol',
            ...             'kind':'scatter'
            ...            }
            >>> qkObj.scatteringFeatures(features=['lwi', 'flow'],
            ...                         relplot_kws=regpl_kws,
            ...                         **sns_pkws, 
            ...                    ) 
            
        """
        if data_fn is not None : 
            self.data_fn = data_fn
            
        if df is not None: self.df = df 
        
        df_= self.df.copy(deep=True)
        
        # controller function
        try:
            hints.featureExistError(superv_features=features, 
                                    features=df_.columns)
        except: 
            warnings.warn(f'Feature {features} controlling failed!')
        else: 
            self._logging.info(
                f'Feature{features} controlling passed !')
            
        if len(features)>2: 
            self._logging.debug(
                'Features length provided is = {0}. The first two '
                'features `{1}` are used for joinplot.'.format(
                    len(features), features[:2]))
            features=list(features)[:2]
        elif len(features)<=1: 
            self._logging.error(
                'Could not scattering. Need two features. Only {0} '
                'is given.'.format(len(features)))
            raise Wex.WATexError_parameter_number(
                'Only {0} is feature is given. Need two '
                'features!'.format(len(features)))

        ax= sns.scatterplot(features[0],features[1], data=df_, **sns_kwargs)
        
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(self.fig_title)
        
        if relplot_kws is not None: 
            sns.relplot(data=df_, x= features[0], y=features[1],
                        **relplot_kws)
            
        plt.show()
        
        if self.savefig is not None :
            plt.savefig(self.savefig,
                        dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
    
    def discussingFeatures(self,df=None, data_fn=None,
                           features:Iterable[T]=['ohmS','sfi', 'geol', 'flow'],
                           map_kws:Generic[T]=None, 
                           map_func: Callable[..., T]= None, 
                           **sns_kws)-> None: 
        """
        Porvides the features names at least 04 and discuss with 
        their distribution. 
        
        This method maps a dataset onto multiple axes arrayed in a grid of
        rows and columns that correspond to levels of features in the dataset. 
        The plots it produces are often called “lattice”, “trellis”, or
        “small-multiple” graphics. 
        
        :param df: refer to :doc:`watex.viewer.plot.QuickPlot`
        :param data_fn: see :doc:`watex.viewer.plot.QuickPlot`
        
        :param features: 
            
            List of features for discussion plot. Can change the *default*
            value to the own features. The number of recommended 
            features for better analysis is four (04) classified as below: 
                
                features_disposal = ['x', 'y', 'col', 'target|hue']
                
            where: 
                - `x` is the features hold to the x-axis, *default* is``ohmS`` 
                - `y` is the feature located on y_xis, *default* is ``sfi`` 
                - `col` is the feature on column subset, *default` is ``col`` 
                - `target` or `hue` for targetted examples, *default* is ``flow``
            
            If 03 `features` are given, the latter is considered as a `target`

        :param map_kws: 
            Extra keyword arguments for mapping plot.
            
        :param func_map: 
            
            Plot style `function`. Can be a .. _matplotlib-pyplot:`mpl.plt` like
            ``plt.scatter`` or .._seaborn-scatterplot:`sns.plt` like 
            ``sns.scatterplot``. The *default* is ``sns.scatterplot``.
  
        :param sns_kwargs:
           kwywords arguments to control what visual semantics are used 
           to identify the different subsets. For more details, please consult
           :ref:`<http://seaborn.pydata.org/generated/seaborn.FacetGrid.html>`. 
        
        :Example: 
            
            >>> from viewer.plot.QuickPlot import discussingFeatures 
            >>> qkObj = QuickPlot(  fig_legend_kws={'loc':'upper right'},
            ...          fig_title = '`sfi` vs`ohmS|`geol`',
            ...            )  
            >>> sns_pkws={'aspect':2 , 
            ...          "height": 2, 
            ...                  }
            >>> map_kws={'edgecolor':"w"}   
            >>> qkObj.discussingFeatures(
            ...    data_fn ='data/geo_fdata/BagoueDataset2.xlsx' , 
            ...                         features =['ohmS', 'sfi','geol', 'flow'],
            ...                           map_kws=map_kws,  **sns_pkws
            ...                         )   
        """
        if data_fn is not None : 
            self.data_fn = data_fn
        if df is not None: 
            self.df = df 
        df_= self.df.copy(deep=True)
        
        try:
            hints.featureExistError(superv_features=features, 
                                    features=df_.columns)
        except: 
            warnings.warn(f'Feature {features} controlling failed!')
        else: 
            self._logging.info(
                f'Feature{features} controlling passed !')
        
        if len(features)>4: 
            self._logging.debug(
                'Features length provided is = {0:02}. The first four '
                'features `{1}` are used for joinplot.'.format(
                    len(features), features[:4]))
            features=list(features)[:4]
            
        elif len(features)<=2: 
            if len(features)==2:verb, pl='are','s'
            else:verb, pl='is',''
            
            self._logging.error(
                'Could not plot features. Need three features. Only {0} '
                '{1} given.'.format(len(features), verb))
            
            raise Wex.WATexError_parameter_number(
                'Only {0:02} feature{1} {2} given. Need at least 03 '
                'features!'.format(len(features),pl,  verb))
            
        elif len(features)==3:
            
            msg='03 Features are given. The last feature `{}` should be'\
                ' considered as the`targetted`feature or `hue` value.'.format(
                    features[-1])
            self._logging.debug(msg)
            
            warnings.warn(
                'The last feature `{}` is used as targetted feature!'.format(
                    features[-1]))
            
            features.insert(2, None)
    
        ax= sns.FacetGrid(data=df_, col=features[-2], hue= features[-1], 
                            **sns_kws)
        
        if map_func is None: 
            map_func = sns.scatterplot #plt.scatter
        if map_func is not None : 
            if not hasattr(map_func, '__call__'): 
                warnings.warn(f'Object {map_func} must be a callable, not {0}.'
                    'Can be `plt.scatter` or <matplotlib.pyplot|'
                    'seaborn.scatterplot>''function supported.'.format(
                    type(map_func)))
                        
                self._logging.error('{map_func} is not callable !'
                                    'Use `plt.scatter` as default function.')
                raise Wex.WATexError_inputarguments(
                    '`Argument `map_func` should be a callable not {0}'.
                    format(type(map_func)))
                
        if map_kws is None : 
            map_kws={'edgecolor':"w"}
            
        try : 
            ax.map(map_func, features[0], features[1], #edgecolor=self.edgecolor,
                   **map_kws).add_legend(**self.fig_legend)
            
        except AttributeError:
            print('`{}` object has not attribute `get`'.format(
                type(self.fig_legend)))
            
        plt.show()
        
        if self.savefig is not None :
            plt.savefig(self.savefig,
                        dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
        
    def discover_and_visualize_data(self, df = None, data_fn:str =None, 
                                    x:str =None, y:str =None, kind:str ='scatter',
                                    s_col:T ='lwi', leg_kws:dict ={}, **pd_kws):
        """ Create a scatter plot to visualize the data using `x` and `y` 
        considered as dataframe features. 
        
        :param df: refer to :doc:`watex.viewer.plot.QuickPlot`
        :param data_fn: see :doc:`watex.viewer.plot.QuickPlot`
        
        :param x: Column name to hold the x-axis values 
        :param y: column na me to hold the y-axis values 
        :param s_col: Size for scatter points. 'Default is 
            ``fs`` time the features colum `lwi`.
            
        :param pd_kws: 
            Pandas plot keywords arguments 
        :param leg_kws: Matplotlib legend keywords arguments 
        
        :Example: 
            
            >>> import watex.utils.ml_utils as mfunc
            >>> from watex.utils.tranformers import StratifiedWithCategoryAdder
            >>> df = mfunc.load_data('data/geo_fdata')
            >>> stratifiedNumObj= StratifiedWithCategoryAdder('flow')
            >>> strat_train_set , strat_test_set = \
            ...    stratifiedNumObj.fit_transform(X=df)
            >>> bag_train_set = strat_train_set.copy()   
            >>> pd_kws ={'alpha': 0.4, 
            ...         'label': 'flow m3/h', 
            ...         'c':'flow', 
            ...         'cmap':plt.get_cmap('jet'), 
            ...         'colorbar':True}
            >>> qkObj=QuickPlot(fs=25.)
            >>> qkObj.discover_and_visualize_data(
            ...    df = bag_train_set, x= 'east', y='north', **pd_kws)
        """
        if data_fn is not None : 
            self.data_fn = data_fn
        if df is not None: 
            self.df = df 
        df_= self.df.copy(deep=True)
        
         # visualize the data and get insights
        if 's' not in pd_kws.keys(): 
            pd_kws['s'] = df_[s_col]* self.fs 
             
        df_.plot(kind=kind, x=x, y=y, **pd_kws)
        
        plt.legend(**leg_kws)
        
        if self.savefig is not None :
            plt.savefig(self.savefig,
                        dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
    
if __name__=='__main__': 
    
    qkObj = QuickPlot(  fig_legend_kws={'loc':'upper right'},
                      fig_title = '`sfi` vs`ohmS|`geol`',
                        )  
    sns_pkws={'aspect':2 , 
              "height": 2, 
                      }

    map_kws={'edgecolor':"w"}   
    qkObj.discussingFeatures(data_fn ='data/geo_fdata/BagoueDataset2.xlsx' , 
                              features =['ohmS', 'sfi','geol', 'flow'],
                                map_kws=map_kws,  **sns_pkws
                              )   

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        