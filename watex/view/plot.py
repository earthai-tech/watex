# -*- coding: utf-8 -*-
# Copyright (c) 2021 LKouadio, Wed Jul  7 22:23:02 2021 hz
# MIT- licence.

from __future__ import annotations 
import os
import warnings
import numpy as np 
import  matplotlib.pyplot  as plt
# import matplotlib.cm as cm 
# import matplotlib.colorbar as mplcb
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.ticker import MultipleLocator, NullLocator
# import matplotlib.gridspec as gspec
import pandas as pd 
import seaborn as sns 

from ..bases import FeatureInspection
from .._watexlog import watexlog

from ..tools.mlutils import (
    cfexist , 
    findIntersectionGenObject,
    featureExistError,
    formatGenericObj
    )
from ..typing import (
    List,
    Dict,
    Optional,
    NDArray, 
    ArrayLike, 
    Iterable,
    DataFrame, 
    Series,
    F, 
    
)
from ..property import ( 
    BasePlot, 
    )
from ..tools.coreutils import ( 
    _is_readable 
    )
from ..tools.funcutils import ( 
    _assert_all_types , 
    repr_callable_obj, 
    smart_strobj_recognition
    )
from ..exceptions import ( 
    ParameterNumberError , 
    FileHandlingError, 
    PlotError, 
    TipError,
    FeatureError, 
    FitError
    )


_logger=watexlog.get_watex_logger(__name__)


class QuickPlot (BasePlot)  : 
    """
    Special class deals with analysis modules. To quick plot diagrams, 
    histograms and bar plots.
    
    Arguments 
    ----------
    **data**: str or pd.core.DataFrame
        Path -like object or Dataframe. If data is given as path-like object,
        QuickPlot`  calls  the module from :mod:`watex.bases.features`
        for data reading and sanitizing before plotting. Be aware in this
        case to provide the target name and possible the `classes` of for 
        data analysis. Both str or dataframe need to provide the name of target. 
        
    **y**: array-like, optional 
        array of the target. Must be the same length as the data. If `y` is 
        provided and `data` is given as ``str`` or ``DataFrame``, all the data 
        should be considered as the X data for analysis. 
        
    **nameoftarget**: str, 
        the name of the target from data analysis. In the both cases where the 
        data is given as string of dataframe, `nameoftarget` must be provided. 
        Otherwise an error will occurs. 
 
    **classes**: list of float, 
        list of the categorial values encoded to numerical. For instance, for
        `flow` data analysis in the Bagoue dataset, the `classes` could be 
        ``[0., 1., 3.]`` which means:: 
            
            - 0 m3/h  --> FR0
            - > 0 to 1 m3/h --> FR1
            - > 1 to 3 m3/h --> FR2
            - > 3 m3/h  --> FR3
            
    **mapflow**: bool, 
        Is refer to the flow rate prediction using DC-resistivity features and 
        work when the `nameoftarget` is set to ``flow``. If set to True, value 
        in the target columns should map to categorical values. Commonly the 
        flow rate values are given as a trend of numerical values. For a 
        classification purpose, flow rate must be converted to categorical 
        values which are mainly refered to the type of types of hydraulic. 
        Mostly the type of hydraulic system is in turn tided to the number of 
        the living population in a specific area. For instance, flow classes 
        can be ranged as follow: 
    
            - FR = 0 is for dry boreholes
            - 0 < FR ≤ 3m3/h for village hydraulic (≤2000 inhabitants)
            - 3 < FR ≤ 6m3/h  for improved village hydraulic(>2000-20 000inhbts) 
            - 6 <FR ≤ 10m3/h for urban hydraulic (>200 000 inhabitants). 
        
        Note that this flow range is not exhaustive and can be modified according 
        to the type of hydraulic required on the project. 
        
    Hold others optionnal attributes infos: 
        
    =================   ========================================================
    Key Words               Description        
    =================   ========================================================
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
    marker              marker of stations *default* is :math:`\blacktriangledown`.
    ms                  size of marker in points. *default* is 5
    marker_style        style  of marker in points. *default* is ``o``.
    marker_facecolor    facecolor of the marker. *default* is ``yellow``
    marker_edgecolor    edgecolor of the marker. *default* is ``cyan``.
    marker_edgewidth    width of the marker. *default* is ``3``.
    xminorticks         minortick according to x-axis size and *default* is 1.
    yminorticks         minortick according to y-axis size and *default* is 1.
    font_size           size of font in inches (width, height)
                        *default* is 3.
    font_style          style of font. *default* is ``italic``
    bins                histograms element separation between two bar. 
                         *default* is ``10``. 
    xlim                limit of x-axis in plot. *default* is None 
    ylim                limit of y-axis in plot. *default* is None 
    xlabel              label name of x-axis in plot. *default* is None 
    ylabel              label name  of y-axis in plot. *default* is None 
    rotate_xlabel       angle to rotate `xlabel` in plot. *default* is None 
    rotate_ylabel       angle to rotate `ylabel` in plot. *default* is None 
    leg_kws             keyword arguments of legend. *default* is empty dict.
    plt_kws             keyword arguments of plot. *default* is empty dict
    rs                  [ '-' | '.' | ':' ] line style of `Recall` metric
                        *default* is '--'
    ps                  [ '-' | '.' | ':' ] line style of `Precision `metric
                        *default* is '-'
    rc                  line color of `Recall` metric *default* is ``(.6,.6,.6)``
    pc                  line color of `Precision` metric *default* is ``k``
    s                   size of items in scattering plots. default is ``fs*40.``
    gls                 [ '-' | '.' | ':' ] line style of grid  
                        *default* is '--'.
    glc                 line color of the grid plot, *default* is ``k``
    glw                 line weight of the grid plot, *default* is ``2``
    galpha              transparency number of grid, *default* is ``0.5``  
    gaxis               axis to plot grid.*default* is ``'both'``
    gwhich              type of grid to plot. *default* is ``major``
    tp_axis             axis  to apply ticks params. default is ``both``
    tp_labelsize        labelsize of ticks params. *default* is ``italic``
    tp_bottom           position at bottom of ticks params. *default*
                        is ``True``.
    tp_top              position at the top  of ticks params. *default*
                        is ``True``.
    tp_labelbottom      see label on the bottom of the ticks. *default* 
                        is ``False``
    tp_labeltop         see the label on the top of ticks. *default* is ``True``
    cb_orientation      orientation of the colorbar. *default* is ``vertical``
    cb_aspect           aspect of the colorbar. *default* is 20.
    cb_shrink           shrink size of the colorbar. *default* is ``1.0``
    cb_pad              pad of the colorbar of plot. *default* is ``.05``
    cb_anchor           anchor of the colorbar. *default* is ``(0.0, 0.5)``
    cb_panchor          proportionality anchor of the colorbar. *default* is 
                        `` (1.0, 0.5)``.
    cb_label            label of the colorbar. *default* is ``None``.      
    cb_spacing          spacing of the colorbar. *default* is ``uniform``
    cb_drawedges        draw edges inside of the colorbar. *default* is ``False``
    cb_format           format of the colorbar values. *default* is ``None``.
    sns_orient          seaborn fig orientation. *default* is ``v`` which refer
                        to vertical 
    sns_style           seaborn style 
    sns_palette         seaborn palette 
    sns_height          seaborn height of figure. *default* is ``4.``. 
    sns_aspect          seaborn aspect of the figure. *default* is ``.7``
    sns_theme_kws       seaborn keywords theme arguments. default is ``{
                        'style':4., 'palette':.7}``
    ================    ========================================================
    
    """

    def __init__(self,  classes = None, nameoftarget= None,  **kws): 
        super().__init__(**kws)
        
        self._logging =watexlog().get_watex_logger(self.__class__.__name__)
        
        self.classes = kws.pop('classes',[0., 1., 3.])
        self.nameoftarget= kws.pop('nameoftarget', None)
        self.mapflow= kws.pop('mapflow', False)

        self.sns_orient =kws.pop('sns_orient', 'v')
        self.sns_style =kws.pop('sns_style', None)
        self.sns_palette = kws.pop('sns_palette', None)
        self.sns_height =kws.pop ('sns_height', 4.)
        self.sns_aspect =kws.pop ('sns_aspect', .7)
        self.sns_theme_kws = kws.pop('sns_theme_kws', 
                                        {'style':self.sns_style, 
                                         'palette':self.sns_palette }
                                        )
        self.data_ =None 
        self.y = None 
        
        for key in kws.keys(): 
            setattr(self, key, kws[key])


    @property 
    def data(self): 
        return self.data_ 
    
    @data.setter 
    def data (self, data):
        """ Read the data file
        
        Can read the data file provided  and set the data into pd.DataFrame by
        calling :class:`watex.bases.features.FeatureInspection`  to populate 
        convenient attributes especially when the target name is specified as 
        `flow`. Be sure to set other name if you dont want to consider flow 
        features inspection."""
          
        if str(self.nameoftarget).lower() =='flow':
           fobj= FeatureInspection( set_index=True, 
                flow_classes = self.classes , 
                target = self.nameoftarget, 
                mapflow= self.mapflow 
                           ).fit(data=data)
           self.data_= fobj.data  
        elif isinstance(data, str) :
            self.data_ = _is_readable(data )
        elif isinstance(data, pd.DataFrame): 
            self.data_ = data
            
        if str(self.nameoftarget).lower() in self.data_.columns.str.lower(): 
            ix = list(self.data.columns.str.lower()).index (
                self.nameoftarget.lower() )
            self.y = self.data_.iloc [:, ix ]

            self.X_ = self.data_.drop(columns =self.data_.columns[ix] , 
                                         )
            
    def fit(self,
            data: str | DataFrame, 
            y: Optional[Series| ArrayLike]=None
            )-> object : 
        """ Fit data and populate the arguments for plotting purposes. 
        
        Parameters 
        ----------
        data: str or pd.core.DataFrame, 
            Path -like object or Dataframe. If data is given as path-like object,
            `QuickPlot` calls  the module from :mod:`watex.bases.features`
            for data reading and sanitizing data before plotting. Be aware in this
            case to provide the target name and possible the `classes` of for 
            data analysis. Both str or dataframe need to provide the name of target. 
        
        y: array-like, optional 
            array of the target. Must be the same length as the data. If `y` is 
            provided and `data` is given as ``str`` or ``DataFrame``, all the data 
            should be considered as the X data for analysis. 
            
        Examples 
        --------

        >>> from watex.view.plot import QuickPlot
        >>> qplotObj= QuickPlot(xlabel = 'Flow classes in m3/h',
                                ylabel='Number of  occurence (%)')
        >>> qplotObj.nameoftarget= None # eith nameof target set to None 
        >>> qplotObj.fit(data)
        >>> qplotObj.data.iloc[1:2, :]
        ...  num name    east      north  ...         ohmS        lwi      geol flow
            1    2   b2  791227  1159566.0  ...  1135.551531  21.406531  GRANITES  0.8
        >>> qplotObj.nameoftarget= 'flow'
        >>> qplotObj.mapflow= True # map the flow from num. values to categ. values
        >>> qplotObj.fit(data)
        >>> qplotObj.data.iloc[1:2, :]
        ... num  power  magnitude shape  ...         ohmS        lwi      geol  flow
        id                               ...                                        
        b2    2   70.0      142.0     V  ...  1135.551531  21.406531  GRANITES   FR1
         
        """
        self.data = data 
        if y is not None: 
            y = _assert_all_types(y, np.ndarray, list, tuple, pd.Series)
            if len(y)!= len(self.data) :
                raise ValueError(
                    f"y and data must have the same length but {len(y)} and"
                    f" {len(self.data)} were given respectively.")
            
            self.y = pd.Series (y , name = self.nameoftarget or 'none')
            # for consistency get the name of target 
            self.nameoftarget = self.y.name 
            
            
        return self 
    
    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        return repr_callable_obj  (self, skip ='y')
       
    def __getattr__(self, name):
        if name.endswith ('_'): 
            if name not in self.__dict__.keys(): 
                if name in ('data_', 'X_'): 
                    raise FitError (
                        f'Fit the {self.__class__.__name__!r} object first'
                        )
                
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        appender  = "" if rv is None else f'. Do you mean {rv!r}'
        
        raise AttributeError (
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{appender}{"" if rv is None else "?"}'
            )        
        
#XXXOPTIMIZE         
    def histCatDistribution(self, data:  str | DataFrame = None, 
                               stacked: bool = False,  **kws): 
        """
        Quick plot a distributions of categorized classes according to the 
        percentage of occurence. 
        
        Parameters 
        -----------
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Both are the sequence of data. If 
            data is given as path-like object,`QuickPlot` reads and sanitizes 
            data before plotting. Be aware in this case to provide the target 
            name and possible the `classes` of for data inspection. Both str or
            dataframe need to provide the name of target. 
            
        stacked: bool 
            Pill bins one to another as a cummulative values. *default* is 
            ``False``. 
            
        bins : int, optional 
             contains the integer or sequence or string
             
        range : list, optional 
            is the lower and upper range of the bins
        
        density : bool, optional
             contains the boolean values 
            
        weights : array-like, optional
            is an array of weights, of the same shape as `data`
            
        bottom : float, optional 
            is the location of the bottom baseline of each bin
            
        histtype : str, optional 
            is used to draw type of histogram. {'bar', 'barstacked', step, 'stepfilled'}
            
        align : str, optional
             controls how the histogram is plotted. {'left', 'mid', 'right'}
             
        rwidth : float, optional,
            is a relative width of the bars as a fraction of the bin width
            
        log : bool, optional
            is used to set histogram axis to a log scale
            
        color : str, optional 
            is a color spec or sequence of color specs, one per dataset
            
        label : str , optional
            is a string, or sequence of strings to match multiple datasets
            
        normed : bool, optional
            an optional parameter and it contains the boolean values. It uses 
            the density keyword argument instead.
            
        Examples 
        ---------
        >>> from watex.view.plot import QuickPlot 
        >>> qplotObj= QuickPlot(xlabel = 'Flow classes in m3/h',
                                ylabel='Number of  occurence (%)'
                                lc='b')
        >>> qplotObj.histCatDistribution()
        
        """
        self._logging.info('Quick plot of categorized classes distributions.'
                           f' the target name: {self.nameoftarget!r}')
        
        if self.data_ is None: 
            self.fit(data)
            
        if self.data is None: 
            raise PlotError( "Can plot histogram with NoneType value!")

        if self.nameoftarget is None and self.y is None: 
            raise FeatureError("Please specify the name of the target. ")

        # reset index 
        df_= self.data_.copy()  #make a copy for safety 
        df_.reset_index(inplace =True)
        
        plt.figure(figsize =self.fig_size)
        plt.hist(df_[self.nameoftarget], bins=self.bins ,
                  stacked = stacked , color= self.lc , **kws)

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.fig_title)

        if self.savefig is not None :
            plt.savefig(self.savefig,dpi=self.fig_dpi,
                        orientation =self.fig_orientation
                        )
        
    def barCatDistribution(self,
                           data: str | DataFrame =None, 
                           basic_plot: bool = True,
                           groupby: List[str] | Dict [str, float] =None,
                           **kws):
        """
        Bar plot distribution. Can plot a distribution according to 
        the occurence of the `target` in the data and other parameters 
        
        Parameters 
        -----------
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Both are the sequence of data. If 
            data is given as path-like object,`QuickPlot` reads and sanitizes 
            data before plotting. Be aware in this case to provide the target 
            name and possible the `classes` of for data inspection. Both str or
            dataframe need to provide the name of target. 
            
        basic_pot: bool, 
            Plot only the occurence of targetted columns from 
            `matplotlib.pyplot.bar` function. 
            
        groupby: list or dict, optional 
            Group features for plotting. For instance it plot others features 
            located in the df columns. The plot features can be on ``list``
            and use default plot properties. To customize plot provide, one may 
            provide, the features on ``dict`` with convenients properties 
            like::

                * `groupby`= ['shape', 'type'] #{'type':{'color':'b',
                                             'width':0.25 , 'sep': 0.}
                                     'shape':{'color':'g', 'width':0.25, 
                                             'sep':0.25}}
        kws: dict, 
            Additional keywords arguments from `seaborn.countplot`
            
        Examples
        ----------
            >>> from watex.view.plot import QuickPlot
            >>> data = 'data/geodata/main.bagciv.data.csv'
            >>> qplotObj= QuickPlot(xlabel = 'Anomaly type',
                                    ylabel='Number of  occurence (%)',
                                    lc='b', nameoftarget='flow')
            >>> qplotObj.sns_style = 'darkgrid'
            >>> qplotObj.fit(data)
            >>> qplotObj. barCatDistribution(basic_plot =False, 
            ...                                groupby=['shape' ])
   
        """
        
        if data is not None: 
            self.data= data 
            
        if self.data_ is None: 
            raise PlotError ("NoneType can not be plotted!")

        fig, ax = plt.subplots(figsize = self.fig_size)
        
        df_= self.data.copy(deep=True)  #make a copy for safety 
        df_.reset_index(inplace =True)
        
        if groupby is None:
            mess= ''.join([
                'Basic plot is turn to``False`` but no specific plot is', 
                "  detected. Please provide a specific column's into "
                " a `specific_plot` argument."])
            self._logging.debug(mess)
            warnings.warn(mess)
            basic_plot =True
            
        if basic_plot : 
            ax.bar(list(set(df_[self.nameoftarget])), 
                        df_[self.nameoftarget].value_counts(normalize =True),
                        label= self.fig_title, color = self.lc, )  
    
        if groupby is not None : 
            if hasattr(self, 'sns_style'): 
                sns.set_style(self.sns_style)
            if isinstance(groupby, str): 
                self.groupby =[groupby]
            if isinstance(groupby , dict):
                groupby =list(groupby.keys())
            for sll in groupby :
                ax= sns.countplot(x= sll,  hue=self.nameoftarget, 
                                  data = df_, orient = self.sns_orient,
                                  ax=ax ,**kws)

        ax.set_xlabel(self. xlabel)
        ax.set_ylabel (self.ylabel)
        ax.set_title(self.fig_title)
        ax.legend() 
        
        if groupby is not None: 
            self._logging.info(
                'Multiple bar plot distribution grouped by  {0}.'.format(
                    formatGenericObj(groupby)).format(*groupby))
        
        if self.savefig is not None :
            plt.savefig(self.savefig,dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
        plt.show()
        

    
    def multiCatDistribution(self, 
                             data : str | DataFrame = None, 
                             *, 
                             x =None, 
                             col=None, 
                             hue =None, 
                             targets: List[str]=None,
                             x_features:List[str]=None ,
                             y_features: List[str]=None, 
                             kind:str='count',
                             **kws): 
        """
        Figure-level interface for drawing categorical plots onto a FacetGrid.
        
        Multiple categorials plots  from targetted pd.series. 
        
        Parameters 
        -----------
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`QuickPlot` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` of for data inspection. Both str or dataframe
            need to provide the name of target. 
            
        x, y, hue: list , Optional, 
            names of variables in data. Inputs for plotting long-form data. 
            See examples for interpretation. Here it can correspond to  
            `x_features` , `y_features` and `targets` from dataframe. Note that
            each columns item could be correspond as element of `x`, `y` or `hue`. 
            For instance x_features could refer to x-axis features and must be 
            more than 0 and set into a list. the `y_features` might match the 
            columns name for `sns.catplot`. If number of feature is more than 
            one, create a list to hold all features is recommended. 
            the `y` should fit the  `sns.catplot` argument ``hue``. Like other 
            it should be on list of features are greater than one. 
        
        row, colnames of variables in data, optional
            Categorical variables that will determine the faceting of the grid.
        
        col_wrapint
            "Wrap" the column variable at this width, so that the column facets 
            span multiple rows. Incompatible with a row facet.
        
        estimator: string or callable that maps vector -> scalar, optional
            Statistical function to estimate within each categorical bin.
        
        errorbar: string, (string, number) tuple, or callable
            Name of errorbar method (either "ci", "pi", "se", or "sd"), or a 
            tuple with a method name and a level parameter, or a function that
            maps from a vector to a (min, max) interval.
        
        n_bootint, optional
            Number of bootstrap samples used to compute confidence intervals.
        
        units: name of variable in data or vector data, optional
            Identifier of sampling units, which will be used to perform a 
            multilevel bootstrap and account for repeated measures design.
        
        seed: int, numpy.random.Generator, or numpy.random.RandomState, optional
            Seed or random number generator for reproducible bootstrapping.
        
        order, hue_order: lists of strings, optional
            Order to plot the categorical levels in; otherwise the levels are 
            inferred from the data objects.
        
        row_order, col_order: lists of strings, optional
            Order to organize the rows and/or columns of the grid in, otherwise
            the orders are inferred from the data objects.
        
        height: scalar
            Height (in inches) of each facet. See also: aspect.
        
        aspect:scalar
            Aspect ratio of each facet, so that aspect * height gives the width
            of each facet in inches.
        
        kind: str, optional
            `The kind of plot to draw, corresponds to the name of a categorical 
            axes-level plotting function. Options are: "strip", "swarm", "box", 
            "violin", "boxen", "point", "bar", or "count".
        
        native_scale: bool, optional
            When True, numeric or datetime values on the categorical axis 
            will maintain their original scaling rather than being converted 
            to fixed indices.
        
        formatter: callable, optional
            Function for converting categorical data into strings. Affects both
            grouping and tick labels.
        
        orient: "v" | "h", optional
            Orientation of the plot (vertical or horizontal). This is usually 
            inferred based on the type of the input variables, but it can be 
            used to resolve ambiguity when both x and y are numeric or when 
            plotting wide-form data.
        
        color: matplotlib color, optional
            Single color for the elements in the plot.
        
        palette: palette name, list, or dict
            Colors to use for the different levels of the hue variable. 
            Should be something that can be interpreted by color_palette(), 
            or a dictionary mapping hue levels to matplotlib colors.
        
        hue_norm: tuple or matplotlib.colors.Normalize object
            Normalization in data units for colormap applied to the hue 
            variable when it is numeric. Not relevant if hue is categorical.
        
        legend: str or bool, optional
            Set to False to disable the legend. With strip or swarm plots, 
            this also accepts a string, as described in the axes-level 
            docstrings.
        
        legend_out: bool
            If True, the figure size will be extended, and the legend will be 
            drawn outside the plot on the center right.
        
        share{x,y}: bool, 'col', or 'row' optional
            If true, the facets will share y axes across columns and/or x axes 
            across rows.
        
        margin_titles:bool
            If True, the titles for the row variable are drawn to the right of 
            the last column. This option is experimental and may not work in 
            all cases.
        
        facet_kws: dict, optional
            Dictionary of other keyword arguments to pass to FacetGrid.
        
        kwargs: key, value pairings
            Other keyword arguments are passed through to the underlying 
            plotting function.

        Examples
        ---------
        >>> from watex.view.plot import QuickPlot 
        >>> data = 'data/geodata/main.bagciv.data.csv'
        >>> qplotObj= QuickPlot(lc='b', nameoftarget='flow')
        >>> qplotObj.sns_style = 'darkgrid'
        >>> fdict={
        ...            'x':['shape', 'type', 'type'], 
        ...            'col':['type', 'geol', 'shape'], 
        ...            'hue':['flow', 'flow', 'geol'],
        ...            } 
        >>> qplotObj.multiCatDistribution(**fdict)
            
        """
        if data is not None: 
            self.data= data 
            
        if self.data_ is None: 
            raise PlotError ("NoneType can not be plotted!")
            
        # set 
        if x is None : x = [None] 
        if col is None: xol =[None] 
        if hue is None: hue =[None] 
        
        maxlen = max([len(i) for i in [x, col, hue]])  
        
        x = [None if n !=None else n for ii, n  in range(maxlen)] 
        col = [None if n !=None else n for n in range(maxlen)]
        hue =  [None if n !=None else n for n in range(maxlen)]
        l=list()
        for i in range (maxlen): 
            if i!= None: 
                l.append(None)
                
        #features_dict=kws.pop('features_dict',None )
        
        # if sns_style is not None: 
        #     self.sns_style = sns_style

        # for key in kws.keys(): 
        #     setattr(self, key, kws[key])
        
        # if data is not None : 
        #     self.data = data
            
        # minlen=9999999
        # if features_dict is None : 
        #     features_dict ={ feature: featvalue  for
        #                     feature, featvalue in zip(
        #                         ['x_features', 'y_features', 'targets'],
        #                     [x_features, y_features, targets] )}
        
        # if features_dict is not None : 
        #     for ffn, ffval in features_dict.items():

        #         if ffval is None: 
        #             warnings.warn(f'Need `{ffn}` value for multiple '
        #                           'categorical plotting.')
        #             raise PlotError(
        #                 f'Need `{ffn}` value for multiple categorial plots.')
        #         if isinstance(ffval, str): 
        #             ffval=[ffval]
                    
        #         if minlen > len(ffval): 
        #             minlen= len(ffval)
        #     features_dict ={ feature: featvalue[:minlen] for
        #                     feature, featvalue in zip(
        #                         ['x_features', 'y_features', 'targets'],
        #                     [x_features, y_features, targets] )}
        print(len(x))
        df_= self.data.copy(deep=True)
        df_.reset_index(inplace=True )
         
        if not hasattr(self, 'ylabel'): 
            self.ylabel= 'Number of  occurence (%)'
            
        if hue is not None: 
            self._logging.info(
                'Multiple categorical plots  from targetted {0}.'.format(
                    formatGenericObj(hue)).format(*hue))
        
        for ii in range(len(x)): 
            sns.catplot( data = df_,
                        kind= kind, 
                        x=  x[ii], #features_dict ['x_features'][ii], 
                        col=col[ii], # features_dict['y_features'][ii], 
                        hue= hue[ii], #features_dict['targets'][ii],
                        linewidth = self.lw, 
                        height = self.sns_height,
                        aspect = self.sns_aspect,
                        **kws
                    ).set_ylabels(self.ylabel)
        
    
        plt.show()
       
        if self.sns_style is not None: 
            sns.set_style(self.sns_style)
            
        print('--> Multiple plots sucessfully done!')    
        
        
    def plot_correlation_matrix(self, df=None, data =None, 
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

        :param df: refer to :doc:`watex.view.plot.QuickPlot`
        :param data: see :doc:`watex.view.plot.QuickPlot`
        
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
            
            >>> from watex.view.plot import QuickPlot 
            >>> qplotObj = QuickPlot(
            ...    data ='data/geo_fdata/BagoueDataset2.xlsx' , lc='b', 
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

        
        if data is not None : 
            self.data = data
        
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
                raise PlotError(
                    f"Feature's name is set to ``{feature_names}``."
                    "Please provided the right `plot_params` argument "
                    "not {plot_params}."
                        )
                
        # Control the existence of providing features into the pd.dataFramename:
        try : 
            reH=  cfexist(features_to= feature_names,
                               features = df_.columns) 
                
        except: 
            raise ParameterNumberError(
                f'Parameters number of {feature_names} is  not found in the '
                ' dataframe columns ={0}'.format(list(df_.columns)))
        else : 
            if reH is False: 
                raise ParameterNumberError(
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
            
        fmObj = formatGenericObj(feature_names)
        
        print(" --> Successfully plot of matrix correlation between "
              f"{'categorial' if plot_params =='qual' else 'numerical'}"
              " features {0}.".format(fmObj).format(*feature_names))
              
        plt.show()
                
    def plot_numerical_features(self, df=None, data =None , target= None,
                                  numerical_features=None, 
                                  trigger_map_lower_kws: bool =False, 
                                  map_lower_kws=None, **sns_kws): 
        """
        Plot qualitative features distribution using correlative aspect. Be 
        sure to provided numerical features arguments. 
        
        :param df: refer to :doc:`watex.view.plot.QuickPlot`
        :param data: see :doc:`watex.view.plot.QuickPlot`
        
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
            
            >>> from watex.view.plot import QuickPlot 
            >>> qkObj = QuickPlot(
            ...         data ='data/geo_fdata/BagoueDataset2.xlsx', lc='b', 
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
        if data is not None : 
            self.data = data
            
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
                raise FeatureError(
                    'Target feature is missing. Could not plot numerical'
                    '  features. Please provide the right target``hue`` name.'
                    )
        elif target is not None : 
            if not target in df_.columns: 
                raise FeatureError(
                    f"The given target {target} is wrong. Please provide the "
                    " the right target (hue)instead.")
        
        
        if target =='flow': 
            if sorted(findIntersectionGenObject(
                    {'ohmS', 'power', 'sfi', 'magnitude'}, df_.columns
                    ))== sorted({'ohmS', 'power', 'sfi', 'magnitude'}):
                numerical_features= sorted({'ohmS', 'power', 'sfi', 'magnitude'})
                
            if target =='flow': 
                numerical_features.append('flow')
                # df_['flow']=df_['flow'].astype('category').cat.codes
        try : 
            resH= cfexist(features_to= numerical_features,
                               features = df_.columns)
        except:
             raise ParameterNumberError(
                f'Parameters number of {numerical_features} is  not found in the '
                ' dataframe columns ={0}'.format(list(df_.columns)))
        
        else: 
            if not resH:  raise ParameterNumberError(
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
                raise ParameterNumberError(
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
   
    def joint2features(self,*, data =None, df=None, 
                      features=['ohmS', 'lwi'], 
                      join_kws=None, marginals_kws=None, 
                      **sns_kwargs)-> None:
        """
        Joint methods allow to visualize correlation of two features. 
        
        Draw a plot of two features with bivariate and univariate graphs. 
        
        :param df: refer to :doc:`watex.view.plot.QuickPlot`
        :param data: see :doc:`watex.view.plot.QuickPlot`
        
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
            
            >>> from watex.view.plot.QuickPlot import joint2features
            >>> qkObj = QuickPlot(
            ...        data ='data/geo_fdata/BagoueDataset2.xlsx', lc='b', 
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
        if data is not None : 
            self.data = data
        if df is not None: self.df = df 
        
        df_= self.df.copy(deep=True)
        
        try : 
            resH= cfexist(features_to= features,features = df_.columns)
        except TypeError: 
            
            print(' Features can not be a NoneType value.'
                  'Please set a right features.')
            self._logging.error('NoneType can not be a features!')
        except :
            raise ParameterNumberError(
               f'Parameters number of {features} is  not found in the '
               ' dataframe columns ={0}'.format(list(df_.columns)))
        
        else: 
            if not resH:  raise ParameterNumberError(
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
                raise  FeatureError(
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
            raise ParameterNumberError(
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
            
    def scatteringFeatures(self,data=None, df=None, 
                           features=['lwi', 'flow'],
                           relplot_kws= None, 
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
        
        :param df: refer to :doc:`watex.view.plot.QuickPlot`
        :param data: see :doc:`watex.view.plot.QuickPlot`
        
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
            
            >>> from watex.view.plot.QuickPlot import  scatteringFeatures
            >>> qkObj = QuickPlot(
            ...    data ='data/geo_fdata/BagoueDataset2.xlsx' , lc='b', 
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
        if data is not None : 
            self.data = data
            
        if df is not None: self.df = df 
        
        df_= self.df.copy(deep=True)
        
        # controller function
        try:
            featureExistError(superv_features=features, 
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
            raise ParameterNumberError(
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
    
    def discussingFeatures(self,df=None, data=None,
                           features=['ohmS','sfi', 'geol', 'flow'],
                           map_kws=None, 
                           map_func= None, 
                           **sns_kws)-> None: 
        """
        Porvides the features names at least 04 and discuss with 
        their distribution. 
        
        This method maps a dataset onto multiple axes arrayed in a grid of
        rows and columns that correspond to levels of features in the dataset. 
        The plots it produces are often called “lattice”, “trellis”, or
        “small-multiple” graphics. 
        
        :param df: refer to :doc:`watex.view.plot.QuickPlot`
        :param data: see :doc:`watex.view.plot.QuickPlot`
        
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
            
            >>> from watex.view.plot.QuickPlot import discussingFeatures 
            >>> qkObj = QuickPlot(  fig_legend_kws={'loc':'upper right'},
            ...          fig_title = '`sfi` vs`ohmS|`geol`',
            ...            )  
            >>> sns_pkws={'aspect':2 , 
            ...          "height": 2, 
            ...                  }
            >>> map_kws={'edgecolor':"w"}   
            >>> qkObj.discussingFeatures(
            ...    data ='data/geo_fdata/BagoueDataset2.xlsx' , 
            ...                         features =['ohmS', 'sfi','geol', 'flow'],
            ...                           map_kws=map_kws,  **sns_pkws
            ...                         )   
        """
        if data is not None : 
            self.data = data
        if df is not None: 
            self.df = df 
        df_= self.df.copy(deep=True)
        
        try:
            featureExistError(superv_features=features, 
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
            
            raise ParameterNumberError(
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
                raise FeatureError(
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
        
    def discover_and_visualize_data(self, df = None, data:str =None, 
                                    x:str =None, y:str =None, kind:str ='scatter',
                                    s_col ='lwi', leg_kws:dict ={}, **pd_kws):
        """ Create a scatter plot to visualize the data using `x` and `y` 
        considered as dataframe features. 
        
        :param df: refer to :class:`watex.view.plot.QuickPlot`
        :param data: see :class:`watex.view.plot.QuickPlot`
        
        :param x: Column name to hold the x-axis values 
        :param y: column na me to hold the y-axis values 
        :param s_col: Size for scatter points. 'Default is 
            ``fs`` time the features colum `lwi`.
            
        :param pd_kws: 
            Pandas plot keywords arguments 
        :param leg_kws: Matplotlib legend keywords arguments 
        
        :Example: 
            
            >>> import watex.tools.mlutils as mfunc
            >>> from watex.bases.tranformers import StratifiedWithCategoryAdder
            >>> from watex.view.plot import QuickPlot
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
        if data is not None : 
            self.data = data
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
    qkObj.discussingFeatures(data ='data/geo_fdata/BagoueDataset2.xlsx' , 
                              features =['ohmS', 'sfi','geol', 'flow'],
                                map_kws=map_kws,  **sns_pkws
                              )   

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        