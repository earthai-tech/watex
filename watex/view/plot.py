# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Create data: Wed Jul  7 22:23:02 2021 hz

"""
Plot templates 
================
Base plot for data exploratory and analysis 

"""
from __future__ import annotations 

import re 
import copy
import warnings

import numpy as np 
import  matplotlib.pyplot  as plt
import pandas as pd 
import seaborn as sns 

from ..bases import FeatureInspection

from .._docstring import ( 
    DocstringComponents,
    _core_docs,
    _baseplot_params
    )
from ..tools.mlutils import (
    existfeatures,
    formatGenericObj, 
    selectfeatures 
    )
from ..typing import (
    Any , 
    Tuple, 
    List,
    Dict,
    Optional,
    ArrayLike, 
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
    _isin, 
    repr_callable_obj, 
    smart_strobj_recognition, 
    smart_format,
    reshape, 
    
    )
from ..exceptions import ( 
    PlotError, 
    FeatureError, 
    NotFittedError
    )

try: 
    import missingno as msno 
except : pass 

from .._watexlog import watexlog    
_logger=watexlog.get_watex_logger(__name__)

#+++++++++++++++++++++++ add seaborn docs +++++++++++++++++++++++++++++++++++++ 
_sns_params = dict( 
    sns_orient="""
sns_orient: 'v' | 'h', optional
    Orientation of the plot (vertical or horizontal). This is usually inferred 
    based on the type of the input variables, but it can be used to resolve 
    ambiguity when both x and y are numeric or when plotting wide-form data. 
    *default* is ``v`` which refer to 'vertical'  
    
    """, 
    sns_style="""
sns_style: dict, or one of {darkgrid, whitegrid, dark, white, ticks}
    A dictionary of parameters or the name of a preconfigured style.
    """, 
    sns_palette="""
sns_palette: seaborn color paltte | matplotlib colormap | hls | husl
    Palette definition. Should be something color_palette() can process. the 
    palette  generates the point with different colors
    """, 
    sns_height="""
sns_height:float, 
    Proportion of axes extent covered by each rug element. Can be negative.
    *default* is ``4.``
    """, 
    sns_aspect="""
sns_aspect: scalar (float, int)
    Aspect ratio of each facet, so that aspect * height gives the width of 
    each facet in inches. *default* is ``.7``
    """, 
    )

_param_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_baseplot_params), 
    sns = DocstringComponents(_sns_params)
    )
#++++++++++++++++++++++++++++++++++ end +++++++++++++++++++++++++++++++++++++++

class ExPlot (BasePlot): 
    
    msg = ( "{expobj.__class__.__name__} instance is not fitted yet."
           " Call 'fit' with appropriate arguments before using"
           " this method"
           )
    
    def __init__(self, **kws):
        super().__init__(**kws)
        self.y= None 
        self.data = None 

        
    def fit(self, data: str |DataFrame =None, y:ArrayLike = None,  **kws ): 
        """ Fit data and populate the arguments for plotting purposes. 
        
        There is no conventional procedure for checking if a method is fitted. 
        However, an class that is not fitted should raise 
        :class:`exceptions.NotFittedError` when a method is called.
        
        Parameters
        ------------
        data: Filepath or Dataframe or shape (M, N) from 
            :class:`pandas.DataFrame`. Dataframe containing samples M  
            and features N

        kws: dict 
            Additional keywords arguments from 
            :func:watex.tools.coreutils._is_readable`
           
        Return
        -------
        ``self``: `Plot` instance 
            returns ``self`` for easy method chaining.
             
        """
        if data is not None: 
            self.data = _is_readable(data, **kws)
        if y is not None: 
            self.y = y 
            
        return self 
    
    def scatter (self, data : str |DataFrame =None,
                 name =None, 
                 y =None,
                 **kwd): 
        """ Shown the relationship between two numeric columns. This is very
        way with pandas. 
        
        Parameters 
        ------------
        data: Dataframe or shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N
            
        
        """
        return self 
    
    def histvstarget (self, 
                      c: Any, *, 
                      y: ArrayLike | Series =None, 
                      name: str ,
                      c_label: str = None, 
                      other_label: str= None,
                      **kws
                      ): 
        """ A histogram of continuous against the target of binary plot. 
        
        Parameters 
        ----------
        data: Dataframe or shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N
        
        y: Array like, shape (M, )
            array of the features M and 
        c: str or  int  
            the class value in `y` to consider. Raise an error if not in `y`. 
            value `c` can be considered as the binary positive class 
        name: str, 
            the column name to consider. Shoud be  an item of the dataframe 
            columns. Raise an error it element does not exist. 
            
        c_label: str 
            legend label for the `c` ( binary positive class )
            
        other_label: str 
            name to give other classes in `y`. Can be considered as the binary 
            negative class. 
          
        Return
        -------
        ``self``: `Plot` instance 
            returns ``self`` for easy method chaining.
            
        Examples
        --------
        >>> import pandas as pd 
        >>> from watex.view import Plot
        >>> from watex.view.plot import histvstarget 
        >>> data = pd.read_csv ( '../../data/geodata/main.bagciv.data.csv' ) 
        >>> p = Plot()
        >>> p.fig_size = (12, 4)
        >>> p.savefig ='../../bbox.png'
        >>> histvstarget (p, data, data.flow, c = 0,  name= 'sfi', 
                          c_label='dried borehole (m3/h)',
                          other_label = 'accept. boreholes'
                          )
        """
        if self.data is None: 
            raise NotFittedError(self.msg.format(
                expobj=self)
            )
            
        self.data = _assert_all_types(self.data, pd.DataFrame)
        if y is not None: 
            self.y = y 
        
        self.y = self.y.values if isinstance(
            self.y, pd.Series) else reshape (self.y)
        
        existfeatures(self.data, name) # assert the name in the columns 
        
        if not _isin (self.y, c): 
            raise ValueError (f"Expect 'c' value in the target 'y', got:{c}")
        
        fig, _ = plt.subplots (figsize = self.fig_size )
        mask = self.y == c 
     
        ax = sns.displot (self.data[mask][name], 
                          label= c_label, 
                          linewidth = self.lw, 
                          height = self.sns_height,
                          aspect = self.sns_aspect,
                          **kws
                      ).set_ylabels(self.ylabel)
          
        ax= sns.displot (self.data[~mask][name], 
                          label= other_label, 
                          linewidth = self.lw, 
                          height = self.sns_height,
                          aspect = self.sns_aspect,
                          
                          **kws,
                          )
    
        if self.sns_style is not None: 
            sns.set_style(self.sns_style)
            
        # ax.set_xlim (self.xlim )
        ax.add_legend ()
        
        if self.savefig is not None: 
            fig.savefig (self.savefig , dpi = self.fig_dpi , 
                         bbox_inches = 'tight')
        
        plt.show() if self.savefig is None else plt.close()
        
        return self 
    
    def histogram (self, *, 
                   name: str , kind:str = 'hist', 
                   fig_size: Tuple [int] =None,
                   **kws 
                   ): 
        """ A histogram visualization of numerica data.  
        
        Parameters 
        ----------
        data: Dataframe or shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N
        name: str 
            feature name in the dataframe. raise an error , if it does not 
            exist in the dataframe 
        kind: str 
            Mode of pandas series plotting. the *default* is ``hist``. 
            
        kws: dict, 
            additional keywords arguments from : func:`pandas.DataFrame.plot` 
            
       Return
        -------
        ``self``: `Plot` instance 
            returns ``self`` for easy method chaining.
            
        """
        if self.data is None: 
            raise NotFittedError(self.msg.format(
                expobj=self)
            )
                
        name = _assert_all_types(name,str )
        # assert whether whether  feature exists 
        existfeatures(self.data, name)
    
        fig, ax = plt.subplots (figsize = fig_size or self.fig_size )
        self.data [name].plot(kind = kind , ax= ax  , **kws )
        fig.savefig ( self.savefig , dpi = self.fig_dpi  )
        
        plt.show() if self.savefig is not None else plt.close () 
        
        return self 
    
    
    def missing(self, *, 
                kind: str =None, 
                sample: float = None,  
                **kwd
                ): 
        """
        Vizualize patterns in the missing data.
        
        Parameters 
        ------------
        data: Dataframe or shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N
    
        kind: str, Optional 
            kind of visualization. Can be ``dendrogramm``, ``mbar`` or ``bar`` plot 
            for dendrogram , :mod:`msno` bar and :mod:`plt` visualization 
            respectively: 
                
            * ``bar`` plot counts the  nonmissing data  using pandas
            *  ``mbar`` use the :mod:`msno` package to count the number 
                of nonmissing data. 
            * dendrogram`` show the clusterings of where the data is missing. 
                leaves that are the same level predict one onother presence 
                (empty of filled). The vertical arms are used to indicate how  
                different cluster are. short arms mean that branch are 
                similar. 
            * ``corr` creates a heat map showing if there are correlations 
                where the data is missing. In this case, it does look like 
                the locations where missing data are corollated.
            * ``None`` is the default vizualisation. It is useful for viewing 
                contiguous area of the missing data which would indicate that 
                the missing data is  not random. The :code:`matrix` function 
                includes a sparkline along the right side. Patterns here would 
                also indicate non-random missing data. It is recommended to limit 
                the number of sample to be able to see the patterns. 
   
            Any other value will raise an error 
            
        sample: int, Optional
            Number of row to visualize. This is usefull when data is composed of 
            many rows. Skrunked the data to keep some sample for visualization is 
            recommended.  ``None`` plot all the samples ( or examples) in the data 
            
        kws: dict 
            Additional keywords arguments of :mod:`msno.matrix` plot. 
    
        Return
        -------
        ``self``: `{self.__class__.__name__}` instance 
            returns ``self`` for easy method chaining.
            
        Example
        --------
        >>> import pandas as pd 
        >>> from watex.view import ExPlot
        >>> data = pd.read_csv ('data/geodata/main.bagciv.data.csv' ) 
        >>> p = ExPlot().fit(data)
        >>> p.fig_size = (12, 4)
        >>> p.missing(kind ='corr')
        
        """
        if self.data is None: 
            raise NotFittedError(self.msg.format(
                expobj=self)
            )
            
        kstr =('dendrogram', 'bar', 'mbar')
        kind = str(kind).lower().strip() 
        
        regex = re.compile (r'none|dendro|corr|base|default|mbar|bar', 
                            flags= re.IGNORECASE)
        kind = regex.search(kind) 
        
        if kind is None: 
            raise ValueError (f"Expect {smart_format(kstr, 'or')} not: {kind!r}")
            
        kind = kind.group()
  
        if kind in ('none', 'default', 'base'): 
            kind ='mbar'
        
        if sample is not None: 
            sample = _assert_all_types(sample, int, float)
            
        if kind =='bar': 
            fig, ax = plt.subplots (figsize = self.fig_size, **kwd )
            (1- self.data.isnull().mean()).abs().plot.bar(ax=ax)
    
        elif kind  in ('mbar', 'dendro', 'corr'): 
            try : 
                msno 
            except : 
                raise ModuleNotFoundError(
                    "Missing 'missingno' package. Can not plot {kind!r}")
                
            if kind =='mbar': 
                ax = msno.bar(
                    self.data if sample is None else self.data.sample(sample),
                              figsize = self.fig_size 
                              )
    
            elif kind =='dendro': 
                ax = msno.dendrogram(self.data, **kwd) 
        
                
            elif kind =='corr': 
                ax= msno.heatmap(self.data, figsize = self.fig_size)
            else : 
                ax = msno.matrix(
                    self.data if sample is None else self.data.sample (sample),
                                 figsize= self.fig_size , **kwd)
        
        if self.savefig is not None:
            fig.savefig(self.savefig, dpi =self.fig_dpi 
                        ) if kind =='bar' else ax.get_figure (
                ).savefig (self.savefig,  dpi =self.fig_dpi)
        
        return self 



class QuickPlot (BasePlot)  : 
    r"""
    Special class deals with analysis modules. To quick plot diagrams, 
    histograms and bar plots.
    
    Arguments 
    ----------
    data: str or pd.core.DataFrame
        Path -like object or Dataframe. If data is given as path-like object,
        QuickPlot`  calls  the module from :mod:`watex.bases.features`
        for data reading and sanitizing before plotting. Be aware in this
        case to provide the target name and possible the `classes` of for 
        data analysis. Both str or dataframe need to provide the name of target. 
        
    y: array-like, optional 
        array of the target. Must be the same length as the data. If `y` is 
        provided and `data` is given as ``str`` or ``DataFrame``, all the data 
        should be considered as the X data for analysis. 
        
    nameoftarget: str, 
        the name of the target from data analysis. In the both cases where the 
        data is given as string of dataframe, `nameoftarget` must be provided. 
        Otherwise an error will occurs. 
 
    classes: list of float, 
        list of the categorial values encoded to numerical. For instance, for
        `flow` data analysis in the Bagoue dataset, the `classes` could be 
        ``[0., 1., 3.]`` which means:: 
            
            - 0 m3/h  --> FR0
            - > 0 to 1 m3/h --> FR1
            - > 1 to 3 m3/h --> FR2
            - > 3 m3/h  --> FR3
            
    mapflow: bool, 
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
        
    Note that the flow range from `mapflow` is not exhaustive and can be 
    modified according to the type of hydraulic required on the project. 
        
    Hold others optional attributes informations: 
        
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
    verbose             control the verbosity. Higher value, more messages. 
                        *default* is ``0``.
    ================    ========================================================
    
    """

    def __init__(self,  classes = None, nameoftarget= None,  **kws): 
        
        self._logging =watexlog().get_watex_logger(self.__class__.__name__)
        self.classes = kws.pop('classes', None)
        self.nameoftarget= kws.pop('nameoftarget', None)
        self.mapflow= kws.pop('mapflow', False)
        
        super().__init__(**kws)
        
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
            # default inspection for DC -flow rate prediction
           fobj= FeatureInspection( set_index=True, 
                flow_classes = self.classes or [0., 1., 3] , 
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
        """ Fit data and populate the attributes for plotting purposes. 
        
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
          
         Returns
         -------
         :class:`QuickPlot` instance
             Returns ``self`` for easy method chaining.
             
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
                    raise NotFittedError (
                        f'Fit the {self.__class__.__name__!r} object first'
                        )
                
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        appender  = "" if rv is None else f'. Do you mean {rv!r}'
        
        raise AttributeError (
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{appender}{"" if rv is None else "?"}'
            )        
       
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
          
        Returns
        -------
        :class:`QuickPlot` instance
            Returns ``self`` for easy method chaining.
             
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
        
        return self 
    
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
          
        Returns
        -------
        :class:`QuickPlot` instance
            Returns ``self`` for easy method chaining.
              
        Examples
        ----------
            >>> from watex.view.plot import QuickPlot
            >>> data = '../data/geodata/main.bagciv.data.csv'
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
            
        plt.show() if self.savefig is None else plt.close () 
        
        print('--> Bar distribution plot successfully done!'
              )if self.verbose > 0 else print()  
        
        return self 
    
    
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
            
        Returns
        -------
        :class:`QuickPlot` instance
            Returns ``self`` for easy method chaining.
             
        Examples
        ---------
        >>> from watex.view.plot import QuickPlot 
        >>> data = '../data/geodata/main.bagciv.data.csv'
        >>> qplotObj= QuickPlot(lc='b', nameoftarget='flow')
        >>> qplotObj.sns_style = 'darkgrid'
        >>> qplotObj.mapflow=True # to categorize the flow rate 
        >>> qplotObj.fit(data)
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
        if x is None :
            x = [None] 
        if col is None:
            col =[None] 
        if hue is None:
            hue =[None] 
        # for consistency put the values in list 
        x, col, hue = list(x) , list(col), list(hue)
        maxlen = max([len(i) for i in [x, col, hue]])  
        
        x.extend ( [None  for n  in range(maxlen - len(x))])
        col.extend ([None  for n  in range(maxlen - len(col))] )
        hue.extend ([None  for n  in range(maxlen - len(hue))])
       
        df_= self.data.copy(deep=True)
        df_.reset_index(inplace=True )
         
        if not hasattr(self, 'ylabel'): 
            self.ylabel= 'Number of  occurence (%)'
            
        if hue is not None: 
            self._logging.info(
                'Multiple categorical plots  from targetted {0}.'.format(
                    formatGenericObj(hue)).format(*hue))
        
        for ii in range(len(x)): 
            sns.catplot(data = df_,
                        kind= kind, 
                        x=  x[ii], 
                        col=col[ii], 
                        hue= hue[ii], 
                        linewidth = self.lw, 
                        height = self.sns_height,
                        aspect = self.sns_aspect,
                        **kws
                    ).set_ylabels(self.ylabel)
        
    
        plt.show()
       
        if self.sns_style is not None: 
            sns.set_style(self.sns_style)
            
        print('--> Multiple distribution plots sucessfully done!'
              ) if self.verbose > 0 else print()     
        
        return self 
    
    def correlationMatrix(self,
                        data: str | DataFrame =None, 
                        cortype:str ='num',
                        features: Optional[List[str]] = None, 
                        method: str ='pearson',
                        min_periods: int=1, 
                        **sns_kws): 
        """
        Method to quick plot the numerical and categorical features. 
        
        Set `features` by providing the quantitative features as well
         as the qualitative feature names. If ``None`` value is provided, It 
        assumes to consider on groundwater exploration therefore the 
        `target` is set to ``flow``. If not the case and ``feature_names`` are 
        still ``None``, an errors raises. 

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

        cortype: str, 
            The typle of parameters to cisualize their coreletions. Can be 
            ``num`` for numerical features and ``cat`` for categorical features. 
            *Default* is ``num`` for quantitative values. 
            
        method: str,  
            the correlation method. can be 'spearman' or `person`. *Default is
            ``pearson``
            
        features: List, optional 
            list of  the name of features for corroleation analysis. If given, 
            must be sure that the names belongs to the dataframe columns,  
            otherwise an error will occur. If features are valid, dataframe 
            is shrunk to the number of features before the correlation plot.
       
        min_periods: 
                Minimum number of observations required per pair of columns
                to have a valid result. Currently only available for 
                ``pearson`` and ``spearman`` correlation. For more details 
                refer to https://www.geeksforgeeks.org/python-pandas-dataframe-corr/
        
        sns_kws: Other seabon heatmap arguments. Refer to 
                https://seaborn.pydata.org/generated/seaborn.heatmap.html
                
        Returns
        -------
        :class:`QuickPlot` instance
            Returns ``self`` for easy method chaining.
             
           
        Example 
        ---------
        >>> from watex.view.plot import QuickPlot 
        >>> qplotObj = QuickPlot().fit('../data/geodata/main.bagciv.data.csv')
        >>> sns_kwargs ={'annot': False, 
        ...            'linewidth': .5, 
        ...            'center':0 , 
        ...            # 'cmap':'jet_r', 
        ...            'cbar':True}
        >>> qplotObj.correlationMatrix(cortype='cat', **sns_kwargs) 
            
        """
        corc = str(copy.deepcopy(cortype))
        cortype= str(cortype).lower().strip() 
        if cortype.find('num')>=0 or cortype in (
                'value', 'digit', 'quan', 'quantitative'): 
            cortype ='num'
        elif cortype.find('cat')>=0 or cortype in (
                'string', 'letter', 'qual', 'qualitative'): 
            cortype ='cat'
        if cortype not in ('num', 'cat'): 
            return ValueError ("Expect 'num' or 'cat' for numerical and"
                               f" categorical features, not : {corc!r}")
        
        if data is not None : 
            self.data = data
        
        df_= self.data.copy(deep=True)
        # df_.reset_index(inplace=True )
        
        df_ = selectfeatures(df_, features = features , 
                             include= 'number' if cortype  =='num' else None, 
                             exclude ='number' if cortype=='cat' else None,
                             )
        features = list(df_.columns ) # for consistency  

        if cortype =='cat': 
            for ftn in features: 
                df_[ftn] = df_[ftn].astype('category').cat.codes 
        
        elif cortype =='num': 
           
            if 'id' in features: 
                features.remove('id')
                df_= df_.drop('id', axis=1)

        ax= sns.heatmap(data =df_[list(features)].corr(
            method= method, min_periods=min_periods), 
                **sns_kws
                )

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(self.fig_title)
        
        if self.savefig is not None :
            plt.savefig(self.savefig,dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
            
        plt.show()  if self.savefig is None else plt.close() 
        
        print(" --> Correlation matrix plot successfully done !" 
              ) if self.verbose > 0 else print()
              
        return self 
    
              
    def numfeatures(self, 
                    data: str | DataFrame =None ,
                    features=None, 
                    coerce: bool= False,  
                    map_lower_kws=None, **sns_kws): 
        """
        Plot qualitative features distribution using correlative aspect. Be 
        sure to provided numerical features arguments. 
        
        Parameters 
        -----------
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`QuickPlot` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 
            
        features: list
            List of numerical features to plot for correlating analyses. 
            will raise an error if features does not exist in the data 
            
        coerce: bool, 
            Constraint the data to read all features and keep only the numerical
            values. An error occurs if ``False`` and the data contains some 
            non-numericalfeatures. *default* is ``False``. 
            
        map_lower_kws: dict, Optional
            a way to customize plot. Is a dictionnary of sns.pairplot map_lower
            kwargs arguments. If the diagram `kind` is ``kde``, plot is customized 
            with the provided `map_lower_kws` arguments. if ``None``, 
            will check whether the `diag_kind` argument on `sns_kws` is ``kde``
            before triggering the plotting map. 
            
        sns_kws: dict, 
            Keywords word arguments of seabon pairplots. Refer to 
            http://seaborn.pydata.org/generated/seaborn.pairplot.html for 
            further details.             
                     
        Returns
        -------
        :class:`QuickPlot` instance
            Returns ``self`` for easy method chaining.
            
              
        Examples
        ---------
            
            >>> from watex.view.plot import QuickPlot 
            >>> data = '../data/geodata/main.bagciv.data.csv'
            >>> qkObj = QuickPlot(mapflow =True, nameoftarget='flow'
                                      ).fit(data)
            >>> qkObj.sns_style ='darkgrid', 
            >>> qkObj.fig_title='Quantitative features correlation'
            >>> sns_pkws={'aspect':2 , 
            ...          "height": 2, 
            ...          'markers':['o', 'x', 'D', 'H', 's'], 
            ...          'diag_kind':'kde', 
            ...          'corner':False,
            ...          }
            >>> marklow = {'level':4, 
            ...          'color':".2"}
            >>> qkObj.numfeatures(coerce=True, map_lower_kws=marklow, **sns_pkws)
                                                
        """
        if data is not None : 
            self.data = data
            
        df_= self.data.copy(deep=True)
        
        try : 
            df_= df_.astype(float) 
        except: 
            if not coerce:
                non_num = list(selectfeatures(df_, exclude='number').columns)
                msg = f"non-numerical features detected: {smart_format(non_num)}"
                warnings.warn(msg + "set 'coerce' to 'True' to only visualize"
                              " the numerical features.")
                raise ValueError (msg + "; set 'coerce'to 'True' to keep the"
                                  " the numerical insights")
   
        df_= selectfeatures(df_, include ='number')
        
        ax =sns.pairplot(data =df_, hue=self.nameoftarget,**sns_kws)
        
        if map_lower_kws is not None : 
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
                    
        if self.savefig is not None :
            plt.savefig(self.savefig, dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
            
        plt.show() if self.savefig is None else plt.close () 
        
        return self 
    
    def joint2features(self,features: List [str], *,
                       data: str | DataFrame=None, 
                       join_kws=None, marginals_kws=None, 
                       **sns_kws):
        """
        Joint methods allow to visualize correlation of two features. 
        
        Draw a plot of two features with bivariate and univariate graphs. 
        
        Parameters 
        -----------
        features: list
            List of numerical features to plot for correlating analyses. 
            will raise an error if features does not exist in the data 
        
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`QuickPlot` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 

        join_kws:dict, optional 
            Additional keyword arguments are passed to the function used 
            to draw the plot on the joint Axes, superseding items in the 
            `joint_kws` dictionary.
            
        marginals_kws: dict, optional 
            Additional keyword arguments are passed to the function used 
            to draw the plot on the marginals Axes. 
            
        sns_kwargs: dict, optional
            keywords arguments of seaborn joinplot methods. Refer to 
            :ref:`<http://seaborn.pydata.org/generated/seaborn.jointplot.html>` 
            for more details about usefull kwargs to customize plots. 
          
        Returns
        -------
        :class:`QuickPlot` instance
            Returns ``self`` for easy method chaining.
              
             
        Example
        --------
        >>> from watex.view.plot import QuickPlot 
        >>> data = r'../data/geodata/main.bagciv.data.csv'
        >>> qkObj = QuickPlot( lc='b', sns_style ='darkgrid', 
        ...             fig_title='Quantitative features correlation'
        ...             ).fit(data)  
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
  
        df_= self.data.copy(deep=True)
        
        if isinstance (features, str): 
            features =[features]
            
        if features is None: 
            self._logging.error(f"Valid features are {smart_format(df_.columns)}")
            raise PlotError("NoneType can not be a feature nor plotted.")
            
        df_= selectfeatures(df_, features)

        # checker whether features is quantitative features 
        df_ = selectfeatures(df_, include= 'number') 
        
        if len(df_.columns) != 2: 
            raise PlotError(f" Joinplot needs two features. {len(df_.columns)}"
                            f" {'was' if len(df_.columns)<=1 else 'were'} given")
            
            
        ax= sns.jointplot(data=df_, x=features[0], y=features[1],   **sns_kws)

        if join_kws is not None:
            join_kws = _assert_all_types(join_kws,dict)
            ax.plot_joint(sns.kdeplot, **join_kws)
            
        if marginals_kws is not None: 
            marginals_kws= _assert_all_types(marginals_kws,dict)
            
            ax.plot_marginals(sns.rugplot, **marginals_kws)
            
        plt.show() if self.savefig is None else plt.close () 
        
        if self.savefig is not None :
            plt.savefig(self.savefig,dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
            
        return self 
          
    def scatteringfeatures(self,features: List [str], *,
                           data: str | DataFrame=None,
                           relplot_kws= None, 
                           **sns_kws ): 
        """
        Draw a scatter plot with possibility of several semantic features 
        groupings.
        
        Indeed `scatteringFeatures` analysis is a process of understanding 
        how features in a dataset relate to each other and how those
        relationships depend on other features. Visualization can be a core 
        component of this process because, when data are visualized properly,
        the human visual system can see trends and patterns that indicate a 
        relationship. 
        
        Parameters 
        -----------
        features: list
            List of numerical features to plot for correlating analyses. 
            will raise an error if features does not exist in the data 
        
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`QuickPlot` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 

        relplot_kws: dict, optional 
            Extra keyword arguments to show the relationship between 
            two features with semantic mappings of subsets.
            refer to :ref:`<http://seaborn.pydata.org/generated/seaborn.relplot.html#seaborn.relplot>`
            for more details. 
            
        sns_kwargs:dict, optional
            kwywords arguments to control what visual semantics are used 
            to identify the different subsets. For more details, please consult
            :ref:`<http://seaborn.pydata.org/generated/seaborn.scatterplot.html>`. 
            
            
        Returns
        -------
        :class:`QuickPlot` instance
            Returns ``self`` for easy method chaining.
              
        Examples
        ----------
        >>> from watex.view.plot import  QuickPlot 
        >>> data = r'../data/geodata/main.bagciv.data.csv'
        >>> qkObj = QuickPlot(lc='b', sns_style ='darkgrid', 
        ...             fig_title='geol vs lewel of water inflow',
        ...             xlabel='Level of water inflow (lwi)', 
        ...             ylabel='Flow rate in m3/h'
        ...            ) 
        >>>
        >>> qkObj.nameoftarget='flow' # target the DC-flow rate prediction dataset
        >>> qkObj.mapflow=True  # to hold category FR0, FR1 etc..
        >>> qkObj.fit(data) 
        >>> marker_list= ['o','s','P', 'H']
        >>> markers_dict = {key:mv for key, mv in zip( list (
        ...                       dict(qkObj.data ['geol'].value_counts(
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
        >>> qkObj.scatteringfeatures(features=['lwi', 'flow'],
        ...                         relplot_kws=regpl_kws,
        ...                         **sns_pkws, 
        ...                    ) 
            
        """
        if data is not None : 
            self.data = data
            
        df_= self.data.copy(deep=True)
        
        # controller function
        if isinstance (features, str): 
            features =[features]
            
        if features is None: 
            self._logging.error(f"Valid features are {smart_format(df_.columns)}")
            raise PlotError("NoneType can not be a feature nor plotted.")
            
        if len(features) < 2: 
            raise PlotError(f" Scatterplot needs at least two features. {len(df_.columns)}"
                            f" {'was' if len(df_.columns)<=1 else 'were'} given")
            
        # assert wether the feature exists 
        selectfeatures(df_, features)
        
        ax= sns.scatterplot(data=df_,  x=features[0], y=features[1],
                              **sns_kws)
        
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(self.fig_title)
        
        if relplot_kws is not None: 
            relplot_kws = _assert_all_types(relplot_kws, dict)
            sns.relplot(data=df_, x= features[0], y=features[1],
                        **relplot_kws)
            
        if self.savefig is not None :
            plt.savefig(self.savefig,dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
            
        plt.show() if self.savefig is None else plt.close ()    
        
        return self 
       
    def discussingfeatures(self, features=['ohmS','sfi', 'geol', 'flow'], *, 
                           data: str | DataFrame= None,
                           map_kws: Optional[dict]=None, 
                           map_func: Optional[F] = None, 
                           **sns_kws)-> None: 
        """
        Provides the features names at least 04 and discuss with 
        their distribution. 
        
        This method maps a dataset onto multiple axes arrayed in a grid of
        rows and columns that correspond to levels of features in the dataset. 
        The plots it produces are often called “lattice”, “trellis”, or
        'small-multiple' graphics. 
        
        Parameters 
        -----------
        features: list
            List of features for discussing. The number of recommended 
            features for better analysis is four (04) classified as below: 
                
                features_disposal = ['x', 'y', 'col', 'target|hue']
                
            where: 
                - `x` is the features hold to the x-axis, *default* is``ohmS`` 
                - `y` is the feature located on y_xis, *default* is ``sfi`` 
                - `col` is the feature on column subset, *default` is ``col`` 
                - `target` or `hue` for targetted examples, *default* is ``flow``
            
            If 03 `features` are given, the latter is considered as a `target`
        
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`QuickPlot` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 

        map_kws:dict, optional 
            Extra keyword arguments for mapping plot.
            
        func_map: callable, Optional 
            callable object,  is a plot style function. Can be a 'matplotlib-pyplot'
            function  like ``plt.scatter`` or 'seaborn-scatterplot' like 
            ``sns.scatterplot``. The *default* is ``sns.scatterplot``.
  
        sns_kwargs: dict, optional
           kwywords arguments to control what visual semantics are used 
           to identify the different subsets. For more details, please consult
           :ref:`<http://seaborn.pydata.org/generated/seaborn.FacetGrid.html>`. 
        
        Returns
        -------
        :class:`QuickPlot` instance
            Returns ``self`` for easy method chaining.

        Example
        --------
        >>> from watex.view.plot import  QuickPlot 
        >>> data = r'../data/geodata/main.bagciv.data.csv'
        >>> qkObj = QuickPlot(  leg_kws={'loc':'upper right'},
        ...          fig_title = '`sfi` vs`ohmS|`geol`',
        ...            ) 
        >>> qkObj.nameoftarget='flow' # target the DC-flow rate prediction dataset
        >>> qkObj.mapflow=True  # to hold category FR0, FR1 etc..
        >>> qkObj.fit(data) 
        >>> sns_pkws={'aspect':2 , 
        ...          "height": 2, 
        ...                  }
        >>> map_kws={'edgecolor':"w"}   
        >>> qkObj.discussingfeatures(features =['ohmS', 'sfi','geol', 'flow'],
        ...                           map_kws=map_kws,  **sns_pkws
        ...                         )   
        """
        if data is not None : 
            self.data = data

        df_= self.data.copy(deep=True)
        
        if isinstance (features, str ): 
            features =[features]
            
        if len(features)>4: 
            if self.verbose:  
                self._logging.debug(
                    'Features length provided is = {0:02}. The first four '
                    'features `{1}` are used for joinplot.'.format(
                        len(features), features[:4]))
                
            features=list(features)[:4]
            
        elif len(features)<=2: 
            if len(features)==2:verb, pl='are','s'
            else:verb, pl='is',''
            
            if self.verbose: 
                self._logging.error(
                    'Expect three features at least. {0} '
                    '{1} given.'.format(len(features), verb))
            
            raise PlotError(
                '{0:02} feature{1} {2} given. Expect at least 03 '
                'features!'.format(len(features),pl,  verb))
            
        elif len(features)==3:
            
            msg='03 Features are given. The last feature `{}` should be'\
                ' considered as the`targetted`feature or `hue` value.'.format(
                    features[-1])
            if self.verbose: 
                self._logging.debug(msg)
            
                warnings.warn(
                    '03 features are given, the last one `{}` is used as '
                    'target!'.format(features[-1]))
            
            features.insert(2, None)
    
        ax= sns.FacetGrid(data=df_, col=features[-2], hue= features[-1], 
                            **sns_kws)
        
        if map_func is None: 
            map_func = sns.scatterplot #plt.scatter
            
        if map_func is not None : 
            if not hasattr(map_func, '__call__'): 
                raise TypeError(
                    f'map_func must be a callable object not {map_func.__name__!r}'
                    )

        if map_kws is None : 
            map_kws = _assert_all_types(map_kws,dict)
            map_kws={'edgecolor':"w"}
            
        if (map_func and map_kws) is not None: 
            ax.map(map_func, features[0], features[1], 
                   **map_kws).add_legend(**self.leg_kws) 
      

        if self.savefig is not None :
            plt.savefig(self.savefig, dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
        
        plt.show() if self.savefig is None else plt.close ()
         
        return self 
         
    def discover_and_visualize(self,
                               data: str | DataFrame= None, 
                               x:str =None, y:str =None, kind:str ='scatter',
                               s_col ='lwi', leg_kws:dict ={}, **pd_kws):
        """ Create a scatter plot to visualize the data using `x` and `y` 
        considered as dataframe features. 
        
        Parameters 
        -----------
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`QuickPlot` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 

        x: str , 
            Column name to hold the x-axis values 
        y: str, 
            column na me to hold the y-axis values 
        s_col: column for scatter points. 'Default is ``fs`` time the features
            column `lwi`.
            
        pd_kws: dict, optional, 
            Pandas plot keywords arguments 
            
        leg_kws:dict, kws 
            Matplotlib legend keywords arguments 
            
        Returns
        -------
        :class:`QuickPlot` instance
            Returns ``self`` for easy method chaining.
            
        Example
        --------- 
        >>> import watex.tools.mlutils as mfunc
        >>> from watex.bases.transformers import StratifiedWithCategoryAdder
        >>> from watex.view.plot import QuickPlot
        >>> data = r'../data/geodata/main.bagciv.data.csv'
        >>> df = mfunc.load_data(data)
        >>> stratifiedNumObj= StratifiedWithCategoryAdder('flow')
        >>> strat_train_set , *_= \
        ...    stratifiedNumObj.fit_transform(X=df) 
        >>> pd_kws ={'alpha': 0.4, 
        ...         'label': 'flow m3/h', 
        ...         'c':'flow', 
        ...         'cmap':plt.get_cmap('jet'), 
        ...         'colorbar':True}
        >>> qkObj=QuickPlot(fs=25.)
        >>> qkObj.fit(strat_train_set)
        >>> qkObj.discover_and_visualize( x= 'east', y='north', **pd_kws)
    
        """
        if data is not None : 
            self.data = data
            
        if self.data is None: 
            raise NotFittedError("Fit the {self.__class__.__name__!r} instance.")
            
        df_= self.data.copy(deep=True)
        
         # visualize the data and get insights
        if 's' not in pd_kws.keys(): 
            pd_kws['s'] = df_[s_col]* self.fs 
             
        df_.plot(kind=kind, x=x, y=y, **pd_kws)
        
        self.leg_kws = self.leg_kws or dict () 
        
        plt.legend(**leg_kws)
        
        if self.savefig is not None :
            plt.savefig(self.savefig,dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
            
        plt.show () if self.savefig is None else plt.close()
        
        return self 

    
ExPlot .__doc__=r"""\
Exploratory plot for data analysis 

`ExPlot` is a shadow class. Explore data is needed to create a model since 
it gives a feel for the data and also at great excuses to meet and discuss 
issues with business units that controls the data. `ExPlot` methods i.e. 
return an instancied object that inherits from :class:`watex.property.Baseplots`
ABC (Abstract Base Class) for visualization.
    
Parameters 
-------------
{params.base.savefig}
{params.base.fig_dpi}
{params.base.fig_num}
{params.base.fig_size}
{params.base.fig_orientation}
{params.base.fig_title}
{params.base.fs}
{params.base.ls}
{params.base.lc}
{params.base.lw}
{params.base.alpha}
{params.base.font_weight}
{params.base.font_style}
{params.base.font_size}
{params.base.ms}
{params.base.marker}
{params.base.marker_facecolor}
{params.base.marker_edgecolor}
{params.base.marker_edgewidth}
{params.base.xminorticks}
{params.base.yminorticks}
{params.base.bins}
{params.base.xlim}
{params.base.ylim}
{params.base.xlabel}
{params.base.ylabel}
{params.base.rotate_xlabel}
{params.base.rotate_ylabel}
{params.base.leg_kws}
{params.base.plt_kws}
{params.base.glc}
{params.base.glw}
{params.base.galpha}
{params.base.gaxis}
{params.base.gwhich}
{params.base.tp_axis}
{params.base.tp_labelsize}
{params.base.tp_bottom}
{params.base.tp_labelbottom}
{params.base.tp_labeltop}
{params.base.cb_orientation}
{params.base.cb_aspect}
{params.base.cb_shrink}
{params.base.cb_pad}
{params.base.cb_anchor}
{params.base.cb_panchor}
{params.base.cb_label}
{params.base.cb_spacing}
{params.base.cb_drawedges} 
{params.sns.sns_orient}
{params.sns.sns_style}
{params.sns.sns_palette}
{params.sns.sns_height}
{params.sns.sns_aspect}

Returns
--------
{returns.self}

Examples
---------
>>> import pandas as pd 
>>> from watex.view import ExPlot
>>> data = pd.read_csv ('data/geodata/main.bagciv.data.csv' ) 
>>> ExPlot(fig_size = (12, 4)).fit(data).missing(kind ='corr')
... <watex.view.plot.ExPlot at 0x21162a975e0>
""".format(
    params=_param_docs,
    returns= _core_docs["returns"],
)
        

        
# import matplotlib.cm as cm 
# import matplotlib.colorbar as mplcb
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.ticker import MultipleLocator, NullLocator
# import matplotlib.gridspec as gspec        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        