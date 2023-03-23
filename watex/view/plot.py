# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
:mod:`~watex.utils.plot` is a set of base plots for :term:`tensor` 
visualization, data exploratory and analyses. 
T-E-Q Plots encompass the tensors plots (:class:`~watex.view.TPlot`) dealing 
with :term:`EM` methods, Exploratory plots ( :class:`~watex.view.ExPlot`) and 
Quick analyses (:class:`~watex.view.QuickPlot`) visualization. 
"""
from __future__ import annotations 

import re 
import copy
import warnings
import itertools
import numpy as np 
import  matplotlib.pyplot  as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec 
import pandas as pd 
from pandas.plotting import ( 
    radviz , 
    parallel_coordinates
    ) 
import seaborn as sns 
from .._docstring import ( 
    DocstringComponents,
    _core_docs,
    _baseplot_params
    )
from .._watexlog import watexlog  
from ..decorators import temp2d 
from ..cases.features import FeatureInspection
from ..exceptions import ( 
    PlotError, 
    FeatureError, 
    NotFittedError, 
    EMError, 
    )
from ..property import BasePlot
from .._typing import (
    Any , 
    List,
    Dict,
    Optional,
    ArrayLike, 
    DataFrame, 
    Series,
    F, 
    EDIO
)
from ..utils._dependency import ( 
    import_optional_dependency )
from ..utils.coreutils import _is_readable
from ..utils.exmath import ( 
    moving_average , fittensor)
from ..utils.funcutils import ( 
    _assert_all_types , 
    _validate_name_in, 
    _isin, 
    repr_callable_obj, 
    smart_strobj_recognition, 
    remove_outliers, 
    smart_format,
    reshape, 
    shrunkformat, 
    is_iterable, 
    station_id, 
    make_ids,
    )
from ..utils.mlutils import (
    existfeatures,
    formatGenericObj, 
    selectfeatures , 
    exporttarget 
    )
from ..utils.plotutils import( 
    make_mpl_properties, 
     plot_errorbar
     )
from ..utils.validator import check_X_y 
try: 
    import missingno as msno 
except : pass 

try : 
    from yellowbrick.features import (
        JointPlotVisualizer, 
        Rank2D, 
        RadViz, 
        ParallelCoordinates,
        )
except: pass 

try : 
    from ..methods.em import ( 
        Processing, 
        ZC  
        )
except: pass 

  
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
_qkp_params = dict (  
    classes ="""
classes: list of int | float, [categorized classes] 
    list of the categorial values encoded to numerical. For instance, for
    `flow` data analysis in the Bagoue dataset, the `classes` could be 
    ``[0., 1., 3.]`` which means:: 
        
    * 0 m3/h  --> FR0
    * > 0 to 1 m3/h --> FR1
    * > 1 to 3 m3/h --> FR2
    * > 3 m3/h  --> FR3    
    """, 
    mapflow ="""   
mapflow: bool, 
    Is refer to the flow rate prediction using DC-resistivity features and 
    work when the `tname` is set to ``flow``. If set to True, value 
    in the target columns should map to categorical values. Commonly the 
    flow rate values are given as a trend of numerical values. For a 
    classification purpose, flow rate must be converted to categorical 
    values which are mainly refered to the type of types of hydraulic. 
    Mostly the type of hydraulic system is in turn tided to the number of 
    the living population in a specific area. For instance, flow classes 
    can be ranged as follow: 

    * FR = 0 is for dry boreholes
    * 0 < FR ≤ 3m3/h for village hydraulic (≤2000 inhabitants)
    * 3 < FR ≤ 6m3/h  for improved village hydraulic(>2000-20 000inhbts) 
    * 6 <FR ≤ 10m3/h for urban hydraulic (>200 000 inhabitants). 
    
    Note that the flow range from `mapflow` is not exhaustive and can be 
    modified according to the type of hydraulic required on the project.   
    """
)
_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"], 
    base=DocstringComponents(_baseplot_params), 
    sns = DocstringComponents(_sns_params), 
    qdoc= DocstringComponents(_qkp_params)
    )
#++++++++++++++++++++++++++++++++++ end +++++++++++++++++++++++++++++++++++++++

class TPlot (BasePlot): 

    _t= (
        "survey_area",
        "distance",
        "prefix",
        "window_size",
        "component",
        "mode",
        "method",
        "out",
        "how",
        "c" 
        )
    
    def __init__ (
        self, 
        survey_area =None , 
        distance = 50., 
        prefix ='S', 
        how= 'py',
        window_size:int =5, 
        component:str ='xy', 
        mode: str ='same', 
        method:str ='slinear', 
        out:str  ='srho', 
        c: int =2,
        **kws
        ): 
        super().__init__(**kws)
        
        self.survey_area=survey_area 
        self.distance=distance 
        self.prefix=prefix
        self.window_size=window_size
        self.component=component 
        self.mode=mode
        self.method=method
        self.out=out
        self.how=how
        self.c=c 
        
    def fit (
        self, 
        data: Optional [str|List[EDIO]]
        ): 
        """
        Fit data and populate attributes. 
        
        Parameters 
        ----------- 
        data : str, or list or :class:`pycsamt.core.edi.Edi` object 
            Full path to EDI files or collection of EDI-objects 
   
        Returns
        -------- 
        ``self``: :class:`watex.view.plot.TPlot` instanciated object
            returns ``self`` for chaining methods.
        
        """

        p = Processing(
            window_size = self.window_size ,  
            component= self.component, 
            mode= self.mode, 
            method= self.method, 
            out=self.out, 
            c=self.c 
            ) 
        p.fit(data)
        # set EM processing module 
        # as an attr
        setattr (self, "p_", p )
        
        # set component slices into a dict
        self._c_= {
              'xx': [slice (None, len(self.p_.freqs_)), 0 , 0] , 
              'xy': [slice (None, len(self.p_.freqs_)), 0 , 1], 
              'yx': [slice (None, len(self.p_.freqs_)), 1 , 0], 
              'yy': [slice (None, len(self.p_.freqs_)), 1,  1] 
        }
        return self
    
    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'p_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 
    
    
    def plot_multi_recovery(
            self,  
            sites:str |List[str | int], 
            colors: List[str] = None, 
            **kws
            ): 
        
        """ 
        Plots mutiple site/stations with signal recovery. 
        
        Parameters 
        -----------
        sites: list 
            list of sites to visualize. Can also be the index of the sites 
        colors: list of str
            matplotlib colors to customize the raw signal and recovery signal
     
        Returns 
        ----------
        ax: Matplotlib suplot axes 
        
        Examples
        --------
        >>> from watex.view.plot import TPlot 
        >>> from watex.datasets import load_edis 
        >>> # takes the 03 samples of EDIs 
        >>> edi_data = load_edis (return_data= True, samples =3 ) 
        >>> TPlot(fig_size =(5, 3)).fit(edi_data).plot_multi_recovery (
            sites =['S00'], colors =['o', 'ok--'])
        <AxesSubplot:title={'center':'Recovered tensor $|Z_{xy}|$'}, 
        xlabel='$Frequency [H_z]$', ylabel='$ App.resistivity \\quad xy \\quad [ \\Omega.m]$'>
        """
        self.inspect 

        if isinstance (sites, str): 
            sites =[sites ] 
        if not is_iterable(sites): 
            sites =[sites] 
       
        site_index = station_id(sites) 
        for i, s in enumerate (site_index): 
            if s > len(self.p_.ediObjs_): 
                raise PlotError(f"Site {sites[i]!r} is out of the expected"
                                f" sites number: {len(self.p_.ediObjs_)}"
                                )
        # read the component XY 
        res2d_r = self.p_.make2d (out=f'res{self.component}') 
        z_xy_rest = self.p_.zrestore() # no buffered data 
        # extracted the station at index 12, 27 for instance.
        zs = [ z_xy_rest[s].resistivity[
            tuple (self._c_.get(self.component))] for s in site_index ]

        ma = [ moving_average ( res2d_r[:,  s_ix  ]) for s_ix in site_index ]

        f= self.p_.getfullfrequency()
        #>>> # ---> make a plot and color 
        # colors = make_mpl_properties(2*len(ma))
        if colors is None: 
            colors =[]
        if isinstance (colors, str): 
            colors =[colors]
        colors +=  make_mpl_properties(2*len(ma))
        
        fs = [f for i in range(len(ma))] # repeat frequency 
        z_norm_args = list( zip (fs, zs, colors[: len(ma)] )) 
        args  = list(itertools.chain(*z_norm_args))
        # >>> #  make a fitting args 
        colors = ['m-'] + colors[len(ma):]
        z_cor_objs = list( zip (fs, ma, ['m-'] + colors[len(ma):] )) 
        fit_args = list(itertools.chain(*z_cor_objs))
        
        xlim = (f.min() -.5 * f.min(), f.max() +.5 * f.max())
        return self._plot_recovery (
            *args, fit_args= fit_args, xlim=xlim, sites = sites,  **kws )

    def _plot_recovery (
            self,
            *args,  
            fit_args = None, 
            leg= None,  
            xlim=None, 
            sites=None, 
            **kws
            ): 
        """" Template to plot two stations with signal recovery 
        
        Isolated part of :meth:`~.TPlot.plot_multi_recovery`. 
        
        Parameters 
        -----------
        *args : args : list 
            Matplotlib logs funtions plot arguments 
            
        fit_args : list or tuple 
            Matplotlib logs funtions plot arguments put on list. It used to 
            visualize the fitting curve after apply anay correction to the data.
            
            X-coordinates. It should have the length M, the same of the ``arr2d``; 
            the columns of the 2D dimensional array.  Note that if `x` is 
            given, the `distance is not needed. 
            
        leg: list 
            legend labels put into a list. It must fit the number of given 
            plots. 
             
        kws : dict 
            Additional keywords arguments of Matplotlib subsplots function  
            :func:`plt.loglog` or :func:`plt.semilog`
        
        Returns 
        ------- 
        ax: Matplotlib.pyplot <AxesSubplot>  

        """
        fig, ax = plt.subplots(
            1,figsize = self.fig_size, 
            #num = self.fig_num,
             )
        p1= ax.loglog(*args,  
                  markersize = self.ms ,
                  linewidth = self.lw ,
                  **kws 
                  )
        p2 =[]
        if fit_args  is not None: 
            fit_args  = _assert_all_types(
                fit_args , list, tuple, objname="Fit arguments")  
            p2 = ax.loglog(*fit_args,
                      markersize = self.ms ,
                      linewidth = self.lw 
                      )

        ax.set_xlabel (self.xlabel or '$Frequency [H_z]$',
                    fontsize =1.5 * self.font_size ) 
        ax.set_ylabel(self.ylabel or '$ App.resistivity \quad xy \quad [ \Omega.m]$',
                   fontsize =1.5*self.font_size)
        
        p1labels= [f'rec.tensor {i}' for i in sites ]
        p2labels= [f'mov-aver. line {i}' for i in sites 
                   ] if fit_args is not None else []
        
        ax.legend (handles = [*p1 ,*p2], 
                   labels= [*p1labels, *p2labels] #['restored data' , 'recovery trend '] 
                   if leg is  None else leg,
                   loc ='best', 
                   # ncol =len(args)//3 if fit_args  is None else (
                   #      (len(args)+len(fit_args )))//3  ,
                    fontsize =1.5 * self.font_size
                    )
    
        if xlim is not None: 
            ax.set_xlim (xlim)
        ax.tick_params (axis= 'both', labelsize = 1.5 * self.font_size )
        plt.title (self.fig_title or  'Recovered tensor $|Z_{xy}|$',
                   fontsize =1.5*self.font_size)
        
        
        if self.show_grid :
            ax.grid (visible =True , alpha =self.galpha,
                     which =self.gwhich, color =self.gc)
            
        if self.savefig is not None: 
             plt.savefig(self.savefig , dpi = self.fig_dpi)
             plt.close (fig =fig ) 

        return ax
    
    @temp2d("Base template for 2D recovery tensors plot.")
    def plot_tensor2d (
        self,    
        tensor='res', 
        sites =None, 
        to_log10=False, 
        ): 
        """ Plot two dimensional tensor. 
         
        Parameters 
        -----------
        freqs: array-like 
            y-coordinates. It should have the length N, the same of the ``arr2d``.
            the rows of the ``arr2d``.Frequency array. It should be the 
            complete frequency used during the survey area. 

        tensor: str , ['res','phase', 'z'], default='res'
            kind of tensor to plot. Can be resistivity or phase. If `phase`, 
            customize your plot to not fit the default 'res' behaviour. 
        to_log10: bool, defaut=False, 
            Convert the resistivity data and frequeny in log10.  
            
        sites: list of str, optional 
            List of stations/sites names. If given, it must have the same 
            length of the positions in of the EDI data. Must fit the number 
            of 'EDI' succesffully read. 

        Returns 
        -------
        ( arr2d , freqs, positions , sites , base_plot_kws): 
            - arr2d: 2D resistivity array from the tensor `component` 
            - freqs: array-like 1d of frequency in the survey.
            - positions: Sites/stations positions. It is equals to the distance
                between stations times the number of sites 
            - sites: list of the names of the station/sites 
            - base_plot_kws: plot keywords arguments inherits from 
                :class:`watex.property.BasePlot`. It composes the last 
                parameters for customizing plot as decorated return function. 
    
        Examples 
        -------- 
        >>> from watex.view.plot import TPlot 
        >>> from watex.datasets import load_edis 
        >>> # get some 3 samples of EDI for demo 
        >>> edi_data = load_edis (return_data =True, samples =3 )
        >>> # customize plot by adding plot_kws 
        >>> plot_kws = dict( ylabel = '$Log_{10}Frequency [Hz]$', 
                            xlabel = '$Distance(m)$', 
                            cb_label = '$Log_{10}Rhoa[\Omega.m$]', 
                            fig_size =(6, 3), 
                            font_size =7. 
                            ) 
        >>> t= TPlot(**plot_kws ).fit(edi_data)
        >>> # plot recovery2d using the log10 resistivity 
        >>> t.plot_tensor2d (to_log10=True)
        <AxesSubplot:xlabel='$Distance(m)$', ylabel='$Log_{10}Frequency [Hz]$'>
 
        """
        
        self.inspect 
        
        assert  str(tensor).lower() in {"res", 'phase'}, (
            "Expect either a resistivity 'res' or 'phase'. Got {tensor!r}")
        tensor =str(tensor).lower() 
        
        arr2d = self.p_.make2d (out = f'{tensor}{self.component}') 

        return self._make_tensor_utils (arr2d, sites , to_log10, tensor )  
    
    @temp2d("Base template for 2D filtered tensors plot.")
    def plot_ctensor2d  (
            self, 
            tensor ='res' , 
            ffilter ='tma', 
            sites = None, 
            to_log10=False
            ): 
        """ Plot filtered tensors 
        
        Parameters 
        -----------
        tensor: str , ['res','phase', 'z'], default='res'
            kind of tensor to plot. Can be resistivity or phase. If `phase`, 
            customize your plot to not fit the default 'res' behaviour.
            
        ffilter: str ['ama', 'flma', 'tma'], default='tma'
            kind of appropriate filter to corrected tensor data. 
            
        to_log10: bool, defaut=False, 
            Convert the resistivity data and frequeny in log10.  
            
        sites: list of str, optional 
            List of stations/sites names. If given, it must have the same 
            length of the positions in of the EDI data. Must fit the number 
            of 'EDI' succesffully read. 
            
        Returns 
        -------
        ( arr2d , freqs, positions , sites , base_plot_kws): 
            - arr2d: 2D filtered tensor array from the `component` 
            - freqs: array-like 1d of frequency in the survey.
            - positions: Sites/stations positions. It is equals to the distance
                between stations times the number of sites 
            - sites: list of the names of the station/sites 
            - base_plot_kws: plot keywords arguments inherits from 
                :class:`watex.property.BasePlot`. It composes the last 
                parameters for customizing plot as decorated return function.
                
        Examples 
        -------- 
        >>> from watex.view.plot import TPlot 
        >>> from watex.datasets import load_edis 
        >>> # get some 3 samples of EDI for demo 
        >>> edi_data = load_edis (return_data =True, samples =3 )
        >>> # customize plot by adding plot_kws 
        >>> plot_kws = dict( ylabel = '$Log_{10}Frequency [Hz]$', 
                            xlabel = '$Distance(m)$', 
                            cb_label = '$Log_{10}Rhoa[\Omega.m$]', 
                            fig_size =(6, 3), 
                            font_size =7. 
                            ) 
        >>> t= TPlot(**plot_kws ).fit(edi_data)
        >>> # plot filtered tensor using the log10 resistivity 
        >>> t.plot_ctensor2d (to_log10=True)
        <AxesSubplot:xlabel='$Distance(m)$', ylabel='$Log_{10}Frequency [Hz]$'>
  
        """
        self.inspect 
        fd = {"tma": self.p_.tma , "flma":self.p_.flma, "ama":self.p_.ama }

        assert str(ffilter).lower() in fd.keys(), (
            "Supports only base filters {tuple (fd.keys())}. Got {ffilter!r}"
            " To apply a simple filter like 'moving average' to a tensor, refer"
            " to <watex.utils.exmath.moving_average>. For other filters like"
            " 'Savitzky Golay1d/2d', 'remove distorsion' or 'remove outliers'"
            " and else, use the package 'pycsamt' instead. "
            ) 
        ffilter= str (ffilter).lower().strip()         
        arr2d = fd.get(ffilter)()
    
        return self._make_tensor_utils (arr2d, sites, to_log10 , tensor ) 
    
    
    def _make_tensor_utils (
            self, arr2d, sites, to_log10= False, tensor=None ): 
        """ Make utilities for plotting tensors   
        
        Parameters 
        ------------
        arr2d: arraylike of shape (n_freq, n_sites): 
            Array of the tensor composed of number of frequency and number 
            of sites that fit the number of EDI correctly read.
        
        sites: list of str, optional 
            List of stations/sites names. If given, it must have the same 
            length of the positions in of the EDI data. Must fit the number 
            of 'EDI' succesffully read. 
        to_log10: bool, defaut=False, 
            Convert the resistivity data and frequeny in log10.
            
        Returns 
        -------
        ( arr2d , freqs, positions , sites , base_plot_kws): 
            - arr2d: 2D filtered tensor array from the `component` 
            - freqs: array-like 1d of frequency in the survey.
            - positions: Sites/stations positions. It is equals to the distance
                between stations times the number of sites 
            - sites: list of the names of the station/sites 
            - base_plot_kws: plot keywords arguments inherits from 
                :class:`watex.property.BasePlot`. It composes the last 
                parameters for customizing plot as decorated return function.
        """
        try : 
            distance = float(self.distance) 
        except : 
            raise TypeError (
                f'Expect a float value not {type(self.distance).__name__!r}')

        freqs = self.p_.freqs_ 

        positions = np.arange(arr2d.shape[1])  * distance
            
        sites = sites or make_ids (
            positions , self.prefix , how = self.how)  
        
        if isinstance(sites, str): 
            sites =[sites] 
        if not is_iterable(sites): 
            raise TypeError("Sites collection must be an iterable" 
                            f" object. Got {type(sites).__name__!r}"
                    )
        if len(sites)!= len(positions): 
            raise TypeError (f"Sites={len(sites)} length must be consistent."
                             "  Expects positions={len(positions)}.")
            
        if tensor in {'phase', 'phs'}: 
            arr2d %=90
            
        if to_log10: 
            arr2d = arr2d if tensor in ("phase", "phs") else np.log10 (arr2d) 
            freqs = np.log10 (freqs)
            
        base_plot_kws = {
            k: v for k, v in self.__dict__.items () 
            if k not in list(self._t ) +['p_']
            }  
        
        return arr2d, freqs, positions ,sites , base_plot_kws  

    def plot_recovery(self, site = 'S00'): 
        """ visualize the restored tensor per site.
        
        Parameters 
        ------------
        site: str, int, default ="S00"
            Site/station name for 
        
        Returns
        -------- 
        ``self``: :class:`watex.view.plot.TPlot` instanciated object
            returns ``self`` for chaining methods.
            
        Examples 
        --------
        >>> from watex.view import TPlot
        >>> from watex.datasets import load_edis 
        >>> edi_data = load_edis (return_data =True, samples =7) 
        >>> plot_kws = dict( ylabel = '$Log_{10}Frequency [Hz]$', 
                    xlabel = '$Distance(m)$', 
                    cb_label = '$Log_{10}Rhoa[\Omega.m$]', 
                    fig_size =(7, 4), 
                    font_size =7. 
                    ) 
        >>> t= TPlot(**plot_kws ).fit(edi_data)
        >>> # plot recovery of site 'S01'
        >>> t.plot_recovery ('S01')
        
        """
        self.inspect 
        
        if isinstance(site, str): 
            site =[site]
        site_index = station_id(site) 
        
        site_index = site_index [0] if isinstance (
            site_index, tuple ) else site_index 
 
        if site_index  > len(self.p_.ediObjs_): 
            raise PlotError(f"Site {site!r} is out of the expected"
                            f" sites number: {len(self.p_.ediObjs_)}"
                            )
    
        ediObjs = self.p_.ediObjs_ 
        # >>> zobjs_b = restoreZ(ediObjs, buffer = buffer) # with buffer 
        zobjs = self.p_.zrestore() # with no buffer 
    
        zxy_restored = np.abs (zobjs[site_index].z [
            tuple (self._c_.get(self.component))])#[:, 0, 1]) 
        # Export the first raw object with missing Z at 
        # the dead dand in ediObjs collection
        z1 = np.abs(ediObjs[site_index].Z.z) 
        z1freq= ediObjs[site_index].Z._freq # the frequency of the first z obj 
        # get the frequency of the clean data knonw as reference frequency
        indice_reffreq = np.argmax (list (map(lambda o: len(o.Z._freq), ediObjs)))
        reffreq = ediObjs [indice_reffreq].Z._freq 
        # >>> # use the real part of component xy for the test 
        zxy = np.abs (z1[tuple (self._c_.get(self.component))])  #[:, 0, 1])  
        # fit zxy to get the missing data in the dead band 
        zfit = fittensor(refreq= reffreq, compfreq= z1freq, z=zxy)

        # not necessary, one can corrected z to get a 
        # smooth resistivity distribution 
        zcorrected = moving_average (zxy_restored)                     
        # plot the two figures 
        plt.figure(figsize =self.fig_size) #(10, 5)
        plt.loglog(reffreq, zfit, '^r', reffreq, zxy_restored, 'ok--')
        plt.loglog( reffreq, zcorrected, '1-.')
        plt.legend (['raw data', 'tensor $res_{xy}$ restored',
                     'moving-average trend line' ],loc ='best')
        plt.xlabel ('$Frequency [H_z]$') 
        plt.ylabel('$ Resistivity_{xy} [ \Omega.m]$')
        plt.title ('Recovered tensor $|Z_{xy}|$' + f" at site {site[0].upper()}")
        plt.grid (visible =True , alpha =0.8, which ='both', color ='k')
        plt.xlim (reffreq.min() -.5* reffreq.min(), 
                  reffreq.max() + .5 * reffreq.max())
        plt.tight_layout()
        
        return self 
    
    def plot_phase_tensors(
        self,
        mode ='frequency',
        stretch = (7000, 20 ),
        linedir ='ns',
        tensor='phimin',
        ellipse_dict = None,
        **kws
        ): 
        """ Plot phase tensor pseudosection and skew ellipsis 
        visualization. 
        
        Method plots the phase tensor ellipses in a pseudo section format.
        It uses `mtpy` as dependency. 
        
        Parameters 
        -----------
        mode: str, default ='frequency'
            Tempoora scale in y-axis. Can be ['frequency' | 'period']

        stretch : float or tuple (xstretch, ystretch), default=200
            Is a factor that scales the distance from one station to the next 
            to make the plot readable. It determines (x,y) aspect ratio of plot.
    
        linedir: str [ 'ns' | 'ew' ], default='ns'
            The predominant direction of profile line. It can be ['ns' | 'ew']
            where: 
               
            * 'ns' refer to North-South Line or line is closer to north-south)
            * 'ew' refer to  East-West line or line is closer to east-west
            *Default* is 'ns'
        tensor: str, default='phimin' 
            Is the tensor skew or ellipsis visualizations. The color for plot 
            style is referred accordingly. Tensor can be: 
                
                [ 'phimin' | 'phimax' | 'skew' |'skew_seg' | 'phidet' |'ellipticity' ]
           where: 
                  
                - 'phimin' -> colors by minimum phase
                - 'phimax' -> colors by maximum phase
                - 'skew' -> colors by skew
                - 'skew_seg' -> colors by skew indiscrete segments defined 
                   by the range
                - 'normalized_skew' -> colors by skew see [Booker, 2014]
                - 'normalized_skew_seg' -> colors by normalized skew in
                   discrete segments defined by the range
                - 'phidet' -> colors by determinant of the phase tensor
                - 'ellipticity' -> colors by ellipticity *default* is 'phimin'  
                
        ellipse_dict: dict, optional
            Dictionary of parameters for the phase tensor ellipses with keys:
            
            * 'size': float, default =2 , is the size of ellipse in points
            * 'colorby' : str, default='phimin' 
               Is the color for plot style referring either to  tensor, 
               skew or ellipsis visualizations. It can be all the `tensor`
               parameter values. see `tensor` parameter values. 
               [ 'phimin' | 'phimax' | 'skew' |'skew_seg' | 'phidet' |'ellipticity' ]
        
            * 'range' : tuple (min, max, step), default='colorby'
               Need to input at least the min and max  and if using 
               'skew_seg' to plot discrete values input step as well
               
            * 'cmap' : [ 'mt_yl2rd' | 'mt_bl2yl2rd' |'mt_wh2bl' | 'mt_rd2bl' |
                        'mt_bl2wh2rd' | 'mt_seg_bl2wh2rd' |'mt_rd2gr2bl' ]

                     - 'mt_yl2rd' -> yellow to red
                     - 'mt_bl2yl2rd' -> blue to yellow to red
                     - 'mt_wh2bl' -> white to blue
                     - 'mt_rd2bl' -> red to blue
                     - 'mt_bl2wh2rd' -> blue to white to red
                     - 'mt_bl2gr2rd' -> blue to green to red
                     - 'mt_rd2gr2bl' -> red to green to blue
                     - 'mt_seg_bl2wh2rd' -> discrete blue to white to red
        kws: dict 
            Additional keywords arguments passed from |MTpy| pseudosection 
            phase tensor class: :class:`~.PlotPhaseTensorPseudoSection` 

        See Also
        ----------
        mtpy.imaging.phase_tensor_pseudosection.PlotPhaseTensorPseudoSection: 
            PlotPhase pseudo section tensor from |MTpy| package. 
        watex.utils.plot_skew: 
            Phase sensitive skew visualization. 
        
        Examples
        ---------
        >>> import watex as wx 
        >>> edi_data = wx.fetch_data ('edis', key='edi', return_data =True , samples =17 ) 
        >>> tplot = wx.TPlot ().fit(edi_data ) 
        >>> tplot.plot_phase_tensors (tensor ='skew')
        
        """
        extra =("Phase tensor plots or skew ellipsis visualization"
                " uses 'mtpy' as dependency. Alternatively, you may"
                " also use the phase sensitive 'skew' visualization"
                " of plot utilities if plot  only refers to 'skew'."
                )
        import_optional_dependency ('mtpy' , extra = extra )
        from mtpy.imaging.phase_tensor_pseudosection import (
            PlotPhaseTensorPseudoSection ) 
        
        self.inspect 
        
        zobjs = [edi_obj.Z for edi_obj in self.p_.ediObjs_]
        
        elrange =  [-7, 7] if 'skew' in str(tensor).lower()  else [0, 90 ]  
        ellipse_dict = ellipse_dict or  {
            'ellipse_colorby':tensor,
            'ellipse_range':elrange,  # Color limits
            'ellip_size': 2, 
            'ellipse_cmap':'mt_bl2wh2rd'
        } 
        # skew_seg need to provide
        # 3 numbers, the 3rd indicates
        # interval, e.g. [-12,12,3]
        #from contextlib import suppress 
        # suppress as possible the external 
        #lib resources
        #with suppress (Exception): 
        ptsection = PlotPhaseTensorPseudoSection(
                        fn_list = self.p_.edifiles,
                        z_object_list = zobjs, 
                        fig_size = self.fig_size, 
                        tscale = mode, 
                        plot_num = self.fig_num, 
                        plot_title = self.fig_title, 
                        xlimits = self.xlim, 
                        ylimits = self.ylim,
                        linedir= linedir,  
                        stretch= stretch, 
                        station_id=(0, len(self.p_.ediObjs_)), 
                        font_size=self.font_size ,
                        lw=self.lw,
                        **ellipse_dict, 
                        **kws,
            )

        ptsection.save_figure(save_fn =self.savefig, fig_dpi=self.fig_dpi
                              ) if self.savefig else  ptsection.plot()

        return self 
    
    def plotSkew (
        self , 
        method ='Bahr', 
        view ='skew', 
        mode=None,
        threshold_line=None, 
        show_average_sensistivity=True, 
        suppress_outliers =True, 
        **plot_kws 
        ): 
        """ Plot phase sensistive skew visualization
        
        'Skew' is also knwown as the conventional asymmetry parameter 
        based on the Z magnitude. 
        
        Mosly, the :term:`EM` signal is influenced by several factors such 
        as the dimensionality of the propagation medium and the physical 
        anomalies, which can distort theEM field both locally and regionally. 
        The distortion of Z was determined from the quantification of its 
        asymmetry and the deviation from the conditions that define its 
        dimensionality. The parameters used for this purpose are all rotational 
        invariant because the Z components involved in its definition are
        independent of the orientation system used. The conventional asymmetry
        parameter based on the Z magnitude is the skew defined by Swift (1967)
        [1]_ and Bahr (1991) [2]_.

        Parameters 
        -----------
        method: str, default='Bahr': 
            Kind of correction. Can be:
                
            - ``swift`` for the remove distorsion proposed by Swift in 1967. 
              The value close to 0. assume the 1D and 2D structures, and 3D 
              otherwise.  However, In general case, the  electrical structure 
              of :math:`\eta < 0.4` can be treated as a 2D medium.
            - ``bahr`` for the remove distorsion proposed  by Bahr in 1991. 
              The latter threshold is set to 0.3. Above this value the 
              structures is 3D.
              
        view: str, default='skew'
           phase sensistive visualization. Can be rotational invariant 
           ``invariant``. In fact, setting to ``mu`` or ``invariant`` does 
           not change any interpretation when since the  distortion of Z 
           are all rotational invariant whether using the ``Bahr`` or ``swift``
           methods. 
           
        mode:str, optional 
           X-axis coordinates for visualisation. plot either ``'frequency'`` or
           ``'periods'``.  The default is ``'frequency'`` 
           
        threshold_line: float, optional
           Visualize th threshold line. Can be ['bahr', 'swift', 'both']:
               
           - Note that when method is set to ``swift``, the value close 
             to close to :math:`0.` assume the 1D and 2D structures 
             (:math:`\eta <0.4`),  and 3D otherwise( :math:`\eta >0.4`). 
             The threshold line for ``swift`` is set to :math:`0.4`. 
             
           - when method is set to ``Bahr``, :math:`\eta > 0.3``  is 3D 
             structures, between :math:`[0.1 - 0.3]` assumes modified 3D/2D 
             structures whereas :math:`<0.1` 1D, 2D or distorted 2D. 
        show_average_sensistivity: bool, default=True 
           Display the averaged value of skew data at all -frequencies. 
           Value can help a dimensionality interpretation purposes. 
           
        suppress_outliers: bool, default=True
           Remove the outliers in the data if exists. It uses the 
           Inter Quartile Range (``IQR``) approach. See the documentation 
           of :func:`watex.utils.remove_outliers`. This is useful for clear 
           interpretation using the skew threshold value. 
          
        See Also
        ---------
        watex.methods.Processing.skew: 
            
            For mathematical skew `Bahr` and `Swift` concept formulations. 
        watex.utils.plot_skew: 
            For phase sensistive skew visualization - naive plot.
  
        Examples
        --------
        >>> import watex 
        >>> test_data = watex.fetch_data ('edis', samples =37, return_data =True )
        >>> watex.TPlot(fig_size =(10,  4), marker ='x').fit(
            test_data).plotSkew(method ='swift', threshold_line=True)
        
        References 
        -----------
        .. [1]  Swift, C., 1967. A magnetotelluric investigation of an 
                electrical conductivity  anomaly in the southwestern United 
                States. Ph.D. Thesis, MIT Press. Cambridge.
        .. [2] Bahr, K., 1991. Geological noise in magnetotelluric data: a 
               classification of distortion types. Physics of the Earth and 
               Planetary Interiors 66 (1–2), 24–38.
        """
        self.inspect 
        
        view = str(view).lower() 
        for ix in ('inv', 'rot', 'mu'): 
            if view.find(ix)>=0: 
                view ='mu' 
                break 
            
        view='skew' if view=='none' else view 
        assert view in {"skew", 'mu'}, ("expect 'skew' or 'rotational invariant'"
                                        f" plot, got {view!r}")
        
        if 'period' in str(mode).lower(): 
            mode ='period'

        skew, mu =self.p_.skew(
            method = method, suppress_outliers = suppress_outliers
            )
        freqs =  1/ self.p_.freqs_ if mode =='period' else self.p_.freqs_ 
        ymat = skew if view =='skew' else mu 
        
        fig, ax = plt.subplots(figsize = self.fig_size )

        #---manage threshold hline ------
        thr_code = {"bahr": [1] , "swift":[ 2] , 'both':[1, 2] }
        
        if str(threshold_line).lower()=='true': 
            threshold_line = str(method).lower() 
            
        if threshold_line is not None: 
            if str(threshold_line).lower() in ("*", "both" ): 
                threshold_line = 'both'
                
        ct = thr_code.get(str(threshold_line).lower(), None ) 
        
        for i in range (skew.shape[1]): 
            ax.scatter ( freqs, reshape (ymat[:, i]),
                        marker = plot_kws.get ('marker', None) or self.marker, 
                        cmap = plot_kws.get('cmap', None) or self.cmap, 
                        alpha = plot_kws.get('alpha', None) or self.alpha, 
                        s = plot_kws.get('s', None) or self.s , 
                        **plot_kws 
                        )
        if ct: 
            for m in ct: 
                plt.axhline(y=0.4 if m==2 else 0.3 , color="k" if m==1 else "g",
                            linestyle="-",
                            label=f'threshold: $\mu={0.4 if m==2 else 0.3}$'
                            )
                ax.legend() 

        # see phase sensitive trend 
        if show_average_sensistivity: 
            plt.text(x= np.nanmin(freqs) , y= np.nanmax(ymat), 
                     s="aver.-{}:{}={}".format(view, str(method).capitalize(), 
                    np.around (np.average(ymat[ ~np.isnan(ymat)]), 3)),  
                    fontdict= dict (style ='italic',  bbox =dict(
                         boxstyle='round',facecolor ='#CED9EF'))
                     ) 
        plt.legend()
        ax.set_xscale('log')
        ax.set_xlabel('Period ($s$)' if mode=='period' 
                      else 'Frequency ($H_z$)' or self.xlabel )
        ax.set_ylabel(f"{'Skew' if view =='skew' else 'Rot.Invariant'}" + "($\mu$)"
                      or self.ylabel )

        plt.xlim ([ freqs.min() , freqs.max()] or self.xlim )
        
        plt.xlim() 

        if self.savefig is not  None: 
            plt.savefig (self.savefig, dpi = self.fig_dpi)
            
        plt.close () if self.savefig is not None else plt.show() 
        
        return self 
    
    def _check_component_validity (self, tensor, components ): 
        """Retrieve resistiviy, phase or impedance tensors from 
        EDI objets if component exists. 
        
        Parameters 
        -----------
        tensor: str, 
          Name of tensor. Could be ['resistivity'| 'phase'|'z']
        components: str, list, 
          Name of components. Could be ['xx', 'xy', 'yx', 'yy']
        
        Returns
        --------
        rp: list of valid 2D dimensional tensors and ``None`` if 
          no valid tensors are found. 
        
        """
        rp =[] 
        tensor =str(tensor) 
        components = is_iterable(components, exclude_string =True,
                                transform =True, parse_string =True )
        for c in components : 
            try: 
                mat2d = self.p_.make2d (f'{tensor+c}')
            except :continue 
            else: rp.append(mat2d )
            
        return rp if len(rp)!=0 else None 
    
    def plot_rhoa(
        self, 
        mode ='TE', 
        onto ='period', 
        site =None, 
        seed = None, 
        how ='py', 
        show_site=True,
        survey= None, 
        style=None, 
        errorbar=True, 
        suppress_outliers=False, 
        **kws
        ): 
        """ Plot apparent resistivity and phase curves 
        
        Parameters 
        ----------
        mode: str, default='TE', 
          Electromagnetic mode. Can be ['TM' |'both']. If ``both``, 
          components `xy` and `yx` are expected in the data. 
          
        onto: str, default='period'
          Visualization on axis labell. can be ``'frequency'``. 
          
        site: int,str, optional 
          index of name of the site to plot. `site` must be composed of 
          a position number. For instance ``'S13'``. If not provided, 
          a random station is selected instead. 
          
        seed : int, optional 
           If site is not provided, seed fetches randomly a site. To fetch 
           the same sime everytimes, it is better to set the seed value. 
           
        how: str, default='py'
          The way the site is fetched for plot. For instance, in Python 
          indexing (default), the site is numbered from 0. For instance 
          'site05' will fetch the data at index 4. If this positioning 
          is not wished, set to 'None'.
        show_site:bool, default=True, 
          Display the number of site. 
        survey: str, optional 
          Method used for the survey. e.g., 'AMT' for |AMT|. 
         
        style:str, default='default'
          Matplotlib style. 
          
        errorbar: bool, default=True 
          display the error bar.  
          
        kws: dict, 
          Addfitional keywords arguments passed to 
          Matplotlib.Axes.Scatter plots. 
         
        Examples
        ---------
        >>> import watex as wx 
        >>> edi_data = wx.fetch_data ('edis', return_data =True, samples =27)
        >>> wx.TPlot(show_grid=True).fit(edi_data).plot_rhoa (
            seed =52, mode ='*')
        """
        self.inspect 
        
        m=_validate_name_in(mode,  ('*', 'both', 'tetm'), expect_name='*')
        
        if m!='*':
            m= _validate_name_in(mode, defaults = 'tm transverse-magnetic',
                                     expect_name ='tm' )
        if not m: 
            m='te' 

        onto = _validate_name_in(onto, deep =True, defaults='periods', 
                                 expect_name='period')

        cpm = {'te': ["xy"] , 'tm': ["yx"], '*': ('xy', 'yx') }
        
        components = cpm.get(m)
        
        res, phs, site, *s= self._validate_correction (
                             components = components, 
                             errorbar = errorbar , 
                             how = how, 
                             seed = seed , 
                             site = site , 
                             style =style , 
                             )  
        s,  res_err, phs_err  = s 
    
        fig = plt.figure(self.fig_num , figsize= self.fig_size,
                         dpi = self.fig_dpi , # layout='constrained'
                         )

        gs = GridSpec(3, 1, figure = fig ) 
        
        ax1 = fig.add_subplot (gs[:-1, 0 ])
        ax2 = fig.add_subplot(gs [-1, 0 ], sharex = ax1 )
        plt.setp(ax1.get_xticklabels(), visible=False)
        
        survey= survey or self.p_.survey_name 
        if not survey: survey=''
        
        colors = [ '#069AF3', '#DC143C']
    
        #==plotlog10 --------
        res= [ np.log10 (r) for r in res] 
        # the complete frequency 
        fp = self.p_.freqs_
        
        fp =  1/ fp if onto =='period' else fp 
        
        fp =  np.log10 ( fp) 
        
        if suppress_outliers: 
            res = remove_outliers(res, fill_value=np.nan) 
            phs = remove_outliers(phs, fill_value=np.nan) 
            if errorbar: 
                res_err = remove_outliers(
                    res_err, fill_value=np.nan) 
                phs_err = remove_outliers(
                    phs_err, fill_value=np.nan) 
                
        min_y =  np.nanmin(res[0][:, site])
        
        # add error bar data to main 
        data = [res, phs ] 
        data +=  [ res_err ,  phs_err ] if errorbar else []
        
        for i, sloop in enumerate (zip (* data )) : 
            r, p, *sl = sloop 
            
            if len(sl) !=0 : 
                e, ep = sl  # mean errorbar is set to True 
            
            y =  reshape (r[:, site])
            if errorbar: 
                plot_errorbar (ax1 , 
                               fp, 
                               y,  
                               y_err = reshape (e[:, site]),
                               )
            ax1.scatter (fp  , y, 
                          marker =self.marker, 
                          color =colors [i],
                          edgecolors='k', 
                          label = fr'{survey}$\rho_a${components[i]}',
                          **kws 
                          ) 
            if errorbar: 
                plot_errorbar (ax2 , 
                               fp, 
                               reshape (p[:, site]),
                               y_err = reshape (ep[:, site]),
                               )
            ax2.scatter( fp, 
                        reshape (p[:, site]),
                        marker =self.marker, 
                        color =colors [i] ,
                        edgecolors='k', 
                        label = f'{survey}$\phi${components[i]}',
                        **kws
                        ) 
            min_y = np.nanmin (y) if np.nanmin (
                y) < min_y else min_y 
            try: 
                ax1.legend(ncols = len(res)) 
                ax2.legend(ncols = len(phs)) 
            except: 
                # For consistency in the case matplotlib  is < 3.3. 
                ax1.legend() 
                ax2.legend() 
                
        if show_site:
            ax1.text (np.nanmin(fp),
                      min_y,
                      f'site {s}', 
                      fontdict= dict (style ='italic',  bbox =dict(
                           boxstyle='round',facecolor ='#CED9EF'), 
                          alpha = 0.5 )
                      )
        
        ax2.set_ylim ([0, 90 ])
        xlabel = self.xlabel or ( 'Log$_{10}$Period($s$)' if onto=='period' 
                                 else 'Frequency ($H_z$)') 
        
        ax2.set_xlabel(xlabel ) 
        ax1.set_ylabel(self.ylabel or r'Log$_{10}\rho_a$($\Omega$.m)') 
 
        ax2.set_ylabel('$\phi$($\degree$)')
        
        if self.show_grid :
            for ax in (ax1, ax2 ): 
                ax.grid (visible =True , alpha =self.galpha,
                         which =self.gwhich, color =self.gc)
          
            
        if self.savefig is not  None: 
            plt.savefig (self.savefig, dpi = self.fig_dpi)
            
        plt.close () if self.savefig is not None else plt.show() 
        
        return self 
    
    def _validate_correction (
        self, 
        components , 
        errorbar , 
        seed , 
        site , 
        how , 
        style , 
        ): 
        """Isolated part to validate the :meth:`plot_corrections` and 
        :meth:`plot_rhoa` arguments. 
        
        Parameters
        ----------
        
        components: str ,
           could be 'xx', 'xy', 'yx' or 'yy' 

        onto: str, default='period'
          Visualization on axis labell. can be ``'frequency'``.
          
        site: int,str, optional 
          index of name of the site to plot. `site` must be composed of 
          a position number. For instance ``'S13'``. If not provided, 
          a random station is selected instead. 
          
        seed : int, optional 
           Get the same site if site is not provided. `seed` fetches 
           a random number of site. 
           
        how: str, default='py'
          The way the site is fetched for plot. For instance, in Python 
          indexing (default), the site is numbered from 0. For instance 
          'site05' will fetch the data at index 4. If this positioning 
          is not wished, set to 'None'.
        
        style:str, default='default'
          Matplotlib style. 
          
        errorbar: bool, default=True 
          display the error bar.
          
        Returns 
        --------
        ( fp, res, phs, site, s ,  res_err , phs_err) : Tuple 
        
          - fp: frequency array 
          - res:  resistivity tensor collected at a specific components 
          - phs: phase tensor collected at a specific component 
          - site: The site number 
          - s : position of the site 
          - res_err: error in resistivity at a specific component 
          - phs_err: error in phase at a specific components. 
          
        """ 
        
        res = self._check_component_validity('res', components)
        phs = self._check_component_validity('phase', components)
        
        res_err , phs_err =[], []
        if errorbar: 
            res_err = self._check_component_validity(
                'res_err', components)
            phs_err = self._check_component_validity(
                'phase_err', components)
  
        
        terror =("{0!r} does not contain component {}. Provide the"
                 " right component of the valid tensor.")
        if res is None: 
            raise EMError(terror.format('resistivity', components))
        if phs is None: 
            raise EMError(terror.format('phase', components))

        if seed: 
            seed = _assert_all_types(seed , int, float, objname ='Seed')
            np.random.seed (seed ) 
           
        if site is None:
            site = np.random.choice (range (res[0].shape[1])) 
           
        s= copy.deepcopy(site)
        site =re.search ('\d+', str(site), flags=re.IGNORECASE).group() 
        try: 
           site= int(site)
        except TypeError: 
            raise TypeError ("Missing position number. Station must be "
                             f"prefixed with position, e.g. 'S7', got {s!r}")
        
        site = abs (site) + 1 if how !='py' else site 
        
        if site > res[0].shape [1] : 
            raise ValueError (
                f"Site position {site} is out of the range. The total"
                f" number of sites/stations ={res[0].shape [1]}")
            
        try: 
            plt.style.use ( style or 'default')
        except : 
            warnings.warn(
                f"{style} is not available. Use `plt.style.available`"
                " to get the list of available styles.")
            plt.style.use ('default')

       
        return res, phs, site, s ,  res_err , phs_err 
 
 
    def plot_corrections(
        self, 
        fltr='ss',
        ss_fx =None, 
        ss_fy=None, 
        r=1000., 
        nfreq=21,
        skipfreq=5, 
        tol=.12,
        rotate=0., 
        distortion=None, 
        distortion_err =None, 
        mode ='TE', 
        onto ='period', 
        site =None, 
        seed = None, 
        how ='py', 
        show_site=True,
        survey= None, 
        style=None, 
        errorbar=True, 
        **kws
        ): 
        """Plot apparent resistivity/phase curves and corrections.  
        
        Parameters 
        ----------
        fltr: str , default='ss'
           Type of filter to apply. ``ss`` is used to remove the static 
           shift using spatial median filter. Whereas ``dist`` is for 
           distorsion removal. Note that `distortion` might be provided 
           otherwise an error raises. 
           
        distortion_tensor: np.ndarray(2, 2, dtype=real) 
           Real distortion tensor as a 2x2
   
        error: np.ndarray(2, 2, dtype=real), Optional 
          Propagation of errors/uncertainties included
          
        ss_fx: float, Optional  
           static shift factor to be applied to x components
           (ie z[:, 0, :]).  This is assumed to be in resistivity scale. 
           If None should be automatically computed using  the 
           spatial median filter. 
           
        ss_fy: float, optional 
           static shift factor to be applied to y components 
           (ie z[:, 1, :]).  This is assumed to be in resistivity scale. If
           ``None`` , should be computed using the spatial filter median.
           
        r: float, default=1000. 
           radius to look for nearby stations, in meters.
 
        nfreq: int, default=21 
           number of frequencies calculate the median static shift.  
           This is assuming the first frequency is the highest frequency.  
           Cause usually highest frequencies are sampling a 1D earth.  
    
        skipfreq** : int, default=5 
           number of frequencies to skip from the highest frequency.  
           Sometimes the highest frequencies are not reliable due to noise 
           or low signal in the :term:`AMT` deadband.  This allows you to 
           skip those frequencies.
     
        tol: float, default=0.12
           Tolerance on the median static shift correction.  If the data is 
           noisy the correction factor can be biased away from 1.  Therefore 
           the shift_tol is used to stop that bias.  If 
           ``1-tol < correction < 1+tol`` then the correction factor is set 
           to ``1``
           
        rotate: float, default=0.  
            Rotate Z array by angle alpha in degrees.  All angles are referenced
            to geographic North, positive in clockwise direction.
            (Mathematically negative!).
            In non-rotated state, X refs to North and Y to East direction. 
            
        mode: str, default='TE', 
          Electromagnetic mode. Can be ['TM' |'both']. If ``both``, 
          components `xy` and `yx` are expected in the data. 
          
        onto: str, default='period'
          Visualization on axis labell. can be ``'frequency'``.
          
        site: int,str, optional 
          index of name of the site to plot. `site` must be composed of 
          a position number. For instance ``'S13'``. If not provided, 
          a random station is selected instead. 
          
        seed : int, optional 
           Get the same site if site is not provided. `seed` fetches 
           a random number of site. T
           
        how: str, default='py'
          The way the site is fetched for plot. For instance, in Python 
          indexing (default), the site is numbered from 0. For instance 
          'site05' will fetch the data at index 4. If this positioning 
          is not wished, set to 'None'.
          
        show_site:bool, default=True, 
          Display the number of site. 
          
        survey: str, optional 
          Method used for the survey. e.g., 'AMT' for |AMT|. 
         
        style:str, default='default'
          Matplotlib style. 
          
        errorbar: bool, default=True 
          display the error bar.  
          
        kws: dict, 
          Addfitional keywords arguments passed to 
          Matplotlib.Axes.Scatter plots. 
         
        Examples
        ---------
        >>> import numpy as np 
        >>> import watex as wx 
        >>> edi_data = wx.fetch_data ('edis', return_data =True, samples =27)
        >>> wx.TPlot(show_grid=True).fit(edi_data).plot_corrections (
            seed =52, )
        >>> distortion = np.array([[1.1 , 0.6 ],[0.23, 1.9 ]])
        >>> wx.TPlot(show_grid=True).fit(edi_data).plot_corrections (
             seed =52, mode ='tm', fltr ='dist', distortion =distortion 
             )
        """
        self.inspect 
    
        m=_validate_name_in(mode,  'tm transverse-magnetic', expect_name='tm')
        if not m: 
            m='te' 
        onto = _validate_name_in(onto, deep =True, defaults='periods', 
                                 expect_name='period')

        cpm = {'te': ["xy"] , 'tm': ["yx"]}
       
        components = cpm.get(m)
        res, phs, site, *s= self._validate_correction (
                             components = components, 
                             errorbar = errorbar , 
                             how = how, 
                             seed = seed , 
                             site = site , 
                             style =style , 
                             )  
        s,  res_err, phs_err  = s 
        
        # Assert filters 
        mc = _validate_name_in(fltr, defaults =('static shift', 'ss', '1'), 
                               expect_name= 'ss')
        if mc!='ss': 
            mc = _validate_name_in(fltr, defaults=('distortion', 'dist', '2'), 
                                   expect_name ='dist')
            if not mc: 
                raise ValueError(f"Wrong filter {fltr}. Expect `ss` or `dist`"
                                 " for static shift or distortion plot.")
           
            if mc and distortion is None: 
                raise TypeError("Distorsion cannot be None!")
        
        # -> compute the corrected values 
        zo = ZC().fit(self.p_.ediObjs_)
        
        if mc =='ss': 
            zc = zo.remove_static_shift (
                ss_fx = ss_fx , 
                ss_fy = ss_fx, 
                nfreq = nfreq ,         
                r=r, 
                skipfreq=skipfreq , 
                tol=tol, 
                rotate = rotate, 
                )
        if mc =='dist': 
            zc = zo.remove_distortion (
                distortion , 
                error = distortion_err 
                )
            
        zc_res = [ z.resistivity[tuple (self._c_.get(components[0])) ] 
                  for z in zc ] 
        zc_res = [ np.log10(r) for r in zc_res ] # convert to log10 res 
        # --> phase 
        zc_phase = [ z.phase[tuple (self._c_.get(components[0])) ] 
                  for z in zc ] 
        # mofulo the phase to be 0 and 90 degree 
        zc_phase = [ np.abs (p)%90  for p in zc_phase ] 
        
        # ----------------------end ---------------------------------
        fig = plt.figure(self.fig_num , figsize= self.fig_size,
                         dpi = self.fig_dpi , # layout='constrained'
                         )

        gs = GridSpec(3, 1, figure = fig ) 
        
        ax1 = fig.add_subplot (gs[:-1, 0 ])
        ax2 = fig.add_subplot(gs [-1, 0 ], sharex = ax1 )
        plt.setp(ax1.get_xticklabels(), visible=False)
        
        survey= survey or self.p_.survey_name 
        if not survey: survey=''
        
        colors = [ '#069AF3', '#DC143C']
        
        #==plotlog10 --------
        res= [ np.log10 (r) for r in res] 
        # to use frequency for individual site rather than 
        # the complete frequency 
        fp = self.p_.ediObjs_[site].Z._freq
        fp =  1/ fp if onto =='period' else fp 
        
        fp =  np.log10 ( fp) 
        
        min_y =  np.nanmin(res[0][:, site])
        
        # add error bar data to main 
        data = [res, phs ] 
        data +=  [ res_err ,  phs_err ] if errorbar else []
        
        for i, sloop in enumerate (zip (* data )) : 
            r, p, *sl = sloop 
            
            if len(sl) !=0 : 
                e, ep = sl  # mean errorbar is set to True 
                
            y =  reshape (r[:, site])
            yc =zc_res [site]
            if errorbar: 
                plot_errorbar (ax1 , 
                               fp, 
                               y,  
                               y_err = reshape (e[:, site]),
                               )
            ax1.scatter (fp  , y, 
                          marker =self.marker, 
                          color =colors [i],
                          edgecolors='k', 
                          label = fr'{survey}$\rho_a${components[i]}',
                          **kws 
                          ) 
            # res_corr 
            ax1.scatter (fp  , yc, 
                          marker ='*', 
                          color="#FF00FF",
                          edgecolors='k', 
                          label = fr'{survey}$\rho_a${components[i]} {mc}',
                          **kws 
                          ) 
            
            if errorbar: 
                plot_errorbar (ax2 , 
                               fp, reshape (p[:, site]),
                               y_err = reshape (ep[:, site]), 
                               )
            
            ax2.scatter( fp, 
                        reshape (p[:, site]),
                        marker =self.marker, 
                        color =colors [i] ,
                        edgecolors='k', 
                        label = f'{survey}$\phi${components[i]}',
                        **kws
                        ) 
            # ----phase_cor 
            ax2.scatter( fp, 
                        zc_phase [site],
                        marker ='*', 
                        color="#FF00FF" ,
                        edgecolors='k', 
                        label = f'{survey}$\phi${components[i]} {mc}',
                        **kws
                        ) 
            
            min_y = np.nanmin (y) if np.nanmin (
                y) < min_y else min_y 
            try: 
                ax1.legend(ncols = len(res)) 
                ax2.legend(ncols = len(phs)) 
            except: 
                # For consistency in the case matplotlib  is < 3.3. 
                ax1.legend() 
                ax2.legend() 
                
        if show_site:
            ax1.text (np.nanmin(fp),
                      min_y,
                      f'site {s}', 
                      fontdict= dict (style ='italic',  bbox =dict(
                           boxstyle='round',facecolor ='#CED9EF'), 
                          alpha = 0.5 )
                      )
        
        ax2.set_ylim ([0, 90 ])
        xlabel = self.xlabel or ( 'Log$_{10}$Period($s$)' if onto=='period' 
                                 else 'Frequency ($H_z$)') 
        
        ax2.set_xlabel(xlabel ) 
        ax1.set_ylabel(self.ylabel or r'Log$_{10}\rho_a$($\Omega$.m)') 
 
        ax2.set_ylabel('$\phi$($\degree$)')
        
        if self.show_grid :
            for ax in (ax1, ax2 ): 
                ax.grid (visible =True , alpha =self.galpha,
                         which =self.gwhich, color =self.gc)
          
            
        if self.savefig is not  None: 
            plt.savefig (self.savefig, dpi = self.fig_dpi)
            
        plt.close () if self.savefig is not None else plt.show() 
        
        return self 

    def __repr__(self): 
        """ Represents the output class format """
        outm = ( '<{!r}:' + ', '.join(
            [f"{k}={getattr(self, k)!r}" for k in self._t]) + '>' 
            ) 
        return  outm.format(self.__class__.__name__)
    
    
class ExPlot (BasePlot): 
    
    msg = ("{expobj.__class__.__name__} instance is not"
           " fitted yet. Call 'fit' with appropriate"
           " arguments before using this method."
           )
    
    def __init__(
        self, 
        tname:str = None, 
        inplace:bool = False, 
        **kws
        ):
        super().__init__(**kws)
        
        self.tname= tname
        self.inplace= inplace 
        self.data= None 
        self.target_= None
        self.y_= None 
        self.xname_=None 
        self.yname_=None 
        

    @property 
    def inspect(self): 
        """ Inspect data and trigger plot after checking the data entry. 
        Raises `NotFittedError` if `ExPlot` is not fitted yet."""
        if self.data is None: 
            raise NotFittedError(self.msg.format(
                expobj=self)
            )
        return 1 
     
    def save (self, fig): 
        """ savefigure if figure properties are given. """
        if self.savefig is not None: 
            fig.savefig (self.savefig, dpi = self.fig_dpi , 
                         bbox_inches = 'tight'
                         )
        plt.show() if self.savefig is None else plt.close () 
        
    def fit(self, data: str |DataFrame,  **fit_params )->'ExPlot': 
        """ Fit data and populate the arguments for plotting purposes. 
        
        There is no conventional procedure for checking if a method is fitted. 
        However, an class that is not fitted should raise 
        :class:`exceptions.NotFittedError` when a method is called.
        
        Parameters
        ------------
        data: Filepath or Dataframe or shape (M, N) from 
            :class:`pandas.DataFrame`. Dataframe containing samples M  
            and features N

        fit_params: dict 
            Additional keywords arguments for reading the data is given as 
            a path-like object passed from 
            :func:watex.utils.coreutils._is_readable`
           
        Return
        -------
        ``self``: `Plot` instance 
            returns ``self`` for easy method chaining.
             
        """
        if data is not None: 
            self.data = _is_readable(data, **fit_params)
        if self.tname is not None:
            self.target_, self.data  = exporttarget(
                self.data , self.tname, self.inplace ) 
            self.y_ = reshape (self.target_.values ) # for consistency 
            
        return self 

    def plotparallelcoords (
            self, 
            classes: List [Any] = None, 
            pkg = 'pd',
            rxlabel: int =45 , 
            **kwd
            )->'ExPlot': 
        """ Use parallel coordinates in multivariates for clustering 
        visualization  
        
        Parameters 
        ------------
        classes: list, default: None
            a list of class names for the legend The class labels for each 
            class in y, ordered by sorted class index. These names act as a 
            label encoder for the legend, identifying integer classes or 
            renaming string labels. If omitted, the class labels will be taken 
            from the unique values in y.
            
            Note that the length of this list must match the number of unique 
            values in y, otherwise an exception is raised.
        pkg: str, Optional, 
            kind or library to use for visualization. can be ['sns'|'pd'] for 
            'yellowbrick' or 'pandas' respectively. *default* is ``pd``.
            
        rxlabel: int, default is ``45`` 
            rotate the xlabel when using pkg is set to ``pd``. 
            
        kws: dict, 
            Additional keywords arguments are passed down to 
            :class:`yellowbrick.ParallelCoordinates` and
            :func:`pandas.plotting.parallel_coordinates`
            
        Returns 
        --------
        ``self``: `ExPlot` instance and returns ``self`` for easy method chaining. 
        
        Examples
        --------
        >>> from watex.datasets import fetch_data 
        >>> from watex.view import ExPlot 
        >>> data =fetch_data('original data').get('data=dfy1')
        >>> p = ExPlot (tname ='flow').fit(data)
        >>> p.plotparallelcoords(pkg='yb')
        ... <'ExPlot':xname=None, yname=None , tname='flow'>
         
        """
        self.inspect 
        
        if str(pkg) in ('yellowbrick', 'yb'): 
            pkg ='yb'
        else: pkg ='pd' 
        
        fig, ax = plt.subplots(figsize = self.fig_size )
        
        df = self.data .copy() 
        df = selectfeatures(df, include ='number')
        
        if pkg =='yb': 
            import_optional_dependency('yellowbrick', (
                "Cannot plot 'parallelcoordinates' with missing"
                " 'yellowbrick'package.")
                )
            pc =ParallelCoordinates(ax =ax , 
                                    features = df.columns, 
                                    classes =classes , 
                                    **kwd 
                                    )
            pc.fit(df, y = None or self.y_)
            pc.transform (df)
            label_format = '{:.0f}'
            
            ticks_loc = list(ax.get_xticks())
            ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            ax.set_xticklabels([label_format.format(x) for x in ticks_loc], 
                                rotation =rxlabel)
            pc.show()
            
        elif pkg =='pd': 
            if self.tname is not None: 
                if self.tname not in df.columns :
                    df[self.tname ]= self.y_ 
            parallel_coordinates(df, class_column= self.tname, 
                                 ax= ax, **kwd
                                 )
        self.save (fig)
        
        return self 
    
    def plotradviz (self,
                    classes: List [Any] = None, 
                    pkg:str = 'pd',
                    **kwd
                    )-> 'ExPlot': 
        """ plot each sample on circle or square, with features on the  
        circonference to vizualize separately between target. 
        
        Values are normalized and each figure has a spring that pulls samples 
        to it based on the value. 
        
        Parameters 
        ------------
        
        classes: list of int | float, [categorized classes] 
            must be a value in the target.  Specified classes must match 
            the number of unique values in target. otherwise an error occurs.
            the default behaviour  i.e. ``None`` detect all classes in unique  
            value in the target. 
            
        pkg: str, Optional, 
            kind or library to use for visualization. can be ['sns'|'pd']  for 
             'yellowbrick' or 'pandas' respectively. default is ``pd``.   
        kws: dict, 
            Additional keywords arguments are passed down to 
            :class:`yellowbrick.RadViZ` and :func:`pandas.plotting.radviz`
            
        Returns 
        -----------
        ``self``: `ExPlot` instance and returns ``self`` for easy method chaining. 
        
        Examples 
        ---------
        (1)-> using yellowbrick RadViz 
        
        >>> from watex.datasets import fetch_data 
        >>> from watex.view import ExPlot 
        >>> data0 = fetch_data('bagoue original').get('data=dfy1')
        >>> p = ExPlot(tname ='flow').fit(data0)
        >>> p.plotradviz(classes= [0, 1, 2, 3] ) # can set to None 
        
        (2) -> Using pandas radviz plot 
        
        >>> # use pandas with 
        >>> data2 = fetch_data('bagoue original').get('data=dfy2')
        >>> p = ExPlot(tname ='flow').fit(data2)
        >>> p.plotradviz(classes= None, pkg='pd' )
        ... <'ExPlot':xname=None, yname=None , tname='flow'>
        """
        self.inspect 
        
        fig, ax = plt.subplots(figsize = self.fig_size )
        df = self.data .copy() 
        
        if str(pkg) in ('yellowbrick', 'yb'): 
            pkg ='yb'
        else: pkg ='pd' 
        
        if classes is None :  
            if self.tname is None: 
                raise TypeError (
                    "target name is missing. Can not fetch the target."
                    " Provide the target name instead."
                    )
            classes = list(np.unique (self.y_))
            
        df = selectfeatures(df, include ='number')

        if pkg =='yb': 
            rv = RadViz( ax = ax , 
                        classes = classes , 
                        features = df.columns,
                        **kwd 
                        )
            rv.fit(df, y =None or self.y_ )
            _ = rv.transform(df )
            
            rv.show()
            
        elif pkg =='pd': 
            if (self.tname is not None)  and (self.y_ is not None) :
                df [self.tname] = self.y_ 
            radviz (df , class_column= self.tname , ax = ax, 
                    **kwd 
                    )
            
        self.save (fig)
        
        return self 
    
        
    def plotpairwisecomparison (
            self ,  
            corr:str = 'pearson', 
            pkg:str ='sns', 
            **kws
            )-> 'ExPlot': 
        """ Create pairwise comparizons between features. 
        
        Plots shows a ['pearson'|'spearman'|'covariance'] correlation. 
        
        Parameters 
        -----------
        corr: str, ['pearson'|'spearman'|'covariance']
            Method of correlation to perform. Note that the 'person' and 
            'covariance' don't support string value. If such kind of data 
            is given, turn the `corr` to `spearman`. 
            *default* is ``pearson``
        
        pkg: str, Optional, 
            kind or library to use for visualization. can be ['sns'|'yb']  for 
            'seaborn' or 'yellowbrick' respectively. default is ``sns``.   
        kws: dict, 
            Additional keywords arguments are passed down to 
            :class:`yellowbrick.Rand2D` and `seaborn.heatmap`
        
        Returns 
        -----------
        ``self``: `ExPlot` instance and returns ``self`` for easy method chaining.
             
        Example 
        ---------
        >>> from watex.datasets import fetch_data 
        >>> from watex.view import ExPlot 
        >>> data = fetch_data ('bagoue original').get('data=dfy1') 
        >>> p= ExPlot(tname='flow').fit(data)
        >>> p.plotpairwisecomparison(fmt='.2f', corr='spearman', pkg ='yb',
                                     annot=True, 
                                     cmap='RdBu_r', 
                                     vmin=-1, 
                                     vmax=1 )
        ... <'ExPlot':xname='sfi', yname='ohmS' , tname='flow'>
        """
        self.inspect 
        
        if str(pkg) in ('yellowbrick', 'yb'): 
            pkg ='yb'
        else: pkg ='sns' 
  
        fig, ax = plt.subplots(figsize = self.fig_size )
        df = self.data .copy() 
        
        if pkg =='yb': 
            pcv = Rank2D( ax = ax, 
                         features = df.columns, 
                         algorithm=corr, **kws)
            pcv.fit(df, y = None or self.y_ )
            pcv.transform(df)
            pcv.show() 
            
        elif pkg =='sns': 
            sns.set(rc={"figure.figsize":self.fig_size}) 
            fig = sns.heatmap(data =df.corr() , **kws
                             )
        
        self.save (fig)
        
        return self 
        
    def plotcutcomparison(
            self, 
            xname: str =None, 
            yname:str =None, 
            q:int =10 , 
            bins: int=3 , 
            cmap:str = 'viridis',
            duplicates:str ='drop', 
            **kws
        )->'ExPlot': 
        """Compare the cut or `q` quantiles values of ordinal categories.
        
        It simulates that the the bining of 'xname' into a `q` quantiles, and 
        'yname'into `bins`. Plot is normalized so its fills all the vertical area.
        which makes easy to see that in the `4*q %` quantiles.  
        
        Parameters 
        -------------
        xname, yname : vectors or keys in data
            Variables that specify positions on the x and y axes. Both are 
            the column names to consider. Shoud be items in the dataframe 
            columns. Raise an error if elements do not exist.
        q: int or list-like of float
            Number of quantiles. 10 for deciles, 4 for quartiles, etc. 
            Alternately array of quantiles, e.g. [0, .25, .5, .75, 1.] for 
            quartiles.
        bins: int, sequence of scalars, or IntervalIndex
            The criteria to bin by.

            * int : Defines the number of equal-width bins in the range of x. 
                The range of x is extended by .1% on each side to include the 
                minimum and maximum values of x.

            * sequence of scalars : Defines the bin edges allowing for non-uniform 
                width. No extension of the range of x is done.

            * IntervalIndex : Defines the exact bins to be used. Note that 
                IntervalIndex for bins must be non-overlapping.
                
        labels: array or False, default None
            Used as labels for the resulting bins. Must be of the same length 
            as the resulting bins. If False, return only integer indicators of 
            the bins. If True, raises an error.
            
        cmap: str, color or list of color, optional
            The matplotlib colormap  of the bar faces.
            
        duplicates: {default 'raise', 'drop}, optional
            If bin edges are not unique, raise ValueError or drop non-uniques.
            *default* is 'drop'
        kws: dict, 
            Other keyword arguments are passed down to `pandas.qcut` .
            
        Returns 
        -------
        ``self``: `ExPlot` instance and returns ``self`` for easy method chaining.
        
        Examples 
        ---------
        >>> from watex.datasets import fetch_data 
        >>> from watex.view import ExPlot 
        >>> data = fetch_data ('bagoue original').get('data=dfy1') 
        >>> p= ExPlot(tname='flow').fit(data)
        >>> p.plotcutcomparison(xname ='sfi', yname='ohmS')
        """
        self.inspect 
        
        self.xname_ = xname or self.xname_ 
        self.yname_ = yname or self.yname_ 
        
        fig, ax = plt.subplots(figsize = self.fig_size )
        
        df = self.data .copy() 
        (df.assign(
            xname_bin = pd.qcut( 
                df[self.xname_], q = q, labels =False, 
                duplicates = duplicates, **kws
                ),
            yname_bin = pd.cut(
                df[self.yname_], bins =bins, labels =False, 
                duplicates = duplicates,
                ), 
            )
        .groupby (['xname_bin', 'yname_bin'])
        .size ().unstack()
        .pipe(lambda df: df.div(df.sum(1), axis=0))
        .plot.bar(stacked=True, 
                  width=1, 
                  ax=ax, 
                  cmap =cmap)
        .legend(bbox_to_anchor=(1, 1))
        )
                  
        self.save(fig)
        return self 
         
    def plotbv (
            self, 
            xname: str =None, 
            yname:str =None, 
            kind:str ='box',
            **kwd
        )->'ExPlot': 
        """Visualize distributions using the box, boxen or violin plots. 
        
        Parameters 
        -----------
        xname, yname : vectors or keys in data
            Variables that specify positions on the x and y axes. Both are 
            the column names to consider. Shoud be items in the dataframe 
            columns. Raise an error if elements do not exist.
            
        kind: str 
            style of the plot. Can be ['box'|'boxen'|'violin']. 
            *default* is ``box``
            
        kwd: dict, 
            Other keyword arguments are passed down to `seaborn.boxplot` .
            
        Returns 
        -----------
        ``self``: `ExPlot` instance and returns ``self`` for easy 
        method chaining.
        
        Example
        --------
        >>> from watex.datasets import fetch_data 
        >>> from watex.view import ExPlot 
        >>> data = fetch_data ('bagoue original').get('data=dfy1') 
        >>> p= ExPlot(tname='flow').fit(data)
        >>> p.plotbv(xname='flow', yname='sfi', kind='violin')
        
        """
    
        self.inspect 
        
        self.xname_ = xname or self.xname_ 
        self.yname_ = yname or self.yname_ 
        
        kind = str(kind).lower() 
    
        if kind.find('violin')>=0: kind = 'violin'
        elif kind.find('boxen')>=0 : kind ='boxen'
        else : kind ='box'
        
        df = self.data.copy() 
        if (self.tname not in df.columns) and (self.y_ is not None): 
            df [self.tname] = self.y_  
        
        if kind =='box': 
            g= sns.boxplot( 
                data = df , 
                x= self.xname_,
                y=self.yname_ , 
                **kwd
                )
        if kind =='boxen': 
            g= sns.boxenplot( 
                data = df , 
                x= self.xname_, 
                y=self.yname_ , 
                **kwd
                )
        if kind =='violin': 
            g = sns.violinplot( 
                data = df , 
                x= self.xname_, 
                y=self.yname_ , 
                **kwd
                )
        
        self.save(g)
        
        return self 
    
    
    def plotpairgrid (
            self, 
            xname: str =None, 
            yname:str =None, 
            vars: List[str]= None, 
            **kwd 
    ) -> 'ExPlot': 
        """ Create a pair grid. 
        
        Is a matrix of columns and kernel density estimations. To color by a 
        columns from a dataframe, use 'hue' parameter. 
        
        Parameters 
        -------------
        xname, yname : vectors or keys in data
            Variables that specify positions on the x and y axes. Both are 
            the column names to consider. Shoud be items in the dataframe 
            columns. Raise an error if elements do not exist.
        vars: list, str 
            list of items in the dataframe columns. Raise an error if items 
            dont exist in the dataframe columns. 
        kws: dict, 
            Other keyword arguments are passed down to `seaborn.joinplot`_ .
            
        Returns 
        -----------
        ``self``: `ExPlot` instance and returns ``self`` for easy method chaining.
            
        Example
        --------
        >>> from watex.datasets import fetch_data 
        >>> from watex.view import ExPlot 
        >>> data = fetch_data ('bagoue original').get('data=dfy1') 
        >>> p= ExPlot(tname='flow').fit(data)
        >>> p.plotpairgrid (vars = ['magnitude', 'power', 'ohmS'] ) 
        ... <'ExPlot':xname=(None,), yname=None , tname='flow'>
        
        """
        self.inspect 
        
        self.xname_ = xname or self.xname_ 
        self.yname_ = yname or self.yname_ 

        df = self.data.copy() 
        if (self.tname not in df.columns) and (self.y_ is not None): 
            df [self.tname] = self.y_ # set new dataframe with a target
        if vars is None : 
            vars = [self.xname_, self.y_name ]
            
        sns.set(rc={"figure.figsize":self.fig_size}) 
        g = sns.pairplot (df, vars= vars, hue = self.tname, 
                            **kwd, 
                             )
        self.save(g)
        
        return self 
    
    def plotjoint (
            self, 
            xname: str, 
            yname:str =None,  
            corr: str = 'pearson', 
            kind:str ='scatter', 
            pkg='sns', 
            yb_kws =None, 
            **kws
    )->'ExPlot': 
        """ fancier scatterplot that includes histogram on the edge as well as 
        a regression line called a `joinplot` 
        
        Parameters 
        -------------
        xname, yname : vectors or keys in data
            Variables that specify positions on the x and y axes. Both are 
            the column names to consider. Shoud be items in the dataframe 
            columns. Raise an error if elements do not exist.
        pkg: str, Optional, 
            kind or library to use for visualization. can be ['sns'|'yb']  for 
            'seaborn' or 'yellowbrick'. default is ``sns``.
            
        kind : str in {'scatter', 'hex'}, default: 'scatter'
            The type of plot to render in the joint axes. Note that when 
            kind='hex' the target cannot be plotted by color.
            
        corr: str, default: 'pearson'
            The algorithm used to compute the relationship between the 
            variables in the joint plot, one of: 'pearson', 'covariance', 
            'spearman', 'kendalltau'.
            
        yb_kws: dict, 
            Additional keywords arguments from 
            :class:`yellowbrick.JointPlotVisualizer`
        kws: dict, 
            Other keyword arguments are passed down to `seaborn.joinplot`_ .
            
        Returns 
        -----------
        ``self``: `ExPlot` instance and returns ``self`` for easy method chaining.
             
        Notes 
        -------
        When using the `yellowbrick` library and array i.e a (x, y) variables 
        in the columns as well as the target arrays must not contain infs or NaNs
        values. A value error raises if that is the case. 
        
        .. _seaborn.joinplot: https://seaborn.pydata.org/generated/seaborn.joinplot.html
        
        """
        pkg = str(pkg).lower().strip() 
        if pkg in ('yb', 'yellowbrick'): pkg ='yb'
        else:  pkg ='sns'
        
        self.inspect 
        
        self.xname_ = xname or self.xname_ 
        self.yname_ = yname or self.yname_ 
        
        # assert yb_kws arguments 
        yb_kws = yb_kws or dict() 
        yb_kws = _assert_all_types(yb_kws, dict)
        
        if pkg =='yb': 
            fig, ax = plt.subplots(figsize = self.fig_size )
            jpv = JointPlotVisualizer(
                ax =ax , 
                #columns =self.xname_,   # self.data.columns, 
                correlation=corr, 
                # feature=self.xname_, 
                # target=self.tname, 
                kind= kind , 
                fig = fig, 
                **yb_kws
                )
            jpv.fit(
                self.data [self.xname_], 
                self.data [self.tname] if self.y_ is None else self.y_,
                    ) 
            jpv.show()
        elif pkg =='sns': 
            sns.set(rc={"figure.figsize":self.fig_size}) 
            sns.set_style (self.sns_style)
            df = self.data.copy() 
            if (self.tname not in df.columns) and (self.y_ is not None): 
                df [self.tname] = self.y_ # set new dataframe with a target 
                
            fig = sns.jointplot(
                data= df, 
                x = self.xname_, 
                y= self.yname_,
                **kws
                ) 
            
        self.save(fig )
        
        return self 
        
    def plotscatter (
            self, 
            xname:str =None, 
            yname:str = None, 
            c:str |Series|ArrayLike =None, 
            s: int |ArrayLike =None, 
            **kwd
        )->'ExPlot': 
        """ Shows the relationship between two numeric columns. 
        
        Parameters 
        ------------
        xname, yname : vectors or keys in data
            Variables that specify positions on the x and y axes. Both are 
            the column names to consider. Shoud be items in the dataframe 
            columns. Raise an error if elements do not exist.
          
        c: str, int or array_like, Optional
            The color of each point. Possible values are:
                * A single color string referred to by name, RGB or RGBA code,
                     for instance 'red' or '#a98d19'.
                * A sequence of color strings referred to by name, RGB or RGBA 
                    code, which will be used for each point’s color recursively.
                    For instance [‘green’,’yellow’] all points will be filled 
                    in green or yellow, alternatively.
                * A column name or position whose values will be used to color 
                    the marker points according to a colormap.
                    
        s: scalar or array_like, Optional, 
            The size of each point. Possible values are:
                * A single scalar so all points have the same size.
                * A sequence of scalars, which will be used for each point’s 
                    size recursively. For instance, when passing [2,14] all 
                    points size will be either 2 or 14, alternatively.
        kwd: dict, 
            Other keyword arguments are passed down to `seaborn.scatterplot`_ .
            
        Returns 
        -----------
        ``self``: `ExPlot` instance 
            returns ``self`` for easy method chaining.

        Example 
        ---------
        >>> from watex.view import ExPlot 
        >>> p = ExPlot(tname='flow').fit(data).plotscatter (
            xname ='sfi', yname='ohmS')
        >>> p
        ...  <'ExPlot':xname='sfi', yname='ohmS' , tname='flow'>
        
        References 
        ------------
        Scatterplot: https://seaborn.pydata.org/generated/seaborn.scatterplot.html
        Pd.scatter plot: https://www.w3resource.com/pandas/dataframe/dataframe-plot-scatter.php
        
        .. _seaborn.scatterplot: https://seaborn.pydata.org/generated/seaborn.scatterplot.html
        
        """
        self.inspect 
        
        hue = kwd.pop('hue', None) 
        
        self.xname_ = xname or self.xname_ 
        self.yname_ = yname or self.yname_ 
        
        if hue is not None: 
            self.tname = hue 

        if xname is not None: 
            existfeatures( self.data, self.xname_ )
        if yname is not None: 
            existfeatures( self.data, self.yname_ )
        
        # state the fig plot and change the figure size 
        sns.set(rc={"figure.figsize":self.fig_size}) #width=3, #height=4
        if self.sns_style is not None: 
           sns.set_style(self.sns_style)
        # try : 
        fig= sns.scatterplot( data = self.data, x = self.xname_,
                             y=self.yname_, hue =self.tname, 
                        # ax =ax , # call matplotlib.pyplot.gca() internally
                        **kwd)
        # except : 
        #     warnings.warn("The following variable cannot be assigned with "
        #                   "wide-form data: `hue`; use the pandas scatterplot "
        #                   "instead.")
        #     self.data.plot.scatter (x =xname , y=yname, c=c,
        #                             s = s,  ax =ax  )
        
        self.save(fig) 
        
        return self 
    
    def plothistvstarget (
            self, 
            xname: str, 
            c: Any =None, *,  
            posilabel: str = None, 
            neglabel: str= None,
            kind='binarize', 
            **kws
        )->'ExPlot': 
        """
        A histogram of continuous against the target of binary plot. 
        
        Parameters 
        ----------
        xname: str, 
            the column name to consider on x-axis. Shoud be  an item in the  
            dataframe columns. Raise an error if element does not exist.  
            
        c: str or  int  
            the class value in `y` to consider. Raise an error if not in `y`. 
            value `c` can be considered as the binary positive class 
            
        posilabel: str, Optional 
            the label of `c` considered as the positive class 
        neglabel: str, Optional
            the label of other classes (categories) except `c` considered as 
            the negative class 
        kind: str, Optional, (default, 'binarize') 
            the kind of plot features against target. `binarize` considers 
            plotting the positive class ('c') vs negative class ('not c')
        
        kws: dict, 
            Additional keyword arguments of  `seaborn displot`_ 
            
        Returns 
        -----------
        ``self``: `ExPlot` instance 
            returns ``self`` for easy method chaining.

        Examples
        --------
        >>> from watex.utils import read_data 
        >>> from watex.view import ExPlot
        >>> data = read_data  ( 'data/geodata/main.bagciv.data.csv' ) 
        >>> p = ExPlot(tname ='flow').fit(data)
        >>> p.fig_size = (7, 5)
        >>> p.savefig ='bbox.png'
        >>> p.plothistvstarget (xname= 'sfi', c = 0, kind = 'binarize',  kde=True, 
                          posilabel='dried borehole (m3/h)',
                          neglabel = 'accept. boreholes'
                          )
        Out[95]: <'ExPlot':xname='sfi', yname=None , tname='flow'>
        """
     
        self.inspect 
            
        self.xname_ = xname or self.xname_ 
        
        existfeatures(self.data, self.xname_) # assert the name in the columns
        df = self.data.copy() 
       
        if str(kind).lower().strip().find('bin')>=0: 
            if c  is None: 
                raise ValueError ("Need a categorical class value for binarizing")
                
            _assert_all_types(c, float, int)
            
            if self.y_ is None: 
                raise ValueError ("target name is missing. Specify the `tname`"
                                  f" and refit {self.__class__.__name__!r} ")
            if not _isin(self.y_, c ): 
                raise ValueError (f"c-value should be a class label, got '{c}'"
                                  )
            mask = self.y_ == c 
            # for consisteny use np.unique to get the classes
            
            neglabel = neglabel or shrunkformat( 
                np.unique (self.y_[~(self.y_ == c)]) 
                )

        else: 
  
            if self.tname is None: 
                raise ValueError("Can't plot binary classes with missing"
                                 " target name ")
            df[self.tname] = df [self.tname].map(
                lambda x : 1 if ( x == c if isinstance(c, str) else x <=c 
                                 )  else 0  # mapping binary target
                )

        #-->  now plot 
        # state the fig plot  
        sns.set(rc={"figure.figsize":self.fig_size}) 
        if  str(kind).lower().strip().find('bin')>=0: 
            g=sns.histplot (data = df[mask][self.xname_], 
                            label= posilabel or str(c) , 
                            linewidth = self.lw, 
                            **kws
                          )
            
            g=sns.histplot( data = df[~mask][self.xname_], 
                              label= neglabel , 
                              linewidth = self.lw,
                              **kws,
                              )
        else : 
            g=sns.histplot (data =df , 
                              x = self.xname_,
                              hue= self.tname,
                              linewidth = self.lw, 
                          **kws
                        )
            
        if self.sns_style is not None: 
            sns.set_style(self.sns_style)
            
        g.legend ()
        
        # self.save(g)
  
        return self 

    def plothist(self,xname: str = None, *,  kind:str = 'hist', 
                   **kws 
                   ): 
        """ A histogram visualization of numerica data.  
        
        Parameters 
        ----------
        xname: str , xlabel 
            feature name in the dataframe  and is the label on x-axis. 
            Raises an error , if it does not exist in the dataframe 
        kind: str 
            Mode of pandas series plotting. the *default* is ``hist``. 
            
        kws: dict, 
            additional keywords arguments from : func:`pandas.DataFrame.plot` 
            
        Return
        -------
        ``self``: `ExPlot` instance 
            returns ``self`` for easy method chaining.
            
        """
        self.inspect  
        self.xname_ = xname or self.xname_ 
        xname = _assert_all_types(self.xname_,str )
        # assert whether whether  feature exists 
        existfeatures(self.data, self.xname_)
    
        fig, ax = plt.subplots (figsize = self.fig_size or self.fig_size )
        self.data [self.xname_].plot(kind = kind , ax= ax  , **kws )
        
        self.save(fig)
        
        return self 
    
    def plotmissing(self, *, 
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
            * ``mbar`` use the :mod:`msno` package to count the number 
                of nonmissing data. 
            * dendrogram`` show the clusterings of where the data is missing. 
                leaves that are the same level predict one onother presence 
                (empty of filled). The vertical arms are used to indicate how  
                different cluster are. short arms mean that branch are 
                similar. 
            * ``corr` creates a heat map showing if there are correlations 
                where the data is missing. In this case, it does look like 
                the locations where missing data are corollated.
            * ``mpatterns`` is the default vizualisation. It is useful for viewing 
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
        ``self``: `ExPlot` instance 
            returns ``self`` for easy method chaining.
            
        Example
        --------
        >>> import pandas as pd 
        >>> from watex.view import ExPlot
        >>> data = pd.read_csv ('data/geodata/main.bagciv.data.csv' ) 
        >>> p = ExPlot().fit(data)
        >>> p.fig_size = (12, 4)
        >>> p.plotmissing(kind ='corr')
        
        """
        self.inspect 
            
        kstr =('dendrogram', 'bar', 'mbar', 'correlation', 'mpatterns')
        kind = str(kind).lower().strip() 
        
        regex = re.compile (r'none|dendro|corr|base|default|mbar|bar|mpat', 
                            flags= re.IGNORECASE)
        kind = regex.search(kind) 
        
        if kind is None: 
            raise ValueError (f"Expect {smart_format(kstr, 'or')} not: {kind!r}")
            
        kind = kind.group()
  
        if kind in ('none', 'default', 'base', 'mpat'): 
            kind ='mpat'
        
        if sample is not None: 
            sample = _assert_all_types(sample, int, float)
            
        if kind =='bar': 
            fig, ax = plt.subplots (figsize = self.fig_size, **kwd )
            (1- self.data.isnull().mean()).abs().plot.bar(ax=ax)
    
        elif kind  in ('mbar', 'dendro', 'corr', 'mpat'): 
            try : 
                msno 
            except : 
                raise ModuleNotFoundError(
                    f"Missing 'missingno' package. Can not plot {kind!r}")
                
            if kind =='mbar': 
                ax = msno.bar(
                    self.data if sample is None else self.data.sample(sample),
                              figsize = self.fig_size 
                              )
    
            elif kind =='dendro': 
                ax = msno.dendrogram(self.data, figsize = self.fig_size , **kwd) 
        
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
    
    def __repr__(self): 
        """ Represent the output class format """
        return  "<{0!r}:xname={1!r}, yname={2!r} , tname={3!r}>".format(
            self.__class__.__name__, self.xname_ , self.yname_ , self.tname 
            )


                       
class QuickPlot (BasePlot): 
    def __init__(
        self,  
        classes = None, 
        tname= None, 
        mapflow=False, 
        **kws
        ): 
        super().__init__(**kws)
        
        self._logging =watexlog().get_watex_logger(self.__class__.__name__)
        self.classes=classes
        self.tname=tname
        self.mapflow=mapflow
        
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
        features inspection.
        
        """
          
        if str(self.tname).lower() =='flow':
            # default inspection for DC -flow rate prediction
           fobj= FeatureInspection( set_index=True, 
                flow_classes = self.classes or [0., 1., 3] , 
                target = self.tname, 
                mapflow= self.mapflow 
                           ).fit(data=data)
           self.data_= fobj.data  
           
        self.data_ = _is_readable(
            data , input_name="'data'")
        
        if str(self.tname).lower() in self.data_.columns.str.lower(): 
            ix = list(self.data.columns.str.lower()).index (
                self.tname.lower() )
            self.y = self.data_.iloc [:, ix ]

            self.X_ = self.data_.drop(columns =self.data_.columns[ix] , 
                                         )
            
    def fit(
        self,
        data: str | DataFrame, 
        y: Optional[Series| ArrayLike]=None
    )-> "QuickPlot" : 
        """ 
        Fit data and populate the attributes for plotting purposes. 
        
        Parameters 
        ----------
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`QuickPlot` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target.  
        
        y: array-like, optional 
            array of the target. Must be the same length as the data. If `y` 
            is provided and `data` is given as ``str`` or ``DataFrame``, 
            all the data should be considered as the X data for analysis. 
  
         Returns
         -------
         self: :class:`QuickPlot` instance
             Returns ``self`` for easy method chaining.
             
        Examples 
        --------
        >>> from watex.datasets import load_bagoue
        >>> data = load_bagoue ().frame
        >>> from watex.view.plot import QuickPlot
        >>> qplotObj= QuickPlot(xlabel = 'Flow classes in m3/h',
                                ylabel='Number of  occurence (%)')
        >>> qplotObj.tname= None # eith nameof target set to None 
        >>> qplotObj.fit(data)
        >>> qplotObj.data.iloc[1:2, :]
        ...     num name      east      north  ...         ohmS        lwi      geol flow
            1  2.0   b2  791227.0  1159566.0  ...  1135.551531  21.406531  GRANITES  0.0
        >>> qplotObj.tname= 'flow'
        >>> qplotObj.mapflow= True # map the flow from num. values to categ. values
        >>> qplotObj.fit(data)
        >>> qplotObj.data.iloc[1:2, :]
        ...    num name      east      north  ...         ohmS        lwi      geol flow
            1  2.0   b2  791227.0  1159566.0  ...  1135.551531  21.406531  GRANITES  FR0
         
        """
        self.data = data 
        if y is not None: 
            _, y = check_X_y(
                self.data, y, 
                force_all_finite="allow-nan", 
                dtype =object, 
                to_frame = True 
                )
            y = _assert_all_types(y, np.ndarray, list, tuple, pd.Series)
            if len(y)!= len(self.data) :
                raise ValueError(
                    f"y and data must have the same length but {len(y)} and"
                    f" {len(self.data)} were given respectively.")
            
            self.y = pd.Series (y , name = self.tname or 'none')
            # for consistency get the name of target 
            self.tname = self.y.name 
            
        return self 
    
    def histcatdist(
        self, 
        stacked: bool = False,  
        **kws
        ): 
        """
        Histogram plot distribution. 
        
        Plots a distributions of categorized classes according to the 
        percentage of occurence. 
        
        Parameters 
        -----------
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
          
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`QuickPlot` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 
            
        Returns
        -------
        :class:`QuickPlot` instance
            Returns ``self`` for easy method chaining.
            
        Notes 
        -------
        The argument for  `data` must be passed to `fit` method. `data` 
        parameter is not allowed in other `QuickPlot` method. The description 
        of the parameter `data` is to give a synopsis of the kind of data 
        the plot expected. An error will raise if force to pass `data` 
        argument as a keyword arguments. 
        
        Examples 
        ---------
        >>> from watex.view.plot import QuickPlot
        >>> from watex.datasets import load_bagoue 
        >>> data = load_bagoue ().frame
        >>> qplotObj= QuickPlot(xlabel = 'Flow classes',
                                ylabel='Number of  occurence (%)',
                                lc='b', tname='flow')
        >>> qplotObj.sns_style = 'darkgrid'
        >>> qplotObj.fit(data)
        >>> qplotObj. histcatdist()
        
        """
        self._logging.info('Quick plot of categorized classes distributions.'
                           f' the target name: {self.tname!r}')
        
        self.inspect 
    
        if self.tname is None and self.y is None: 
            raise FeatureError(
                "Please specify 'tname' as the name of the target")

        # reset index 
        df_= self.data_.copy()  #make a copy for safety 
        df_.reset_index(inplace =True)
        
        if kws.get('bins', None) is not None: 
            self.bins = kws.pop ('bins', None)
            
        plt.figure(figsize =self.fig_size)
        plt.hist(df_[self.tname], bins=self.bins ,
                  stacked = stacked , color= self.lc , **kws)

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.fig_title)

        if self.savefig is not None :
            plt.savefig(self.savefig,dpi=self.fig_dpi,
                        orientation =self.fig_orientation
                        )
        
        return self 
    
    def barcatdist(
        self,
        basic_plot: bool = True,
        groupby: List[str] | Dict [str, float] =None,
        **kws):
        """
        Bar plot distribution. 
        
        Plots a categorical distribution according to the occurence of the 
        `target` in the data. 
        
        Parameters 
        -----------
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
          
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`QuickPlot` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 
            
        Returns
        -------
        :class:`QuickPlot` instance
            Returns ``self`` for easy method chaining.
            
        Notes 
        -------
        The argument for  `data` must be passed to `fit` method. `data` 
        parameter is not allowed in other `QuickPlot` method. The description 
        of the parameter `data` is to give a synopsis of the kind of data 
        the plot expected. An error will raise if force to pass `data` 
        argument as a keyword arguments. 
        
        Examples
        ----------
        >>> from watex.view.plot import QuickPlot
        >>> from watex.datasets import load_bagoue 
        >>> data = load_bagoue ().frame
        >>> qplotObj= QuickPlot(xlabel = 'Anomaly type',
                                ylabel='Number of  occurence (%)',
                                lc='b', tname='flow')
        >>> qplotObj.sns_style = 'darkgrid'
        >>> qplotObj.fit(data)
        >>> qplotObj. barcatdist(basic_plot =False, 
        ...                      groupby=['shape' ])
   
        """
        self.inspect 
        
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
            ax.bar(list(set(df_[self.tname])), 
                        df_[self.tname].value_counts(normalize =True),
                        label= self.fig_title, color = self.lc, )  
    
        if groupby is not None : 
            if hasattr(self, 'sns_style'): 
                sns.set_style(self.sns_style)
            if isinstance(groupby, str): 
                self.groupby =[groupby]
            if isinstance(groupby , dict):
                groupby =list(groupby.keys())
            for sll in groupby :
                ax= sns.countplot(x= sll,  hue=self.tname, 
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
    
    
    def multicatdist(
        self, 
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
        Figure-level interface for drawing multiple categorical distributions
        plots onto a FacetGrid.
        
        Multiple categorials plots  from targetted pd.series. 
        
        Parameters 
        -----------
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
            
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`QuickPlot` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 
            
        Returns
        -------
        :class:`QuickPlot` instance
            Returns ``self`` for easy method chaining.
            
        Notes 
        -------
        The argument for  `data` must be passed to `fit` method. `data` 
        parameter is not allowed in other `QuickPlot` method. The description 
        of the parameter `data` is to give a synopsis of the kind of data 
        the plot expected. An error will raise if force to pass `data` 
        argument as a keyword arguments. 
        
        Examples
        ---------
        >>> from watex.view.plot import QuickPlot 
        >>> from watex.datasets import load_bagoue 
        >>> data = load_bagoue ().frame
        >>> qplotObj= QuickPlot(lc='b', tname='flow')
        >>> qplotObj.sns_style = 'darkgrid'
        >>> qplotObj.mapflow=True # to categorize the flow rate 
        >>> qplotObj.fit(data)
        >>> fdict={
        ...            'x':['shape', 'type', 'type'], 
        ...            'col':['type', 'geol', 'shape'], 
        ...            'hue':['flow', 'flow', 'geol'],
        ...            } 
        >>> qplotObj.multicatdist(**fdict)
            
        """
        self.inspect 
            
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
    
    def corrmatrix(
        self,
        cortype:str ='num',
        features: Optional[List[str]] = None, 
        method: str ='pearson',
        min_periods: int=1, 
        **sns_kws): 
        """
        Method to quick plot the numerical and categorical features. 
        
        Set `features` by providing the names of  features for visualization. 

        Parameters 
        -----------
        cortype: str, 
            The typle of parameters to cisualize their coreletions. Can be 
            ``num`` for numerical features and ``cat`` for categorical features. 
            *Default* is ``num`` for quantitative values. 
            
        method: str,  
            the correlation method. can be 'spearman' or `person`. *Default is
            ``pearson``
            
        features: List, optional 
            list of  the name of features for correlation analysis. If given, 
            must be sure that the names belong to the dataframe columns,  
            otherwise an error will occur. If features are valid, dataframe 
            is shrunk to the number of features before the correlation plot.
       
        min_periods: 
                Minimum number of observations required per pair of columns
                to have a valid result. Currently only available for 
                ``pearson`` and ``spearman`` correlation. For more details 
                refer to https://www.geeksforgeeks.org/python-pandas-dataframe-corr/
        
        sns_kws: Other seabon heatmap arguments. Refer to 
                https://seaborn.pydata.org/generated/seaborn.heatmap.html
                
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`QuickPlot` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 
            
        Returns
        -------
        :class:`QuickPlot` instance
            Returns ``self`` for easy method chaining.
            
        Notes 
        -------
        The argument for  `data` must be passed to `fit` method. `data` 
        parameter is not allowed in other `QuickPlot` method. The description 
        of the parameter `data` is to give a synopsis of the kind of data 
        the plot expected. An error will raise if force to pass `data` 
        argument as a keyword arguments. 
             
           
        Example 
        ---------
        >>> from watex.view.plot import QuickPlot 
        >>> from watex.datasets import load_bagoue 
        >>> data = load_bagoue ().frame
        >>> qplotObj = QuickPlot().fit(data)
        >>> sns_kwargs ={'annot': False, 
        ...            'linewidth': .5, 
        ...            'center':0 , 
        ...            # 'cmap':'jet_r', 
        ...            'cbar':True}
        >>> qplotObj.corrmatrix(cortype='cat', **sns_kwargs)
            
        """
        self.inspect 
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
    
              
    def numfeatures(
        self, 
        features=None, 
        coerce: bool= False,  
        map_lower_kws=None,
        **sns_kws): 
        """
        Plots qualitative features distribution using correlative aspect. Be 
        sure to provide numerical features as data arguments. 
        
        Parameters 
        -----------
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
                     
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`QuickPlot` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 
            
        Returns
        -------
        :class:`QuickPlot` instance
            Returns ``self`` for easy method chaining.
            
        Notes 
        -------
        The argument for  `data` must be passed to `fit` method. `data` 
        parameter is not allowed in other `QuickPlot` method. The description 
        of the parameter `data` is to give a synopsis of the kind of data 
        the plot expected. An error will raise if force to pass `data` 
        argument as a keyword arguments. 
            
              
        Examples
        ---------
        >>> from watex.view.plot import QuickPlot 
        >>> from watex.datasets import load_bagoue 
        >>> data = load_bagoue ().frame
        >>> qkObj = QuickPlot(mapflow =False, tname='flow'
                                  ).fit(data)
        >>> qkObj.sns_style ='darkgrid', 
        >>> qkObj.fig_title='Quantitative features correlation'
        >>> sns_pkws={'aspect':2 , 
        ...          "height": 2, 
        # ...          'markers':['o', 'x', 'D', 'H', 's',
        #                         '^', '+', 'S'],
        ...          'diag_kind':'kde', 
        ...          'corner':False,
        ...          }
        >>> marklow = {'level':4, 
        ...          'color':".2"}
        >>> qkObj.numfeatures(coerce=True, map_lower_kws=marklow, **sns_pkws)
                                                
        """
        self.inspect 
            
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
        ax =sns.pairplot(data =df_, hue=self.tname,**sns_kws)
        
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
    
    def joint2features(
        self,
        features: List [str], *,
        join_kws=None, marginals_kws=None, 
        **sns_kws):
        """
        Joint method allows to visualize correlation of two features. 
        
        Draw a plot of two features with bivariate and univariate graphs. 
        
        Parameters 
        -----------
        features: list
            List of numerical features to plot for correlating analyses. 
            will raise an error if features does not exist in the data 

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
          
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`QuickPlot` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 
            
        Returns
        -------
        :class:`QuickPlot` instance
            Returns ``self`` for easy method chaining.
            
        Notes 
        -------
        The argument for  `data` must be passed to `fit` method. `data` 
        parameter is not allowed in other `QuickPlot` method. The description 
        of the parameter `data` is to give a synopsis of the kind of data 
        the plot expected. An error will raise if force to pass `data` 
        argument as a keyword arguments. 
              
             
        Examples
        ----------
        >>> from watex.view.plot import QuickPlot 
        >>> from watex.datasets import load_bagoue 
        >>> data = load_bagoue ().frame
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
        self.inspect 
  
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
          
    def scatteringfeatures(
        self,features: List [str], 
        *,
        relplot_kws= None, 
        **sns_kws 
        ): 
        """
        Draw a scatter plot with possibility of several semantic features 
        groupings.
        
        Indeed `scatteringfeatures` analysis is a process of understanding 
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
            
        relplot_kws: dict, optional 
            Extra keyword arguments to show the relationship between 
            two features with semantic mappings of subsets.
            refer to :ref:`<http://seaborn.pydata.org/generated/seaborn.relplot.html#seaborn.relplot>`
            for more details. 
            
        sns_kwargs:dict, optional
            kwywords arguments to control what visual semantics are used 
            to identify the different subsets. For more details, please consult
            :ref:`<http://seaborn.pydata.org/generated/seaborn.scatterplot.html>`. 
            
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`QuickPlot` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 
            
        
        Returns
        -------
        :class:`QuickPlot` instance
            Returns ``self`` for easy method chaining.
          
        Notes 
        -------
        The argument for  `data` must be passed to `fit` method. `data` 
        parameter is not allowed in other `QuickPlot` method. The description 
        of the parameter `data` is to give a synopsis of the kind of data 
        the plot expected. An error will raise if force to pass `data` 
        argument as a keyword arguments. 
            
        Examples
        ----------
        >>> from watex.view.plot import  QuickPlot 
        >>> from watex.datasets import load_bagoue 
        >>> data = load_bagoue ().frame
        >>> qkObj = QuickPlot(lc='b', sns_style ='darkgrid', 
        ...             fig_title='geol vs lewel of water inflow',
        ...             xlabel='Level of water inflow (lwi)', 
        ...             ylabel='Flow rate in m3/h'
        ...            ) 
        >>>
        >>> qkObj.tname='flow' # target the DC-flow rate prediction dataset
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
        self.inspect 
            
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
       
    def discussingfeatures(
        self, features, *, 
        map_kws: Optional[dict]=None, 
        map_func: Optional[F] = None, 
        **sns_kws)-> None: 
        """
        Provides the features names at least 04 and discuss with 
        their distribution. 
        
        This method maps a dataset onto multiple axes arrayed in a grid of
        rows and columns that correspond to levels of features in the dataset. 
        The plots produced are often called "lattice", "trellis", or
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
           
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`QuickPlot` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 
            
        Returns
        -------
        :class:`QuickPlot` instance
            Returns ``self`` for easy method chaining.
            
        Notes 
        -------
        The argument for  `data` must be passed to `fit` method. `data` 
        parameter is not allowed in other `QuickPlot` method. The description 
        of the parameter `data` is to give a synopsis of the kind of data 
        the plot expected. An error will raise if force to pass `data` 
        argument as a keyword arguments. 

        Examples
        --------
        >>> from watex.view.plot import  QuickPlot 
        >>> from watex.datasets import load_bagoue 
        >>> data = load_bagoue ().frame 
        >>> qkObj = QuickPlot(  leg_kws={'loc':'upper right'},
        ...          fig_title = '`sfi` vs`ohmS|`geol`',
        ...            ) 
        >>> qkObj.tname='flow' # target the DC-flow rate prediction dataset
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
        self.inspect 

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
         
    def naiveviz(
        self,
        x:str =None, 
        y:str =None, 
        kind:str ='scatter',
        s_col ='lwi', 
        leg_kws:dict ={}, 
        **pd_kws
        ):
        """ Creates a plot  to visualize the samples distributions 
        according to the geographical coordinates `x` and `y`.
        
        Parameters 
        -----------
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
           
        data: str or pd.core.DataFrame
            Path -like object or Dataframe. Long-form (tidy) dataset for 
            plotting. Each column should correspond to a variable,  and each 
            row should correspond to an observation. If data is given as 
            path-like object,`QuickPlot` reads and sanitizes data before 
            plotting. Be aware in this case to provide the target name and 
            possible the `classes` for data inspection. Both str or dataframe
            need to provide the name of target. 
            
        Returns
        -------
        :class:`QuickPlot` instance
            Returns ``self`` for easy method chaining.
            
        Notes 
        -------
        The argument for  `data` must be passed to `fit` method. `data` 
        parameter is not allowed in other `QuickPlot` method. The description 
        of the parameter `data` is to give a synopsis of the kind of data 
        the plot expected. An error will raise if force to pass `data` 
        argument as a keyword arguments. 
        
        Examples
        --------- 
        >>> from watex.transformers import StratifiedWithCategoryAdder
        >>> from watex.view.plot import QuickPlot
        >>> from watex.datasets import load_bagoue 
        >>> df = load_bagoue ().frame
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
        >>> qkObj.naiveviz( x= 'east', y='north', **pd_kws)
    
        """
        self.inspect
            
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
    
    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        return repr_callable_obj  (self, skip ='y') 
    
       
    def __getattr__(self, name):
        if not name.endswith ('__') and name.endswith ('_'): 
            raise NotFittedError (
                f"{self.__class__.__name__!r} instance is not fitted yet."
                " Call 'fit' method with appropriate arguments before"
               f" retreiving the attribute {name!r} value."
                )
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        appender  = "" if rv is None else f'. Do you mean {rv!r}'
        
        raise AttributeError (
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{appender}{"" if rv is None else "?"}'
            )      
       
    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'data_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1
     
ExPlot .__doc__="""\
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
 
QuickPlot.__doc__="""\
Special class dealing with analysis modules for quick diagrams, 
histograms and bar visualizations. 

Originally, it was designed for the flow rate prediction, however, it still 
works with any other dataset by following the parameters details. 
  
Parameters 
-------------
{params.core.data}
{params.core.y}
{params.core.tname}
{params.qdoc.classes}
{params.qdoc.mapflow}
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
>>> from watex.view.plot import  QuickPlot 
>>> data = 'data/geodata/main.bagciv.data.csv'
>>> qkObj = QuickPlot(  leg_kws= dict( loc='upper right'),
...          fig_title = '`sfi` vs`ohmS|`geol`',
...            ) 
>>> qkObj.tname='flow' # target the DC-flow rate prediction dataset
>>> qkObj.mapflow=True  # to hold category FR0, FR1 etc..
>>> qkObj.fit(data) 
>>> sns_pkws= dict ( aspect = 2 , 
...          height= 2, 
...                  )
>>> map_kws= dict( edgecolor="w")    
>>> qkObj.discussingfeatures(features =['ohmS', 'sfi','geol', 'flow'],
...                           map_kws=map_kws,  **sns_pkws
...                         )   
""".format(
    params=_param_docs,
    returns= _core_docs["returns"],
)
    
TPlot.__doc__="""\
Tensor plot from EM processing data.

`TPlot` is a :term:`Tensor` (Impedances , resistivity and phases ) plot class. 
Explore SEG ( Society of Exploration Geophysicist ) class data.  Plot recovery 
tensors. `TPlot` methods returns an instancied object that inherits 
from :class:`watex.property.Baseplots` ABC (Abstract Base Class) for 
visualization.
    
Parameters 
------------

window_size : int
    the length of the window. Must be greater than 1 and preferably
    an odd integer number. Default is ``5``
    
component: str 
   field tensors direction. It can be ``xx``, ``xy``,``yx``, ``yy``. If 
   `arr2d`` is provided, no need to give an argument. It become useful 
   when a collection of EDI-objects is provided. If don't specify, the 
   resistivity and phase value at component `xy` should be fetched for 
   correction by default. Change the component value to get the appropriate 
   data for correction. Default is ``xy``.
   
mode: str , ['valid', 'same'], default='same'
    mode of the border trimming. Should be 'valid' or 'same'.'valid' is used 
    for regular trimimg whereas the 'same' is used for appending the first
    and last value of resistivity. Any other argument except 'valid' should 
    be considered as 'same' argument. Default is ``same``.     
   
method: str, default ``slinear``
    Interpolation technique to use. Can be ``nearest``or ``pad``. Refer to 
    the documentation of :doc:`~.interpolate2d`. 
    
out : str 
    Value to export. Can be ``sfactor``, ``tensor`` for corrections factor 
    and impedance tensor. Any other values will export the static corrected  
    resistivity ``srho``. 
    
c : int, 
    A window-width expansion factor that must be input to the filter 
    adaptation process to control the roll-off characteristics
    of the applied Hanning window. It is recommended to select `c` between 
    ``1``  and ``4``.  Default is ``2``.
    
distance: float 
    The step between two stations/sites. If given, it creates an array of  
    position for plotting purpose. Default value is ``50`` meters. 
 
prefix: str 
    string value to add as prefix of given id. Prefix can be the site 
    name. Default is ``S``. 
    
how: str 
    Mode to index the station. Default is 'Python indexing' i.e. 
    the counting of stations would starts by 0. Any other mode will 
    start the counting by 1.
     
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
>>> from watex.view.plot import TPlot 
>>> from watex.datasets import load_edis 
>>> plot_kws = dict( ylabel = '$Log_{{10}}Frequency [Hz]$', 
                    xlabel = '$Distance(m)$', 
                    cb_label = '$Log_{{10}}Rhoa[\Omega.m$]', 
                    fig_size =(6, 3), 
                    font_size =7., 
                    rotate_xlabel=45, 
                    imshow_interp='bicubic', 
                    ) 
>>> edi_data =load_edis (return_data= True, samples=7 ) 
>>> t= TPlot(**plot_kws ).fit(edi_data)
>>> t.fit(edi_data ).plot_tensor2d (to_log10=True )
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
|Data collected =  7      |EDI success. read=  7      |Rate     =  100.0  %|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Out[150]: <AxesSubplot:xlabel='$Distance(m)$', ylabel='$Log_{{10}}Frequency [Hz]$'>
""".format(
    params=_param_docs,
    returns= _core_docs["returns"],
)
         
def viewtemplate (y, /, xlabel=None, ylabel =None,  **kws):
    """
    Quick view template
    
    Parameters 
    -----------
    y: Arraylike , shape (N, )
    xlabel: str, Optional 
        Label for naming the x-abscissia 
    ylabel: str, Optional, 
        Label for naming the y-coordinates.
    kws: dict, 
        keywords argument passed to :func:`matplotlib.pyplot.plot`

    """
    label =kws.pop('label', None)
    # create figure obj 
    obj = ExPlot()
    fig = plt.figure(figsize = obj.fig_size)
    ax = fig.add_subplot(1,1,1)
    ax.plot(y,
            color= obj.lc, 
            linewidth = obj.lw,
            linestyle = obj.ls , 
            label =label, 
            **kws
            )
    
    if obj.xlabel is None: 
        obj.xlabel =xlabel or ''
    if obj.ylabel is None: 
        obj.ylabel =ylabel  or ''

    ax.set_xlabel( obj.xlabel,
                  fontsize= .5 * obj.font_size * obj.fs 
                  )
    ax.set_ylabel (obj.ylabel,
                   fontsize= .5 * obj.font_size * obj.fs
                   )
    ax.tick_params(axis='both', 
                   labelsize=.5 * obj.font_size * obj.fs
                   )
    
    if obj.show_grid is True : 
        if obj.gwhich =='minor': 
              ax.minorticks_on() 
        ax.grid(obj.show_grid,
                axis=obj.gaxis,
                which = obj.gwhich, 
                color = obj.gc,
                linestyle=obj.gls,
                linewidth=obj.glw, 
                alpha = obj.galpha
                )
          
        if len(obj.leg_kws) ==0 or 'loc' not in obj.leg_kws.keys():
             obj.leg_kws['loc']='upper left'
        
        ax.legend(**obj.leg_kws)
        

        plt.show()
        
        if obj.savefig is not None :
            plt.savefig(obj.savefig,
                        dpi=obj.fig_dpi,
                        orientation =obj.fig_orientation
                        )     

# import matplotlib.cm as cm 
# import matplotlib.colorbar as mplcb
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.ticker import MultipleLocator, NullLocator
# import matplotlib.gridspec as gspec        
        
        
        