# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Created on Tue May 18 12:33:15 2021
"""
ERP 
=======
DC-1D Resistivty drilling location Auto-detecting 

Notes
---------
This module is one of the earlier module designed for predicting flow rate by 
computing the electrical parameters. Originaly the idea was to automate 
everything to ease the task for the users. All things,  the users need to do, 
is to arrange the electrical data according to the arrangement proposed 
by the library in :class:`watex.property.P` such as:
    
    === ======== =========== ========== ======== ======= ========
    *pk  *x         *y          *rho      sloc     shape  type   
    === ======== =========== ========== ======== ======= ========
    0   790210      1093010     230        low   
    10  790214      1093016     93         se       V       CP
    20  790218      1093026     93         up
        ...
    140 790255      1093116     138        
    === ======== =========== ========== ======== ======= ========
    
    where headers with `*` means compulsory data and optional otherwise. 
    `x`  and `y` are  utm easting and northing coordinates respectively, 
    while `rho` is the apparent resistivity at each measurement point(`pk`).
    `sloc` is the column of anomaly boundaries definition. The optional 
    column names such as `sloc`, `shape` and  `type` can be None.
    Inside the table: 
        
    - `low` means the lower boundary of the selected anomaly,  can also be '1'
    - `up` means the uper boundary of selected anomaly, can be `2` 
    - `se` means the sounding location on the survey area. can be `ves` or `0`. 
    - `V` anomaly-shape and can be 'W', 'K', 'U', 'H', 'C' and 'M' 
    - `CP` anomaly type and can be 'CB2P', 'NC' or 'EC' 

The main interesting part of this module is the collection of ERP where 
the module can rewrite the data and arrange it following the aforementioned  
disposal (above proposed by the library). If data  is given in a separate sheets 
(from excel files), the parser exports each sheet and rewrite accordingly. The 
limit of this approach is that the parser only read the excel format. 

Warnings 
----------    
Thus, once the data is  well organized, the module is able to compute all the 
parameters and select the best location for drilling after analyzing all the 
different points in dataset. 
However, this seems too much perfect (not realistic) and far from the practice
since in DC - resistivity, the low resistivity does not mean there is a water 
at that place thereby leading to a misinsterpretationin the choice of locating 
of the drilling points. To handle this issue, we recommended to use the module 
:mod:`watex.method.electrical` instead. To force use ERP module, be sure you 
are a background of the geology of the area and whether you are not in marshes 
or a complex geological area which unfortunately is something difficult 
to know in advance. 
To well organize the watex API and remediate to the problem of automation, it 
is recommended to use the :class:`watex.methods.electrical.DCProfiling`. The 
latter provides fast and efficient way to compute the electrical 
parameters with a few margin of errors. The module will be deprecated in the  
future and should be rewritten. The automation of parameter computation from 
the erp parser sheets such as shape and type of anomaly will henceforth use 
the deep neural networks.  

"""
import os
import re 
import warnings
import datetime
import  shutil

import numpy as np 
import pandas as pd

from ..decorators import deprecated 
from .._watexlog import watexlog 
from .._typing import  ( 
    List, 
    Optional, 
    )
from ..exceptions import ( 
    FileHandlingError,
    ERPError, 
    NotFittedError,
    DCError, 
    )
from ..utils.exmath import ( 
    select_anomaly, 
    compute_anr,
    compute_sfi,
    compute_power, 
    compute_magnitude, 
    gettype, 
    getshape 
    )
from ..utils.funcutils import ( 
    display_infos, 
    savepath_, 
    get_boundaries, 
    wrap_infos, 
    repr_callable_obj, 
    smart_strobj_recognition,
    get_xy_coordinates, 
    listing_items_format, 

    )
from ..utils.gistools import ( 
    ll_to_utm, 
    utm_to_ll, 
    project_point_ll2utm, 
    project_point_utm2ll 
    )

from ..property import ElectricalMethods 
from .electrical import ( 
    DCProfiling , DCSounding, 
    ) 
from ..utils.coreutils import  ( 
    _is_readable , 
    vesSelector, 
    erpSelector
    )
from ..utils.validator import get_estimator_name 

_logger =watexlog.get_watex_logger(__name__)


__all__=['DCMagic', 'ERPCollection'] 

class DCMagic (ElectricalMethods ): 
    """A super class that deals with ERP and VES objects to generate 
    single DC features for prediction. 
    
    `DCMagic reads the :term:`VES` and :term:`ERP` data and compute the 
    corresponding features through its `summary` method. Note the number of 
    ERP profiles and sounding sites must be consistent as well as the  
    coordinates at this points.  
    The best practice to have full control of the computed parameters is to  
    used the :class:`watex.methods.DCProfiling` and :class:`watex.methods.DCSounding`
    to compute the parameters of each line and site with their coordinates 
    and constraints then call the `fit` methods to read each objects. 
    
    
    Parameters 
    ----------
    stations: list or str (path-like object )
        list of station name where the drilling is expected to be located. It 
        strongly linked to the name of used to specify the center position of 
        each dipole when the survey data is collected. Each survey can have its 
        own way for numbering the positions, howewer if the station is given 
        it should be one ( presumed to be the suitable point for drilling) in 
        the survey lines. Commonly it is called the `sves` which mean at this 
        point, the DC-sounding will be operated. Be sure to provide the correct
        station to compute the electrical parameters. 
        
        It is recommed to provide the positioning of the station expected to 
        hold the drillings. However if `stations` is None, the auto-way for 
        computing electrical features should be triggered. User can also 
        provide the list of stations by hand. In that case, each station should
        numbered from 1 not 0. For instance: 
            
            *  in a survey line of 20 positions. We considered the station 13 
                as the best point to locate the drilling. Therefore the name  
                of the station should be 'S13'. In other survey line (line2)
                the second point of my survey is considered the suitable one 
                to locate my drilling. Considering the two survey lines, the 
                list of stations sould be '['S13', 'S2']
                
            * `stations` can also be arrange in a single to be parsed which 
                refer to the string arguments. 
            
    dipole: float 
        The dipole length used during the exploration area. If `dipole` value 
        is set as keyword argument,i.e. the station name is overwritten and 
        is henceforth named according to the  value of the dipole. For instance
        for `dipole` equals to ``10m``, the first station should be ``S00``, 
        the second ``S10`` , the third ``S20`` and so on. However, it is 
        recommend to name the station using counting numbers rather than using 
        the dipole  position.
        
    auto: bool 
        Auto dectect the best conductive zone. If ``True``, the station 
        position should be  the  `station` of the lower resistivity value 
        in |ERP|. 
        

    read_sheets: bool, 
        Read the data in sheets. Here its assumes the data  of each survey 
        lines are arrange in a single excell worksheets. Note that if 
        `read_sheets` is set to ``True`` and the file is not in excell format, 
        a TypError will raise. 
        
    search: float , list of float
        The collection of the depth in meters from which one expects to find a 
        fracture zone outside of pollutions. Indeed, the `search` parameter is 
        used to speculate about the expected groundwater in the fractured rocks 
        under the average level of water inrush in a specific area. For 
        instance in `Bagoue region`_ , the average depth of water inrush 
        is around ``45m``.So the `search` can be specified via the water inrush 
        average value. 
        
    rho0: float 
        Value of the starting resistivity model. If ``None``, `rho0` should be
        the half minumm value of the apparent resistivity  collected. Units is
        in Ω.m not log10(Ω.m)
        
    h0: float 
        Thickness  in meter of the first layers in meters.If ``None``, it 
        should be the minimum thickess as possible ``1.m`` . 
    
    strategy: str 
        Type of inversion scheme. The defaut is Hybrid Monte Carlo (HMC) known
        as ``HMCMC``. Another scheme is Bayesian neural network approach (``BNN``). 
        
    vesorder: int 
        The index to retrieve the resistivity data of a specific sounding point.
        Sometimes the sounding data are composed of the different sounding 
        values collected in the same survey area into different |ERP| line.
        For instance:
            
            +------+------+----+----+----+----+----+
            | AB/2 | MN/2 |SE1 | SE2| SE3| ...|SEn |
            +------+------+----+----+----+----+----+
            
        Where `SE` are the electrical sounding data values  and `n` is the 
        number of the sounding points selected. `SE1`, `SE2` and `SE3` are 
        three  points selected for |VES| i.e. 3 sounding points carried out 
        either in the same |ERP| or somewhere else. These sounding data are 
        the resistivity data with a  specific numbers. Commonly the number 
        are randomly chosen. It does not refer to the expected best fracture
        zone selected after the prior-interpretation. After transformation 
        via the function :func:`~watex.utils.coreutils.vesSelector`, the header  
        of the data should hold the `resistivity`. For instance, refering to 
        the table above, the data should be:
            
            +----+----+-------------+-------------+-------------+-----+
            | AB | MN |resistivity  | resistivity | resistivity | ... |
            +----+----+-------------+-------------+-------------+-----+
        
        Therefore, the `vesorder` is used to select the specific resistivity
        values i.e. select the corresponding sounding number  of the |VES| 
        expecting to locate the drilling operations or for computation. For 
        esample, `vesorder`=1 should figure out: 
            
            +------+------+----+--------+----+----+------------+
            | AB/2 | MN/2 |SE2 |  -->   | AB | MN |resistivity |
            +------+------+----+--------+----+----+------------+
        
        If `vesorder` is ``None`` and the number of sounding curves are more 
        than one, by default the first sounding curve is selected ie 
        `rhoaIndex` equals to ``0``
        
    typeofop: str 
        Type of operation to apply  to the resistivity 
        values `rhoa` of the duplicated spacing points `AB`. The *default* 
        operation is ``mean``. Sometimes at the potential electrodes ( `MN` ),the 
        measurement of `AB` are collected twice after modifying the distance
        of `MN` a bit. At this point, two or many resistivity values are 
        targetted to the same distance `AB`  (`AB` still remains unchangeable 
        while while `MN` is changed). So the operation consists whether to the 
        average ( ``mean`` ) resistiviy values or to take the ``median`` values
        or to ``leaveOneOut`` (i.e. keep one value of resistivity among the 
        different values collected at the same point `AB` ) at the same spacing 
        `AB`. Note that for the ``LeaveOneOut``, the selected 
        resistivity value is randomly chosen.
        
    objective: str 
        Type operation to output. By default, the function outputs the value
        of pseudo-area in :math:`$ohm.m^2$`. However, for plotting purpose by
        setting the argument to ``view``, its gives an alternatively outputs of
        X and Y, recomputed and projected as weel as the X and Y values of the
        expected fractured zone. Where X is the AB dipole spacing when imaging 
        to the depth and Y is the apparent resistivity computed.

    fit_params: dict 
         Additional |ERP| keywords arguments  
         
    Attributes 
    -----------
    rtable_: pd.DataFrame 
       :term:`ERP` table that contains the different parameters computed at 
       the selected drilling points `sves`. 
       
    vtable_: pd.DataFrame 
        :term:`VES` table that contains the different parameters computed at 
        the selected drilling points. 
        
    table_:  pd.DataFrame 
       The complete table that contains :term:`VES` and term`ERP` data
       composing the DC-Features. 
       
    Examples
    ---------
    >>> import watex as wx 
    >>> from watex.methods.erp import DCMagic
    >>> erp_data = wx.make_erp ( seed =33 ).frame  
    >>> ves_data = wx.make_ves (seed =42).frame 
    >>> v = wx.DCSounding ().fit(wx.make_ves (seed =10, as_frame =True, add_xy =True))
    >>> r = wx.DCProfiling().fit( wx.make_erp ( seed =77 , as_frame =True))
    >>> res= wx.methods.ResistivityProfiling(station='S4').fit(erp_data) 
    >>> ves= wx.methods.VerticalSounding(search=60).fit(ves_data)
    dc-ves  : 100%|################################| 1/1 [00:00<00:00, 111.13B/s]
    dc-erp  : 100%|################################| 1/1 [00:00<00:00, 196.77B/s]
    >>> m = DCMagic().fit(erp_data, ves_data, v, r, ves, res ) 
    dc-erp  : 100%|################################| 2/2 [00:00<00:00, 307.40B/s]
    dc-o:erp: 100%|################################| 1/1 [00:00<00:00, 499.74B/s]
    dc-ves  : 100%|################################| 2/2 [00:00<00:00, 222.16B/s]
    dc-o:ves: 100%|################################| 1/1 [00:00<00:00, 997.46B/s]
    >>> m.summary(keep_params =True)
        longitude  latitude shape  ...       sfi  sves_resistivity  ohmic_area
    0         NaN       NaN     W  ...  1.310417        707.609756  263.213572
    1         NaN       NaN     K  ...  1.300024          1.000000  964.034554
    2  109.332932  28.41193     U  ...  1.184614          1.000000  276.340744
    """
    
    def __init__ (self, 
        stations: List[str]= None,
        dipole: float = 10.,
        auto: bool = False,
        read_sheets:bool=False, 
        force:bool=False,
        search:float=45.,
        rho0:float=None, 
        h0 :float=1., 
        strategy:str='HMCMC',
        vesorder:int=None, 
        typeofop:str='mean',
        objective: Optional[str] = 'coverall',
        **kws
        ):
        super().__init__(**kws)

        self.stations=stations 
        self.dipole=dipole 
        self.auto=auto 
        self.read_sheets=read_sheets
        self.search=search 
        self.vesorder=vesorder 
        self.typeofop=typeofop
        self.objective=objective 
        self.rho0=rho0, 
        self.h0=h0
        self.strategy=strategy 
        self.read_sheets= read_sheets

    def fit ( self, *data,  **fit_params ): 
        """ Fit the DC- electrical profiling and sounding objects.  
       
        Fit the |ERP| and |VES| curves and computed the DC-parameters. 
       
        Parameters 
        -----------
        data:  list of path-like object, or DataFrames
           When reading the |VES| objects , data should be in ([D|F|P-types]).
           The string argument is a path-like object. It must be a valid file
           wich encompasses the collected data on the field. It shoud be
           composed for |VES|, spacing values `AB` and  the apparent resistivity 
           values `rhoa`. By convention `AB` is half-space data i.e `AB/2`. 
           So, if `data` is given, params `AB` and `rhoa` should be kept to
           ``None``. If `AB` and `rhoa` is expected to be inputted, user must
           set the `data`  to ``None`` values for API purpose. If not an error
           will raise. Or the recommended way is to use the `vesSelector` tool
           in :func:`watex.utils.vesSelector` to buid the |VES| data before 
           feeding it to the algorithm. See the example below.
           
        fit_params: dict 
           Does nothing here, just for API purpose. 
    
        Returns 
        -------
        self: `DCMAgic` instanced object for chaining method. 
        """ 
        def format_invalid_data(data, kind = 'Unrecognized objects'): 
            fmterp_text = '|{0:<7}|{1:>45}|{2:>12}|{3:>14}|' 
            print()
            print('+'*83)
            print(fmterp_text.format(
                'N°', f'{kind}: {len(data):02}', 'Type',  'status'))
            print('+'*83)
            
            if len(data)==0: 
                print("|{:^81}|".format(" There is NO invalid data ")) 
                print('-'*83)
                return 
            for ii, d in enumerate( data): 
                if hasattr (d, '__module__'): 
                    if hasattr ( d, '__qualname__'): 
                        d = f"{d.__module__}.{d.__qualname__}"
                    else: d = get_estimator_name( d )
                print(fmterp_text.format(
                    ii+1, d, type (d).__name__,'*Failed') )
                
            print('-'*83)
            print()
   
        erp_data, ves_data , self.isnotvalid_ =  _parse_dc_data(
            *data , vesorder = self.vesorder )
    
        # get the doc objects if exist in the data 
        self.rtable_  = None ; self.vtable_ = None 
        if len(erp_data)!=0:
            dcp, _, erp_data = get_dc_objects(
                *erp_data , return_diff= True )

            if len(erp_data)!=0: 
                # if DC ovbjects is given 
                # dont need to fit again 
                po= DCProfiling(stations= self.stations, 
                                 dipole = self.dipole, 
                                 auto = self.auto , 
                                 read_sheets= self.read_sheets , 
                                 keep_params = False , 
                                 force = True 
                                 ) 
                po.fit(*erp_data )
                self.rtable_= po.summary(return_table= True ) 
                
            if len(dcp)!=0:
                self._make_table_if (*dcp ) 

        if len(ves_data)!=0: 
            # check whether there are some DC Sounding object 
            # inside the collection 
            _, dcv , ves_data = get_dc_objects(
                *ves_data ,return_diff= True, method ='DCSounding') 
            if len(ves_data)!=0: 
                    # expect it will read Ves data 
                vo= DCSounding(search= self.search , 
                                rho0= self.rho0, 
                                h0= self.h0, 
                                vesorder = self.vesorder, 
                                read_sheets = self.read_sheets , 
                                typeofop=self.typeofop , 
                                objective= self.objective ,
                                keep_params= False, 
                                    ) 
                vo.fit(*ves_data )                
                self.vtable_= vo.summary(return_table= True ) 
            
            if len(dcv)!=0:
                self._make_table_if (
                    *dcv, method = 'DCSounding', fit_attr='nareas_' ) 

        if self.verbose: 
            format_invalid_data (self.isnotvalid_)

        # resert tables index 
        self._reset_table_index ()
        
        return self 
    
    def summary (
        self, 
        *, 
        force=False, 
        coerce=False, 
        return_table=True, 
        keep_params=False, 
        work_as=None, 
        ): 
        """ Retrieve sites details and aggregate the table to 
        compose unique :term:`DC` features. 
        
        Parameters 
        -----------
        force: bool, default=True
          In principle, number of profiles should be equals to number of sites
          where the drilling operations is perfomed. ``Force`` allows to 
          aggregate the dataframe even this condition is not met, otherwise, 
          an error raises. 
          
        coerce: bool, default=True 
          If coordinates data of sites are  missing in a profile/site, 
          setting ``coerce`` to ``True`` will use the |ERP| coordinates 
          by defaults.  
            
        force: bool, default=False, 
          By default, :class:`DCProfiling` expects users to provide either DC 
          objects or pandas dataframe. This assumes users have already 
          transformed its data from sheets to data frame. If not the case, setting
          `force` to ``True`` constraints the algorithm to do the both tasks at
          once. 
           
        return_tables: bool, default=True, 
          Returns DC-features in a pandas dataframe. 
          
        keep_params: bool, default=False, 
            If ``True`` , keeps only the predicted parameters in the summary 
            table, otherwise, returns returns all main DC-resistivity details 
            of the site. 
            
        work_as: str, Optional 
           Can be ['ERP' | 'VES']. When one of DC-methods such as :term:`VES`
           or :term:`ERP` is not supplied. `summary` methods of `DCMagic` 
           returns an   `DCError` because `DCMagic` expects each sounding 
           point to have its profiling data. However to work like `DCSounding`
           and `DCProfiling` in order to return the table of VES or ERP, 
           the parameter `work_as` can be turn to `ERP` or `VES`.
           
        Returns 
        --------
        self or table_: :class:`~.DCMagic` or class:`pd.DataFrame` 
          Returns DCMagic object or dataframe. 
        
        """
        #xxxxxxxxxxxxxxxxxxx
        self.inspect 
        emsg =("Number of profiles and sites must be consistent. Got"
               " '{0}' and '{1}' respectively. Indeed, each sounding"
               " point is expected to be located in each individual"
               " profile therefore the coordinates of sounding site"
               " should fit the one used for positionning the drilling."
               " When using different coordinates, it might lead to"
               " unexpected results. To force performing a cross"
               " merge, set parameter ``force=True`` or ``coerce=True`` to"
               " truncate the data to fit the number of sounding points."
               " Note that this is not recommended and will probably lead "
               " to a bad DC-features arrangement. Use at your own risk.")
        
        main_params = ('longitude', 'latitude', 'shape', 'type', 'magnitude', 
                       'power',  'sfi', 'sves_resistivity', 'ohmic_area')
        #xxxxxxxxxxxxxxxxxxxx
 
        if ( 
                self.rtable_ is None 
                or self.vtable_ is None
            ): 
            # behave like DCProfiling or DCSounding
            # 'need' is used to indicate which kind of methods, 
            # DCMagic must work as. 
            need=None 
            
            if  self.rtable_ is None:
    
                if str(work_as).lower().strip() =='none':
                    raise DCError ('.'.join(emsg.split('.')[:2]).format(
                        0, len(self.vtable_) ) + 
                        (". Missing profiling data that fits the numberof"
                         f" each sounding points({len(self.vtable_)}). Use"
                         " `watex.DCProfiling` for dealing with profilings"
                         " data or set ``work_as='ves'``")
                                   )
                elif ( 
                        str(work_as).lower().strip() .find ('ves')>=0 
                        or str(work_as).lower().strip() .find('dcs')>=0
                        ): 
                    return self.vtable_ 
                need = 'ves'
            elif self.vtable_ is None:  
                if str(work_as).lower().strip() =='none':
                    raise DCError ('.'.join(emsg.split('.')[:2]).format(
                        len(self.rtable_), 0 ) + 
                        (". Missing vertical sounding data that fits the number"
                        f" of sites of the profiling ({len(self.rtable_)}). Use"
                        " `watex.DCSounding` for dealing with profilings data"
                        " or set ``work_as='erp'``")
                                   )
                elif ( 
                        str(work_as).lower().strip() .find ('erp')>=0 
                        or str(work_as).lower().strip() .find('dcp')>=0
                        ) : 
                    return self.rtable_ 
                need ='erp'
                
            raise ValueError(
                "`work_as` expects arguments {0!r}. Got {1!r}".format(
                    need if need  else "'ves' or 'erp'", work_as)
                )
            
        # check whether the coordinates exist in both 
        # tables. If not the case, coerce instead if set to 
        # True 
        for d , name in zip ( ( self.rtable_ , self.vtable_), 
                             ('dc-erp', 'dc-ves')) : 
            
            xy_coords, _, xynames  = get_xy_coordinates(
                d, as_frame =True, raise_exception='mute' )
            
            if xy_coords is None: 
                if name =='dc-erp': 
                    raise DCError("Missing sounding coordinates in the DC "
                                  "profiling data. Please specify the coordinates"
                                  " of station positions in DC readable data"
                                  " formats (D|F|P-types)"
                                  )
                elif  name =='dc-ves' and not coerce: 
                    msg = ("Missing sounding coordinates in VES. We assume each"
                           " sounding point 'sves_' from  profiling fits the"
                           " location where the drill is expected to be"
                           " performed. The ERP coordinates should be used."
                           " To avoid such behavior turn off  ``coerce=False``."
                           )
                    warnings.warn (msg)

      
        if len(self.vtable_ ) != len(self.rtable_): 
            if force: 
                self.table_ =  pd.merge (
                    self.rtable_ , self.vtable_, on =['longitude', 'latitude'] , 
                                         how ='outer')
            elif coerce: 
                # take the small length of tables
                
                sm_tab , tab_to = ( self.rtable_ , self.vtable_) if len(
                    self.rtable_) < len(self.vtable_) else (
                        (self.vtable_, self.rtable_) ) 
                
                trunc_rtab = tab_to.iloc [:len(sm_tab), :  ] 

                self.table_ = pd.concat (
                    [   # discarded the ERP and takes the VES coordinates
                       trunc_rtab[['longitude', 'latitude']].reset_index(), 
                       sm_tab.drop ( columns = ['longitude', 'latitude']
                                          ).reset_index (), 
                       trunc_rtab.drop ( columns = ['longitude', 'latitude']
                                          ).reset_index (), 
                     ]
                    , axis =1 
                    )
                self.table_.drop(columns ='index', inplace =True ) 
                
                if self.verbose : 
                    warnings.warn("Sites and profiles are not consistent."
                                  " `coerce` will truncate the rows equals"
                                  f" to {len(tab_to)} to fit the valid"
                                  f" {len(trunc_rtab)} positionning sites.")
                    
            else: raise DCError (emsg.format(
                len(self.rtable_), len(self.vtable_) ))
            
        else: 
            if coerce: 
                # then add /rername coordinates points to ves.
                # using the ERP 
                self.vtable_[['longitude', 'latitude']] =  self.rtable_ [
                    ['longitude', 'latitude']]
            # validate coordinates 
            self._validate_xy_coordinates()
            self.table_ = pd.concat (
                [   # discarded the ERP and takes the VES coordinates
                    self.vtable_[['longitude', 'latitude']].reset_index(), 
                   self.rtable_.drop ( columns = ['longitude', 'latitude']
                                      ).reset_index (), 
                   self.vtable_.drop ( columns = ['longitude', 'latitude']
                                      ).reset_index (), 
                 ]
                , axis =1 
                )
            self.table_.drop(columns ='index', inplace =True ) 

        if keep_params: 
            self.table_= self.table_[list(main_params)] 
            
        return self.table_ if return_table else self 
    
    def _validate_xy_coordinates (self): 
        """ Validate whether coordinates are identical or are consistent."""
        
        incons_mgs = ("Inconsistent coordinates founds in {} sites."
                      "ERP coordinates are discarded instead")
        # convert longtude type to float for consistency 
        self.rtable_ [['longitude', 'latitude']]= self.rtable_ [[
            'longitude', 'latitude']].astype (float)
        self.vtable_ [['longitude', 'latitude']] = self.vtable_ [
            ['longitude', 'latitude']].astype (float) 
        
        def replace_to ( *, action ='forward'): 
            """ Replace NaN value in coordinate by 0. and vice versa """
            # in the array from ERP , if there is 0. Considers as NaN 
            # and reverse back 
            
            self.rtable_[['longitude', 'latitude']]=self.rtable_[
                ['longitude', 'latitude']].replace(
                np.nan if action=='forward' else 0.  ,
                0.if action=='forward' else np.nan  )
            self.vtable_[['longitude', 'latitude']]=self.vtable_[
                ['longitude', 'latitude']].replace(
                np.nan if action=='forward' else 0.  ,
                0.if action=='forward' else np.nan  )
        # Replace NaN to 0. 
        replace_to()
        
        # make the differences if equals to zero 
        
        ar = self.rtable_ [['longitude', 'latitude']].values - self.vtable_ [
            ['longitude', 'latitude']].values
        
        if np.any(ar): 
            # get index where there is inconsistent coordinates 
            # numpy.flatnonzero(x == 0) when working with 1d array 
            # to find zero 
            incons_indices = list( np.nonzero ( ar )[0]) 
            incons_mgs =incons_mgs.format(len(incons_indices))
 
            # get the elements.
            if self.verbose: 
                items = ["line{0}/site{0}".format(i+1) for i in 
                         incons_indices ] 
                b, e = incons_mgs.split('.')
                listing_items_format(
                    items, b ,  endtext=e +".", inline =True )  
            # else :  
            #     warnings.warn(incons_mgs)
            # discarded the lonlat from ERP to VES 

        # Replace back  0. to NaN 
        replace_to (action= 'reverse') 

        return self 
  
    def _make_table_if (self, *dc , method = 'DCProfling', 
                        fit_attr ='sves_'): 
        """ Build object if DCProfiling or Sounding is passed
        
        Parameters 
        ------------
        dc: list 
           A collection of DC objects 
        method: str  
           DC method ['DCSounding' | 'DCProfiling']
        fit_attr: str, 
           Fitted attribute to check whether object is fitted yet. 
        
        """ 
        def tables (o,  table = None ):
            """ get table if summary method is not fitted yet or 
            concatenate table one to onother"""
            if not hasattr ( do, 'table_'): 
                o.summary(return_table = False )  
               
            if table is not None: 
                table = pd.concat ( [ table , o.table_ ], axis = 0)
                
            else : table = do.table_ 
            
            return table 
        
        msg = ("{!r} object is detected while it is not fitted yet. You must"
               " fit the object or remove the DC object from the collection"
               " and simply pass the corresponding data as frame or a pathlike"
               " object.")
        # set tqdm 
        try: 
            import tqdm 
            pbar =  tqdm.tqdm(dc ,ascii=True, unit='B', 
                              desc ="dc-o" + (
                                  ':erp' if fit_attr=='sves_' else ':ves'), 
                              ncols =77)
            has_tqdm =True 
        except NameError: 
            # force pbar to hold the data value
            # if trouble occurs 
            pbar =dc
            has_tqdm=False 
            
        if len(dc) !=0:  
            for ii, do in enumerate ( pbar) : 
                if not hasattr (do, fit_attr): 
                    raise NotFittedError(msg.format(method))
                    
                if not hasattr ( do, 'table_'): 
                    do.summary(return_table = False ) 
                    
                # aggregate table with r_table if exists 
                if get_estimator_name (do)== "DCProfiling": # --> ERP methods 
                    self.rtable_ = tables ( do, self.rtable_ ) 
                    
                if get_estimator_name(do)=="DCSounding": 
                    self.vtable_ = tables ( do, self.vtable_)

                pbar.update(ii)  if has_tqdm else None 
                
        return self 
    
    def _reset_table_index ( self ): 
        """ reset DC profiling and sounding tables indexes """
        
        if self.rtable_ is not None: 
            self.rtable_.index = [f'line{k+1}' for k in range (
                len(self.rtable_))]
            
        if self.vtable_ is not None: 
            self.vtable_.index = [f'site{k+1}' for k in range (
                len(self.vtable_))]
            
    def __repr__(self):
        """ Pretty format for developers following the API... """
        return repr_callable_obj(self)
       
    def __getattr__(self, name):
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        appender  = "" if rv is None else f'. Do you mean {rv!r}'
        
        if name =='table_': 
            err_msg =(". Call 'summary' method to fetch attribute 'table_'")
        else: err_msg =  f'{appender}{"" if rv is None else "?"}' 
        
        raise AttributeError (
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{err_msg}'
            )

    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ("{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if ( 
                not hasattr (self, 'rtable_') or not hasattr (self, 'vtable_')
        ): 
            raise NotFittedError(msg.format(obj=self)
            )
        return 1    
    
def _parse_dc_data (*data,vesorder =0  ): 
    """ Select ERP and VES data """ 
    
    erp_data = [] 
    ves_data = []
    unreadf= []
    # check wether a path-like object is supplied 
    # then collect all the files and check whether 
    # there are some ERP and VES data 
    dtemp =[] # collect all files in different path. 
    d0=[]
    for d in data : 
        if isinstance ( d, str ) and os.path.isdir ( d ): 
            # collect all the data 
            dtemp += [ os.path.join (d, f ) for f in os.listdir ( d )] 
        else: d0.append (d ) 
        
    data = d0 + dtemp # new data
    
    for d in data : 
        if isinstance ( d, str ): 
            # assume there is a file object 
            # check instead 
            if os.path.isfile (d ):
                try: 
                    d = _is_readable(d, input_name= f'Data {d!r}')
                except : 
                    unreadf.append (os.path.basename (d) ) 
                    continue 
               
        if hasattr ( d, '__array__') and hasattr ( d, 'columns'): 
            # assume a pandas data frame is given 
            # check instead 
            try : 
                ves_data.append ( vesSelector (d, index_rho= vesorder , 
                                 input_name= f'Data {d!r}'))
               # is_valid_dc_data(d, method ='ves')
            except : 
                try : 
                    erp_data.append (erpSelector (d , force= True ) ) 
                    #is_valid_dc_data(d )
                except : 
                    # save the file name and skip 
                    unreadf.append (
                        "Frame: [{d.shape[0]} rows x {d.shape[1]} columns]") 
                    continue 
                
        # Assume DC object is given 
        elif  get_estimator_name ( d) in (
                'DCSounding', 'VerticalSounding') : 
            ves_data.append ( d ) 
            
        elif  get_estimator_name ( d) in (
                'DCProfiling', 'ResistivityProfiling'): 
            erp_data.append ( d ) 

        else: 
            # if nothing fit VES or ERP 
            # collect d instead as invalid data. 
            unreadf.append (d ) 

    return erp_data, ves_data , unreadf 

def get_dc_objects (
        *data, method = 'DCProfiling', return_diff=False, **kws  ): 
    """ Get DC Profiling and DCSounding objects if exists in the whole 
    valid data.
    
    Parameters 
    -----------
    data: list 
       Collection of DC objects 
       
    method: str, 
       Method of selection. 
    return_diff: bool, default=False, 
       Retuns the remain objects which are not DCProfiling or DCSounding 
       
    Returns
    --------
    dcp, dcv, remain_data: Tuple of list 
    
       A collection of selected DC Profiling and DCSounding objects 
       
     
    """ 
    
    dcp=[]; dcv =[]
    if method == 'DCProfiling': 
        dcp = list(filter ( lambda o : get_estimator_name (
            o)== 'DCProfiling', data )
                   ) 
    elif method =='DCSounding': 
        dcv =  list(filter ( 
            lambda o : get_estimator_name(o) =='DCSounding', data )
            )
    
    if return_diff : 
        # get the remain data which are not a 
        # DCObjects 
        remain_data = list(filter ( lambda o: get_estimator_name(o) not in (
           ( "DCProfiling", "DCSounding")), data )
            ) 

    return dcp, dcv,  remain_data 

@deprecated("Deprecated ERP class and should be removed in the next release." 
            " Use 'watex.methods.DCMagic' instead.")
class ERPCollection: 
    """
    Collection objects. The class collects all `erp` survey lines.
    Each `erp` is an singleton class object with their corresponding 
    attributes. The goal is to build a container  geao-elecricals to 
    straigthforwardly given to :class:`watex.bases.features.GeoFeatures`
    class.
    
    Parameters 
    ------------
    listOferpfn: list, ndarray
        list of different `erp` files.
            
    listOfposMinMax : list 
        collection of different selected anomaly boundaries. 
        If not provided, the :attr:`~.methods.erp.ERP.auto` 
        will triggered. It's recommanded to provided for all 
        `erp` your convenient anomaly boundaries like:: 
        
            listOfposMinMax=[(90, 130), (10, 70), ...]
    
        where ``(90,130)``is boundaries of selected anomaly on 
        the first `erp` line and ``(10,70)`` is the boundaries
        of the second `erp` survey line and so on. 
            
    erpObjs: list, ndarray 
        Collection of objects from :class:`~.methods.erp.ERP`. If objects 
        are alread created. Gather them on a list and pass though the 
        argument `erpObjs`.
    
    Holds others optionals infos passed as  keyword arguments. 
    
    ======================  =============   ===================================
    Attributes              Type                Description  
    ======================  =============   ===================================
    list_of_dipole_lengths  list            Collection of `dipoleLength`. User 
                                            can provide the distance between 
                                            sites measurements as performed on 
                                            investigations site. If given, the 
                                            automaticall `dipoleLength` 
                                            computation will be turned off. 
    fnames                  array_like      Array of `erp` survey lines name. 
                                            If each survey name is the location 
                                            name then will keep it. 
    id                      array_like      Each `erp` obj reference numbers
    erps_data               nd.array        Array composed of geo-electrical
                                            parameters. ndarray(nerp, 8) where 
                                            num is the number of `erp`obj
                                            collected. 
    erpdf                   pd.DataFrame    A dataFrame of collected `erp` line 
                                            and the number of lines correspond 
                                            to the number of collected `erp`.
    ======================  =============   ===================================
    
    It's posible to get from each `erp` collection the singular array of 
    different parameters considered as properties:: 
    
        >>> from watex.methods.erp import ERP_collection as ERPC
        >>> erpcol = ERPC(listOferpfn='list|path|filename')
        >>> erpcol.survey_ids
        >>> erpcol.selectedPoints
    
    List of the :class:`ERP_collection` attribute properties:
    
    ====================   ==============   ===================================
    Properties              Type                Description  
    ====================   ==============   ===================================
    selectedPoints          array_like      Collection of Best anomaly 
                                            position points. 
    survey_ids              array_like      Collection of all `erp` survey 
                                            survey ids. Note that each ids is
                                            following by the prefix **e**.
    sfis                    array_like      Collection of best anomaly standard 
                                            fracturation index value. 
    powers                  array_like      Collection of best anomaly `power`
    magnitudes              array_like      Colection of best anomaly
                                            magnitude in *ohm.m*.
    shapes                  array_like      Collection of best anomaly shape. 
                                            For more details please refer to
                                            :doc:`ERP`.
    types                   array_like      Collection of best anomaly type. 
                                            Refer to :doc:`ERP` for
                                            more details.
    ====================   ==============   ===================================
    
    Examples
    ---------
    >>> from watex.methods.erp import ERPCollection 
    >>> erpObjs =ERP_collection(listOferpfn= 'data/erp')
    >>> erpObjs.erpdf
    >>> erpObjs.survey_ids
    ... ['e2059734331848' 'e2059734099144' 'e2059734345608']
    
    """
    erpColums =['id', 
                'east', 
                'north', 
                'power', 
                'magnitude', 
                'shape', 
                'type', 
                'sfi']
    
    def __init__(self, listOferpfn=None, listOfposMinMax=None, erpObjs=None,
                  **kws): 
        self._logging =watexlog().get_watex_logger(self.__class__.__name__)
        
        self.erpObjs = erpObjs 
        self.anomBoundariesObj= listOfposMinMax
        self.dipoleLengthObjs= kws.pop('list_of_dipole_lengths', None)
        self.export_data =kws.pop('export', False)
        self.export_fex= kws.pop('extension', 'csv')
        self.verbose =kws.pop('verbose', False)

        self.listOferpfn =listOferpfn 
        self.id =None
        
        for key in list(kws.keys()): 
            setattr(self, key, kws[key])
        
        self._readErpObjs()
        
    def _readErpObjs(self, listOferpfn=None,listOfposMinMax=None, 
                 erpObjs=None, **kwargs): 
        """
        Read or cread `erp` objects and populate attributes.
        """
        
        self._logging.info('Collecting `erp` files and populates '
                           'main attributes')
        
        dipoleLengthObjs =kwargs.pop('list_of_dipole_lengths', None)
        
        if listOferpfn is not None :
            self.listOferpfn =listOferpfn 
        if listOfposMinMax is not None : 
            self.anomBoundariesObj =listOfposMinMax 
        if erpObjs is not None : 
            self.erpObjs = erpObjs
        
        if dipoleLengthObjs is not None : 
            self.dipoleLengthObjs = dipoleLengthObjs
            
        
        if self.listOferpfn is None and self.erpObjs is None : 
            self._logging.error('No ERP file nor ERP object detected.'
                                'Please provide at least `ERP` file or '
                                ' `erp` object.')
        if self.listOferpfn is not None: 
            if isinstance(self.listOferpfn, str):
                if os.path.isfile(self.listOferpfn): 
                    self.listOferpfn=[self.listOferpfn]
                elif os.path.isdir(self.listOferpfn): 
                    self.listOferpfn=[os.path.join(self.listOferpfn,file) 
                                      for file in os.listdir (
                                              self.listOferpfn)]
                else : 
                    raise ERPError(
                        'File or path provided is wrong! Please give a'
                        ' a right path.')
                        
            if self.dipoleLengthObjs is not None : 
                assert len(self.listOferpfn)== len(
                    self.dipoleLengthObjs),'Length of dipoles lenghths is'\
                    ' = {0}. It must be equal to number of `erp line'\
                        ' provided (is ={1}).'.format(len(
                            self.dipoleLengthObjs), len(self.listOferpfn))
            else : 
                self.dipoleLengthObjs = [
                    None for ii in range(len(self.listOferpfn ))]
                

        if self.anomBoundariesObj is not None : 
            assert len(self.anomBoundariesObj)== len(self.listOferpfn ), \
                'Length of selected anomalies boundaries (is={0}) must be '\
                    'equal to the length of number of `erp` line provided '\
                        '(is={1}).'.format(len(self.anomBoundariesObj), 
                                           len(len(self.listOferpfn )))
        
        else : 
            self.anomBoundariesObj= [None for nn in range(
            len(self.listOferpfn ))]
        
        unreadfiles =[]  # collected uncesse
        fmterp_text = '|{0:<7}|{1:>45}|{2:>15}|'
        if self.erpObjs is None: 
            self.erpObjs=[]
            
            print('-'*70)
            print(fmterp_text.format('Num', 'ERPlines', 'status'))
            print('-'*70)
            
            for ii, erp_filename in enumerate(self.listOferpfn) :
                
                try : 
                    name_file = os.path.basename(
                        os.path.splitext(erp_filename)[0])
                except : 
                    # for consistency takes at least the basename
                    # to display.
                    name_file = os.path.basename(erp_filename)
                    
                try : 
   
                    erpObj = ERP(erp_fn= erp_filename,
                                dipole_length=self.dipoleLengthObjs[ii], 
                                posMinMax=self.anomBoundariesObj[ii])
                except : 
                    
                    unreadfiles.append(name_file)
                    print(fmterp_text.format(ii+1,
                                              name_file, 
                                              '*Failed'))
                    
                else: 
                    print(fmterp_text.format(ii+1,
                                              name_file, 
                                              'Passed'))
                    self.erpObjs.append(erpObj)
            print('-'*70)
             
        if len(unreadfiles)>=1 : 
            self._logging.error (
                f'Unable to read the files `{unreadfiles}`')
            warnings.warn(f'Unable to read file `{len(unreadfiles)}`.'
                          ' Please check your files.')
            
            print(' --!> {0} ERP file not read. Please check your file'
                  f' {"s" if len(unreadfiles)>1 else ""} enumerate below:'
                  .format(len(unreadfiles)))
            
            display_infos(infos=unreadfiles,
                    header=f"Unread file{'s' if len(unreadfiles)>1 else ''}")
            
        if self.erpObjs is not None and self.listOferpfn is None : 
            
            lenOrpfn = len(self.erpObjs)
            
        elif self.erpObjs is not None and self.listOferpfn is  not None :
            lenOrpfn= len(self.listOferpfn)
            
        print('-'*70)  
        print(' {0}/{1} ERP files have been succesffuly read.'.format(
            len(self.erpObjs),lenOrpfn ))
        
        print('-'*70)  
        
        # collected the ERP filenames and generated the id from each object.
        self.fnames = self.get_property_infos('_name')
        self.id = np.array([id(obj) for obj in self.fnames])
        
        # create a dataframe object
        self._logging.info('Setting and `ERP` data array '
                           'and create pd.Dataframe')
        try : 
            self.erps_data= np.c_[
                                self.survey_ids, 
                                self.easts, 
                                self.norths, 
                                self.powers, 
                                self.magnitudes, 
                                self.shapes, 
                                self.types, 
                                self.sfis]
            self.erpdf =pd.DataFrame(data = self.erps_data, 
                                      columns=self.erpColums) 
                                      
            self.erpdf=self.erpdf.astype( {'east':np.float, 
                                            'north': np.float, 
                                            'power': np.float, 
                                            'magnitude':np.float, 
                                            'sfi':np.float})
            
            if self.export_data is True : 
                self.exportErp()
        except : pass 
            
    def get_property_infos(self, attr_name , objslist =None): 
        """
        From each obj `erp` ,get the attribute infos and set 
        on data array 
         
        :param attr_name: 
            Name of attribute to get the informations of the properties. 
             
        :type attra_name: str 
        
        :param objslist: list of collection objects. 
        :type objslist: list 
        
        :Example:
            >>> from watex.methods.erp.ERPCollection as ERPcol
            >>> erpObjs =ERPcol(listOferpfn= 'data/erp', 
            ...                export_erpFeatures=True,
            ...                    filename='ykroS')
            
        """
        
        if objslist is not None : 
            self.erpObjs = objslist 
        
        return np.array([getattr(obj, attr_name) for obj in self.erpObjs ])
        
    def exportErp(self, extension_file=None, savepath =None, **kwargs ):
        """
        Export `erp` data after computing different geo_electrical features.
        
        :param extension_file: 
            Extension type to export the files. Can be ``xlsx`` or ``csv``. 
            The default   `extension_file` is ``csv``. 
        :type extension_file: str 
        
        :param savepath: Path like string to save the output file.
        :type savepath: str 
    
        """
        
        filename = kwargs.pop('filename', None)
        if filename is not None : 
            self.filename =filename
        if extension_file is not None : 
            self.export_fex = extension_file 
        if  savepath is not None :
            self.savepath = savepath 
            
        if self.export_fex.find('csv') <0 and self.export_fex.find('xlsx') <0: 
            self.export_fex ='.csv'
        self.export_fex= self.export_fex.replace('.', '')
        
        erp_time = '{0}_{1}'.format(datetime.datetime.now().date(), 
                                    datetime.datetime.now().time())
        
        # check whether `savepath` and `filename` attributes are set.
        for addf in ['savepath', 'filename']: 
            if not hasattr(self, addf): 
                setattr(self, addf, None)
                
        if self.filename is None : 
            self.filename = 'erpdf-{0}'.format(
                erp_time + '.'+ self.export_fex).replace(':','-')
        elif self.filename is not None :
            self.filename += '.'+ self.export_fex
            
        # add name into the workbooks
        exportdf = self.erpdf.copy() 
        
        exportdf.insert(loc=1, column='name', value =self.fnames )
        exportdf.reset_index(inplace =True)
        exportdf.insert(loc=0, column='num', value =exportdf['index']+1  )
        exportdf.drop(['id', 'index'], axis =1 , inplace=True)
        
        if self.export_fex =='xlsx':

           # with pd.ExcelWriter(self.filename ) as writer: 
           #      exportdf.to_excel(writer, index=False, sheet_name='data')
                
           exportdf.to_excel(self.filename , sheet_name='data',
                            index=False) 
           
        elif self.export_fex =='csv': 
            
            exportdf.to_csv(self.filename, header=True,
                              index =False)

        if self.savepath is None :
            self.savepath = savepath_('_erpData_')
            
        if self.savepath is not None :
            if not os.path.isdir(self.savepath): 
                self.savepath = savepath_('_erpData_')
            try : 
                shutil.move(os.path.join(os.getcwd(),self.filename) ,
                        os.path.join(self.savepath , self.filename))
            except : 
                self._logging.debug("We don't find any path to save ERP data.")
            else: 
                print('--> ERP features file <{0}> is well exported to {1}'.
                      format(self.filename, self.savepath))
        
                
    @property 
    def survey_ids (self) : 
        """Get the `erp` filenames """
        return np.array(['e{}'.format(idd) for idd in self.id])
    
    @property 
    def selectedPoints (self): 
        """Keep on array the best selected anomaly points"""
        return self.get_property_infos('select_best_point_')
    @property
    def powers(self):
        """ Get the `power` of select anomaly from `erp`"""
        return self.get_property_infos('best_power')
    
    @property
    def magnitudes(self):
        """ Get the `magnitudes` of select anomaly from `erp`"""
        return self.get_property_infos('best_magnitude')
    
    @property 
    def shapes (self):
        """ Get the `shape` of the selected anomaly. """
        return self.get_property_infos('best_shape')
    @property 
    def types(self): 
        """ Collect selected anomalies types from `erp`."""
        return self.get_property_infos('best_type')
    @property 
    def sfis (self): 
        """Collect `sfi` for selected anomaly points """
        return self.get_property_infos('best_sfi')
    
    @property 
    def easts(self): 
        """Collect the utm_easting value from `erp` survey line. """
        return self.get_property_infos('best_east')
    
    @property 
    def norths(self): 
        """Collect the utm_northing value from `erp` survey line. """
        return self.get_property_infos('best_north')  
            

class ERP : 
    """
    Electrical resistivity profiling class . Define anomalies and compute
    its features. Can select multiples anomalies  on ERP and give their
    features values. 
    
    Arguments 
    ----------
    * erp_fn: str 
        Path to electrical resistivity profile 
           
    * dipole_length: float
        Measurement electrodes. Distance between two electrodes in 
        meters. 
    * auto: bool 
        Trigger the automatic computation . If the `auto` is set to 
        ``True``, dont need to provide the `posMinMax` argument
        otherwise `posMinMax` must be given. 
    * posMinMax: tuple, list, nd.array(1,2)
        Selected anomaly boundary. The boundaries matches the startpoint 
        as the begining of anomaly position and the endpoint as the end 
        of anomaly position. If provided , `auto` will be turn off at
        ``False`` even ``True``. 
                    
    Notes 
    ------
    Provide the `posMinMax` is strongly recommended for accurate 
    geo-electrical features computation. If not given, the best anomaly 
    will be selected automatically and probably could not match what you 
    expect.

            
    Hold others informations: 
        
    =================  ===================  ===================================
    Attributes              Type                Description  
    =================  ===================  ===================================
    lat                 float               sation latitude 
    lon                 float               station longitude 
    elev                float               station elevantion 
                                            in m or ft 
    east                float               station easting coordinate (m)
    north               float               station northing coordinate (m)
    azim                float               station azimuth in meter (m)
    utm_zone            str                 UTM location zone 
    resistivity         dict                resistivity value at each
                                            station (ohm.m)
    name                str                 survey location name 
    turn_on             bool                turn on/off the displaying computa-
                                            tion parameters. 
    best_point          float/int           position of the selected anomaly
    best_rhoa           float               selected anomaly app.resistivity 
    display_autoinfos   bool                display the selected three best 
                                            anomaly  points selected automatic-
                                            cally. 
    =================  ===================  ===================================

    - To get the geo-electrical-features,  create an `erp` object by calling:: 
        
        >>> from watex.methods.erp import ERP 
        >>> anomaly_obj =ERP(erp_fn = '~/location_filename')
        
    The call of the following `erp` properties attributes:
    
    ====================  ================  ===================================
    properties              Type                Description  
    ====================  ================  ===================================
    select_best_point_      float           Best anomaly position points 
    select_best_value_      float           Best anomaly app.resistivity value.
    best_points             float           Best positions points selected 
                                            automatically. 
    best_sfi                float           Best anomaly standart fracturation 
                                            index value. 
    best_anr                float           Best 
    best_power              float           Best anomaly power  in *meter(m)*.
    best_magnitude          float           Best anomlay magnitude in *ohm.m*
    best_shape              str             Best anomaly shape. can be ``V``, 
                                            ``W``,``K``, ``H``, ``C``, ``M``.
    best_type               str             Best anomaly type. Can be : 
                                            - ``EC`` for Extensive conductive. 
                                            - ``NC`` for narrow conductive. 
                                            - ``CP`` for conductive plane. 
                                            - ``CB2P`` for contact between two
                                            planes. 
    ====================  ================  ===================================
    
    Examples 
    ---------
        
    >>> from watex.methods.erp import ERP  
    >>> anom_obj= ERP(erp_fn = 'data/l10_gbalo.xlsx', auto=False, 
    ...                  posMinMax= (90, 130),turn_off=True)
    >>> anom_obj.name 
    ... l10_gbalo
    >>> anom_obj.select_best_point_
    ...110 
    >>> anom_obj.select_best_value_
    ...132
    >>> anom_obj.best_magnitude
    ...5
    >>> nom_obj.best_power
    ..40
    >>> anom_obj.best_sfi
    ...1.9394488747363936
    >>> anom_obj.best_anr
    ...0.5076113145430543
        
    """ 
    erpLabels =['pk', 
                'east', 
                'north', 
                'rhoa'
                ]
    
    dataType ={
                ".csv":pd.read_csv, 
                 ".xlsx":pd.read_excel,
                 ".json":pd.read_json,
                 ".html":pd.read_json,
                 ".sql" : pd.read_sql
                 }
    
    def __init__(self, erp_fn =None , dipole_length =None, auto =False, 
                 posMinMax=None, **kwargs)  : 
        """ Read :ref:`erp` file and  initilize  the following
        attributes attributes. Set `auto` to ``True`` to let the program 
        selecting the best anomaly points. """
        
        self._logging =watexlog.get_watex_logger(self.__class__.__name__)

        self.erp_fn =erp_fn 
        self._dipoleLength =dipole_length
        self.auto =auto 
        
        self.anom_boundaries = posMinMax
        self._select_best_point =kwargs.pop('best_point', None)
        self.turn_on =kwargs.pop('display', 'off')
        self._select_best_value =kwargs.pop('best_rhoa', None)
        
        self._power =None 
        self._magnitude =None 
  
        
        self._lat =None
        self._name = None 
        
        self._lon =None 
        self._east=None 
        self._north =None 
        
        self._sfi = None 
        self._type =None 
        self._shape= None 
        self.utm_zone =kwargs.pop('utm_zone', None)
        
        
        self.data=None
        
        self._fn =None 
        self._df =None 
        
        
        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])
            

        if self.erp_fn is not None : 
            self._read_erp()
            
    @property 
    def fn(self): 
        """
        ``erp`` file type 
        """
        return self._fn 
    
    @fn.setter 
    def fn(self, erp_f): 
        """ Find the type of data and call pd.Dataframe for reading. 
        numpy array data can get from Dataframe 
        
        :param erp_f: path to :ref:`erp` file
        :type erp_f: str
        
        """
        if erp_f is not None : self.erp_fn = erp_f 

        if not os.path.isfile(self.erp_fn): 
            raise FileHandlingError (
                'No right file detected ! Please provide the right path.')
        name , exT=os.path.splitext(self.erp_fn)

        if exT in self.dataType.keys(): 
            self._fn =exT 
        else: self._fn ='?'
        
        df_ = self.dataType[exT](self.erp_fn)
        # Check into the dataframe whether the souding location and anomaly 
        #boundaries are given 
        self.auto, self._shape, self._type, self._select_best_point,\
            self.anom_boundaries, self._df = \
                get_boundaries(df_)
 
        self.data =self._df.to_numpy()
        self._name = os.path.basename(name)
        
        
    def _read_erp(self, erp_fn=None ):
        """
        Read :ref:`erp` file and populate attribute 
        
        :param erp_fn: Path to electrical resistivity profile 
        :type erp_fn: str 
        
        """
        if erp_fn is not None : 
            self.erp_fn = erp_fn 

        self.fn = self.erp_fn 
        
        self.sanitize_columns()
        
        if self.coord_flag ==1 : 
            self._longitude= self.df['lon'].values 
            self._latitude = self.df['lat'].values
            easting= np.zeros_like(self._longitude)
            northing = np.zeros_like (self._latitude)

            for ii in range(len(self._longitude)):
                try : 
                    self.utm_zone, utm_easting, utm_northing = ll_to_utm(
                                            reference_ellipsoid=23, 
                                              lon=self._longitude[ii],
                                              lat = self._latitude[ii])
                except : 
                    utm_easting, utm_northing, \
                        self.utm_zone= project_point_ll2utm(
                        lon=self._longitude[ii],
                        lat = self._latitude[ii])
                    
                easting[ii] = utm_easting
                northing [ii] = utm_northing
            
            self.df.insert(loc=1, column ='east', value = easting)
            self.df.insert(loc=2, column='north', value=northing)
            
        # get informations from anomaly 
        if self.coord_flag ==0 : 
            # compute  latitude and longitude coordinates if not given 
            self._latitude = np.zeros_like(self.df['east'].values)
            self._longitude = np.zeros_like(self._latitude)
            
            if self.utm_zone is None :
                self._logging.debug("UTM zone must be provide for accurate"
                                    "location computation. We'll use `30N`"
                                    "as default value")
                warnings.warn("Please set the `UTM_zone` for accurating "
                              "`longitude` and `latitude` computing. If not"
                              " given, 30N `lon` and `lat` is used as"
                              " default value.")
                
                self.utm_zone = '30N'
            
            for ii, (north, east) in enumerate(zip(self.df['north'].values,
                                                self.df['east'].values)): 
                try : 
                    self._latitude [ii],\
                        self._longitude [ii] = utm_to_ll(23,
                            northing = north, 
                            easting = east, 
                            zone = self.utm_zone) 
                except: 
                    self._latitude[ii], \
                        self._longitude [ii] = project_point_utm2ll(
                                        northing = north, 
                                        easting = east, 
                                        utm_zone = self.utm_zone) 
                        
        if self.anom_boundaries is None or \
            None in np.array(self.anom_boundaries): 
            # for consistency enable `automatic option`
            if not self.auto : 
                self._logging.info ('Automatic trigger is set to ``False``.'
                                    " For accuracy it's better to provide "
                                    'anomaly location via its positions '
                                    'boundaries. Can be a tuple or a list of '
                                    'startpoint and endpoint.')
                self._logging.debug('Automatic option is triggered!')
                
            self.auto=True 
            
        if self.turn_on in ['off', False]: 
            self.turn_on =False 
        elif self.turn_on in ['on', True]: 
            self.turn_on =True 
        else : 
            self.turn_on =False 
        
    
        if self._dipoleLength is None : 
            self._dipoleLength=(max(self.df['pk']) - min(self.df['pk']))/(
                len(self.df['pk'])-1)
                    
   
        self.aBestInfos= select_anomaly(
                             rhoa_array= self.df['rhoa'].values, 
                             pos_array= self.df['pk'].values, 
                             auto = self.auto, 
                             dipole_length=self._dipoleLength , 
                             pos_bounds=self.anom_boundaries, 
                             pos_anomaly = self._select_best_point, 
                             display=self.turn_on
                             )
        
        self._best_keys_points = list(self.aBestInfos.keys())
        
        for ckey in self._best_keys_points : 
            if ckey.find('1_pk')>=0 : 
                self._best_key_point = ckey 
                break 
        

    def sanitize_columns(self): 
        """
        Get the columns of electrical resistivity profiling  dataframe and set
        new names according to :attr:`.ERP.erpLabels` . 
    
        """ 

        self.coord_flag=0
        columns =[ c.lower().strip() for c in self._df.columns]

        for ii, sscol in enumerate(columns): 
            try : 
            
                if re.match(r'^sta+', sscol) or re.match(r'^site+', sscol) or \
                    re.match(r'^pk+', sscol)  : 
                    columns[ii] = 'pk'
                if re.match(r'>east+', sscol) or re.match(r'^x|X+', sscol): 
                    columns[ii] = 'east'
                if re.match(r'^north+', sscol) or re.match(r'^y|Y+', sscol): 
                    columns[ii] = 'north'
                if re.match(r'^lon+', sscol): 
                    columns[ii] = 'lon'
                    self._coord_flag = 1
                if re.match(r'^lat+', sscol):
                    columns[ii] = 'lat'
                if re.match(r'^rho+', sscol) or re.match(r'^res+', sscol): 
                    columns[ii] = 'rhoa'
                    
            except KeyError: 
                print(f'keys {self.erpLabels} are not found in {sscol}')
            except: 
                self._logging.error(
                    f"Unrecognized header keys {sscol}. "
                    f"Erp keys are ={self.erpLabels}"
                      )

        self.df =pd.DataFrame(data =self.data, columns= columns)
        
    @property
    def select_best_point_(self): 
        """ Select the best anomaly points."""
        self._select_best_point_= self.aBestInfos[self._best_key_point][0]
        
        
        mes ='The best point is found  at position (pk) = {0} m. '\
            '----> Station number {1}'.format(
                self._select_best_point_,
                int(self._select_best_point_/self.dipoleLength)+1
                                              )
        wrap_infos(mes, on =self.turn_on) 
        
        return self._select_best_point_
    
    @property 
    def dipoleLength(self): 
        """Get the dipole length i.e the distance between two measurement."""
        
        wrap_infos(
            'Distance bewteen measurement is = {0} m.'.
            format(self._dipoleLength), off = self.turn_on)
        
        return self._dipoleLength
    
    @property 
    def best_points (self) : 
        """ Get the best points from auto computation """
        
        if len(self._best_keys_points)>1 : verb, pl='were','s'
        else: verb, pl='was',''
        mess =['{0} best point{1} {2} found :\n '.
               format(len(self._best_keys_points),pl,verb)] 
        self._best_points ={}
        for ii,  bp in enumerate (self._best_keys_points): 
            cods = float(bp.replace('{0}_pk'.format(ii+1), ''))
            pmes='{0:02} : position = {1} m ----> rhoa = {2} Ω.m\n '.format(
                ii+1, cods, 
                self.aBestInfos[bp][1]) 
            mess.append(pmes)
            self._best_points['{}'.format(cods)]=self.aBestInfos[bp][1]
            
        mess[-1]=mess[-1].replace('\n', '')
        
        wrap_infos(''.join([ss for ss in mess]),
                         on = self.turn_on)
        return self._best_points  
    
    @property 
    def best_power (self):
        """Get the power from the select :attr:`~.ERP.select_best_point`. """
        self._power =compute_power(
            posMinMax=self.aBestInfos[self._best_key_point][2])
        
        wrap_infos(
            'The power of selected best point is = {0}'.format(self._power),
                        on = self.turn_on)
        
        return self._power 
    @property 
    def best_magnitude(self): 
        """ Get the magnitude of the select :attr:`~.ERP.select_best_point`."""
        
        self._magnitude =compute_magnitude(
            rhoa_max=self.rhoa_max,rhoa_min=self.select_best_value_)
                                                 
        wrap_infos(
           'The magnitude of selected best point is = {0}'.
           format(self._magnitude),
          on = self.turn_on)
        
        return self._magnitude
    
    @property 
    def best_sfi(self) : 
        """Get the standard fraturation index from 
        :attr:`~.ERP.select_best_point_`"""
        
        self._sfi = compute_sfi(pk_min=self.posi_min,
                                      pk_max=self.posi_max,
                                      rhoa_min=self.rhoa_min,
                                      rhoa_max=self.rhoa_max,
                                      rhoa=self.select_best_value_, 
                                      pk=self.select_best_point_)
        
        wrap_infos('SFI computed at the selected best point is = {0}'.
                        format(self._sfi), 
                        on =self.turn_on)
        return self._sfi
    
    @property 
    def posi_max (self):
        """Get the right position of :attr:`select_best_point_ boundaries 
        using the station locations of unarbitrary positions got from
        :attr:`~.ERP.dipoleLength`."""
        
        return np.array(self.aBestInfos[self._best_key_point][2]).max()
    
    @property 
    def posi_min (self):
        """Get the left position of :attr:`select_best_point_ boundaries 
        using the station locations of unarbitrary positions got from
        :attr:`~.ERP.dipoleLength`."""
        
        return np.array(self.aBestInfos[self._best_key_point][2]).min()
        
    @property 
    def rhoa_min (self):
        """Get the buttom position of :attr:`select_best_point_ boundaries 
        using the magnitude  got from :attr:`~.ERP.abest_magnitude`."""
    
        return np.array(self.aBestInfos[self._best_key_point][3]).min()
    
    @property 
    def rhoa_max (self):
        """Get the top position of :attr:`select_best_point_ boundaries 
        using the magnitude  got from :attr:`~.ERP.abest_magnitude`."""
    
        return np.array(self.aBestInfos[self._best_key_point][3]).max()
           
    @property
    def select_best_value_(self): 
        """ Select the best anomaly points."""   
        self._select_best_value= float(
            self.aBestInfos[self._best_key_point][1]
            )
        
        wrap_infos('Best conductive value selected is = {0} Ω.m'.
                        format(self._select_best_value), 
                        on =self.turn_on) 
        
        return self._select_best_value
        
    @property 
    def best_anr (self ): 
        """Get the select best anomaly ratio `abest_anr` along the
        :class:`~watex.methods.erp.ERP`"""
        
        pos_min_index = int(np.where(self.df['pk'].to_numpy(
            ) ==self.posi_min)[0])
        pos_max_index = int(np.where(self.df['pk'].to_numpy(
            ) ==self.posi_max)[0])

        self._anr = compute_anr(sfi = self.best_sfi,
                                rhoa_array = self.df['rhoa'].to_numpy(), 
                                pos_bound_indexes= [pos_min_index ,
                                                    pos_max_index ])
        wrap_infos('Best cover   = {0} % of the whole ERP line'.
                        format(self._anr*100), 
                        on =self.turn_on) 
        
        return self._anr
    
    @property 
    def best_type (self): 
        """ Get the select best anomaly type """
        if self._type is None: 
            self._type = gettype(erp_array= self.df['rhoa'].to_numpy() , 
                                  posMinMax = np.array([float(self.posi_max),
                                                        float(self.posi_min)]), 
                                  pk= self.select_best_point_ ,
                                  pos_array=self.df['pk'].to_numpy() , 
                                  dl= self.dipoleLength)
        
        wrap_infos('Select anomaly type is = {}'.
                       format(self._type), 
                       on =self.turn_on) 
        return self._type 
    
    @property 
    def best_shape (self) : 
        """ Find the selected anomaly shape"""
        if self._shape is None: 

            self._shape = getshape(
                rhoa_range=self.aBestInfos[self._best_key_point][4])
        
        wrap_infos('Select anomaly shape is = {}'.
                       format(self._shape), 
                       on =self.turn_on) 
        return self._shape 
    
    
    @property
    def best_east(self): 
        """ Get the easting coordinates of selected anomaly"""
        
        self._east = self.df['east'][self.best_index]
        return self._east
    
    @property
    def best_north(self): 
        """ Get the northing coordinates of selected anomaly"""
        self._north = self.df['north'][self.best_index]
        return self._north
        
    @property 
    def best_index(self): 
        """ Keep the index of selected best anomaly """
        v_= (np.where( self.df['pk'].to_numpy(
            )== self.select_best_point_)[0]) 
        
        if len(v_)>1: 
            v_=v_[0]
        try : 
            v_ = int(v_)
        except : 
            v_= np.nan 
            pass 
        return v_
            
    @property
    def best_lat(self): 
        """ Get the latitude coordinates of selected anomaly"""
        self._lat = self._latitude[self.best_index]
        return self._lat
    
    @property
    def best_lon(self): 
        """ Get the longitude coordinates of selected anomaly"""
        self._lat = self._longitude[self.best_index]
        return self._lon
    
    @property 
    def best_rhoaRange(self):
        """
        Collect the resistivity values range from selected anomaly boundaries.
        """
        return self.aBestInfos[self._best_key_point][4]
    
      


 

    

    
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        