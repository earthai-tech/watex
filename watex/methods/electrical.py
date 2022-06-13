# -*- coding: utf-8 -*-
#   Copyright (c) 2021  @Daniel03 <etanoyau@gmail.com>
#   Created date: Thu Apr 14 17:45:55 2022
#   Licence: MIT Licence

from __future__ import annotations 

import os 
import copy
import warnings
import numpy as np 
import pandas as pd

from ..tools.funcutils import (
    repr_callable_obj,
    smart_format,
    smart_strobj_recognition 
    )
from ..tools.coreutils import (
    _define_conductive_zone, 
    _assert_station_positions,  
    fill_coordinates, 
    erpSelector, 
    vesSelector,
    
) 
from ..tools.exmath import (
    shape_, 
    type_, 
    power_, 
    magnitude_, 
    sfi_,
    ohmicArea, 
    invertVES,
    )
from ..typing import  ( 
    List, 
    Optional, 
    Array, 
    NDArray, 
    Series , 
    DataFrame, 

    )
from ..property import assert_arrangement 
from ..tools._watexlog import watexlog 
from ..exceptions import (
    FitError, 
    VESError
    )

class ElectricalMethods : 
    """ Base class of geophysical electrical methods 
    
    The electrical geophysical methods are used to determine the electrical
    resistivity of the earth's subsurface. Thus, electrical methods are 
    employed for those applications in which a knowledge of resistivity 
    or the resistivity distribution will solve or shed light on the problem 
    at hand. The resolution, depth, and areal extent of investigation are 
    functions of the particular electrical method employed. Once resistivity 
    data have been acquired, the resistivity distribution of the subsurface 
    can be interpreted in terms of soil characteristics and/or rock type and 
    geological structure. Resistivity data are usually integrated with other 
    geophysical results and with surface and subsurface geological data to 
    arrive at an interpretation.
    get more infos by consulting https://wiki.aapg.org/Electrical_methods 
    
    
    The :class:`watex.methods.electrical.ElectricalMethods` compose the base 
    class of all the geophysical methods that images the underground using 
    the resistivity values. 
    
    Holds on others optionals infos in ``kws`` arguments: 
       
    ======================  ==============  ===================================
    Attributes              Type                Description  
    ======================  ==============  ===================================
    AB                      float, array    Distance of the current electrodes
                                            in meters. `A` and `B` are used 
                                            as the first and second current 
                                            electrodes by convention. Note that
                                            `AB` is considered as an array of
                                            depth measurement when using the
                                            vertical electrical sounding |VES|
                                            method i.e. AB/2 half-space. Default
                                            is ``200``meters. 
    MN                      float, array    Distance of the current electrodes 
                                            in meters. `M` and `N` are used as
                                            the first and second potential 
                                            electrodes by convention. Note that
                                            `MN` is considered as an array of
                                            potential electrode spacing when 
                                            using the collecting data using the 
                                            vertical electrical sounding |VES|
                                            method i.e MN/2 halfspace. Default 
                                            is ``20.``meters. 
    arrangement             str             Type of dipoles `AB` and `MN`
                                            arrangememts. Can be *schlumberger*
                                            *Wenner-alpha /wenner beta*,
                                            *Gradient rectangular* or *dipole-
                                            dipole*. Default is *schlumberger*.
    area                    str             The name of the survey location or
                                            the exploration area. 
    fromlog10               bool            Set to ``True`` if the given 
                                            resistivities values are collected 
                                            on base 10 logarithm.
    utm_zone                str             string (##N or ##S). utm zone in 
                                            the form of number and North or South
                                            hemisphere, 10S or 03N. 
    datum                   str             well known datum ex. WGS84, NAD27,
                                            etc.         
    projection              str             projected point in lat and lon in 
                                            Datum `latlon`, as decimal degrees 
                                            or 'UTM'. 
    epsg                    int             epsg number defining projection (see 
                                            http://spatialreference.org/ref/ 
                                            for moreinfo). Overrides utm_zone
                                            if both are provided.                           
    ======================  ==============  ===================================
               
    Notes
    -------
        The  `ElectricalMethods` consider the given resistivity values as 
        a normal values and not on base 10 logarithm. So if log10 values 
        are given, set the argument `fromlog10` value to ``True``.
    
    """
    def __init__(self, 
                AB: float = 200. , 
                MN: float = 20.,
                arrangement: str  = 'schlumberger', 
                area : str = None, 
                projection: str ='UTM', 
                datum: str ='WGS84', 
                epsg: int =None, 
                utm_zone: str = None,  
                fromlog10:bool =False, 
                verbose: int = 0, 
                ) -> None:
        self._logging = watexlog.get_watex_logger(self.__class__.__name__)
        self.AB=AB 
        self.MN=MN 
        self.arrangememt=assert_arrangement(arrangement) 
        self.utm_zone=utm_zone 
        self.projection=projection 
        self.datum=datum
        self.epsg=epsg 
        self.area=area 
        self.fromlog10=fromlog10 
        self.verbose=verbose 
        
      
    def __repr__(self):
        """ Pretty format for programmer following the API... """
        
        return repr_callable_obj  (self)
           
        
class ResistivityProfiling(ElectricalMethods): 
    """ Class deals with the Electrical Resistivity Profiling (ERP).
    
    The electrical resistivity profiling is one of the cheap geophysical 
    subsurface imaging method. It is most preferred to find groundwater during
    the campaigns of drinking water supply, especially in developing countries.
    Commonly, it is used in combinaision with the  the vertical electrical
    sounding |VES| to speculated about the layer thickesses and the existence
     of the fracture zone. 
    
    Arguments 
    ----------
    
    **station**: str 
            Station name where the drilling is  expected to be located. The 
            station should numbered from 1 not 0. So if ``S00` is given, the 
            station name should be set to ``S01``. Moreover, if `dipole` value 
            is set as keyword argument,i.e. the station is  named according
            to the  value of the dipole. For instance for `dipole` equals to 
            ``10m``, the first station should be ``S00``, the second ``S10`` , 
            the third ``S20`` and so on. However, it is recommend to name the
            station using counting numbers rather than using the dipole 
            position.
            
    **dipole**: float 
        Value of  the dipole length used during the exploration area. 
        
    **auto**: bool 
            Auto dectect the best conductive zone. If ``True``, the station 
            position should be  the position `station` of the lower
            resistivity value in |ERP|. 
    
    **kws**: dict 
         Additional |ERP| keywords arguments  
         
    
    Examples
    --------
    >>> from watex.methods.electrical import ResistivityProfiling 
    >>> rObj = ResistivityProfiling(AB= 200, MN= 20,station ='S7') 
    >>> rObj.fit('data/erp/testunsafedata.csv')
    >>> rObj.sfi_
    ... array([0.03592814])
    >>> rObj.power_, robj.position_zone_
    ... 90, array([ 0, 30, 60, 90])
    >>> rObj.magnitude_, robj
    >>> rObj.magnitude_, rObj.conductive_zone_
    ... 268, array([1101, 1147, 1345, 1369], dtype=int64)
    >>> robj.dipole
    ... 30
    
    """
    
    def __init__ (self, 
                  station: str | None = None,
                  dipole: float = 10.,
                  auto: bool = False, 
                  **kws): 
        super().__init__(**kws) 
        
        self.dipole=dipole
        self.station=station
        self.auto=auto 
        
        for key in list( kws.keys()): 
            setattr(self, key, kws[key])
            

            
    def fit(self, data : str | NDArray | Series | DataFrame ,
             columns: str | List [str] = None, 
             **kws
            ) -> object: 
        """ Fitting the :class:`ResistivityProfiling` 
        and populate the class attributes.
        
        Parameters 
        ---------- 
            **data**: Path-like obj, Array, Series, Dataframe. 
                Data containing the the collected resistivity values in 
                survey area. 
                    
            **columns**: list, 
                Only necessary if the `data` is given as an array. No need to 
                to explicitly defin when `data` is a dataframe or a Pathlike
                object.
                
            **kws**: dict, 
                Additional keyword arguments; e.g. to force the station to 
                match at least the best minimal resistivity value in the 
                whole data collected in the survey area. 
                
        Returns 
        -------
               object instanciated 
            
        Notes
        ------
                The station should numbered from 1 not 0. So if ``S00` 
                is given, the station name should be set to ``S01``. 
                Moreover, if `dipole` value is set as keyword argument,
                i.e. the station is  named according to the  value of 
                the dipole. For instance for `dipole` equals to ``10m``, 
                the first station should be ``S00``, the second ``S10`` , 
                the third ``S20`` and so on. However, it is recommend to 
                name the station using counting numbers rather than using 
                the dipole position.
        """
        
        self._logging.info('`Fit` method from {self.__class__.__name__!r}'
                           ' is triggered ')
        if isinstance(data, str): 
            if not os.path.isfile (data): 
                raise TypeError ( f'{data!r} object should be a file,'
                                 f' got {type(data).__name__!r}'
                                 )

        data = erpSelector(data, columns) 
        self.data_ = copy.deepcopy(data) 
        
        self.data_, self.utm_zone = fill_coordinates(
            self.data_, utm_zone= self.utm_zone, 
            datum = self.datum , epsg= self.epsg ) 
        self.resistivity_ = self.data_.resistivity 
        # convert app.rho to the concrete value 
        # if log10 rho are provided.
        if self.fromlog10: 
            self.resistivity_ = np.power(
                10, self.resistivity_)
            if self.verbose > 7 : 
                print("Resistivity profiling data should be overwritten to "
                      " take the concrete values rather than log10(ohm.meters)"
                      )
            self.data_['resistivity'] = self.resistivity_
        
        self._logging.info(f'Retrieving the {self.__class__.__name__!r} '
                           ' components and recompute the coordinate values...')
        
        self.position_ = self.data_.station 
        self.lat_ = self.data_.latitude  
        self.lon_= self.data_.longitude 
        self.east_ = self.data_.easting 
        self.north_ = self.data_.northing 
        
        if self.verbose > 7: 
            print(f'Compute {self.__class__.__name__!r} parameter numbers.' )
            
        self._logging.info(f'Assert the station {self.station!r} if given' 
                           'or auto-detected otherwise.')
        
        # assert station and use the automatic station detection  
        ##########################################################
        if self.auto and self.station is not None: 
            warnings.warn (
                f"Station {self.station!r} is given while 'auto' is 'True'."
                  " Only the auto-detection is used instead...", UserWarning)
                
            self.station = None 
        
        if self.station is None: 
            if not self.auto: 
                warnings.warn("Station number is missing! By default the " 
                              "automatic-detection should be triggered.")
                self.auto = True 

        if self.station is not None: 
            if self.verbose > 7 : 
                print("Assert the given station and recomputed the array position."
                      )
                self._logging.warn(
                    f'Station value {self.station!r} in the given data '
                    'should be overwritten...')
                
        # recompute the position and dipolelength 
        self.position_, self.dipole = _assert_station_positions(
            df = self.data_, **kws)
        self.data_['station'] = self.position_ 
        
        ############################################################
        # Define the selected anomaly (conductive_zone )
        # ix: s the index of drilling point in the selected 
        # conductive zone while 
        # pos: is the index of drilling point in the whole 
        # survey position  
        ix, pos, self.conductive_zone_, self.position_zone_ =\
            _define_conductive_zone(
                        self.resistivity_,
                        s= self.station, 
                        auto = self.auto,
                        #keep Python numbering index (from 0 ->...),
                        keepindex = True, 
                        
                        # No need to implement the dipole computation 
                        # for detecting the sation position if the 
                        # station is given
                        # dipole =self.dipole if self.station is None else None,
                        
                        p = self.position_
            )

        if self.verbose >7 : 
            print('Compute the property values at the station location ' 
                  ' expecting for drilling location <`sves`> at'
                  f' position {str(pos+1)!r}')
            
        # Note that `sves` is the station location expecting to 
        # hold the drilling operations at this point. It is considered  
        # as the select point of the conductive zone. 
        self.sves_ = f'S{pos:03}' 
        
        self._logging.info ('Loading main params value from the expecting' 
                            f' drilling location: {self.sves_!r}')
    
        self.sves_lat_ = self.lat_[pos] 
        self.sves_lon_= self.lon_[pos] 
        self.sves_east_ = self.east_[pos]
        self.sves_north_= self.north_[pos] 
        self.sves_resistivity_= self.resistivity_[pos]
        
        # Compute the predictor parameters 
        self.power_ = power_(self.position_zone_)
        self.shape_ = shape_(self.conductive_zone_ ,
                             s= ix , 
                             p= self.position_)
        self.magnitude_ = magnitude_(self.conductive_zone_)
        self.type_ = type_ (self.resistivity_)
        self.sfi_ = sfi_(cz = self.conductive_zone_, 
                         p = self.position_zone_, 
                         s = ix, 
                         dipolelength= self.dipole
                         )
        
        if self.verbose > 7 :
            pn = ('type', 'shape', 'magnitude', 'power' , 'sfi')
            print(f"Parameter numbers {smart_format(pn)}"
                  " were successfully computed.") 

        return self 

    def summary(self, keeponlyparams: bool = False) -> DataFrame : 
        """ Summarize the most import parameters for prediction purpose.
        
        If `keeponlyparams` is set to ``True``. Method should output only 
        the main important params for prediction purpose... 
        """
        
        try:
             getattr(self, 'type_'); getattr(self, 'sfi_')
        except FitError:
            raise FitError(
                "Can't call the method 'summary' without fitting the"
                f" {self.__class__.__name__!r} object first.")
        
        usefulparams = (
            'station','dipole',  'sves_lon_', 'sves_lat_','sves_east_', 
            'sves_north_', 'sves_resistivity_', 'power_', 'magnitude_',
            'shape_','type_','sfi_')
        
        table_= pd.DataFrame (
            {f"{k[:-1] if k.endswith('_') else k }": getattr(self, k , np.nan )
             for k in usefulparams}, index=range(1)
            )
        table_.station = self.sves_
        table_.set_index ('station', inplace =True)
        table_.rename (columns= {'sves_lat':'latitude', 'sves_lon':'longitude',
                        'sves_east':'easting', 'sves_north':'northing'},
                       inplace =True)
        if keeponlyparams: 
            table_.reset_index(inplace =True )
            table_.drop(
                ['station', 'dipole', 'sves_resistivity', 
                 'latitude', 'longitude', 'easting', 'northing'],
                axis='columns',  inplace =True )
            
        return table_ 
        
            
    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        return repr_callable_obj  (self)
       
    
    def __getattr__(self, name):
        if name.endswith ('_'): 
            if name not in self.__dict__.keys(): 
                if name in ('data_', 'resistivity_', 'lat_', 'lon_', 
                            'easting_', 'northing_', 'sves_lon_', 'sves_lat_',
                            'sves_east_', 'sves_north_', 'sves_resistivity_',
                            'power_', 'magnitude_','shape_','type_','sfi_'): 
                    raise FitError (
                        f'Fit the {self.__class__.__name__!r} object first'
                        )
                
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        appender  = "" if rv is None else f'. Do you mean {rv!r}'
        
        raise AttributeError (
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{appender}{"" if rv is None else "?"}'
            )

    
    
class VerticalSounding (ElectricalMethods): 
    """ 
    Vertical Electrical Sounding (|VES|) class; inherits of ElectricalMethods 
    base class. 
    
    The VES is carried out to speculate about the existence of a fracture zone
    and the layer thicknesses. Commonly, it comes as supplement methods to |ERP| 
    after selecting the best conductive zone when survey is made on 
    one-dimensional. 
    
    Arguments 
    ----------

    **fromS**: float 
        The depth in meters from which one expects to find a fracture zone 
        outside of pollutions. Indeed, the `fromS` parameter is used to  
        speculate about the expected groundwater in the fractured rocks 
        under the average level of water inrush in a specific area. For 
        instance in :ref:`Bagoue region`, the average depth of water inrush 
        is around ``45m``.So the `fromS` can be specified via the water inrush 
        average value. 
        
    **rho0**: float 
        Value of the starting resistivity model. If ``None``, `rho0` should be
        the half minumm value of the apparent resistivity  collected. Units is
        in Ω.m not log10(Ω.m)
        
    **h0**: float 
        Thickness  in meter of the first layers in meters.If ``None``, it 
        should be the minimum thickess as possible ``1.``m. 
    
    **strategy**: str 
        Type of inversion scheme. The defaut is Hybrid Monte Carlo (HMC) known
        as ``HMCMC``. Another scheme is Bayesian neural network approach (``BNN``). 
        
    **vesorder** int 
        The index to retrieve the resistivity data of a specific sounding point.
        Sometimes the sounding data are composed of the different sounding 
        values collected in the same survey area into different |ERP| line.
        For instance::
            
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
        via the function `vesSelector`, the header of the data should hold 
        the `resistivity`. For instance, refering to the table above, the 
        data should be::
            
            +----+----+-------------+-------------+-------------+----
            | AB | MN |resistivity  | resistivity | resistivity | ...
            +----+----+-------------+-------------+-------------+----
        
        Therefore, the `vesorder` is used to select the specific resistivity
        values i.e. select the corresponding sounding number  of the |VES| 
        expecting to locate the drilling operations or for computation. For 
        esample, `vesorder`=1 should figure out:: 
            
            +------+------+----+        +-----+----+------------+
            | AB/2 | MN/2 |SE2 |  -->   | AB | MN |resistivity |
            +------+------+----+        +----+----+------------+
        
        If `vesorder` is ``None`` and the number of sounding curves are more 
        than one, by default the first sounding curve is selected ie 
        `rhoaIndex` equals to ``0``
        
    **typeofop**: str 
        Type of operation to apply  to the resistivity 
        values `rhoa` of the duplicated spacing points `AB`. The *default* 
        operation is ``mean``. Sometimes at the potential electrodes (`MN`),the 
        measurement of `AB` are collected twice after modifying the distance
        of `MN` a bit. At this point, two or many resistivity values are 
        targetted to the same distance `AB`  (`AB` still remains unchangeable 
        while while `MN` is changed). So the operation consists whether to the 
        average (``mean``) resistiviy values or to take the ``median``values
        or to ``leaveOneOut`` (i.e. keep one value of resistivity among the 
        different values collected at the same point`AB`) at the same spacing 
        `AB`. Note that for the `LeaveOneOut``, the selected 
        resistivity value is randomly chosen.
        
    **objective**: str 
        Type operation to outputs. By default, the function outputs the value
        of pseudo-area in :math:`$\ohm.m^2$`. However, for plotting purpose by
        setting the argument to ``view``, its gives an alternatively outputs of
        X and Y, recomputed and projected as weel as the X and Y values of the
        expected fractured zone. Where X is the AB dipole spacing when imaging 
        to the depth and Y is the apparent resistivity computed.
        
    **kws**: dict 
        Additionnal keywords arguments from |VES| data operations. 
        See :doc:`watex.utils.exmath.vesDataOperator` for futher details.
        
    See also 
    ---------
    .. _Kouadio et al 2022: https://doi.org/10.1029/2021WR031623
    
    References
    ----------

    Koefoed, O. (1970). A fast method for determining the layer distribution 
        from the raised kernel function in geoelectrical sounding. Geophysical
        Prospecting, 18(4), 564–570. https://doi.org/10.1111/j.1365-2478.1970.tb02129.x
        
    Koefoed, O. (1976). Progress in the Direct Interpretation of Resistivity 
        Soundings: an Algorithm. Geophysical Prospecting, 24(2), 233–240.
        https://doi.org/10.1111/j.1365-2478.1976.tb00921.x
        
    Examples
    --------
    >>> from watex.methods import VerticalSounding 
    >>> vobj = VerticalSounding(fromS= 45, vesorder= 3)
    >>> vobj.fit('data/ves/ves_gbalo.xlsx')
    >>> vobj.ohmic_area_ # in ohm.m^2
    ... 349.6432550517697
    >>> vobj.nareas_ # number of areas computed 
    ... 2
    >>> vobj.area1_, vobj.area2_ # value of each area in ohm.m^2 
    ... (254.28891096053943, 95.35434409123027) 
    >>> vobj.roots_ # different boundaries in pairs 
    >>> array([ 45.        ,  57.55255255,  96.91691692, 100.        ])
    """
    def __init__(self,
                 fromS: float = 45.,
                 rho0: float = None, 
                 h0 : float = 1., 
                 strategy: str = 'HMCMC',
                 vesorder: int = None, 
                 typeofop: str = 'mean',
                 objective: Optional[str] = 'coverall',
                 **kws) -> None : 
        super().__init__(**kws) 
        
        self.fromS=fromS 
        self.vesorder=vesorder 
        self.typeofop=typeofop
        self.objective=objective 
        self.rho0=rho0, 
        self.h0=h0
        self.strategy = strategy
        
        for key in list( kws.keys()): 
            setattr(self, key, kws[key])
            
        
    def fit(self, data: str | DataFrame, **kwd ): 
        """ Fit the sounding |VES| curves and computed the ohmic as well as 
        the thicknesses of different layers ... 
        
        :param data: DataFrame - It is composed of spacing values `AB` and  the 
        apparent resistivity values `rhoa`. If `data` is given, params `AB` and 
        `rhoa` should be kept to ``None``.   
            
        :param AB: array-like - Spacing of the current electrodes when exploring
        in deeper. Units are in meters. Note that the `AB` is by convention 
        equals to `AB/2`. It's taken as half-space of the investigation depth... 
        
        :param MN: array-like - Potential electrodes distances at each investigation 
        depth. Note by convention the values are half-space and equals to `MN/2`.
        
        :param rhoa: array-like - Apparent resistivity values collected in imaging 
        in depth. Units are in Ω.m not log10(Ω.m)
        
        :param readableformats: tuple -Specific readable files. The defaults of 
        files reading is ``xlsx`` and ``csc``. 
        
        """
        def prettyprinter (n, r,v): 
            """ Display some details when verbose is higher... 
            
            :param n: int : number of areas 
            :param r: array-like. Pair values of integral bounds (-inf, +inf)
            :param v: array-float - values of pseudo-areas computed.  """
            print('=' * 73 )
            print('| {0:^15} | {1:>15} | {2:>15} | {3:>15} |'.format(
                'N-area', 'lb:-AB/2 (m)','ub:-AB/2(m)', 'ohmS (Ω.m^2)' ))
            print('=' * 73 )
            for ii in range (n): 
                print('| {0:^15} | {1:>15} | {2:>15} | {3:>15} |'.format(
                    ii+1, round(r[ii][0]), round(r[ii][1]), round(v[ii], 3)))
                print('-'*73)
        
        self._logging.info (f'`Fit` method from {self.__class__.__name__!r}'
                           ' is triggered')
        
        if self.verbose >= 7 : 
            print(f'Range {str(self.vesorder)!r} of resistivity data of the  '
                  'sshould be selected as the main sounding data. ')
        self.data_ = vesSelector(
            data = data, rhoaIndex= self.vesorder, **kwd )
        self.max_depth_ = self.data_.AB.max()
        
        if self.fromS >= self.max_depth_ : 
            raise VESError(
                " Process of the depth research monitoring is aborted!"
                f" The searching point 'fromS'<{self.fromS}m> ' is expected "
                 f"to be less than the maximum depth <{self.max_depth_}m>.")
        
        if self.verbose >= 3 : 
            print("Pseudo-area should be computed from AB/2 ={str(self.fromS)}"
                  f" to {self.max_depth_} meters. "
                  )
        r = ohmicArea( data = self.data_ , sum = False, ohmSkey = self.fromS,  
                    objective = self.objective , typeofop = self.typeofop, 
                    )
        self._logging.info(f'Populating {self.__class__.__name__!r} property'
                           ' attributes.')
        oc, gc = r 
        
        ohmS, self.err_, self.roots_ = list(oc) 
        self.nareas_ = len(ohmS) 
        
        self._logging.info(f'Setting the {self.nareas_} pseudo-areas calculated.')
        for ii in range(self.nareas_): 
            self.__setattr__(f"area{ii+1}_", ohmS[ii])
            
        self.roots_ = np.split(self.roots_, len(self.roots_)//2 ) if len(
            self.roots_) > 2 else [np.array(self.roots_)]
        
        if self.verbose >= 7 : 
            prettyprinter(n= self.nareas_, r= self.roots_, v= ohmS)

        self.ohmic_area_= sum(ohmS) # sum the different spaces 
        
        self.XY_ , _, self.XYarea_ = list(gc) 
        self.AB_ = self.XY_[:, 0] 
        self.resistivity_ = self.XY_[:, 1] 
        self.fz_= self.XYarea_[:, 0] 
        self.fz_resistivity_ = self.XYarea_[:, 1] 
        
        if self.verbose > 7 :
            print("The Parameter numbers were successfully computed.") 
        return self 
    
    def summary(self, keeponlyparams: bool = False) -> DataFrame : 
        """ Summarize the most import parameters for prediction purpose.
        
        If `keeponlyparams` is set to ``True``. Method should output only 
        the main important params for prediction purpose... 
        """
        
        try:
             getattr(self, 'ohmic_area_'); getattr(self, 'fz_')
        except FitError:
            raise FitError(
                "Can't call the method 'summary' without fitting the"
                f" {self.__class__.__name__!r} object first.")
        
        usefulparams = (
            'area', 'AB','MN',  'arrangement','utm_zone', 'objective', 'rho0',
             'h0', 'fromS', 'max_depth_', 'ohmic_area_', 'nareas_')
        
        table_= pd.DataFrame (
            {f"{k }": getattr(self, k , np.nan )
             for k in usefulparams}, index=range(1)
            )
        table_.area = self.area
        table_.set_index ('area', inplace =True)
        table_.rename (columns= {'max_depth_':'max_depth',
                                 'ohmic_area_':'ohmic_area',
                            'nareas_':'nareas'},
                           inplace =True)
        if keeponlyparams: 
            table_.reset_index(inplace =True )
            table_.drop( 
                [ el for el in list(table_.columns) if el !='ohmic_area'],
                axis='columns',  inplace =True
                )
            
        return table_ 
        
    def invert( self, data: str | DataFrame , strategy=None,  **kwd): 
        """ Invert the |VES| data collected in the exporation area.
        
        :param data: Dataframe pandas - contains the depth measurement AB from 
        current electrodes, the potentials electrodes MN and the collected 
        apparents resistivities. 
        
        :param rho0: float - Value of the starting resistivity model. If ``None``, 
            `rho0` should be the half minumm value of the apparent resistivity  
            collected. Units is in Ω.m not log10(Ω.m)
        :param h0: float -  Thickness  in meter of the first layers in meters.
         If ``None``, it should be the minimum thickess as possible ``1.``m. 
        
        :param strategy: str - Type of inversion scheme. The defaut is Hybrid Monte 
        Carlo (HMC) known as ``HMCMC``. Another scheme is Bayesian neural network
        approach (``BNN``). 
        
        :param kws: dict - Additionnal keywords arguments from |VES| data  
        operations. See :doc:`watex.utils.exmath.vesDataOperator` for futher
        details.
        
        """
        self.data_ = getattr(self, 'data_', None)
        if self.data_ is None: 
            self.data_ = vesSelector(
                data = data, rhoaIndex= self.typeofop, **kwd )
        # invert data 
        #XXX TODO 
        if strategy is not None: 
            self.strategy = strategy 
            
        invertVES(data= self.data_, h0 = self.h0 , rho0 = self.rho0,
                  typeof = self.strategy )
 
    def __repr__(self):
        """ Pretty format for programmer following the API... """
        return repr_callable_obj(self)
       
    def __getattr__(self, name):
        if name.endswith ('_'): 
            if name not in self.__dict__.keys(): 
                if name in ('data_', 'resistivity_', 'ohmic_area_', 'err_', 
                            'roots_', 'XY_', 'XYarea_', 'AB_',
                            'resistivity_', 'fz_', 'fz_resistivity_'): 
                    raise FitError (
                        f'Fit the {self.__class__.__name__!r} object first'
                        )
                
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        appender  = "" if rv is None else f'. Do you mean {rv!r}'
        
        raise AttributeError (
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{appender}{"" if rv is None else "?"}'
            )

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        