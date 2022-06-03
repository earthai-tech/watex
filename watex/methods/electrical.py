# -*- coding: utf-8 -*-
#   Copyright (c) 2021  @Daniel03 <etanoyau@gmail.com>
#   Created date: Thu Apr 14 17:45:55 2022
#   Licence: MIT Licence

from __future__ import annotations 

import os 
import copy
import numpy as np 

from ..utils.funcutils import (
    repr_callable_obj,
    smart_format,
    smart_strobj_recognition 
    )
from ..utils.coreutils import (
    _parse_args,
    _define_conductive_zone, 
    _assert_station_positions, 
    erp_selector, 
    fill_coordinates, 
) 
from ..utils.exmath import (
    shape_, 
    type_, 
    power_, 
    magnitude_, 
    sfi_, 
    )
from .._typing import  ( 
    Any, 
    List, 
    NDArray, 
    Series , 
    DataFrame, 

    )
from .._property import assert_arrangement 
from ..utils._watexlog import watexlog 
from ..utils.decorators import deprecated 
from ..exceptions import StationError


class ElectricalMethods : 
    """ Base class of geophys electrical methods 
    
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
    AB                      float           Distance of the current electrodes
                                            in meters. `A` and `B` are used 
                                            as the first and second current 
                                            electrodes by convention.
    MN                      float           Distance of the current electrodes 
                                            in meters. `M` and `N` are used as
                                            the first and second potential 
                                            electrodes by convention.
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
                                            if both are provided                           
    ======================  ==============  ===================================
               
    Notes
    -------
        The  `ElectricalMethods` consider the given resistivity values as 
        a normal values and not on base 10 logarithm. So if log10 values 
        are given, set the argument `fromlog10` value to ``True``.
    
    """
    def __init__(self, 
                AB: float  =None , 
                MN: float = None,
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
    >>> rObj.power_
    ... 268
    >>> robj.dipole
    ... 30
    >>> rObj.conductive_zone_
    ... array([1101, 1147, 1345, 1369], dtype=int64)
    
    """
    
    def __init__ (self, 
                  station: str | None = None,
                  dipole: float = 10.,
                  auto: bool = False, 
                  **kws): 
        super().__init__(ResistivityProfiling) 
        
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
                Only necessary if the `data` is given as array. No need to 
                explicitly defined when `data` is a dataframe of Pathlike
                object.
                
            **kws**: dict, 
                Additional keyword arguments to force the station to 
                match as least the best minimal resistivity value  in the 
                resistivity data collected in the survey area. 
                
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

        data = erp_selector(data, columns) 
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
            
        self._logging.info(f'Assert the station `{self.station}` if given' 
                           'or auto-detected  otherwise.')
        
        if self.station is not None: #np.all(self.position_)==0. and 
            if self.verbose > 7 : 
                print("Assert the given station and recomputed the array position."
                      )
                self._logging.warn(
                    f'Station value {self.station!r} in the given data '
                    'should be overwritten...')
                
            self.position_, self.dipole = _assert_station_positions(
                df = self.data_, **kws)
            self.data_['station'] = self.position_ 
            
        # assert station and use auto station detect 
        ##########################################################
        if self.station is None: 
           if not self.auto: 
               raise StationError (
                   "Parameter 'auto' is set to 'False'. No station location found. "
                   " Unable to detect a position expecting to locate the drilling." 
                   )
        ############################################################
        ix, self.conductive_zone_, self.position_zone_ = _define_conductive_zone(
                        self.resistivity_, s= self.station, 
                        auto= self.auto , dipole =self.dipole, 
                        p = self.position_
            )
        # get the station resistivity value 
        self.station_resistivity_= self.resistivity_ [ix]
        # find the index of the resistivity value in the conductive zone 
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

     
    def __repr__(self):
        """ Pretty format for programmer following the API... """
        return repr_callable_obj  (self)
       
    
    def __getattr__(self, name):
        if name.endswith ('_'): 
            if name not in self.__dict__.items(): 
                raise AttributeError (
                    f'Fit the {self.__class__.__name__!r} object first'
                    )
                
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        appender  = "" if rv is None else f'. Do you mean {rv!r}'
        
        raise AttributeError (
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{appender}{"" if rv is None else "?"}'
            )
        
    @deprecated ('Should be removed for the next release. Use the '
                 ' :meth:`watex.methods.electrical.~ResistivityProfiling.fit`'
                 'instead.')
    def fit_(self,
            *args:Any, 
            station:str |None  =None,
            auto_station:bool  =False,
            fromlog10:bool =False,
            **kws): 
        """ Fitting the :class:`~ElectricalResistivityProfile` 
        to populate the class attributes. 
        
        :param args: argument data. It  be a list of data,
            a dataframe or a Path like object. If a Path-like object
            is given, it should be the priority of reading.
            
        :param station: str, int. The station name of station position
        :param auto_station: bool. If the station is not given, turn the 
            `auto_station` to ``True`` will select the station with 
            very low resistivity values. To have full control, it is 
            strongly recommended to provide the station name or index.
            
        :param fromlog10: bool. Setting to ``True`` will convert the given 
            log10 apparent resistivity values into real values in ohm.m. 
            It is usefull when data is collected in log10/ohm.m. 
            
        .. note:: The station should numbered from 1 not 0. SO if ``S00` 
                is given, the station name should be set to ``S01``. 
                Moreover, if `dipole` value is set as keyword argument,
                i.e. the station is  named according to the  value of 
                the dipole. For instance for `dipole` equals to ``10m``, 
                the first station should be ``S00``, the second ``S10`` , 
                the third ``S30`` and so on. However, it is recommend to 
                name the station using counting numbers rather than using 
                the dipole position.
        """
        msg =''.join(['Should provide the station expecting to be the'
            ' position to locate the drilling. Or use the default station'
            ' by setting the argument `auto_station` to ``True``. '
            
            ])
        if station is not None :
            self.station = station 
        # Get the data and keep the resistivity at the first index 
        self.data, col  = _parse_args(list(args))
        # If the index is get 
        self.resistivities_ = self.data [:, 0]
        if fromlog10: 
            self.resistivities_ = np.power(
                10, self.resistivities_)
        
        # assert station and use auto station detect 
        if self.station is None: 
           if not auto_station: 
               raise StationError (msg
                   )
           self.station = np.argmin( self.resistivities_) + 1
           
        # get the conductive zone 
        self.conductive_zone_ , ix = _define_conductive_zone(
            self.resistivities_ , self.station , **kws) 
        
        # get the station resistivity value 
        self.station_res_= self.resistivities_ [ix ]
        self.station_ = 'S{:02}'.format(ix +1)
        self.pks_ = np.arange(1, len (self.resistivities_) +1) 
        
        # Numerise the positions  
        if col is not None: 
            self.positions_, self.dipole = _assert_station_positions(
                self.data, col) 
        else : 
            self.positions_ = self.pks_ * self.dipole 
        
        return self 
    
    
class VerticalSounding (ElectricalMethods): 
    """ 
    Vertical Electrical Sounding (|VES|) class; inherits of ElectricalMethods 
    base class. 
    
    The VES is carried out to speculate about the existence of a fracture zone
    and the layer thicknesses. Commonly, it comes as supplement methods to |ERP| 
    after selecting the best conductive zone when survey is made on 
    one-dimensional. 
    
    
    
    """
    def __init__(self, ): 
        super().__init__(VerticalSounding) 
        
    def fit(self, data ): 
        pass 
    
    
    def __repr__(self):
        """ Pretty format for programmer following the API... """
        return repr_callable_obj  (self)
       
    def __getattr__(self, name):
        if name.endswith ('_'): 
            if name not in self.__dict__.items(): 
                raise AttributeError (
                    f'Fit the {self.__class__.__name__!r} object first'
                    )
                
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        appender  = "" if rv is None else f'. Do you mean {rv!r}'
        
        raise AttributeError (
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{appender}{"" if rv is None else "?"}'
            )
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        