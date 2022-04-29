# -*- coding: utf-8 -*-
#   Copyright (c) 2021  @Daniel03 <etanoyau@gmail.com>
#   Created date: Thu Apr 14 17:45:55 2022
#   Licence: MIT Licence 
from typing import Any 
import numpy as np 

from ..utils.coreutils import (_parse_args,
                               _define_conductive_zone, 
                               plot_anomaly, 
                               _assert_station_positions 
) 

from ..exceptions import WATexError_station



class ElectricalResistivityProfiling: 
    """ Class deals with the Electrical Resistivity Profiling (ERP)"""
    
    def __init__ (self, *,
                  station:str |None =None,
                  dipole:float =10.,
                  **kwargs): 

        self.dipole = dipole
        self.station = station 
        self.data =None 
        
        self.shape_= None
        self.magnitude_= None 
        self.power_= None 
        self.sfi_= None 
        self.conductive_zone_= None
        self.resistivities_=None 
        
    def fit(self,
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
               raise WATexError_station(msg
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
    
    def show(self,  **kwargs): 
        """ Visualize plot and conductive zone"""
        
        if self.station_ is None : 
            raise ValueError(
                'Please fit the data before calling the ``poof`` method.')
            
        plot_anomaly(self.resistivities_, self.conductive_zone_,
                       self.station_, **kwargs)
        
        return self
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        