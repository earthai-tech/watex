# -*- coding: utf-8 -*-
#   Copyright (c) 2021  @Daniel03 <etanoyau@gmail.com>
#   Created date: Fri Apr 15 10:46:56 2022
#   Licence: MIT Licence 

# from abc import ABCMeta 
# import warnings 

"""
`WATex <https://github.com/WEgeophysics/watex/>`_ properties objects 
====================================================================

.. |ERP| replace: Electrical resistivity profiling 

"""

__all__ = [ 'P' ]

class P:
    """
    Data properties are values that are hidden to avoid modifications alongside 
    the packages. Its was used for assertion, comparison etceteara. These are 
    enumerated below into a property objects.

    **frcolortags**: Stands for flow rate colors tags. Values are :: 
                    '#CED9EF','#9EB3DD', '#3B70F2', '#0A4CEF'
    **ididctags**: Stands for the list of index set in dictionary which encompasses 
                key and values of all different prefixes.
                
    **isation**: List of prefixes used for indexing the stations in the |ERP|. 

    **ieasting**: List of prefixes used for indexing the easting coordinates array. 

    **inorthing**: List of prefixes used to index the northing coordinates. 
     
    **iresistivity** List of prefix used for indexing the apparent resistivity 
                values in the |ERP| data collected during the survey. 

    **isenr**: Is the list of heads columns during the data collections. Any data 
                head |ERP| data provided should be converted into 
                the following arangement::
                    
            +----------+-----------+-----------+-------------+
            |station   | easting   | northing  | resistivity | 
            +----------+-----------+-----------+-------------+
            
    **P**: Typing class for fectching the properties. For instance:: 
        
        >>> from watex._properties import P 
        >>> P.idicttags 
        ... <property at 0x1ec1f2c3ae0>
        >>> P().idictags 
        ... 
        {'station': ['pk', 'sta', 'pos'], 'easting': ['east', 'x', 'long'],
         'northing': ['north', 'y', 'lat'], 'resistivity': ['rho', 'app', 'res']}
        >>> {k:v for k, v in  P.__dict__.items() if '__' not in k}
        ... {'_station': ['pk', 'sta', 'pos'],
             '_easting': ['east', 'x', 'long'],
             '_northing': ['north', 'y', 'lat'],
             '_resistivity': ['rho', 'app', 'res'],
             'frcolortags': <property at 0x1ec1f2fee00>,
             'idicttags': <property at 0x1ec1f2c3ae0>,
             'istation': <property at 0x1ec1f2c3ea0>,
             'ieasting': <property at 0x1ec1f2c39f0>,
             'inorthing': <property at 0x1ec1f2c3c70>,
             'iresistivity': <property at 0x1ec1f2c3e00>,
             'isenr': <property at 0x1ec1f2c3db0>}
        >>> P().isenr 
        ... ['station','easting','northing', 'resistivity' ]
    
    """
    station_prefix   = [
        'pk','sta','pos'
    ]
    easting_prefix   =[
        'east','x','long'
                ]
    northing_prefix = [
        'north','y','lat'
    ]
    resistivity_prefix = [
        'rho','app','res'
    ]
    erp_head= [
        'station','easting','northing', 'resistivity' 
    ]
    param_options = [
        ['bore', 'for'], 
        ['x','east'], 
        ['y', 'north'], 
        ['pow', 'puiss', 'pa'], 
        ['magn', 'amp', 'ma'], 
        ['shape', 'form'], 
        ['type'], 
        ['sfi', 'if'], 
        ['lat'], 
        ['lon'], 
        ['lwi', 'wi'], 
        ['ohms', 'surf'], 
        ['geol'], 
        ['flow', 'deb']
    ]
    param_ids =[
        'id', 
        'east', 
        'north', 
        'power', 
        'magnitude', 
        'shape', 
        'type', 
        'sfi', 
        'lat', 
        'lon', 
        'lwi', 
        'ohmS', 
        'geol', 
        'flow'
    ]
   
    all_prefixes = { f'_{k}':v for k, v in zip (erp_head , [
        station_prefix, easting_prefix, northing_prefix,
        resistivity_prefix])}

    def __init__( self) :
        for key , value in self.all_prefixes.items() : 
            self.__setattr__( key , value)
            
    @property 
    def frcolortags (self): 
        """ set the dictionnar"""
        return  dict ((f'fr{k}', f'#{v}') for k, v in zip(
                        range(4), ('CED9EF','9EB3DD', '3B70F2', '0A4CEF' )
                        )
        )
    @property 
    def idicttags (self): 
        """ Is the collection of data properties """ 
        return  dict ( (k, v) for k, v in zip(self.isenr,
              [self.istation, self.ieasting, self.inorthing ,
                self.iresistivity]))
    @property 
    def istation(self) : 
        """ Use prefix to identify station location positions """
        return self._station 
    
    @property 
    def ieasting (self): 
        """ Use prefix to identify easting coordinates if given in the
        dataset. """
        return self._easting 
    
    @property 
    def inorthing(self): 
        """ Use prefix to identify northing coordinates if given in the
        dataset. """
        return self._northing
    
    @property 
    def iresistivity(self): 
        """ Use prefix to identify the resistivity values in the dataset"""
        return self._resistivity 
    
    @property 
    def isenr(self): 
        """ `SENR` is the abbreviation of `S`for ``Stations``, `E` for ``Easting`, 
        `N` for ``Northing`` and `R``for resistivity. `SENR` is the expected 
        columns in Electrical resistivity profiling. Indeed, it keeps the 
        traditional collections sheets during the survey. """
        return self.erp_head



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    