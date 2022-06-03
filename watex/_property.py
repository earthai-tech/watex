# -*- coding: utf-8 -*-
#   Copyright (c) 2021  @Daniel03 <etanoyau@gmail.com>
#   Created date: Fri Apr 15 10:46:56 2022
#   Licence: MIT Licence 

# from abc import ABCMeta 
# import warnings 

from __future__ import annotations 
"""
`WATex <https://github.com/WEgeophysics/watex/>`_ properties objects 
====================================================================

.. |ERP| replace: Electrical resistivity profiling 

"""

__all__ = [ 'P', 'BagoueNotes' ]

array_configuration ={
    1 : (
        ['Schlumberger','AB>> MN','slbg'], 
        'S'
        ), 
    2 : (
        ['Wenner','AB=MN'], 
         'W'
         ), 
    3: (
        ['Dipole-dipole','dd','AB<BM>MN','MN<NA>AB'],
        'DD'
        ), 
    4: (
        ['Gradient-rectangular','[AB]MN', 'MN[AB]','[AB]'],
        'GR'
        )
    }

    
utm_zone_designator ={
    'X':[72,84], 
    'W':[64,72], 
    'V':[56,64],
    'U':[48,56],
    'T':[40,48],
    'S':[32,40], 
    'R':[24,32], 
    'Q':[16,24], 
    'P':[8,16],
    'N':[0,8], 
    'M':[-8,0],
    'L':[-16, 8], 
    'K':[-24,-16],
    'J':[-32,-24],
    'H':[-40,-32],
    'G':[-48,-40], 
    'F':[-56,-48],
    'E':[-64, -56],
    'D':[-72,-64], 
    'C':[-80,-72],
    'Z':[-80,84]
}
    


# XXX EDI-Jand AVG property indexes 

# A. Jones file indexes 
_j=[
    '>AZIMUTH',
    '>LATITUDE',
    '>LONGITUDE',
    '>ELEVATION',
    
    'RXX','RXY','RYX',
    'RYY','RTE','RTM',
    'RAV','RDE','RZX',
    'RZY','SXX','SXY',
    'SYX','SYY','STE',
    'STM','SAV','SDE',
    'SZX','SZY','ZXX',
    'ZXY','ZYX', 'ZYY',
    'ZTE','ZTM','ZAV',
    'ZDE','ZZX','ZZY', 
    'QXX','QXY','QYX',
    'QYY','QTE','QTM',
    'QAV','QDE','QZX',
    'QZY','CXX','CXY',
    'CYX','CYY', 'CTE',
    'CTM','CAV', 'CDE',
    'CZX','CZY','TXX',
    'TXY','TYX','TYY',
    'TTE','TTM','TAV',
    'TDE','TZX','TZY',
    'ZXX','ZXY','ZYX'
]
    
# Zonge Engineering file indexes 
_avg=[
    'skp','Station',
    'Freq','Comp',
    'Amps','Emag',
    'Ephz','Hmag',
    'Hphz','Resistivity',
    'Phase','%Emag',
    'sEphz','%Hmag',
    'sHphz','%Rho',
    'sPhz', 'Tx.Amp',
    'E.mag','E.phz',
    'B.mag','B.phz',
    'Z.mag','Z.phz',
    'ARes.mag','SRes',
    'E.wgt', 'B.wgt',
    'E.%err','E.perr',
    'B.%err','B.perr',
    'Z.%err','Z.perr',
    'ARes.%err'
]
    
    
_edi =[
    #Head=Infos-Frequency-Rhorot
    # Zrot and end blocks
    '>HEAD','>INFO',
    
    #Definitions Measurments Blocks
    '>=DEFINEMEAS',
    '>=MTSECT',
    '>FREQ ORDER=INC', 
    '>ZROT',
    '>RHOROT',
    '>!Comment',
    '>FREQ',
    '>HMEAS',
    '>EMEAS',
    
    #Apparents Resistivities
    #and Phase Blocks
    '>RHOXY','>PHSXY',
    '>RHOXY.VAR','>PHSXY.VAR',
    '>RHOXY.ERR','>PHSXY.ERR',
    '>RHOXY.FIT','>PHSXY.FIT', 
    '>RHOYX','>PHSYX',
    '>RHOYX.VAR','>PHSYX.VAR',
    '>RHOYX.ERR','>PHSYX.ERR',
    '>RHOYX.FIT','>PHSYX.FIT',
    '>RHOXX','>PHSXX',
    '>RHOXX.VAR','>PHSXX.VAR',
    '>RHOXX.ERR','>PHSXX.ERR',
    '>RHOXX.FIT','>PHSXX.FIT',
    '>RHOYY','>PHSYY',
    '>RHOYY.VAR','>PHSYY.VAR',
    '>RHOYY.ERR','>PHSYY.ERR',
    '>RHOYY.FIT','>PHSYY.FIT',
    '>FRHOXY','>FPHSXY',
    '>FRHOXY.VAR','>FPHSXY.VAR',
    '>FRHOXY.ERR','>FPHSXY.ERR',
    '>FRHOXY.FIT','>FPHSXY.FIT', 
    '>FRHOXX','>FPHSXX',
    '>FRHOXX.VAR','>FPHSXX.VAR',
    '>FRHOXX.ERR','>FPHSXX.ERR',
    '>FRHOXX.FIT','>FPHSXX.FIT',
    
     #Time series-Sepctra 
     #and EM/OTHERSECT
    '>TSERIESSECT', '>TSERIES', 
    '>=SPECTRASECT', '>=EMAPSECT',
    '>=OTHERSECT',
    
    #Impedance Data Blocks

    '>ZXYR','>ZXYI',
    '>ZXY.VAR', '>ZXYR.VAR',
    '>ZXYI.VAR', '>ZXY.COV', 
    '>ZYXR','>ZYXI',
    '>ZYX.VAR', '>ZYXR.VAR',
    '>ZYXI.VAR', '>ZYX.COV',
    '>ZYYR','>ZYYI',
    '>ZYY.VAR', '>ZYYR.VAR',
    '>ZYYI.VAR', '>ZYY.COV',
    '>ZXXR', '>ZXXI',
    '>ZXXR.VAR','>ZXXI.VAR',
    '>ZXX.VAR','>ZXX.COV',
    '>FZXXR','>FZXXI',
    '>FZXYR','>FZXYI', 
    
    #Continuous 1D inversion 
    '>RES1DXX', '>DEP1DXX', 
    '>RES1DXY', '>DEP1DXY',
    '>RES1DYX', '>DEP1DYX', 
    '>RES1DYY', '>DEP1DYY',
    '>FRES1DXX', '>FDEP1DXX',
    '>FRES1DXY', '>FDEP1DXY',
    
    # Coherency and 
    #Signal Data Blocks
    '>COH','>EPREDCOH',
    '>HPREDCOH','>SIGAMP',
    '>SIGNOISE', 
    
    #Tipper Data blocks 
    '>TIPMAG','>TIPPHS',
    'TIPMAG.ERR', '>TIPMAG.FIT',
    '>TIPPHS.FIT', '>TXR.EXP',
    '>TXI.EXP', '>TXVAR.EXP',
    '>TYR.EXP','>TYI.EXP',
    '>TYVAR.EXP',
    
     # Strike, Skew, and
     # Ellipticity Data Blocks 
    '>ZSTRIKE','>ZSKEW',
    '>ZELLIP', '>TSTRIKE',
    '>TSKEW','>TELLIP',
    
    #Spatial filter blocks
    '>FILWIDTH','>FILANGLE',
    '>EQUIVLEN' , '>END'
] 

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

    **isren**: Is the list of heads columns during the data collections. Any data 
                head |ERP| data provided should be converted into 
                the following arangement::
                    
            +----------+-------------+-----------+-----------+
            |station   | resistivity | easting   | northing  | 
            +----------+-------------+-----------+-----------+
            
   **isrll**: Is the list of heads columns during the data collections. Any data 
               head |ERP| data provided should be converted into 
               the following arangement::
                   
            +----------+-------------+-------------+----------+
            |station   | resistivity | longitude   | latitude | 
            +----------+-------------+-------------+----------+
            
    **P**: Typing class for fectching the properties. For instance:: 
        
        >>> from watex._properties import P 
        >>> P.idicttags 
        ... <property at 0x1ec1f2c3ae0>
        >>> P().idictags 
        ... 
        {'station': ['pk', 'sta', 'pos'], 'longitude': ['east', 'x', 'long', 'lon'],
         'latitude': ['north', 'y', 'lat'], 'resistivity': ['rho', 'app', 'res']}
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
        >>> P().isrll 
        ... ['station','resistivity','longitude','latitude']
    
    """
    station_prefix   = [
        'pk','sta','pos'
    ]
    easting_prefix   =[
        'east','x',
                ]
    northing_prefix = [
        'north','y',
    ]
    lon_prefix   =[
        'long', 'lon'
                ]
    
    lat_prefix = [
        'lat'
    ]
    
    resistivity_prefix = [
        'rho','app','res'
    ]
    erp_headll= [
        'station', 'resistivity',  'longitude','latitude',
    ]
    erp_headen= [
        'station', 'resistivity',  'easting','northing',
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
   
    all_prefixes = { f'_{k}':v for k, v in zip (
        erp_headll +erp_headen[2:] , [
            station_prefix,
            resistivity_prefix,
            lon_prefix,
            lat_prefix, 
            easting_prefix, 
            northing_prefix,
            northing_prefix
        ]
        )}

    def __init__( self, hl =None ) :
        self.hl = hl
        for key , value in self.all_prefixes.items() : 
            self.__setattr__( key , value)
            
    
    def _check_header_item (self, it ): 
        """ Check whether the item exists in the property dictionnary."""
        for k, val in self.idicttags.items(): 
            for s in val : 
                if str(it).lower().find(s)>=0: 
                    return k 
        return  
                
    def __call__(self, hl: list = None):
        """ Rename the given header to hold the  properties 
        header values. 
        
        Call function could  return ``None`` whether the 
        given header item in `hl` does not match any item in property 
        headers. 
        
        :param hl: list or array, 
            list of the given headers 
            
        :Example: 
    
            >>> from watex._property import P 
            >>> test_v= ['pos', 'easting', 'north', 'rhoa', 'lat', 'longitud']
            >>> pobj = P(test_v)
            >>> pobj ()
            ... ['station', 'easting', 'northing', 'resistivity',
                 'latitude', 'longitude']
            >>> test_v2 = test_v + ['straa', 'nourmai', 'opirn'] 
            >>> pobj (test_v2)
            ... ['station', 'easting', 'northing', 'resistivity', 
                 'latitude', 'longitude']
        """
        v_ =list()
        
        self.hl = hl or self.hl 
        if self.hl is not None: 
            self.hl = [self.hl] if isinstance(self.hl, str ) else self.hl
            if hasattr(self.hl, '__iter__'):
                for item in self.hl : 
                    v_.append( self._check_header_item(item)) 
                v_=list(filter((None).__ne__, v_))
                return None if len (v_) ==0 else v_
            
    @property 
    def frcolortags (self): 
        """ set the dictionnary"""
        return  dict ((f'fr{k}', f'#{v}') for k, v in zip(
                        range(4), ('CED9EF','9EB3DD', '3B70F2', '0A4CEF' )
                        )
        )
    @property 
    def idicttags (self): 
        """ Is the collection of data properties """ 
        return  dict ( (k, v) for k, v in zip(self.isrll + self.isren[2:],
              [self.istation, self.iresistivity, self.ilon, 
                self.ilat, self.ieasting, self.inorthing ])
                      )
    @property 
    def istation(self) : 
        """ Use prefix to identify station location positions """
        return self._station 
    
    @property 
    def ilon (self): 
        """ Use prefix to identify longitude coordinates if given in the
        dataset. """
        return self._longitude 
    
    @property 
    def ilat(self): 
        """ Use prefix to identify latitude coordinates if given in the
        dataset. """
        return self._latitude
    @property 
    def ieasting  (self): 
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
    def isrll(self): 
        """ `SRLL` is the abbreviation of `S`for ``Stations``,`R``for 
        resistivity, `L` for ``Longitude` and `L` for ``Latitude``. 
        `SRLL` is the expected columns in Electrical resistivity profiling.
        Indeed, it keeps the traditional collections sheets
        during the survey. """
        return self.erp_headll
    
    @property 
    def isren(self): 
        """ `SREN` is the abbreviation of `S`for ``Stations``,`R``for 
        resistivity, `E` for ``easting` and `N` for ``northing``. 
        `SREN` is the expected columns in Electrical resistivity profiling.
        Indeed, it keeps the traditional collections sheets
        during the survey. """
        return self.erp_headen

class BagoueNotes: 
    """"
    `Bagoue dataset` are are Bagoue region is located in WestAfrica and lies
    between longitudes 6° and 7° W and latitudes 9° and 11° N in the north of 
    Cote d’Ivoire.
    
    The average FR observed in this area fluctuates between 1
    and 3 m3/h. Refer to the link of case story paper in the `repository 
    part <https://github.com/WEgeophysics/watex#documentation>`_ to visualize the
    location map of the study area with the geographical distribution of the
    various boreholes in the region. The geophysical data and boreholes
    data were collected from National  Office of Drinking Water (ONEP) and
    West-Africa International Drilling  Company (FORACO-CI) during  the 
    Presidential Emergency Program (PPU) in 2012-2013 and the National Drinking 
     Water Supply Program (PNAEP) in 2014.
    
    The data are firstly composed of Electrical resistivity profile (ERP) data
    collected from geophysical survey lines with various arrays such as
    Schlumberger, gradient rectangle and Wenner (α or β) and the Vertical 
    electricalsounding (VES) data carried out on the selected anomalies.
    The configuration used during the ERP is Schlumberger with distance of
    AB is 200m and MN =20m.
    """
    bagkeys = ('num',
                'name',
                'east',
                'north',
                'power', 
                'magnitude',
                'shape', 
                 'type', 
                'geol',
                'ohmS',
                'flow'
                    )
    bagvalues = [
            'Numbering the data-erp-ves-boreholes',
            'Borehole code', 
            'easting :UTM:29P-30N-WGS84', 
            'northing :UTM:29P-30N-WGS84', 
             'anomaly `power` or anomaly width in meter(m).'
             '__ref :doc:watex.utils.exmath.compute_power', 
            'anomaly `magnitude` or `height` in Ω.m '
            '__ref :doc:watex.utils.exmath.compute_magnitude', 
            'anomaly `standard fracturing index`, no unit<=sqrt(2)'
            '__ref :doc:watex.utils.exmath.compute_sfi', 
             'anomaly `shape`. Can be `V`W`M`K`L`U`H`'
             '__ref :doc:watex.utils.exmath.get_shape`', 
              'anomaly `shape`. Can be `EC`CP`CB2P`NC`__'
                'ref :doc:watex.utils.exmath.get_type`', 
            'most dominant geology structure of the area where'
                ' the erp or ves (geophysical survey) is done.', 
            'Ohmic surface compute on sounding curve in '
                'relationship with VES1D inversion (Ω.m2)'
                    '__ref :doc:`watex.core.ves`',
             'flow rate value of drilling in m3/h'
                    ]
    
   
    bagattr_infos ={
        key:val  for key, val in zip(bagkeys, bagvalues)
                    }


## XXXX properties functions 
def assert_arrangement(a: int | str ): 
    """ Assert whether the given arrangement is correct. 
    :param a: int, float, str - Type of given electrical arrangement. 
    :returns:
        - The correct arrangement name 
        - ``0`` which means ``False`` or a wrong given arrangements.   
    """
    
    for k, v in array_configuration.items(): 
        if a == k  or str(a).lower().strip() in ','.join (
                v[0]).lower() or a ==v[1]: 
            return  v[0][0].lower()
        
    return 0

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   