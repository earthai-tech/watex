# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, zju-ufhb
# This module is part of the WATex core package, which is released under a
# MIT- licence.

"""
===============================================================================
Copyright (c) 2021 Kouadio K. Laurent

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
===============================================================================

.. synopsis:: 'watex.core.erp'
            Module to deal with Electrical resistivity profile (ERP)
            exploration tools 


Created on Tue May 18 12:33:15 2021

@author: @Daniel03
"""
import os, re 
import json 
import numpy as np 
import pandas as pd 
import  watex.utils.exceptions as Wex
import watex.utils.wmathandtricks as wfunc
import watex.utils.gis_tools as gis

from watex.utils._watexlog import watexlog 

_logger =watexlog.get_watex_logger(__name__)


            
class ERP : 
    """
    Electrical resistivity profiling class . computes and plot ERP 
    define anomalies and compute its features. can select multiples anomalies 
    on ERP and gve their features values. 
    
    Arguments: 
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
                    
            
    :Note: Provide the `posMinMax` is strongly recommended for accurate 
            geo-electrical features computation. If not given, the best anomaly 
            will be selected automatically and probably could not match what you 
            expect.
            ... 
            
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

    - To get the geo-electrical-features,  create an `erp` object by calling: 
        
        >>> from watex.core.erp import ERP 
        >>> anomaly_obj =ERP(erp_fn = '~/location_filename')
        
    The call the following `erp` properties attributes:
    
    =================  ===================  ===================================
    properties          Type                Description  
    =================  ===================  ===================================
    select_best_point_      float           Best anomaly position points 
    select_best_value_      float           Best anomaly app.resistivity value.
    best_points             float           Best positions points selected 
                                            automatically. 
    abest_sfi               float           Best anomaly standart fracturation 
                                            index value. 
    abest_anr               float           Best 
    abest_power             float           Best anomaly power  in *meter(m)*.
    abest_magnitude         float           Best anomlay magnitude in *ohm.m*
    abest_shape             str             Best anomaly shape. can be ``V``, 
                                            ``W``,``K``, ``H``, ``C``, ``M``.
    abest_type              str             Best anomaly type. Can be : 
                                            - ``EC`` for Extensive conductive. 
                                            - ``NC`` for narrow conductive. 
                                            - ``CP`` for conductive PLANE 
                                            - ``CB2P`` for contact between two
                                            planes. 
    =================  ===================  ===================================
    
    :Example: 
        
        >>> from watex.core.erp import ERP  
        >>> anom_obj= ERP(erp_fn = 'data/l10_gbalo.xlsx', auto=False, 
        ...                  posMinMax= (90, 130),turn_off=True)
        >>> anom_obj.name 
        ... l10_gbalo
        >>> anom_obj.select_best_point_
        ...110 
        >>> anom_obj.select_best_value_
        ...132
        >>> anom_obj.abest_magnitude
        ...5
        >>> nom_obj.abest_power
        ..40
        >>> anom_obj.abest_sfi
        ...1.9394488747363936
        >>> anom_obj.abest_anr
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
    
    def __init__(self, erp_fn =None , dipole_length =10., auto =False, posMinMax=None, **kwargs)  : 
        """ Read :ref:`erp` file and  initilize  the following
        attributes attributes. Set `auto` to ``True`` to let the program 
        selecting the best anomaly points. """
        
        self._logging =watexlog.get_watex_logger(self.__class__.__name__)

        self.erp_fn =erp_fn 
        self._dipoleLength =dipole_length
        self.auto =auto 
        
        self.anom_boundaries = posMinMax
        self._select_best_point =kwargs.pop('best_point', None)
        self.turn_on =kwargs.pop('turn_on', False)
        self.display_auto_infos= kwargs.pop ('display_autoinfos', False)
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
        self.utm_zone =None
        
        
        self.data=None
        
        self._fn =None 
        
        
        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])
            
        if self.auto is False and posMinMax is None : 
            raise Wex.WATexError_ERP('Automatic trigger is set to ``False``.'
                                     'Please provide anomalylocation via'
                                     'its positions boundaries. Can be a tuple'
                                     'or a list of startpoint and endpoint.')
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
            raise Wex.WATexError_file_handling(
                'No right file detected ! Please provide the right path.')
        name , exT=os.path.splitext(self.erp_fn)

        if exT in self.dataType.keys(): 
            self._fn =exT 
        else: self._fn ='?'
        
        self._df = self.dataType[exT](self.erp_fn)
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
            lon_array = self.df['lon'].to_numpy()
            lat_array = self.df['lat'].to_numpy()
            easting= np.zeros_like(lon_array)
            northing = np.zeros_like (lat_array)

            for ii in range(len(lon_array)): 
                self.utm_zone, utm_easting, utm_northing = gis.ll_to_utm(
                                        reference_ellipsoid=23, 
                                          lat=lon_array[ii],
                                          lon = lat_array [ii])
                easting[ii] = utm_easting
                northing [ii] = utm_northing
            
            self.df.insert(loc=1, column ='east', value = easting)
            self.df.insert(loc=2, column='north', value=northing)
            
        # get informations form anomaly 
        
        self.aBestInfos= wfunc.select_anomaly(
                            rhoa_array= self.df['rhoa'].to_numpy(), 
                             pos_array= self.df['pk'].to_numpy(), 
                             auto = self.auto, 
                             dipole_length=self._dipoleLength , 
                             pos_bounds=self.anom_boundaries, 
                             pos_anomaly = self._select_best_point, 
                             display_infos=self.display_auto_infos
                             )
        
        self._best_keys_points = list(self.aBestInfos.keys())
        
        for ckey in self._best_keys_points : 
            if ckey.find('1_pk')>=0 : 
                self._best_key_point = ckey 
                break 
        

    def sanitize_columns(self): 
        """
        Get the columns of :ref:`erp` dataframe and set new names according to 
        :class:`~watex.core.ERP.erpLabels` . 
    
        """ 

        self.coord_flag=0
        columns =[ c.lower() for c in self._df.columns]

        for ii, sscol in enumerate(columns): 
            if re.match(r'^sta+', sscol) or re.match(r'^site+', sscol) : 
                columns[ii] = 'pk'
            if re.match(r'>east+', sscol) or re.match(r'^x+', sscol): 
                columns[ii] = 'east'
            if re.match(r'>north+', sscol) or re.match(r'^y+', sscol): 
                columns[ii] = 'north'
            if re.match(r'>lon+', sscol): 
                columns[ii] = 'lon'
                self._coord_flag = 1
            if re.match(r'>lat+', sscol):
                columns[ii] = 'lat'
            if re.match(r'^rho+', sscol) or re.match(r'^res+', sscol): 
                columns[ii] = 'rhoa'

        self.df =pd.DataFrame(data =self.data, columns= columns)
        
    @property
    def select_best_point_(self): 
        """ Select the best anomaly points."""
        self._select_best_point_= self.aBestInfos[self._best_key_point][0]
        
        
        mes ='The best point is found  at position (pk) = {0} m. '\
            '----> Station number {1}'.format(self._select_best_point_,
                                              int(self._select_best_point_/self.dipoleLength)+1
                                              )
        wfunc.wrap_infos(mes, on =self.turn_on) 
        
        return self._select_best_point_
    
    @property 
    def dipoleLength(self): 
        """Get the dipole length  i.e the distance between two measurement."""
        
        wfunc.wrap_infos(
            'Distance bewteen measurement is = {0} m.'.
            format(self._dipoleLength), off = self.turn_on)
        
        return self._dipoleLength
    
    @property 
    def best_points (self) : 
        """ Get the best points from auto computation """
        
        mess =['{0} best points was found :\n '] 
        self._best_points ={}
        for ii,  bp in enumerate (self._best_keys_points): 
            cods = float(bp.replace('{0}_pk'.format(ii+1), ''))
            pmes='{0} : position = {1} m ----> rhoa = {2} Ω.m\n'.format(
                ii+1, cods, 
                self.aBestInfos[bp][1]) 
            mess.append(pmes)
            self._best_points['{}'.format(cods)]=self.aBestInfos[bp][1]
            
        mess[-1]=mess[-1].replace('\n', '')
        
        wfunc.wrap_infos(''.join([ss for ss in mess]),
                         on = self.turn_on)
        return self._best_points  
    
    @property 
    def abest_power (self):
        """Get the power from the select :attr:`select_best_point_`"""
        self._power =wfunc.compute_power(
            posMinMax=self.aBestInfos[self._best_key_point][2])
        
        wfunc.wrap_infos(
            'The power of selected best point is = {0}'.format(self._power),
                        on = self.turn_on)
        
        return self._power 
    @property 
    def abest_magnitude(self): 
        """ Get the magnitude of the select :attr:`select_best_point_"""
        
        self._magnitude =wfunc.compute_magnitude(
            rhoaMinMax= self.aBestInfos[self._best_key_point][3])
        
        wfunc.wrap_infos(
           'The magnitude of selected best point is = {0}'.
           format(self._magnitude),
          on = self.turn_on)
        
        return self._magnitude
    
    @property 
    def abest_sfi(self) : 
        """Get the standard fraturation index from 
        :attr:`select_best_point_"""
        
        self._sfi = wfunc.compute_sfi(pk_min=self.posi_min,
                                      pk_max=self.posi_max,
                                      rhoa_min=self.rhoa_min,
                                      rhoa_max=self.rhoa_max,
                                      rhoa=self.select_best_value_, 
                                      pk=self.select_best_point_)
        
        wfunc.wrap_infos('SFI computed at the selected best point is = {0}'.
                        format(self._sfi), 
                        on =self.turn_on)
        return self._sfi
    
    @property 
    def posi_max (self):
        """Get the right position of :attr:`select_best_point_ boundaries 
        using the station locations of unarbitrary positions got from
        :attr:`dipoleLength`."""
        
        return np.array(self.aBestInfos[self._best_key_point][2]).max()
    
    @property 
    def posi_min (self):
        """Get the left position of :attr:`select_best_point_ boundaries 
        using the station locations of unarbitrary positions got from
        :attr:`dipoleLength`."""
        
        return np.array(self.aBestInfos[self._best_key_point][2]).min()
        
    @property 
    def rhoa_min (self):
        """Get the buttom position of :attr:`select_best_point_ boundaries 
        using the magnitude  got from :attr:`abest_magnitude`."""
    
        return np.array(self.aBestInfos[self._best_key_point][3]).min()
    
    @property 
    def rhoa_max (self):
        """Get the top position of :attr:`select_best_point_ boundaries 
        using the magnitude  got from :attr:`abest_magnitude`."""
    
        return np.array(self.aBestInfos[self._best_key_point][3]).max()
           
    @property
    def select_best_value_(self): 
        """ Select the best anomaly points."""   
        self._select_best_value= float(
            self.aBestInfos[self._best_key_point][1]
            )
        
        wfunc.wrap_infos('Best conductive value selected is = {0} Ω.m'.
                        format(self._select_best_value), 
                        on =self.turn_on) 
        
        return self._select_best_value
        
    @property 
    def abest_anr (self ): 
        """Get the select best anomaly ratio `abest_anr` along the
        :class:`~watex.core.erp.ERP`"""
        
        pos_min_index = int(np.where(self.df['pk'].to_numpy(
            ) ==self.posi_min)[0])
        pos_max_index = int(np.where(self.df['pk'].to_numpy(
            ) ==self.posi_max)[0])

        self._anr = wfunc.compute_anr(sfi = self.abest_sfi,
                                      rhoa_array = self.df['rhoa'].to_numpy(), 
                                      pos_bound_indexes= [pos_min_index ,
                                                          pos_max_index ])
        wfunc.wrap_infos('Best cover   = {0} % of the whole ERP line'.
                        format(self._anr*100), 
                        on =self.turn_on) 
        
        return self._anr
    
    @property 
    def abest_type (self): 
        """ Get the select best anomaly type """
        self._type = get_type(erp_array= self.df['rhoa'].to_numpy() , 
                              posMinMax = self.anom_boundaries , 
                              pk= self.select_best_point_ ,
                              pos_array=self.df['pk'].to_numpy() , 
                              dl= self.dipoleLength)
        
        wfunc.wrap_infos('Select anomaly type is = {}'.
                       format(self._type), 
                       on =self.turn_on) 
        return self._type 
    
    @property 
    def abest_shape (self) : 
        """ Find the selected anomaly shape"""
        
        self._shape = get_shape(
            rhoa_range=self.aBestInfos[self._best_key_point][4])
        
        wfunc.wrap_infos('Select anomaly shape is = {}'.
                       format(self._shape), 
                       on =self.turn_on) 
        return self._shape 
    
def get_shape (rhoa_range): 
    """
    Find anomaly `shape`  from apparent resistivity values framed to
    the best points. 
 
    :param rhoa_range: The apparent resistivity from selected anomaly bounds
                        :attr:`~core.erp.ERP.anom_boundaries`
    :type rhoa_range: array_like or list 
    
    :returns: 
        - V
        - W
        - K 
        - C
        - M
        - U
    
    :Example: 
        
        >>> from watex.core.erp import get_shape 
        >>> x = [60, 70, 65, 40, 30, 31, 34, 40, 38, 50, 61, 90]
        >>> shape = get_shape (rhoa_range= np.array(x))
        ...U

    """
    from scipy.signal import argrelextrema 
    # find minimum locals.
    minlocals = argrelextrema(rhoa_range, np.less)

    shape ='V'
    average_curve = rhoa_range.mean()
    if len (minlocals[0]) >1 : 
        shape ='W'
        average_curve = rhoa_range.mean()
        minlocals_slices = rhoa_range[int(minlocals[0][0]):int(minlocals[0][-1])+1]
        average_minlocals_slices  = minlocals_slices .mean()

        if average_curve >= 1.2 * average_minlocals_slices: 
            shape = 'U'
            if rhoa_range [-1] < average_curve and\
                rhoa_range [-1]> minlocals_slices[
                    int(argrelextrema(minlocals_slices, np.greater)[0][0])]: 
                shape ='K'
        elif rhoa_range [0] < average_curve and \
            rhoa_range [-1] < average_curve :
            shape ='M'
    elif len (minlocals[0]) ==1 : 
        if rhoa_range [0] < average_curve and \
            rhoa_range [-1] < average_curve :
            shape ='M'
        elif rhoa_range [-1] <= average_curve : 
            shape = 'C'
            
    return shape 

def get_type (erp_array, posMinMax, pk, pos_array, dl): 
    """
    Find anomaly type from app. resistivity values and positions locations 
    
    :param erp_array: App.resistivty values of all `erp` lines 
    :type erp_array: array_like 
    
    :param posMinMax: Selected anomaly positions from startpoint and endpoint 
    :type posMinMax: list or tuple or nd.array(1,2)
    
    :param pk: Position of selected anomaly in meters 
    :type pk: float or int 
    
    :param pos_array: Stations locations or measurements positions 
    :type pos_array: array_like 
    
    :param dl: 
        
        Distance between two receiver electrodes measurment. The same 
        as dipole length in meters. 
    
    :returns: 
        - ``EC`` for Extensive conductive. 
        - ``NC`` for narrow conductive. 
        - ``CP`` for conductive PLANE 
        - ``CB2P`` for contact between two planes. 
        
    :Example: 
        
        >>> from watex.core.erp import get_type 
        >>> x = [60, 61, 62, 63, 68, 65, 80,  90, 100, 80, 100, 80]
        >>> pos= np.arange(0, len(x)*10, 10)
        >>> ano_type= get_type(erp_array= np.array(x),
        ...            posMinMax=(10,90), pk=50, pos_array=pos, dl=10)
        >>> ano_type
        ...CB2P

    """
    # Get position index 
    anom_type ='PC'
    index_pos = int(np.where(pos_array ==pk)[0])
    if erp_array [:index_pos +1].mean() < np.median(erp_array) or\
        erp_array[index_pos:].mean() < np.median(erp_array) : 
            anom_type ='CB2P'
    elif dl <= (max(posMinMax)- min(posMinMax)) <= 4* dl : 
        anom_type = 'NC'
    elif (max(posMinMax)- min(posMinMax))> 4 *dl and (
            erp_array [:index_pos +1].mean() >= np.median(erp_array) or 
            erp_array[index_pos:].mean() >= np.median(erp_array) ): 
        anom_type = 'EC'

    return anom_type
            
         
if __name__=='__main__'   : 

    erp_data='data/erp/l10_gbalo.xlsx'# 'data/l11_gbalo.csv'
    
    anom_obj =ERP(erp_fn = erp_data, 
                  auto=False, posMinMax=(90, 130),turn_off=True)

    print(anom_obj.abest_type) 
    print(anom_obj.abest_shape)
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        