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
import os, re , sys
import json , warnings
import datetime, shutil
import numpy as np 
import pandas as pd
from scipy.signal import argrelextrema 

from watex.utils.__init__ import savepath as savePath  
import  watex.utils.exceptions as Wex
import watex.utils.wmathandtricks as wfunc
import watex.utils.func_utils as func
import watex.utils.gis_tools as gis

from watex.utils._watexlog import watexlog 

_logger =watexlog.get_watex_logger(__name__)

sys.path.insert(0, os.path.abspath('.'))

class ERP_collection: 
    """
    Collection objects. The class collects all `erp` survey lines.
    Each `erp` is an singleton class object with their corresponding 
    attributes. The goal is to build a container  geao-elecricals to 
    straigthforwardly given to :class:`~watex.core.geofeatures.Features`
    class.
    
    Arguments:
    ----------
            *listOferpfn*: list, ndarray
                        list of different `erp` files. 
            *listOfposMinMax* : list 
                        collection of different selected anomaly boundaries. 
                        If not provided, the :attr:`~.core.erp.ERP.auto` 
                        will triggered. It's recommanded to provided for all 
                        `erp` your convenient anomaly boundaries like:: 
                            
                            listOfposMinMax=[(90, 130), (10, 70), ...]
                        
                        where ``(90,130)``is boundaries of selected anomaly on 
                        the first `erp` line and ``(10,70)``is the boundaries
                        of the second `erp` survey line and so on. 
            *erpObjs*: list, ndarray 
                Collection of objects from :class:~core.erp.ERP`. If objects 
                are alread created. Gather them on a list and pass though the 
                argument `erpObjs`
    
    Holds on others optionals infos in ``kws`` arguments: 
       
    ======================  ==============  ===================================
    Attributes              Type                Description  
    ======================  ==============  ===================================
    list_of_dipole_lengths  list            Collection of `dipoleLength`. User 
                                            can provide the distance between 
                                            sites measurements as performed on 
                                            investigations site. If given, the 
                                            automaticall `dipoleLength` compu-
                                            tation will be turned off. 
    fnames                 array_like       Array of `erp`survey lines name. 
                                            If each survey name is the location 
                                            name then will keep it. 
    id                      array_like      Each `erp`obj reference numbers
    erps_data               nd.array        Array composed of geo-electrical
                                            parameters. ndarray(nerp, 8) where 
                                            num is the number of `erp`obj
                                            collected. 
    erpdf                   pd.DataFrame    A dataFrame of collected `erp` line 
                                            and the number of lines correspond 
                                            to the number of collected `erp`.
    ======================  ==============  ===================================
        
    It's posible to get from each `erp` collection the singular array of 
    different parameters considered as properties params:: 
        
        >>> from watex.core.erp import ERP_collection as ERPC
        >>> erpcol = ERPC(listOferpfn='list|path|filename')
        >>> erpcol.survey_ids
        >>> erpcol.selectedPoints
        
   Call the following :class:`~.erp.ERP_collection` properties attributes:
    
    ====================  ================  ===================================
    properties              Type                Description  
    ====================  ================  ===================================
    selectedPoints          array_like      Collection of Best anomaly 
                                            position points 
    survey_ids              array_like      Collection of all `erp` survey 
                                            survey ids. :Note:Each ids is fol-
                                            lowing by the prefix **e**.
    sfis                    array_like      Collection of best anomaly standard 
                                            fracturation index value. 
    powers                  array_like      Collection of best anomaly `power`
    magnitudes              array_like      Colection of best anomaly
                                            magnitude in *ohm.m*
    shapes                  array_like      Collection of best anomaly shape. 
                                            For more details please refer to
                                            :doc:`~core.erp.ERP`.
    types                   array_like      Collection of best anomaly type. 
                                             Refer to :doc:`~core.erp.ERP` for
                                             more details.
    ====================  ================  ===================================
        
    :Example: 
        
        >>> from watex.core.erp import ERP_collection 
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
        self.export_data =kws.pop('export_data', False)
        self.export_fex= kws.pop('file_extension', 'csv')
        
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
                    raise Wex.WATexError_ERP(
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
   
                    erpObj = ERP(erp_fn= erp_filename,
                                dipole_length=self.dipoleLengthObjs[ii], 
                                posMinMax=self.anomBoundariesObj[ii])
                except : 
                    
                    unreadfiles.append(os.path.basename(erp_filename))
                    print(fmterp_text.format(ii+1,
                                             os.path.basename(erp_filename), 
                                             'Failed'))
                    
                else: 
                    print(fmterp_text.format(ii+1,
                                             os.path.basename(erp_filename), 
                                             'Passed'))
                    self.erpObjs.append(erpObj)
            print('-'*70)
             
        if len(unreadfiles)>=1 : 
            self._logging.error (
                f'Unable to read the files `{unreadfiles}`')
            warnings.warn(f'Unable to read file `{unreadfiles}`.'
                          ' Please check your files.')
            
            print(' --!> {0} ERP file not read. Please check your file'
                  f' {"s" if len(unreadfiles)>1 else ""} ='
                  '`{0}`'.format(len(unreadfiles), unreadfiles))
            
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
        self.erps_data= func.concat_array_from_list([
                                        self.survey_ids, 
                                        self.easts, 
                                        self.norths, 
                                        self.powers, 
                                        self.magnitudes, 
                                        self.shapes, 
                                        self.types, 
                                        self.sfis], concat_axis=1)
        self.erpdf =pd.DataFrame(data = self.erps_data, 
                                  columns=self.erpColums) 
                                  
        self.erpdf=self.erpdf.astype( {'east':np.float, 
                                        'north': np.float, 
                                        'power': np.float, 
                                        'magnitude':np.float, 
                                        'sfi':np.float})
        
        if self.export_data is True : 
            self.exportErp()
            
    def get_property_infos(self, attr_name , objslist =None): 
        """
        From each obj `erp` ,get the attribute infos and set 
        on data array 
         
        :param attr_name: 
            Name of attribute to get the informations of the properties 
             
        :type attra_name: str 
        
        :param objslist: list of collection objects 
        :type objslist; list 
        
        :Example:
            
            >>> from watex.core.erp.ERP_collection as ERPcol
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
            
        if self.export_fex =='xlsx':
            
            self.erpdf.to_excel(self.filename , sheet_name='data',
                                index=False) 
        elif self.export_fex =='csv': 
            
            self.erpdf.to_csv(self.filename, header=True,
                              index =False)

        if self.savepath is None :
            self.savepath = savePath('_erpData_')
            
        if self.savepath is not None :
            if not os.path.isdir(self.savepath): 
                self.savepath = savePath('_erpData_')
            try : 
                shutil.move(os.path.join(os.getcwd(),self.filename) ,
                        os.path.join(self.savepath , self.filename))
            except : 
                self.logging.debug("We don't find any path to save ERP data.")
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
        return self.get_property_infos('selected_best_point_')
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

    - To get the geo-electrical-features,  create an `erp` object by calling:: 
        
        >>> from watex.core.erp import ERP 
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
        # self.display_auto_infos= kwargs.pop ('display', False)
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
            raise Wex.WATexError_file_handling(
                'No right file detected ! Please provide the right path.')
        name , exT=os.path.splitext(self.erp_fn)

        if exT in self.dataType.keys(): 
            self._fn =exT 
        else: self._fn ='?'
        
        df_ = self.dataType[exT](self.erp_fn)
        # Check into the dataframe whether the souding location and anomaly 
        #boundaries are given 
        self.auto, self._shape, self._type, self._select_best_point,\
            self.anom_boundaries, self._df =\
                wfunc.sanitizedf_and_findBoundaries(df_)
 
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
            self._longitude= self.df['lon'].to_numpy()
            self._latitude = self.df['lat'].to_numpy()
            easting= np.zeros_like(self._longitude)
            northing = np.zeros_like (self._latitude)

            for ii in range(len(self._longitude)):
                try : 
                    self.utm_zone, utm_easting, utm_northing = gis.ll_to_utm(
                                            reference_ellipsoid=23, 
                                              lon=self._longitude[ii],
                                              lat = self._latitude[ii])
                except : 
                    utm_easting, utm_northing, \
                        self.utm_zone= gis.project_point_ll2utm(
                        lon=self._longitude[ii],
                        lat = self._latitude[ii])
                    
                easting[ii] = utm_easting
                northing [ii] = utm_northing
            
            self.df.insert(loc=1, column ='east', value = easting)
            self.df.insert(loc=2, column='north', value=northing)
            
        # get informations from anomaly 
        if self.coord_flag ==0 : 
            # compute  latitude and longitude coordinates if not given 
            self._latitude = np.zeros_like(self.df['east'].to_numpy())
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
            
            for ii, (north, east) in enumerate(zip(self.df['north'].to_numpy(),
                                                self.df['east'].to_numpy())): 
                try : 
                    self._latitude [ii],\
                        self._longitude [ii] = gis.utm_to_ll(23,
                            northing = north, 
                            easting = east, 
                            zone = self.utm_zone) 
                except: 
                    self._latitude[ii], \
                        self._longitude [ii] = gis.project_point_utm2ll(
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
            
        if self.turn_on in ['off', False]: self.turn_on =False 
        elif self.turn_on in ['on', True]: self.turn_on =True 
        else : self.turn_on =False 
        
    
        if self._dipoleLength is None : 
            self._dipoleLength=(self.df['pk'].to_numpy().max()- \
                self.df['pk'].to_numpy().min())/(len(
                    self.df['pk'].to_numpy())-1)
        self.aBestInfos= wfunc.select_anomaly(
                            rhoa_array= self.df['rhoa'].to_numpy(), 
                             pos_array= self.df['pk'].to_numpy(), 
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
        Get the columns of :ref:`erp` dataframe and set new names according to 
        :class:`~watex.core.ERP.erpLabels` . 
    
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
        
        wfunc.wrap_infos(''.join([ss for ss in mess]),
                         on = self.turn_on)
        return self._best_points  
    
    @property 
    def best_power (self):
        """Get the power from the select :attr:`select_best_point_`"""
        self._power =wfunc.compute_power(
            posMinMax=self.aBestInfos[self._best_key_point][2])
        
        wfunc.wrap_infos(
            'The power of selected best point is = {0}'.format(self._power),
                        on = self.turn_on)
        
        return self._power 
    @property 
    def best_magnitude(self): 
        """ Get the magnitude of the select :attr:`select_best_point_"""
        
        self._magnitude =wfunc.compute_magnitude(
            rhoa_max=self.rhoa_max,rhoa_min=self.select_best_value_)
                                                 
        wfunc.wrap_infos(
           'The magnitude of selected best point is = {0}'.
           format(self._magnitude),
          on = self.turn_on)
        
        return self._magnitude
    
    @property 
    def best_sfi(self) : 
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
    def best_anr (self ): 
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
    def best_type (self): 
        """ Get the select best anomaly type """
        if self._type is None: 
            self._type = get_type(erp_array= self.df['rhoa'].to_numpy() , 
                                  posMinMax = np.array([float(self.posi_max),
                                                        float(self.posi_min)]), 
                                  pk= self.select_best_point_ ,
                                  pos_array=self.df['pk'].to_numpy() , 
                                  dl= self.dipoleLength)
        
        wfunc.wrap_infos('Select anomaly type is = {}'.
                       format(self._type), 
                       on =self.turn_on) 
        return self._type 
    
    @property 
    def best_shape (self) : 
        """ Find the selected anomaly shape"""
        if self._shape is None: 
            self._shape = get_shape(
                rhoa_range=self.aBestInfos[self._best_key_point][4])
        
        wfunc.wrap_infos('Select anomaly shape is = {}'.
                       format(self._shape), 
                       on =self.turn_on) 
        return self._shape 
    
    
    @property
    def best_east(self): 
        """ Get the easting coordinates of selected anomaly"""
        
        self._east = self.df['east'].to_numpy()[self.best_index]
        return self._east
    
    @property
    def best_north(self): 
        """ Get the northing coordinates of selected anomaly"""
        self._north = self.df['north'].to_numpy()[self.best_index]
        return self._north
        
    @property 
    def best_index(self): 
        """ Keeop the index of selected best anomaly """
        v_= (np.where( self.df['pk'].to_numpy(
            )== self.select_best_point_)[0]) 
        if len(v_)>1: 
            v_=v_[0]
        return int(v_)
            
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
        
        Distance between two receiver electrodes measurement. The same 
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
    anom_type ='CP'
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

def get_shape(rhoa_range): 
    
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
    shape ='V'
    
    minlocals_ix, = argrelextrema(rhoa_range, np.less)
    maxlocals_ix, = argrelextrema(rhoa_range, np.greater)
    value_of_median = np.median(rhoa_range)
    
    coef_UH = 1.2 
    c_=[rhoa_range[0] , rhoa_range[-1] ]

    if len(minlocals_ix)==0 : 
        if len(maxlocals_ix)==0 and\
            (max(c_) and min(c_)) > value_of_median : 
            return 'U'
        
        return 'C' 

    if len(minlocals_ix) ==1 : 

        if max(c_) > np.median(rhoa_range) and min(c_) <  value_of_median/2: 
            return 'C'

        elif rhoa_range[minlocals_ix] > value_of_median or \
            rhoa_range[minlocals_ix] > max(c_): 
            return 'M'
        
    if len(minlocals_ix)>1 : 
        if (rhoa_range[0] or rhoa_range[-1]) > value_of_median:
            shape ='W'
            
            if max(c_) > value_of_median and\
                min(c_) > value_of_median: 
                if rhoa_range[maxlocals_ix].mean()> value_of_median : 
                    if  coef_UH * rhoa_range[minlocals_ix].mean(): 
                        shape ='H'
                        
                        coef_UH = 1.
                        
                        if rhoa_range[minlocals_ix].mean() <= coef_UH * \
                            rhoa_range[maxlocals_ix].mean():
                            shape = 'U'
                        
            else : shape ='K'
            
        elif (rhoa_range[0] and rhoa_range[-1]) < np.median(rhoa_range): 
            shape =  'M'    

        return shape 
    
                
    
        
    return shape             
         
if __name__=='__main__'   : 
    erp_data='data/erp/l10_gbalo.xlsx'# 'data/l11_gbalo.csv'
    erp_path ='data/erp/test_anomaly.xlsx'
    test_fn = 'l10_gbalo.xlsx'
    # erpObj =ERP(erp_fn=erp_path, turn_on ='off')
    # erpObj.best_rhoaRange
    # pathBag = r'F:\repositories\watex\data\Bag.main&rawds\ert_copy\nt'
    
    # erpObjs =ERP_collection(listOferpfn= pathBag, 
    #                         )
    # alldfs= erpObjs.erpdf
    # erpObjs.exportErp()
    # print(erpObjs.survey_ids)
    # gFname, exT=os.path.splitext(test_fn)
    # gFame = os.path.basename(gFname)
    # print(gFname, exT)
    # erpObj = ERP(erp_path, posMinMax=(0, 80))
    # print(erpObj.best_type)
    # print(erpObj.best_power)
    # print(erpObj.best_magnitude)
    # print(erpObj.best_shape)
    # print(erpObj.rhoa_max)
    # print(erpObj.rhoa_min)
    # print(erpObj.best_rhoaRange)
    # print(erpObj.best_points)
    # print(erpObj.best_sfi)
    # print(erpObj.select_best_point_)
    # print(erpObj.select_best_value_)
    test=np.array([95,  130,  163,  140,  167,  154, 93])#, 140, 167])
    print(get_shape(test))
    

    
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        