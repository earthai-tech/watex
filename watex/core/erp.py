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

import xml.etree.ElementTree as ET

from watex.utils._watexlog import watexlog 

_logger =watexlog.get_watex_logger(__name__)

class Features: 
    """
    Features class. Deals  with Electrical Resistivity profile (VES) and
    Vertical electrical Sounding (VES). Set all features values of sites area 
    Features class is  composed of `features_labels` for ``ML`` appliactions 
    Once parameters are computed , the class uses each features for ML purposes 
   
    """
    
    dataType ={
                ".csv":pd.read_csv, 
                 ".xlsx":pd.read_excel,
                 ".json":pd.read_json,
                 ".html":pd.read_json,
                 ".sql" : pd.read_sql
                 }  
    feature_labels = [
                        'boreh', 
                        'x_m',
                        "y_m",
                        'pa',
                        "ma",
                        "shape",
                        "type",
                        "sfi",
                        'ohms',
                        'wi', 
                        'geol',
                        'flow'
            ]
    

    def __init__(self, erpData=None, horizonDis=None , rhoApp=None,**kwargs):
        self._logging = watexlog.get_watex_logger(self.__class__.__name__)
        
        self.erp_data=erpData
        
        self.utmX =kwargs.pop('utmX', None)
        self.utmY =kwargs.pop('utmY', None)
        
        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])
            

    def from_csv(self, erp_fn):
        """
        Method essentially created to read file from csv , collected 
        horizontal distance value and apparent resistivy values. 
        then send to the class for computation purposes. 
        
        :param erp_fn: path_like string of CSV file 
        :type erp_fn: str 
        
        :return: horizontal distance im meters 
        :rtype: np.array of all data
        """
        if not os.path.isfile(erp_fn):
            raise Wex.WATexError_file_handling('{} is not a file. '
                                               'Please provide a right file !'
                                               .format(erp_fn))
        with open(erp_fn, 'r') as fcsv: 
            csvData = fcsv.readlines()
            
        # retrieve ;locations,  coordinates values for each pk, 
        # horizontal distance in meter and apparent resistivity values 
        

    def from_xml (self, xml_fn, columns=None):
        """
        collected data from xml  and build dataFrame 
        :param xxlm_fn: full path to xml file 
        :type xml: str 
        
        :param columns: list of columns of dataset 
        :type columns: list 
        
        """
        tree = ET.parse(xml_fn)

        root = tree.getroot()
        dataframe = pd.DataFrame(columns = columns)
        # loop the node and collect files from loop 
        seriesList =[]
        for ii, node in enumerate(root): 
            seriesList.append(node.find(columns[ii]).text)
            
        # after loop , use series to create pandas dataframes 
        #create dataframe but ignore index 
    
        dataframe=dataframe.append(pd.Series(seriesList, index=columns ), 
                                   ignore_index=True )
        

    def from_json (self, json_fn , indent =4):
        """
        Collected data from json files and retrieve the most insights contents 
        
        :param json_fn: json file 
        :type json_fn: str 
        
        """
    
    def data_to_numpy(self, data_fn): 
        """
        Method to get datatype and set different features into nympy array
        
        """
        if data_fn is not None : 
            self.erp_data =data_fn 
        
        if not os.path.isfile(self.erp_data): 
            raise Wex.WATexError_file_handling('{} is not a file. '
                                               'Please provide a right file !'
                                               .format(self.erp_data))
        ex_file = os.path.splitext(self.erp_data)[1] 
        if not ex_file in self.dataType.keys(): 
            pass 
            
class ERP : 
    """
    Electrical resistivity profiling class . computes and plot ERP 
    define anomalies and compute its features. can select multiples anomalies 
    on ERP and gve their features values. 
    
    
    =================  ===================  ===================================
    Attributes              Type                Description  
    =================  ===================  ===================================
    lat                 float/ndarray,1     sation latitude 
    lon                 float/ndarray,1     station longitude 
    elev                float/ndarray       station elevantion 
                                            in m or ft 
    east                float/ndarray.1     station easting coordinate (m)
    north               float/ndarray,1     station northing coordinate (m)
    azim                float/ndarray,1     station azimuth in meter (m)
    station             ndarray,1           sation id from survey
    utm_zone            str                 UTM location zone 
    resistivity         dict                resistivity value at each
                                            station (ohm.m)
    resistivity_err     dict                res.error at each  station 
    phase               dict                phase value at each station
                                            in degree 
    phase_err           dict                phase error at each station
    zxy                 dict                impedanceTensor at each station 
                                            from xy
    zxy_error           dict                imped. Tensor error  at each  
                                            station from yx
    zyx                 dict                impedanceTensor  at each station 
                                             from xy
    zyx_err             dict                imped. Tensor error  at each
                                            station from yx
    freq                ndarray,1           frequency array from survey   
    =================  ===================  ===================================

    ===========================  ==============================================
    Methods                         Description 
    ===========================  ==============================================
    _read_csamt_objs                read in CSAMT file [EDI|AVG|J]
    _read_edi_obj                   read_edi_obj and set attributes 
    _read_avg_obj                   read Zonge Eng. Avg file and set attributes
    _read_j_obj                     read A.G. Jones file and set attributes.
    ===========================  ==============================================
    
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
    
    def __init__(self, erp_fn =None , **kwargs)  : 
        self._logging =watexlog.get_watex_logger(self.__class__.__name__)
        self.erp_fn =erp_fn 
        self._Pa =None 
        self._Ma =None 
        self._anPos =None 
        self._lat =None
        self._name = None 
        self._lon =None 
        self._easting =None 
        self._northing =None 
        self._sfi = None 
        self._type =None 
        self._shape= None 
        self.utm_zone =None
        
        self.pdData =None 
        self.data=None
        
        self._fn =None 
        
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
        
        :param erp_ex:  erp_extension file
        :type erp_ex: 
        
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
        


       
if __name__=='__main__'   : 
    # pathData ='data/BagoueDataset2.xlsx'
    # ex_file =  os.path.splitext(pathData)[1]
    # print(ex_file)
    erp_data='data/l10_gbalo.xlsx'# 'data/l11_gbalo.csv'
    # df=pd.read_excel(pathData )
    
    anom_obj =ERP(erp_fn = erp_data)
    print(anom_obj._name )
    print(anom_obj._df.columns)
    print(anom_obj.df.columns)
    
    # array= df.to_numpy()
    # pk=array[:,0]
    # data=array[:,-1]
    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        