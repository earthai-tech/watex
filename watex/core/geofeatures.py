# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, zju-ufhb
# This module is part of the WATex core package, which is released under a
# MIT- licence.

"""
 .. Synopsis: Module features collects geo-electricals features computed from 
     :class:`watex.core.erp.ERP` and :class:`watex.core.ves.VES` and keeps 
     on a new dataFrame for analysis and modeling. 
     
Created on Mon Jul  5 16:51:14 2021

@author: @Daniel03

"""
from __future__ import print_function 

__docformat__='restructuredtext'

import os
import re 
import pandas as pd 
import json 
import numpy as np 
import pandas as pd 
import  watex.utils.exceptions as Wex
import watex.utils.gis_tools as gisf 

from watex.core.erp import ERP_collection 
from watex.core.ves import VES_collection 
from watex.core.geology import Geology, Borehole 

import watex.utils.wmathandtricks as wfunc
import watex.utils.gis_tools as gis

import xml.etree.ElementTree as ET

from watex.utils._watexlog import watexlog 

class Features: 
    """
    Features class. Deals  with Electrical Resistivity profile (VES), 
    Vertical electrical Sounding (VES), Geological (Geol) data and 
    Borehole data(Boreh). Set all features values of differents
    investigation sites. Features class is  composed of :: 
    
        - `erp` class  get from :class:`watex.core.erp.ERP_colection`
        - `ves` collected from :class:`watex.core.ves.VES_collection
        - `geol`  obtained from :class:`watex.core.geol.Geology` 
        - `boreh ` get from :class:`watex.core.boreh.Borehole` 
        
    Arguments: 
   -----------
            *features_fn* :str , Path_like 
                File to geoelectical  features files 
            *ErpColObjs*: object 
                    Collection object from `erp` survey lines. 
            *vesObjs*: object, 
                    Collection object from vertical electrical sounding (VES)
                    curves. 
            *geoObjs*: object, 
                    Collection object from `geol` class.
                    see :doc:`watex.core.geol.Geology`
            *boreholeObjs*: object
                    Collection of boreholes of all investigation sites.
                    Refer to :doc:`watex.core.boreh.Borehole`
    
    :Note: 
        Be sure to not miss any coordinates files. Indeed, 
        each selected anomaly should have a borehole performed at 
        that place for supervising learing. That means, each selected 
        anomaly referenced by location coordinates and `id` on `erp` must 
        have it own `ves`, `geol` and `boreh` data. 
            ... 
            
    Holds on others optionals infos in ``kwargs`` arguments: 
       
    ============  ========================  ===================================
    Attributes              Type                Description  
    ============  ========================  ===================================
    df              pd.core.DataFrame       Container of all features composed 
                                            of :attr:`~Features.featureLabels`
    site_ids        array_like              ID of each survey locations.
    site_names      array_like              Survey locations names. 
    gFname          str                     Filename of `features_fn`.                                      
    ErpColObjs      obj                     ERP `erp` class object. 
    vesObjs         obj                     VES `ves` class object.
    geoObjs         obj                     Geology `geol` class object.
    borehObjs       obj                     Borehole `boreh` class obj.
    ============  ========================  ===================================   
   
    :Note: 
        
        For furher details about classes object , please refer to the classes 
        documentation aforementionned.
        
    :Example: 
        
        >>> from watex.core.geofeatures import Features 
        >>> featurefn ='data/geo_fdata/BagoueDataset2.xlsx' 
        >>> featObj =Features(features_fn= featurefn)
        >>> featObj.site_ids
        >>> featObj.site_names
        >>> featObj.df
        
    """
    
    readFeafmt ={
                ".csv":pd.read_csv, 
                 ".xlsx":pd.read_excel,
                 ".json":pd.read_json,
                 ".html":pd.read_json,
                 ".sql" : pd.read_sql
                 }  
    featureLabels = [
                        'id', 
                        'east',
                        "north",
                        'power',
                        "magnitude",
                        "shape",
                        "type",
                        "sfi",
                        'ohmS',
                        'lwi', 
                        'geol',
                        'flow'
            ]

    def __init__(self, features_fn =None, ErpColObjs=None , vesObjs=None,
                 geoObjs=None, boreholeObjs=None,  **kwargs):
        
        self._logging = watexlog.get_watex_logger(self.__class__.__name__)
        
        self.features_fn =features_fn
        self.ErpColObjs=ErpColObjs
        self.vesObjs=vesObjs
        self.geoObjs=geoObjs
        self.borehObjs=boreholeObjs
        
        self.gFname= None 
        
        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])
            
        self._readFeatures_()   
        
    @property 
    def fn (self): 
        """ Control the Feature-file extension provide. Usefull to select 
        pd.DataFrame construction."""
        return self._fn 
    
    @fn.setter 
    def fn(self, features_fn) :
        """ Get the Features file and  seek for pd.Core methods 
        construction."""
        
        if not os.path.isfile(features_fn): 
            raise Wex.WATexError_file_handling(
                'No file detected. Could read `{0}`,`{1}`,`{2}`,'
                '`{3}` and `{4}`files.'.format(*list(self.readFeafmt.keys())))
        
        self.gFname, exT=os.path.splitext(features_fn)
        if exT in self.readFeafmt.keys(): self._fn =exT 
        else: self._fn ='?'
        self._df = self.readFeafmt[exT](features_fn)
        self.gFname = os.path.basename(self.gFname)
 
    def _readFeatures_(self, features_fn =None, ErpColObjs=None , vesObjs=None,
                 geoObjs=None, boreholeObjs=None, **kws): 
        """
        Reading class and attributes populating. Please refer to 
        :doc:`~.core.geofeatures.Features` for arguments details.
        
        """
        fimp, objsCoun =0,0 
        for nname, vval in zip(['features_fn' , 'ErpColObjs' , 'vesObjs',
                 'geoObjs', 'borehObjs'],[features_fn , ErpColObjs , vesObjs,
                 geoObjs, boreholeObjs]): 
            if vval is not None: 
                setattr(self,nname, vval )
                if nname !='features_fn':
                    objsCoun +=1
        # call object
        for fObjs in ['ErpColObjs' , 'vesObjs',
                'geoObjs', 'borehObjs']: 
            if getattr(self, fObjs) is None : 
                fimp =1        
                
        if self.features_fn is None and fimp ==1:
            raise Wex.WATexError_geoFeatures(
                'Features file is not given. Please provided specific'
                ' objects from`erp`, `ves`, `geology` and `borehole` data'
                'Call each specific collection class to build each'
                ' collection features.')
        elif self.features_fn is not None : 
            self.fn = self.features_fn 
            self.sanitize_fdataset()

            self.site_names =np.copy(self.df['id'].to_numpy())
            
            if self.utm_flag ==0 :
                # convert lat and lon to utm 

                self._easting = np.zeros_like(self.df['lat'].to_numpy())
                self._northing =np.zeros_like (self._easting)
                for ii in range(len(self._northing)):
                    try : 
                        self.utm_zone, utm_easting, utm_northing = gisf.ll_to_utm(
                                        reference_ellipsoid=23, 
                                        lat=self.df['lon'].to_numpy()[ii],
                                        lon = self.df['lat'].to_numpy()[ii])
                    except : 
                        utm_easting, utm_northing, \
                            self.utm_zone= gisf.project_point_ll2utm(
                            lat=self._longitude[ii],
                            lon = self._latitude[ii])
                        
                    self._easting[ii] = utm_easting
                    self._northing [ii] = utm_northing
            
                self.df.insert(loc=1, column ='east', value = self._easting)
                self.df.insert(loc=2, column='north', value=self._northing)
                
                try : 
                    del self.df['lat']
                    del self.df['lon']
                except : 
                    try : 
                        self.df = self.df.drop(['lat'], axis=1)
                        self.df = self.df.drop(['lon'], axis=1)
                    except : 
                        try: 
                            self.df.pop('lat')
                            self.df.pop('lon')
                        except: 
                           self._logging.debug(
                               'No way to delete `lat` and `lon` in features '
                               "dataframe. It seems there is no `lat` & `lon`"
                               " series in your dataFrame.") 
            
            #Keep location names 
            self.site_ids =np.array(['e{0}'.format(id(name.lower())) 
                                  for name in self.df['id']])
            self.df['id'] = np.array(['e{0:07}'.format(ii+1) 
                                     for ii in range(len(self.site_ids))])
            # rebuild the dataframe from main features
            self.df = pd.concat({
                featkey: self.df[featkey] 
                for featkey in self.featureLabels}, axis =1)


        if objsCoun ==4 : 
            # mean all object is provided corrected 
            # self.ErpColObjs.fnames
            #initit df
            temlen= [len(obj) for obj in [self.ErpColObjs.erpdf['id'], 
                                self.borehObjs.borehdf['id'], 
                                self.geoObjs.geoldf['id'], 
                                self.vesObjs.vesdf['id'] ]]
            
            if all(temlen) is False:
                raise Wex.WATexError_geoFeatures(
                    '`ERP`, `VES`, `Geology` and `Borehole` Features must '
                    'have the same length. You  give <{0},{1},{2}, and '
                    '{3} respectively.'.format(*temlen))
                
            
            self.df =pd.DataFrame(data = np.array((len(self.ErpColObjs.fnames), 
                                                   len(self.featureLabels))), 
                                  columns = self.featureLabels)
            
            newID= self.controlObjId(erpObjID=self.ErpColObjs.erpdf['id'], 
                              boreObjID=self.borehObjs.borehdf['id'], 
                              geolObjID=self.geoObjs.geoldf['id'], 
                              vesObjsID= self.vesObjs.vesdf['id'])
            
            self.df =self.merge(self.ErpColObjs.erpdf.drop(['id'], axis=1),
                                self.vesObjs.vesdf['ohmS'],
                                self.geoObjs.geoldf['geol'], 
                                self.borehObjs.borehdf[['lwi', 'flow']], 
                                right_index=True, 
                                left_index=True)
            
            self.df.insert(loc=0, column ='id', value = newID)

        self.df.set_index('id', inplace =True)
        self.df =self.df.astype({'east':np.float, 
                      'north':np.float, 
                      'power':np.float, 
                      'magnitude':np.float, 
                      'sfi':np.float, 
                      'ohmS': np.float, 
                      'lwi':np.float, 
                      'flow':np.float
                      })
            
   
            
            
    def sanitize_fdataset(self): 
        """ Sanitize the feature dataset. Recognize the columns provided 
        by the users and resset according to the features labels disposals
        :attr:`~Features.featureLabels`."""
        
        self.utm_flag =0
        OptsList, paramsList =[['bore', 'for'], 
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
                                ], ['id', 
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
 
        def getandReplace(optionsList, params, df): 
            """
            Function to  get parames and replace to the main features params.
            
            :param optionsList: 
                User options to qualified the features headlines. 
            :type optionsList: list
            
            :param params: Exhaustive parameters names. 
            :type params: list 
            
            :param df: pd.DataFrame collected from `features_fn`. 
            
            :return: sanitize columns
            :rtype: list 
            """
            columns = [c.lower() for c in df.columns] 
            
            for ii, celemnt in enumerate(columns): 
                for listOption, param in zip(optionsList, params): 
                     for option in listOption:
    
                         if param =='lwi': 
                            if celemnt.find('eau')>=0 : 
                                columns[ii]=param 
                                break
                         if re.match(r'^{0}+'.format(option), celemnt):
                             columns[ii]=param
                             if columns[ii] =='east': 
                                 self.utm_flag=1
                             break

    
                        
            return columns

        new_df_columns= getandReplace(optionsList=OptsList, params=paramsList,
                                      df= self._df)
        self.df = pd.DataFrame(data=self._df.to_numpy(), 
                               columns= new_df_columns)
        
 

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
        # with open(erp_fn, 'r') as fcsv: 
        #     csvData = fcsv.readlines()
            
        # retrieve ;locations,  coordinates values for each pk, 
        # horizontal distance in meter and apparent resistivity values 
        pass

    def from_xml (self, xml_fn, columns=None):
        """
        collected data from xml  and build dataFrame 
        
        :param xxlm_fn: Full path to xml file 
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
        pass 
    
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
    
    @staticmethod
    def controlObjId( erpObjID, boreObjID, geolObjID, vesObjsID): 
        """
        Control object id whether the each selected anomaly from `erp` matchs 
        with its`ves` and `geol` and `borehole`
        
        :param erpObjID: ERP object ID. Refer to :doc`~core.erp.ERP_collection` 
        :type erpObjID: str 
        
        :param boreObjID: Borehole ID.  Refer to :doc`~core.geology.Borehole`
        :type boreObjID: str 
        
        :param boreObjID: Geology ID.  Refer to :doc`~core.geology.Geology`
        :type boreObjID:str
        
        :param vesObjsID: VES object ID. see :doc:`~core.ves.VES`
        :param vesObjsID
        
        :return: New survey ID
        
        """
        new_id =np.zeros_like(erpObjID)
        for ii, ( erObj, bhObj, geolObj, vesObj) in enumerate(zip(
                                erpObjID,boreObjID,geolObjID, vesObjsID)): 
            if erObj.replace('e', '') == bhObj.replace(
                    'e', '') and erObj.replace('e', '')== bhObj.replace(
                        'e', '') and erObj.replace(
                            'e', '')== bhObj.replace('e', '') :
                
                new_id [ii] ='e{0:07}'.format(ii+1)
            else: 
                raise Wex.WATexError_geoFeatures(
                    "Survey location's name must be the same for `erp`, "
                    ' `ves` and `geol` but you are given '
                    '<{0}, {1}, {2}> respectively. Please rename'
                    ' names to be the same everywhere'.format(erObj, bhObj,
                                                           geolObj))
        return new_id 
        
        
        
if __name__=='__main__': 
    featurefn ='data/geo_fdata/BagoueDataset2.xlsx' 
    
    featObj =Features(features_fn= featurefn)
    
    print(featObj.site_ids)
    print(featObj.site_names)
    dff= featObj.df
    print(dff)
       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        