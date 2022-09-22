# -*- coding: utf-8 -*-
# Copyright (c) 2021 LKouadio
# MIT- licence.

from __future__ import print_function 

import os
import re 
import xml.etree.ElementTree as ET
import pandas as pd 
import numpy as np 

from ..tools.funcutils import  ( 
    savepath_ , 
    sanitize_fdataset, 
    categorize_flow, 
    )
    
from ..typing import ( 
    Iterable,
    T
    )

from .base import  ( 
    exportdf, 
    ) 
from ..property import ( 
    Config 
    )
from ..decorators import ( 
    writef
    ) 
from ..exceptions import ( 
    FileHandlingError, 
    FeatureError, 
    ArgumentError, 
    ParameterNumberError, 
   )
from ..tools.gistools import ( 
    ll_to_utm, 
    project_point_ll2utm
    
    )
from watex._watexlog import watexlog 

_logger =watexlog().get_watex_logger(__name__)


__docformat__='restructuredtext'

class GeoFeatures: 
    """
    Features class. Deals  with Electrical Resistivity profile (VES), 
    Vertical electrical Sounding (VES), Geological (Geol) data and 
    Borehole data(Boreh). Set all features values of differents
    investigation sites. Features class is  composed of :: 
    
    - `erp` class  get from :class:`watex.methods.erp.ERP_colection`
    - `geol`  obtained from :class:`watex.geology.geology.Geology` 
    - `boreh` get from :class:`watex.geology.geology.Borehole` 
    
    Arguments 
    -----------
    *features_fn* :str , Path_like 
        File to geoelectical  features files.
        
    *ErpColObjs*: object 
        Collection object from `erp` survey lines. 
        
    *vesObjs*: object, 
        Collection object from vertical electrical sounding (VES) curves.
        
    *geoObjs*: object, 
        Collection object from `geol` class. See :doc:`watex.geology.geology.Geology`.
        
    *boreholeObjs*: object
        Collection of boreholes of all investigation sites.
        Refer to :doc:`watex.geology.geology.Borehole`


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
    
         
    Notes 
    ------
    Be sure to not miss any coordinates files. Indeed, each selected anomaly
    should have a borehole performed at that place for supervising learing.
    That means, each selected anomaly referenced by location coordinates and 
    `id` on `erp` must have it own `ves`, `geol` and `boreh` data. For furher
    details about classes object , please refer to the classes documentation 
    aforementionned.

    Examples
    ---------
    >>> from watex.bases.features import GeoFeatures 
    >>> featurefn ='data/geo_fdata/BagoueDataset2.xlsx' 
    >>> featObj =Features(features_fn= featurefn)
    >>> featObj.site_ids
    >>> featObj.site_names
    >>> featObj.df
     
    """
    
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
        cpl = Config().parsers
        if not os.path.isfile(features_fn): 
            raise FileHandlingError (
                'No file detected. Could read `{0}`,`{1}`,`{2}`,'
                '`{3}` and `{4}`files.'.format(*list(cpl.keys())))
        
        self.gFname, exT=os.path.splitext(features_fn)
        if exT in cpl.keys(): self._fn =exT 
        else: self._fn ='?'
        self._df = cpl[exT](features_fn)
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
            raise FeatureError (
                'Features file is not given. Please provide specific'
                ' objects from`erp`, `ves`, `geology` and `borehole` data'
                'Call each specific collection class to build each'
                ' collection features.')
        elif self.features_fn is not None : 
            self.fn = self.features_fn 
            self.sanitize_fdataset()
            try : 
                self.site_names =np.copy(self.df['id'].to_numpy())
            except KeyError: 
                 # force to set id 
                self.df=self.df.rename(columns = {'name':'id'})
                self.site_names =np.copy(self.df['id'].to_numpy())
                # self._index_col_id ='id'
            
            if self.utm_flag ==0 :
                # convert lat and lon to utm 

                self._easting = np.zeros_like(self.df['lat'].to_numpy())
                self._northing =np.zeros_like (self._easting)
                for ii in range(len(self._northing)):
                    try : 
                        self.utm_zone, utm_easting, utm_northing = ll_to_utm(
                                        reference_ellipsoid=23, 
                                        lat=self.df['lon'].to_numpy()[ii],
                                        lon = self.df['lat'].to_numpy()[ii])
                    except : 
                        utm_easting, utm_northing, \
                            self.utm_zone= project_point_ll2utm(
                            lat=self.df['lat'].to_numpy()[ii],
                            lon = self.df['lon'].to_numpy()[ii])
                        
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
                               'No way to remove `lat` and `lon` in features '
                               "dataFrame. It seems there is no `lat` & `lon`"
                               " pd.series in your dataFrame.") 
            
            #Keep location names 
            self.df['id']=np.array(['e{0}'.format(id(name.lower())) 
                                  for name in self.df['id']])
    
            self.id =np.copy(self.df['id'].to_numpy())
            self.id_ = np.array(['e{0:07}'.format(ii+1) 
                                     for ii in range(len(self.df['id']))])
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
                raise FeatureError  (
                    '`ERP`, `VES`, `Geology` and `Borehole` Features must '
                    'have the same length. You  give <{0},{1},{2}, and '
                    '{3} respectively.'.format(*temlen))
                
            
            self.df =pd.DataFrame(data = np.array((len(self.ErpColObjs.fnames), 
                                                   len(self.featureLabels))), 
                                  columns = self.featureLabels)
            
            self.id_= self.controlObjId(
                              erpObjID=self.ErpColObjs.erpdf['id'], 
                              boreObjID=self.borehObjs.borehdf['id'], 
                              geolObjID=self.geoObjs.geoldf['id'], 
                              vesObjsID= self.vesObjs.vesdf['id']
                              )
            
            self.df =self.merge(self.ErpColObjs.erpdf, #.drop(['id'], axis=1),
                                self.vesObjs.vesdf['ohmS'],
                                self.geoObjs.geoldf['geol'], 
                                self.borehObjs.borehdf[['lwi', 'flow']], 
                                right_index=True, 
                                left_index=True)
            
            #self.df.insert(loc=0, column ='id', value = newID)
            self.id =self.ErpColObjs.erpdf['id'].to_numpy()
            
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
            
        # populate site names attributes 
        for attr_ in self.site_names: 
            if not hasattr(self, attr_): 
                setattr(self, attr_, ID()._findFeaturePerSite_(
                                        _givenATTR=attr_, 
                                        sns=self.site_names,
                                        df_=self.df,
                                        id_=self.id, 
                                        id_cache= self.id_))
            
            
    def sanitize_fdataset(self): 
        """ Sanitize the feature dataset. Recognize the columns provided 
        by the users and resset according to the features labels disposals
        :attr:`~.GeoFeatures.featureLabels`."""
        
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
        :rtype: np.array of all data.
        
        """
        if not os.path.isfile(erp_fn):
            raise FileHandlingError (
                '{} is not a file. Please provide a right file !'.format(erp_fn))
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
            raise FileHandlingError(
                '{} is not a file. Please provide a '
                'right file !'.format(self.erp_data))
        ex_file = os.path.splitext(self.erp_data)[1] 
        if not ex_file in self.dataType.keys(): 
            pass 
    
    @staticmethod
    def controlObjId( erpObjID, boreObjID, geolObjID, vesObjsID): 
        """
        Control object id whether the each selected anomaly from `erp` matchs 
        with its`ves` and `geol` and `borehole`.
        
        :param erpObjID: ERP object ID. Refer to :doc`~.methods.erp.ERP_collection` 
        :type erpObjID: str 
        
        :param boreObjID: Borehole ID.  Refer to :doc`~.geology.geology.Borehole`
        :type boreObjID: str 
        
        :param boreObjID: Geology ID.  Refer to :doc:`~.geology.geology.Geology`
        :type boreObjID: str
        
        :param vesObjsID: VES object ID. see :doc:`~core.ves.VES`
        
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
                raise FeatureError(
                    "Survey location's name must be the same for `erp`, "
                    ' `ves` and `geol` but you are given '
                    '<{0}, {1}, {2}> respectively. Please rename'
                    ' names to be the same everywhere.'.format(erObj, bhObj,
                                                           geolObj))
        return new_id 
    
    
    @writef(reason='write', from_='df')
    def exportdf (self, refout=None, to =None, savepath=None, **kwargs): 
        """ Export dataframe from :attr:`~.features.GeoFeatures.df` to files 
        can be Excell sheet file or '.json' file. To get more details about 
        the `writef` decorator, see :doc:`watex.tools.decorators.writef`. 
        
        :param refout: 
            Output filename. If not given will be created refering to  the 
            exported date. 
        :param to: 
            Export type. Can be `.xlsx` , `.csv`, `.json` and else
        :type to: str 
        
        :param savepath: 
            Path to save the `refout` filename. If not given
            will be created.
        :returns: 
            - `ndf`: new dataframe from `attr:`~.geofeatures.Features.df` 
     
        :Example: 
            
            >>> from watex.bases.features import Features 
            >>> featObj = Features(
            ...    features_fn= 'data/geo_fdata/BagoueDataset2.xlsx' )
            >>> featObj.exportdf(refout=ybro, to='csv')
    
        """
        df =kwargs.pop('df', None)
        modname =kwargs.pop('moduleName', '_geoFeatures_')
        writeindex =kwargs.pop('writeindex', False)
        
        if df is not None : 
            self.df =df 
            
        for attr in ['to', 'savepath', 'refout']:
            if not hasattr(self, attr): 
                setattr(self, attr,  None)

        if savepath is not None : self.savepath = savepath  
        if to is not None: self.to = to
        if refout is not None: self.refout = refout
        
        # create new data and replace id by site name 
        ndf =self.df.copy(deep=True)
        ndf.reset_index(inplace =True)
        ndf['id'] =self.site_names 
        
        if self.savepath is None :
            self.savepath = savepath_(modname)
            
        return ndf, self.to,  self.refout,\
            self.savepath, writeindex
        

class ID: 
    """
    Special class to manage Feature's ID. Each `erp` or `ves` or `geol` and
    `borehole` name can be an attribute of the each collection class. 
    Eeach survey line is identified with its  common `ID` and point to 
    the same name.
    
    :param _givenATTR:  Station or location name considered a new name for 
        attribute creating
    :type _givenATTR: str 
    
    :param sns: Station names from `erp`, `ves` and `geol`. 
    :type sns: array_like or sns
    
    :param id_: 
        Indentification site number. See col ``id`` of :attr:`~geofeatures.id_`
    :param id_cache: 
        New id of station number kept on caches 
    :param df_: Features dataFrame. Refer to :attr:~geofeatures.df 
    
    :Example: 
        
        >>> from watex.core.geofeatures import Features, ID
        >>> featObj =Features(features_fn= 
        ...                      'data/geo_fdata/BagoueDataset2.xlsx' )
        >>> featObj.b126
        
    where ``b126`` is the surveyname and `featObj.b126` is data value 
    extracted from features dataFrame :attr:`watex.core.geofeatures.Features.df`
    
    :Note: To extract data from station location name `sns`, be sure to write 
        the right name. If not an `AttributeError` occurs. 
    
    """
    
    def __init__(self, **kwargs): 
        self._logging = watexlog().get_watex_logger(self.__class__.__name__)

        for key in list(kwargs.keys()): 
            setattr(self, key, kwargs[key])
        
        if hasattr(self, '_givenATTR'): 
            self._findFeaturePerSite_()
            
    def _findFeaturePerSite_(self, _givenATTR, sns=None, df_=None, id_=None, 
                             id_cache=None ): 
        """Check the report between `site_names ` and `ids`. If `givenATTR is 
        among `sns` or reference object `id_` or `id_cache` then value of 
        given station name will be selected as a dataframe.
        
        :param givenATTR: 
            Station or location name considered a new name for attribute 
            creating:
            
            .. code-block::
            
                >>> from watex.bases.features import Features
                >>> location_name ='gbalo_l10'
                >>> Features.gbalo
                
        :return: As select part of DataFrame
        :rtype: pd.DataFrame 
        
        """
        for attr, value in zip(['_givenATTR', 'df_', 'sns', 'id_cache', 'id_'], 
                             [_givenATTR, df_, sns, id_cache, id_]): 
            if not hasattr(self, attr): 
                setattr(self, attr, value)

        for ii, (name, id_, idp_) in enumerate(zip(self.sns,self.id_,
                                                    self.id_cache)) : 
             if self._givenATTR.lower() == name or \
                 self._givenATTR.lower()==id_ \
                 or self._givenATTR.lower() == idp_ : 
                 self.select_ = self.df_.iloc[
                     [int( idp_.replace('e', '')) - 1]]
                 
        return self.select_ 

                    
class FeatureInspection: 
    """ 
    This class summarizes supervised learning methods inspection. It deals with
    data features categorization, when numericall values is provided standard 
    anlysis either `qualitative` or `quantitative analyis`. 
    
    Arguments 
    -----------
    *dataf_fn*: str 
        Path to analysis data file. 
        
    *df*: pd.Core.DataFrame 
        Dataframe of features for analysis . Must be contains of 
        main parameters including the `target` pd.Core.series 
        as columns of `df`. 
     
    
    Holds on others optionals infos in ``kwargs`` arguments: 
       
    ============  ========================  ===================================
    Attributes              Type                Description  
    ============  ========================  ===================================
    df              pd.core.DataFrame       raw container of all features for 
                                            data analysis.
    target          str                     Traget name for superving learnings
                                            It's usefull to  for clearly  the 
                                            name.
    flow_classes    array_like              How to classify the flow?. Provided 
                                            the main specific values to convert 
                                            numerical value to categorial trends.
    slmethod        str                     Supervised learning method name.The 
                                            methods  can be: 
                                            - Support Vector Machines ``svm``                                      
                                            - Kneighbors: ``knn` 
                                            - Decision Tree: ``dtc``. 
                                            The *default* `sl` is ``svm``. 
    sanitize_df     bool                    Sanitize the columns name to match 
                                            the correct featureslables names
                                            especially in groundwater 
                                            exploration.
    drop_columns    list                    To analyse the data, you can drop 
                                            some specific columns to not corrupt 
                                            data analysis. In formal dataframe 
                                            collected straighforwardly from 
                                            :class:`~features.GeoFeatures`,the
                                            default `drop_columns` refer to 
                                            coordinates positions as : 
                                                ['east', 'north'].
    fn              str                     Data  extension file.                                        
    ============  ========================  ===================================   
    
    :Example:
        >>> from watex.analysis.bases.features import FeatureInspection
        >>> slObj =FeatureInspection(data_fn =' data/geo_fdata/BagoueDataset2.xlsx')
        >>> sObj.df 
 
    """
    
    def __init__(self, df =None , data_fn =None , **kws): 
        self._logging =watexlog().get_watex_logger(self.__class__.__name__)
        
      
        self.data_fn = data_fn 
        self._df =df 
        self._target =kws.pop('target', 'flow')
        self._set_index =kws.pop('set_index', False)
        self._flow_classes=kws.pop('flow_classes',[0., 1., 3.])
        self._slmethod =kws.pop('slm', 'svm')
        self._sanitize_df = kws.pop('sanitize_df', True)
        self._index_col_id =kws.pop('col_id', 'id')
        self._drop_columns =kws.pop('drop_columns', ['east', 'north'])
        
        self._fn =None
        self._df_cache =None 
        
        for key in list(kws.keys()): 
            setattr(self, key, kws[key])
            
        if self.data_fn is not None or self._df is not None: 
            self._dropandFlow_classifier()
            
    @property 
    def target (self): 
        """ Reterun the prediction learning target. Default is ``flow``. """
        return self._target 
    @target.setter 
    def target(self, targ): 
        """ Chech the target and must be in ``str``."""
        if not isinstance(targ, str): 
            raise  FeatureError(
                'Features `target` must be a str'
                ' value not `{}`.'.format(type(targ)))
        self._target =str(targ)
    
    @property 
    def flow_classes(self): 
        return self._flow_classes 
    @flow_classes.setter 
    def flow_classes(self, flowClasses:Iterable[float]): 
        """ When targeting features is numerically considered, The `setter` 
        try to classified into different classes provided. """
        try: 
            self._flow_classes = np.array([ float(ss) for ss in flowClasses])
        except: 
            mess ='`flow_classes` argument must be an `iterable` of '+\
                '``float``values.'
            if flowClasses is None : 
                self._flow_classes =np.array([0.,  3.,  6., 10.])
                self.logging.info (
                    '`flowClasses` argument is set to default'
                    ' values<{0},{1},{2} and {3} m3/h>.'.format(
                        *list(self._flow_classes)))
                
            else: 
                raise ArgumentError(mess)
   
        else: 
            self._logging.info('Flow classes is successfully set.')
            
    @property 
    def fn (self): 
        """ Control the Feature-file extension provide. Usefull to select 
        pd.DataFrame construction."""
        return self._fn
    
    @fn.setter 
    def fn(self, features_fn) :
        """ Get the Features file and  seek for pd.Core methods 
        construction."""
        cpl = Config().parsers
        if not os.path.isfile(features_fn): 
            raise FileHandlingError(
                'No file detected. Could read `{0}`,`{1}`,`{2}`,'
                '`{3}` and `{4}`files.'.format(*list(cpl.keys() )))
        
        self.gFname, exT=os.path.splitext(features_fn)
        if exT in cpl.key() : self._fn =exT 
        else: self._fn ='?'
        
        self._df = cpl[exT](features_fn)
    
        self.gFname = os.path.basename(self.gFname)
    
    @property 
    def df_cache(self): 
        """ Generate cache `df_` for all eliminate features and keep on 
        new pd.core.frame.DataFrame. """
        return self._df_cache 
        
    @df_cache.setter 
    def df_cache(self, cache: Iterable[T]): 
        """ Holds the remove features and keeps on new dataframe """
        try:
            
            temDict={'id': self._df['id'].to_numpy()}
        except KeyError: 
            # if `id` not in colums try 'name'
            temDict={'id': self._df['name'].to_numpy()}
            self._index_col_id ='name'
            
        temc=[]
        if isinstance(cache, str): 
            cache = [cache] 
        elif isinstance(cache, (set,dict, np.ndarray)): 
            cache=list(cache)
            
        if self._drop_columns is not None: 
            if isinstance(self._drop_columns, str) :
                self._drop_columns =[self._drop_columns]
            cache = cache + self._drop_columns 
            if isinstance(self._target, str): 
                cache.append(self._target)
            else : cache + list(self._target)
        for cc in cache: 
            if cc not in self._df.columns: 
                temDict[cc]= np.full((self._df.shape[0],), np.nan)
                temc.append(cc)
            else: 
                if cc=='id': continue # id is already in dict
                temDict [cc]= self._df[cc].to_numpy()
        
        # check into the dataset whether the not provided features exists.
        if self.data_fn is not None : 
            # try: 
            featObj = GeoFeatures(features_fn= self.data_fn)
            # except: 
            self._logging.error(
                'Trouble occurs when calling `Features` class from '
                '`~.core.features.GeoFeatures` module !')
            # else: 

            df__= featObj.df.copy(deep=True)
            df__.reset_index(inplace= True)
            
            if 'id' in df__.columns: 
                temDict['id_']= df__['id'].to_numpy()

            if len(temc) !=0 : 
                for ad in temc: 
                    if ad in df__.columns: 
                        temDict[ad]= df__[ad]
         
            
        self._df_cache= pd.DataFrame(temDict)
        
        if 'id_' in self._df_cache.columns: 
            self._df_cache.set_index('id_', inplace=True)
       
        
    def _dropandFlow_classifier(self, data_fn =None, df =None ,
                                target: str ='flow', 
                                flow_cat_values: Iterable[float] =None, 
                                set_index: bool = False, sanitize_df: bool =True,
                                col_name: str =None, **kwargs): 
        """
        Main goals of this method is to classify the different flow classes
        into four(04) considered as default values according to:
            
            CIEH. (2001). L’utilisation des méthodes géophysiques pour
            la recherche d’eaux dans les aquifères discontinus. 
            Série Hydrogéologie, 169.
            
        which mention 04 types of hydraulic according to the population 
        target inhabitants. Thus:: 
            - FR = 0 is for dry boreholes
            - 0 < FR ≤ 3m3/h for village hydraulic (≤2000 inhabitants)
            - 3 < FR ≤ 6m3/h  for improved village hydraulic(>2000-20 000inhbts) 
            - 6 <FR ≤ 10m3/h for urban hydraulic (>200 000 inhabitants). 
            
        :Note: flow classes can be modified according 
        to the type of hydraulic porposed for your poject. 
        
        :param df: Dataframe of features for analysis 
        
        :param set_index: Considered one column as dataframe index
                        It set to ``True``, please provided the `col_name`, 
                        if not will set the d efault value as ``id``
        :param target:
            
            The target for predicting purposes. Here for groundwater 
            exploration, we specify the target as ``flow``. Hower can be  
            change for another purposes. 
            
        :param flow_cat_values: 
            Different project targetted flow either for village hydraulic, 
            or imporved village hydraulic  or urban hydraulics. 
        
        :param sanitize_df: 
            When using straightforwardly `data_fn` in th case of groundwater  
            exploration :class:erp`
            
        :Example:
            
            >>> from watex.analysis.bases.features import FeatureInspection
            >>> slObj = FeatureInspection(
            ...    data_fn='data/geo_fdata/BagoueDataset2.xlsx',
            ...    set_index =True)
            >>> slObj._df
            >>> slObj.target

        """
        
        drop_columns = kwargs.pop('drop_columns', None)
        mapflow2c = kwargs.pop('map_flow2classes', True)
        if col_name is not None: 
            self._index_col_id= col_name 
        
        if flow_cat_values is not None: 
            self.flow_classes = flow_cat_values 
  
        if data_fn is not None : 
            self.data_fn  = data_fn 
        if self.data_fn is not None : 
            self.fn = self.data_fn 
            
        if df is not None: 
            self._df =df 
        if target is not None : self.target =target 
        
        if sanitize_df is not None : 
            self._sanitize_df = sanitize_df
        
        if drop_columns is not None : 
            # get the columns from dataFrame and compare to the given given 
            if isinstance(drop_columns, (list, np.ndarray)): 
                if  len(set(list(self._df.columns)).intersection(
                        set(drop_columns))) !=len(drop_columns):
                    raise  ParameterNumberError (
                        'Drop values are not found on dataFrame columns. '
                        'Please provided the right names for droping.')
            self._drop_columns = list(drop_columns) 
            
        if self.fn is not None : 
             if self._sanitize_df is True : 
                 self._df , utm_flag = sanitize_fdataset(self._df)
        # test_df = self._df.copy(deep=True)
        if self._drop_columns is not None :
            if isinstance(self._drop_columns, np.ndarray): 
                self._drop_columns = [l.lower() for
                                      l in self._drop_columns.tolist()]
        self.df = self._df.copy()
        if self._drop_columns is not None: 
            self.df = self.df.drop(self._drop_columns, axis =1)
            
        if mapflow2c is True : 
  
            self.df[self.target]= categorize_flow(
                target_array= self.df[self.target], 
                flow_values =self.flow_classes)
  
        if self._set_index :
            # id_= [name  for name in self.df.columns if name =='id']
            if self._index_col_id !='id': 
                self.df=self.df.rename(columns = {self._index_col_id:'id'})
                self._index_col_id ='id'
                
            try: 
                self.df.set_index(self._index_col_id, inplace =True)
            except KeyError : 
                # force to set id 
                self.df=self.df.rename(columns = {'name':'id'})
                self._index_col_id ='id'
                # self.df.set_index('name', inplace =True)

        if self.target =='flow': 
            self.df =self.df.astype({
                             'power':np.float, 
                             'magnitude':np.float, 
                             'sfi':np.float, 
                             'ohmS': np.float, 
                              'lwi': np.float, 
                              })  
            
    def writedf(self, df=None , refout:str =None,  to:str =None, 
              savepath:str =None, modname:str ='_anEX_',
              reset_index:bool =False): 
        """
        Write the analysis `df`. 
        
        Refer to :doc:`watex.__init__.exportdf` for more details about 
        the reference arguments ``refout``, ``to``, ``savepath``, ``modename``
        and ``rest_index``. 
        
        :Example: 
            
            >>> from watex.analysis.bases.features import FeatureInspection
            >>> slObj =FeatureInspection(
            ...   data_fn='data/geo_fdata/BagoueDataset2.xlsx',
            ...   set_index =True)
            >>> slObj.writedf()
        
        """
        for nattr, vattr in zip(
                ['df', 'refout', 'to', 'savepath', 'modname', 'reset_index'], 
                [df, refout, to, savepath, modname, reset_index]): 
            if not hasattr(self, nattr): 
                setattr(self, nattr, vattr)
                
        exportdf(df= self.df , refout=self.refout,
               to=self.to, savepath =self.savepath, 
               reset_index =self.reset_index, modname =self.modname)
            
        
if __name__=='__main__': 
    featurefn ='data/geo_fdata/BagoueDataset2.xlsx' 
    
    # featObj =GeoFeatures(features_fn= featurefn)
    
    # df=featObj.df
    # print(df)
    #df2, *_ = 
    # featObj.exportdf()
    
    # print(featObj.site_names)
    # print(featObj.id_)
    # print(featObj.id)
    # print(featObj.b125)
    # dff= featObj.df
    # print(dff)
       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        