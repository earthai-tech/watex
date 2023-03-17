# -*- coding: utf-8 -*-
#   Licence:BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import ( 
    print_function , 
    annotations
    )

import os
import re 
import warnings 
import xml.etree.ElementTree as ET
import pandas as pd 
import numpy as np 

from ..utils.funcutils import  ( 
    savepath_ , 
    sanitize_fdataset, 
    )   
from .._typing import ( 
    List,
    Optional, 
    DataFrame,
    )
from ..utils.hydroutils import ( 
    writef,  
    exportdf, 
    categorize_flow
    ) 
from ..exceptions import ( 
    FileHandlingError, 
    FeatureError, 
   )
from ..utils.gistools import ( 
    ll_to_utm, 
    project_point_ll2utm
    )
from ..utils.coreutils import _is_readable 
from watex._watexlog import watexlog 

__all__=['GeoFeatures', 'FeatureInspection'] 

__docformat__='restructuredtext' 
_logger =watexlog().get_watex_logger(__name__)


class GeoFeatures: 
    """
    Features class. Deals  with Electrical Resistivity profile (VES), 
    Vertical electrical Sounding (VES), Geological (Geol) data and 
    Borehole data(Boreh). Set all features values of differents
    investigation sites. Features class is  composed of: 
    
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
    >>> from watex.cases.features import GeoFeatures 
    >>> data ='data/geodata/main.bagciv.data.csv' 
    >>> featObj =GeoFeatures().fit(data )
    >>> featObj.id_
    Out[114]: 
    array(['e0000001', 'e0000002', 'e0000003', 'e0000004', 'e0000005',
           'e0000006', 'e0000007'], dtype='<U8')
    >>> featObj.site_names_
    >>> featObj.site_names_[:7] 
    Out[115]: array(['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'], dtype=object)
     
    """
    
    featureLabels_ = [
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

    def __init__(self, **kws):
        
        self._logging = watexlog.get_watex_logger(self.__class__.__name__)

        for key in list(kws.keys()):
            setattr(self, key, kws[key])
            
         
    @property 
    def data  (self): 
        """ Control the Feature-file extension provide. Usefull to select 
        pd.DataFrame construction."""
        return self.data_
    
    @data.setter 
    def data(self, data ) :
        """ Get the Features file and  seek for pd.Core methods construction."""
    
        self.data_ = _is_readable(data)

 
    def fit(self, 
            data: Optional[str |DataFrame]=None , 
            geoObj=None, 
            erpObj=None, 
            vesObj=None, 
            boreholeObj=None, 
            **kws): 
        """
        Reading class and attributes populating. Please refer to 
        :doc:`~.core.geofeatures.Features` for arguments details.
        
        """
        
        if data is not None: 
            self.data = data 
            
        fimp, objsCoun =0,0 
        for nname, vval in zip(['data' , 'erpObj' , 'vesObj',
                 'geoObj', 'borehObj'],[data , erpObj , vesObj,
                 geoObj, boreholeObj]): 
            if vval is not None: 
                setattr(self,nname, vval )
                if nname !='data':
                    objsCoun +=1
        # call object
        for fObjs in ['ErpColObjs' , 'vesObjs',
                'geoObjs', 'borehObjs']: 
            if getattr(self, fObjs, None) is None : 
                fimp =1        
                
        if self.data is None and fimp ==1:
            raise FeatureError (
                'Features file is not given. Please provide specific'
                ' objects from`erp`, `ves`, `geology` and `borehole` data'
                'Call each specific collection class to build each'
                ' collection features.')
        elif self.data is not None : 
            #self.fn = self.features_fn 
            self.sanitize_fdataset()
            try : 
                self.site_names_ =np.copy(self.df_['id'].to_numpy())
            except KeyError: 
                 # force to set id 
                self.df_=self.df_.rename(columns = {'name':'id'})
                self.site_names_ =np.copy(self.df_['id'].to_numpy())
                # self._index_col_id ='id'
            
            if self.utm_flag_ ==0 :
                # convert lat and lon to utm 

                self.easting_ = np.zeros_like(self.df_['lat'].to_numpy())
                self.northing_ =np.zeros_like (self._easting)
                for ii in range(len(self.northing_)):
                    try : 
                        self.utm_zone_, utm_easting, utm_northing = ll_to_utm(
                                        reference_ellipsoid=23, 
                                        lat=self.df_['lon'].to_numpy()[ii],
                                        lon = self.df_['lat'].to_numpy()[ii])
                    except : 
                        utm_easting, utm_northing, \
                            self.utm_zone= project_point_ll2utm(
                            lat=self.df_['lat'].to_numpy()[ii],
                            lon = self.df_['lon'].to_numpy()[ii])
                        
                    self.easting_[ii] = utm_easting
                    self.northing_ [ii] = utm_northing
            
                self.df_.insert(loc=1, column ='east', value = self.easting_)
                self.df_.insert(loc=2, column='north', value=self.northing_)
                
                try : 
                    del self.df_['lat']
                    del self.df_['lon']
                except : 
                    try : 
                        self.df_ = self.df_.drop(['lat'], axis=1)
                        self.df_ = self.df_.drop(['lon'], axis=1)
                    except : 
                        try: 
                            self.df_.pop('lat')
                            self.df_.pop('lon')
                        except: 
                           self._logging.debug(
                               'No way to remove `lat` and `lon` in features '
                               "dataFrame. It seems there is no `lat` & `lon`"
                               " pd.series in your dataFrame.") 
            
            #Keep location names 
            self.df_['id']=np.array(['e{0}'.format(id(name.lower())) 
                                  for name in self.df_['id']])
    
            self.id =np.copy(self.df_['id'].to_numpy())
            self.id_ = np.array(['e{0:07}'.format(ii+1) 
                                     for ii in range(len(self.df_['id']))])
            # rebuild the dataframe from main features
            self.df_ = pd.concat({
                featkey: self.df_[featkey] 
                for featkey in self.featureLabels_}, axis =1)


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
                
            
            self.df_ =pd.DataFrame(data = np.array((len(self.ErpColObjs.fnames), 
                                                   len(self.featureLabels_))), 
                                  columns = self.featureLabels_)
            
            self.id_= self.controlObjId(
                              erpObjID=self.ErpColObjs.erpdf['id'], 
                              boreObjID=self.borehObjs.borehdf['id'], 
                              geolObjID=self.geoObjs.geoldf['id'], 
                              vesObjsID= self.vesObjs.vesdf['id']
                              )
            
            self.df_ =self.merge(self.ErpColObjs.erpdf, #.drop(['id'], axis=1),
                                self.vesObjs.vesdf['ohmS'],
                                self.geoObjs.geoldf['geol'], 
                                self.borehObjs.borehdf[['lwi', 'flow']], 
                                right_index=True, 
                                left_index=True)
            
            #self.df.insert(loc=0, column ='id', value = newID)
            self.id =self.ErpColObjs.erpdf['id'].to_numpy()
            
        self.df_.set_index('id', inplace =True)
        self.df_ =self.df_.astype({'east':np.float, 
                      'north':np.float, 
                      'power':np.float, 
                      'magnitude':np.float, 
                      'sfi':np.float, 
                      'ohmS': np.float, 
                      'lwi':np.float, 
                      'flow':np.float
                      })
            
        # populate site names attributes 
        for attr_ in self.site_names_: 
            if not hasattr(self, attr_): 
                setattr(self, attr_, ID()._findFeaturePerSite_(
                                        _givenATTR=attr_, 
                                        sns=self.site_names_,
                                        df_=self.df_,
                                        id_=self.id, 
                                        id_cache= self.id_))
            
        return self 
    
    def sanitize_fdataset(self): 
        """ Sanitize the feature dataset. Recognize the columns provided 
        by the users and resset according to the features labels disposals
        :attr:`~.GeoFeatures.featureLabels`."""
        
        self.utm_flag_ =0
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
                                 self.utm_flag_=1
                             break

    
                        
            return columns

        new_df_columns= getandReplace(optionsList=OptsList, params=paramsList,
                                      df= self.data)
        self.df_= pd.DataFrame(data=self.data.to_numpy(), 
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
            self.data_fn_ =data_fn 
        
        if not os.path.isfile(self.data_fn_): 
            raise FileHandlingError(
                '{} is not a file. Please provide a '
                'right file !'.format(self.data_fn_))
        ex_file = os.path.splitext(self.data_fn_)[1] 
        if not ex_file in self.dataType.keys(): 
            pass 
    
    @staticmethod
    def controlObjId( erpObjID, boreObjID, geolObjID, vesObjsID): 
        """
        Control object id whether the each selected anomaly from `erp` matchs 
        with its`ves` and `geol` and `borehole`.
        
        :param erpObjID: ERP object ID. Refer to 
            :class:`watex.methods.erp.ERP_collection` 
        :type erpObjID: str 
        
        :param boreObjID: Borehole ID.  Refer to 
            :class:`watex.geology.drilling.Borehole`
        :type boreObjID: str 
        
        :param boreObjID: Geology ID.  Refer to 
            :class:`watex.geology.geology.Geology`
        :type boreObjID: str
        
        :param vesObjsID: VES object ID. Refer to 
            :class:`watex.methods.electrical.VerticalSounding`
        
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
        the `writef` decorator, see :func:`watex.decorators.writef`. 
        
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
    Summarizes the flow features. 
    
    It deals with data features categorization. When numericall values are 
    provided standard `qualitative` or `quantitative`  analysis is performed.
    
    Parameters  
    -----------
    *data*: str or pd.core.DataFrame  
        Path-like object or pandas Dataframe. Must contain the  main 
        parameters including the `target`. 
        
    **tname**:str 
        The tname for predicting purposes. Here for groundwater exploration, 
        we specify the name of the target as ``flow``. 
        
    **flow_classes**: list or array_like 
        The way to classify the flow. Provide the main specific values to convert 
        the  categorial trends to numerical values.  Different projects have 
        different tnameing flow rate. Might specify either for village hydraulic, 
        or improved village hydraulic  or urban hydraulics. 
        
    **drop_columns**: list  
        items for dropping. To analyse the data, we can drop some specific 
        columns to not corrupt data analysis. In formal dataframe  collected 
        straighforwardly from :class:`~features.GeoFeatures`,the default
        `drop_columns` refer to coordinates positions as : ['east', 'north'].
        
    **mapflow: bool, 
        if set to True, value in the target columns should map to categorical 
        values. Commonly the flow rate values are given as a trend of numerical
        values. For a classification purpose, flow rate must be converted to 
        categorical values which are mainly refered to the type of types of
        hydraulic. Mostly the type of hydraulic system is in turn tided to the 
        the number of the living population in a specific area. For instance, 
        flow classes can be ranged as follow: 
    
            - FR = 0 is for dry boreholes
            - 0 < FR ≤ 3m3/h for village hydraulic (≤2000 inhabitants)
            - 3 < FR ≤ 6m3/h  for improved village hydraulic(>2000-20 000inhbts) 
            - 6 <FR ≤ 10m3/h for urban hydraulic (>200 000 inhabitants). 
        
        Note that this flow range is not exhaustive and can be modified according 
        to the type of hydraulic required on the project. 
    
    **set_index**: bool, 
        condired a column as dataframe index. If set to ``True``, 
        please provided the `col_name`, otherwise it should be the ``id`` as 
        as a default columns item. 
        
    **sanitize**: 
        polish the data and remove inconsistent columns in the data which are 
        not refer to the predicting features. It is able to change for instance
        the french name of water ``eau`` to 'water` wich is related to the value 
        of water inflow features ``lwi``. This could be  usefull when the data 
        is given as a Path-Like object and features are not described correctly
        in the case of groundwater. Default is ``False``
        
 
    Examples
    --------
    >>> from watex.cases.features import FeatureInspection
    >>> data = 'data/geodata/main.bagciv.data.csv'
    >>> fobj = FeatureInspection().fit(data) 
    >>> fobj.data_.columns
    Out[117]: 
    Index(['num', 'name', 'east', 'north', 'power', 'magnitude', 'shape', 'type',
           'sfi', 'ohmS', 'lwi', 'geol', 'flow'],
          dtype='object')
    """
    
    def __init__(self,
                 tname:str ='flow' ,
                 mapflow:bool =True, 
                 sanitize:bool =False,
                 flow_classes: List[float] = [0., 1., 3.],
                 set_index:bool = False, 
                 col_name: str = None, 
                 **kws): 
        self._logging =watexlog().get_watex_logger(self.__class__.__name__)
        
        self.tname =tname
        self.set_index =set_index 
        self.sanitize =sanitize
        self.mapflow =mapflow
        self.flow_classes_=flow_classes
        self.col_name= col_name 
        
        self.index_col_id =kws.pop('col_id', 'id')
        self.drop_columns =kws.pop('drop_columns', None)

        
        self.cache_ =None 

        for key in list(kws.keys()): 
            setattr(self, key, kws[key])
            

    @property 
    def flow_classes(self): 
        return self.flow_classes_
    @flow_classes.setter 
    def flow_classes(self, flow_classes): 
        """ When tnameing features is numerically considered, The `setter` 
        try to classified into different classes provided. """
        if flow_classes is None: 
            flow_classes = [0.,  3.,  6., 10.]
            warnings.warn("default flow classes argument range is set to"
                          f"'{flow_classes}' m3/h !")
            self.logging.info (' The flow classes argument is set to default'
                ' values : <{0},{1},{2} and {3} m3/h>.'.format( *list(
                 flow_classes)))
            
        try: 
            self.flow_classes_ = np.array(flow_classes).astype(float)
        except: 
            raise FeatureError (f"Not supported flow classes "
                                f"arguments: {flow_classes}")
            
    @property 
    def data (self): 
        """ Control the Feature-file extension provide. Usefull to select 
        pd.DataFrame construction."""
        return self.data_
    
    @data.setter 
    def data(self, data) :
        """ Get the Features file and  seek for pd.Core methods 
        construction."""
        self.data_ = _is_readable(data )

    
    @property 
    def cache(self): 
        """ Generate cache `df_` for all eliminate features and keep on 
        new pd.core.frame.DataFrame. """
        return self.cache_
        
    @cache.setter 
    def cache(self, cache): 
        """ Holds the remove features and keeps on new dataframe
        
        :param cache: iterable object containing the item to drop"""
        
        try:
            
            temDict={'id': self.data_['id'].to_numpy()}
        except KeyError: 
            # if `id` not in colums try 'name'
            temDict={'id': self.data_['name'].to_numpy()}
            self.index_col_id ='name'
            
        temc=[]
        if isinstance(cache, str): 
            cache = [cache] 

        if self.drop_columns is not None: 
            if isinstance(self.drop_columns, str) :
                self.drop_columns =[self.drop_columns]
            cache = cache + self.drop_columns 
            if isinstance(self.tname, str): 
                cache.append(self.tname)
            else : cache + list(self.tname)
            
        for cc in cache: 
            if cc not in self.data_.columns: 
                temDict[cc]= np.full((self.data_.shape[0],), np.nan)
                temc.append(cc)
            else: 
                if cc=='id': continue # id is already in dict
                temDict [cc]= self.data_[cc].to_numpy()
        
        # check into the dataset whether the not provided features exists.
        if self.data_ is not None : 
            df_= self.data_.copy()
            df_.reset_index(inplace= True)
            
            if 'id' in df_.columns: 
                temDict['id_']= df_['id'].to_numpy()

            if len(temc) !=0 : 
                for ad in temc: 
                    if ad in df_.columns: 
                        temDict[ad]= df_[ad]
         
            
        self.cache_= pd.DataFrame(temDict)
        self.col_name = self.col_name or 'id_'
        if self.col_name in self.cache_.columns: 
            self.cache_.set_index('id_', inplace=True)
       
        
    def fit(self, data: str | DataFrame): 
        """
        Main goals of this method is to fit and classify the different flow 
        classes in the dataset. However by default, four(04) flow classes are 
        considered according to the reference below 
        
        Parameters 
        -----------
        
        *data*: str or pd.core.DataFrame  
            Path-like object or pandas Dataframe. Must contains of the  main 
            parameters including the `tname` the tname. 
            
        Returns
        --------
        object: :class:`~.FeatureInspection` object 
        
        Examples
        ---------
        >>> from watex.bases.features import FeatureInspection
        >>> data = 'data/geodata/main.bagciv.data.csv'
        >>> fobj = FeatureInspection() 
        >>> fobj.fit(data)
        >>> fobj.data.iloc[1:3 , :]
        ...    num name  power  magnitude  ...         ohmS        lwi      geol  flow
        1    2   b2   70.0      142.0  ...  1135.551531  21.406531  GRANITES   FR1
        2    3   b3   80.0       87.0  ...   767.562500   0.000000  GRANITES   FR1
        
        
        Notes 
        --------
        The paper mentions 04 types of hydraulic according to the population 
        demand and the number of living inhabitants. The hydraulic system are
        defined as:
         
         *  FR = 0 is for dry boreholes
         *  0 < FR ≤ 3m3/h for village hydraulic (≤2000 inhabitants)
         *  3 < FR ≤ 6m3/h  for improved village hydraulic(>2000-20 000inhbts) 
         *  6 <FR ≤ 10m3/h for urban hydraulic (>200 000 inhabitants). 
         
        The flow classes can be modified according to the type of hydraulic
        proposed for the project. 
        
        References 
        ------------
            
        .. [1] CIEH. (2001). L’utilisation des méthodes géophysiques pour
            la recherche d’eaux dans les aquifères discontinus. 
            Série Hydrogéologie, 169.
            
        """
        
        self.data = data 
            
        if self.data_ is None: 
            raise FeatureError("NoneType can not be a data of features.")
            
              
        if self.sanitize is True : 
            self.data_ , utm_flag = sanitize_fdataset(self.data_)
        # test_df = self._df.copy(deep=True)

        # df = self.data_.copy()
        if self.drop_columns is not None:
            if isinstance(self.drop_columns, str): 
                    self.drop_columns = [self.drop_columns]
           
            if  len(set(list(self.data.columns)).intersection(
                    set(self.drop_columns))) !=len(self.drop_columns):
                raise  FeatureError (
                    'Drop values are not found on dataFrame columns. '
                    'Please provide the right names for droping.')
                
            self.cache = self.drop_columns  
            self.data_.drop(columns = self.drop_columns, inplace =True)
            
        if self.mapflow is True : 
            self.data_[self.tname]= categorize_flow(
                target= self.data_[self.tname], 
                flow_values =self.flow_classes)
  
        if self.set_index :
            # id_= [name  for name in self.df.columns if name =='id']
            if self.index_col_id !='id': 
                self.data_=self.data_.rename(columns = {self.index_col_id:'id'})
                self.index_col_id ='id'
                
            try: 
                self.data_.set_index(self.index_col_id, inplace =True)
            except KeyError : 
                # force to set id 
                self.data_=self.data.rename(columns = {'name':'id'})
                self.index_col_id ='id'
                # self.df.set_index('name', inplace =True)

        if self.tname =='flow': 
            self.data_ =self.data_.astype({
                             'power':np.float, 
                             'magnitude':np.float, 
                             'sfi':np.float, 
                             'ohmS': np.float, 
                              'lwi': np.float, 
                              }
                )  
        return self 
    
    
    def writedf(self, df=None , refout:str =None,  to:str =None, 
              savepath:str =None, modname:str ='_anEX_',
              reset_index:bool =False): 
        """
        Write the analysis `df`. 
        
        Refer to :func:`watex.decorators.exportdf` for more details about 
        the arguments ``refout``, ``to``, ``savepath``, ``modename``
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
        
      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        