# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Wed Jul  7 22:23:02 2021 hz
# This module is part of the WATex analysis package, which is released under a
# MIT- licence.
"""
Created on Mon Jul 12 14:24:18 2021

    :synopsis: 
        
        Module :mod:`watex.analysis.basics` is actually focused on supervised 
        analyses.

@author: @Daniel03

"""

import os
from typing import Iterable,TypeVar 
import pandas as pd 
import numpy as np 

from ..bases import sanitize_fdataset as STZ_DF
from ..bases import exportdf as EXP_DF
from . import PD_READ_FEATURES
from ..core import features
from ..utils import decorator as deco
from ..utils._watexlog import watexlog 
import watex.utils.exceptions as Wex

T=TypeVar('T', list, tuple) 
_logger =watexlog().get_watex_logger(__name__)


class SLAnalyses: 
    """ 
    This class summarizes supervised learning methods analysis. It  
    deals with data features categorization, when numericall values is 
    provided standard anlysis either `qualitatif` or `quantitatives analyis`. 
    
    Arguments: 
    ---------
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
                                            methods  can be:: 
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
                                                ['east', 'north']
    fn              str                     Data  extension file.                                        
    ============  ========================  ===================================   
    
    :Example:
        
        >>> from watex.analysis.basics import SLAnalyses
        >>> slObj =SLAnalyses(data_fn =' data/geo_fdata/BagoueDataset2.xlsx')
        >>> sObj.df 
        >>> sObj.
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
            raise Wex.WATexError_geoFeatures(
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
                raise Wex.WATexError_inputarguments(mess)
   
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
        
        if not os.path.isfile(features_fn): 
            raise Wex.WATexError_file_handling(
                'No file detected. Could read `{0}`,`{1}`,`{2}`,'
                '`{3}` and `{4}`files.'.format(*list(PD_READ_FEATURES.keys())))
        
        self.gFname, exT=os.path.splitext(features_fn)
        if exT in PD_READ_FEATURES.keys(): self._fn =exT 
        else: self._fn ='?'
        
        self._df = PD_READ_FEATURES[exT](features_fn)
    
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
            featObj = features.GeoFeatures(features_fn= self.data_fn)
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
         into four(04) considered as default values according to::
            
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
            
            >>> from watex.analysis.basics import SLAnalyses
            >>> slObj = SLAnalyses(
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
                    raise Wex.WATexError_parameter_number(
                        'Drop values are not found on dataFrame columns. '
                        'Please provided the right names for droping.')
            self._drop_columns = list(drop_columns) 
            
        if self.fn is not None : 
             if self._sanitize_df is True : 
                 self._df , utm_flag = STZ_DF(self._df)
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
        Write analysis `df`. 
        
        Refer to :doc:`watex.__init__.exportdf` for more details about 
        the reference arguments ``refout``, ``to``, ``savepath``, ``modename``
        and ``rest_index``. 
        
        :Example: 
            
            >>> from watex.analysis.slfeatures import SLAnalyses 
            >>> slObj =SLAnalyses(
            ...   data_fn='data/geo_fdata/BagoueDataset2.xlsx',
            ...   set_index =True)
            >>> slObj.writedf()
        
        """
        for nattr, vattr in zip(
                ['df', 'refout', 'to', 'savepath', 'modname', 'reset_index'], 
                [df, refout, to, savepath, modname, reset_index]): 
            if not hasattr(self, nattr): 
                setattr(self, nattr, vattr)
                
        EXP_DF(df= self.df , refout=self.refout,
               to=self.to, savepath =self.savepath, 
               reset_index =self.reset_index, modname =self.modname)
        
            
@deco.catmapflow2(cat_classes=['FR0', 'FR1', 'FR2', 'FR3'])#, 'FR4'] )
def categorize_flow(target_array, flow_values: Iterable[float],
                    **kwargs) -> Iterable[T]: 
    """ 
    Categorize `flow` into different classes. If the optional
    `flow_classes`  argument is given, it should be erased the
    `cat_classes` argument of decororator `deco.catmapflow`.
    
    :param target_array: Flow array to be categorized 
    
    :param flow_values: 
        
        The way to be categorized. Distribute the flow values 
        of numerical values considered as pseudo_classes like:: 
    
            flow_values= [0.0, [0.0, 3.0], [3.0, 6.0], [6.0, 10.0], 10.0] (1)
            
        if ``flow_values`` is given as follow:: 
            
            flow_values =[0. , 3., 6., 10.] (2)
        
        It should convert the type (2) to (1).
        
    :param flow_classes: 
        Values of categorized flow rates 
        
    :returns: 
        
        - ``new_flow_values``: Iterable object as type (2) 
        - ``target_array``: Raw flow iterable object to be categorized
        - ``flowClasses``: If given , see ``flow_classes`` param. 
            
    """
    flowClasses =  kwargs.pop('classes', None)

    new_flow_values = []
    inside_inter_flag= False
    
    if isinstance(flow_values, (tuple, np.ndarray)): 
        flow_values =list(flow_values)
    # Loop and find 
    for jj, _iter in enumerate(flow_values) : 
        if isinstance(_iter, (list, tuple, np.ndarray)): 
            inside_inter_flag = True 
            flow_values[jj]= list(_iter)
 
    if inside_inter_flag: 
        new_flow_values =flow_values 
    
    if inside_inter_flag is False: 
        flow_values= sorted(flow_values)
        # if 0. in flow_values : 
        #     new_flow_values.append(0.) 
        for ss, val in enumerate(flow_values) : 
            if ss ==0 : 
                #append always the first values. 
                 new_flow_values.append(val) 
            # if val !=0. : 
            else:
                if val ==flow_values[-1]: 
                    new_flow_values.append([flow_values[ss-1], val])
                    new_flow_values.append(val)
                else: 
                   new_flow_values.append([flow_values[ss-1], val])
 
    return new_flow_values, target_array, flowClasses
        
  
if __name__=='__main__': 

    featurefn ='data/geo_fdata/BagoueDataset3.xlsx' 
    featurefn ='data/geo_fdata/_bagoue_civ_loc_ves&erpdata3.csv'
    
    # slObj =sl_analysis(data_fn=featurefn, set_index =True,
    #                    drop_columns='num', col_id ='name',
    #                    flow_classes =[0., 1., 3.])
    # slObj.df_cache=['op']
    # df= slObj.df
    # print(set(df['flow'].to_numpy()))
    # slObj.writedf()
    # import matplotlib.pyplot as plt 
    # plt.figure(figsize=(10,5))
    # df['flow'].value_counts(normalize =True).plot.bar(label='flow_classes')
    # # print(cache +['east', 'north'])
    # print(df['flow'].value_counts(normalize = True))
        
        
        
        
        
        
        
        
        
        
        
        
        




