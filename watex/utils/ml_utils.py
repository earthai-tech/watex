# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Wed Jul 14 20:00:26 2021
# This module is a set of utils for data prepprocessing
# released under a MIT- licence.
"""
Created on Sat Aug 28 16:26:04 2021

@author: @Daniel03

"""

from typing import TypeVar, Generic, Iterable , Callable, Text

T= TypeVar('T')
KT=TypeVar('KT')
VT=TypeVar('VT')

import os 
import hashlib 
import tarfile 
import inspect
import warnings 
 
from six.moves import urllib 
from abc import ABC, abstractmethod, ABCMeta  

import pandas as pd 
import numpy as np 

from sklearn.model_selection import StratifiedShuffleSplit 
from watex.utils._watexlog import watexlog

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
 
_logger = watexlog().get_watex_logger(__name__)

DOWNLOAD_ROOT = 'https://github.com/WEgeophysics/watex/master/'
#'https://zenodo.org/record/4896758#.YTWgKY4zZhE'
DATA_PATH = 'data/tar.tgz_files'
TGZ_FILENAME = '/bagoue.main&rawdata.tgz'
CSV_FILENAME = '_bagoue_civ_loc_ves&erpdata4.csv'

DATA_URL = DOWNLOAD_ROOT + DATA_PATH  + TGZ_FILENAME


def read_from_excelsheets(erp_file: T = None ) -> Iterable[VT]: 
    
    """ Read all Excelsheets and build a list of dataframe of all sheets.
   
    :param erp_file:
        Excell workbooks containing `erp` profile data.
    :return: A list composed of the name of `erp_file` at index =0 and the 
            datataframes.
    """
    
    allfls:Generic[KT, VT] = pd.read_excel(erp_file, sheet_name=None)
    
    list_of_df =[os.path.basename(os.path.splitext(erp_file)[0])]
    for sheets , values in allfls.items(): 
        list_of_df.append(pd.DataFrame(values))

    return list_of_df 

def write_excel(listOfDfs: Iterable[VT], csv:bool =False , sep:T =','): 
    """ 
    Rewrite excell workbook with dataframe for :ref:`read_from_excelsheets`. 
    
    Its recover the name of the files and write the data from dataframe 
    associated with the name of the `erp_file`. 
    
    :param listOfDfs: list composed of `erp_file` name at index 0 and the
     remains dataframes. 
    :param csv: output workbook in 'csv' format. If ``False`` will return un 
     `excel` format. 
    :param sep: type of data separation. 'default is ``,``.'
    
    """
    site_name = listOfDfs[0]
    listOfDfs = listOfDfs[1:]
    for ii , df in enumerate(listOfDfs):
        
        if csv:
            df.to_csv(df, sep=sep)
        else :
            with pd.ExcelWriter(f"z{site_name}_{ii}.xlsx") as writer: 
                df.to_excel(writer, index=False)
    
   
def fetch_geo_data (data_url:str = DATA_URL, data_path:str =DATA_PATH,
                    tgz_filename =TGZ_FILENAME ) -> Text: 
    """ Fetch data from data repository in zip of 'targz_file. 
    
    I will create a `datasets/data` directory in your workspace, downloading
     the `~.tgz_file and extract the `data.csv` from this directory.
    
    :param data_url: url to the datafilename where `tgz` filename is located  
    :param data_path: absolute path to the `tgz` filename 
    :param filename: `tgz` filename. 
    """
    if not os.path.isdir(data_path): 
        os.makedirs(data_path)
    tgz_path = os.path.join(data_url, tgz_filename.replace('/', ''))
    
    urllib.request.urlretrieve(data_url, tgz_path)
    data_tgz = tarfile.open(tgz_path)
    data_tgz.extractall(path = data_path )
    data_tgz.close()
    
    
def load_data (data_path:str = DATA_PATH,
               filename:str =CSV_FILENAME, sep =',' )-> Generic[VT]:
    """ Load CSV file to pd.dataframe. 
    
    :param data_path: path to data file 
    :param filename: name of file. 
    
    """ 
    csv_path = os.path.join(data_path , filename)
    
    return pd.read_csv(csv_path, sep)


def split_train_test (data:Generic[VT], test_ratio:T)-> Generic[VT]: 
    """ Split dataset into trainset and testset from `test_ratio` 
    and return train set and test set.
        
    ..note: `test_ratio` is ranged between 0 to 1. Default is 20%.
    """
    shuffled_indices =np.random.permutation(len(data)) 
    test_set_size = int(len(data)* test_ratio)
    test_indices = shuffled_indices [:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    return data.iloc[train_indices], data.iloc[test_indices]
    
def test_set_check_id (identifier, test_ratio, hash:Callable[..., T]) -> bool: 
    """ 
    Get the testset id and set the corresponding unique identifier. 
    
    Compute the a hash of each instance identifier, keep only the last byte 
    of the hash and put the instance in the testset if this value is lower 
    or equal to 51(~20% of 256) 
    has.digest()` contains object in size between 0 to 255 bytes.
    
    :param identifier: integer unique value 
    :param ratio: ratio to put in test set. Default is 20%. 
    
    :param hash:  
        Secure hashes and message digests algorithm. Can be 
        SHA1, SHA224, SHA256, SHA384, and SHA512 (defined in FIPS 180-2) 
        as well as RSA’s MD5 algorithm (defined in Internet RFC 1321). 
        
        Please refer to :ref:`<https://docs.python.org/3/library/hashlib.html>` 
        for futher details.
    """
    return hash(np.int64(identifier)).digest()[-1]< 256 * test_ratio

def split_train_test_by_id(data, test_ratio:T, id_column:T=None,
                           hash=hashlib.md5)-> Generic[VT]: 
    """Ensure that data will remain consistent accross multiple runs, even if 
    dataset is refreshed. 
    
    The new testset will contain 20%of the instance, but it will not contain 
    any instance that was previously in the training set.

    :param data: Pandas.core.DataFrame 
    :param test_ratio: ratio of data to put in testset 
    :id_colum: identifier index columns. If `id_column` is None,  reset  
                dataframe `data` index and set `id_column` equal to ``index``
    :param hash: secures hashes algorithms. Refer to 
                :func:`~test_set_check_id`
    :returns: consistency trainset and testset 
    """
    if id_column is None: 
        id_column ='index' 
        data = data.reset_index() # adds an `index` columns
        
    ids = data[id_column]
    in_test_set =ids.apply(lambda id_:test_set_check_id(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

def discretizeCategoriesforStratification(data, in_cat:str =None,
                               new_cat:str=None, **kws) -> Generic[VT]: 
    """ Create a new category attribute to discretize instances. 
    
    A new category in data is better use to stratified the trainset and 
    the dataset to be consistent and rounding using ceil values.
    
    :param in_cat: column name used for stratified dataset 
    :param new_cat: new category name created and inset into the 
                dataframe.
    :return: new dataframe with new column of created category.
    """
    divby = kws.pop('divby', 1.5) # normalize to hold raisonable number 
    combined_cat_into = kws.pop('higherclass', 5) # upper class bound 
    
    data[new_cat]= np.ceil(data[in_cat]) /divby 
    data[new_cat].where(data[in_cat] < combined_cat_into, 
                             float(combined_cat_into), inplace =True )
    return data 

def stratifiedUsingDiscretedCategories(data:VT , cat_name:str , n_splits:int =1, 
                    test_size:float= 0.2, random_state:int = 42)-> Generic[VT]: 
    """ Stratified sampling based on new generated category  from 
    :func:`~DiscretizeCategoriesforStratification`.
    
    :param data: dataframe holding the new column of category 
    :param cat_name: new category name inserted into `data` 
    :param n_splits: number of splits 
    """
    
    split = StratifiedShuffleSplit(n_splits, test_size, random_state)
    for train_index, test_index in split.split(data, data[cat_name]): 
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index] 
        
    return strat_train_set , strat_test_set 

 
class SearchedGrid: 
    """ Fine tune hyperparameters. 
    
    `Search Grid will be able to  fiddle with the hyperparameters until to 
    find the great combination for model predictions. 
    
    :param base_estimator: Estimator to be fined tuned hyperparameters
    
    :grid_params: list of hyperparamters params  to be tuned 
    
    :param cv: Cross validation sampling. Default is `4` 
    
    :pram kind: Kind of search. Could be ``'GridSearchCV'`` or
    ``RandomizedSearchCV``. Default is ``gridSearchCV`.
    
    :param scoring: Type of score for errors evaluating. Default is 
        ``neg_mean_squared_error``. 
        
    :Example: 
        
        >>> from pprint import pprint 
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from watex.utils._data_preparing_ import bagoue_train_set_prepared 
        >>> from watex.utils._data_preparing_ import bagoue_label_encoded  
        >>> grid_params = [
        ...        {'n_estimators':[3, 10, 30], 'max_features':[2, 4, 6, 8]}, 
        ...        {'bootstrap':[False], 'n_estimators':[3, 10], 
        ...                             'max_features':[2, 3, 4]}]
        >>> forest_clf = RandomForestClassifier()
        >>> grid_search = SearchedGrid(forest_clf, grid_params)
        >>> grid_search.fit(X= bagoue_train_set_prepared ,
        ...                    y = bagoue_label_encoded)
        >>> pprint(grid_search.best_params_ )
        >>> pprint(grid_search.cv_results_)
    """
    
    __slots__=('_base_estimator',
                'grid_params', 
                'scoring',
                'cv', 
                '_kind', 
                '_logging',
                 'grid_kws',
                'best_params_',
                'best_estimator_',
                'cv_results_',
                'feature_importances_',
                )
               
    def __init__(self,
                 base_estimator:Callable[..., T],
                 grid_params:Iterable[T],
                 cv:int =4,
                 kind:str ='GridSearchCV',
                 scoring:str = 'neg_mean_squared_error',
                 **grid_kws): 
        
        self._logging = watexlog().get_watex_logger(self.__class__.__name__)
        
        self._base_estimator = base_estimator 
        self.grid_params = grid_params 
        self.scoring = scoring 
        self.cv = cv 
        self._kind = kind 
        
        self.best_params_ =None 
        self.cv_results_= None
        self.feature_importances_= None
        self.best_estimator_=None 
    
        if len(grid_kws)!=0: 
            self.__setattr__('grid_kws', grid_kws)
            
    @property 
    def base_estimator (self): 
        """ Return the base estimator class"""
        return self._base_estimator 
    
    @base_estimator.setter 
    def base_estimator (self, baseEstim): 
        if not inspect.isclass(baseEstim) or\
            type(self.estimator) != ABCMeta: 
            raise TypeError(f'Expected an Estimator not {type(baseEstim)!r}')
            
        self._base_estimator =baseEstim 
        
    @property 
    def kind(self): 
        """ Kind of searched. `RandomizedSearchCV` or `GridSearchCV`."""
        return self._kind 
    
    @kind.setter 
    def kind (self, typeOfsearch): 
        """`kind attribute checker"""
        if typeOfsearch ==1 or 'GridSearchCV'.lower(
                ).find(typeOfsearch.lower())>=0: 
            typeOfsearch = 'GridSearchCV'
            
        if typeOfsearch ==2 or  'RandomizedSearchCV'.lower(
                ).find(typeOfsearch.lower())>=0:
            typeOfsearch = 'RandomizedSearchCV'
    
        else: 
            raise ValueError('Expected %r or %r not %s.'
                             %('gridSearchCV','RandomizedSearchCV', 
                               typeOfsearch ))
            
        self._kind = typeOfsearch 

    def fit(self,  X, y, **grid_kws): 
        """ Fit method using base Estimator.
        
        Populate gridSearch attributes. 
        
        :param X: Train dataset 
        :param y: Labels
        :param grid_kws: Additional keywords arguments of Gird search.
            keywords arguments must be the inner argunents of `GridSearchCV` or 
            `RandomizedSearchCV`.
        """
        
        if hasattr(self, 'grid_kws'): 
            grid_kws = getattr(self, 'grid_kws')
            
        if type(self.base_estimator) == ABCMeta :
            
            baseEstimatorObj = self.base_estimator()
            # get the base estimators parameters values in case for logging 
            # and users warnings except the `self`.
            init_signature = inspect.signature(baseEstimatorObj.__init__)
            
            parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
            
            self._logging.info('%s estimator (type %s) could not be cloned. Need'
                               ' to create an instance with default arguments '
                               ' %r for cross validatation grid search.'
                               %(repr(baseEstimatorObj.__class__),
                                 repr(type(baseEstimatorObj)), parameters))
            
            warnings.warn('%s estimator (type %s) could not be cloned.'
                          'Need to create an instance with default arguments '
                          ' %r for cross validatation grid search.'
                            %(repr(baseEstimatorObj.__class__),
                            repr(type(baseEstimatorObj)), parameters), 
                            UserWarning)
            
        else : 
            # suppose an instance is created before running the 
            # `SearchedGrid` class. 
            baseEstimatorObj  = self.base_estimator 
        
        if self.kind =='GridSearchCV': 
            try: 
                gridObj = GridSearchCV(baseEstimatorObj  , 
                                        self.grid_params,
                                        scoring = self.scoring , 
                                        cv = self.cv,
                                        **grid_kws)
            except TypeError: 
                warnings.warn('%s does not accept the param %r arguments.'
                              %(GridSearchCV.__class__, grid_kws),
                              RuntimeWarning)
                _logger.warnings('Unacceptable params %r arguments'
                                      % grid_kws)
            
        elif self.kind =='RandomizedSearchCV':
            
            try: 
                gridObj = RandomizedSearchCV(baseEstimatorObj ,
                                            self.grid_params,
                                            scoring = self.scoring,
                                            **grid_kws
                                                     )
            except TypeError:
                warnings.warn('%s does not accept the param %r arguments.'
                              %(RandomizedSearchCV.__class__, grid_kws),
                              RuntimeWarning)
                self.logging.warnings('Unacceptable params %r arguments'
                                      %self.grid_kws)
        try : 
            # fit gridSearchObject.
            gridObj.fit(X,y)
            
        except TypeError : 
  
            init_signature = inspect.signature(baseEstimatorObj.__init__)
            parameters = [p for p in init_signature.parameters.values()
                          if p.name != 'self' ] 
            
            warnings.warn('sklearn.clone error. Cannot clone object %s.'
                          'To avoid future warning, Create an instance of'
                          'estimator and set the instance as %s arguments.' 
                          %(repr(baseEstimatorObj ),type(baseEstimatorObj )),
                          FutureWarning)
            
            self._logging.warning('Trouble of clone estimator. Create an instance '
                                  ' of estimator and set as %r base_estimator'
                                  ' arguments before runing the {type(self)!r}'
                                  'class. Please create instance with %s params'
                                  'values.'%(repr(type(baseEstimatorObj)), 
                                             repr(parameters)))
            
            return 
        
        for param_ , param_value_ in zip(
                ['best_params_','best_estimator_',
                  'cv_results_','feature_importances_'],
                            [gridObj.best_params_, gridObj.best_estimator_, 
                             gridObj.cv_results_, 
                             gridObj.best_estimator_.feature_importances_]):
            
            setattr(self, param_, param_value_)
            
        #resetting the grid-kws attributes 
        setattr(self, 'grid_kws', grid_kws)
        

class AttributeCkecker(ABC): 
    """ Check attributes and inherits from module `abc` for Data validators. 
    
    Validate DataType mainly `X` train or test sets and `y` labels or
    and any others params types.
    """
    
    def __set_name__(self, owner, name): 
        try: 
            self.private_name = '_' + name 
        except AttributeError: 
            warnings.warn('Object {owner!r} has not attribute {name!r}')
            
    def __get__(self, obj, objtype =None):
        return getattr(obj, self.private_name) 
    
    def __set__(self, obj, value):
        self.validate(value)
        setattr(obj, self.private_name, value) 
        
    @abstractmethod 
    def validate(self, value): 
        pass 

class checkData (AttributeCkecker): 
    """ Descriptor to check data type `X` or `y` or else."""
    def __init__(self, Xdtypes):
        self.Xdtypes =eval(Xdtypes)

    def validate(self, value) :
        """ Validate `X` and `y` type."""
        if not isinstance(value, self.Xdtypes):
            raise TypeError(
                f'Expected {value!r} to be one of {self.Xdtypes!r} type.')
            
class checkValueType_ (AttributeCkecker): 
    """ Descriptor to assert parameters values. Default assertion is 
    ``int`` or ``float``"""
    def __init__(self, type_):
        self.six =type_ 
        
    def validate(self, value):
        """ Validate `cv`, `s_ix` parameters type"""
        if not isinstance(value,  self.six ): 
            raise ValueError(f'Expected {self.six} not {type(value)!r}')
   
class  checkClass (AttributeCkecker): 
    def __init__(self, klass):
        self.klass = klass 
       
    def validate(self, value): 
        """ Validate the base estimator whether is a class or not. """
        if not inspect.isclass(value): 
            raise TypeError('Estimator might be a class object '
                            f'not {type(value)!r}.')
        
class BaseEvaluation: 
    """ Evaluation of dataset using a base estimator.
    
    Quick evaluation after data preparing and pipeline constructions. 
    
    :param base_estimator: obj 
        estimator for trainset and label evaluating 
        
    :param X: ndarray of dataframe of trainset data
    
    :param y: array of labels data 
    
    :param s_ix: int, sampling index. 
        If given, will sample the `X` and `y` 
            
    :param columns: list of columns. Use to build dataframe `X` when `X` is 
        given as numpy ndarray. 
        
    :param pipeline: callable func 
            Tranformer data and preprocessing 
    :param cv: cross validation splits. Default is ``4``.
            
    """
   
    # descriptors to control the attributes values 
    # estimator = checkClass(object)
    X= checkData('(pd.DataFrame, np.ndarray, pd.Series)')
    y= checkData('(pd.DataFrame, np.ndarray, pd.Series)')
    s_ix = checkValueType_((int, float))
    cv = checkValueType_((int, float))
    
    def __init__(self, 
                 base_estimator,
                 X, 
                 y=None,
                 s_ix=None,
                 cv=7,  
                 pipeline:Callable[..., T]= None, 
                 columns =None, 
                 **kwargs): 
        
        self._logging =watexlog().get_watex_logger(self.__class__.__name__)
        
        self.estimator = base_estimator
        self.X= X 
        self.y =y 
        self.s_ix =s_ix 
        self.cv = cv 
        self.columns =columns 
        self.pipeline =pipeline
        
        for key in list(kwargs.keys()): 
            setattr(self, key, kwargs[key])
        
        if (self.estimator and X) is not None : 
            self.quickEvaluateEstimator()
            
    def quickEvaluateEstimator(self, fit='yes', **kws): 
        
        """ Quick methods used to evaluate eastimator, display the 
        error results as well as the sample model_predictions.
        
        :param X: Dataframe  be trained 
        :param y: labels from trainset 
        :param sample_ix: index to sample in the trainset and labels. 
        :param kws: Estmator additional keywords arguments. 
        :param fit: Fit the method for quick estimating 
            Default is ``yes`` 
            
        """
        pprint =kws.pop('pprint', True) 
        compute_cross = kws.pop('compute_cross', True)
        scoring = kws.pop('scoring', 'neg_mean_squared_error' )
        
        for objattr, objvalue in zip(['pprint', 'compute_cross', 'scoring'],
                               [pprint, compute_cross, scoring]) :
            
             if not hasattr(self, objattr): 
                 setattr(self, objattr, objvalue)
                 
        self._logging.info ('Quick estimation using the %r estimator with'
                            'config %r arguments %s.'
                            %(repr(self.estimator),self.__class__.__name__, 
                            inspect.getfullargspec(self.__init__)))
        
        if hasattr(self, 'random_state'): 
            kws.__setitem__('random_state', getattr(self, 'random_state'))
            
        try: 
            obj = self.estimator (**kws)
        except TypeError: 
            obj = self.estimator()
        # if self.pipeline is None:
        #     self.pipeline = full_pipeline 
            
        if self.s_ix is not None: 
            if isinstance(self.X, pd.DataFrame): 
                self.X= self.X.iloc[: int(self.s_ix)]
    
            elif isinstance(self.X, np.ndarray): 
                if self.columns is None:
                    warnings.warn(
                        f'{self.X!r} must be a dataframe but nocolumns found!'
                          ' Could not create a new dataframe.',UserWarning)
                                 
                if self.columns is not None: 
                    if self.X.shape[2] !=len(self.columns): 
                        warnings.warn(
                            f'Expected {self.X.shape[2]!r} but {len(self.columns)} '
                            f'{"is" if len(self.columns)< 2 else"are"}',
                            RuntimeWarning)
                        
                        raise IndexError('Expectted %i not %i self.columns.'
                                         %(self.X.shape[2], len(self.columns)))
                        
                    self.X= pd.DataFrame(self.X, self.columns)
                    
                self.X= self.X.iloc[: int(self.s_ix)]
    
    
            self.y= self.y[:int(self.s_ix )]  
    
        if isinstance(self.y, pd.Series): 
            self.y =self.y.values 
   
        if fit =='yes': 
            self.fit_data(obj, pprint= self.pprint,
                          compute_cross=self.compute_cross,
                          scoring = self.scoring)
            
            
    def fit_data (self, obj , pprint=True, compute_cross=True, 
                  scoring ='neg_mean_squared_error' ): 
        """ Fit data once verified and compute the ``rmse`` scores.
        
        :paramm obj: base estimator with base params
        :param pprint: Display prediction of the quick evaluation 
        ;param compute_cross: compute the cross validation 
        :param scoring: Type of scoring for cross validation. Please refer to  
                 :doc:~slkearn.sklearn.model_selection.cross_val_score
                 for further details.
        """
        def display_scores(scores): 
            """ Display scores..."""
            print('scores:', scores)
            print('Mean:', scores.mean())
            print('standard deviation:', scores.std())
            
        self._logging.info('Fit data X with shape {X.shape!r}.')
        
        train_prepared_obj =self.pipeline.fit_transform(self.X)
        
        obj.fit(train_prepared_obj, self.y)
 
        if pprint: 
             print("predictions:\t", obj.predict(train_prepared_obj ))
             print("Labels:\t\t", list(self.y))
            
        y_obj_predicted = obj.predict(train_prepared_obj)
        
        obj_mse = mean_squared_error(self.y ,
                                     y_obj_predicted)
        self.rmse = np.sqrt(obj_mse )
        
        
        if compute_cross : 
            
            scores = cross_val_score(obj, train_prepared_obj,
                                     self.y, 
                                     cv=self.cv,
                                     scoring='neg_mean_squared_error' 
                                     )
            
            if scoring == 'neg_mean_squared_error': 
                
                self.rmse_scores = np.sqrt(-scores)
            else: 
                self.rmse_scores = np.sqrt(scores)
    
            if pprint:
                display_scores(self.rmse_scores)   
                
    def __getattribute__(self, item):
        """ Return attribute when trying to get `rmse_scores`. when is not 
        computed(`compute_cross` is set to ``False``.) Otherwise raise 
        errors for others attributes that are not sets.
        """
        try: 
            if item.startwith('rme_s'): 
                return super().__getattribute__(item)
            
        except AttributeError: 
            warnings.warn(f'{type(self)!r} has no attribute {item!r}'
                          '{item!r} is set to ``None``.', UserWarning)
            self.__dict__[item] =None 
            
            return 
        

if __name__=="__main__": 
    from sklearn.ensemble import RandomForestClassifier
    from watex.utils._data_preparing_ import bagoue_train_set_prepared #as TRAINSET_PREPARED 
    from watex.utils._data_preparing_ import bagoue_label_encoded #as TRAINSET_LABEL_ENCODED 
    grid_params = [
            {'n_estimators':[3, 10, 30], 'max_features':[2, 4, 6, 8]}, 
            {'bootstrap':[False], 'n_estimators':[3, 10], 'max_features':[2, 3, 4]}
            ]
    
    forest_clf = RandomForestClassifier()
    grid_search = SearchedGrid(forest_clf, grid_params, kind='RandomizedSearchCV')
    grid_search.fit(X= bagoue_train_set_prepared , y = bagoue_label_encoded)
    from pprint import pprint  
    pprint(grid_search.best_params_ )
    pprint(grid_search.cv_results_)
    
    # df = load_data('data/geo_fdata')
    # # df.hist(bins=50, figsize=(20, 15))
    
    # data = discretizeCategoriesforStratification(df, in_cat='flow', new_cat='tempf_',
    #                                       divby =1, higherclass=3)

    # print(searchObj.X)
    
 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        