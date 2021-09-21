# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Wed Jul 14 20:00:26 2021
# This module is a set of utils for data prepprocessing
# released under a MIT- licence.
"""
Created on Sat Aug 28 16:26:04 2021

@author: @Daniel03

"""
import os 
import hashlib 
import tarfile 
import inspect
import warnings  
from six.moves import urllib 
from typing import TypeVar, Generic, Iterable , Callable, Text
from abc import ABC, abstractmethod, ABCMeta  
import pandas as pd 
import numpy as np 

from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from sklearn.model_selection import cross_val_predict 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import confusion_matrix , f1_score
from sklearn.metrics import roc_curve, roc_auc_score

from watex.utils._watexlog import watexlog
import watex.utils.decorator as deco
import watex.utils.exceptions as Wex
import watex.viewer.hints as Hints

T= TypeVar('T')
KT=TypeVar('KT')
VT=TypeVar('VT')

_logger = watexlog().get_watex_logger(__name__)

DOWNLOAD_ROOT = 'https://github.com/WEgeophysics/watex/master/'
#'https://zenodo.org/record/4896758#.YTWgKY4zZhE'
DATA_PATH = 'data/tar.tgz_files'
TGZ_FILENAME = '/bagoue.main&rawdata.tgz'
CSV_FILENAME = 'main.bagciv.data.csv'#'_bagoue_civ_loc_ves&erpdata4.csv'

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
    
    :param grid_params: list of hyperparamters params  to be tuned 
    
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
            
            return self
        
        for param_ , param_value_ in zip(
                ['best_params_','best_estimator_','cv_results_'],
                [gridObj.best_params_, gridObj.best_estimator_, 
                             gridObj.cv_results_ ]
                             ):
            
            setattr(self, param_, param_value_)
        try : 
            attr_value = gridObj.best_estimator_.feature_importances_
        except AttributeError: 
            warnings.warn ('{0} object has no attribute `feature_importances_`'.
                           format(gridObj.best_estimator_.__class__.__name__))
            setattr(self,'feature_importances_', None )
        else : 
            setattr(self,'feature_importances_', attr_value)
            
        #resetting the grid-kws attributes 
        setattr(self, 'grid_kws', grid_kws)
        
        return self
    
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
        
class BaseEvaluation (object): 
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
   
    def __init__(self, 
                 base_estimator,
                 X, 
                 y,
                 s_ix=None,
                 cv=7,  
                 pipeline= None, 
                 columns =None, 
                 pprint=True, 
                 cvs=True, 
                 scoring ='neg_mean_squared_error',
                 **kwargs): 
        
        self._logging =watexlog().get_watex_logger(self.__class__.__name__)
        
        self.base_estimator = base_estimator
        self.X= X 
        self.y =y 
        self.s_ix =s_ix 
        self.cv = cv 
        self.columns =columns 
        self.pipeline =pipeline
        self.pprint =pprint 
        self.cvs = cvs
        self.scoring = scoring
        
        for key in list(kwargs.keys()): 
            setattr(self, key, kwargs[key])

        if self.X is not None : 
            self.quickEvaluation()
            
    def quickEvaluation(self, fit='yes', **kws): 
        
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
        if pprint is not None: 
            self.pprint = pprint 
        cvs = kws.pop('cvs', True)
        if cvs is not None: 
            self.cvs = cvs 
        scoring = kws.pop('scoring', 'neg_mean_squared_error' )
        if scoring is not None: 
            self.scoring  = scoring 
            
        self._logging.info ('Quick estimation using the %r estimator with'
                            'config %r arguments %s.'
                            %(repr(self.base_estimator),self.__class__.__name__, 
                            inspect.getfullargspec(self.__init__)))
        
        if not hasattr(self, 'random_state'):
            self.random_state =42 
            
            try:
                if kws.__getitem__('random_state') is not None : 
                    setattr(self, 'random_state', kws['random_state'])
            except KeyError: 
                self.random_state =42 
  
        if not inspect.isclass(self.base_estimator) or \
              type(self.base_estimator) !=ABCMeta:
                if type(self.base_estimator).__class__.__name__ !='type':
                    raise TypeError('Estimator might be a class object '
                                    f'not {type(self.base_estimator)!r}.')
                
        if type(self.base_estimator) ==ABCMeta:  
            try: 
                self.base_estimator  = self.base_estimator (**kws)
            except TypeError: 
                self.base_estimator  = self.base_estimator()

        if  self.s_ix is None: 
            self.s_ix = int(len(self.X)/2)

        if self.s_ix is not None: 
            if isinstance(self.X, pd.DataFrame): 
                self.X= self.X.iloc[: int(self.s_ix)]
            elif isinstance(self.X, np.ndarray): 
                if self.columns is None:
                    warnings.warn(
                        f'{self.columns!r} must be a dataframe columns!'
                          f' not {type(self.columns)}.',UserWarning)
                    
                    if self.X.ndim ==1 :
                        size =1 
                    elif self.X.ndim >1: 
                        size = self.X.shape[1]
                    
                    return TypeError(f'Expected {size!r} column name'
                                      '{"s" if size >1 else 1} for array.')

                elif self.columns is not None: 
                    if self.X.shape[1] !=len(self.columns): 
                        warnings.warn(f'Expected {self.X.shape[1]!r}' 
                                      f'but {len(self.columns)} '
                                      f'{"is" if len(self.columns) < 2 else"are"} '
                                      f'{len(self.columns)!r}.',RuntimeWarning)
         
                        raise IndexError('Expected %i not %i self.columns.'
                                          %(self.X.shape[2], 
                                            len(self.columns)))
                        
                    self.X= pd.DataFrame(self.X, self.columns)
                    
                self.X= self.X.iloc[: int(self.s_ix)]
    
            self.y= self.y[:int(self.s_ix )]  
    
        if isinstance(self.y, pd.Series): 
            self.y =self.y.values 
   
        if fit =='yes': 
            self.fit_data(self.base_estimator , pprint= self.pprint,
                          compute_cross=self.cvs,
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
            print('rmse scores:', np.sqrt(scores))
            print('standard deviation:', scores.std())
            
        self._logging.info('Fit data X with shape {X.shape!r}.')
        
        if self.pipeline is not None: 
            train_prepared_obj =self.pipeline.fit_transform(self.X)
            
        elif self.pipeline is None: 
            warnings.warn('No Pipeline is applied. Could estimate with purely'
                          '<%r> given estimator.'%(self.base_estimator.__name__))
            self.logging.info('No Pipeline is given. Evaluation should be based'
                              'using  purely  the given estimator <%r>'%(
                                  self.base_estimator.__name__))
            
            train_prepared_obj =self.base_estimator.fit_transform(self.X)
        
        obj.fit(train_prepared_obj, self.y)
 
        if pprint: 
             print("predictions:\t", obj.predict(train_prepared_obj ))
             print("Labels:\t\t", list(self.y))
            
        y_obj_predicted = obj.predict(train_prepared_obj)
        
        obj_mse = mean_squared_error(self.y ,
                                     y_obj_predicted)
        self.rmse = np.sqrt(obj_mse )

        if compute_cross : 
            
            self.scores = cross_val_score(obj, train_prepared_obj,
                                     self.y, 
                                     cv=self.cv,
                                     scoring=self.scoring
                                     )
            
            if self.scoring == 'neg_mean_squared_error': 
                self.rmse_scores = np.sqrt(-self.scores)
            else: 
                self.rmse_scores = np.sqrt(self.scores)
    
            if pprint:
                if self.scoring =='neg_mean_squared_error': 
                    self.scores = -self.scores 
                display_scores(self.scores)   
                

class DimensionReduction: 
    """ Reduce dimension for data visualisation.
    
    Reduce number of dimension down to two (or to three) make  it possible 
    to plot high-dimension trainsing set on the graph and often gain some 
    important insights by visually detecting patterns, such as clusters. 
    """
    
    def PCA(self,X, n_components=None, plot_projection=False, 
            plot_kws=None, n_axes =None,  **pca_kws ): 
        """Principal Components analysis (PCA) is by far themost popular
        dimensional reduction algorithm. First it identifies the hyperplane 
        that lies closest to the data and project it to the data onto it.
        
        :param X: Dataset compose of n_features items for dimension reducing
        
        :param n_components: Number of dimension to preserve. If`n_components` 
                is ranged between float 0. to 1., it indicated the number of 
                variance ratio to preserve. If ``None`` as default value 
                the number of variance to preserve is ``95%``.
        :param plot_projection: Plot the explained varaince as a function 
        of number of dimension. Deafualt is``False``.
        
        :param n_axes: Number of importance components to retrieve the 
            variance ratio. If ``None`` the features importance is computed 
            usig the cumulative variance representative of 95% .
        
        :param plot_kws: Additional matplotlib.pyplot keywords arguments. 
        
        :Example: 
            
            >>> from watex.utils import DimensionReduction
            >>> from .datasets.data_preparing import X_train_2
            >>> DimensionReduction().PCA(X_train_2, 0.95, n_axes =3)
            >>> pca.components_
            >>> pca.features_importances_
        """
        def findFeaturesImportances(fnames, components, n_axes=2): 
            """ Retrive the features importance with variance ratio.
            
            :param fnames: array_like of feature's names
            :param components: pca components on different axes 
            """
            pc =list()
            if components.shape[0] < n_axes : 
                
                warnings.warn(f'Retrieved axes {n_axes!r} no more than'
                              f' {components.shape[0]!r}. Reset to'
                              f'{components.shape[0]!r}', UserWarning)
                n_axes = int(components.shape[0])
            
            for i in range(n_axes): 
                # reverse from higher values to lower 
                index = np.argsort(abs(components[i, :]))
                comp_sorted = components[i, :][index][::-1]
                numf = fnames [index][::-1]
                pc.append((f'pc{i+1}', numf, comp_sorted))
                
            return pc 
        
        from sklearn.decomposition import PCA 
        
        if n_components is None: 
            # choose the right number of dimension that add up to 
            # sufficiently large proportion of the variance 0.95%
            pca=PCA(**pca_kws)
            pca.fit(X)
            cumsum =np.cumsum( pca.explained_variance_ratio_ )
            # d= np.argmax(cumsum >=0.95) +1 # for index 
            
            # we can set the n_components =d then run pca again or set the 
            # value of n_components betwen 0. to 1. indicating the ratio of 
            # the variance we wish to preserve.
        pca = PCA(n_components=n_components, **pca_kws)
        self.X_= pca.fit_transform(X) # X_reduced = pca.fit_transform(X)
  
        if n_components is not None: 
            cumsum = np.cumsum(pca.explained_variance_ratio_ )
        
        if plot_projection: 
            import matplotlib.pyplot as plt
            
            if plot_kws is None: 
                plot_kws ={'label':'Explained variance as a function of the'
                           ' number of dimension' }
            plt.plot(cumsum, **plot_kws)
            # plt.plot(np.full((cumsum.shape), 0.95),
            #          # np.zeros_like(cumsum),
            #          ls =':', c='r')
            plt.xlabel('Dimensions')
            plt.ylabel('Explained Variance')
            plt.title('Explained variance as a function of the'
                        ' number of dimension')
            plt.show()
            
        # make introspection and set the all pca attributes to self.
        for key, value in  pca.__dict__.items(): 
            setattr(self, key, value)
        
        if n_axes is None : 
            self.n_axes = pca.n_components_
        else : 
            setattr(self, 'n_axes', n_axes)
            
        # get the features importance and features names
        self.feature_importances_= findFeaturesImportances(
                                        np.array(list(X.columns)), 
                                        pca.components_, 
                                        self.n_axes)
        
        return self 

class Metrics: 
    """ Metric class.
    
    Metrics are measures of quantitative assessment commonly used for 
    assessing, comparing, and tracking performance or production. Generally,
    a group of metrics will typically be used to build a dashboard that
    management or analysts review on a regular basis to maintain performance
    assessments, opinions, and business strategies.
    
    Here we implement some Scikit-learn metrics like `precision`, `recall`
    `f1_score` , `confusion matrix`, and `receiving operating characteristic`
    (R0C)
    """ 
    
    def precisionRecallTradeoff(self, 
                                clf,
                                X,
                                y,
                                cv =3,
                                classe_ =None,
                                method="decision_function",
                                cross_val_pred_kws =None,
                                y_tradeoff =None, 
                                **prt_kws):
        """ Precision/recall Tradeoff computes a score based on the decision 
        function. 
        
        Is assign the instance to the positive class if that score on 
        the left is greater than the `threshold` else it assigns to negative 
        class. 
        
        Parameters
        ----------
        
        clf: obj
            classifier or estimator
            
        X: ndarray, 
            Training data (trainset) composed of n-features.
            
        y: array_like 
            labelf for prediction. `y` is binary label by defaut. 
            If '`y` is composed of multilabel, specify  the `classe_` 
            argumentto binarize the label(`True` ot `False`). ``True``  
            for `classe_`and ``False`` otherwise.
            
        cv: int 
            K-fold cross validation. Default is ``3``
            
        classe_: float, int 
            Specific class to evaluate the tradeoff of precision 
            and recall. If `y` is already a binary classifer, `classe_` 
            does need to specify. 
            
        method: str
            Method to get scores from each instance in the trainset. 
            Ciuld be ``decison_funcion`` or ``predict_proba`` so 
            Scikit-Learn classifuier generally have one of the method. 
            Default is ``decision_function``.
        
        y_tradeoff: float
            check your `precision score` and `recall score`  with a 
            specific tradeoff. Suppose  to get a precision of 90%, you 
            might specify a tradeoff and get the `precision score` and 
            `recall score` by setting a `y-tradeoff` value.

        Notes
        ------
            
        Contreverse to the `confusion matrix`, a precision-recall 
        tradeoff is very interesting metric to get the accuracy of the 
        positive prediction named ``precison`` of the classifier with 
        equation is:: 
            
            precision = TP/(TP+FP)
            
        where ``TP`` is the True Positive and ``FP`` is the False Positive
        A trival way to have perfect precision is to make one single 
        positive precision (`precision` = 1/1 =100%). This would be usefull 
        since the calssifier would ignore all but one positive instance. So 
        `precision` is typically used along another metric named `recall`,
         also `sensitivity` or `true positive rate(TPR)`:This is the ratio of 
        positive instances that are corectly detected by the classifier.  
        Equation of`recall` is given as:: 
            
            recall = TP/(TP+FN)
            
        where ``FN`` is of couse the number of False Negatives. 
        It's often convenient to combine `preicion`and `recall` metrics into
        a single metric call the `F1 score`, in particular if you need a 
        simple way to compared two classifiers. The `F1 score` is the harmonic 
        mean of the `precision` and `recall`. Whereas the regular mean treats 
        all  values equaly, the harmony mean gives much more weight to low 
        values. As a result, the classifier will only get the `F1 score` if 
        both `recalll` and `preccion` are high. The equation is given below::
            
            F1= 2/((1/precision)+(1/recall))
            F1= 2* precision*recall/(precision+recall)
            F1 = TP/(TP+ (FN +FP)/2)
        
        The way to increase the precion and reduce the recall and vice versa
        is called `preicionrecall tradeoff`
        
        Examples
        --------
        
        >>> from sklearn.linear_model import SGDClassifier
        >>> from watex.utils.ml_utils import Metrics 
        >>> sgd_clf = SGDClassifier()
        >>> mObj = Metrics(). precisionRecallTradeoff(clf = sgd_clf, 
        ...                                           X= X_train_2, 
        ...                                         y = y_prepared, 
        ...                                         classe_=1, cv=3 )                                
        >>> mObj.confusion_matrix 
        >>> mObj.f1_score
        >>> mObj.precision_score
        >>> mObj.recall_score
        """
        
        # check y if value to plot is binarized ie.True of false 
        y_unik = np.unique(y)
        if len(y_unik )!=2 and classe_ is None: 

            warnings.warn('Classes value of `y` is %s, but need 2.' 
                          '`PrecisionRecall Tradeoff` is used for training '
                           'binarize classifier'%len(y_unik ), UserWarning)
            self._logging.warning('Need a binary classifier(2). %s are given'
                                  %len(y_unik ))
            raise ValueError(f'Need binary classes but {len(y_unik )!r}'
                             f' {"are" if len(y_unik )>1 else "is"} given')
            
        if classe_ is not None: 
            try : 
                classe_= int(classe_)
            except ValueError: 
                raise Wex.WATexError_inputarguments(
                    'Need integer value. Could not convert to Float.')
            except TypeError: 
                raise Wex.WATexError_inputarguments(
                    'Could not convert {type(classe_)!r}') 
        
            if classe_ not in y: 
                raise Wex.WATexError_inputarguments(
                    'Value must contain a least a value of label '
                        '`y`={0}'.format(
                            Hints.format_generic_obj(y).format(*list(y))))
                                     
            y=(y==classe_)
            
        if cross_val_pred_kws is None: 
            cross_val_pred_kws = dict()
            
        self.y_scores = cross_val_predict(clf,
                                          X, 
                                          y, 
                                          cv =cv,
                                          method= method,
                                          **cross_val_pred_kws )

        y_scores = cross_val_predict(clf,
                                     X,
                                     y, 
                                     cv =cv,
                                     **cross_val_pred_kws )
        self.confusion_matrix =confusion_matrix(y, y_scores )
        
        self.f1_score = f1_score(y,y_scores)
        self.precision_score = precision_score(y, y_scores)
        self.recall_score= recall_score(y, y_scores)
            
        if method =='predict_proba': 
            # if classifier has a `predict_proba` method like `Random_forest`
            # then use the positive class probablities as score 
            # score = proba of positive class 
            self.y_scores =self.y_scores [:, 1] 
            
        if y_tradeoff is not None:
            try : 
                float(y_tradeoff)
            except ValueError: 
                raise Wex.WATexError_float(
                    f'Could not convert {y_tradeoff!r} to float.')
            except TypeError: 
                raise Wex.WATexError_inputarguments(
                    f'Invalid type `{type(y_tradeoff)}`')
                
            y_score_pred = (self.y_scores > y_tradeoff) 
            self.precision_score_tradeoff = precision_score(y,
                                                            y_score_pred)
            self.recall_score_tradeoff = recall_score(y, 
                                                      y_score_pred)
            
        self.precisions, self.recalls, self.thresholds =\
            precision_recall_curve(y,
                                   self.y_scores,
                                   **prt_kws)
            
        self.y =y
        
        return self 
    
    @deco.docstring(precisionRecallTradeoff, start ='Parameters', end ='Notes')
    def ROC_curve(self, roc_kws=None, **tradeoff_kws): 
        """The Receiving Operating Characteric (ROC) curve is another common
        tool  used with binary classifiers. 
        
        It s very similar to preicision/recall , but instead of plotting 
        precision versus recall, the ROC curve plots the `true positive rate`
        (TNR)another name for recall) against the `false positive rate`(FPR). 
        The FPR is the ratio of negative instances that are correctly classified 
        as positive.It is equal to one minus the TNR, which is the ratio 
        of  negative  isinstance that are correctly classified as negative.
        The TNR is also called `specify`. Hence the ROC curve plot 
        `sensitivity`(recall) versus 1-specifity.
        
        Parameters 
        ----------
        clf: callable
            classifier or estimator
                
        X: ndarray, 
            Training data (trainset) composed of n-features.
            
        y: array_like 
            labelf for prediction. `y` is binary label by defaut. 
            If '`y` is composed of multilabel, specify  the `classe_` 
            argumentto binarize the label(`True` ot `False`). ``True``  
            for `classe_`and ``False`` otherwise.
            
        roc_kws: dict 
            roc_curve additional keywords arguments
            
        See also
        --------
        
            `ROC_curve` deals wuth optional and positionals keywords arguments 
            of :meth:`~watex.utlis.ml_utils.Metrics.precisionRecallTradeoff`
            
        Examples
        ---------
        
            >>> from sklearn.linear_model import SGDClassifier
            >>> from watex.utils.ml_utils import Metrics 
            >>> sgd_clf = SGDClassifier()
            >>> rocObj = Metrics().ROC_curve(clf = sgd_clf,  X= X_train_2, 
            ...                                 y = y_prepared, classe_=1, cv=3 )
            >>> rocObj.fpr
        """
        self.precisionRecallTradeoff(**tradeoff_kws)
        if roc_kws is None: roc_kws =dict()
        self.fpr , self.tpr , thresholds = roc_curve(self.y, 
                                           self.y_scores,
                                           **roc_kws )
        self.roc_auc_score = roc_auc_score(self.y, self.y_scores)
        
        return self 
    
if __name__=="__main__": 
#     if __package__ is None : 
#         __package__='watex'
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.linear_model import SGDClassifier
    # from .datasets import X_, y_,  X_prepared, y_prepared, default_pipeline
    from watex.datasets.data_preparing import X_train_2


#     grid_params = [
#             {'n_estimators':[3, 10, 30], 'max_features':[2, 4, 6, 8]}, 
#             {'bootstrap':[False], 'n_estimators':[3, 10], 'max_features':[2, 3, 4]}
#             ]

    pca= DimensionReduction().PCA(X_train_2, 0.95, plot_projection=True, n_axes =3)
    print('columnsX=', X_train_2.columns)
    print('components=', pca.components_)
    print('feature_importances_:', pca.feature_importances_)
    # print(pca.X_.shape)
#     sgd_clf = SGDClassifier()
#     from watex.utils.ml_utils import Metrics 
#     # mObj = Metrics(). precisionRecallTradeoff(clf = sgd_clf,  X= X_train_2, 
#     #                                           y = y_prepared, classe_=1, cv=3 )
                                              
#     # print('conf_mx:=', mObj.confusion_matrix )
#     # print('f1_score=:', mObj.f1_score)
#     # print('precision_score=:', mObj.precision_score)
#     # print('recall_score=:', mObj.recall_score)
#     rocObj = Metrics().ROC_curve(clf = sgd_clf,  X= X_train_2, 
#                                               y = y_prepared, classe_=1, cv=3 )
#     print(rocObj.fpr)
#     # print('thresholds=:', mObj.thresholds)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        