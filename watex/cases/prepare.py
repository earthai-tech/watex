# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Base data preparation for case studies 
========================================

Base module helps to automate data preparation at once. It is created for fast 
data preparation in real engineering case study. The base steps has been 
used to solve a flow rate prediction problems [1]_. Its steps procedure 
can straighforwardly help user to fast reach the same goal as the published 
paper. An example of  different kind of Bagoue dataset [2]_ , is prepared  
using the `BaseSteps` module. 

References
------------
    
.. [1] Kouadio, K.L., Kouame, L.N., Drissa, C., Mi, B., Kouamelan, K.S., 
    Gnoleba, S.P.D., Zhang, H., et al. (2022) Groundwater Flow Rate 
    Prediction from Geo‐Electrical Features using Support Vector Machines. 
    Water Resour. Res. :doi:`10.1029/2021wr031623`
.. [2] Kouadio, K.L., Nicolas, K.L., Binbin, M., Déguine, G.S.P. & Serge, 
    K.K. (2021, October) Bagoue dataset-Cote d'Ivoire: Electrical profiling,
    electrical sounding and boreholes data, Zenodo. :doi:`10.5281/zenodo.5560937`

"""
from __future__ import annotations 

import inspect
import copy
import warnings
from pprint import pprint

import numpy as np
import pandas as pd 
 
from .._watexlog import watexlog
from ..exlib.sklearn import  (
    Pipeline,
    FeatureUnion, 
    SimpleImputer, 
    StandardScaler,
    OrdinalEncoder, 
    OneHotEncoder,
    LabelBinarizer,
    LabelEncoder, 
    MinMaxScaler,
    PCA,
) 
from .._typing import (
    Tuple, 
    List, 
    Optional, 
    F, 
    NDArray,
    ArrayLike, 
    Series,
    DataFrame, 
    )

from ..transformers import (
    DataFrameSelector,
    CombinedAttributesAdder, 
    CategorizeFeatures,
    StratifiedWithCategoryAdder
)
from .features import categorize_flow 
from ..utils.coreutils import _is_readable 
from ..utils.funcutils import smart_format,to_numeric_dtypes
from ..utils.mlutils import ( 
    formatGenericObj,
    split_train_test_by_id,
    selectfeatures, 
    )

_logger = watexlog().get_watex_logger(__name__)


class BaseSteps(object): 
    """
    Default Data  preparation steps

    By default, the :class:`BaseSteps` is used to prepare the DC 1d -resistivity 
    geoelectrical features before prediction. The predicted target was the 
    `flow` rate. 

    Parameters
    ----------
    tname: str, 
        A target name or label. In supervised learning the target name is 
        considered as the reference name of `y` or label variable.

    return_all: bool 
        return all the stratified trainset. When data is too large, can set 
        to ``False`` to take an sample of the stratified trainset. to evaluate 
        your model.
        
    drop_features: list 
        List of useless features  and clean the dataset.
        
    categorizefeature_props: list 
        list of properties to categorize a particular features in the dataset.
        It composed of the 'name of feature' to convert its numerical values 
        into categorical values , then the value range of data to be categorize 
        and finally the categorical name of that values range. For instance:: 
            
            categorizefeature_props= [
                ('flow', ([0., 1., 3.], ['FR0', 'FR1', 'FR2', 'FR3']))
                ]
        Please refer to :doc:`watex.utils.transformers.CategorizeFeatures` 
        fot furthers details.
        
    hash: bool, 
        If ``True``, it ensure that data will remain consistent accross 
        multiple runs, even if dataset is refreshed. Use test by id to hash 
        training and test sets when data is splitting. 
        
    add_attributes: list, optional
        Experience the combinaison <numerical> attributes. 
        List of features for combinaison. Decide to combine features to create
        a new feature value from `operator` parameters. By default, the 
        combinaison is ratio of the given attribute/numerical features. 
        For instance, ``attribute_names=['lwi', 'ohmS']`` will divide the 
        feature 'lwi' by 'ohmS'.
        
   operator: str, default ='/' 
        Type of operation to perform when combining features. 
        Can be ['/', '+', '-', '*', '%']  
        
    attribute_indexes: list of int,
        List of attributes indexes to combines. For instance:: 
            
            attribute_indexes = [1, 0] # or [4, 3]

        The operator by default is `division` . Indexes of each 
        attribute/feature for experiencing combinaison. User warning should 
        raise if any index does match the dataframe of array 
        columns.For more details, refer to 
        :class:`~watex.transformers.CombinedAttributesAdder`
 
    imputer_strategy: str 
        Type of strategy to replace the missing values. Refer to 
        :class:`~.sklearn.SimpleImputer`. Default is ``median``.
        
    missing_values : float
        The value to be replaced.  Default is ``np.nan`` values.
        
    pipeline: callable
        Pipeline to prepare the dataset. Default is :func:`defaultPipeline`.
        
    test_size: float, default=.2 i.e. 20% (X, y)
        The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
        respectively. 
        
    random_state : int, RandomState instance or None, default=42
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        
    verbose: int, `default` is ``0``    
        Control the level of verbosity. Higher value lead to more messages. 
        
    data: Filepath or Dataframe or shape (M, N) 
        Data is passed here as additional keyword arguments just for making
        under the `X` and `y` using method :neth:`~.stratifydata`. It 
        is :class:`pandas.DataFrame` containing samples of M  and features N. 
        
    Notes
    ------
    The data preparing includes is composed of two steps. The first step 
    includes: 
    -   The data cleaning by fixing and removing outliers, to replace the missing 
        values by the ``other values``  using param `imputer_strategy`rather than 
        to  get rid of the different instances (examples) or the whole feature. 
    -	The handling text and features consist to convert the categorial features
        labels to numbers to let the algorithm to well perform with non-numerical
        values. 
    -   The data stratification process is done before separating the dataset 
        into trainset and test set.  Indeed, the stratification consist to 
        divide the whole dataset into homogeneous subgroup to guarantee that
        the test set is most representative of the overall dataset. This is
        useful in our case because the dataset is not large enough to avoid 
        the risk of introducing a significant bias.  Once data are stratified,
        data are divided into a trainset (80%) and test set (20%). 
        
    ..   

    The second steps consist of features selection, features engineering, 
    encoding and data scaling using the pipeline via a parameter `pipeline`. 
    If None pipeline is given, the default pipline is triggered.The features
    engineering’s consist to aggregate features with experiencing combinations
    of attributes into promising new features using the params `attribute_indexes`
    after setting the argument `add_attributes` to ``True``. The final step of
    transformation consists of features scaling. The type of scaling used 
    by default in this module is the standardization because it less affected 
    by the outliers. 
    Each transformation step must be executed in right order therefore a
    full pipeline is created, composed of the numerical pipeline (deals with 
    numerical features) and categorical pipeline (deals with categorial 
    features). Both pipelines are combined and applied to the trainset and 
    later to the test set. 
     

    Examples
    ---------

    ../datasets/_p.py
    
    """
    def __init__(
            self,
            tname: Optional [str] = None, 
            return_all=True,
            drop_features: Optional[List [str]] = None, 
            categorizefeature_props: Tuple [str, List[int, str]] = None,
            add_attributes: bool = True, 
            attribute_indexes: List[Tuple[int]]= None, 
            operator:str='/', 
            imputer_strategy: str ='median', 
            missing_values: float | int  = np.nan, 
            pipeline: Optional[F] = None,
            test_size: float = 0.2,
            hash: bool = False,
            random_state: int = 42,
            verbose: int =0,
            **kwargs
       ):
        self._logging = watexlog.get_watex_logger(self.__class__.__name__)
        
        self.tname_=tname 
        self.drop_features=drop_features 
        self.categorizefeatures_props=categorizefeature_props 
        self.return_train=return_all
        self.add_attributes=add_attributes 
        self.attribute_indexes=attribute_indexes
        self.imputer_strategy=imputer_strategy 
        self.missing_values=missing_values
        self.pipeline=pipeline 
        self.test_size=test_size
        self.hash=hash
        self.random_state=random_state
        self.verbose=verbose

        self.data_ = None 
        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])
            
    @property 
    def data (self): 
        return self.data_ 
    @data.setter 
    def data(self, d): 
        """ Control whether a pd.datafrme of filetype."""
        self.data_ = _is_readable(d)

    def stratifydata (self,  data: str |DataFrame = None ): 
        """ Split and stratified data  and return stratified training and 
        test sets """
        if data is not None: 
            self.data_= data 
        if self.tname_ is None: 
            raise TypeError("Missing target name 'tname'. Data stratification"
                            " cannot be performed.")
            
        self.X, self.y, *__ = self.stratifyFolds(self.data )

        return self  
    
    def fit (self, 
             X: NDArray, 
             y: ArrayLike | Series =None, 
             ): 
        """ Preparing steps. 
        
        Parameters
        -----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.
            
        y: array-like, shape (M, ) ``M=m-samples``, 
            train target; Denotes data that may be observed at training time 
            as the dependent variable in learning, but which is unavailable 
            at prediction time, and is usually the target of prediction. 
            
        Return
        -------
        ``self``: `BaseSteps` instance 
            returns ``self`` for easy method chaining.
            
        """
        
        if self.verbose: 
            self._logging.info('Start the default preparing steps including'
                               ' Data cleaning,features combinaisons ... using '
                               ' the `fit` method!')
     
        if isinstance(y, pd.Series): 
            y =y.values 
        # convert to pandas X with features names 
        try:
            X =pd.DataFrame (X, columns = self.attribute_names_)
        except AttributeError: 
            
            self.attribute_names_= X.columns
            X =pd.DataFrame (X, columns = self.attribute_names_)
        # drop features if features are useles
        if self.drop_features is not None : 
            if isinstance(self.drop_features, str): 
                self.drop_features=[ self.drop_features]
                
            if self.verbose > 3 :
                self._logging.info('Dropping useless features {0}'.format(
                    formatGenericObj(self.drop_features)).format(
                        *self.drop_features))
            
            X.drop(self.drop_features, inplace =True, axis =1)
            
            # get new attributes names 
            self.attribute_names_ = X.columns 
        #convert numerical data in dataframe 
        for fname in self.attribute_names_: 
            try : 
                X= X.astype({fname:np.float64})
            except : 
                continue 
        #--->  try to convert labels into numerical features 
        # --> This  tranformers is called for classification problems
        if  self.categorizefeatures_props is not None :
            
            # control whether the target needs to be categorize 
            feat_to_ =[f[i] for i, f in enumerate(
                self.categorizefeatures_props)]
  
            if self.tname in feat_to_: 
                if self.verbose > 3: 
                    self._logging.info('Change the numerical target `%s` into'
                                      ' categorical features.'%self.tname)
                
                feat_ix = feat_to_.index(self.tname)
                gettarget_props =[self.categorizefeatures_props[feat_ix]]
                
                cObj = CategorizeFeatures(
                    num_columns_properties=gettarget_props)
                y = cObj.fit_transform(X=y)
                
                self.categorizefeatures_props.pop(feat_ix)
                feat_to_.pop(feat_ix)
                
            if len(self.categorizefeatures_props) !=0 : 
                if self.verbose > 3: 
                    self._logging.info(
                        f'Change the numerical features `{feat_to_}` into'
                                      ' categorical features.')
                cObj = CategorizeFeatures(
                    num_columns_properties=self.categorizefeatures_props)
                X = cObj.fit_transform(X)
 
        if isinstance(X, np.ndarray): 
            X= pd.DataFrame(X, columns = self.attribute_names_)
            
        # --> experiencing features combinaisons 
        if self.add_attributes : 
            if self.attribute_indexes is  None: 
                if self.verbose > 3: 
                    warnings.warn( 'Attributes indexes|names are None.'
                                  ' Set attributes indexes or names to experience'
                                  ' the attributes combinaisons.'
                                  )
            elif self.attribute_indexes is  not None:
                if self.verbose > 7 : 
                    try:
                        
                        self._logging.info('Experiencing combinaisons attributes'
                                          ' {0}.'.format(formatGenericObj (
                                              self.attribute_indexes)).format(
                                                  *self.attribute_indexes))
                    except : 
                        self._logging.info('Experiencing combinaisons attributes'
                                          ' {0}.'.format(self.attribute_indexes))
                                          
                cObj = CombinedAttributesAdder(
                    attribute_indexes= self.attribute_indexes , 
                    attribute_names =self.add_attributes
                                )
                X=cObj.fit_transform(X)  # return numpy array        
                    
               # get the attributes and create new pdFrame with new attributes 
                self.attribute_names_  = list(self.attribute_names_) +\
                    cObj.attribute_names_
            
        
            if isinstance(X, np.ndarray): 
                X= pd.DataFrame(X, columns= self.attribute_names_)
                
        #convert final data the default numerical features.
        self.cat_attributes_, self.num_attributes_=list(), list()
        for fname in self.attribute_names_: 
            if self.verbose > 7: 
                self._logging.info('Convert dataframe `X` to numeric ')
            try : 
                X = X.astype({fname:np.float64})
            except : 
                self.cat_attributes_.append(fname)
                continue 
            else: 
                self.num_attributes_.append(fname)

        self.X0 = pd.concat([X[self.num_attributes_],
                             X[self.cat_attributes_]],
                            axis =1)  
        self.y0 = y 
        
        return self
    
    def transform (self, X=None, y=None, on_testset =False): 
        """ Transform data applying the pipeline transformation.
        
        Parameters
        ---------
        X: ndarray, pd.DataFrame 
             X or dataframe X 
        y: array_like, 
            ylabel or target values 
        on_testset:str 
            Check whether the dataframe is evaluating on 
                testset or trainset 
        Returns
        -------
        - X_prepared. Data prepared after transformation 
        -y-prepared. label prepared after transformation.
        
        """
        if self.verbose:
            self._logging.info('Transform the data using the `transform` method.'
                               'Applying the pipeline.')
        
        for name, value in zip(['sparse_output','label_encoding'],
                                     [True,'LabelEncoder']): 
                                    
                if not hasattr(self, name) :
                    setattr(self, name, value)
                    
        pkws={'num_attributes': self.num_attributes_, 
                'cat_attributes': self.cat_attributes_, 
                'missing_values' : self.missing_values, 
                'strategy' : self.imputer_strategy, 
                'sparse_output': self.sparse_output,
                'label_encoding': self.label_encoding
                }
        
        if X is not None: 
            self.X0 = X 
        if y is not None: 
            self.y0=y 
        
        if self.pipeline is None :
            _, _, self.pipeline, self.y_prepared= default_pipeline(
                X= self.X0, 
                y = self.y0,
                **pkws
                )
        # keep another type of encodage using the Ordinal encoder           
        _, _, self.pca_pipeline,_ = default_pipeline(
            X= self.X0, 
            pca=True,
            **pkws
            )
            
        if on_testset : 
            if self.verbose > 3:
                self._logging.info(
                    '`You are running the sest set, so pipeline `transform`' 
                    ' method is applied not `fit_transform`.')
            try :
                self.X_prepared = self.pipeline.transform(self.X0)
            except: 
                self.X_prepared = self.pipeline.fit_transform(self.X0)
            
        if not on_testset: 
            if self.verbose > 3: 
                self._logging.info('Train set is running so `fit_transform` '
                                   'method is used.')
 
            self.X_prepared = self.pipeline.fit_transform(self.X0)
            # --> pipeline 
            self._X = self.pca_pipeline.fit_transform(self.X0)
            names = self.num_attributes_ + self.cat_attributes_

            self._Xpd = pd.DataFrame(self._X, 
                                     columns =names )
        
        return self.X_prepared, self.y_prepared 
    

    def fit_transform (self, X=None, y =None, 
                       on_testset=False ): 
        """ Fit transform apply `fit` and `transform` at Once. 
        
        Parameters
        ---------
         X: ndarray, pd.DataFrame 
             X or dataframe X 
        y: array_like, 
            ylabel or target 
        on_testset:str 
            Check whether the dataframe is evaluating on 
                testset or trainset 
        data: 
        Returns
        -------
            - X_prepared. Data prepared after transformation 
            -y-prepared. label prepared after transformation.
            
        """
        
        self.fit(X, y)
    
        self.transform(on_testset =on_testset)
        
        return self.X_prepared, self.y_prepared 
    
    @property 
    def tname (self): 
        return self.tname_
    @tname.setter
    def tname(self, label): 
        """ Check whether the target is among the data frame columns."""
        if not label in self.data.columns: 
            raise TypeError(
                f"{'None' if self.tname_ is None else self.tname_!r} target"
                f" is not found in {smart_format(self.data.columns)}.")
        self.tname_ =label 
        
    def stratifyFolds(self, data): 
        """ Stratified the dataset and return the trainset. Get more details 
        in :doc:`watex.bases.transformers.StratifiedWithCategoryAdder`."""

        smsg =''
        if not self.hash:
            sObj= StratifiedWithCategoryAdder(base_num_feature=self.tname, 
                                              test_size= self.test_size, 
                                              random_state = self.random_state,
                                              return_train=self.return_train)
            # return data with labels stratified
            # func_signature = inspect.signature(sObj)
            STRAT_PARAMS_VALUES = {k: v.default
                    for k, v in inspect.signature(sObj.__init__).parameters.items()
                    if v.default is not inspect.Parameter.empty
                }
            if self.verbose > 3 : 
                
                smsg =''.join([
                    f"Object {sObj.__class__.__name__!r} sucessffuly run",
                    f" to stratify data from base feature `{self.tname}`. ",
                    f' Default arguments are: `{STRAT_PARAMS_VALUES}`'])
                
            _X , __X = sObj.fit_transform(X=data)
        
        if self.hash :
            # ensure data to remain consistent even multiples runs.
            # usefull pratice to always keep test set as unseen data.
            _X , __X = split_train_test_by_id(data, test_ratio=self.test_size)
            if 'index' in list(_X.columns): 
                self._logging.debug(
                    '`index` used for hashing training and test sets  are '
                    'still on the dataset. So should be dropped to keep '
                    'the dataset safe as original.')
                
                mes='`index` used for hashing training and test '+\
                       ' sets should be dropped.'
                
                _X = _X.drop('index', axis =1)
                __X= __X.drop('index', axis=1)
      
                if self.verbose >1: 
                    pprint(mes + '\nNew columns of mileages'
                           f' are `{list(_X.columns)}`')

        # make a copy to keep
        X, y = _X.drop(self.tname, axis =1).copy(), _X[self.tname].copy() 
        X_, y_ = __X.drop(self.tname, axis =1).copy(), __X[self.tname].copy()
    
        # self.__X= __X.drop(self.tname, axis =1).copy()
        # self.__y= __X[self.tname].copy()
        self.__X = X_.copy()
        self.__y = y_.copy()
        
        self.attribute_names_= X.columns 
        if self.verbose >1 : 
            
            pprint(f"Training shapes (X ={X.shape!r}, y= {y.shape!r})"
                   f"Test set shapes ((X ={self.__X.shape!r},"
                   f" y= {self.__y.shape!r}")
            
        df0= self.data.copy()
        df1= df0.copy()
        _check_existingtarget_ =False 
        if self.categorizefeatures_props is not None: 
            cmsg=''
            ix_feat= [t[i] for  i, t in  enumerate(
                      self.categorizefeatures_props)]
            
            _check_existingtarget_= self.tname in ix_feat
            
        if _check_existingtarget_: 
            target_ix= ix_feat.index(self.tname) 
    
            try: 
                ycat = self.categorizefeatures_props[
                    target_ix][1][0]
                ytext= self.categorizefeatures_props[
                    target_ix][1][1]
            except : 
                ycat, ytext =None 
                
                self._logging.error('Unable to find labelcategories values. '
                                    'Could not convert y to text attributes.')
                warnings.warn('Unable to find labelcategories values. '
                                    'Could not convert y to text attributes')
                if self.verbose > 1:
                    cmsg=''.join(['Unable to find label categories. ',
                           'Could not convert to text attributes.'])
            else: 
                
                cmsg ="".join([
                    "\ It seems y label need to be categorized. ",
                    f"The given y categories are `{np.unique(ycat)}`",
                    f" and should be converted to text values = {ytext}"])
       
                df0[self.tname]= categorize_flow(
                target= df0[self.tname], 
                flow_values =ycat, classes=ytext)
                
                cmsg +='\ Conversion successfully done!'
    
                df1= df0.copy()
                df1[self.tname]= df1[self.tname].astype(
                    'category').cat.codes
                    
        self._df0= df0
        self._df1=df1            
                    
        if self.verbose >1 : pprint(cmsg)
        if self.verbose >3 :pprint(smsg)
   
        return X,  y, X_, y_
                

    @property 
    def X_(self): 
        """ keep the stratified testset X"""
        return self.__X
    @property 
    def y_(self): 
        """ keep the stratified label y"""
        return self.__y 
    
def default_pipeline(X,  num_attributes, cat_attributes, y=None,
                    label_encoding='LabelEncoder', **kws): 
    """ Default pipeline use for preprocessing the`Bagoue` dataset
    
    The pipeline can be improved  to achieve a good results. 
    
    Parameters
    ---------
     X: ndarray, pd.DataFrame 
         X or dataframe X 
    y: array_like, 
        ylabel or target 
    num_attributes:list 
        Numerical attributes 
    cat_attributes: list 
        categorical attributes 
    lableEncodage: str 
        Type of encoding used to encode the label 
        Default is ``labelEncoder` but can be ``LabelBinarizer``
        
    Returns
    -------
        - `mum_pipeline`: Pipeline to process numerical features 
        -`cat_pipeline`: pipeline to process categorical features.
        - `full_pipeline`: Full pipeline as the union of two pipelines 
        -`y`: ylabel encoded if not None.
    """
    
    missing_values = kws.pop('missing_values', np.nan)
    strategy = kws.pop('strategy', 'median')
    sparse_output = kws.pop('sparse_output', True)
    pca=kws.pop('pca', False)
    
    if y is not None: 
        if label_encoding =='LabelEncoder': 
            encodage_Objy =LabelEncoder()
        elif label_encoding =='LabelBinarizer':
            encodage_Objy =LabelBinarizer(sparse_output=sparse_output)
            
        y= encodage_Objy.fit_transform(y)

    num_pipeline =Pipeline([
        ('selectorObj', DataFrameSelector(attribute_names= num_attributes)),
        ('imputerObj',SimpleImputer(missing_values=missing_values , 
                                    strategy=strategy)),                
        ('scalerObj', StandardScaler()), 
        ])
        
    if not pca: 
        encode_ =  ('OneHotEncoder',OneHotEncoder())
        
    if pca : 
        encode_=  ('OrdinalEncoder',OrdinalEncoder())
        
    cat_pipeline = Pipeline([
        ('selectorObj', DataFrameSelector(attribute_names= cat_attributes)),
        encode_
        ])
    
    full_pipeline =FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline), 
        ('cat_pipeline', cat_pipeline)
        ])
    
    return num_pipeline, cat_pipeline, full_pipeline, y   

def default_preparation(
        X: NDArray| DataFrame, 
        imputer_strategy: Optional[str] =None,
        missing_values : float =np.nan,
        num_indexes: List[int] =None, 
        cat_indexes: List[int] =None, 
        scaler: Optional[str] =None ,
        encode_cat_features: bool = True, 
        columns:List[str] =None, 
        )-> NDArray: 
    
    """ Automate the data preparation to be ready for PCA analyses  

    Data preparation consist to imput missing values, scales the numerical 
    features and encoded the categorial features. 
    
    Parameters 
    ----------
    X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        Training set; Denotes data that is observed at training and 
        prediction time, used as independent variables in learning. 
        When a matrix, each sample may be represented by a feature vector, 
        or a vector of precomputed (dis)similarity with each training 
        sample. :code:`X` may also not be a matrix, and may require a 
        feature extractor or a pairwise metric to turn it into one  before 
        learning a model.
        
    imputer_strategy: str, default ='most_frequent' 
        Strategy proposed to replace the missing  values. Can be ``mean`` or
        ``median`` or ``most_frequent``. 
        Be aware , it `mean` or `median` are given, be sure that the data 
        are not composed of categorial fatures. 
        
    missing_values: float
        Value to replace the missing value in `X` ndarray or dataframe. 
        Default is ``np.nan`
    num_indexes: 
        list of indexes to select the numerical data if categorical data  
        columns exist in  `X` ndarray. 
        
    cat_indexes: 
        list of indexes to select the categorical data if numerical data 
        columns exists in  `X` ndarray. 
        
    scaler: str, default, is
        type of feature scaling applied on numerical features. 
        Can be ``MinMaxScaler``. Default is ``StandardScaler``
        
    encode_cat_features: bool
        Encode categorical data or text attributes. 
        Default is :class:`sklearn.preprocessing.OrdinalEncoder`. 
        
    columns: list, Optional, 
        list of columns to compose a dataframe if `X` is given as 
        an `NDAarray`.
        
    Returns
    --------
    X: NDArray | Dataframe
    
    Notes
    -----
    `num_indexes` and `cat_indexes` are mainly used when type of data `x` is 
    np.ndarray(m, nf) where `m` is number of instances or examples and 
    `nf` if number of attributes or features. `selector_` is used  for 
    dataframe preprocessing.
    
    """
    
    imputer_strategy = imputer_strategy or 'most_frequent'
    scaler = scaler or 'StandardScaler'
    
    sc=copy.deepcopy(scaler)
    scaler = str(scaler).lower().strip() 
    if scaler.find ('std')>=0 or scaler.find ('stand')>=0: 
        scaler ='std'
    if scaler .find('minmax')>=0: scaler ='minmax'
    if scaler not in ('std', 'minmax'): 
        raise ValueError(
            f"Expect 'MinMaxScaler' or 'StandardScaler' not {sc!r}")
        
    #regex = re.compile (r"[''|NAN|np.nan", re.IGNORECASE )
    is_frame =False 
    if isinstance(X, pd.DataFrame): 
        is_frame= True 
        
    elif isinstance(X, np.ndarray): 
        # if cat_indexes are given whereas num_indexes is None
        # considere the remain indexes are cat_index and vice-versa 
        if cat_indexes is not None: 
            num_indexes = list(set([i for i in range(X.shape[1])]
                                   ).difference( set(cat_indexes)))
               
        if num_indexes is not None: 
             cat_indexes = list(set([i for i in range(X.shape[1])]
                                    ).difference( set(num_indexes)))
               
        if num_indexes is not None: 
            X_num = np.float(X[:, num_indexes])
            
        if cat_indexes is not None: 
            X_cat =X[:, cat_indexes]
        
    # --> processing numerical features 
    numcols =list()
    if is_frame:
        c = X.columns 
        # replace empty string by Nan 
        X= X.replace(r'^\s*$', np.nan, regex=True) 
    # replace nan by np.nan by the most frequent no matter 
    # the data contain categorical or numerical. 
    # if stragey is other than most_frequent, be sure that all 
    # the data are numerical data. 
    imputer_obj = SimpleImputer(missing_values=missing_values,
                                strategy=imputer_strategy)
    X =imputer_obj.fit_transform(X)
    if is_frame: 
        # reconvert data to frame
        # and convert to numerical dtypes 
        X = to_numeric_dtypes(X, columns= c )
        # select feature before triggering 
        # the preprocessing 
        #-> coerce to True , return dataframe if columns is None 
        X = selectfeatures(X, features= columns, coerce =True ) 
        X_num = selectfeatures(X, include= 'number')
        X_cat = selectfeatures(X, exclude='number')
        catcols = list(X_cat.columns) 
        numcols = list(X_num.columns)
        
        if encode_cat_features: 
             encodeObj= OrdinalEncoder()
             X_cat= encodeObj.fit_transform(X_cat)
    # scale the dataset 
    if scaler=='std': 
        scalerObj =StandardScaler()
        X_num= scalerObj .fit_transform(X_num)
        
    if scaler=='minmax':
        scalerObj =MinMaxScaler()
        X_num= scalerObj .fit_transform(X_num)

    X= np.c_[X_num, X_cat]

    if is_frame: 
        X= pd.DataFrame (X, columns = numcols + catcols)
        
    # for consistency replace all np.nan by median values 
     # replace nan by np.nan
    impObj = SimpleImputer(missing_values=missing_values,
                           strategy=imputer_strategy)
    X =impObj.fit_transform(X )
    
    if is_frame: 
        # return dataframe like it was 
        X= pd.DataFrame (X, columns = numcols + catcols)
        
    return X 
    
def base_transform(
    X: NDArray | DataFrame, 
    n_components: float|int=0.95,
    attr_names: List[str] =None,
    attr_indexes:List[int]= None, 
    operator:str= None, 
    view: bool =False,
    **kws 
    )-> NDArray: 

    if (attr_names  and attr_indexes) is None: 
        warnings.warn(
            "NoneType can not experienced attributes combinaisons."
            " Attributes indexes for combinaison must be supplied.")
          
    if (attr_names or attr_indexes ) : 
        operator = operator or '/'
        cObj = CombinedAttributesAdder(attribute_names=attr_names, 
                                   attribute_indexes=attr_indexes, 
                                   operator=operator )
        X=cObj.fit_transform(X) 
        X = pd.DataFrame (X, columns = cObj.attribute_names_ )
    # reduce dimension of the datasets 
    
    X = default_preparation(X=X, **kws)
    pca =PCA(n_components=n_components)
    pca.fit_transform(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)

    if view: 
        from watex.view import viewtemplate 
        viewtemplate (cumsum, xlabel ='Dimensions', 
                      ylabel='Explained variance', 
                      label='Explained variance curve'
                      )
        
    return X 

base_transform.__doc__="""\
Tranformed X using PCA and plot variance ratio by experiencing 
the attributes combinaisons. 

Create a new attributes using features index or litteral string operator.
and prepared data for `PCA` variance plot.

Parameters 
-----------
X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
    Training set; Denotes data that is observed at training and 
    prediction time, used as independent variables in learning. 
    When a matrix, each sample may be represented by a feature vector, 
    or a vector of precomputed (dis)similarity with each training 
    sample. :code:`X` may also not be a matrix, and may require a 
    feature extractor or a pairwise metric to turn it into one  before 
    learning a model.
    
n_components: float oR int
    Number of dimension to preserve. If`n_components` 
    is ranged between float 0. to 1., it indicated the number of 
    variance ratio to preserve. If ``None`` as default value 
    the number of variance to preserve is ``95%``.

attr_names: list of str , optional
    List of features for combinaison. Decide to combine new feature
    values by from `operator` parameters. By default, the combinaison it 
    is ratio of the given attribute/numerical features. For instance, 
    ``attribute_names=['lwi', 'ohmS']`` will divide the feature 'lwi' by 
    'ohmS'.
                
attr_indexes : list of int,
    index of each feature/feature for experience combinaison. User 
    warning should raise if any index does match the dataframe of array 
    columns.
        
operator: str, default ='/' 
    Type of operation to perform when combining features. Can be 
    ['/', '+', '-', '*', '%']    
    
Returns 
-------
    X: n_darray, or pd.dataframe
    New array of dataframe with new attributes combined. 
    
Examples
--------
>>> from from watex.view.mlplot import MLPlots
>>> from watex.datasets import fetch_data 
>>> from watex.analysis import pcaVarianceRatio
>>> plot_kws = {'lc':(.9,0.,.8),
        'lw' :3.,           # line width 
        'font_size':7.,
        'show_grid' :True,        # visualize grid 
       'galpha' :0.2,              # grid alpha 
       'glw':.5,                   # grid line width 
       'gwhich' :'major',          # minor ticks
        # 'fs' :3.,                 # coeff to manage font_size 
        }
>>> X, _ = fetch_data ('Bagoue data analysis')
>>> mlObj =MLPlots(**plot_kws)
>>> pcaVarianceRatio(mlObj,X, plot_var_ratio=True)

"""