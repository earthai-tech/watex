# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Sep 16 11:31:38 2021
# This module is a set of processing tips 
# released under a MIT- licence.
#Created on Wed Sep 22 15:04:52 2021
#@author: @Daniel03

import os
import warnings
import numpy as np
import pandas as pd 
from typing import TypeVar 

from sklearn.preprocessing import (OrdinalEncoder, 
                                   OneHotEncoder,
                                   LabelBinarizer,
                                   LabelEncoder) 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA 

from ..analysis.__init__ import PD_READ_FEATURES
from ..utils.decorator import docstring
from ..utils._watexlog import watexlog
from ..utils.transformers import (DataFrameSelector,
                                  CombinedAttributesAdder, 
                                  CategorizeFeatures,
                                  StratifiedWithCategoryAdder)
import watex.viewer.hints as HI 
import watex.utils.ml_utils as MLU
import watex.utils.exceptions as Wex

T= TypeVar('T', list, tuple)
_logger = watexlog().get_watex_logger(__name__)

DEFAULT_PATH = 'data/geo_fdata'
DEFAULT_FILENAME = 'main.bagciv.data.csv'


class PrepareDATA (object): 
    """ Prepare the Data for preparation steps.
    
    Apply method `fit` and `transform` or `fit_transform`.
    
    Parameters
    ----------
    data: str, Path_Likeobj or Dataframe obj 
        Path to data file or pandas dataframe 
        
    target: str 
        Label name.In supervised learning target is considered as `y`.
    
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
        
    add_attributes: bool, 
        Experience the combinaison <numerical> attributes. 
        
    attributes_ix: list 
        List of attributes indexes to combines. For instance:: 
            
            attributes_ix = [(1, 0), (4,3)] 
    
        The operator by default is `division` . For more details, please 
        refer to :doc:`~watex.utils.transformers.CombinedAttributesAdder`
    imputer_strategy: str 
        Type of strategy to replace the missing values. Refer to 
        :class:`sklearn.impute.SimpleImputer`. Default is ``median``.
        
    missing_values : float
        The value to be replaced.  Default is ``np.nan`` values.
        
    pipeline: callable
        Pipeline to prepare the dataset. Default is :func:`defaultPipeline`.
        
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

     The second steps consist of features selection, features engineering, 
     encoding and data scaling using the pipeline via a parameters `pipeline`. 
     If None pipeline is given, the default pipline is triggered.The features
     engineering’s consist to aggregate features with experiencing combinations
     of attributes into promising new features using the params `attributes_ix`
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
     --------
     
     ../datasets/config.py
     
    """
    
    def __init__(self,
                 data=None, 
                 target =None, 
                 return_all=True,
                 drop_features=None, 
                 categorizefeature_props =None,
                 add_attributes =True, 
                 attributes_ix =None, 
                 imputer_strategy ='median', 
                 missing_values =np.nan, 
                 pipeline =None,
                 test_size=0.2, 
                 random_state=42,
                 **kwargs):
        self._logging = watexlog.get_watex_logger(self.__class__.__name__)
        
        self._data =data 
        self._target= target 
        self.drop_features =drop_features 
        self.categorizefeatures_props = categorizefeature_props 
        self.return_train = return_all
        self.add_attributes = add_attributes 
        self.attributes_ix = attributes_ix
        self.imputer_strategy = imputer_strategy 
        self.missing_values = missing_values
        self.pipeline =pipeline 
        self.test_size =test_size 
        self.random_state =random_state 

        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])
            
        if self._data is not None : 
            self.data = self._data 
            
            self.X, self.y,*__ = self.stratifyFolds(self.data )
            
    @property 
    def data (self): 
        return self._data 
    @data.setter 
    def data (self, _data): 
        """ Control whether a pd.datafrme of filetype."""
        
        if isinstance(_data , str): 
            self._data = self.getDataFrameFromFile(_data)

    def fit (self, X, y=None): 
        """ Preparing steps 
        
        Parameters
        ---------
        X: ndarray, pd.DataFrame 
             X or dataframe X 
        y: array_like, 
            ylabel or target values 
        """
        self._logging.info('Start the default preparing steps including'
                           ' Data cleaning,features combinaisons ... using '
                           ' the `fit` method!')
        if isinstance(y, pd.Series): 
            y =y.values 
        # convert to pandas X with features names 
        X =pd.DataFrame (X, columns = self.attribute_names_)
        # drop features if features are useles
        if self.drop_features is not None : 
            if isinstance(self.drop_features, str): 
                self.drop_features=[ self.drop_features]
                
            self._logging.info('Dropping useless features {0}'.format(
                HI.format_generic_obj(self.drop_features)).format(
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
  
            if self.target in feat_to_: 
                self._logging.info('Change the numerical target `%s` into'
                                  ' categorical features.'%self.target)
                
                feat_ix = feat_to_.index(self.target)
                get_targetprops =[self.categorizefeatures_props[feat_ix]]
                
                cObj = CategorizeFeatures(
                    num_columns_properties=get_targetprops)
                y = cObj.fit_transform(X=y)
                
                self.categorizefeatures_props.pop(feat_ix)
                feat_to_.pop(feat_ix)
                
            if len(self.categorizefeatures_props) !=0 : 
                
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
            if self.attributes_ix is  None: 
                warnings.warn( 'Attributes indexes|names are None.'
                              ' Set attributes indexes or names to experience'
                              ' the attributes combinaisons.'
                              )
            elif self.attributes_ix is  not None:
                try:
                    self._logging.info('Experiencing combinaisons attributes'
                                      ' {0}.'.format(HI.format_generic_obj(
                                          self.attributes_ix)).format(
                                              *self.attributes_ix))
                except : 
                    self._logging.info('Experiencing combinaisons attributes'
                                      ' {0}.'.format(self.attributes_ix))
                                          
                cObj = CombinedAttributesAdder(
                    add_attributes=self.add_attributes, 
                    attributes_ix=self.attributes_ix )
                X=cObj.fit_transform(X)  # return numpy array        
                    
               # get the attributes and create new pdFrame with new attributes 
                self.attribute_names_  = list(self.attribute_names_) +\
                    cObj.attribute_names_
            
        
            if isinstance(X, np.ndarray): 
                X= pd.DataFrame(X, columns= self.attribute_names_)
                
        #convert final data the default numerical features.
        self.cat_attributes_, self.num_attributes_=list(), list()
        for fname in self.attribute_names_: 
            self._logging.info('Convert dataframe `X` to numeric !')
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
        self._logging.info('Transform the data using the `transform` method.'
                           'Applying the pipeline.')
        
        for name, value in zip(['sparse_output','labelEncodage'],
                                     [True,'LabelEncoder']): 
                                    
                if not hasattr(self, name) :
                    setattr(self, name, value)
                    
        pkws={'num_attributes': self.num_attributes_, 
                'cat_attributes': self.cat_attributes_, 
                'missing_values' : self.missing_values, 
                'strategy' : self.imputer_strategy, 
                'sparse_output': self.sparse_output,
                'labelEncodage': self.labelEncodage
                }
        
        if X is not None: 
            self.X0 = X 
        if y is not None: 
            self.y0=y 
        
        if self.pipeline is None :
            _, _, self.pipeline, self.y_prepared= defaultPipeline(
                X= self.X0, 
                y = self.y0,
                **pkws
                )
        # keep another type of encodage using the Ordinal encoder           
        _, _, self.pca_pipeline,_ = defaultPipeline(
            X= self.X0, 
            pca=True,
            **pkws
            )
            
        if on_testset : 
            self._logging.info(
                '`You are running the Test Set, so pipeline `transform` method'
                ' is applied not `fit_transform`.')
            self.X_prepared = self.pipeline.transform(self.X0)
            
        if not on_testset: 
            self._logging.info('Train Set is running so `fit_transform` '
                               'method is used.')
 
            self.X_prepared = self.pipeline.fit_transform(self.X0)
            # --> pipeline 
            self._X = self.pca_pipeline.fit_transform(self.X0)
            names = self.num_attributes_ + self.cat_attributes_
            
            self._Xpd = pd.DataFrame(self._X, 
                                     columns =names )
        return self.X_prepared, self.y_prepared 
    

    def fit_transform (self, X=None, y =None, on_testset=False ): 
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
        Returns
        -------
        - X_prepared. Data prepared after transformation 
        -y-prepared. label prepared after transformation.
        """
        self.fit(X, y)
    
        self.transform(on_testset =on_testset)
        
        return self.X_prepared, self.y_prepared 
    
    @property 
    def target (self): 
        return self._target
    @target.setter
    def target(self, label): 
        """ Check whether the target is among the data frane columns."""
        if not label in self.data.columns: 
            raise TypeError('No target {self._target !} found in dataframe'
                            'columns `{self.data.columns}`')
        self._target =label 
        
    def stratifyFolds(self, data): 
        """ Stratified the dataset and return the trainset. Get more details 
        in :doc:`watex.utils.transformers.StratifiedWithCategoryAdder`."""
        sObj= StratifiedWithCategoryAdder(base_num_feature=self.target, 
                                          test_size= self.test_size, 
                                          random_state = self.random_state,
                                        return_train=self.return_train)
        # return data with labels stratified
        _X , __X = sObj.fit_transform(X=data)
        
        # make a copy to keep
        X, y = _X.drop(self.target, axis =1).copy(), _X[self.target].copy() 
        X_, y_ = __X.drop(self.target, axis =1).copy(), __X[self.target].copy()
    
        self.__X= __X.drop(self.target, axis =1).copy()
        self.__y= __X[self.target].copy()
        
        self.attribute_names_= X.columns 
        
        return X,  y, X_, y_
                
            
    @staticmethod 
    def getDataFrameFromFile(data_fn =None, read_default_file =False, 
                             ):
        """ Get the dataframe from file and convert to pandas dataframe.
        :param data_fn: str or Path_Like obj 
            Path to data file. 
        :param read_default_file: bool 
            Read the default file. Default file is the dataset to testing 
            the whole scripts.
            
        :return: 
            object data.
        """
        key_read = list(PD_READ_FEATURES.keys())
        
        if read_default_file: 
            
            return  MLU.load_data(DEFAULT_PATH)
        
        if not os.path.isfile(data_fn): 
            raise Wex.WATexError_file_handling(
                'None file detected. NoneType can not be read.'
                ' Could only read {0}'.format(
                    HI.format_generic_obj(key_read)).format(
                        *key_read)
                )
        
        _fname, _fex =os.path.splitext(data_fn)
        if _fex not in PD_READ_FEATURES.keys(): 
            raise Wex.WATexError_file_handling(
                'Unrecognized file Type. Unable to read the file {0}.'
                ' Only types {1} could be read. Please Check your right file.'
                .format(data_fn,HI.format_generic_obj(key_read).format(
                    *key_read) ))

        return PD_READ_FEATURES[_fex](data_fn)
        
    @property 
    def X_(self): 
        """ keep the stratified testset X"""
        return self.__X
    @property 
    def y_(self): 
        """ keep the stratified label y"""
        return self.__y 
    
def defaultPipeline(X,  num_attributes, cat_attributes, y=None,
                    labelEncodage='LabelEncoder', **kws): 
    """ Default pipeline use for preprocessing the`Bagoue` dataset  used'
    for implement of this workflow. 
    
    The pipleine can be improved  to achieve a good results. 
    
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
        if labelEncodage =='LabelEncoder': 
            encodage_Objy =LabelEncoder()
        elif labelEncodage =='LabelBinarizer':
            encodage_Objy =LabelBinarizer(sparse_output=sparse_output)
            
        y= encodage_Objy.fit_transform(y)

    num_pipeline =Pipeline([
        ('selectorObj', DataFrameSelector(attribute_names= num_attributes)),
        ('imputerObj',SimpleImputer(missing_values=missing_values , 
                                    strategy=strategy)),                
        ('scalerObj', StandardScaler()), 
        ])
        
    if not pca: 
        encode__ =  ('OneHotEncoder',OneHotEncoder())
        
    if pca : 
        encode__=  ('OrdinalEncoder',OrdinalEncoder())
        
    cat_pipeline = Pipeline([
        ('selectorObj', DataFrameSelector(attribute_names= cat_attributes)),
        encode__
        ])
    
    full_pipeline =FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline), 
        ('cat_pipeline', cat_pipeline)
        ])
    
    return num_pipeline, cat_pipeline, full_pipeline, y       
     
        
def prepareDataForPCA(X, 
                      imputer_strategy:str ='median',
                      missing_values :float=np.nan,
                      num_indexes:T =None, 
                      cat_indexes:T=None, 
                      selector__:bool=True, 
                      scaler__:str='StandardScaler',
                      encode_categorical_features__:bool=True, 
                      pd_column_list:T =None): 
    """ Preparing the data.

    Data preparation consist to imput missing values, scales the numerical 
    features and encoded the categorial features. 
    
    Parameters 
    ----------
    imputer_strategy: str 
        strategy propose to replace the missing  values. Can be ``mean`` or
        another numbers. Default is ``median``
    missing_values: float
        Value to replace the missing value in `X` ndarray or dataframe. 
        Default is ``np.nan`
    num_indexes: 
        list of indexes to select the numerical data if numerical data  
        exist in  `X` ndarray. 
    cat_indexes: 
        list of indexes to select the categorical data if categorical 
        datta exists in  `X` ndarray. 
    selector__: bool, str (True|'auto') 
        Trigger the numerical or categorical features in Dataframe selection
        and apply all the for each category the convenient data transformation.
        Default is ``True``
    scaler__: str
        type of feature scaling applied on numerical features. 
        Can be ``MinMaxScaler``. Default is ``StandardScaler``
        
    encode_categorical_features__: bool
        Encode categorical data or text attributes. 
        Default is :class:`sklearn.preprocessing.OrdinalEncoder` . 
        
     Notes
     -----
     `num_indexes` and cat_indexes` is mainly used when type of data `x` is 
     np.ndarray(m, nf) where `m` is number of instances or examples and 
     `nf` if number of attributes or features. `selector__` is used  for 
     dataframe preprocessing.
    """
    
    if isinstance(X, pd.DataFrame): 
        type__= 'pd.df'
        
    elif isinstance(X, np.ndarray): 
        
        if num_indexes =='auto' and cat_indexes is not None: 
            num_indexes = list(set([i for i in range(X.shape[1])]
                                   ).difference( set(cat_indexes)))
               
        if cat_indexes =='auto' and num_indexes is not None: 
             cat_indexes = list(set([i for i in range(X.shape[1])]
                                    ).difference( set(num_indexes)))
               
        
        if num_indexes is not None: 
            X_num = np.float(X[:, num_indexes])
            
        if cat_indexes is not None: 
            X_cat =X[:, cat_indexes]
            
        type__='np.nd'
        
    # --> processing numerical features 
    num_col_ =list()
    
    if type__=='pd.df':
        
        if selector__ or selector__ =='auto':
            # --> process the categorical features
            if pd_column_list is not None :
                 X_columns = pd_column_list
            else :X_columns = X.columns 
            # try to convert pandas to numeric
            
            for serie in X_columns : 
                try:
                    X= X.astype(
                            {serie:np.float64})
                except:
                    continue
                # else: # pd.to_numeric(X[serie], errors ='ignore')
            try: 
                num_col_ = X.select_dtypes('number').columns.tolist()
                X_num = X[num_col_]
 
            except: 
                datafameObj =  DataFrameSelector(select_type='num')
                X_num = datafameObj.fit_transform(X)
                num_col_ =datafameObj.attribute_names 
 
            cat_col_ = list(set(X_columns).difference(set(num_col_)))
        
            X_cat = X[cat_col_]
            
        if encode_categorical_features__: 
             encodeObj= OrdinalEncoder()
             X_cat= encodeObj.fit_transform(X_cat)
   
    # replace nan by np.nan
    if imputer_strategy  is not None and not imputer_strategy:
        imputer_obj = SimpleImputer(missing_values=missing_values,
                                    strategy=imputer_strategy)
        X_num =imputer_obj.fit_transform(X_num )
        
        
    # scale the dataset 
    if scaler__=='StandardScaler' or scaler__.lower().find('stand')>=0: 
        scalerObj =StandardScaler()
        X_num= scalerObj .fit_transform(X_num)
        
    if scaler__=='MinMaxScaler' or scaler__.lower().find('min')>=0 or\
        scaler__.lower().find('max')>=0:
        scalerObj =MinMaxScaler()
        X_num= scalerObj .fit_transform(X_num)

    
    X= np.c_[X_num, X_cat]

    if type__=='pd.df': 
        X= pd.DataFrame (X, columns =num_col_ + cat_col_)
        
    # for consistency replace all np.nan by median values 
     # replace nan by np.nan
    impObj = SimpleImputer(missing_values=missing_values,
                           strategy=imputer_strategy)
    X =impObj.fit_transform(X )
      
    return X 

@docstring(prepareDataForPCA, start ='Parameters', end='Notes')
def pcaVarianceRatio(self,
                        X, 
                        n_components:float=0.95,
                        add_attributes:bool=False,
                        attributes_ix:T= None, 
                        plot_var_ratio:bool =False,
                        **ppca_kws ): 
    
    if add_attributes and attributes_ix is None: 
        warnings.warn( 'Attributes indexes|names are None. Set attributes '
                      'indexes or names to experience the attributes combinaisons'
                      )
    col=list()
    if isinstance(X, pd.DataFrame): 
        col = list(X.columns)
        
    cObj = CombinedAttributesAdder(add_attributes=add_attributes, 
                               attributes_ix=attributes_ix)
    X=cObj.fit_transform(X)  # return numpy array 


    # get the attributes and create new pdFrame
    if len(col)!=0:
        col +=cObj.attribute_names_
        X= pd.DataFrame(X, columns=col )
    if len(col) ==0 :col =None 
    # --> prepare the data 
    X= prepareDataForPCA(X=X, pd_column_list =col, **ppca_kws)

    pca =PCA(n_components=n_components)
    pca.fit_transform(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)

    if plot_var_ratio: 
        import matplotlib.pyplot as plt 
        # create figure obj 
        fig = plt.figure(figsize = self.fig_size)
        ax = fig.add_subplot(1,1,1)
        ax.plot(cumsum,
                color= self.lc, 
                linewidth = self.lw,
                linestyle = self.ls , 
                label ='pca explained variance vs Ndimension')
        
        if self.xlabel is None: 
            self.xlabel ='Ndimensions'
        if self.ylabel is None: 
            self.ylabel ='Explained Variance'
    
        ax.set_xlabel( self.xlabel,
                      fontsize= .5 * self.font_size * self.fs )
        ax.set_ylabel (self.ylabel,
                       fontsize= .5 * self.font_size * self.fs)
        ax.tick_params(axis='both', 
                       labelsize=.5 * self.font_size * self.fs)
        
        if self.show_grid is True : 
            if self.gwhich =='minor': 
                  ax.minorticks_on() 
            ax.grid(self.show_grid,
                    axis=self.gaxis,
                    which = self.gwhich, 
                    color = self.gc,
                    linestyle=self.gls,
                    linewidth=self.glw, 
                    alpha = self.galpha
                    )
              
            if len(self.leg_kws) ==0 or 'loc' not in self.leg_kws.keys():
                 self.leg_kws['loc']='upper left'
            
            ax.legend(**self.leg_kws)
            
    
            plt.show()
            
            if self.savefig is not None :
                plt.savefig(self.savefig,
                            dpi=self.fig_dpi,
                            orientation =self.fig_orientation)

        
    return X 

pcaVarianceRatio.__doc__="""\
Plot variance ratio by experiencing  the attributes combinaisons. 

Create a new attributes using features index or litteral string operator.
and prepared data for `PCA` variance plot.

Parameters 
-----------
    X: pd.DataFrame of ndarray 
        dataset composed of initial features 
    n_components: float of int
        Number of dimension to preserve. If`n_components` 
        is ranged between float 0. to 1., it indicated the number of 
        variance ratio to preserve. If ``None`` as default value 
        the number of variance to preserve is ``95%``.

    add_attributes: bool,
        Decide to add new features values by combining 
        numerical features operation. By default ease to 
        divided two numerical features.
                    
    attributes_ix : str or list of int,
        Divide two features from string litteral operation betwen or 
        list of features indexes. 
    
Returns 
-------
    X: n_darry, or pd.dataframe
    new array of dataframe with new attributes. 
    
Examples
--------

    >>> from from watex.viewer.mlplot import MLPlots
    >>> from watex.datasets.data_preparing import bagoue_train_set
    >>> plot_kws = {'lc':(.9,0.,.8),
            'lw' :3.,           # line width 
            'font_size':7.,
            'show_grid' :True,        # visualize grid 
           'galpha' :0.2,              # grid alpha 
           'glw':.5,                   # grid line width 
           'gwhich' :'major',          # minor ticks
            # 'fs' :3.,                 # coeff to manage font_size 
            }
    >>> mlObj =MLPlots(**plot_kws)
    >>> pcaVarianceRatio(mlObj,
    ...                     bagoue_train_set,
    ...                     plot_var_ratio=True,
    ...                     add_attributes=True)
"""
# import inspect 
# func_signature = inspect.signature(pcaVarianceRatio)
# PARAMS_VALUES = {k: v.default
#             for k, v in func_signature.parameters.items()
#             if v.default is not inspect.Parameter.empty
#             }
# @docstring(prepareDataForPCA, start ='Parameters', end='Notes')
# def pcaVarianceRatio():
#     ...
    

    
    
    
    
    
    
    
    
    
    
    





