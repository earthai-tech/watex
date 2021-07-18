# Copyright (c) 2021 Kouadio K. Laurent, Wed Jul 14 20:00:26 2021
# This module is part of the WATex viewer package, which is released under a
# MIT- licence.

import os 
import warnings 
import numpy as np 
import pandas as pd 

from typing import TypeVar, Generic, Iterable , Callable


T= TypeVar('T', float, int)

from watex.processing.__init__ import (DecisionTreeClassifier, SGDClassifier,
                                       KNeighborsClassifier, SVC  )
from watex.processing.__init__ import _HAS_ENSEMBLE_ 

from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import RobustScaler 
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures 

from sklearn.compose import make_column_transformer 
from sklearn.compose import make_column_selector 


from sklearn.model_selection import validation_curve, cross_val_score
from sklearn.model_selection import RandomizedSearchCV,  GridSearchCV
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GroupKFold 
from sklearn.model_selection import learning_curve 

from sklearn.metrics import confusion_matrix, f1_score, classification_report 

from sklearn.feature_selection import SelectKBest, f_classif 

from watex.analysis.features import sl_analysis 
from watex.viewer.plot import hints 

import  watex.utils.exceptions as Wex 
from watex.utils._watexlog import watexlog 

_logger =watexlog().get_watex_logger(__name__)


d_estimators__={'dtc':DecisionTreeClassifier, 
                'svc':SVC, 
                'sgd':SGDClassifier, 
                'knn':KNeighborsClassifier 
                 }

if _HAS_ENSEMBLE_ :
    from watex.processing.__init__ import skl_ensemble__
    for es_, esf_ in zip(['rdf', 'ada', 'vtc', 'bag','stc'], skl_ensemble__): 
        d_estimators__[es_]=esf_ 


class Preprocessing : 
    """
    Preprocessing class deal with supervised learning `sl` . 

    """
    _estim_prefix = ['dtc',
                     'svc',
                     'sgd',
                     'knn', 
                     'rdf', 
                     'ada', 
                     'vtc',
                     'bag',
                     'stc']

    def __init__(self, data_fn =None , df=None , **kwargs)->None : 
        self._logging = watexlog().get_watex_logger(self.__class__.__name__)
        
        self._data_fn = data_fn 
        self._df =df   
        
        self.categorial_features =kwargs.pop('categorial_features', None)
        self.numerical_features =kwargs.pop('numerical_features', None)
        
        self.target =kwargs.pop('target', 'flow')
        self._drop_features = kwargs.pop('drop_features', ['lwi'])
        self.random_state = kwargs.pop('random_state', 0)
        
        self._df_cache =None 
        self._features = None 
        
        self.y = None 
        self.X = None 
        self.X_train =None 
        self.X_test = None 
        self.y_train =None 
        self.y_test =None 
        

        self._num_column_selector = make_column_selector(
            dtype_include=np.number)
        self._cat_colum_selector =make_column_selector(
            dtype_exclude=np.number)
        self._features_engineering =PolynomialFeatures(
            10, include_bias=False) 
        self._selectors= SelectKBest(f_classif, k=4) 
        self._scalers =RobustScaler()
        self._encodages =OneHotEncoder()
        
        self._select_estimator_ =None 
        
        
        for key in kwargs.keys(): 
            setattr(self, key, kwargs[key])

        if self._data_fn is not None: 
            self.data_fn = self._data_fn 

        if self.df is not None : 
            self._read_and_encode_catFeatures()
            
    @property 
    def df (self): 
        """ Retrieve the pd.core.frame.DataFrame """
        return self._df 
    
    @property 
    def data_fn(self): 
        return self._data_fn 
    
    @data_fn.setter
    def data_fn (self, datafn): 
        """ Read the given data and create a pd.core.frame.DataFrame . 
        Call :class:~analysis.features.sl_analysis` and retrieve the best 
         information. """
        if datafn is not None : 
           self._data_fn = datafn 

        slObj= sl_analysis(data_fn=self._data_fn, set_index=True)
        self.df= slObj.df 
        
        slObj.df_cache =[] # initialize the cache 
        self._df_cache =slObj.df_cache  
        
    @df.setter 
    def df (self, dff): 
        """ Ressetting dataframe when comming from raw file. """
        if dff is not None : 
            self._df = dff
            
    @property 
    def df_cache(self): 
        """ keed the data on the trash"""
        return self._df_cache 
    
    @df_cache.setter 
    def df_cache (self, feature_name): 
        """ Keep the feature considered as a pd.Series  for other purposes."""
        
        if isinstance(feature_name, str) :
            feature_name=[feature_name]
        elif isinstance(feature_name, (dict,tuple)): 
            feature_name = list(feature_name)
            
        for fc in feature_name : 
            if fc == self.target : 
                fc = self.target +'_' # flow_
                placement = self.target 
            else : placement = fc 
                
            if fc in self._df_cache.columns : 
                self._logging.info(
                    f'Feature `{feature_name}` already exists on the cache.')
                warnings.warns( 
                    f'Feature `{feature_name}` already exists on the cache.')
            else:
                self._df_cache.insert(loc=len(self._df_cache.columns), 
                            column =fc, value = self.df[placement].to_numpy())
    
    @property 
    def features(self): 
        """ Collect the list of features"""
        return self._features 
    @features.setter 
    def features(self, feats): 
        """ Set the features once given"""
        
        if isinstance(feats, pd.core.frame.DataFrame ): 
            self._features = list(feats.columns ) 
        elif isinstance(feats , str): self._features= [self._features ]
        elif isinstance(feats, (dict,tuple)): 
            self._features = list(feats)
            
        
    def _read_and_encode_catFeatures(self, data_fn =None , df=None,
                                     features:Iterable[T]=None, 
                                     categorial_features:Iterable[T] =None,
                                     numerical_features:Iterable[T]=None, 
                                     **kws) -> None: 
        """ 
        Read the whole dataset and encode the categorial features  and 
        populate class attributes.
        
        :param df: Container `pd.DataFrame` of all features in the dataset.
        :param features:
            Iterable object ``list`` in the dataset. If `df` is given, don't need
            to set any others arguments. 
        :param categorial_features: 
            list of selected categorial features. Need to provide the whole 
            `features` to find the `numerical_values` 
        :param numerical_features: 
            list of selected `numerical_features`. If given, provides the 
            `features` of the whole dataframe to find the `categorial_features`
            
        :Note: Once the `features` argument is set, provide at least 
            `categorial_features` or `numerical_features` to find the one of 
            the targetted features. 
        
        
        """
        drop_features = kws.pop('drop_features', None)
        target =kws.pop('target', None)
        random_state = kws.pop('random_state', None )
        
        if target is not None : self.target =target 
        if random_state is not None : self.random_state = random_state 
        
        if drop_features is not None : 
            self._drop_features = drop_features 
            
        if data_fn is not None : 
            self.data_fn = data_fn
        
        if features is not None : self.features = features 
        
        if  df is not None : 
            self.df =df 
        if isinstance(self.df, pd.core.frame.DataFrame ): 
            self.categorial_features,self.numerical_features =\
                            find_categorial_and_numerical_features(df=self.df)
                            
        if categorial_features is not None : 
            self.categorial_features = categorial_features 
        if numerical_features is not None:
            self.numerical_features =numerical_features 
            
        if self.features is not None : 
            if self.categorial_features is not None : 
                self.categorial_features,self.numerical_features =\
                            find_categorial_and_numerical_features(
                            features =self.features, 
                            categorial_features=self.categorial_features)
  
            elif  self.numerical_features is not None : 
                self.categorial_features, self.numerical_features =\
                            find_categorial_and_numerical_features(
                            features =self.features, 
                             numerical_features= self.numerical_features)            
                            
        # make a copy to hold the raw dataframe                    
        self.X = self.df.copy(deep=True) 
        
        # encode all  categories 
        for catfeatures in self.categorial_features : 
            self.X[catfeatures] = self.X[catfeatures ].astype(
                'category').cat.codes 
        
        #droping unecessaries features 
        if isinstance(self._drop_features, str) :
            self._drop_features =[self._drop_features]
            
        elif isinstance(self._drop_features, (dict,tuple)): 
            self._drop_features = list(self._drop_features)
            
        if not self.target  in self._drop_features : 
            if isinstance(self.target, str ): 
                self._drop_features += [self.target] 
            elif isinstance(self.target, (dict,tuple)): 
                self._drop_features += list(self.target)
                
        # drop unecessaries collections and put on caches 
        self.df_cache = self._drop_features 
        self.X = self.X.drop(self._drop_features, axis =1 )
    
        self.y = self.df[self.target].astype('category').cat.codes 
        
        # splitted dataset 
        self.X_train , self.X_test, self.y_train, self.y_test =\
            train_test_split (self.X, self.y, 
                              random_state = self.random_state )
     
    def make_preprocessor(self, num_column_selector_:Callable[...,T] =None, 
                               cat_column_selector_:Callable[...,T] =None, 
                               features_engineering_:Callable[...,T] =None, 
                               selectors_ :Callable[...,T]=None , 
                               scalers_:Callable[...,T]=None, 
                               encodages_ :Callable [..., T]=None, 
                               **kwargs)-> Callable[..., T]:
        """
        Create a composite estimator based on multiple pipeline creation. 
        
        `make_preprocessor` arguments are only callable function of method or 
        preprocessor making.Different modules from the :mod:`sklearn` are 
        used for the preprocessor building like:: 
            
            - :meth:`sklearn.pipeline.make_pipeline ` for pipeline creation. 
            - :meth:`sklearn.preprocessing.OneHotEncoder` for categorial 
                `features` encoding. 
            - :meth:`sklearn.preprocessing.PolynomialFeatures` for features 
               engineering. 
            - :meth:`sklearn.preprocessing.RobustScaler` for data scaling 
            - :meth:`sklearn.compose.make_column_transformer` for data 
                transformation. 
            - :meth:`sklearn.compose.make_column_selector` for features 
                composing.
                
        :param num_column_selector_: Callable method from sckitlearn 
            Numerical column maker. Refer to  sklearn site for more details. 
            :ref:`<https://scikit-learn.org/stable/modules/classes.html>` 
            The default is:: 
                
                `make_column_selector(dtype_include=np.number)`
            
        :param cat_column_selector_`: 
            Callable method. Categorical column selector. The default is::
            
                `make_column_selector(dtype_exclude=np.number)`
 
        :param features_engineering_: 
            Callable argument using :mod:`sklearn.preprocessing` different 
            method. the default is::
                
                `PolynomialFeatures(10, include_bias=False)`
                
        :param selectors_: Selector callable argument including many test 
            methods like `f_classif` or Anova test.The default is: 
                `SelectKBest(f_classif, k=4),` 
           
        :param scalers_: Scaling data using many normalization or standardization 
            methodike. The default is  `RobustScaler()`. 
            
        :param kwargs: Other additionals keywords arguments in 
        `make_column_transformer()` and `make_pipeline()` methods. 
        
        
                      
        """
        preprocessor_kws = kwargs.pop('prepreocessor_kws', None)
        make_pipeline_kws =kwargs.pop('make_pipeline_kws', None)

        
        if num_column_selector_ is not None :
            self._num_column_selector = num_column_selector_
        if cat_column_selector_ is not None : 
            self._cat_colum_selector = cat_column_selector_ 
            
        if features_engineering_ is not None: 
            self._features_engineering = features_engineering_
        if selectors_ is not None : self._selectors_= selectors_ 
        if scalers_ is not None : self._scalers = scalers_ 
        if encodages_ is not None: self._encodages = encodages_ 
        
        
        numerical_features = self._num_column_selector
        categorical_features =self._cat_column_selector
        
        #create a pipeline 
        numerical_pipeline = make_pipeline(self._features_engineering,
                                           self._selectors , self._scalers,
                                           **make_pipeline_kws )
        categorical_pipeline= make_pipeline(self._encodages)
        
        self._preprocessor =make_column_transformer(
            (numerical_pipeline, numerical_features), 
            (categorical_pipeline, categorical_features), **preprocessor_kws)
        
        return self._preprocessor 
    
    def make_preprocessing_model(self, preprocessor: Callable[..., T]= None, 
                                 estimator_:Callable[..., T]=None,
                                 defaut_estimator ='svc', **kws)->T: 
        """
        Test your preprocessing model by providing an `sl` estimator. 
        
        If `estimator` is None, set the default estimator by the predix of 
        the estimator to test the fit of the preprocessing model. The prefix 
        of some defaults estimator are enumerated below:: 
            
            - 'dtc': For DecisionTreeClassifier 
            - 'svc': Support Vector Classifier 
            - 'sdg': SGDClassifier
            - 'knn': KNeighborsClassifier
            - 'rdf`: RandmForestClassifier 
            - 'ada': AdaBoostClassifier 
            - 'vtc': VotingClassifier
            - 'bag': BaggingClassifier 
            - 'stc': StackingClassifier
            
        :param preprocessor: Callable preprocessor methods. Can build a 
            preprocessor  by creating your own pipeline with different 
            composite methods. Refer to the method  
            :meth:`watex.preprocessing.sl.Preprocessing.make_preprocessor`. 
            
        """
        self._logging.info('Preprocessing model creating using %s method.'% 
                           self.make_preprocessing_model.__name__)
        
        if preprocessor is None: 
            self.preprocessor = self.make_preprocessing_model()
        
        if estimator_ is not  None : 
            self._select_estimator_ = estimator_ 
        
        # set default configuration of estimators 
        if estimator_ is None : 
            for e_pref, e_v in d_estimators__.keys(): 
                if e_pref =='knn': 
                    d_estimators__[e_pref]=e_v(n_neighbors=10, 
                                               metric='manhattan')
                elif e_pref =='svc': 
                     d_estimators__[e_pref]=e_v(C=100, gamma=1e-3, 
                                                random_state=self.random_state)
                elif e_pref =='rdf': 
                    d_estimators__[e_pref]=e_v(n_estimators=200, 
                                               random_state=self.random_state)
                else: 
                    d_estimators__[e_pref]=e_v(random_state=self.random_state)

            self._select_estimator_= d_estimators__[defaut_estimator]        
        
        if hasattr(self._select_estimator_, '__call__'): 
            
            self._logging.info('Evaluation using single estimator.')
            
            self.preprocessing_model = make_pipeline(self._preprocessor, 
                                                     self._select_estimator_)
            self.model_dict={'none':  self.preprocessing_model}
            
        #keep all model on a dictionnary of model 
        
        elif isinstance(self._select_estimator_, dict): 
            self._logging.info('Evaluate model using multiple estimators')
            
            for es_key, es_val in self._select_estimator_.items(): 
    
                self.model_dict[es_key] =make_pipeline(self._preprocessor,
                                                       es_val) 
                
            
                    
        
    
def find_categorial_and_numerical_features(*, df= None, features= None,  
                                           categorial_features: Iterable[T]=None ,
                                           numerical_features:Iterable[T]=None 
                                           ) -> Generic[T]: 
    """ 
    Retrieve the categorial or numerical features on whole features 
    of dataset. 
    
    :param df: Container `pd.DataFrame` of all features in the dataset.
    :param features:
        Iterable object ``list`` in the dataset. If `df` is given, don't need
        to set any others arguments. 
    :param categorial_features: 
        list of selected categorial features. Need to provide the whole 
        `features` to find the `numerical_values` 
    :param numerical_features: 
        list of selected `numerical_features`. If given, provides the 
        `features` of the whole dataframe to find the `categorial_features`
        
    :Note: Once the `features` argument is set, provide at least 
        `categorial_features` or `numerical_features` to find the one of 
        the targetted features. 
        
    :return: 
        - `categorial_features`: list of qualitative parameters
        - `numerical_features`: list of quantitative parameters 
    
    :Example: 
        
        >>> from watex.processing.sl import find_categorial_and_numerical_features
        >>> preObj = Preprocessing(
        ...     data_fn ='data/geo_fdata/BagoueDataset2.xlsx',
        ...                )
        >>> cat, num = find_categorial_and_numerical_features(df=preObj.df)
        
    """
    
    if isinstance(df, pd.core.frame.DataFrame ): 
        if features is None : 
            features = list(df.columns)
            
    if isinstance(features, (set, dict, np.ndarray)): 
        features = list(features)
    if isinstance(features, str): 
        features =[features]
        
        
    if df is None and features is None : 
        if categorial_features is None or numerical_features is None: 

            _logger.debug(
                'NoneType is found! At least a `df` or `features` and ' 
                ' or `numerical_features|categorial features ` '
                'must be supplied.')
            warnings.warn('NoneType is found. Could not parsed the categorial'
                ' features and numerical features.')
    if df is not None:
        categorial_features, numerical_features =[],[] 

        for ff in features: 
            try : 
                df.astype({
                         ff:np.float})
            except:
                categorial_features.append(ff)

            else: 
                numerical_features.append(ff) 
                
        if len(numerical_features) ==0 : numerical_features =None 
        if len(categorial_features)==0 : categorial_features =None
        
    elif features is not None :

        if (categorial_features or numerical_features ) is None : 
            _logger.error(
                'NoneType is detected. Set at least the `numerical_features` '
                ' or the `categorial_features`.')
            warnings.warn(
                'NoneType is found! Provided at least the `numerical_features`'
                ' or the `categorial_features`.')
          
        for ii, (fname , ftype) in enumerate(zip(['cat', 'num'],
                                 [categorial_features,numerical_features ])): 
            
            if ftype is  None : continue 
            else: 
                res=  hints.cfexist(features_to =ftype,
                                features=features)
                if res is True :
                    if fname =='cat': 
                        numerical_features= list(hints.findDifferenceGenObject(
                        ftype, features))

                    elif fname =='num': 
                        categorial_features= list(hints.findDifferenceGenObject(
                        ftype, features))
                        
                    break 
                
                elif res is False and ii==1 : 
                     _logger.error(
                        f'Feature `{ftype}` is not found in the `dataset`'
                        '  `{features}`. Please provide the right feature`.')
                     warnings.warn(
                        f'Feature `{ftype}` not found! Provide the right'
                        " feature'name`.")
                    
    return categorial_features , numerical_features 
            


if __name__=='__main__': 
    
    preObj = Preprocessing(data_fn ='data/geo_fdata/BagoueDataset2.xlsx',
                        )
    #cat, num = find_categorial_and_numerical_features(df=preObj.df)
    df_pre = preObj.df 
    df_X= preObj .X 
    df_y = preObj.y 
    df_cache = preObj.df_cache 
    
    # print(preObj.y_train)
    # print(preObj.X_train)
    print(preObj.X_test)
    # print(preObj.df_cache )

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    