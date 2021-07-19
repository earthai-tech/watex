# Copyright (c) 2021 Kouadio K. Laurent, Wed Jul 14 20:00:26 2021
# This module is part of the WATex processing package, which is released under a
# MIT- licence.

import os 
import warnings 
import inspect
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
import  watex.utils.decorator as deco
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
    
    This class summarizeed supervised learning methods analysis. It  
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
       
    ====================  ============  =======================================
    Attributes              Type                Description  
    ====================  ============  =======================================
    categorial_features     list        Categorical features list. If not given
                                        it should be find automatically.           
    numerical_features      list        Numerical features list. If not given, 
                                        should be find automatically. 
    target                  str         The name of predicting feature. The 
                                        *default* target is ``flow``.
    random_state            int         The state of data shuffling. The 
                                        default is ``7``.
    default_estimator       str         The default estimator name for predic-
                                        ting the target value. A predifined 
                                        defaults estimators prameters are set 
                                        and keep in cache for quick prepro-
                                        cessing like: 
                                            - 'dtc': For DecisionTreeClassifier 
                                            - 'svc': Support Vector Classifier 
                                            - 'sdg': SGDClassifier
                                            - 'knn': KNeighborsClassifier
                                            - 'rdf`: RandmForestClassifier 
                                            - 'ada': AdaBoostClassifier 
                                            - 'vtc': VotingClassifier
                                            - 'bag': BaggingClassifier 
                                            - 'stc': StackingClassifier
                                        If not given the default is ``svm`` or 
                                        ``svc``.
    test_size               float       The testset size. Must be <1.The 
                                        sample test size is ``0.2`` either 
                                        20% of dataset.      
    drop_columns            list        List collection of the unusefull 
                                        `features` for predicting. Can be elimi-
                                        nated by puting them on a drop list. 
    ====================  ============  =======================================  

    Can get other interessing attributes after buiding your preprocessor using 
    the :meth:``watex.processing.sl.Preprocessing.make_preprocessor` or 
    builing your test model using the
    :meth:``watex.processing.sl.Preprocessing.make_preprocessing_model`.
    
    ==============================  ==============  ===========================
    Attributes                      Type            Description  
    ==============================  ==============  ===========================
    X_train                         pd.DataFrame    Selected trainset
    x_test                          pd.DataFrame    selected Data for testset 
    y_train                         array_like      selected data for evaluation 
                                                    set. 
    y_test                          array_like      selected data for model test 
    preprocessor                    Callable        composite pipeline creating 
    confusion_matrix                ndarray         Confuse on array the right 
                                                    and bad prediction. 
    classification_report           str/dict        Get the report of model 
                                                    evaluation.    
    preprocessing_model_score       float/dict      Preprocessing model test 
                                                    score. 
    preprocessing_model_prediction  array/dict      Preprocessing model 
                                                    prediction on array_like. 
    ==============================  ==============  ===========================
    
    Example: 
        
        >>> from watex.processing.sl import Preprocessing
        >>> prepObj = Preprocessing(drop_features = ['lwi', 'x_m', 'y_m'],
        ...    data_fn ='data/geo_fdata/BagoueDataset2.xlsx')
        >>> prepObj.X_train, prepObj.X_test, prepObj.y_train, prepObj.y_test
        >>> prepObj.categorial_features, prepObj.numerical_features 
        >>> prepObj.random_state = 25 
        >>> preObj.test_size = 0.25
        >>> prepObj.make_preprocessor()         # use default preprocessing
        >>> preObj.preprocessor
        >>> prepObj.make_preprocessing_model( default_estimator='SVM')
        >>> prepObj.preprocessing_model_score
        >>> prepObj.preprocess_model_prediction
        >>> prepObj.confusion_matrix
        >>> prepObj.classification_report

    """

    def __init__(self, data_fn =None , df=None , **kwargs)->None : 
        self._logging = watexlog().get_watex_logger(self.__class__.__name__)
        
        self._data_fn = data_fn 
        self._df =df   
        
        self.categorial_features =kwargs.pop('categorial_features', None)
        self.numerical_features =kwargs.pop('numerical_features', None)
        
        self.target =kwargs.pop('target', 'flow')
        self._drop_features = kwargs.pop('drop_features', ['lwi'])
        self.random_state = kwargs.pop('random_state', 7)
        self.default_estimator = kwargs.pop('default_estimator', 'svc')
        self.test_size = kwargs.pop('test_size', 0.2)
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
        self._cat_column_selector =make_column_selector(
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
            train_test_split (self.X, self.y, test_size = self.test_size,
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
        
        :Note: 
            Can build the default preprocessor by merely call::
            
                 >>> from watex.processing.sl import Preprocessing
                 >>> preObj = Preprocessing(
                     ...    data_fn ='data/geo_fdata/BagoueDataset2.xlsx',
                     ...            ) 
                 >>> preObj.make_preprocessor()
                 >>> preObj.preprocessor
                 
            Or build your own preprocesor object using the example below: 
                
        :Example: 
            
            >>> from watex.processing.sl import Preprocessing
            >>> from sklearn.preprocessing import StandardScaler 
            >>> preObj = Preprocessing(
            ...    data_fn ='data/geo_fdata/BagoueDataset2.xlsx',
            ...            )
            >>> preObj.random_state = 23
            >>> preObj.make_preprocessor(
            ...    num_column_selector_= make_column_selector(
            ...                            dtype_include=np.number),
            ...    cat_column_selector_= make_column_selector(
            ...                            dtype_exclude=np.number),
            ...    features_engineering_=PolynomialFeatures(7,
            ...                            include_bias=True),
            ...    selectors_=SelectKBest(f_classif, k=4), 
            ...        encodages_= StandardScaler())
            >>> preObj.preprocessor
             
        """
        preprocessor_kws = kwargs.pop('prepreocessor_kws', {})
        make_pipeline_kws =kwargs.pop('make_pipeline_kws', {})

        
        if num_column_selector_ is not None :
            self._num_column_selector = num_column_selector_
        if cat_column_selector_ is not None : 
            self._cat_column_selector = cat_column_selector_ 
            
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
        
        self.preprocessor =make_column_transformer(
            (numerical_pipeline, numerical_features), 
            (categorical_pipeline, categorical_features), **preprocessor_kws)
        
        return self.preprocessor 
    
    def make_preprocessing_model(self, preprocessor: Callable[..., T]= None, 
                                 estimators_:Callable[..., T]=None,
                                  **kws)->T: 
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
            
        :param preprocessor: 
            Callable preprocessor method. Can build a preprocessor by creating 
           your own pipeline with different composite estimator.Refer to the  
           :meth:`watex.preprocessing.sl.Preprocessing.make_preprocessor` for
           details. 
           
        :param estimator_: 
            Callable estimator method to fit the model :: 
                
                `estimator_`= SGDClassifier(random_state=13)
            
            It's possible to 
            provide multiple estimator with configuration arguments into 
            estimator dictionnary like: 
                
                `estimator_`={'knn': KNeighborsClassifier(n_neighbors=10, 
                                                          metric='manhattan') , 
                              'svc':SVC(C=100, gamma=1e-3, random_state=25)}
            when multiple estimators is provided, the results of model fit and 
            prediction score should be in dict with estimator name. 
        
        :param defaut_estimator: 
            The default estimator for preprocessing model testing
            
        :Note: If ``None`` estimator is given, the *default* estimator is `svm`
            otherwise, provide the only prefix of the select  estimator into 
            the `default_estimator` keywords argument like::
                
                >>> from watex.processing.sl import Preprocessing
                >>> preObj = Preprocessing(
                    data_fn ='data/geo_fdata/BagoueDataset2.xlsx')
                >>> preObj.random_state = 7
                >>> mfitspred= preObj.make_preprocessing_model(
                    default_estimator='ada')
                >>> preObj.preprocessing_model_score
                >>> preObj.preprocess_model_prediction
                >>> preObj.confusion_matrix
                >>> preObj.classification_report
        
        Providing multiple estimator is possible like the example below: 
            
        :Example: 
            
            >>> from watex.processing.sl import Preprocessing
            >>> preObj = Preprocessing(
            ...    data_fn ='data/geo_fdata/BagoueDataset2.xlsx')
            >>>   from sklearn.ensemble import RandomForestClassifier
            >>> preObj.random_state = 7
            >>> preObj.make_preprocessing_model(estimators_={
            ...    'RandomForestClassifier':RandomForestClassifier(
            ...        n_estimators=200, random_state=0), 
            ...    'SDGC':SGDClassifier(random_state=0)})
            >>> preObj.preprocessing_model_score)
            >>> preObj.preprocessing_model_prediction)
        """
        
        def model_evaluation(model, X_train, y_train, X_test, y_test): 
            """ Evaluating model prediction """
            
            model.fit(X_train, y_train)
            
            return  model.score(X_test, y_test), model.predict(X_test)
        
        
        self._logging.info('Preprocessing model creating using %s method.'% 
                           self.make_preprocessing_model.__name__)
        
        default_estimator =kws.pop('default_estimator', None)
        
        if default_estimator is not None : 
            default_estimator = hints.controlExistingEstimator(
                default_estimator)
   
            if default_estimator is None :
                self._logging.error(
                    f'Estimator `{default_estimator}` not found! Please '
                    ' provide the right `estimator` name! ')
                warnings.warn(
                    f'Estimator `{default_estimator}` not found! Please '
                    ' provide the right `estimator` name! ')
                return 
                
       
            self.default_estimator, estim_full_name = default_estimator
                

        if preprocessor is None: 
            self.preprocessor = self.make_preprocessor()
 
        if estimators_ is not  None : 
            self._select_estimator_ = estimators_ 
        
        # set default configuration of estimators 
        if estimators_ is None : 
            self._logging.info('Loading default parameters into estimators.')
            #load all default config parameters 
            for e_pref, e_v in d_estimators__.items(): 
                if e_pref =='knn': 
                    d_estimators__[e_pref]=e_v(n_neighbors=10, 
                                               metric='manhattan')
                elif e_pref =='svc': 
                     d_estimators__[e_pref]=e_v(C=100, gamma=1e-3, 
                                                random_state=self.random_state)
                elif e_pref =='rdf': 
                    d_estimators__[e_pref]=e_v(n_estimators=200, 
                                               random_state=self.random_state)
                elif e_pref in[ 'vtc', 'stc']:
                    compose_estimators = [('SGDC', SGDClassifier(
                                random_state=self.random_state)),
                            ('DTC', DecisionTreeClassifier(
                                max_depth=100, random_state=self.random_state)), 
                            ('KNN', KNeighborsClassifier())]
                    
                    d_estimators__[e_pref]=e_v(compose_estimators)
                   
                elif e_pref =='bag':
                    
                    d_estimators__[e_pref]=e_v(
                        base_estimator=KNeighborsClassifier(), n_estimators=100)
                elif e_pref =='stc': 
                    d_estimators__[e_pref]=e_v(
                            [('SGDC', SGDClassifier(
                                random_state=self.random_state)),
                            ('DTC', DecisionTreeClassifier(
                                max_depth=100, random_state=self.random_state)), 
                            ('KNN', KNeighborsClassifier())])
                else:
                    try :
                        d_estimators__[e_pref]=e_v(random_state=self.random_state)
                    except : 
                        d_estimators__[e_pref]=e_v()
                        
                        
            # once loaded, select the estimator 
            self._select_estimator_= d_estimators__[self.default_estimator]
            try : 
                self._logging.info(
                    'End loading default parameters! The selected default estimator '
                    f'is {estim_full_name}')
            except : 
                self._logging.info(
                    'End loading default parameters! The selected default estimator '
                    f'is {default_estimator}')
            
        self.model_dict ={}
       
        if  not isinstance(self._select_estimator_, dict):
            self._logging.info(
                'Evaluation using single {0} '
                'estimator.'.format(self._select_estimator_))
            
            self.preprocessing_model = make_pipeline(self.preprocessor, 
                                                     self._select_estimator_)
      
            self.model_dict['___'] =  self.preprocessing_model
     
            for m_key, m_value in self.model_dict.items(): 
                self.preprocessing_model_score, self.preprocessing_model_prediction =\
                    model_evaluation(m_value, self.X_train , 
                                 self.y_train,  self.X_test, self.y_test)
                    
            return self.preprocessing_model_score,\
                self.preprocessing_model_prediction
        
        #keep all model on a dictionnary of model 
        elif isinstance(self._select_estimator_, dict): 
            self.preprocessing_model_score={}
            self.preprocessing_model_prediction={}
    
            self._logging.info(
                'Evaluate model using multiples estimators `{}`'.format(
                    list(self._select_estimator_.keys())))
            
            for es_key, es_val in self._select_estimator_.items():
                self.model_dict[es_key] =make_pipeline(self.preprocessor,
                                                       es_val) 
                
            for m_key, m_value in self.model_dict.items(): 
                    preprocessing_model_score,\
                        preprocessing_model_prediction =\
                        model_evaluation(m_value, self.X_train , 
                                     self.y_train,  self.X_test, self.y_test)        
                    self.preprocessing_model_score[
                        m_key]= preprocessing_model_score
                    self.preprocessing_model_prediction[
                        m_key]= preprocessing_model_prediction
         
        if not isinstance(self.preprocessing_model_prediction, dict): 
            self.confusion_matrix= confusion_matrix(self.y_test,
                                   self.preprocessing_model_prediction)
            self.classification_report= classification_report(self.y_test,
                                   self.preprocessing_model_prediction)
        else : 
             self.confusion_matrix ={keycfm: confusion_matrix(
                 self.y_test, cfmV) for keycfm, cfmV in 
                 self.preprocessing_model_prediction.items() 
                                     }  
             self.classification_report={keycR: classification_report(
                 self.y_test, clrV) for keycR, clrV in 
                 self.preprocessing_model_prediction.items()
                 }
             
        return self.preprocessing_model_score,\
            self.preprocessing_model_prediction
                
        
class Processing (Preprocessing) : 
    """
    Processing class deal with preprocessing. 
    
     Arguments: 
    ---------
        *dataf_fn*: str 
            Path to analysis data file. 
        *df*: pd.Core.DataFrame 
                Dataframe of features for analysis . Must be contains of 
                main parameters including the `target` pd.Core.series 
                as columns of `df`. 
 
    
    Holds on others optionals infos in ``kwargs`` arguments: 
    
    """  
    
    def __init__(self, data_fn =None , df=None, **kwargs):
        Preprocessing.__init__(self, data_fn=data_fn , df=df, **kwargs)
        
        self._auto =kwargs.pop('auto', False)
        self.preprocessor =kwargs.pop('preprocessor', None)
        self._select_estimator_ = kwargs.pop('estimators', None)
        
        # print('processor oooo', self.preprocessor)
        # if self._auto: 
        #     self.auto =True
            
        for key in kwargs.keys(): 
            setattr(self, key, kwargs[key])    
    @property 
    def auto (self): 
        """ Trigger the composite pipeline building and greate 
        a composite default model estimator `CE-SVC` """
        return self._auto 
    @auto.setter 
    def auto (self, auto): 
        """ Trigger the `CE-SVC` buiLding using default parameters"""
        if auto: 
            self.make_preprocessing_model(
                preprocessor= self.preprocessor, 
                estimators_= self._select_estimator_)
            
            self.model_score = self.preprocessing_model_score
            print('-'*77)
            print('>{0:<15}: {1:^10} %'.format(' Score',round(
                self.model_score *100,3 )))
            print('-'*77)
            self.model_prediction = self.preprocessing_model_prediction
            self._auto =True 
        
    
    @deco.visualize_validation_curve(turn='off', k= np.arange(1,210,10), 
               plot_style='scatter',savefig=None)               
    def get_validation_curve(self, estimator=None, X_train=None, 
                         y_train=None, val_curve_kws:Generic[T]=None, 
                         **kws):
        """ Compute the validation score and plot the validation curve if 
         the argument `turn` of decorator is switched to ``on``. If 
         validation keywords arguments `val_curve_kws` doest not contain a 
         `param_range` key, the default param_range should be the one of 
             decorator.
  
        :param model: The creating model. If ``None`` 
        :param X_train: pd.core.frame.DataFrame  of selected trainset
        :param x_test:  pd.DataFrame of  selected Data for testset 
        :param y_train: array_like of selected data for evaluation set.        
        :param y_test: array_like of selected data for model test 
        
        :param val_curve_kws:
            `validation_curve` keywords arguments.  if none the *default* 
            should be::
                
                val_curve_kws = {"param_name":'C', 
                             "param_range": np.arange(1,210,10), 
                             "cv":4}
        :returns: 
            
            - `train_score`: float|dict of trainset score 
            - `val_score` : float/dict of valisation score 
            - `switch`: Turn ``on`` or ``off`` the validation_plot.
            - `kk`: the validation `param_range` for plot.
        
        :Example: 
            
            >>> from watex.processing.sl import Processing 
            >>> processObj = Processing(
                data_fn = 'data/geo_fdata/BagoueDataset2.xlsx')
            >>> processObj.get_validation_curve(
                switch_plot='on', preprocess_step=True)
        """
        preprocess_step =kws.pop('preprocess_step', False)
        switch = kws.pop('switch_plot', 'off')

        if val_curve_kws is None :
    
            val_curve_kws = {"param_name":'C', 
                             "param_range": np.arange(1,210,10), 
                             "cv":4}
            self._logging.debug(
                f'Use default `SVM` params configurations <{val_curve_kws}>.')
            
            if inspect.isfunction(self.get_validation_curve): 
                _code = self.get_validation_curve.__code__
                filename = _code.co_filename
                lineno = _code.co_firstlineno + 1
            else: 
               filename = self.get_validation_curve.__module__
               lineno = 1
    
            warnings.warn_explicit(
                'Use default `SVM` params configurations <{val_curve_kws}>.',
                                   category=DeprecationWarning, 
                                   filename =filename, lineno= lineno)
        # get the param range 
        try: 
            kk= val_curve_kws['param_range']
        except : kk=None 
        
        
        if estimator is None: 
            if preprocess_step : 
                print('---> Preprocessing step is enabled !')
                self._logging.info(
                    'By default, the`preprocessing_step` is activated.')
                self.auto =True 
            else: 
                warnings.warn('At least one `estimator` must be supplied!')
                self._logging.error(
                    'Need a least a one `estimator` but NoneType is found.')
                raise Wex.WATexError_processing(
                    'None `estimator` detected. Please provide at least'
                    ' One `estimator`.')

        if estimator is not None :
            self._select_estimator_= estimator
            if not isinstance(self._select_estimator, dict) : 
                self.model_dict={'___':self._select_estimator_ }
            else : 
                self.model_dict = self._select_estimator_
                
        for mkey , mvalue in self.model_dict.items(): 
            if len(self.model_dict) ==1:
                self.train_score, self.val_score = validation_curve(
                                        self._select_estimator_,
                                        self.X_train, self.y_train,
                                        **val_curve_kws)
                
            elif len(self.model_dict) > 1 :
                trainScore, valScore = validation_curve(mvalue,
                                       self.X_train, self.y_train,
                                       **val_curve_kws)
                self.train_score [mkey] = trainScore
                self.val_score[mkey] = valScore 

        return self.train_score, self.val_score, switch , kk      
    
        
    def quick_estimation(self, estimators: Callable[...,T] = None, 
                         random_state:float = None, **kws): 
        """ Quick run the model without any processing.  If none estimator 
        is provided ``SVC``estimator is used.
        
        :param estimators: Callable estimator. If none a ``svc`` is used to 
                            quick estimate prediction. 
        :param random_state: The state of data shuffling.The default is ``7``.
                                        
        :Example: 
            
            >>> from watex.processing.sl import Processing 
            >>> processObj = Processing(
                data_fn = 'data/geo_fdata/BagoueDataset2.xlsx')
            >>> processObj.quick_estimation(estimators=DecisionTreeClassifier(
                max_depth=100, random_state=13)
            >>> processObj.model_score)
            >>> processObj.model_prediction)
        
        """
        
        X_train =kws.pop('X_train', None )
        if X_train is not None : self.X_train =X_train 
        
        y_train =kws.pop('y_train', None)
        if y_train is not None : self.y_train =y_train 
        
        X_test =kws.pop('X_test', None)
        if X_test is not None: self.X_test = X_test 
        
        y_test = kws.pop('y_test', None)
        if y_test is not None : self.y_test 
        
        if random_state is not None: self.random_state = random_state 
        
        if estimators is not None: 
            quick_estimator= estimators
        else : 
            quick_estimator = SVC(self.random_state)
        self.model_dict= {'SVC':quick_estimator }
        
        quick_estimator.fit(self.X_train, self.y_train)
        
        self.model_score = quick_estimator.score(self.X_test, self.y_test)
        self.model_prediction = quick_estimator.predict(self.X_test)
        
        self.confusion_matrix= confusion_matrix(self.y_test,
                                   self.model_prediction)
        self.classification_report= classification_report(self.y_test,
                               self.model_prediction)
        print('-'*77)
        print('>{0:<15}: {1:^10} %'.format(' Score',round(
            self.model_score *100,3 )))
        print('-'*77)
        
        return self.model_score , self.model_prediction
        

        
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
    
    # preObj = Preprocessing(data_fn ='data/geo_fdata/BagoueDataset2.xlsx',
    #                     )
    processObj = Processing(data_fn = 'data/geo_fdata/BagoueDataset2.xlsx', 
                            auto=False)
    processObj.get_validation_curve(switch_plot='ff', preprocess_step=True)
    print(processObj.model_score)
    print(processObj.model_prediction)
    # processObj.quick_estimation(estimators=DecisionTreeClassifier(
    #             max_depth=100, random_state=13))
    # print(processObj.model_score)
    # print(processObj.model_prediction)
    # processObj.get_validation_curve(switch_plot='on', preprocess_step=True)
    # print(processObj.train_score)
    # print(processObj.val_score)
    # preObj.random_state = 7
    # mfitspred= preObj.make_preprocessing_model()
    #cat, num = find_categorial_and_numerical_features(df=preObj.df)
    # df_pre = preObj.df 
    # # df_X= preObj .X 
    # df_y = preObj.y 
    # from sklearn.preprocessing import StandardScaler 
    # from sklearn.ensemble import RandomForestClassifier
    # preObj.random_state = 7
    # mfitspred= preObj.make_preprocessing_model(estimators_={
    #     'RandomForestClassifier':RandomForestClassifier(n_estimators=200, random_state=0), 
    #     'SDGC':SGDClassifier(random_state=0)})
    # print(preObj.preprocessing_model_score)
    # print(preObj.preprocessing_model_prediction)
    # print(preObj.confusion_matrix)
    # print(preObj.classification_report)
    # make_pre =preObj.make_preprocessor(
    #     num_column_selector_= make_column_selector(dtype_include=np.number),
    #     cat_column_selector_= make_column_selector(dtype_exclude=np.number),
    #     features_engineering_=PolynomialFeatures(7, include_bias=True),
    #     selectors_=SelectKBest(f_classif, k=4), encodages_= StandardScaler())
    # print(preObj.preprocessor )
    # print(preObj.random_state)
    
    
    # print(mfitspred)
    # print(preObj.df_cache )

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    