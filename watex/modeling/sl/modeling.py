# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Wed Jul  7 22:23:02 2021 hz
# This module is part of the WATex modeling package, which is released under a
# MIT- licence.

from __future__ import print_function, division 

import os 
import warnings 
import numpy as np 
import pandas as pd 

from typing import TypeVar, Generic, Iterable , Callable


T= TypeVar('T', float, int, dict, list, tuple)
from sklearn.pipeline import make_pipeline 

from sklearn.model_selection import validation_curve#, cross_val_score
from sklearn.model_selection import RandomizedSearchCV,  GridSearchCV
# from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GroupKFold 
from sklearn.model_selection import learning_curve 

# from sklearn.metrics import confusion_matrix, classification_report 
# from sklearn.metrics import mean_squared_error, f1_score 

from watex.processing.sl import Processing , d_estimators__ 

import  watex.utils.exceptions as Wex 
import  watex.utils.decorator as deco
from watex.viewer import hints 
from watex.utils._watexlog import watexlog 

_logger =watexlog().get_watex_logger(__name__)


class Modeling: 
    """
    Modeling class. The most interesting and challenging part of modeling 
    is the `tuning hyperparameters` after designing a composite estimator. 
    Getting the best params is a better way to reorginize the created pipeline 
    `{transformers +estimators}` so to have a great capability 
    of data generalization. 
    
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
    auto                    bool        Trigger the composite estimator.
                                        If ``True`` a SVC-composite estimator 
                                        `preprocessor` is given. 
                                        *default* is False.
    pipelines               dict        Collect your own pipeline for model 
                                        preprocessor trigging.
                                        it should be find automatically.           
    estimators              Callable    A given estimator. If ``None``, `SVM`
                                        is auto-selected as default estimator.
    model_score             float/dict  Model test score. Observe your test 
                                        model score using your compose estimator 
                                        for enhacement 
    model_prediction        array_like  Observe your test model prediction for 
                                        as well as the compose estimator 
                                        enhancement.
    preprocessor            Callable    Compose piplenes and estimators for 
                                        default model scorage.
    ====================  ============  =======================================  
        
    """
    def __init__(self, data_fn =None, df=None , **kwargs)->None: 
        self._logging = watexlog().get_watex_logger(self.__class__.__name__)
        
        self._data_fn = data_fn 
        self._df =df 
        
        self.pipelines = kwargs.pop('pipelines', None) 
        self.estimator =kwargs.pop('estimator', None) 
        self.random_state =kwargs.pop('random_state', 7)
        
        self.auto= kwargs.pop('auto', False)
        self.lc_kws = kwargs.pop('lc_kws', {
            'train_sizes':np.linspace(0.2, 1, 10), 
            'cv':4, 'scoring':'accuracy'})
        self.vc_kws = kwargs.pop('vc_kws', {'param_name':'C',
                                            'param_range':np.arange(1, 200, 10), 
                                            'cv':4})
        self.Processing = Processing() 
        
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.train_score =None 
        self.val_score =None 
        self._processor =None
        self._composite_model=None
        self._model_score =None 
        
        self.best_params_ =None 
        self.best_score_= None 
        self._model_pred =None 

        for key in list(kwargs.keys()): 
            setattr(self, key, kwargs[key])
            
        if self._data_fn is not None or self._df is not None: 
            self._read_modelingObj()
     
    @property 
    def model_(self): 
        """ Get a set of `processor` and `eestimator` composed of 
        the composite model """
        
        if (self.processor and self.estimator) is not None : 
            self._composite_model = make_pipeline(self.processor, 
                                                self.estimator)
        return self._composite_model 
    
    @model_.setter 
    def model_(self, pipeline_and_estimator):
        """ Set a composite estimator usinng a tuple of pipeline 
        plus  estimator"""
        if len(pipeline_and_estimator) <=1 : 
            warnings.warn(
                'A Composite model creation need at least the `pipeline` and `estimator`.')
            self._logging.debug('Need at least a `pipeline` and `estimators`')
        
        if self.processor is None : 
            self.processor, self.estimator = pipeline_and_estimator
            
        # self._composite_model = make_pipeline(self.processor, 
        #                                         self.estimator)
        
    @property 
    def processor (self): 
        """ Get te `processor` after supplying the `pipelines` """
        return self._processor
    
    @processor.setter 
    def processor (self, pipeline): 
        """ Build your processor with your pipelines. If pipeline  is not
        given , the default preprocessor will be considered instead."""
        import sklearn 
        if pipeline is None :
            if self.Processing.preprocessor  is None : 
                if (self._data_fn or self._df) is not None : 
                    self.Processing=Processing(data_fn = self._data_fn, 
                                    df=self._df ,auto=True)
            self._processor = self.Processing.preprocessor
  
        elif pipeline is not None : 
            
            if isinstance(pipeline,
                        sklearn.compose._column_transformer.ColumnTransformer): 
                self._processor = pipeline 
            else : 
                self.Processing.preprocessor = pipeline 
                self._processor = self.Processing.preprocessor 
            
        if self.estimator is None : 
                self.estimator = self.Processing._select_estimator_
    @property 
    def model_score(self):
        """ Estimate your composite model prediction """ 
        if self.model_ is not None: 
            self.model_.fit(self.X_train, self.y_train)
            self._model_score = self.model_.score(self.X_test, self.y_test) 

            try : 
                hints.formatModelScore(self._model_score,
                                       self.Processing._estimator_name)
            except: 
                self._logging.debug(
                    f'Error finding the {self.Processing._estimator_name}')
                warnings.warn(
                    f'Error finding the {self.Processing._estimator_name}')
            
        return self._model_score
   
           
    def _read_modelingObj (self, data_fn=None, df=None, 
                              pipelines: Generic[T]=None, 
                              estimator:Callable[...,T] =None)->None: 
        """ Modeling object implicity inherits from ``Processing`` usefull 
        attributes.
        
        Read the `Processing` class and from that super class populate the 
        usefull attributes. 
        
        :param data_fn:
                    Full path to features data files. Refer to `../data`
                      directory to have a look how data are look like.
                    To get this list of features. Call `Features` class 
                    to automatic generate this datafile. 
        :param df: `pd.core.frame.DataFrame` 
        
        """
        self._logging.info('Reading and populating modeling <%s> object'
            ' attributes.'%self.__class__.__name__)
            
        
        if data_fn is not None : self._data_fn = data_fn 
        if df is not None: self._df = df 

        if pipelines is not None : self.pipelines =pipelines 
        if estimator is not None : self.estimator = estimator 
        
        if self._data_fn is not None or self._df is not None : 
            self.Processing= Processing(data_fn = self._data_fn, df =self._df, 
                estimator = self.estimator, pipelines = self.pipelines,
                auto= self.auto, random_state=self.random_state)

            self.X_train = self.Processing.X_train 
            self.y_test = self.Processing.y_test 
            self.X_test = self.Processing.X_test 
            self.y_train = self.Processing.y_train 
        
        if self.estimator is not None: 
            self.Processing.estimator = self.estimator 
            self.estimator = self.Processing._select_estimator_
            
        if self.pipelines is not None : 
            self.Processing.preprocessor= self.pipelines 
            self.pipelines = self.Processing.preprocessor
            
        self.processor = self.pipelines 

    @deco.visualize_valearn_curve(reason ='learn_curve',turn='off', plot_style='line',
        train_kws={'color':'blue', 'linewidth':2, 'marker':'o', 'linestyle':'dashed'} , 
        val_kws ={'color':'r', 'linewidth':3,'marker':'H', 'linestyle':'--'}, 
        xlabel={'xlabel':'Training set '}, ylabel={'ylabel':'Validation set '})
    def get_learning_curve (self, estimator=None, X_train=None, 
                         y_train=None, learning_curve_kws:Generic[T]=None,
                         validation_curve_kws:Generic[T]=None,
                         **kws)-> Iterable[T]: 
        """ Compute the train sore and validation curve to visualize 
        your learning curve. 
  
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
            - `switch`: Turn ``on`` or ``off`` the learning curve of 
                    validation curve.
            -`trigDec`: Trigger the decorator 
            - `N`: number of param range for plotting. 

        
        """
        def compute_validation_curve(model, X_train, y_train, param_ks):
            """ Compute learning curve and plot 
            errors with training set size"""
            train_score , val_score = validation_curve(model,
                                                       X_train, y_train, 
                                                       **param_ks )
            return train_score , train_score 

        valPlot =kws.pop('val_plot', False)
        learning_curve_kws = kws.pop('lc_kws', None)
        trigDec = kws.pop('switch', 'off')
        trig_preprocessor = kws.pop('preprocessor', False)
        
        if learning_curve_kws is not None: 
            self.lc_kws =learning_curve_kws
        if validation_curve_kws is not None : 
            self.vc_kws = validation_curve_kws
            
        if estimator is not None : 
            self.estimator = estimator 
            
        elif estimator is None : 
            if trig_preprocessor: 
                if not self.Processing._auto:
                    self.Processing._auto=True 
                    self.Processing.auto = self.Processing._auto 
                self.estimator = self.Processing.estimator
         
            else : 
                self._logging.info(
                    'Estimator is not provide! Trigger the `preprocessor`'
                    ' by setting to ``True`` to visualize the default pipelines '
                    ' implementations.')
                warnings.warn(
                    'Estimator is not given! Set `preprocessor` to ``True``'
                    ' to get the default estimator curve.')
                raise Wex.WATexError_Estimators(
                    'Estimator not found! Please provide your estimator model '
                    ' or trigger the default composite estimator by enabling '
                    '`preprocessor` to ``True``.')
                
        if X_train is not None : 
            self.X_train = X_train 
        if y_train is not None: 
            self.y_train = y_train 
            
        if valPlot: 
            N = self.vc_kws['param_range']

            self.train_score, self.val_score = compute_validation_curve(
                        model= self.estimator, X_train=self.X_train,
                           y_train= self.y_train, param_ks=self.vc_kws)
     
        else : 
            N, self.train_score, self.val_score = learning_curve(
                estimator= self.estimator, X=self.X_train, y=self.y_train,
                **self.lc_kws)

        return N, self.train_score, self.val_score , trigDec
    
    def tuning_hyperparameters (self, estimator: Callable[...,T]=None, 
                                 hyper_params:Generic[T]=None, cv:T=4, 
                                 grid_kws:Generic[T]=None,
                                 **kws): 
        """ Tuning hyperparametres from existing estimator to evaluate 
        performance. Boosting the model using the model `best_param` 
        
        :param estimator: Callable estimator or model to boost 
        :param hyper_params: dict of hyperparameters of the `estimator`
        :param cv: Cross validation cutting off. the *default* is 4
        
        :param grid_kws:dict of other gridSearch parameters
        
        :Example: 
            
            >>> from watex.modeling.sl.modeling import Modeling 
            >>> from sklearn.preprocessing import RobustScaler, PolynomialFeatures 
            >>> from sklearn.feature_selection import SelectKBest, f_classif 
            >>> from sklearn.svm import SVC 
            >>> from sklearn.compose import make_column_selector 
            >>> my_own_pipelines= {
                    'num_column_selector_': make_column_selector(
                        dtype_include=np.number),
                    'cat_column_selector_': make_column_selector(
                        dtype_exclude=np.number),
                    'features_engineering_':PolynomialFeatures(
                        3, include_bias=False),
                    'selectors_': SelectKBest(f_classif, k=3), 
                    'encodages_': RobustScaler()
                      }
            >>> my_estimator = SVC(C=1, gamma=1e-4, random_state=7)
            >>> modelObj = Modeling(data_fn ='data/geo_fdata/BagoueDataset2.xlsx', 
                           pipelines =my_own_pipelines , 
                           estimator = my_estimator)
            >>> hyperparams ={
                'columntransformer__pipeline-1__polynomialfeatures__degree': np.arange(2,10), 
                'columntransformer__pipeline-1__selectkbest__k': np.arange(2,7), 
                'svc__C': [1, 10, 100],
                'svc__gamma':[1e-1, 1e-2, 1e-3]}
            >>> my_compose_estimator_ = modelObj.model_ 
            >>> modelObj.tuning_hyperparameters(
                                        estimator= my_compose_estimator_ , 
                                        hyper_params= hyperparams, 
                                        search='rand') 
            >>> modelObj.best_params_
            >>> modelObj.best_score_
            
        """
        with_gridS= kws.pop('Search','GridSearchCV' )
        X_train =kws.pop('X_train', None)
        y_train =kws.pop('y_train', None)
        if grid_kws is None : 
            grid_kws={}
        
        if X_train is not None : self.X_train = X_train 
        if y_train is not None: self.y_train = y_train 
        
        if with_gridS is None : 
            self._logging.debug(
                ' `Search` is set to ``None``. ``GgridSearchCV`` is used as'
                'as default tuning hyperparameters.')
            warnings.warn(
                ' `Search` is set to ``None``. ``GgridSearchCV`` is used as'
                'as default tuning hyperparameters.')
            with_gridS = 'gridsearchcv'
            
        if 'grid' in with_gridS.lower(): 
            with_gridS = 'gridsearchcv'
        elif 'rand' in with_gridS.lower() :
             with_gridS = 'randomizedsearchcv'
        
        if  with_gridS == 'gridsearchcv': 
            model_grid = GridSearchCV(estimator, hyper_params, cv =cv, **grid_kws )
        elif  with_gridS == 'randomizedsearchcv': 
            model_grid = RandomizedSearchCV(estimator, hyper_params, cv=cv, **grid_kws)
            

        model_grid.fit(self.X_train, self.y_train)
        self._model_pred = model_grid.predict(self.X_test)
        self.best_score_= model_grid.best_score_ 
        self.best_params_= model_grid.best_params_
        
        return self.best_score_ , self.best_params_ 
    
        
if __name__=='__main__': 
    
    from sklearn.preprocessing import RobustScaler,  PolynomialFeatures 
    from sklearn.feature_selection import SelectKBest, f_classif 
    # from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC 
    from sklearn.compose import make_column_selector 
    
    # my_own_pipelines= {
    #                 'num_column_selector_': make_column_selector(dtype_include=np.number),
    #                 'cat_column_selector_': make_column_selector(dtype_exclude=np.number),
    #                 'features_engineering_':PolynomialFeatures(3, include_bias=False),
    #                 'selectors_': SelectKBest(f_classif, k=3), 
    #                 'encodages_': RobustScaler()
    #                   }
    # my_estimator = SVC(C=1, gamma=1e-4, random_state=7)#RandomForestClassifier(max_depth=100, random_state=7)
    # modelObj = Modeling(data_fn ='data/geo_fdata/BagoueDataset2.xlsx', 
    #                        pipelines =my_own_pipelines , 
    #                        estimator = my_estimator)
    # hyperparams1= {'svc__C': [1, 10, 100],
    #            'svc__gamma':[1e-1, 1e-2, 1e-3]
    #            }
    # hyperparams ={'columntransformer__pipeline-1__polynomialfeatures__degree': np.arange(2,10), 
    #          'columntransformer__pipeline-1__selectkbest__k': np.arange(2,7), 
    #          'svc__C': [1, 10, 100],
    #          'svc__gamma':[1e-1, 1e-2, 1e-3]}
    
    # model_estimator = modelObj.model_ 
    # modelObj.tuning_hyperparameters(estimator= model_estimator, hyper_params= hyperparams, 
    #                                 search='rand') 
    # print(modelObj.best_params_)
    # print(modelObj.best_score_)
    
    pipeline2={
                    'num_column_selector_': make_column_selector(dtype_include=np.number),
                    'cat_column_selector_': make_column_selector(dtype_exclude=np.number),
                    'features_engineering_':PolynomialFeatures(2, include_bias=False),
                    'selectors_': SelectKBest(f_classif, k=2), 
                    'encodages_': RobustScaler()
                      } 
    estimator2= SVC(C=1, gamma=0.1, random_state=7)
    modelObj = Modeling(data_fn ='data/geo_fdata/BagoueDataset2.xlsx', 
                           pipelines =pipeline2 , 
                           estimator = estimator2)
    print(modelObj.processor)
    print(modelObj.model_score)
    # myipeles 
    # modelObj.get_learning_curve()
    
    # print(modelObj.train_scores) 
    # print(modelObj.val_scores)

    