# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import ( 
    print_function,
    division 
)

import warnings 
import numpy as np 
import pandas as pd 

from .._typing import ( 
    T, 
    Generic,
    Iterable , 
    Dict ,
    Callable,
    Optional,
    Union 
  )
from ..exlib.sklearn  import ( 
    make_pipeline, 
    validation_curve, 
    RandomizedSearchCV, 
    GridSearchCV, 
    learning_curve, 
    permutation_importance, 
    confusion_matrix
    )
from .processing import Processing 
from ..utils.mlutils import ( 
    formatModelScore 
    )
from ..decorators import ( 
    pfi, 
    visualize_valearn_curve, 
    predplot, 
    )
from .._watexlog import watexlog 
from ..exceptions import ( 
    EstimatorError, 
    ArgumentError
    )

# import  watex.exceptions as Wex 
# import  watex.decorators as deco

_logger =watexlog().get_watex_logger(__name__)

__all__=["BaseModel"] 

class BaseModel: 
    """
    Base model class. The most interesting and challenging part of modeling 
    is the `tuning hyperparameters` after designing a composite estimator. 
    Getting the best params is a better way to reorginize the created pipeline 
    `{transformers +estimators}` so to have a great capability 
    of data generalization. 
    
    Arguments 
    ----------
    *dataf_fn*: str 
        Path to analysis data file. 
    *df*: pd.Core.DataFrame 
        Dataframe of features for analysis . Must be contains of 
        main parameters including the target name of pd.Core.series 
        of columns of `df`. 

    Holds on others optionals infos in ``kwargs`` arguments: 

    =================   ============    =======================================
    Attributes              Type        Description  
    =================   ============    =======================================
    auto                 bool           Trigger the composite estimator.
                                        If ``True`` a SVC-composite estimator 
                                        `preprocessor` is given. 
                                        *default* is False.
    pipelines            dict           Collect your own pipeline for model 
                                        preprocessor trigging.
                                        it should be find automatically.           
    estimators           Callable       A given estimator. If ``None``, `SVM`
                                        is auto-selected as default estimator.
    model_score          float/dict     Model test score. Observe your test 
                                        model score using your compose estimator 
                                        for enhancement or your own pipelines. 
    model_prediction     array_like     Observe your test model prediction for 
                                        as well as the compose estimator 
                                        enhancement.
    processor            Callable       Compose piplenes and estimators for 
                                        default model scorage.
    =================   ============    =======================================  
     
    Examples
    --------
    >>> from watex.bases.modeling import BaseModel
    >>> from sklearn.preprocessing import RobustScaler,  PolynomialFeatures 
    >>> from sklearn.feature_selection import SelectKBest, f_classif 
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.compose import make_column_selector 
    >>> estimator2= RandomForestClassifier()
    >>> modelObj = BaseModel(
    ...     data_fn ='data/geo_fdata/BagoueDataset2.xlsx',
    ...     pipelines = {
    ...            'num_column_selector_': make_column_selector(
    ...                dtype_include=np.number),
    ...            'cat_column_selector_': make_column_selector(
    ...                dtype_exclude=np.number),
    ...            'features_engineering_':PolynomialFeatures(
    ...                2, include_bias=False),
    ...            'selectors_': SelectKBest(f_classif, k=2), 
    ...            'encodages_': RobustScaler()
    ...              }, 
    ...     estimator = RandomForestClassifier()
    ...        )
    """
    def __init__(self, data_fn =None, df=None , **kwargs)->None: 
        self._logging = watexlog().get_watex_logger(self.__class__.__name__)
        
        self._data_fn = data_fn 
        self._df =df 
        
        self.pipelines = kwargs.pop('pipelines', None) 
        self.estimator =kwargs.pop('estimator', None) 
        self.auto= kwargs.pop('auto', False)
        self.random_state =kwargs.pop('random_state', 7)
        self.savefig = kwargs.pop('savefig', None)
        self.Processing = Processing() 
        self.lc_kws = kwargs.pop('lc_kws', {
            'train_sizes':np.linspace(0.2, 1, 10), 
            'cv':4, 'scoring':'accuracy'})
        self.vc_kws = kwargs.pop('vc_kws', {'param_name':'C',
                                            'param_range':np.arange(1, 200, 10), 
                                            'cv':4})
        
        self.figsize =kwargs.pop('fig_size', (12, 8))
        self.fimp_kws=kwargs.pop('fimp_kws', {"width": 0.3, "color":'navy',    
                            "edgecolor" : 'blue', "linewidth" : 2,
                            "ecolor" : 'magenta', "capsize" :5, 
                            'figsize':self.figsize})
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X= None 
        
        self.train_score =None 
        self.val_score =None 
        self._processor =None
        self._composite_model=None
        self._model_score =None 
        
        self.best_params_ =None 
        self.best_score_= None 
        self._model_pred =None 
        self.y_pred= None
        
        
        self.confusion_matrix=None 

        for key in list(kwargs.keys()): 
            setattr(self, key, kwargs[key])
            
        if (self._data_fn  or self._df) is not None: 
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
                'A Composite model creation need '
                'at least the `pipeline` and `estimator`.')
            self._logging.debug(
                'Need at least a `pipeline` and `estimators`')
        
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
        given, the default preprocessor will be considered instead."""
        import sklearn 
        if pipeline is None :
            if self.Processing.preprocessor  is None : 
                if (self._data_fn or self._df) is not None : 
                    self.Processing=Processing(data_fn = self._data_fn, 
                                    df=self._df ,auto=True, 
                                    random_state=self.random_state)
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
                formatModelScore(self._model_score,
                                       self.Processing._estimator_name)
            except: 
                self._logging.debug(
                    f'Error finding the {self.Processing._estimator_name}')
                warnings.warn(
                    f'Error finding the {self.Processing._estimator_name}')
            
        return self._model_score
   
           
    def _read_modelingObj (self, data_fn:Optional[T]=None, df=None, 
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
            self.X= self.Processing.X
        
        if self.estimator is not None: 
            try: 
                self.estimator.__class__.__name__
            except : 
                self.Processing.estimator = self.estimator 
                self.estimator = self.Processing._select_estimator_
                
            
        if self.pipelines is not None : 
            self.Processing.preprocessor= self.pipelines 
            # self.pipelines = self.Processing.preprocessor
        if self.auto:
            self.processor = self.pipelines 

    @visualize_valearn_curve(reason ='learn_curve',turn='off',
         plot_style='line', train_kws={'color':'blue', 'linewidth':2, 
                   'marker':'o','linestyle':'dashed', 'label':'Training set'}, 
        val_kws ={'color':'r', 'linewidth':3,'marker':'H',
                  'linestyle':'-', 'label':'Validation set'}, 
        xlabel={'xlabel':'Training set '},
        ylabel={'ylabel':'performance on the validation set '})
    
    def get_learning_curve (self, 
                            estimator:Callable[..., T]=None,
                            X_train=None, 
                             y_train=None,
                             learning_curve_kws:Generic[T]=None,
                             **kws
                             )-> Iterable[T]: 
        """ Compute the train score and validation curve to visualize 
        your learning curve. 
          
        :param estimator: The creating model. If ``None`` 
        :param X_train: pd.core.frame.DataFrame  of selected trainset
        :param x_test:  pd.DataFrame of  selected Data for testset 
        :param y_train: array_like of selected data for evaluation set.        
        :param y_test: array_like of selected data for model test 
        
        :param val_kws:
            `validation_curve` keywords arguments.  if none the *default* 
            should be::
                
                val_curve_kws = {"param_name":'C', 
                             "param_range": np.arange(1,210,10), 
                             "cv":4}
        :returns: 
            - `train_score`: float|dict of trainset score. 
            - `val_score` : float/dict of valisation score. 
            - `switch`: Turn ``on`` or ``off`` the learning curve of validation
                curve.
            -`trigDec`: Trigger the decorator. 
            - `N`: number of param range for plotting.
            
        :Example:
            >>> from watex.bases.modeling import BaseModel
            >>> processObj = BaseModel(
                data_fn = 'data/geo_fdata/BagoueDataset2.xlsx')
            >>> processObj.get_learning_curve (
                switch_plot='on', preprocessor=True)
        """
        
        def compute_validation_curve(model, X_train, y_train, **param_kws):
            """ Compute learning curve and plot 
            errors with training set size"""
            train_score , val_score = validation_curve(model,
                                                       X_train, y_train, 
                                                       **param_kws )
            return train_score , train_score 

        valPlot =kws.pop('val_plot', False)
        learning_curve_kws = kws.pop('lc_kws', None)
        trigDec = kws.pop('switch_plot', 'off')
        trig_preprocessor = kws.pop('preprocessor', False)
        val_kws = kws.pop('val_kws', None)
        train_kws = kws.pop('train_kws', None)
        
        if learning_curve_kws is not None: 
            self.lc_kws =learning_curve_kws
        if val_kws  is not None : 
            self.vc_kws = val_kws 
            
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
                raise EstimatorError(
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
                           y_train= self.y_train,**self.vc_kws)
            try : 
                pname = self.vc_kws['param_name']
            except : 
                pname =''
                
        else : 
            N, self.train_score, self.val_score = learning_curve(
                self.estimator, X=self.X_train, y=self.y_train,
                **self.lc_kws)
            pname =''
            
        return (N, self.train_score, self.val_score ,
                trigDec, pname, val_kws, train_kws)
    
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
            >>> from watex.modeling.basics import SLModeling 
            >>> from sklearn.preprocessing import RobustScaler,PolynomialFeatures 
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
            >>> modelObj = SLModeling(data_fn ='data/geo_fdata/BagoueDataset2.xlsx', 
                           pipelines =my_own_pipelines , 
                           estimator = my_estimator)
            >>> hyperparams ={
                'columntransformer__pipeline-1__polynomialfeatures__degree': 
                    np.arange(2,10), 
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
            model_grid = GridSearchCV(estimator, hyper_params, 
                                      cv =cv, **grid_kws )
        elif  with_gridS == 'randomizedsearchcv': 
            model_grid = RandomizedSearchCV(estimator, hyper_params,
                                            cv=cv, **grid_kws)
            

        model_grid.fit(self.X_train, self.y_train)
        self._model_pred = model_grid.predict(self.X_test)
        self.best_score_= model_grid.best_score_ 
        self.best_params_= model_grid.best_params_
        
        return self.best_score_ , self.best_params_ 
    
    @predplot(turn='off', fig_size =(10, 5), ObsLine =('on','ypred'))
    def get_model_prediction(self, estimator:Callable[..., T]=None,
                                 X_test:Optional[T]=None , 
                                 y_test:Optional[T]=None, 
                                **kws) -> Iterable[T]: 
        """
        Get the model prediction and quick plot using the surche decorator.
        
        The decorator holds many keyword arguments to customize plot. Refer to 
        :class:`watex.utils.decorator.predPlot`. 
        
        :param estimator: The creating model. If ``None`` 
        :param x_test:  pd.DataFrame of  selected Data for testset 
        :param y_test: array_like of selected data for model test 
        
        :param kws: Additional keywords arguments which refer to the `data_fn`
                    `df` and `pipelines` parameters. 
        :param switch: Turn `on` or `off` the decorator.

        :Example: 
            
            >>> from watex.modeling.sl import Modeling 
            >>> modelObj = Modeling(
                data_fn ='data/geo_fdata/BagoueDataset2.xlsx', 
                pipelines ={
                    'num_column_selector_': make_column_selector(
                        dtype_include=np.number),
                    'cat_column_selector_': make_column_selector(
                        dtype_exclude=np.number),
                    'features_engineering_':PolynomialFeatures(2,
                                                    include_bias=False),
                    'selectors_': SelectKBest(f_classif, k=2), 
                    'encodages_': RobustScaler()
                      }, estimator = SVC(C=1, gamma=0.1))
            >>> modelObj.get_model_prediction(estimator =testim, switch ='on')
        """

        data_fn: Optional[T]= kws.pop('data_fn', None)
        df:Optional[T] = kws.pop('df', None)
        pipelines:Callable[..., Generic[T]]=kws.pop('pipelines', None)
        switch:Union [bool, str]= kws.pop('switch', 'off')
        
        if estimator is not None :
            self.estimator =estimator 
        if pipelines is not None: 
            self.pipelines =pipelines 
            
        if X_test is not None: self.X_test= X_test 
        if y_test is not None: self.y_test =y_test 
        
        if (self._data_fn and self._df) is None : 
            if data_fn is not None : 
                self._data_fn =data_fn 
            if df is not None: self._df = df 
   
            if (self._data_fn or self._df ) is not None: 
                self._read_modelingObj(data_fn=self._data_fn, df=self._df , 
                                    pipelines = self.pipelines,
                                    estimator= self.estimator)
            else: 
                raise ArgumentError(
                    "Could not find any data for reading!")
        
        self.y_pred= self.estimator.predict(self.X_test)
        self.confusion_matrix = confusion_matrix(self.y_test, self.y_pred)
        # create y_pred dataframe 
        df_ypred = pd.DataFrame(self.y_pred, index=self.y_test.index)
        
        return  self.y_test,  df_ypred , switch 
    
     
    @property 
    def feature_importances_(self): 
        """ Get the bar plot of features importances.
        If the estimator has not `feature_importances_` attributes, it will 
        raise an error."""
        import matplotlib.pyplot as plt 
        
        try : 
            estim_name = self.estimator.__class__.__name__ 
        except : 
            warnings.warn(
                'Error occurs when trying to find the estimator name.')
        else : 
            self.estimator.fit(self.X_train, self.y_train)
            self.estimator.score(self.X_test, self.y_test)
            try : 
                pd.DataFrame(self.estimator.feature_importances_ *100, 
                   index=self.X_train.columns).plot.bar(**self.fimp_kws)
                
            except AttributeError as e: 
                print(e.args)
                plt.close()
            except: 
                self._logging.info(
                    f"{estim_name} object has no attribute "
                    "feature_importances_'" )
                warnings.warn('Could not plot the `feature_importances_`.'
                              f' The `{estim_name}` estimator has no'
                              ' attributes`feature_importances_')
                plt.close()
                
            else: 
                plt.xlabel('Name of features')
                plt.ylabel('Importance of feature in %')
                plt.show()

    @pfi(reason ='pfi', turn='off', fig_size= (10,3),savefig=None,
              barh_kws= {'color':'blue','edgecolor':'k', 'linewidth':2},
              box_kws= {'vert':False}, dendro_kws={'leaf_rotation':90},
              fig_title= 'PFI diagram')    
    def permutation_feature_importance(self, 
                            estimator:Callable[..., T]=None,
                            X_train:Optional[T] =None ,
                            y_train:Optional[T]=None, 
                            pfi_kws:Dict[str, T]=None,
                            **kws):
        """
        Evaluation of features importance with tree estimators before 
        shuffle and after shuffling trees. 
        
        Permutation feature importance is a model inspection technique that
        can be used for any fitted estimator when the data is tabular.
        This is especially useful for non-linear or opaque estimators. Refer to
        :ref:`this link <https://scikit-learn.org/stable/modules/permutation_importance.html>`_
        for more details. 
        
        :param estimator: The estimator to evaluate the importance of
            features. The default is ``RandomForestClassifier``.
                      
        :param X_train: pd.core.frame.DataFrame  of selected trainset.
        
        :param y_train: array_like of selected data for evaluation set.  
        
        :param n_estimators: 
            Number of estimator composed the tree. The *default* is 100 
        :param n_repeats: Number of tree shuffling. The *default* is 10.
        
        :param pfi_kws: 
            `permution_importance` callable additional keywords arguments. 
        :param pfi_stype: Type of plot. Can be : 
            - ``pfi`` for permutation feature importance before
                and after shuffling trees  
            -``dendro`` for dendrogram plot . 
            The *default* is `pfi`.
            
        :param switch: Turn ``on`` or ``off`` the decorator.
            
        :Example:
            
            >>> from watex.bases.modeling import BaseModel
            >>> from sklearn.ensemble import AdaBoostClassifier
            >>> modelObj = BaseModel()
            >>> modelObj.permutation_feature_importance(
            ...    estimator = AdaBoostClassifier(random_state=7),
            ...    data_fn ='data/geo_fdata/BagoueDataset2.xlsx',  
            ...     switch ='on', pfi_style='pfi')
            
        """
        
        savefig:Optional[T] =kws.pop('savefig', None)
        random_state:Optional[T] = kws.pop('random_state', None)
        n_estimators: int = kws.pop('n_estimators', 100)
        n_repeats: int = kws.pop('n_repeats', 10)
        X_train:Optional[T] =kws.pop('X_train', None)
        y_train:Optional[T] =kws.pop('y_train', None) 
        X:Optional[T]=kws.pop('X', None)
        data_fn: Optional[T]= kws.pop('data_fn', None)
        df:Optional[T] = kws.pop('df', None)
        pfi_type= kws.pop('pfi_style','pfi')
        switch: Union[str, bool]= kws.pop('switch', 'off')
        n_jobs: Optional[T]=kws.pop('n_jobs', -1)
        
        if pfi_kws is None : pfi_kws={}
        
        if savefig is not None : self.savefig = savefig 
        if X_train is not None : self.X_train = X_train 
        if y_train is not None: self.y_train = y_train 
        
        if X is not None :
            self.X = X
        
        if random_state is not None : self.random_state = random_state 
        if estimator  is None : 
            from sklearn.ensemble import RandomForestClassifier
            self.estimator = RandomForestClassifier(
                n_estimators=n_estimators, random_state= self.random_state)
                                        
            # self.estimator = clf
        if (self._data_fn and self._df) is None : 
            if data_fn is not None : 
                self._data_fn =data_fn 
            if df is not None: self._df = df 
            
            if (self._data_fn or self._df ) is None: 
                self._logging.error(
                    'No data found ! Could not read modeling object.')
                warnings.warn(
                    'No data found to read. Could not read the modeling object.')
                raise ArgumentError(
                    "Could not find any data to read!")
                
            elif (self._data_fn or self._df ) is not None: 
                self._read_modelingObj(data_fn=self._data_fn, df=self._df, 
                                       estimator = self.estimator)

        if estimator is not None :
            self.estimator = estimator 

        self.estimator.fit(self.X_train, self.y_train)

        try : 
            print("Accuracy on test data:")
            formatModelScore(self.estimator.score(
                                        self.X_test, self.y_test),
                                       self.estimator.__class__.__name__)
        except AttributeError as e: 
            print(e.args)
        except: pass 

        result = permutation_importance(self.estimator,
                                        self.X_train, self.y_train, 
                                        n_repeats=n_repeats,
                                random_state=self.random_state, n_jobs=n_jobs,
                                **pfi_kws)
        perm_sorted_idx = result.importances_mean.argsort()
        try: 
            # check whether the estimator has attribute `feature_importances_`
            tree_importance_sorted_idx = np.argsort(
                self.estimator.feature_importances_)
        except Exception: 
           raise AttributeError(
               f' `{self.estimator}` estimator nas no attribute'
               ' `feature_importances_`')
        else:
            tree_indices = np.arange(
                0, len(self.estimator.feature_importances_)) + 0.5

        return self.X, result, tree_indices,\
            self.estimator, tree_importance_sorted_idx,\
            self.X_train.columns, perm_sorted_idx, pfi_type, switch, savefig
        
        

 

    