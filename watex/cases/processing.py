# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Wed Jul 14 20:00:26 2021
from __future__ import (
    print_function ,
    division, 
    annotations
)
import copy
import warnings 
import inspect
import numpy as np 
import pandas as pd 

from .._docstring import  DocstringComponents, _core_docs
from .._watexlog import watexlog 
from ..decorators import visualize_valearn_curve
from ..exceptions import ( 
    FeatureError , 
    NotFittedError , 
    ProcessingError, 
    EstimatorError
  )
from ..exlib.sklearn  import ( 
    DecisionTreeClassifier, 
    KNeighborsClassifier, 
    OneHotEncoder, 
    SelectKBest,  
    SGDClassifier,
    SVC, 
    PolynomialFeatures, 
    RobustScaler, 
    ColumnTransformer,
    confusion_matrix , 
    classification_report, 
    f_classif, 
    train_test_split, 
    validation_curve ,
    SimpleImputer,
    Pipeline,
    _HAS_ENSEMBLE_,   
    ) 
from ..utils.coreutils import ( 
    _is_readable , 
    _assert_all_types
    )
from ..utils.funcutils import ( 
    format_notes, 
    repr_callable_obj, 
    smart_strobj_recognition, 
    smart_format
    )
from ..utils.mlutils import (
    formatModelScore, 
    controlExistingEstimator,
    findCatandNumFeatures,
    selectfeatures, 
    evalModel 
    )
from ..typing import ( 
    T, 
    List, 
    Generic,
    Callable, 
    NDArray, 
    ArrayLike,
    F
    ) 

_logger =watexlog().get_watex_logger(__name__)
d_estimators_={'dtc':DecisionTreeClassifier, 
                'svc':SVC, 
                'sgd':SGDClassifier, 
                'knn':KNeighborsClassifier 
                 }
if _HAS_ENSEMBLE_ :
    from ..exlib.sklearn import skl_ensemble_
    
    for es_, esf_ in zip(['rdf', 'ada', 'vtc', 'bag','stc'], skl_ensemble_): 
        d_estimators_[es_]=esf_ 
        
# append repeat docs to dictdocs
_preproces_params =dict ( 
    pipe_ = """
pipe_:Callable, preprocessor object from :mod:`sklearn.pipeline`
    Pipeline can  be buit by your own pipeline with different transformer. 
    For base model prediction, it is possible to use the default pipeline.
    Call `get_default_pipe` to get the transformation list and steps. 
    """, 
    estimator_="""
estimator: Callable, F or :mod:`sklearn.metaestimator`
    Callable estimator method to fit the model:: 
        
        estimators= SGDClassifier(random_state=13)    
    """
    )
_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"], 
    base=DocstringComponents(_preproces_params), 
    )
class Preprocessing : 
    def __init__(self, 
                 tname:str ='flow', 
                 drop_features: List[str]=None,  #['lwi']
                 random_state: int =42 , 
                 default_estimator: F|str= 'svc', 
                 test_size: float =.2 ,                
                 verbose: int = 0 , 
                 ): 
        self._logging = watexlog().get_watex_logger(self.__class__.__name__)
        
        self.tname=tname
        self.drop_features=drop_features
        self.random_state=random_state
        self.default_estimator=default_estimator
        self.test_size=test_size
        self.verbose=verbose 
        self.X=None 
        self.y=None 
        self.Xt= None
        self.yt=None 
        self.features_ = None 
        self.cat_features_ =None
        self.num_features_ =None
        self.y_=None
        self.X_= None 
        self.estimator_ =None 
        self.pipe_ = None 
        self.ypred_= None 
        self.model_results_=None 
        self.base_score_=None 
        self.data_ = None 
        self.model_ =None 
  
    @property 
    def data(self): 
        return self.data_ 
    
    @data.setter
    def data (self, d): 
        """ Read the given data and create a pd.core.frame.DataFrame . 
        Call :class:~analysis.features.sl_analysis` and retrieve the best 
          information. """

        self.data_ = _is_readable(d)
    

    @property 
    def features(self): 
        """ Collect the list of features"""
        return self.features_ 
    @features.setter 
    def features(self, feats): 
        """ Set the features once given"""
        
        if isinstance(feats , str):
            self.features_= [feats]
        else: self.features_ = list( self.data.columns )
            
        
    def fit (self, 
             X:NDArray =None, 
             y:ArrayLike = None, 
             **fit_params
             ) -> 'Preprocessing': 
        """ 
        Read the whole dataset, encode the categorial features and 
        populate class attributes.
        
        If `X` and `y` are provided, they are considered as a features set
        and target respectively. They should be splitted to the training set 
        and test set respectively.
        
        Parameters 
        -----------
        X: N-d array, shape (N, M) 
            the feature arrays composed of N-columns and the M-samples. The 
            feature set excludes the target `y`. 
        y: arraylike , shape (M)
            the target is composed of M-examples in supervised learning. 
            
        data: Dataframe or shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N including the 
            target `y`. 
            Note that if the data is given, it is not necessary to provide the
            `X` and `y`. By specifying the target name `tname`, the target 
            should be remove to the data. 
        split_X_y: bool, default {'True'}
            split the datatset to training set {X, y } and test set {Xt, yt}. 
            Otherwise `X` and `y` should be considered as traning sets.  
            
        Returns 
        --------
        ``self``: `Preprocessing` instance for easy method chaining.
        
        Examples
        ---------
        >>> from watex.cases.processing import Preprocessing 
        >>> from watex.datasets import fetch_data 
        >>> data = fetch_data('bagoue original').get('data=dfy2')
        >>> pc = Preprocessing (drop_features = ['lwi', 'num', 'name']
                                ).fit(data =data )
        >>> len(pc.X ),  len(y), len(pc.Xt ),  len(pc.yt)
        ... (344, 344, 87, 87) # trainset (X,y) and testset (Xt, yt)

        """
        data = fit_params.pop('data', None)
        split_X_y= fit_params.pop('split_X_y', True)
        
        self.X_ = None or X 
        self.y_ = None or y 
        
        if data is not None: 
            self.data = data 
            self.X_= self.data.copy()
        
        if not isinstance(self.X_, (pd.DataFrame, np.ndarray) ) : 
            msg  =f"Expect an nd-array not {type (self.X_).__name__!r}."
            raise FeatureError( 
                (msg + "Use param 'data' in fit params to read the file") 
                               if isinstance(self.X_, str) else msg )
            
        if self.y_ is not None: 
            self.y_ = _assert_all_types(self.y_, pd.Series, np.ndarray)
            
        if self.drop_features is not None: 
            if isinstance (self.drop_features , str): 
                self.drop_features =[self.drop_features ]
            self.X_ = self.X_.drop(columns = self.drop_features)
             
        # find numerical and categorial features 
        self.cat_features_, self.num_features_ = findCatandNumFeatures(
            self.X_ 
            )
        # encode categorical values if exists 
        self.X_[self.cat_features_] = (self.X_[self.cat_features_ ]
                                       .apply ( lambda c: c.astype(
                                               'category').cat.codes)
                                       )
        if self.tname is not None: 
            self.y_ = selectfeatures(self.X_, features=self.tname)
            self.X_.drop(columns=self.tname, inplace =True)
            # remove the tname and update cat_features or num_features list 
            if self.tname in (self.cat_features_): 
                self.cat_features_.remove (self.tname) 
            elif self.tname in self.num_features_ : 
                self.num_features_.remove (self.tname )
                
        # for consistency, encode label y and let it untouchable if numerical 
        # value is given
        self.y_ = self.y_.astype ('category').cat.codes
        
        # splitted dataset 
        if split_X_y: 
            if self.y_ is None :
                warnings.warn("target name 'tname' is None. Cannot retrieve"
                              " the target 'y' from the dataset")
                raise FeatureError("'tname' is missing. specify the target name"
                                   " before splitting the datasets.")
                
            self.X , self.Xt, self.y, self.yt =\
                train_test_split (self.X_, self.y_, test_size = self.test_size,
                                  random_state = self.random_state )
        else: 
            # consider X and y as a trainig set. 
            self.X, self.y = copy.deepcopy(self.X_) , copy.deepcopy(self.y_ )
            
        return self 
    
    @property 
    def inspect(self): 
        """ Inspect data and trigger plot after checking the data entry. 
        Raises `NotFittedError` if ``self`` is not fitted yet."""
        if self.X is None: 
            raise NotFittedError(self.msg.format(
                expobj=self)
            )
        return 1
     
    def makeModel( self, pipe: F=None, estimator:F=None, 
                     )-> Callable[..., F]:
        """
        Assemble pipes and estimator to create the model 
        
        the model is composed of the transformers and estimator, If one is set 
        to None, it uses the default pipe and estimator which might be not the 
        one expected. Therefore providing a pipe and estimator is suggested.
        
        Parameters 
        -----------
        pipe: Callable, pipeline or preprocessor 
            Callable pipeline. Pipeline can your own pipeline with different 
            transformer. Refer to the  :class:`sklearn.pipeline.Pipeline` 
            for futher details. Call `get_default_pipe` to get the default 
            pipe.
            
        estimator: Callable, F or {sklearn estimator}
            Callable estimator method to fit the model:: 
                
                estimators= SGDClassifier(random_state=13)
                
             `Some pre-estimators can be fetched by providing the prefix as  
             a key of the estimator default dict. For instance to fetch the 
             `DecisionTreeClassifier` estimators:: 
                
                 >>> from watex.cases.processing import Preprocessing 
                 >>> Preprocessing._getdestimators()['dtc']
                 ... DecisionTreeClassifier(max_depth=100, random_state=42)
        
        Returns 
        ---------
        `model_`: Callable, {preprocessor + estimator } 
        
        Examples 
        ----------
        (1) We can get the default preprocessor by merely calling: 

        >>> from watex.cases.processing import Preprocessing 
        >>> pc = Preprocessing (tname = 'flow', drop_features =['lwi', 'name', 'num'])
        >>> data = fetch_data ('bagoue original').get('data=dfy2')
        >>> pc.fit(data =data) 
        >>> pc.makeModel() # use default model and preprocessor 
        >>> pc.model_ 
                 
        (2)-> Or build your own preprocesor object using the example below: 

        >>> from sklearn.pipeline import Pipeline  
        >>> from sklearn.compose import ColumnTransformer
        >>> from sklearn.impute import SimpleImputer
        >>> from sklearn.preprocessing import StandardScaler, OneHotEncoder
        >>> from sklearn.linear_model import LogisticRegression
        >>> from watex.datasets import fetch_data 
        >>> from watex.cases.processing import Preprocessing 
        >>> pc = Preprocessing (tname = 'flow', drop_features =['lwi', 'name', 'num'])
        >>> numeric_features = ['east', 'north', 'power', 'magnitude', 'sfi', 'ohmS']
        >>> numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), 
                   ("scaler", StandardScaler())]
            )
        >>> categorical_features = ['shape', 'geol', 'type']
        >>> categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        >>> preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ])
        >>> pc.makeModel (pipe = preprocessor, 
                          estimator =  LogisticRegression()) 
        >>> pc.model_
        
        """
        
        self.pipe_ = pipe or self.get_default_pipe ()
        if estimator is not  None : 
            self.estimator_= estimator 
        
        # set default configuration of estimators 
        if self.estimator_ is None:
            if self.verbose: 
                self._logging.info('Loading default parameters into estimators.')
                print("### -> Use default estimator instead ...")
            #load all default config parameters
            des= copy.deepcopy(self.default_estimator)
            self.default_estimator = str(self.default_estimator).lower().strip() 
            
            if self.default_estimator not in d_estimators_.keys(): 
                raise ValueError (f"Unknow default estimator :{des!r}")
                
            self.estimator_ = self._getdestimators()[self.default_estimator]
            
        self.model_ = Pipeline ( 
            steps = [( 'preprocessor', self.pipe_), 
                     (f'{self.estimator_.__class__.__name__}', self.estimator_) 
                     ]
            )
        return self.model_ 
    
    def get_default_pipe(self ):
        """ make a default pipe to preprocess the data. 
        
        Create a preprocessor by assembling multiple transformers. The default 
        pipeline is not exhaustive so to have full control of the data, it 
        is recommended to provide a strong preprocessor for the data 
        processing at once. 
        
        the method returns `self.pipe_`as  callable, preprocessor pipeline 
        from :class:`sklearn.pipeline.Pipeline` object. Basically since, the 
        default transformers are composed of: 
            
            - :meth:`sklearn.pipeline.make_pipeline` for pipeline creation. 
            - :meth:`sklearn.preprocessing.OneHotEncoder` for categorial 
                `features` encoding. 
            - :meth:`sklearn.preprocessing.PolynomialFeatures` for features 
               engineering. 
            - :meth:`sklearn.preprocessing.RobustScaler` for data scaling 
            - :meth:`sklearn.compose.make_column_transformer` for data 
                transformation. 
            - :meth:`sklearn.compose.make_column_selector` for features 
                composing.
                
        Default pipeline composition  
        -----------------------------
        * imputer
            callable to fit the missing NaN values in the dataset.the default 
            behaviour use the `strategy` equals to ``mean``. Refer to 
            :class:`sklearn.imputer.SimpleImputer`
            
        * num_column_selector 
            Callable method from `sklearn.compose.make_column_selector`
            Numerical column maker. Refer to  sklearn site for  
            :ref:'more details <https://scikit-learn.org/stable/modules/classes.html>` 
            The default is ``make_column_selector(dtype_include=np.number)``
            
        * cat_column_selector
            Callable from `sklearn.compose.make_column_selector`
            Callable method. Categorical column selector. The default is
            ``make_column_selector(dtype_exclude=np.number)``. 
            
        * features_engineering applies the `Polynomial features`
            callable from `sklearn.feature_selection`
            Callable argument using :mod:`sklearn.preprocessing` different 
            method. the default is::
            
                `PolynomialFeatures(10, include_bias=False)`
                
        * selectors
            Selector callable argument including many test 
            methods like `f_classif` or Anova test.The default is::
                
                `SelectKBest(f_classif, k=4),` 
           
        * scalers 
            Scaling data using many normalization or standardization. The 
            default is  ``RobustScaler``. 

        """
        num_pipe = Pipeline(
            steps = [ 
            # since fit method alread separated the numerical 
            # and categorical columns, not need to add as 
            # a transformer again 
            # ('num_selector', make_column_selector(dtype_include=np.number, ) ), 
            ('imputer', SimpleImputer()), 
            ('polynomialfeatures', PolynomialFeatures(10, include_bias=False) ), 
            ('selectors', SelectKBest(f_classif, k=4) ), 
            ('scalers', RobustScaler()), 
            ] 
        )
        cat_pipe = Pipeline(
            steps = [ 
            # ('num_selector', make_column_selector( dtype_exclude=np.number) ),
            ('imputer', SimpleImputer()), 
            ('onehotencoder', OneHotEncoder(handle_unknown="ignore") )
            ]
            
            )
        self.pipe_ =  ColumnTransformer ( 
            transformers=[ 
                    ('numpipe', num_pipe , self.num_features_), 
                    ( 'catpipe', cat_pipe, self.cat_features_ )
                                ]  )
            
        return self.pipe_  
        
    def baseEvaluation(self, model:F=None, eval_metric=False, **kws
                 )->float: 
        """
        Dummy baseline model from preprocessing pipeline. 
        
        onto a model by providing an estimator. 
        
        Parameters 
        -----------
        model: Callable, {'preprocessor + estimator },
            A model is scikit-learn estimator or or  composite model  built 
            from a Pipeline. If `model` is ``None`` , use the default model 
            from the default `preprocessor and `estimator`. `model` can be 
            a dict of multiples estimators. Therefore the evaluation of each 
            estimator is set to dictionnary where the key is each estimator 
            name. 
     
        eval_metric: bool, 
            if set to ``True``, confusion matrix and classification report scores
            are evaluated assuming the the supervised learning is a classification
            problem. *default* is ``False``. 
            
        scorer: str, Callable, 
            a scorer is a metric  function for model evaluation. If given as 
            string it should be the prefix of the following metrics: 
                
                * "classification_report"     -> for classification_report,
                * 'precision_recall'          -> for precision_recall_curve,
                * "confusion_matrix"          -> for a confusion_matrix,
                * 'precision'                 -> for  precision_score,
                * "accuracy"                  -> for  accuracy_score
                * "mse"                       -> for mean_squared_error, 
                * "recall"                    -> for  recall_score, 
                * 'auc'                       -> for  roc_auc_score, 
                * 'roc'                       -> for  roc_curve 
                * 'f1'                        -> for f1_score,
                
            Other string prefix values should raises an errors 
            
        kws: dict, 
            Additionnal keywords arguments from scklearn metric function.
            
        Returns 
        ----------
        `self.base_score_` : base score after predicting 
        
        Notes
        ------
        If ``None`` estimator is given, the *default* estimator is `svm`
        otherwise, provide the  prefix to select  the convenience estimator 
        into the  default dict `default_estimator`. Get the default dict by 
        calling `<instance>._getdestimators()>`

        Examples 
        ---------
        >>> from watex.cases.processing import Preprocessing 
        >>> pc = Preprocessing (tname = 'flow', drop_features =['lwi', 'name', 'num'])
        >>> data = fetch_data ('bagoue original').get('data=dfy2')
        >>> pc.fit(data =data)
        
        (1) -> default estimator 
        >>> pc.baseEvaluation (eval_metric=True)
        ... 0.47126436781609193
        
        (2) -> multiples estimators 
        >>> from sklearn.ensemble import RandomForestClassifier 
        >>> from sklearn.linear_model import SGDClassifier
        >>> from slearn.imputer import SimpleImputer 
        >>> estimators={'RandomForestClassifier':RandomForestClassifier
                        (n_estimators=200, random_state=0), 
                        'SDGC':SGDClassifier(random_state=0)}
        >>> pc.X= SimpleImputer().fit_transform(pc.X)
        >>> pc.Xt= SimpleImputer().fit_transform(pc.Xt) # remove NaN values 
        >>> pc.BaseEvaluation(estimator={
        ...    'RandomForestClassifier':RandomForestClassifier(
        ...        n_estimators=200, random_state=0), 
        ...    'SDGC':SGDClassifier(random_state=0)}, eval_metric =True)
        >>> pc.score
        ... 
        
        """
        self.inspect 
        self.model_results_ ={} 
        if model is not  None : 
            self.model_= model 
        elif self.model_ is None: 
            self.model_ = self.makeModel() 
            
        # ---> run model for prediction 
        if hasattr (self.model_, '__dict__') and hasattr(
                self.model_, '__class__'): 
            self.ypred_, self.base_score_ = evalModel(
                model=self.model_, X=self.X, y=self.y,  Xt=self.Xt, yt=self.yt,
                eval = eval_metric, **kws)
            
            self.model_results_[f'{self.model_.__class__.__name__}']= (
                self.base_score_ , self.ypred_ )

            return self.base_score_ 
        
        if isinstance(self.model_, dict): 
            print(self.model_)
            # when mutiples estimators are given 
            for est in list(self.model_.values())  : 
                psc, msc= evalModel(
                    model = est, X= self.X , y=self.y,  Xt=self.Xt, yt=self.yt, 
                    eval= eval_metric, **kws)
                self.model_results_  [f'{est.__class__.__name__}'] =  (psc, msc)
                
            self.base_score_ ={
                k: s for k, (s, _) in self.model_results_.items() 
                              }  
            self.ypred_ ={
                k: s for k, (_, s) in self.model_results_.items() 
                              }  
                                    
        return self.base_score_  

    def _getdestimators (self): 
        """ Load default estimator fit default arguments and returns a dict 
        of each default estimator with default hyperparameters already set."""
        ens={}
        d= dict ( 
            knn= dict(
                n_neighbors=10,  metric='manhattan'), 
            svc = dict (
                C=100, gamma=1e-3, random_state=self.random_state), 
            dtc = dict(
                max_depth=100,random_state=self.random_state), 
            rdf = dict(
                n_estimators=200, random_state=self.random_state), 
        
            bag = dict (base_estimator=KNeighborsClassifier(), 
                        n_estimators=100), 
            sdg = dict(random_state=self.random_state)
            )
        
        for key in ('vtc', 'stc'): 
            d[key] = dict (estimators = [
                ('sdg', SGDClassifier(
                 random_state=self.random_state)),
                ('dtc', DecisionTreeClassifier(
                    max_depth=100, 
                    random_state=self.random_state)), 
                ('knn', KNeighborsClassifier())
                ]
            )
        
        for key , func in d_estimators_.items () :
            if key not in d.keys() :
                ens [key] = func (** dict(random_state=self.random_state)) 
            
            else : ens[key] = func (**d[key])
            
        return ens 
     
    def __repr__(self):
       """ Pretty format for programmer guidance following the API... """
       return repr_callable_obj  (self, skip = ('data', 'y', 'X', 'Xt', 'yt') )
       
    def __getattr__(self, name):
        if name.endswith ('_'): 
            if name not in self.__dict__.keys(): 
                if name in ('data_', 'X_', 'y_'): 
                    raise NotFittedError (
                        f'Fit the {self.__class__.__name__!r} object first'
                        )
                
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        appender  = "" if rv is None else f'. Do you mean {rv!r}'
        
        raise AttributeError (
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{appender}{"" if rv is None else "?"}'
            )        

Preprocessing.__doc__="""\
Base preprocessing class. 

Give a baseline preprocessing model with a base score. Usefull before fidlling 
the model hyperparameters. 

Parameters 
-------------
{params.core.tname}

drop_features: list or str, Optional
    List the useless `features` for predicting or list of column names to drop 
    out. 
random_state: int, default is ``42``
    The state of data shuffling. The default is ``42``.
    
default_estimator: callable, F or sckitlearn estimator 
    The default estimator name for predicting the tname value. A predifined 
    defaults estimators prameters are set and keep in cache for quick 
    preprocessing like: 
    - 'dtc': For DecisionTreeClassifier 
    - 'svc': Support Vector Classifier 
    - 'sdg': SGDClassifier
    - 'knn': KNeighborsClassifier
    - 'rdf`: RandmForestClassifier 
    - 'ada': AdaBoostClassifier 
    - 'vtc': VotingClassifier
    - 'bag': BaggingClassifier 
    - 'stc': StackingClassifier
    If estimator is not given the default is ``svm`` or 
                                    ``svc``.
test_size: float,       
    The test set data size. Must be less than 1.The sample test size is 
    ``0.2`` either 20% of dataset.      

{params.core.verbose} 

Attributes
-----------
{params.core.X}
{params.core.y}
{params.core.Xt}
{params.core.yt}
{params.core.data}
{params.base.pipe_}
{params.base.estimator_}
{params.core.model}

cat_features_: list or str, Optional
     list of categorical features list. If not given it should be find 
     automatically.           
num_features_ : list of str, Optional
     list Numerical features list. If not given, should be find automatically. 
     
model: Callable, {{preprocessor + estimator }},
    Use the predifined pipelines i.e can be a Pipeline can your build 
    by your own pipeline with different composite estimator.
    If `model` is ``None`` , use the default model from the default 
    `preprocessor` and `estimator`. 

Examples
--------- 
>>> from sklearn.ensemble import RandomForestClassifier 
>>> from sklearn.linear_model import SGDClassifier
>>> from sklearn.impute import SimpleImputer 
>>> estimators=dict(
...    RandomForestClassifier=RandomForestClassifier(
...        n_estimators=200, random_state=0), 
>>> pc.X= SimpleImputer().fit_transform(pc.X)
>>> pc.Xt= SimpleImputer().fit_transform(pc.Xt) # remove NaN values 
>>> pc.baseEvaluation(estimator=estimators, eval_metric =True)
>>> pc.base_score_
... 0.72586369
""".format(
    params=_param_docs,
)  
    
   
class Processing (Preprocessing) : 
    def __init__(self, 
                 pipeline:F=None, 
                 estimator:F= None, 
                 auto:bool = False,  
                 **kws
                 ):
        super().__init__(**kws)
        
        self.pipeline=pipeline 
        self.auto_= auto
        self.estimator_=estimator
        self.model_score_=None 
        self.model_prediction_=None 
        self.estimator_name_=None 
        self.processing_model_=None
        
        if self.auto_:
            self.auto = True 
            
  
    @property 
    def auto (self): 
        """ Trigger the composite pipeline building and greate 
        a composite default model estimator `CE-SVC` """
        return self.auto_ 
    
    @auto.setter 
    def auto (self, auto): 
        """ Trigger the `CE-SVC` buiLding using default parameters with 
        default pipeline."""
        if not auto: return 
    
        format_notes(text= ''.join(
            [f'Automatic Option is set to ``{self.auto_}``.Composite',
            '  estimator building is auto-triggered with default ',
            'pipeline. The default estimation score should be displayed.',
            '  ']), 
            cover_str='*',inline = 70, margin_space = 0.05)
    
        self._logging.info(
            ' Automatic Option to design a default composite estimator'
            f' is triggered <`{self.auto_}``> with default pipeline.')
        warnings.warn(
            ' Automatic Option to design a composite estimator is '
            f' triggered <`auto={self.auto_}``> with default pipeline '
            'construction. The default estimation score should be '
            ' displayed.')
        
        self.model_score_ = self.baseEvaluation(eval_metric=True)
        self.preprocessor_ = self.pipe_ 
        formatModelScore(self.model_score_, self.default_estimator)
        self.model_prediction_ = self.ypred_
    
    @property 
    def processing_model(self): 
        """ Get the default composite model """
        return self.processing_model_ 
    
    @property 
    def preprocessor (self): 
        """ Preoprocessor for `composite_estimator` design """
        return self.preprocesor_ 
    
    def _validate_estimator (self, e):
        """ Assert whether estimator is valid refering to scikit-learn "
        conventions"""
        msg = ( ":https://scikit-learn.org/stable/developers/develop.html &&"
            "https://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline"
            )
        try : 
            from sklearn.utils.estimator_checks import check_estimator
            check_estimator(e )
        except: 
            if not  ( hasattr(e, '__dict__') and hasattr(
                       e, '__class__') ):
                warnings.warn("'estimator does not adhere to sckit-learn conventions."
                    f" Refer to {msg!r} for more guidelines.")
                raise ProcessingError(f"wrong estimator. Refer to {msg}"
                                      " for furher details.")
        return True 
    
    @preprocessor.setter 
    def preprocessor(self, pipe): 
        """ Create your preprocessor. If `preprocess` is given, it must be
        the collection of transformer and encoders which composed of
        the pipeline like:: 
            
            my_own_pipelines= {'num_column_selector_': make_column_selector(
                                        dtype_include=np.number),
            'cat_column_selector_': make_column_selector(
                                        dtype_exclude=np.number),
            'features_engineering_':PolynomialFeatures(3,
                                        include_bias=True),
            'selectors_': SelectKBest(f_classif, k=4), 
             'encodages_': StandardScaler()
                         }
        """
        self._validate_estimator(pipe) 
        self.preprocesor_= pipe  
    
    @property 
    def model (self):
        """ Concatenate preprocessor and estimator to var"""
        if self.model_ is None: 
            self.model_ = self.makeModel(
                pipe= self.preprocesor_, estimator=self.estimator_)
            
        return self.model_ 
    
    @property 
    def estimator (self): 
        """ Get your estimator of  the existing default estimator """
        return self.estimator_ 
    
    @estimator.setter 
    def estimator (self, e): 
        """ Set estimator value. If string value is given, it is considered 
        as the default estimator is expected. Raise and error is not found."""
        msg=("A string value assumes to be a default estimator prefix.")
        
        if isinstance (e, str): 
            if e not in d_estimators_.keys(): 
                raise EstimatorError( msg + 
                    f"Expect {e!r} being in {smart_format(d_estimators_.keys())}"
                    )
            e = self._getdestimators()[e]
        elif isinstance(e, dict): 
            # estimator is a dict or many estimators 
            # check wether each given values much scikit 
            # conventions estimators 
            self.estimator_name_ = [
                f'{es.__class__.__name__}' for es in e.values()
                ]
        else : 
            self._validate_estimator(e)
            
        if self.estimator_name_ is None: 
            self.estimator_name_ = self.estimator_.__class__.__name__  
        
    
    @property 
    def model_score(self): 
        """ Get the composite estimator score """
        self.model_score_ = self.baseEvaluation(
            self.model , eval_metric=True )
        self.model_prediction_ = self.ypred_
        
        try : 
            formatModelScore(self.model_score_, self.estimator_name_)
        except: 
            self._logging.debug(
                f'{self.estimator_name_ !r} name not found')
            warnings.warn(
                f'Unable to find esimator {self.estimator_name_!r} name')
            
        return self.model_score_ 
    
    @model_score.setter 
    def model_score (self, print_score): 
        """ Display score value """
        if isinstance(print_score, str): 
            self.estimator_name_ = print_score 
        try : 
            self.estimator_name_ = self.estimator_.__class__.__name__
        except : 
            self.estimator_name_ = print_score
        # hints.formatModelScore(self.model_score_, self.estimator_name_)
        
    @property 
    def model_prediction(self):
        """ Get the model prediction after composite estimator designed"""
        return self.model_prediction_ 
        
    @visualize_valearn_curve(reason ='valcurve', turn='off', 
               k= np.arange(1,210,10),plot_style='line',savefig=None)               
    def get_validation_curve(self, estimator=None, X=None, 
                         y=None, val_curve_kws:Generic[T]=None, 
                         **kws):
        """ Compute the validation score and plot the validation curve if 
        the argument `turn` of decorator is switched to ``on``. If 
        validation keywords arguments `val_curve_kws` does not contain a 
        `param_range` key, the default param_range should be the one of 
            decorator.
        
        :param model: The creating model. If ``None``.
        
        :param X: pd.core.frame.DataFrame  of selected trainset.
        
        :param Xt:  pd.DataFrame of  selected Data for testset.
        
        :param y: array_like of selected data for evaluation set. 
        
        :param yt: array_like of selected data for model test 
        
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
            
        >>> from watex.cases.processing  import Processing 
        >>> processObj = Processing(
            data = 'data/geo_fdata/BagoueDataset2.xlsx')
        >>> processObj.get_validation_curve(
            switch_plot='on', preprocess_step=True)
        """
        
        preprocess_step =kws.pop('preprocess_step', False)
        switch = kws.pop('switch_plot', 'off')
        val_kws = kws.pop('val_kws', None)
        train_kws = kws.pop('train_kws', None)
        
        if X is not None:
            self.X =X
        if y is not None:
            self.y =y 
        
        if val_curve_kws is None:
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
                if self.verbose :
                    print('---> Preprocessing step is enabled !')
                    self._logging.info(
                        'By default, the`preprocessing_step` is activated.')
                self.auto =True 
            else: 
                warnings.warn("Expect one 'estimator' at least")
                self._logging.error("Expect one 'estimator' at least")
                raise ProcessingError( "'Estimator' not found. Expect one "
                                      "'estimator' at least")
    
        if estimator is not None :
            self.estimator_= estimator
    
            if not isinstance(self.estimator_, dict) : 
                self.model_dict={'___':self.estimator_ }
            else : 
                self.model_dict = self.estimator_
                
        for mkey , mvalue in self.model_dict.items(): 
            if len(self.model_dict) ==1:
                self.train_score, self.val_score = validation_curve(
                                        self.estimator_,
                                        self.X, self.y,
                                        **val_curve_kws)
                
            elif len(self.model_dict) > 1 :
                trainScore, valScore = validation_curve(mvalue,
                                       self.X, self.y,
                                       **val_curve_kws)
                self.train_score [mkey] = trainScore
                self.val_score[mkey] = valScore 
        try:
            pname = val_curve_kws['param_name']
        except KeyError: 
            pname =''
        except : 
            pname =''
        
        return (self.train_score, self.val_score, switch ,
                kk , pname,  val_kws, train_kws)     
    
        
    def quick_estimation(self, estimator: Callable[...,T] = None, 
                         random_state:float = None, **kws): 
        """ Quick run the model without any processing.  If none estimator 
        is provided ``SVC`` estimator is used.
        
        :param estimators: Callable estimator. If ``None``, a ``svc`` is 
            used to quick estimate prediction. 
                            
        :param random_state: The state of data shuffling.The default is ``7``.
                                        
        :Example: 
        >>> from watex.cases.processing import Processing 
        >>> processObj = Processing(
            data = 'data/geo_fdata/BagoueDataset2.xlsx')
        >>> processObj.quick_estimation(estimator=DecisionTreeClassifier(
            max_depth=100, random_state=13)
        >>> processObj.model_score
        >>> processObj.model_prediction
        
        """
        
        X =kws.pop('X', None )
        if X is not None : self.X =X 
        
        y =kws.pop('y', None)
        if y is not None : self.y =y 
        
        Xt =kws.pop('Xt', None)
        if Xt is not None: self.Xt = Xt 
        
        yt = kws.pop('yt', None)
        if yt is not None : self.yt 
        
        if random_state is not None: self.random_state = random_state 
        
        if estimator is not None: 
            quick_estimator = estimator 
        elif estimator is None : 
            quick_estimator = SVC(self.random_state)
            
        try : 
            self.estimator_name_ = quick_estimator.__class__.__name__
        except : pass 
        else : 
            try : 
                estim_names = controlExistingEstimator(
                    self.estimator_name_)
            except: 
                self.estimator_name_ = '___'
            else : 
                if estim_names is not None :
                    self.estimator_name_ = estim_names[1]
    
        self.estimator_= quick_estimator
     
        self.model_dict= {f'{self.estimator_name_}':self.estimator_ }
        
        self.estimator_.fit(self.X, self.y)
        
        self.model_score_ = self.estimator_.score(
            self.Xt, self.yt)
        self.model_prediction_ = self.estimator_.predict(
            self.Xt)
        
        self.confusion_matrix= confusion_matrix(self.yt,
                                   self.model_prediction_)
        self.classification_report= classification_report(self.yt,
                               self.model_prediction_)
        
        
        return self.model_score_ , self.model_prediction_
                
Processing.__doc__="""\
Processing class for managing baseline model evaluation and learning and 
validation curves after fiddling a little bit an estimator hyperparameters. 

Processing is usefull before modeling step. To process data, a default 
implementation is given for  data `preprocessor` build. It consists of creating 
a model pipeline using different transformers. If None pipeline is setting  
and auto is set to 'True', a default pipeline is created though the 
`prepocessor`to raun the base model evaluation. Indeed  a `preprocessor` is a 
set of `transformers + estimators`.

Parameters 
-------------
auto: bool, default is {{'False'}}
    trigger the composite estimator.If ``True`` a composite  `preprocessor` 
    is built and use for base model evaluation. *default* is False.
pipeline: Callable, F or  dict of callable F            
   preprocessing steps encapsulated. If not supplied a default pipe is 
   used as auto is set to ``True``.   
     
estimator: Callable, 
    An object which manages the estimation and decoding of a model. Estimators 
    must provide a fit method, and should provide set_params and `get_params`, 
    although these are usually provided by inheritance from `base.BaseEstimator`.
    The core functionality of some estimators may also be available as a function.
       
{params.core.tname}

drop_features: list or str, Optional
    List the useless `features` for predicting or list of column names to drop 
    out. 
random_state: int, default is ``42``
    The state of data shuffling. The default is ``42``.
    
default_estimator: callable, F or sckitlearn estimator 
    The default estimator name for predicting the tname value. A predifined 
    defaults estimators prameters are set and keep in cache for quick 
    preprocessing like: 
    - 'dtc': For DecisionTreeClassifier 
    - 'svc': Support Vector Classifier 
    - 'sdg': SGDClassifier
    - 'knn': KNeighborsClassifier
    - 'rdf`: RandmForestClassifier 
    - 'ada': AdaBoostClassifier 
    - 'vtc': VotingClassifier
    - 'bag': BaggingClassifier 
    - 'stc': StackingClassifier
    If estimator is not given the default is ``svm`` or ``svc``.
test_size: float,       
    The test set data size. Must be less than 1.The sample test size is 
    ``0.2`` either 20% of dataset.      

{params.core.verbose} 

Attributes
-----------
{params.core.X}
{params.core.y}
{params.core.Xt}
{params.core.yt}
{params.core.data}
{params.base.pipe_}
{params.base.estimator_}
{params.core.model}

cat_features_: list or str, Optional
     list of categorical features list. If not given it should be find 
     automatically.           
num_features_ : list of str, Optional
     list Numerical features list. If not given, should be find automatically. 
     
model: Callable, {{preprocessor + estimator }},
    Use the predifined pipelines i.e can be a Pipeline can your build 
    by your own pipeline with different composite estimator.
    If `model` is ``None`` , use the default model from the default 
    `preprocessor` and `estimator`. 
                                     
model_score_:  float/dict      
    Model test score. Observe your test model score using your compose 
    estimator for enhacement 
model_prediction_: array_like      
    Observe your test model prediction for as well as the compose estimator 
    enhancement.
preprocessor_: Callable , F       
    Compose piplenes and estimators for default model scorage.


See also
---------
.. [1] https://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline
.. [2] https://scikit-learn.org/stable/modules/classes.html#module-sklearn.compose


Examples 
---------
>>> from watex.cases.processing  import Processing
>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.ensemble import RandomForestClassifier 
>>> my_own_pipeline= {{'num_column_selector_': 
...                       make_column_selector(dtype_include=np.number),
...                'cat_column_selector_': 
...                    make_column_selector(dtype_exclude=np.number),
...                'features_engineering_':
...                    PolynomialFeatures(3,include_bias=True),
...                'selectors_': SelectKBest(f_classif, k=4), 
...               'encodages_': StandardScaler()
...                 }}
>>> my_estimator={{
...    'RandomForestClassifier':RandomForestClassifier(
...    n_estimators=200, random_state=0)
...    }}
>>> processObj = Processing(
...                    data ='data/geo_fdata/BagoueDataset2.xlsx', 
...                    pipeline= my_own_pipeline,
...                    estimator=my_estimator)
>>> print(processObj.preprocessor_)
>>> print(processObj.estimator_)
>>> print(processObj.model_score_)
>>> print(processObj.model_prediction_)
    
""".format(
    params=_param_docs,
)      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    