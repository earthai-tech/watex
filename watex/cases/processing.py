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

from ..exlib.sklearn  import ( 
    DecisionTreeClassifier, 
    KNeighborsClassifier, 
    OneHotEncoder, 
    SelectKBest,  
    SGDClassifier,
    SVC, 
    PolynomialFeatures, 
    RobustScaler, 
    make_column_selector,
    ColumnTransformer,
    # make_pipeline, 
    confusion_matrix , 
    classification_report, 
    f_classif, 
    # make_column_transformer , 
    train_test_split, 
    validation_curve ,
    SimpleImputer,
    Pipeline,
    # FeatureUnion,
    _HAS_ENSEMBLE_,   
) 
from .._docstring import ( 
    DocstringComponents, _core_docs
    )
from .._watexlog import watexlog 
from ..tools.mlutils import (
    _estimators,
    formatModelScore, 
    controlExistingEstimator,
    formatGenericObj,
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
from ..exceptions import ( 
    FeatureError , 
    NotFittedError , 
    ProcessingError, 
    EstimatorError
    )
import  watex.decorators as deco
from ..tools.funcutils import ( 
    format_notes, 
    repr_callable_obj, 
    smart_strobj_recognition 
    
    )
from ..tools.coreutils import ( 
    _is_readable , 
    _assert_all_types
    )

_logger =watexlog().get_watex_logger(__name__)

d_estimators_={'dtc':DecisionTreeClassifier, 
                'svc':SVC, 
                'sgd':SGDClassifier, 
                'knn':KNeighborsClassifier 
                 }
if _HAS_ENSEMBLE_ :
    from ..exlib import skl_ensemble_
    
    for es_, esf_ in zip(['rdf', 'ada', 'vtc', 'bag','stc'], skl_ensemble_): 
        d_estimators_[es_]=esf_ 

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
        split_xy: bool, default {'True'}
            split the datatset to training set {X, y } and test set {Xt, yt}. 
            Otherwise `X` and `y` should be considered as traning sets.  
            
        Returns 
        --------
        ``self``: `Preprocessing` instance for easy method chaining.
        
        Examples
        ---------
        >>> from watex.bases.processing import Preprocessing 
        >>> from watex.datasets import fetch_data 
        >>> data = fetch_data('bagoue original').get('data=dfy2')
        >>> pc = Preprocessing (drop_features = ['lwi', 'num', 'name']
                                ).fit(data =data )
        >>> len(pc.X ),  len(y), len(pc.Xt ),  len(pc.yt)
        ... (344, 344, 87, 87) # trainset (X,y) and testset (Xt, yt)

        """
        data = fit_params.pop('data', None)
        split_Xy= fit_params.pop('split_Xy', True)
        
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
        if split_Xy: 
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
                
                 >>> from watex.bases.processing import Preprocessing 
                 >>> Preprocessing._getdestimators()['dtc']
                 ... DecisionTreeClassifier(max_depth=100, random_state=42)
        
        Returns 
        ---------
        `model_`: Callable, {preprocessor + estimator } 
        
        Examples 
        ----------
        (1) We can get the default preprocessor by merely calling: 

        >>> from watex.bases.processing import Preprocessing 
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
        >>> from watex.bases.processing import Preprocessing 
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
            
        Notes
        ------
        If ``None`` estimator is given, the *default* estimator is `svm`
        otherwise, provide the  prefix to select  the convenience estimator 
        into the  default dict `default_estimator` keywords argument like::
            
            >>> from watex.bases.processing import Preprocessing
            >>> from watex.datasets import fetch_data 
            >>> data = fetch_data ('bagoue original').get('data=dfy2')
            >>> preObj = Preprocessing()
            >>> preObj.tname = 'flow' # specify the target name 
            >>> preObj.random_state = 7
            >>> preObj.fit(data =data )
            >>> preObj.basemodel ()
            >>> preObj.base_score_ 
            >>> preObj.confusion_matrix_
            >>> preObj.classification_report_
        
        Providing multiple estimator is possible like the example below. 
            
        Examples 
        ---------
        >>> from watex.bases.processing import Preprocessing 
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
>>> from watex.bases.processing import Preprocessing
>>> prepObj = Preprocessing(dropfeatures_ = ['lwi', 'x_m', 'y_m'],
...    data ='data/geo_fdata/BagoueDataset2.xlsx')
>>> prepObj.X, prepObj.Xt, prepObj.y, prepObj.yt
>>> prepObj.cat_features_, prepObj.num_features_ 
>>> prepObj.random_state = 25 
>>> preObj.test_size = 0.25
>>> prepObj.makepipe_()         # use default preprocessing
>>> preObj.preprocessor
>>> prepObj.make_preprocessing_model( default_estimator='SVM')
>>> prepObj.preprocessing_model_score
>>> prepObj.preprocess_model_prediction
>>> prepObj.confusion_matrix
>>> prepObj.classification_report

""".format(
    params=_param_docs,
)  
    
   
class Processing (Preprocessing) : 
    """
    Processing class deal with preprocessing. 
    
    Processing is usefull before modeling step. To process data, a default 
    implementation is given for  data `preprocessing` after data sanitizing.
    It consists of creating a model pipeline using different supervised 
    learnings methods. A default pipeline is created though the `prepocessor` 
    designing. Indeed  a `preprocessor` is a set of `transformers + estimators`
    and multiple other functions to boost the prediction. 
    
    Arguments 
    -----------
    *dataf_fn*: str 
        Path to analysis data file. 
    *df*: pd.Core.DataFrame 
            Dataframe of features for analysis . Must be contains of 
            main parameters including the `tname` pd.Core.series 
            as columns of `df`. 
 
    Holds on others optionals infos in ``kwargs`` arguments: 

    ==================    ============      ===================================
    Attributes              Type                Description  
    ==================    ============      ===================================
    auto                    bool            Trigger the composite estimator.
                                            If ``True`` a SVC-composite  
                                            estimator `preprocessor` is given. 
                                            *default* is False.
    pipelines               dict            Collect your own pipeline for model 
                                            preprocessor trigging.
                                            it should be find automatically.           
    estimators              Callable        A given estimator. If ``None``,`SVM`
                                            is auto-selected as default estimator.
    model_score             float/dict      Model test score. Observe your test 
                                            model score using your compose  
                                            estimator for enhacement 
    model_prediction        array_like      Observe your test model prediction for 
                                            as well as the compose estimator 
                                            enhancement.
    preprocessor            Callable        Compose piplenes and estimators for 
                                            default model scorage.
    ==================    ============      ===================================  
    
    
    Examples 
    ---------
    >>> from watex.bases.processing  import Processing
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.ensemble import RandomForestClassifier 
    >>> my_own_pipeline= {'num_column_selector_': 
    ...                       make_column_selector(dtype_include=np.number),
    ...                'cat_column_selector_': 
    ...                    make_column_selector(dtype_exclude=np.number),
    ...                'features_engineering_':
    ...                    PolynomialFeatures(3,include_bias=True),
    ...                'selectors_': SelectKBest(f_classif, k=4), 
    ...               'encodages_': StandardScaler()
    ...                 }
    >>> my_estimator={
    ...    'RandomForestClassifier':RandomForestClassifier(
    ...    n_estimators=200, random_state=0)
    ...    }
    >>> processObj = Processing(
    ...                    data ='data/geo_fdata/BagoueDataset2.xlsx', 
    ...                    pipeline= my_own_pipeline,
    ...                    estimator=my_estimator)
    >>> print(processObj.preprocessor)
    >>> print(processObj.estimator)
    >>> print(processObj.model_score)
    >>> print(processObj.model_prediction)
    
    """  
    
    def __init__(self, data = None, df=None , **kws):
        Preprocessing.__init__(self,  data , df, **kws)
             
        self.pipelines =kws.pop('pipelines', None)
        
        self._auto =kws.pop('auto', False)
        self.select_estimator_ = kws.pop('estimator', None)

        if self._auto:
            self.auto = True 
             
        self._model_score =None 
        self._model_prediction =None 
        self._estimator_name =None 
        self._processing_model =None
    
        if self.pipelines is not None:
            self.preprocessor = self.pipelines 
            
        for key in list(kws.keys()): 
            setattr(self, key, kws[key]) 
            
    @property 
    def auto (self): 
        """ Trigger the composite pipeline building and greate 
        a composite default model estimator `CE-SVC` """
        return self._auto 
    
    @auto.setter 
    def auto (self, auto): 
        """ Trigger the `CE-SVC` buiLding using default parameters with 
        default pipelines."""
        if auto: 
            format_notes(text= ''.join(
                [f'Automatic Option is set to ``{self._auto}``.Composite',
                '  estimator building is auto-triggered with default ',
                'pipelines construction.The default estimation score ',
                '  should be displayed.']), 
                cover_str='*',inline = 70, margin_space = 0.05)

            self._logging.info(
                ' Automatic Option to design a default composite estimator'
                f' is triggered <`{self._auto}``> with default pipelines.')
            warnings.warn(
                ' Automatic Option to design a composite estimator is '
                f' triggered <`auto={self._auto}``> with default pipelines '
                'construction. The default estimation score should be '
                ' displayed.')
            
            self.make_preprocessing_model()
            self._model_score = self.preprocessing_model_score
            self._processing_model = self.preprocessing_model
            formatModelScore(self._model_score, self.default_estimator)
            self._model_prediction = self.preprocessing_model_prediction
            self._auto =True 
    @property 
    def processing_model(self): 
        """ Get the default composite model """
        return self._processing_model 
    
    @property 
    def preprocessor (self): 
        """ Preoprocessor for `composite_estimator` design """
        return self.pipe_ 
    
    @preprocessor.setter 
    def preprocessor(self, pipelines): 
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
        
        import sklearn 
        
        if pipelines is None: 
            self.pipelines == {}
            self.pipe_ = self.makepipe_()
            
        elif  pipelines is not None:
            self.pipelines = pipelines
            if isinstance(self.pipelines,
                        sklearn.compose._column_transformer.ColumnTransformer): 
                self._preprocesor= pipelines  
            else:
                self._preprocesor = self.makepipe_(
                    **self.pipelines)
        
        self.make_preprocessing_model(preprocessor= self._preprocesor, 
                                      estimators_=self.select_estimator_)
        
        self._processing_model = self.preprocessing_model
        self._model_score = self.preprocessing_model_score
        self._model_prediction = self.preprocessing_model_prediction
        
        self._estimator_name = self.select_estimator_.__class__.__name__
        
    @property 
    def estimator (self): 
        """ Get your estimator of  the existing default estimator """
        return self.select_estimator_ 
    
    @estimator.setter 
    def estimator (self, estim): 
        """ Set estimator value"""
        f_search =False 
        try : 
            self._estimator_name = estim.__class__.__name__
        except : 
            warnings.warn(
                "It'seems the estimator ``{estim}`` is not a Callable ")
            self._logging.error(
                "The given estimator ``{estim}`` is not Callable object")
            f_search =True 
        else :
            self.select_estimator_ = estim 
            
        if f_search is True : 
            # get the list of default estimator full names.
            estfullname = [ e_key[0] for e_key in _estimators.values()]
            
            if isinstance(estim, str): 
                self._logging.debug(
                    f'Estimator name <``{estim}``> is string type. Will search '
                    'in default estimator list  whether its exits !')
                warnings.warn(f'A given estimator ``{estim}`` is string type.'
                              'Will try to search its corresponding in default'
                              'estimators wheter it exists' )
                try : 
                    estim_codecs = controlExistingEstimator(estim)
                except : 
                    warnings(
                        f'Given estimator``{estim}`` does not exist in the '
                        '  list of default estimators {}.'.format(
                            formatGenericObj(
                            estfullname)).format(*estfullname))
                else: 
                    if estim_codecs is None: 
                        raise EstimatorError (
                            f' Estimator `{estim}` not found! Please provide'
                            ' the estimator as Callable or class object.')
                    if len(estim_codecs) ==2: 
                        self.select_estimator_= d_estimators_[
                            estim_codecs[0]]
                        self._estimator_name = estim_codecs[0]
    @property 
    def model_score(self): 
        """ Get the composite estimator score """
        try : 
            formatModelScore(self._model_score, self._estimator_name)
        except: 
            self._logging.debug(f'Error finding the {self._estimator_name}')
            warnings.warn(f'Error finding the {self._estimator_name}')
            
        return self._model_score 
    
    @model_score.setter 
    def model_score (self, print_score): 
        """ Display score value """
        if isinstance(print_score, str): 
            self._estimator_name = print_score 
        try : 
            self._estimator_name = self.select_estimator_.__class__.__name__
        except : 
            self._estimator_name = print_score
        
        # hints.formatModelScore(self._model_score, self._estimator_name)
        
    @property 
    def model_prediction(self):
        """ Get the model prediction after composite estimator design"""
        return self._model_prediction 
        
    @deco.visualize_valearn_curve(reason ='valcurve', turn='off', 
               k= np.arange(1,210,10),plot_style='line',savefig=None)               
    def get_validation_curve(self, estimator=None, X=None, 
                         y=None, val_curve_kws:Generic[T]=None, 
                         **kws):
        """ Compute the validation score and plot the validation curve if 
        the argument `turn` of decorator is switched to ``on``. If 
        validation keywords arguments `val_curve_kws` doest not contain a 
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
            >>> from watex.bases.processing  import Processing 
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
                print('---> Preprocessing step is enabled !')
                self._logging.info(
                    'By default, the`preprocessing_step` is activated.')
                self.auto =True 
            else: 
                warnings.warn('At least one `estimator` must be supplied!')
                self._logging.error(
                    'Need a least a one `estimator` but NoneType is found.')
                raise ProcessingError(
                    'None `estimator` detected. Please provide at least'
                    ' One `estimator`.')

        if estimator is not None :
            self.select_estimator_= estimator

            if not isinstance(self.select_estimator_, dict) : 
                self.model_dict={'___':self.select_estimator_ }
            else : 
                self.model_dict = self.select_estimator_
                
        for mkey , mvalue in self.model_dict.items(): 
            if len(self.model_dict) ==1:
                self.train_score, self.val_score = validation_curve(
                                        self.select_estimator_,
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
            >>> from watex.bases.processing import Processing 
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
            self._estimator_name = quick_estimator.__class__.__name__
        except : pass 
        else : 
            try : 
                estim_names = controlExistingEstimator(
                    self._estimator_name)
            except: 
                self._estimator_name = '___'
            else : 
                if estim_names is not None :
                    self._estimator_name = estim_names[1]

        self.select_estimator_= quick_estimator
     
        self.model_dict= {f'{self._estimator_name}':self.select_estimator_ }
        
        self.select_estimator_.fit(self.X, self.y)
        
        self._model_score = self.select_estimator_.score(
            self.Xt, self.yt)
        self._model_prediction = self.select_estimator_.predict(
            self.Xt)
        
        self.confusion_matrix= confusion_matrix(self.yt,
                                   self._model_prediction)
        self.classification_report= classification_report(self.yt,
                               self._model_prediction)
        
        
        return self._model_score , self._model_prediction
                

if __name__=='__main__': 

    from sklearn.preprocessing import StandardScaler
    # from sklearn.ensemble import RandomForestClassifier 
    my_own_pipelines= {'num_column_selector_': make_column_selector(
                        dtype_include=np.number),
                        'cat_column_selector_': make_column_selector(
                            dtype_exclude=np.number),
                        'features_engineering_':PolynomialFeatures(
                            3,include_bias=True),
                        'selectors_': SelectKBest(f_classif, k=4), 
                        'encodages_': StandardScaler()
                          }
    # estimators={
    #         'RandomForestClassifier':RandomForestClassifier(
    #         n_estimators=200, random_state=0)
    #         }
    # processObj = Processing(data ='data/geo_fdata/BagoueDataset2.xlsx', 
                            # pipelines= my_own_pipelines,
                            # estimator=estimators
                            # )
    # processObj.get_validation_curve(switch_plot ='on', preprocess_step='True')
    
    # print(processObj.estimator)
    # print(processObj.model_score)
    # print(processObj.model_prediction)
    

    # processObj.model_score
    # processObj.model_prediction
    # from watex.processing.sl import Preprocessing
    # from sklearn.preprocessing import StandardScaler 
    preObj = Preprocessing(
        data ='data/geo_fdata/BagoueDataset2.xlsx',
            )

    preObj.random_state = 23
    preObj.makepipe_(**my_own_pipelines)
    # num_column_selector_= make_column_selector(
    #                         dtype_include=np.number),
    # cat_column_selector_= make_column_selector(
    #                         dtype_exclude=np.number),
    # features_engineering_=PolynomialFeatures(7,
    #                         include_bias=True),
    # selectors_=SelectKBest(f_classif, k=4), 
    #     encodages_= StandardScaler())
    print(preObj.pipe_)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    