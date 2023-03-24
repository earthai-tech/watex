# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Created on Sat Sep 25 10:10:31 2022

from __future__ import annotations 

import inspect
import warnings  
from pprint import pprint 
import numpy as np 

from .._docstring import ( 
    DocstringComponents, 
    _core_docs 
    ) 
from .._watexlog import watexlog
from ..exlib.sklearn import (
     mean_squared_error,
     cross_val_score,
     GridSearchCV , 
     RandomizedSearchCV, 
     LogisticRegression, 
     Pipeline,
)
from .._typing import (
    List,
    Tuple,
    F, 
    ArrayLike, 
    NDArray, 
    Dict,
    Any, 
    DataFrame, 
    Series,
    
    )
from ..exceptions import ( 
    EstimatorError, 
    NotFittedError, 
    ) 
from ..utils.funcutils import ( 
    _assert_all_types, 
    get_params, 
    savejob, 
    listing_items_format, 
    pretty_printer, 

    )
from ..utils.box import Boxspace 
from ..utils.validator import ( 
    check_X_y, check_array, 
    check_consistent_length, 
    get_estimator_name
    )

_logger = watexlog().get_watex_logger(__name__)

__all__=[
    "BaseEvaluation", 
    "GridSearch", 
    "GridSearchMultiple", 
    "get_best_kPCA_params", 
    "get_scorers", 
    "getGlobalScores", 
    "getSplitBestScores", 
    "displayCVTables", 
    "displayFineTunedResults", 
    "displayModelMaxDetails"
    ]

_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"], 
    )

class GridSearch: 
    __slots__=(
        '_base_estimator',
        'grid_params', 
        'scoring',
        'cv', 
        '_kind', 
        'grid_kws', 
        'best_params_',
        'cv_results_',
        'feature_importances_',
        'best_estimator_',
        'verbose',
        'grid_kws',
        )
    def __init__(
        self,
        base_estimator:F,
        grid_params:Dict[str,Any],
        cv:int =4,
        kind:str ='GridSearchCV',
        scoring:str = 'nmse',
        verbose:int=0, 
        **grid_kws
        ): 
        
        self._base_estimator = base_estimator 
        self.grid_params = grid_params 
        self.scoring = scoring 
        self.cv = cv 
        self.best_params_ =None 
        self.cv_results_= None
        self.feature_importances_= None
        self.grid_kws = grid_kws 
        self._kind = kind 
        self.verbose=verbose

    @property 
    def base_estimator (self): 
        """ Return the base estimator class"""
        return self._base_estimator 
    
    @base_estimator.setter 
    def base_estimator (self, base_est): 
        if not hasattr (base_est, 'fit'): 
            raise EstimatorError(
                f"Wrong estimator {get_estimator_name(base_est)!r}. Each"
                " estimator must have a fit method. Refer to scikit-learn"
                " https://scikit-learn.org/stable/modules/classes.html API"
                " reference to build your own estimator.") 

        self._base_estimator =base_est 
        
    @property 
    def kind(self): 
        """ Kind of searched. `RandomizedSearchCV` or `GridSearchCV`."""
        return self._kind 
    
    @kind.setter 
    def kind (self, ksearch): 
        """`kind attribute checker"""
        if 'gridsearchcv1'.find( str(ksearch).lower())>=0: 
            ksearch = 'GridSearchCV' 
        elif 'randomizedsearchcv2'.find( str(ksearch).lower())>=0:
            ksearch = 'RandomizedSearchCV'
        else: raise ValueError (
            " Unkown the kind of parameter search {ksearch!r}."
            " Supports only 'GridSearchCV' and 'RandomizedSearchCV'.")
        self._kind = ksearch 

    def fit(self,  X, y): 
        """ Fit method using base Estimator and populate gridSearch 
        attributes.
 
        Parameters
        ----------
        X:  Ndarray ( M x N) matrix where ``M=m-samples``, & ``N=n-features``)
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

        Returns
        ----------
        ``self``: `GridSearch`
            Returns :class:`~.GridSearch` 
    
        """
        if callable (self.base_estimator): 
            self.base_estimator= self.base_estimator () 
            parameters = get_params (self.base_estimator.__init__)
            
            if self.verbose > 0: 
                msg = ("Estimator {!r} is cloned with default arguments{!r}"
                       " for cross validation search.".format(
                           get_estimator_name (self.base_estimator), parameters)
                       )
                warnings.warn(msg)
        
        self.kind =self._kind 
        
        if self.kind =='GridSearchCV': 
            searchGridMethod = GridSearchCV 
        elif self.kind=='RandomizedSearchCV': 
            searchGridMethod= RandomizedSearchCV 
            
        if self.scoring in ( 'nmse', None): 
            self.scoring ='neg_mean_squared_error'
        # assert scoring values 
        get_scorers(scorer= self.scoring , check_scorer= True, error ='raise' ) 
         
        gridObj = searchGridMethod(
            self.base_estimator, 
            self.grid_params,
            scoring = self.scoring , 
            cv = self.cv,
            **self.grid_kws
            )
        gridObj.fit(X, y)
        
        #make_introspection(self,  gridObj)
        params = ('best_params_','best_estimator_','cv_results_')
        params_values = [getattr (gridObj , param, None) for param in params] 
        
        for param , param_value in zip(params, params_values ):
            setattr(self, param, param_value)
        # set feature_importances if exists 
        try : 
            attr_value = gridObj.best_estimator_.feature_importances_
        except AttributeError: 
            setattr(self,'feature_importances_', None )
        else : 
            setattr(self,'feature_importances_', attr_value)
 
        return self
    
GridSearch.__doc__="""\
Fine-tune hyperparameters using grid search methods. 

Search Grid will be able to  fiddle with the hyperparameters until to 
      
Parameters 
------------
base_estimator: Callable,
    estimator for trainset and label evaluating; something like a 
    class that implements a fit method. Refer to 
    https://scikit-learn.org/stable/modules/classes.html

grid_params: list of dict, 
    list of hyperparameters params  to be fine-tuned.For instance::
    
        param_grid=[dict(
            kpca__gamma=np.linspace(0.03, 0.05, 10),
            kpca__kernel=["rbf", "sigmoid"]
            )]

pipeline: Callable or :class:`~sklearn.pipeline.Pipeline` object 
    If `pipeline` is given , `X` is transformed accordingly, Otherwise 
    evaluation is made using purely the base estimator with the given `X`. 

prefit: bool, default=False, 
    If ``False``, does not need to compute the cross validation score once 
    again and ``True`` otherwise.
{params.core.cv}
    The default is ``4``.
kind:str, default='GridSearchCV' or '1'
    Kind of grid parameter searches. Can be ``1`` for ``GridSearchCV`` or
    ``2`` for ``RandomizedSearchCV``. 
{params.core.scoring} 
{params.core.random_state}

Examples
-----------
>>> from pprint import pprint 
>>> from watex.datasets import fetch_data 
>>> from watex.models.validation import GridSearch
>>> from watex.exlib.sklearn import RandomForestClassifier
>>> X_prepared, y_prepared =fetch_data ('bagoue prepared')
>>> grid_params = [ dict(
...        n_estimators=[3, 10, 30], max_features=[2, 4, 6, 8]), 
...        dict(bootstrap=[False], n_estimators=[3, 10], 
...                             max_features=[2, 3, 4])
...        ]
>>> forest_clf = RandomForestClassifier()
>>> grid_search = GridSearch(forest_clf, grid_params)
>>> grid_search.fit(X= X_prepared,y =  y_prepared,)
>>> pprint(grid_search.best_params_ )
{{'max_features': 8, 'n_estimators': 30}}
>>> pprint(grid_search.cv_results_)
""".format (params=_param_docs,
)
    
class GridSearchMultiple:
    def __init__ (
        self, 
        estimators: F, 
        scoring:str,  
        grid_params: Dict[str, Any],
        *, 
        kind:str ='GridSearchCV', 
        cv: int =7, 
        random_state:int =42,
        savejob:bool =False,
        filename: str=None, 
        verbose:int =0,
        **grid_kws, 
        ):
        self.estimators = estimators 
        self.scoring=scoring 
        self.grid_params=grid_params
        self.kind=kind 
        self.cv=cv
        self.savejob=savejob
        self.filename=filename 
        self.verbose=verbose 
        self.grid_kws=grid_kws
        
    def fit(
            self, 
            X: NDArray, 
            y:ArrayLike, 
        ):
        """ Fit methods, evaluate each estimator and store models results.
        
        Parameters 
        -----------
        {params.core.X}
        {params.core.y}
        
        Returns 
        --------
        {returns.self}

        """.format( 
            params =_param_docs , 
            returns = _core_docs['returns'] 
        ) 
        err_msg = (" Each estimator must have its corresponding grid params,"
                   " i.e estimators and grid params must have the same length."
                   " Please provide the appropriate arguments.")
        try: 
            check_consistent_length(self.estimators, self.grid_params)
        except ValueError as err : 
            raise ValueError (str(err) +f". {err_msg}")

        self.best_estimators_ =[] 
        self.data_ = {} 
        models_= {}
        msg =''
        
        self.filename = self.filename or '__'.join(
            [get_estimator_name(b) for b in self.estimators ])
        
        for j, estm in enumerate(self.estimators):
            estm_name = get_estimator_name(estm)
            msg = f'{estm_name} is evaluated with {self.kind}.'
            searchObj = GridSearch(base_estimator=estm, 
                                    grid_params= self.grid_params[j], 
                                    cv = self.cv, 
                                    kind=self.kind, 
                                    scoring=self.scoring, 
                                    **self.grid_kws
                                      )
            searchObj.fit(X, y)
            best_model_clf = searchObj.best_estimator_ 
            
            if self.verbose > 7 :
                msg += ( 
                    f"\End {self.kind} search. Set estimator {estm_name!r}"
                    " best parameters, cv_results and other importances" 
                    " attributes\n'"
                 )
            self.data_[estm_name]= {
                                'best_model_':searchObj.best_estimator_ ,
                                'best_params_':searchObj.best_params_ , 
                                'cv_results_': searchObj.cv_results_,
                                'grid_params':self.grid_params[j],
                                'scoring':self.scoring, 
                                "grid_kws": self.grid_kws
                                    }
            
            models_[estm_name] = searchObj
            
            
            msg += ( f"Cross-evaluatation the {estm_name} best model."
                    f" with KFold ={self.cv}"
                   )
            bestim_best_scores, _ = naive_evaluation(
                best_model_clf, 
                X,
                y,
                cv = self.cv, 
                scoring = self.scoring,
                display ='on' if self.verbose > 7 else 'off'
                )
            # store the best scores 
            self.data_[f'{estm_name}']['best_scores']= bestim_best_scores
    
            self.best_estimators_.append((estm, searchObj.best_estimator_,
                          searchObj.best_params_, 
                          bestim_best_scores) 
                        )
            
        # save models into a Box 
        d = {**models_, ** dict( 
            keys_ = list (models_.values() ), 
            values_ = list (models_.values() ), 
            models_= models_, 
            )
            
            }
        self.models= Boxspace(**d) 
        
        if self.verbose:
            msg += ('\Pretty print estimators results using'
                    f' scoring ={self.scoring!r}')
            pretty_printer(clfs=self.best_estimators_, scoring =self.scoring, 
                          clf_scores= None)
        if self.savejob:
            msg += ('\Serialize the dict of fine-tuned '
                    f'parameters to `{self.filename}`.')
            savejob (job= self.data_ , savefile = self.filename )
            _logger.info(f'Dumping models `{self.filename}`!')
            
            if self.verbose: 
                pprint(msg)
                bg = ("Job is successfully saved. Try to fetch your job from "
                       f"{self.filename!r} using")
                lst =[ "{}.load('{}') or ".format('joblib', self.filename ),
                      "{}.load('{}')".format('pickle', self.filename)]
                
                listing_items_format(lst, bg )
    
        if self.verbose:  
            pprint(msg)    

        return self 

GridSearchMultiple.__doc__="""\
Search and find multiples best parameters from differents
estimators.

Parameters
----------
estimators: list of callable obj 
    list of estimator objects to fine-tune their hyperparameters 
    For instance::
        
    random_state=42
    # build estimators
    logreg_clf = LogisticRegression(random_state =random_state)
    linear_svc_clf = LinearSVC(random_state =random_state)
    sgd_clf = SGDClassifier(random_state = random_state)
    svc_clf = SVC(random_state =random_state) 
               )
    estimators =(svc_clf,linear_svc_clf, logreg_clf, sgd_clf )
 
grid_params: list 
    list of parameters Grids. For instance:: 
        
        grid_params= ([
        dict(C=[1e-2, 1e-1, 1, 10, 100], gamma=[5, 2, 1, 1e-1, 1e-2, 1e-3],
                     kernel=['rbf']), 
        dict(kernel=['poly'],degree=[1, 3,5, 7], coef0=[1, 2, 3], 
         'C': [1e-2, 1e-1, 1, 10, 100])], 
        [dict(C=[1e-2, 1e-1, 1, 10, 100], loss=['hinge'])], 
        [dict()], [dict()]
        )
{params.core.cv} 

{params.core.scoring}
   
kind:str, default='GridSearchCV' or '1'
    Kind of grid parameter searches. Can be ``1`` for ``GridSearchCV`` or
    ``2`` for ``RandomizedSearchCV``. 
    
{params.core.random_state} 

savejob: bool, default=False
    Save your model parameters to external file using 'joblib' or Python 
    persistent 'pickle' module. Default sorted to 'joblib' format. 
    
{params.core.verbose} 

grid_kws: dict, 
    Argument passed to `grid_method` additional keywords. 
    
Examples
--------
>>> from watex.models import GridSearchMultiple , displayFineTunedResults
>>> from watex.exlib import LinearSVC, SGDClassifier, SVC, LogisticRegression
>>> X, y  = wx.fetch_data ('bagoue prepared') 
>>> X
... <344x18 sparse matrix of type '<class 'numpy.float64'>'
... with 2752 stored elements in Compressed Sparse Row format>
>>> # As example, we can build 04 estimators and provide their 
>>> # grid parameters range for fine-tuning as ::
>>> random_state=42
>>> logreg_clf = LogisticRegression(random_state =random_state)
>>> linear_svc_clf = LinearSVC(random_state =random_state)
>>> sgd_clf = SGDClassifier(random_state = random_state)
>>> svc_clf = SVC(random_state =random_state) 
>>> estimators =(svc_clf,linear_svc_clf, logreg_clf, sgd_clf )
>>> grid_params= ([dict(C=[1e-2, 1e-1, 1, 10, 100], 
                        gamma=[5, 2, 1, 1e-1, 1e-2, 1e-3],kernel=['rbf']), 
                   dict(kernel=['poly'],degree=[1, 3,5, 7], coef0=[1, 2, 3],
                        C= [1e-2, 1e-1, 1, 10, 100])],
                [dict(C=[1e-2, 1e-1, 1, 10, 100], loss=['hinge'])], 
                [dict()], # we just no provided parameter for demo
                [dict()]
                )
>>> #Now  we can call :class:`watex.models.GridSearchMultiple` for
>>> # training and self-validating as:
>>> gobj = GridSearchMultiple(estimators = estimators, 
                       grid_params = grid_params ,
                       cv =4, 
                       scoring ='accuracy', 
                       verbose =1,   #> 7 put more verbose 
                       savejob=False ,  # set true to save job in binary disk file.
                       kind='GridSearchCV').fit(X, y)
>>> # Once the parameters are fined tuned, we can display the fined tuning 
>>> # results using displayFineTunedResults`` function
>>> displayFineTunedResults (gobj.models.values_) 
MODEL NAME = SVC
BEST PARAM = {{'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}}
BEST ESTIMATOR = SVC(C=100, gamma=0.01, random_state=42)

MODEL NAME = LinearSVC
BEST PARAM = {{'C': 100, 'loss': 'hinge'}}
BEST ESTIMATOR = LinearSVC(C=100, loss='hinge', random_state=42)

MODEL NAME = LogisticRegression
BEST PARAM = {{}}
BEST ESTIMATOR = LogisticRegression(random_state=42)

MODEL NAME = SGDClassifier
BEST PARAM = {{}}
BEST ESTIMATOR = SGDClassifier(random_state=42)

Notes
--------
Call :func:`~.get_scorers` or use `sklearn.metrics.SCORERS.keys()` to get all
the metrics used to evaluate model errors. Can be any others metrics  in 
`~metrics.metrics.SCORERS.keys()`. Furthermore if `scoring` is set to ``None``
``nmse`` is used as default value for 'neg_mean_squared_error'`.
 
""".format (params=_param_docs,
)
    
class BaseEvaluation: 
    def __init__(
        self, 
        estimator: F,
        cv: int = 4,  
        pipeline: List[F]= None, 
        prefit:bool=False, 
        scoring: str ='nmse',
        random_state: int=42, 
        verbose: int=0, 
        ): 
        self._logging =watexlog().get_watex_logger(self.__class__.__name__)
        
        self.estimator = estimator
        self.cv = cv 
        self.pipeline =pipeline
        self.prefit =prefit 
        self.scoring = scoring
        self.random_state=random_state
        self.verbose=verbose 

    def _check_callable_estimator (self, base_est ): 
        """ Check wether the estimator is callable or not.
        
        If callable use the default parameter for initialization. 
        """
        if not hasattr (base_est, 'fit'): 
            raise EstimatorError(
                f"Wrong estimator {get_estimator_name(base_est)!r}. Each"
                " estimator must have a fit method. Refer to scikit-learn"
                " https://scikit-learn.org/stable/modules/classes.html API"
                " reference to build your own estimator.") 
            
        self.estimator  = base_est 
        if callable (base_est): 
            self.estimator  = base_est () # use default initialization 
            
        return self.estimator 
        
    def fit(self, X, y, sample_weight= .75 ): 
        
        """ Quick methods used to evaluate eastimator, display the 
        error results as well as the sample model_predictions.
        
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
        
        sample_weight: float,default = .75 
            The ratio to sample X and y. The default sample 3/4 percent of the 
            data. 
            If given, will sample the `X` and `y`.  If ``None``, will sample the 
            half of the data.
            
        Returns 
        ---------
        `self` : :class:`~.BaseEvaluation` 
            :class:`~.BaseEvaluation` object. 
        """ 
        # pass when pipeline is supplied. 
        # we expect data be transform into numeric dtype 
        dtype = object if self.pipeline is not None else "numeric"
        X, y = check_X_y ( X,y, to_frame =True, dtype =dtype, 
            estimator= get_estimator_name(self.estimator), 
            )
        
        self.estimator = self._check_callable_estimator(self.estimator )
        
        self._logging.info (
            'Quick estimation using the %r estimator with config %r arguments %s.'
                %(repr(self.estimator),self.__class__.__name__, 
                inspect.getfullargspec(self.__init__)))

        sample_weight = float(
            _assert_all_types(sample_weight, int, float, 
                              objname ="Sample weight"))
        if sample_weight <= 0 or sample_weight >1: 
            raise ValueError ("Sample weight must be range between 0 and 1,"
                              f" got {sample_weight}")
            
        # sampling train data. 
        # use 75% by default of among data 
        n = int ( sample_weight * len(X)) 
        if hasattr (X, 'columns'): X = X.iloc [:n] 
        else : X=X[:n, :]
        y= y[:n]
 
        if self.pipeline is not None: 
            X =self.pipeline.fit_transform(X)
            
        if not self.prefit: 
            #for consistency 
            if self.scoring is None: 
                warnings.warn("'neg_mean_squared_error' scoring is used when"
                              " scoring parameter is ``None``.")
                self.scoring ='neg_mean_squared_error'
            self.scoring = "neg_mean_squared_error" if self.scoring in (
                None, 'nmse') else self.scoring 
            
            self.mse_, self.rmse_ , self.cv_scores_ = self._fit(
                X, y, 
                self.estimator, 
                cv_scores=True,
                scoring = self.scoring
                )
            
        return self 
    
    def _fit(self, 
        X, 
        y, 
        estimator,  
        cv_scores=True, 
        scoring ='neg_mean_squared_error' 
        ): 
        """Fit data once verified and compute the ``rmse`` scores.
        
        Parameters 
        ----------
        X: array-like of shape (n_samples, n_features) 
            training data for fitting 
        y: arraylike of shape (n_samples, ) 
            target for training 
        estimator: callable or scikit-learn estimator 
            Callable or something that has a fit methods. Can build your 
            own estimator following the API reference via 
            https://scikit-learn.org/stable/modules/classes.html 
   
        cv_scores: bool,default=True 
            compute the cross validations scores 
       
        scoring: str, default='neg_mean_squared_error' 
            metric dfor scores evaluation. 
            Type of scoring for cross validation. Please refer to  
            :doc:`~.slkearn.model_selection.cross_val_score` for further 
            details.
            
        Returns 
        ----------
        (mse, rmse, scores): Tuple 
            - mse: Mean Squared Error  
            - rmse: Root Meam Squared Error 
            - scores: Cross validation scores 

        """
        mse = rmse = None  
        def display_scores(scores): 
            """ Display scores..."""
            n=("scores:", "Means:", "RMSE scores:", "Standard Deviation:")
            p=(scores, scores.mean(), np.sqrt(scores), scores.std())
            for k, v in zip (n, p): 
                pprint(k, v )
                
        self._logging.info(
            "Fit data with a supplied pipeline or using purely estimator")

        estimator.fit(X, y)
 
        y_pred = estimator.predict(X)
        
        if self.scoring !='accuracy': # if regression task
            mse = mean_squared_error(y , y_pred)
            rmse = np.sqrt(mse)
        scores = None 
        if cv_scores: 
            scores = cross_val_score(
                estimator, X, y, cv=self.cv, scoring=self.scoring
                                     )
            if self.scoring == 'neg_mean_squared_error': 
                rmse= np.sqrt(-scores)
            else: 
                rmse= np.sqrt(scores)
            if self.verbose:
                if self.scoring =='neg_mean_squared_error': 
                    scores = -scores 
                display_scores(scores)   
                
        return mse, rmse, scores 
    
    def predict (self, X ): 
        """ Quick prediction and get the scores.
        
        Parameters 
        -----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Test set; Denotes data that is observed at testing and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.
            
        Returns 
        -------
        y: array-like, shape (M, ) ``M=m-samples``, 
            test predicted target. 
        """
        self.inspect 
        
        dtype = object if self.pipeline is not None else "numeric"
        
        X = check_array(X, accept_sparse= False, 
                        input_name ='X', dtype= dtype, 
                        estimator=get_estimator_name(self.estimator), 
                        )
        
        if self.pipeline is not None: 
            X= self.pipeline.fit_transform (X) 

        return self.estimator.predict (X ) 
    
    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'cv_scores_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 
        
BaseEvaluation.__doc__="""\
Evaluation of dataset using a base estimator.

Quick evaluation of the data after preparing and pipeline constructions. 

Parameters 
-----------
estimator: Callable,
    estimator for trainset and label evaluating; something like a 
    class that implements a fit methods. Refer to 
    https://scikit-learn.org/stable/modules/classes.html

{params.core.cv}
    The default is ``4``.
{params.core.scoring} 

pipeline: Callable or :class:`~sklearn.pipeline.Pipeline` object 
    If `pipeline` is given , `X` is transformed accordingly, Otherwise 
    evaluation is made using purely the base estimator with the given `X`. 
    Refer to https://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline
    for further details. 
    
kind: str, default ='GridSearchCV'
    Kind of grid search method. Could be ``GridSearchCV`` or 
    ``RandomizedSearchCV``.

prefit: bool, default=False, 
    If ``False``, does not need to compute the cross validation score once 
    again and ``True`` otherwise.
{params.core.random_state}
        
Examples 
-------- 
>>> import watex as wx 
>>> from watex.datasets import load_bagoue 
>>> from watex.models import BaseEvaluation 
>>> X, y = load_bagoue (as_frame =True ) 
>>> # categorizing the labels 
>>> yc = wx.smart_label_classifier (y , values = [1, 3, 10 ], 
                                 # labels =['FR0', 'FR1', 'FR2', 'FR4'] 
                                 ) 
>>> # drop the subjective columns ['num', 'name'] 
>>> X = X.drop (columns = ['num', 'name']) 
>>> # X = wx.cleaner (X , columns = 'num name', mode='drop') 
>>> X.columns 
Index(['shape', 'type', 'geol', 'east', 'north', 'power', 'magnitude', 'sfi',
       'ohmS', 'lwi'],
      dtype='object')
>>> X =  wx.naive_imputer ( X, mode ='bi-impute') # impute data 
>>> # create a pipeline for X 
>>> pipe = wx.make_naive_pipe (X) 
>>> Xtrain, Xtest, ytrain, ytest = wx.sklearn.train_test_split(X, yc) 
>>> b = BaseEvaluation (estimator= wx.sklearn.RandomForestClassifier, 
                        scoring = 'accuracy', pipeline = pipe)
>>> b.fit(Xtrain, ytrain ) # accepts only array 
>>> b.cv_scores_ 
Out[174]: array([0.75409836, 0.72131148, 0.73333333, 0.78333333])
>>> ypred = b.predict(Xtest)
>>> scores = wx.sklearn.accuracy_score (ytest, ypred) 
0.7592592592592593
""".format (params=_param_docs,
)

def get_best_kPCA_params(
    X:NDArray | DataFrame,
    n_components: float | int =2,
    *,
    y: ArrayLike | Series=None,
    param_grid: Dict[str, Any] =None, 
    clf: F =None,
    cv: int =7,
    **grid_kws
    )-> Dict[str, Any]: 

    from ..analysis.dimensionality import ( 
        get_component_with_most_variance, KernelPCA) 
    if n_components is None: 
        n_components= get_component_with_most_variance(X)
    if clf is None: 

        clf =Pipeline([
            ('kpca', KernelPCA(n_components=n_components)),
            ('log_reg', LogisticRegression())
            ])
    gridObj =GridSearch(base_estimator= clf,
                        grid_params= param_grid, 
                        cv=cv,
                        **grid_kws
                        ) 
    gridObj.fit(X, y)
    
    return gridObj.best_params_

get_best_kPCA_params.__doc__="""\
Select the Kernel and hyperparameters using GridSearchCV that lead 
to the best performance.

As kPCA( unsupervised learning algorithm), there is obvious performance
measure to help selecting the best kernel and hyperparameters values. 
However dimensionality reduction is often a preparation step for a 
supervised task(e.g. classification). So we can use grid search to select
the kernel and hyperparameters that lead the best performance on that 
task. By default implementation we create two steps pipeline. First reducing 
dimensionality to two dimension using kPCA, then applying the 
`LogisticRegression` for classification. AFter use Grid searchCV to find 
the best ``kernel`` and ``gamma`` value for kPCA in oder to get the best 
clasification accuracy at the end of the pipeline.

Parameters
----------
{params.core.X} 
{params.core.y}

n_components:int, 
     Number of dimension to preserve. If `n_components` is ranged between 
     0. to 1., it indicated the number of variance ratio to preserve. 
    
param_grid: list 
    list of parameters grids. For instance::
    
        param_grid=[dict(
            kpca__gamma=np.linspace(0.03, 0.05, 10),
            kpca__kernel=["rbf", "sigmoid"]
            )]
    
{params.core.clf} 
    It can also be a base estimator or a composite estimor with pipeline. For 
    instance::
    clf =Pipeline([
    ('kpca', KernelPCA(n_components=2))
    ('log_reg', LogisticRegression())
    ])
    
{params.core.cv}

grid_kws: dict, 
    Additional keywords arguments passed to Grid parameters from 
    :class:`~watex.models.validation.GridSearch`

Examples
---------
>>> from watex.analysis.dimensionality import get_best_kPCA_params
>>> from watex.datasets import fetch_data 
>>> X, y=fetch_data('Bagoue analysis data')
>>> param_grid=[dict(
    kpca__gamma=np.linspace(0.03, 0.05, 10),
    kpca__kernel=["rbf", "sigmoid"]
    )]
>>> clf =Pipeline([
    ('kpca', KernelPCA(n_components=2)), 
    ('log_reg', LogisticRegression())
     ])
>>> kpca_best_params =get_best_kPCA_params(
            X,y=y,scoring = 'accuracy',
            n_components= 2, clf=clf, 
            param_grid=param_grid)
>>> kpca_best_params
... {{'kpca__gamma': 0.03, 'kpca__kernel': 'rbf'}}

""".format(
    params=_param_docs,
    )
   
def getGlobalScores (
        cvres : Dict[str, ArrayLike] 
        ) -> Tuple [float]: 
    """ Retrieve the global mean and standard deviation score  from the 
    cross validation containers. 
    
    Parameters
    ------------
    cvres: dict of (str, Array-like) 
        cross validation results after training the models of number 
        of parameters equals to N. The `str` fits the each parameter stored 
        during the cross-validation while the value is stored in Numpy array.
    
    Returns 
    ---------
    ( mean_test_scores', 'std_test_scores') 
         scores on CV test data and standard deviation 
        
    """
    return  ( cvres.get('mean_test_score').mean() ,
             cvres.get('std_test_score').mean()) 

def getSplitBestScores(cvres:Dict[str, ArrayLike], 
                       split:int=0)->Dict[str, float]: 
    """ Get the best score at each split from cross-validation results
    
    Parameters 
    -----------
    cvres: dict of (str, Array-like) 
        cross validation results after training the models of number 
        of parameters equals to N. The `str` fits the each parameter stored 
        during the cross-validation while the value is stored in Numpy array.
    split: int, default=1 
        The number of split to fetch parameters. 
        The number of split must be  the number of cross-validation (cv) 
        minus one.
        
    Returns
    -------
    bests: Dict, 
        Dictionnary of the best parameters at the corresponding `split` 
        in the cross-validation. 
        
    """
    #if split ==0: split =1 
    # get the split score 
    split_score = cvres[f'split{split}_test_score'] 
    # take the max score of the split 
    max_sc = split_score.max() 
    ix_max = split_score.argmax()
    mean_score= split_score.mean()
    # get parm and mean score 
    bests ={'param': cvres['params'][ix_max], 
        'accuracy_score':cvres['mean_test_score'][ix_max], 
        'std_score':cvres['std_test_score'][ix_max],
        f"CV{split}_score": max_sc , 
        f"CV{split}_mean_score": mean_score,
        }
    return bests 

def displayModelMaxDetails(cvres:Dict[str, ArrayLike], cv:int =4):
    """ Display the max details of each stored model from cross-validation.
    
    Parameters 
    -----------
    cvres: dict of (str, Array-like) 
        cross validation results after training the models of number 
        of parameters equals to N. The `str` fits the each parameter stored 
        during the cross-validation while the value is stored in Numpy array.
    cv: int, default=1 
        The number of KFlod during the fine-tuning models parameters. 

    """
    for k in range (cv):
        print(f'split = {k}:')
        b= getSplitBestScores(cvres, split =k)
        print( b)

    globalmeansc , globalstdsc= getGlobalScores(cvres)
    print("Global split scores:")
    print('mean=', globalmeansc , 'std=',globalstdsc)


def displayFineTunedResults ( cvmodels: list[F] ): 
    """Display fined -tuning results 
    
    Parameters 
    -----------
    cvmnodels: list
        list of fined-tuned models.
    """
    bsi_bestestimators = [model.best_estimator_ for model in cvmodels ]
    mnames = ( get_estimator_name(n) for n in bsi_bestestimators)
    bsi_bestparams = [model.best_params_ for model in cvmodels]

    for nam, param , estimator in zip(mnames, bsi_bestparams, 
                                      bsi_bestestimators): 
        print("MODEL NAME =", nam)
        print('BEST PARAM =', param)
        print('BEST ESTIMATOR =', estimator)
        print()

def displayCVTables(cvres:Dict[str, ArrayLike],  cvmodels:list[F] ): 
    """ Display the cross-validation results from all models at each 
    k-fold. 
    
    Parameters 
    -----------
    cvres: dict of (str, Array-like) 
        cross validation results after training the models of number 
        of parameters equals to N. The `str` fits the each parameter stored 
        during the cross-validation while the value is stored in Numpy array.
    cvmnodels: list
        list of fined-tuned models.
        
    Examples 
    ---------
    >>> from watex.datasets import fetch_data
    >>> from watex.models import GridSearchMultiple, displayCVTables
    >>> X, y  = fetch_data ('bagoue prepared') 
    >>> gobj =GridSearchMultiple(estimators = estimators, 
                                 grid_params = grid_params ,
                                 cv =4, scoring ='accuracy', 
                                 verbose =1,  savejob=False , 
                                 kind='GridSearchCV')
    >>> gobj.fit(X, y) 
    >>> displayCVTables (cvmodels=[gobj.models.SVC] ,
                         cvres= [gobj.models.SVC.cv_results_ ])
    ... 
    """
    modelnames = (get_estimator_name(model.best_estimator_ ) 
                  for model in cvmodels  )
    for name,  mdetail, model in zip(modelnames, cvres, cvmodels): 
        print(name, ':')
        displayModelMaxDetails(cvres=mdetail)
        
        print('BestParams: ', model.best_params_)
        try:
            print("Best scores:", model.best_score_)
        except: pass 
        finally: print()
        
        
def get_scorers (*, scorer:str=None, check_scorer:bool=False, 
                 error:str='ignore')-> Tuple[str] | bool: 
    """ Fetch the list of available metrics from scikit-learn or verify 
    whether the scorer exist in that list of metrics. 
    This is prior necessary before  the model evaluation. 
    
    :param scorer: str, 
        Must be an metrics for model evaluation. Refer to :mod:`sklearn.metrics`
    :param check_scorer:bool, default=False
        Returns bool if ``True`` whether the scorer exists in the list of 
        the metrics for the model evaluation. Note that `scorer`can not be 
        ``None`` if `check_scorer` is set to ``True``.
    :param error: str, ['raise', 'ignore']
        raise a `ValueError` if `scorer` not found in the list of metrics 
        and `check_scorer `is ``True``. 
        
    :returns: 
        scorers: bool, tuple 
            ``True`` if scorer is in the list of metrics provided that 
            ` scorer` is not ``None``, or the tuple of scikit-metrics. 
            :mod:`sklearn.metrics`
    """
    from sklearn import metrics 
    scorers = tuple(metrics.SCORERS.keys()) 
    
    if check_scorer and scorer is None: 
        raise ValueError ("Can't check the scorer while the scorer is None."
                          " Provide the name of the scorer or get the list of"
                          " scorer by setting 'check_scorer' to 'False'")
    if scorer is not None and check_scorer: 
        scorers = scorer in scorers 
        if not scorers and error =='raise': 
            raise ValueError(
                f"Wrong scorer={scorer!r}. Supports only scorers:"
                f" {tuple(metrics.SCORERS.keys())}")
            
    return scorers 
              
def naive_evaluation(
        clf: F,
        X:NDArray,
        y:ArrayLike,
        cv:int =7,
        scoring:str  ='accuracy', 
        display: str ='off', 
        **kws
        ): 
    scores = cross_val_score(clf , X, y, cv = cv, scoring=scoring, **kws)
                         
    if display is True or display =='on':
        print('clf=:', clf.__class__.__name__)
        print('scores=:', scores )
        print('scores.mean=:', scores.mean())
    
    return scores , scores.mean()

naive_evaluation.__doc__="""\
Quick scores evaluation using cross validation. 

Parameters
----------
clf: callable 
    Classifer for testing default data. 
X: ndarray
    trainset data 
    
y: array_like 
    label data 
cv: int 
    KFold for data validation.
    
scoring: str 
    type of error visualization. 
    
display: str or bool, 
    show the show on the stdout
kws: dict, 
    Additional keywords arguments passed to 
    :func:`watex.exlib.slearn.cross_val_score`.
Returns 
---------
scores, mean_core: array_like, float 
    scaore after evaluation and mean of the score
    
Examples 
---------
>>> import watex as wx 
>>> from watex.models.validation import naive_evaluation
>>> X,  y = wx.fetch_data ('bagoue data prepared') 
>>> clf = wx.sklearn.DecisionTreeClassifier() 
>>> naive_evaluation(clf, X, y , cv =4 , display ='on' )
clf=: DecisionTreeClassifier
scores=: [0.6279 0.7674 0.7093 0.593 ]
scores.mean=: 0.6744186046511629
Out[57]: (array([0.6279, 0.7674, 0.7093, 0.593 ]), 0.6744186046511629)
"""
# deprecated in scikit-learn 0.21 to 0.23 
# from sklearn.externals import joblib 
# import sklearn.externals    
# from abc import ABC,abstractmethod#  ABCMeta  

# class AttributeCkecker(ABC): 
#     """ Check attributes and inherits from module `abc` 
#     for Data validators. 
    
#     Validate DataType mainly `X` train or test sets and `y` labels or
#     and any others params types.
    
#     """
#     def __set_name__(self, owner, name): 
#         try: 
#             self.private_name = '_' + name 
#         except AttributeError: 
#             warnings.warn('Object {owner!r} has not attribute {name!r}')
            
#     def __get__(self, obj, objtype =None):
#         return getattr(obj, self.private_name) 
    
#     def __set__(self, obj, value):
#         self.validate(value)
#         setattr(obj, self.private_name, value) 
        
#     @abstractmethod 
#     def validate(self, value): 
#         pass 

# class checkData (AttributeCkecker): 
#     """ Descriptor to check data type `X` or `y` or else."""
#     def __init__(self, Xdtypes):
#         self.Xdtypes =eval(Xdtypes)

#     def validate(self, value) :
#         """ Validate `X` and `y` type."""
#         if not isinstance(value, self.Xdtypes):
#             raise TypeError(
#                 f'Expected {value!r} to be one of {self.Xdtypes!r} type.')
            
# class checkValueType_ (AttributeCkecker): 
#     """ Descriptor to assert parameters values. Default assertion is 
#     ``int`` or ``float``"""
#     def __init__(self, type_):
#         self.six =type_ 
        
#     def validate(self, value):
#         """ Validate `cv`, `s_ix` parameters type"""
#         if not isinstance(value,  self.six ): 
#             raise ValueError(f'Expected {self.six} not {type(value)!r}')
   
# class  checkClass (AttributeCkecker): 
#     def __init__(self, klass):
#         self.klass = klass 
       
#     def validate(self, value): 
#         """ Validate the base estimator whether is a class or not. """
#         if not inspect.isclass(value): 
#             raise TypeError('Estimator might be a class object '
#                             f'not {type(value)!r}.')  
    