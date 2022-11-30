# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Created on Tue May 17 11:30:51 2022

import warnings 
from .._docstring import refglossary
from .._watexlog import watexlog 
from .._typing import (  
    Optional, 
    ArrayLike, 
    NDArray, 
    )
from ..decorators import refAppender 
from ..exceptions import (
    EstimatorError, 
    NotFittedError 
    )
from ..utils.funcutils import (
    repr_callable_obj,
    smart_format,
    smart_strobj_recognition, 
    )
from ..utils.mlutils import (
    controlExistingEstimator , 
    fetchModel 
    )
from ._metapredictors import ( 
        _pMODELS 
    )

__all__=["p", "pModels"]

def cloneObj (cls, attributes ): 
    """ Clone object and update attributes """
    obj = cls.__new__(cls) 
    obj.__dict__.update (attributes ) 
    
    return obj

@refAppender(refglossary.__doc__)
class pModels : 
    """ Pretrained Models class. 
    
    The pretrained model class is composed of  estimators already 
    trained in a case study region in West -Africa `Bagoue region`_. Refer 
    to `Kouadio et al`_, 2022 for furher details. It is a set of ``support 
    vector machines``, `decision tree``, ``k-nearest neighbors``, ``Extreme
    ``gradient boosting machines``, benchmart ``voting classifier``, and ``
    ``bagging classifier``. 
    Each retrained model is considered as a class object and attributes compose 
    the training parameters from cross-validation results. 
    
    Parameters
    ----------- 
    model: str 
        Name of the pretrained model. Note that the pretrained SVMs is composed 
        of 04 kernels such as the ``rbf`` for radial basis function , the 
        ``poly`` for polynomial , ``sig`` for sigmoid and ``lin`` for linear. 
        Default is ``rbf``. Each kernel is a model attributes of SVM class. 
        For instance to retreived the pretrained model with kernel = 'poly', we 
        must use after fitting :class:`.pModels` class:: 
            
            >>> pModels(model='svm').fit().SVM.poly.best_estimator_ 
            ... SVC(C=128.0, coef0=7, degree=5, gamma=0.00048828125, kernel='poly', tol=0.01)
            >>> # or 
            >>> pModels(model='svm', kernel='poly').fit().estimator_
            ... SVC(C=128.0, coef0=7, degree=5, gamma=0.00048828125, kernel='poly', tol=0.01)
        
    kernel: str 
        kernel refers to SVM machines kernels. It can be ``rbf`` for radial basis
        function , the ``poly`` for polynomial , ``sig`` for sigmoid and
        ``lin`` for linear. No need to provide since it can be retrieved as an 
        attribute of the SVM model like:: 
            
            >>> pModels(model='svm').fit().SVM.rbf # is an object instance 
            >>> # to retreive the rbf values use attribute `best_estimator_ 
            >>> pModels(model='svm').fit().SVM.rbf.best_estimator_ 
            ...  SVC(C=2.0, coef0=0, degree=1, gamma=0.125)
            
    target: str 
        Two types of classification is predicted. The binary classification ``bin``
        and the multiclass classification ``multi``. default is ``bin``. When  
        turning target to ``multi``, be aware that only the SVMs are trained 
        for multilabels predictions. Futhernore, the `bin` consisted to predict 
        the flow rate (FR) with label {0} and {1} where {0} means the 
        :math:`FR <=1 m^3/hr` and {1} for :math:`FR> 1m^3/hr`. About `multi`, 
        four classes are predicted such as: 
            
        .. math:: 
            
            FR0 & = & FR = 0 
            FR1 & = & 0 < FR <=1 m^3/hr
            FR2 & = & 1< FR <=3 m^3/hr 
            FR3 & = & FR> 3 m^3/hr 
            
    oob_score: bool, 
        Out-of-bag. Setting `oob_score` to ``true``, you will retrieve some 
        pretrained model with ``obb_score`` set to true when training. The  
        pretrained models with fine-tuned model with `oob_score` set to true 
        are 'RandomForest' and  'Extratrees'. 
        
    objective: str, default='fr'
        Is the prediction aim goal, the reason for storing the pretrained 
        models. the default `objective` is 'fr' i.e. for flow rate prediction. 
        
    Examples 
    ----------
    >>> from watex.models.premodels import pModels 
    >>> # fetch the  the pretrained Adaboost model 
    >>> p= pModels (model ='ada') 
    >>> p.fit() 
    >>> p.AdaBoost.best_estimator_ 
    ... AdaBoostClassifier(base_estimator=LogisticRegression(), learning_rate=0.09,
                       n_estimators=500)
    >>> p.model = 'vot' 
    >>> p.fit() 
    >>> p.Voting.best_estimator_ 
    ... VotingClassifier(estimators=[('lr', LogisticRegression()),
    ...                             ('knn',
    ...                              KNeighborsClassifier(metric='manhattan',
    ...                                                   n_neighbors=9)),
    ...                             ('dt',
    ...                              DecisionTreeClassifier(criterion='entropy',
    ...                                                     max_depth=7)),
    ...                             ('pSVM',
    ...                              SVC(C=2.0, coef0=0, degree=1, gamma=0.125))])
    >>> p2 = pModels(model='extree', oob_score= True ).fit()
    >>> p2.ExtraTrees.best_estimator_ 
    ... ExtraTreesClassifier(bootstrap=True, criterion='entropy', max_depth=18,
                         max_features='auto', n_estimators=300, oob_score=True)
    
    """
    
    pdefaults_ = list(map ( lambda e: controlExistingEstimator(e), 
                ['xgboost', 'svm', 'dtc', 'stc', 'bag', 'logit', 'vtc',
                 'rfc', 'ada', 'extree', 'knn']))
    
    def __init__(
        self, 
        model:str='svm',  
        target:str='bin', 
        kernel:Optional[str]=None , 
        oob_score:bool=False, 
        objective: str='fr',
        ): 
        self._logging=watexlog.get_watex_logger(self.__class__.__name__)
        self.model=model 
        self.target=target 
        self.objective=objective 
        self.oob_score=oob_score
        self.kernel=kernel 
        
        
    def fit (self, X:NDArray = None , y: ArrayLike = None ): 
        """ Fit the pretrained model data and populate its corresponding 
        attributes. 
        
        :param X: NoneType 
            X does nothing, it is used for API consistency 
        :param y: NoneType 
            y does nothing, it is used for API consistency
            
        :example: 
        >>> from watex.models.premodels import pModels 
        >>> # fetch the  the pretrained Adaboosting 
        >>> p= pModels (model ='ada') 
        >>> p.fit() 
        >>> p.AdaBoost.best_estimator_ 
        ... AdaBoostClassifier(base_estimator=LogisticRegression(), learning_rate=0.09,
                           n_estimators=500)
        """
        if self.model is None: 
            raise TypeError( "NoneType can't be a model.")
        self.objective = str(self.objective).lower() 
        
        assert self.objective =='fr',(
            f"Pretrained objective is for flow rate prediction 'fr' passed to"
            f" parameter 'objective'; not {self.objective}"
            ) 
        assert self.target not in ("bin", "multi"), (
            "Two types of learning targets are expected: the multiclass"
            f"'multi' and binary 'bin'. Got {self.target!r}"
            )
        self.model, self.name_ = controlExistingEstimator(
            self.model , raise_err = True )
        # change the name of SVC 
        if self.name_ =='SupportVectorClassifier' : 
            self.name_ = 'SVM' 
        else: self. name_ = self.name_.replace('Classifier', '')

        if self.name_ =='ExtremeGradientBoosting': 
            self.name_ ='XGB' 
        
        if self.model not in list(map(lambda d: d[0], self.pdefaults_)): 
            pl = list(map(lambda d: str(d[0]) + ' -> ' + str(d[1]),
                          self.pdefaults_))
            raise EstimatorError( f"Unsupport model : {self.model}."
                                 f" Expects {smart_format(pl, 'or')}")
        try : 
            data_= _pDATA 
            # fetch data from module 
            # force to fetch default 
            # values in exceptions
            if data_ is None: raise 
        except : 
            data_ = _pMODELS 
          
        if self.oob_score: 
            if self.model in ('svc', 'extree', 'rdf'): 
                self.name_ +='_'
            else :
                raise EstimatorError(
                    "Pretrained model for 'oob_score=True' is only available"
                    " for RandomForest <'rdf'> and Extratrees <'extree'>',"
                   f" not {self.model!r}"
                   )
        obj = type (self.name_, (), {})

        try: 
            obj = cloneObj(obj, attributes=data_.get(self.name_).__dict__)
        except AttributeError: 
            obj = cloneObj(obj, attributes=data_.get(self.name_))
            
        if self.target =='multi': 
             if self.name_== 'SVM_': 
                 self.name_= 'SVM'
        else : 
            if '_' in self.name_ : 
                self.name_= self.name_.replace ('_', '')
                
        self.__setattr__(self.name_, obj )
        
        try: 
            self.estimator_ = getattr(self, self.name_).best_estimator_ 
            self.params_  = getattr(self, self.name_).best_params_ 
            
        except AttributeError : 
            # collect some data for quick access 
            if self.kernel is None: 
                self._logging.info("Default kernel 'rbf' is used "
                                   "as default estimator")
                warnings.warn("Default kernel 'rbf' is used "
                              "as default estimator")
                
                self.estimator_ = getattr(self, self.name_).poly.best_estimator_ 
                self.params_  = getattr(self, self.name_).poly.best_params_ 
            else : 
                self.estimator_ = getattr ( 
                    getattr(self, self.name_), self.kernel) .best_estimator_ 
                self.params_ = getattr ( 
                    getattr(self, self.name_),self.kernel) .best_params_
                
        return self 
    
    def predict(self, X: NDArray ) : 
        """ Predict object from the pretrained model 
        
        :param X: Ndarray , shape (M, N) where 'M' is the number of examples 
            and 'N' the number of features. 
        :returns: Array-like, shape (M, ) of the predicted values. 
        """
        try:
             getattr(self, 'estimator_')
        except NotFittedError:
            raise NotFittedError("Can't call the method 'predict'. Call"
                                 f" {self.__class__.__name__!r} 'fit' method"
                                 " first.")
            
        return self.estimator_.predict (X)
    
    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        return repr_callable_obj  (self)
       
    
    def __getattr__(self, name:str ):
        if name.endswith ('_'): 
            if name not in self.__dict__.keys(): 
                if name in ('name_'): 
                    raise NotFittedError (
                        f'Fit the {self.__class__.__name__!r} object first'
                        )
                
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        appender  = "" if rv is None else f'. Do you mean {rv!r}'
        
        raise AttributeError (
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{appender}{"" if rv is None else "?"}'
            )

  

class _objectview(object):
    """ View object of a superclass created from each subclasss of dict 
    elements.
    
    Is a container of dict element resulting  from model instance element. 
    Thus, each element can be retrieved as its own attribute. For instance:: 
        
        >>> from watex.models.premodels import p 
        >>> p.SVM.poly.best_estimator_
        ... SVC(C=128.0, coef0=7, degree=5, gamma=0.00048828125, kernel='poly', tol=0.01)
        >>> p.XGB.best_estimator_ 
        ... XGBClassifier(base_score=None, booster='gbtree', colsample_bylevel=None,
                      colsample_bynode=None, colsample_bytree=None,
                      ... 
                      tree_method=None, validate_parameters=None, verbosity=None)
        >>> p.RandomForest.best_estimator_ 
        ... RandomForestClassifier(criterion='entropy', max_depth=16, n_estimators=350)
        >>> p.keys 
        ... ('SVM', 'SVM_', 'LogisticRegression', 'KNeighbors', 'DecisionTree',
             'Voting', 'RandomForest', 'RandomForest_', 'ExtraTrees', 
             'ExtraTrees_', 'Bagging', 'AdaBoost', 'XGB', 'Stacking'
             ) 
    """
    def __init__(self, kwds ):
        for key in list (kwds.keys()): 
            setattr(self, key, kwds[key])
        setattr(self ,'keys', tuple(self.__dict__.keys()) )
        

p = _objectview(_pMODELS)
  
p.__doc__= """\
p Object is a supclass that contains all the pretrained models. 
each pretrained model composes its own class object with dict element as 
attributes. 

Each pretrained model can fetched  as an attribute. For instance:: 
    
    >>> from watex.models.premodels import p 
    >>> # get the pretrained models using the key attributes 
    >>> p.keys 
    ... ('SVM', 'SVM_', 'LogisticRegression', 'KNeighbors', 'DecisionTree',
         'Voting', 'RandomForest', 'RandomForest_', 'ExtraTrees', 
         'ExtraTrees_', 'Bagging', 'AdaBoost', 'XGB', 'Stacking'
         ) 
    >>> # fetch the pretrained LogisticRegression best parameters 
    >>> p.LogisticRegression.best_params_ 
    ... {'penalty': 'l2',
         'dual': False,
         'tol': 0.0001,
         'C': 1.0,
         'fit_intercept': True,
         'intercept_scaling': 1,
         'class_weight': None,
         'random_state': None,
         'solver': 'lbfgs',
         'max_iter': 100,
         'multi_class': 'auto',
         'verbose': 0,
         'warm_start': False,
         'n_jobs': None,
         'l1_ratio': None
     }
    >>> # fetcth the pretrained RandomForest with out-of-bagg equal to True 
    >>> p.RandomForest_.best_estimator_ 
    ... RandomForestClassifier(max_depth=15, oob_score=True)
    
Note
------
To fetch the pretrained model with parameter (out-of-bag ), need to use the 
'_' at the end of the model name like 'ExtraTrees_'. 
However the pretrained model of Support Vector Machines  with underscore means 
the fine tuned multiclassification targets not 'out-of-bag' parameters. 

"""
#-- Fetch the pretrained model data 
try : 
    _pDATA,  = fetchModel ('./watex/etc/p.models.pkl', default = False )
except: 
    # set to None if something goes wrong 
    _pDATA = None 
     



    
    
    
    
    
    
    
    
    
    
    
    
    
    