# -*- coding: utf-8 -*-
#       Author: Kouadio K.Laurent<etanoyau@gmail.con>
#       Create:on Fri Sep 10 15:37:59 2021
#       Licence: MIT
import os
import inspect
import warnings  
import pickle 
import joblib
from typing import TypeVar, Iterable , Callable
from abc import ABC, abstractmethod, ABCMeta  
from pprint import pprint 
import pandas as pd 
import numpy as np 

# from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
# from sklearn.model_selection import cross_val_predict 
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import roc_curve, roc_auc_score

from watex.utils._watexlog import watexlog

T= TypeVar('T')
KT=TypeVar('KT')
VT=TypeVar('VT')

__logger = watexlog().get_watex_logger(__name__)

"""
Created on Sat Sep 25 10:10:31 2021

@author: @Daniel03
"""
def multipleGridSearches(X, 
                        y,
                        estimators, 
                        grid_params,
                        scoring ='neg_mean_squared_error', 
                        cv=7, 
                        kindOfSearch ='GridSearchCV',
                        random_state =42,
                        save_to_joblib=False,
                        get_metrics_SCORERS=False, 
                        verbose=0,
                        **pkws):
    """ Search and find multiples best parameters from differents
    estimators.
    
    Parameters
    ----------
    X: dataframe or ndarray
        Training set data 
    y: array_like 
        label or target data 
        
    estimators: list of callable obj 
        list of estimator objects to fine-tune their hyperparameters 
        For instance::
            
            estimators= (LogisticRegression(random_state =random_state), 
             LinearSVC(random_state =random_state), 
             SVC(random_state =random_state) )
            
    grid_params: list 
        list of parameters Grids. For instance:: 
            
            grid_params= ([
            {'C':[1e-2, 1e-1, 1, 10, 100], 'gamma':[5, 2, 1, 1e-1, 1e-2, 1e-3],
                         'kernel':['rbf']}, 
            {'kernel':['poly'],'degree':[1, 3,5, 7], 'coef0':[1, 2, 3], 
             'C': [1e-2, 1e-1, 1, 10, 100]}], 
            [{'C':[1e-2, 1e-1, 1, 10, 100], 'loss':['hinge']}], 
            [dict()]
            )
    cv: int 
        Number of K-Fold to cross validate the training set.
    scoring: str 
        Type of scoring to evaluate your grid search. Use 
        `sklearn.metrics.metrics.SCORERS.keys()` to get all the metrics used 
        to evaluate model errors. Default is `'neg_mean_squared_error'`. Can be 
        any others metrics `~metrics.metrics.SCORERS.keys()` of scikit learn.
        
    kindOfSearch:str
        Kinf of grid search. Can be ``GridSearchCV`` or ``RandomizedSearchCV``. 
        Default is ```GridSearchCV``
        
    random_state: int 
        State to shuffle the cross validation data. 
    
    save_to_joblib: bool, 
        Save your model ad parameters to sklearn.external.joblib. 
        
    get_metrics_SCORERS: list 
        list of diferent metrics to evaluate the scores of the models. 
        
    verbose: int , level=0
        control the verbosity, higher value, more messages.
    
    Examples
    --------
    
    .../scripts/fine_tune_hyperparams.py
    """

    if get_metrics_SCORERS: 
        from sklearn import metrics 
        if verbose >0: 
            pprint(','.join([ k for k in metrics.SCORERS.keys()]))
            
        return tuple(metrics.SCORERS.keys())
    
    if len(estimators)!=len(grid_params): 
        warnings.warn('Estimators and grid parameters must have the same .'
                      f'length. But {len(estimators)!r} and {len(grid_params)!r} '
                      'were given.'
                      )
        raise ValueError('Estimator and the grid parameters for fine-tunning '
                         'must have the same length. %s and %s are given.'
                         %(len(estimators),len(grid_params)))
    
    _clfs =list()
    _dclfs=dict()
    msg =''
    pickfname= '__'.join([f'{b.__class__.__name__}' for b in estimators ])
    
    for j , estm_ in enumerate(estimators):
        
        msg = f'{estm_.__class__.__name__} is evaluated.'
        searchObj = GridSearch(base_estimator=estm_, 
                                  grid_params= grid_params[j], 
                                  cv = cv, 
                                  kind=kindOfSearch, 
                                  scoring=scoring
                                  )
        searchObj.fit(X, y)
        best_model_clf = searchObj.best_estimator_ 
        
        if verbose >7 :
            msg+= ''.join([
                f'\End Gridsearch. Resetting {estm_.__class__.__name__}',
                ' `.best_params_`, `.best_estimator_`, `.cv_results` and', 
                ' `.feature_importances_` and grid_kws attributes\n'])

        _dclfs[f'{estm_.__class__.__name__}']= {
                                'best_model':searchObj.best_estimator_ ,
                                'best_params_':searchObj.best_params_ , 
                                'cv_results_': searchObj.cv_results_,
                                'grid_kws':searchObj.grid_kws,
                                'grid_param':grid_params[j],
                                'scoring':scoring
                                }
        
        msg +=''.join([ f' Cross evaluate with KFold ={cv} the',
                       ' {estm_.__class.__name__} best model.'])
        if verbose >7: display ='on'
        else :display='off'
        bestim_best_scores,_ = quickscoring_evaluation_using_cross_validation(
            best_model_clf, 
            X,
            y,
            cv = cv, 
            scoring = scoring,
            display =display)
        # store the best scores 
        _dclfs[f'{estm_.__class__.__name__}'][
            'best_scores']= bestim_best_scores
        # for k, v in dclfs.items():     
        _clfs.append((estm_,
                      searchObj.best_estimator_,
                      searchObj.best_params_, 
                      bestim_best_scores) )
    msg +=f'\Pretty print estimators results using scoring ={scoring!r}'
    if verbose >0:
        prettyPrinter(clfs=_clfs, scoring =scoring, 
                       clf_scores= None, **pkws )
    
        
    msg += f'\Serialize dict of parameters fine-tune to `{pickfname}`.'
    
    if save_to_joblib:
        __logger.info(f'Dumping models `{pickfname}`!')
        
        try : 
            joblib.dump(_dclfs, f'{pickfname}.pkl')
            # and later ....
            # f'{pickfname}._loaded' = joblib.load(f'{pickfname}.pkl')
            dmsg=f'Model `{pickfname} dumped using to ~.externals.joblib`!'
            
        except : 
            # piclke data Serializing data 
            with open(pickfname, 'wb') as wfile: 
                pickle.dump( _dclfs, wfile)
            # new_dclfs_infile = open(names,'rb')
            # new_dclfs= pickle.load(new_dclfs_infile)
            # new_dclfs_infile.close()
            
            pprint(f'Models are serialized  in `{pickfname}`. Please '
                   'refer to your current work directory.'
                   f'{os.getcwd()}')
            __logger.info(f'Model `{pickfname} serialized to {pickfname}.pkl`!')
            dmsg=f'Model `{pickfname} serialized to {pickfname}.pkl`!'
            
        else: __logger.info(dmsg)   
            
        if verbose >1: 
            pprint(
                dmsg + '\nTry to retrieve your model using`:meth:.load`'
                'method. For instance: slkearn --> joblib.load(f"{pickfname}.pkl")'
                'or pythonPickle module:-->pickle.load(open(f"{pickfname},"rb")).'
                )
            
    if verbose > 1:  
        pprint(msg)    
      
    return _clfs, _dclfs, joblib


def prettyPrinter(clfs,  clf_score=None, 
                   scoring =None,
                  **kws): 
    """ Format and pretty print messages after gridSearch sing multiples
    estimators.
    
    display for each estimator, its name, it best params with higher score 
    and the mean scores. 
    
    Parameters
    ----------
    clfs:Callables 
        classifiers or estimators 
    
    clf_scores: array-like
        for single classifier, usefull to provided the 
        cross validation score.
    
    scoring: str 
        Scoring used for grid search.
    """
    empty =kws.pop('empty', ' ')
    e_pad =kws.pop('e_pad', 2)
    p=list()

    if not isinstance(clfs, (list,tuple)): 
        clfs =(clfs, clf_score)

    for ii, (clf, clf_be, clf_bp, clf_sc) in enumerate(clfs): 
        s_=[e_pad* empty + '{:<20}:'.format(
            clf.__class__.__name__) + '{:<20}:'.format(
                'Best-estimator <{}>'.format(ii+1)) +'{}'.format(clf_be),
         e_pad* empty +'{:<20}:'.format(' ')+ '{:<20}:'.format(
            'Best paramaters') + '{}'.format(clf_bp),
         e_pad* empty  +'{:<20}:'.format(' ') + '{:<20}:'.format(
            'scores<`{}`>'.format(scoring)) +'{}'.format(clf_sc)]
        try : 
            s0= [e_pad* empty +'{:<20}:'.format(' ')+ '{:<20}:'.format(
            'scores mean')+ '{}'.format(clf_sc.mean())]
        except AttributeError:
            s0= [e_pad* empty +'{:<20}:'.format(' ')+ '{:<20}:'.format(
            'scores mean')+ 'None']
            s_ +=s0
        else :
            s_ +=s0

        p.extend(s_)
    

    for i in p: 
        print(i)


class GridSearch: 
    """ Fine tune hyperparameters. 
    
    `Search Grid will be able to  fiddle with the hyperparameters until to 
    find the great combination for model predictions. 
    
    :param base_estimator: Estimator to be fined tuned hyperparameters
    
    :param grid_params: list of hyperparamters params  to be tuned 
    
    :param cv: Cross validation sampling. Default is `4` 
    
    :pram kind: Kind of search. Could be ``'GridSearchCV'`` or
    ``RandomizedSearchCV``. Default is ``gridSearchCV`.
    
    :param scoring: Type of score for errors evaluating. Default is 
        ``neg_mean_squared_error``. 
        
    :Example: 
        
        >>> from pprint import pprint 
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from watex.utils._data_preparing_ import bagoue_train_set_prepared 
        >>> from watex.utils._data_preparing_ import bagoue_label_encoded  
        >>> grid_params = [
        ...        {'n_estimators':[3, 10, 30], 'max_features':[2, 4, 6, 8]}, 
        ...        {'bootstrap':[False], 'n_estimators':[3, 10], 
        ...                             'max_features':[2, 3, 4]}]
        >>> forest_clf = RandomForestClassifier()
        >>> grid_search = GridSearch(forest_clf, grid_params)
        >>> grid_search.fit(X= bagoue_train_set_prepared ,
        ...                    y = bagoue_label_encoded)
        >>> pprint(grid_search.best_params_ )
        >>> pprint(grid_search.cv_results_)
    """
    
    __slots__=('_base_estimator',
                'grid_params', 
                'scoring',
                'cv', 
                '_kind', 
                 'grid_kws',
                'best_params_',
                'best_estimator_',
                'cv_results_',
                'feature_importances_',
                )
               
    def __init__(self,
                 base_estimator:Callable[..., T],
                 grid_params:Iterable[T],
                 cv:int =4,
                 kind:str ='GridSearchCV',
                 scoring:str = 'neg_mean_squared_error',
                 **grid_kws): 
        
        self._base_estimator = base_estimator 
        self.grid_params = grid_params 
        self.scoring = scoring 
        self.cv = cv 
        self._kind = kind 
        
        self.best_params_ =None 
        self.cv_results_= None
        self.feature_importances_= None
        self.best_estimator_=None 
    
        if len(grid_kws)!=0: 
            self.__setattr__('grid_kws', grid_kws)
            
    @property 
    def base_estimator (self): 
        """ Return the base estimator class"""
        return self._base_estimator 
    
    @base_estimator.setter 
    def base_estimator (self, baseEstim): 
        if not inspect.isclass(baseEstim) or\
            type(self.estimator) != ABCMeta: 
            raise TypeError(f"Expected an Estimator not {type(baseEstim)!r}")
            
        self._base_estimator =baseEstim 
        
    @property 
    def kind(self): 
        """ Kind of searched. `RandomizedSearchCV` or `GridSearchCV`."""
        return self._kind 
    
    @kind.setter 
    def kind (self, typeOfsearch): 
        """`kind attribute checker"""
        if typeOfsearch ==1 or 'GridSearchCV'.lower(
                ).find(typeOfsearch.lower())>=0: 
            typeOfsearch = 'GridSearchCV'
            
        if typeOfsearch ==2 or  'RandomizedSearchCV'.lower(
                ).find(typeOfsearch.lower())>=0:
            typeOfsearch = 'RandomizedSearchCV'
    
        else: 
            raise ValueError('Expected %r or %r not %s.'
                             %('gridSearchCV','RandomizedSearchCV', 
                               typeOfsearch ))
            
        self._kind = typeOfsearch 

    def fit(self,  X, y, **grid_kws): 
        """ Fit method using base Estimator.
        
        Populate gridSearch attributes. 
        
        :param X: Train dataset 
        :param y: Labels
        :param grid_kws: Additional keywords arguments of Gird search.
            keywords arguments must be the inner argunents of `GridSearchCV` or 
            `RandomizedSearchCV`.
        """
        
        if hasattr(self, 'grid_kws'): 
            grid_kws = getattr(self, 'grid_kws')
            
        if type(self.base_estimator) == ABCMeta :
            
            baseEstimatorObj = self.base_estimator()
            # get the base estimators parameters values in case for logging 
            # and users warnings except the `self`.
            init_signature = inspect.signature(baseEstimatorObj.__init__)
            
            parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
            
            self._logging.info('%s estimator (type %s) could not be cloned. Need'
                               ' to create an instance with default arguments '
                               ' %r for cross validatation grid search.'
                               %(repr(baseEstimatorObj.__class__),
                                 repr(type(baseEstimatorObj)), parameters))
            
            warnings.warn('%s estimator (type %s) could not be cloned.'
                          'Need to create an instance with default arguments '
                          ' %r for cross validatation grid search.'
                            %(repr(baseEstimatorObj.__class__),
                            repr(type(baseEstimatorObj)), parameters), 
                            UserWarning)
        else : 
            # suppose an instance is created before running the 
            # `GridSearch` class. 
            baseEstimatorObj  = self.base_estimator 
        
        if self.kind =='GridSearchCV': 
            try: 
                gridObj = GridSearchCV(baseEstimatorObj  , 
                                        self.grid_params,
                                        scoring = self.scoring , 
                                        cv = self.cv,
                                        **grid_kws)
            except TypeError: 
                warnings.warn('%s does not accept the param %r arguments.'
                              %(GridSearchCV.__class__, grid_kws),
                              RuntimeWarning)
                __logger.error('Unacceptable params %r arguments'
                                      % grid_kws)
            
        elif self.kind =='RandomizedSearchCV':
            try: 
                gridObj = RandomizedSearchCV(baseEstimatorObj ,
                                            self.grid_params,
                                            scoring = self.scoring,
                                            **grid_kws
                                                     )
            except TypeError:
                warnings.warn('%s does not accept the param %r arguments.'
                              %(RandomizedSearchCV.__class__, grid_kws),
                              RuntimeWarning)
                __logger.warnings('Unacceptable params %r arguments'
                                      %self.grid_kws)
        try : 
            # fit gridSearchObject.
            gridObj.fit(X,y)
            
        except TypeError : 
  
            init_signature = inspect.signature(baseEstimatorObj.__init__)
            parameters = [p for p in init_signature.parameters.values()
                          if p.name != 'self' ] 
            
            warnings.warn('sklearn.clone error. Cannot clone object %s.'
                          'To avoid future warning, Create an instance of'
                          'estimator and set the instance as %s arguments.' 
                          %(repr(baseEstimatorObj ),type(baseEstimatorObj )),
                          FutureWarning)
            
            __logger.warning("Trouble of clone estimator. Create an instance "
                            " of estimator and set as %r base_estimator"
                            " arguments before runing the {type(self)!r}"
                            "class. Please create instance with %s params"
                            "values."%(repr(type(baseEstimatorObj)), 
                                       repr(parameters)))
            
            return self
        
        for param_ , param_value_ in zip(
                ['best_params_','best_estimator_','cv_results_'],
                [gridObj.best_params_, gridObj.best_estimator_, 
                             gridObj.cv_results_ ]
                             ):
            setattr(self, param_, param_value_)
        try : 
            attr_value = gridObj.best_estimator_.feature_importances_
        except AttributeError: 
            warnings.warn ('{0} object has no attribute `feature_importances_`'.
                           format(gridObj.best_estimator_.__class__.__name__))
            setattr(self,'feature_importances_', None )
        else : 
            setattr(self,'feature_importances_', attr_value)
            
        #resetting the grid-kws attributes 
        setattr(self, 'grid_kws', grid_kws)
        
        return self
    
class AttributeCkecker(ABC): 
    """ Check attributes and inherits from module `abc` for Data validators. 
    
    Validate DataType mainly `X` train or test sets and `y` labels or
    and any others params types.
    """
    
    def __set_name__(self, owner, name): 
        try: 
            self.private_name = '_' + name 
        except AttributeError: 
            warnings.warn('Object {owner!r} has not attribute {name!r}')
            
    def __get__(self, obj, objtype =None):
        return getattr(obj, self.private_name) 
    
    def __set__(self, obj, value):
        self.validate(value)
        setattr(obj, self.private_name, value) 
        
    @abstractmethod 
    def validate(self, value): 
        pass 

class checkData (AttributeCkecker): 
    """ Descriptor to check data type `X` or `y` or else."""
    def __init__(self, Xdtypes):
        self.Xdtypes =eval(Xdtypes)

    def validate(self, value) :
        """ Validate `X` and `y` type."""
        if not isinstance(value, self.Xdtypes):
            raise TypeError(
                f'Expected {value!r} to be one of {self.Xdtypes!r} type.')
            
class checkValueType_ (AttributeCkecker): 
    """ Descriptor to assert parameters values. Default assertion is 
    ``int`` or ``float``"""
    def __init__(self, type_):
        self.six =type_ 
        
    def validate(self, value):
        """ Validate `cv`, `s_ix` parameters type"""
        if not isinstance(value,  self.six ): 
            raise ValueError(f'Expected {self.six} not {type(value)!r}')
   
class  checkClass (AttributeCkecker): 
    def __init__(self, klass):
        self.klass = klass 
       
    def validate(self, value): 
        """ Validate the base estimator whether is a class or not. """
        if not inspect.isclass(value): 
            raise TypeError('Estimator might be a class object '
                            f'not {type(value)!r}.')
        
class BaseEvaluation (object): 
    """ Evaluation of dataset using a base estimator.
    
    Quick evaluation after data preparing and pipeline constructions. 
    
    :param base_estimator: obj 
        estimator for trainset and label evaluating 
        
    :param X: ndarray of dataframe of trainset data
    
    :param y: array of labels data 
    
    :param s_ix: int, sampling index. 
        If given, will sample the `X` and `y` 
            
    :param columns: list of columns. Use to build dataframe `X` when `X` is 
        given as numpy ndarray. 
        
    :param pipeline: callable func 
            Tranformer data and preprocessing 
    :param cv: cross validation splits. Default is ``4``.
            
    """
   
    def __init__(self, 
                 base_estimator,
                 X, 
                 y,
                 s_ix=None,
                 cv=7,  
                 pipeline= None, 
                 columns =None, 
                 pprint=True, 
                 cvs=True, 
                 scoring ='neg_mean_squared_error',
                 **kwargs): 
        
        self._logging =watexlog().get_watex_logger(self.__class__.__name__)
        
        self.base_estimator = base_estimator
        self.X= X 
        self.y =y 
        self.s_ix =s_ix 
        self.cv = cv 
        self.columns =columns 
        self.pipeline =pipeline
        self.pprint =pprint 
        self.cvs = cvs
        self.scoring = scoring
        
        for key in list(kwargs.keys()): 
            setattr(self, key, kwargs[key])

        if self.X is not None : 
            self.quickEvaluation()
            
    def quickEvaluation(self, fit='yes', **kws): 
        
        """ Quick methods used to evaluate eastimator, display the 
        error results as well as the sample model_predictions.
        
        :param X: Dataframe  be trained 
        :param y: labels from trainset 
        :param sample_ix: index to sample in the trainset and labels. 
        :param kws: Estmator additional keywords arguments. 
        :param fit: Fit the method for quick estimating 
            Default is ``yes`` 
            
        """ 
        pprint =kws.pop('pprint', True) 
        if pprint is not None: 
            self.pprint = pprint 
        cvs = kws.pop('cvs', True)
        if cvs is not None: 
            self.cvs = cvs 
        scoring = kws.pop('scoring', 'neg_mean_squared_error' )
        if scoring is not None: 
            self.scoring  = scoring 
            
        self._logging.info ('Quick estimation using the %r estimator with'
                            'config %r arguments %s.'
                            %(repr(self.base_estimator),self.__class__.__name__, 
                            inspect.getfullargspec(self.__init__)))
        
        if not hasattr(self, 'random_state'):
            self.random_state =42 
            
            try:
                if kws.__getitem__('random_state') is not None : 
                    setattr(self, 'random_state', kws['random_state'])
            except KeyError: 
                self.random_state =42 
  
        if not inspect.isclass(self.base_estimator) or \
              type(self.base_estimator) !=ABCMeta:
                if type(self.base_estimator).__class__.__name__ !='type':
                    raise TypeError('Estimator might be a class object '
                                    f'not {type(self.base_estimator)!r}.')
                
        if type(self.base_estimator) ==ABCMeta:  
            try: 
                self.base_estimator  = self.base_estimator (**kws)
            except TypeError: 
                self.base_estimator  = self.base_estimator()

        if  self.s_ix is None: 
            self.s_ix = int(len(self.X)/2)

        if self.s_ix is not None: 
            if isinstance(self.X, pd.DataFrame): 
                self.X= self.X.iloc[: int(self.s_ix)]
            elif isinstance(self.X, np.ndarray): 
                if self.columns is None:
                    warnings.warn(
                        f'{self.columns!r} must be a dataframe columns!'
                          f' not {type(self.columns)}.',UserWarning)
                    
                    if self.X.ndim ==1 :
                        size =1 
                    elif self.X.ndim >1: 
                        size = self.X.shape[1]
                    
                    return TypeError(f'Expected {size!r} column name'
                                      '{"s" if size >1 else 1} for array.')

                elif self.columns is not None: 
                    if self.X.shape[1] !=len(self.columns): 
                        warnings.warn(f'Expected {self.X.shape[1]!r}' 
                                      f'but {len(self.columns)} '
                                      f'{"is" if len(self.columns) < 2 else"are"} '
                                      f'{len(self.columns)!r}.',RuntimeWarning)
         
                        raise IndexError('Expected %i not %i self.columns.'
                                          %(self.X.shape[2], 
                                            len(self.columns)))
                        
                    self.X= pd.DataFrame(self.X, self.columns)
                    
                self.X= self.X.iloc[: int(self.s_ix)]
    
            self.y= self.y[:int(self.s_ix )]  
    
        if isinstance(self.y, pd.Series): 
            self.y =self.y.values 
   
        if fit =='yes': 
            self.fit_data(self.base_estimator , pprint= self.pprint,
                          compute_cross=self.cvs,
                          scoring = self.scoring)
            
            
    def fit_data (self, obj , pprint=True, compute_cross=True, 
                  scoring ='neg_mean_squared_error' ): 
        """ Fit data once verified and compute the ``rmse`` scores.
        
        :paramm obj: base estimator with base params
        :param pprint: Display prediction of the quick evaluation 
        ;param compute_cross: compute the cross validation 
        :param scoring: Type of scoring for cross validation. Please refer to  
                 :doc:~slkearn.sklearn.model_selection.cross_val_score
                 for further details.
        """
        def display_scores(scores): 
            """ Display scores..."""
            print('scores:', scores)
            print('Mean:', scores.mean())
            print('rmse scores:', np.sqrt(scores))
            print('standard deviation:', scores.std())
            
        self._logging.info('Fit data X with shape {X.shape!r}.')
        
        if self.pipeline is not None: 
            train_prepared_obj =self.pipeline.fit_transform(self.X)
            
        elif self.pipeline is None: 
            warnings.warn('No Pipeline is applied. Could estimate with purely'
                          '<%r> given estimator.'%(self.base_estimator.__name__))
            self.logging.info('No Pipeline is given. Evaluation should be based'
                              'using  purely  the given estimator <%r>'%(
                                  self.base_estimator.__name__))
            
            train_prepared_obj =self.base_estimator.fit_transform(self.X)
        
        obj.fit(train_prepared_obj, self.y)
 
        if pprint: 
             print("predictions:\t", obj.predict(train_prepared_obj ))
             print("Labels:\t\t", list(self.y))
            
        y_obj_predicted = obj.predict(train_prepared_obj)
        
        obj_mse = mean_squared_error(self.y ,
                                     y_obj_predicted)
        self.rmse = np.sqrt(obj_mse )

        if compute_cross : 
            
            self.scores = cross_val_score(obj, train_prepared_obj,
                                     self.y, 
                                     cv=self.cv,
                                     scoring=self.scoring
                                     )
            
            if self.scoring == 'neg_mean_squared_error': 
                self.rmse_scores = np.sqrt(-self.scores)
            else: 
                self.rmse_scores = np.sqrt(self.scores)
    
            if pprint:
                if self.scoring =='neg_mean_squared_error': 
                    self.scores = -self.scores 
                display_scores(self.scores)   
    
                
def quickscoring_evaluation_using_cross_validation(
        clf, X, y, cv=7, scoring ='accuracy', display='off'): 
    scores = cross_val_score(clf , X, y, cv = cv, scoring=scoring)
                         
    if display is True or display =='on':
        
        print('clf=:', clf.__class__.__name__)
        print('scores=:', scores )
        print('scores.mean=:', scores.mean())
    
    return scores , scores.mean()

quickscoring_evaluation_using_cross_validation.__doc__="""\
Quick scores evaluation using cross validation. 

Parameters
----------
clf: callable 
    Classifer for testing default data 
X: ndarray
    trainset data 
y: array_like 
    label data 
cv: int 
    KFold for data validation. 
scoring: str 
    type of error visualization 
display: str or bool, 
    show the show on the stdout
Returns 
-------
scores, mean_core: array_like, float 
    scaore after evaluation and mean of the score
"""
# deprecated in scikit-learn 0.21 to 0.23 
# from sklearn.externals import joblib 
# import sklearn.externals    
    
    