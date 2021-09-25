# -*- coding: utf-8 -*-
#       Author: Kouadio K.Laurent<etanoyau@gmail.con>
#       Create:on Fri Sep 10 15:37:59 2021
#       Licence: MIT
import os 
import warnings
from pprint import pprint 
import pickle 

import joblib
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.svm import SVC, LinearSVC 

from watex.datasets import fetch_data 
from watex.utils.ml_utils import SearchedGrid 
from watex.utils._watexlog import watexlog 

__logger = watexlog().get_watex_logger(__name__)

X_prepared, y_prepared = fetch_data('Bagoue dataset prepared')


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
        list of parameters Grids. For instance:; 
            
            grid_params= ([
            {'C':[1e-2, 1e-1, 1, 10, 100], 'gamma':[5, 2, 1, 1e-1, 1e-2, 1e-3],
                         'kernel':['rbf']}, 
            {'kernel':['poly'],'degree':[1, 3,5, 7], 'coef0':[1, 2, 3], 
             'C': [1e-2, 1e-1, 1, 10, 100]}], 
            [{'C':[1e-2, 1e-1, 1, 10, 100], 'loss':['hinge']}], 
            [dict()]
            )
    cv: int 
        number of K-Fold to cross validate the training set.
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
        searchObj = SearchedGrid(base_estimator=estm_, 
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
                                'grid_param':grid_params[j]
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
if __name__=='__main__': 
    #cross validation Kfold 
    cv = 4
    # type of scores 
    scoring ='roc_auc_ovo'#'neg_mean_squared_error'#'accuracy'
    # display scores 
    display= 'off'
    # random state for estimator s
    random_state =42 
    # kind of grid search 
    kind ='GridSearchCV'
    
    
    logreg_clf = LogisticRegression(random_state =random_state)
    linear_svc_clf = LinearSVC(random_state =random_state)
    sgd_clf = SGDClassifier(random_state = random_state)
    svc_clf = SVC(random_state =random_state) 
    
    gridParams =([
            {'C':[1e-2, 1e-1, 1, 10, 100], 'gamma':[5, 2, 1, 1e-1, 1e-2, 1e-3],'kernel':['rbf']}, 
            {'kernel':['poly'],'degree':[1, 3,5, 7], 'coef0':[1, 2, 3], 'C': [1e-2, 1e-1, 1, 10, 100]}
            ], 
            [{'C':[1e-2, 1e-1, 1, 10, 100], 'loss':['hinge']}], 
            [dict()]
        )
    multipleGridSearches(X_prepared,
                         y_prepared,
                         (svc_clf, linear_svc_clf,logreg_clf ), 
                         gridParams , scoring ='accuracy',
                         )
    
# from sklearn import metrics 
# print(metrics.SCORERS.keys())
# dict_keys(['explained_variance', 'r2', 'max_error', 'neg_median_absolute_error',
#            'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 
#            'neg_mean_squared_error', 'neg_mean_squared_log_error', 
#            'neg_root_mean_squared_error', 'neg_mean_poisson_deviance',
#            'neg_mean_gamma_deviance', 'accuracy', 'top_k_accuracy', 
#            'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted',
#            'roc_auc_ovo_weighted', 'balanced_accuracy', 'average_precision',
#            'neg_log_loss', 'neg_brier_score', 'adjusted_rand_score', 'rand_score',
#            'homogeneity_score', 'completeness_score', 'v_measure_score', 
#            'mutual_info_score', 'adjusted_mutual_info_score', 
#            'normalized_mutual_info_score', 'fowlkes_mallows_score',
#            'precision', 'precision_macro', 'precision_micro',
#            'precision_samples', 'precision_weighted', 'recall', 
#            'recall_macro', 'recall_micro', 'recall_samples', 
#            'recall_weighted', 'f1', 'f1_macro', 'f1_micro', 'f1_samples',
#            'f1_weighted', 'jaccard', 'jaccard_macro', 'jaccard_micro', 
#            'jaccard_samples', 'jaccard_weighted'])






















