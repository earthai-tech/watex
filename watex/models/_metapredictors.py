# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from ..exlib.sklearn import (
    AdaBoostClassifier, 
    BaggingClassifier, 
    DecisionTreeClassifier, 
    ExtraTreesClassifier, 
    KNeighborsClassifier, 
    LogisticRegression , 
    RandomForestClassifier,
    StackingClassifier, 
    SVC, 
    VotingClassifier, 
    )
from ..exlib.gbm import XGBClassifier 
from ..utils.funcutils import get_params 

__all__=['_pMODELS']


_linear = SVC (
    **{
        'C': 0.5,
        'coef0': 0,
        'degree': 1,
        'gamma': 0.00048828125,
        'kernel': 'linear',
        'tol': 1.0
        }
    )

_poly = SVC (
    **{
        'C': 128.0,
        'coef0': 7,
        'degree': 5,
        'gamma': 0.00048828125,
        'kernel': 'poly',
        'tol': 0.01
       }
    )
_sigmoid= SVC (
    **{
        'C': 512.0,
        'coef0': 0,
        'degree': 1,
        'gamma': 0.001953125,
        'kernel': 'sigmoid',
        'tol': 1.0 
        }
    )
_rbf = SVC (
    **{
        'C': 2.0,
        'coef0': 0,
        'degree': 1,
        'gamma': 0.125,
        'kernel': 'rbf',
        'tol': 0.001
       }
    )
_svm = {'poly': {'best_estimator_': _poly , 
                 'best_params_': get_params(_poly)
                 },
        'sigmoid': {'best_estimator_': _sigmoid , 
                    'best_params_': get_params(_sigmoid)
                    }, 
        'rbf': {'best_estimator_': _rbf , 
                    'best_params_': get_params(_rbf)
                    }, 
        'linear': {'best_estimator_': _linear , 
                    'best_params_': get_params(_linear)
                    }, 
        }

_linear_ = SVC (
    **{
        'C': 8.0,
         'coef0': 0,
         'degree': 1,
         'gamma': 0.0078125,
         'kernel': 'linear',
         'tol': 1.0
        }
    )

_poly_ = SVC (
    **{
        'C': 0.5,
         'coef0': 2,
         'degree': 4,
         'gamma': 0.125,
         'kernel': 'poly',
         'tol': 0.1
       }
    )
_sigmoid_= SVC (
    **{
        'C': 128.0,
         'coef0': 1,
         'degree': 1,
         'gamma': 0.0078125,
         'kernel': 'sigmoid',
         'tol': 0.001
        }
    )
_rbf_ = SVC (
    **{
        'C': 32.0,
         'coef0': 0,
         'degree': 1,
         'gamma': 0.0078125,
         'kernel': 'rbf',
         'tol': 0.001
       }
    )
_svm_ = {'poly': {'best_estimator_': _poly_ , 
                 'best_params_': get_params(_poly_)
                 },
        'sigmoid': {'best_estimator_': _sigmoid_ , 
                    'best_params_': get_params(_sigmoid_)
                    }, 
        'rbf': {'best_estimator_': _rbf_ , 
                    'best_params_': get_params(_rbf_)
                    }, 
        'linear': {'best_estimator_': _linear_ , 
                    'best_params_': get_params(_linear_)
                    }, 
        }


#svmBinaryModels= (svm_lin, svm_poly, svm_sig, svm_rbf )

_svmBinaryModels = { 
        'lin': _linear, 
        'poly': _poly, 
        'sig': _sigmoid, 
        'rbf': _rbf 
        
    }

_lr = LogisticRegression (
    **{
        'penalty': 'l2',
        'C': 1.0
        }
    )
_knn = KNeighborsClassifier (
    **{
        'p': 2,
        'n_neighbors': 9,
         'metric': 'manhattan'
        }
    )
_dt = DecisionTreeClassifier (
    **{
        'max_depth': 7,
        'criterion': 'entropy'
        }
    )

#base_learners= [LRc, KNNc , DTc , svm_rbf] 
_baseModels = (_lr, _knn, _dt, _rbf) 

# Best vmodel 
_vmodels = [(f'{m.__class__.__name__}', m) for m in _baseModels ] 

_vtm =VotingClassifier(
    estimators =_vmodels ,
    voting ='hard'
    ) # 0.8517441860465117
_rfm = RandomForestClassifier (
    **{
        'n_estimators': 350,
       'max_depth': 16,
       'criterion': 'entropy', 
       'bootstrap': True
       }
    ) # 0.8430232558139535
_rfoobm=RandomForestClassifier (
    **{
       'oob_score': True,
       'n_estimators': 100,
       'max_depth': 15, 
       'criterion': 'gini',
       'bootstrap': True
       }
    ) # 0.8488372093023256
_extreem = ExtraTreesClassifier (
    **{
       'oob_score': False,
       'n_estimators': 450,
       'max_depth': 19, 
       'criterion': 'entropy',
       'bootstrap': True
       }
    ) # 0.8284883720930233
_extreeoobm = ExtraTreesClassifier (
    **{
       'oob_score': True,
       'n_estimators': 300,
       'max_depth': 18,
       'criterion': 'entropy',
       'bootstrap': True
       }
    ) # 0.8343023255813954
_pastm = BaggingClassifier (
    **{
       'n_estimators': 150,
       'bootstrap': False, 
       'base_estimator': SVC(C=2.0, coef0=0, degree=1, gamma=0.125)
       }
    ) # 0.8517441860465117
_adam = AdaBoostClassifier (
    **{
       'n_estimators': 50, 
       'learning_rate': 0.06, 
       'base_estimator': DecisionTreeClassifier(
            criterion='entropy', max_depth=7)
       }
    ) # 0.8546 
_xgboostm = XGBClassifier (
    **{
       'n_estimators': 300,
       'max_depth': 2,
       'learning_rate': 0.07, 
       'gamma': 1.5,
       'booster': 'gbtree'
       }
    ) # 0.8633
_stcm = StackingClassifier (
    **{
       'estimators': _vmodels , 
       'final_estimator':LogisticRegression (C=1, penalty= 'l2')
       }
    ) # 0.86662

_ensembleModels = (
    _lr, 
    _knn, 
    _dt, 
    _vtm,
    _rfm,
    _rfoobm,
    _extreem, 
    _extreeoobm,
    _pastm, 
    _adam,
    _xgboostm,
    _stcm
    )

def _set2dict ( *objs , s= False, names = None ): 
    """ Create an object dict. Each subset of dict element constitues a 
    subclass. Each dict key must compose an attribute objects of metaclass. """
    d={}
    names = names or [' ' for i in range(len(objs))]
    for o, ne in zip( objs, names)  : 
        if s: 
            d_ ={}
            for o, ne in zip( objs, names)  : 
                for k, value in  o.items(): 
                    d_[k] = type (k, (), value )
        
                d[ne]= type (ne, (), d_ )
                d_={}
        else: 
           for  o  in objs:
       
               n= o.__class__.__name__.replace(
                       'Classifier', '') +'_' if 'oob' in str(
                           o)else  o.__class__.__name__.replace(
                               'Classifier', '') 
               v = {'best_estimator_': o , 
                     'best_params_': get_params(o)}        
         
               d[n] = type (n, (), v)
               
    return d

_pMODELS  = {
    **_set2dict(*(_svm, _svm_), s= True, names = ('SVM', 'SVM_')), 
    **_set2dict(*_ensembleModels)
        } 
