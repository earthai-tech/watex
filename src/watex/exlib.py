# -*- coding: utf-8 -*-
# scikit-learn module importation
# Created on Thu May 19 13:40:53 2022 

import sys 
import warnings 
import inspect 
import subprocess
   
from sklearn.base import(
    BaseEstimator,
    TransformerMixin
)
from sklearn.compose import ( 
    make_column_transformer, 
    make_column_selector 
)
from sklearn.decomposition import (
    PCA ,
    IncrementalPCA,
    KernelPCA
)
  
from sklearn.feature_selection import ( 
    SelectKBest, 
    f_classif
) 

from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (
    LogisticRegression, 
    SGDClassifier
    )
from sklearn.metrics import ( 
    confusion_matrix,
    classification_report ,
    mean_squared_error, 
    f1_score,
    precision_recall_curve, 
    precision_score,
    recall_score, 
    roc_auc_score, 
    roc_curve
)  
from sklearn.model_selection import ( 
    train_test_split , 
    validation_curve, 
    StratifiedShuffleSplit , 
    RandomizedSearchCV,
    GridSearchCV, 
    learning_curve , 
    cross_val_score,
    cross_val_predict 
)
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.pipeline import (
    Pipeline, 
    make_pipeline ,
    FeatureUnion
)
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures, 
    RobustScaler ,
    OrdinalEncoder, 
    StandardScaler,
    MinMaxScaler, 
    LabelBinarizer,
    LabelEncoder,
) 

from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier


def is_installing (
        module: str , 
        upgrade: bool=True , 
        action: bool=True, 
        DEVNULL: bool=False,
        verbose: int=0,
        **subpkws
    )-> bool: 
    """ Install or uninstall a module/package using the subprocess 
    under the hood.
    
    Parameters 
    ------------
    module: str,
        the module or library name to install using Python Index Package `PIP`
    
    upgrade: bool,
        install the lastest version of the package. *default* is ``True``.   
        
    DEVNULL:bool, 
        decline the stdoutput the message in the console 
    
    action: str,bool 
        Action to perform. 'install' or 'uninstall' a package. *default* is 
        ``True`` which means 'intall'. 
        
    verbose: int, Optional
        Control the verbosity i.e output a message. High level 
        means more messages. *default* is ``0``.
         
    subpkws: dict, 
        additional subprocess keywords arguments 
    Returns 
    ---------
    success: bool 
        whether the package is sucessfully installed or not. 
        
    Example
    --------
    >>> from watex import is_installing
    >>> is_installing(
        'tqdm', action ='install', DEVNULL=True, verbose =1)
    >>> is_installing(
        'tqdm', action ='uninstall', verbose =1)
    """
    #implement pip as subprocess 
    # refer to https://pythongeeks.org/subprocess-in-python/
    if not action: 
        if verbose > 0 :
            print("---> No action `install`or `uninstall`"
                  f" of the module {module!r} performed.")
        return action  # DO NOTHING 
    
    success=False 

    action_msg ='uninstallation' if action =='uninstall' else 'installation' 

    if action in ('install', 'uninstall', True) and verbose > 0:
        print(f'---> Module {module!r} {action_msg} will take a while,'
              ' please be patient...')
        
    cmdg =f'<pip install {module}> | <python -m pip install {module}>'\
        if action in (True, 'install') else ''.join([
            f'<pip uninstall {module} -y> or <pip3 uninstall {module} -y ',
            f'or <python -m pip uninstall {module} -y>.'])
        
    upgrade ='--upgrade' if upgrade else '' 
    
    if action == 'uninstall':
        upgrade= '-y' # Don't ask for confirmation of uninstall deletions.
    elif action in ('install', True):
        action = 'install'

    cmd = ['-m', 'pip', f'{action}', f'{module}', f'{upgrade}']

    try: 
        STDOUT = subprocess.DEVNULL if DEVNULL else None 
        STDERR= subprocess.STDOUT if DEVNULL else None 
    
        subprocess.check_call(
            [sys.executable] + cmd, stdout= STDOUT, stderr=STDERR,
                              **subpkws)
        if action in (True, 'install'):
            # freeze the dependancies
            reqs = subprocess.check_output(
                [sys.executable,'-m', 'pip','freeze'])
            [r.decode().split('==')[0] for r in reqs.split()]

        success=True
        
    except: 

        if verbose > 0 : 
            print(f'---> Module {module!r} {action_msg} failed. Please use'
                f' the following command: {cmdg} to manually do it.')
    else : 
        if verbose > 0: 
            print(f"{action_msg.capitalize()} of `{module}` "
                      "and dependancies was successfully done!") 
        
    return success 


_HAS_ENSEMBLE_=False


try : 
    from sklearn.ensemble import  (  
        RandomForestClassifier,
        AdaBoostClassifier, 
        VotingClassifier, 
        BaggingClassifier,
        StackingClassifier , 
        ExtraTreesClassifier, 
        )
except: 
    from .exceptions import ScikitLearnImportError 

    _HAS_ENSEMBLE_ = is_installing('sklearn')
    
    if not _HAS_ENSEMBLE_: 
        warnings.warn(
            'Autoinstallation of `Ensemble` methods from '
            ':mod:`sklearn.ensemble` failed. Try to install it '
            'manually.', ImportWarning)
        raise ScikitLearnImportError('Module importation error.')

else : 
    
    skl_ensemble_= [
        RandomForestClassifier,
        AdaBoostClassifier,
        VotingClassifier,
        BaggingClassifier, 
        StackingClassifier
        ]
    
    _HAS_ENSEMBLE_=True
    

IS_GBM = False 
try : 
    import xgboost 
except : 
    warnings.warn("Gradient Boosting Machine is installing."
                  " Please wait... ")
    print('!-> Please wait for Gradient Boosting Machines to be installed...')
    IS_GBM = is_installing('xgboost')
    if IS_GBM : 
        print("!---> 'xgboost' installation is complete")
    else : 
        warnings.warn ("'xgoost' installation failed.")
        print("Fail to install 'xgboost', please install it mannualy.")
else :IS_GBM =True 
 
if IS_GBM: 
    from xgboost import XGBClassifier 
        
    
def get_params (obj: object 
                ) -> dict: 
    """
    Get object parameters. 
    
    Object can be callable or instances 
    
    :param obj: object , can be callable or instance 
    
    :return: dict of parameters values 
    
    :examples: 
    >>> from sklearn.svm import SVC 
    >>> from watex.tools.funcutils import get_params 
    >>> sigmoid= SVC (
        **{
            'C': 512.0,
            'coef0': 0,
            'degree': 1,
            'gamma': 0.001953125,
            'kernel': 'sigmoid',
            'tol': 1.0 
            }
        )
    >>> pvalues = get_params( sigmoid)
    >>> {'decision_function_shape': 'ovr',
         'break_ties': False,
         'kernel': 'sigmoid',
         'degree': 1,
         'gamma': 0.001953125,
         'coef0': 0,
         'tol': 1.0,
         'C': 512.0,
         'nu': 0.0,
         'epsilon': 0.0,
         'shrinking': True,
         'probability': False,
         'cache_size': 200,
         'class_weight': None,
         'verbose': False,
         'max_iter': -1,
         'random_state': None
     }
    """
    if hasattr (obj, '__call__'): 
        cls_or_func_signature = inspect.signature(obj)
        PARAMS_VALUES = {k: None if v.default is (inspect.Parameter.empty 
                         or ...) else v.default 
                    for k, v in cls_or_func_signature.parameters.items()
                    # if v.default is not inspect.Parameter.empty
                    }
    elif hasattr(obj, '__dict__'): 
        PARAMS_VALUES = {k:v  for k, v in obj.__dict__.items() 
                         if not (k.endswith('_') or k.startswith('_'))}
    
    return PARAMS_VALUES

# +----Defaults Models------+  
# def dModels (kind = 'svm'): 
#     """ Default pretrained models """
#     kind= str(kind).lower().strip() 

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
