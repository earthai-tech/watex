# -*- coding: utf-8 -*-
# scikit-learn module importation
# Created on Thu May 19 13:40:53 2022 
import warnings 

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
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ( 
    confusion_matrix,
    classification_report ,
    mean_squared_error 
    #, f1_score,
)  
from sklearn.model_selection import ( 
    train_test_split , 
    validation_curve, 
    StratifiedShuffleSplit 
)

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures, 
    RobustScaler ,
    OrdinalEncoder, 
    StandardScaler,
    MinMaxScaler
) 

from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier

_HAS_ENSEMBLE_=False

try : 
    from sklearn.ensemble import  (  
        RandomForestClassifier,
        AdaBoostClassifier, 
        VotingClassifier, 
        BaggingClassifier,
        StackingClassifier 
        )
except: 
    from .exceptions import ScikitLearnImportError 
    from .utils.funcutils import is_installing 
    _HAS_ENSEMBLE_ = is_installing('sklearn')
    
    if not _HAS_ENSEMBLE_: 
        warnings.warn(
            'Autoinstallation of `Ensemble` methods from '
            ':mod:`sklearn.ensemble` failed. Try to install it '
            'manually.', ImportWarning)
        raise ScikitLearnImportError('Module importation error.')

else : 
    
    skl_ensemble__= [
        RandomForestClassifier,
        AdaBoostClassifier,
        VotingClassifier,
        BaggingClassifier, 
        StackingClassifier
        ]
    
    _HAS_ENSEMBLE_=True 
    

