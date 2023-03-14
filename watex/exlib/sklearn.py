# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Scikit-learn external 
=======================

The module gives a sypnosis of the module suse throughout the watex packages. 
The mecanism of importation associates 'exlib' to 'sklearn' like 
`watex.exlib.sklearn` to different the estimators classes from scikit and 
others machines learnings algorithms. 

"""
sklearndoc = type ('sklearndoc', () , dict (
    __doc__ ="""\ 
Machine Learning in Python

Scikit-learn (Sklearn) is the most useful and robust library for machine 
learning in Python. It provides a selection of efficient tools for machine 
learning and statistical modeling including classification, regression, 
clustering and dimensionality reduction via a consistence interface in Python. 
This library, which is largely written in Python, is built upon NumPy, SciPy 
and Matplotlib.

It was originally called scikits.learn and was initially developed by David 
Cournapeau as a Google summer of code project in 2007. Later, in 2010, 
Fabian Pedregosa, Gael Varoquaux, Alexandre Gramfort, and Vincent Michel,
from FIRCA (French Institute for Research in Computer Science and Automation), 
took this project at another level and made the first public release 
(v0.1 beta) on 1st Feb. 2010. At this time, itts version history is listed above 

    * May 2019: scikit-learn 0.21.0
    * March 2019: scikit-learn 0.20.3
    * December 2018: scikit-learn 0.20.2        
    * November 2018: scikit-learn 0.20.1        
    * September 2018: scikit-learn 0.20.0        
    * July 2018: scikit-learn 0.19.2        
    * July 2017: scikit-learn 0.19.0        
    * September 2016. scikit-learn 0.18.0        
    * November 2015. scikit-learn 0.17.0        
    * March 2015. scikit-learn 0.16.0        
    * July 2014. scikit-learn 0.15.0        
    * August 2013. scikit-learn 0.14

Installation
-------------
If you already installed NumPy and Scipy, following are the two easiest 
ways to install scikit-learn. Following command can be used to install 
scikit-learn via::
    
    * Using pip
    
    :code:`pip install -U scikit-learn`
    
    * Using conda
    
    :conda:`install scikit-learn`
    
On the other hand, if NumPy and Scipy is not yet installed on your Python 
workstation then, you can install them by using either pip or conda.

Another option to use scikit-learn is to use Python distributions like 
Canopy and Anaconda because they both ship the latest version of scikit-learn.

References 
----------
.. https://scikit-learn.org/stable/index.html

"""
    ) 
)
    
from ..utils._dependency import import_optional_dependency

_HAS_ENSEMBLE_=False

msg= ("Sckit-learn <'sklearn'> is used throughout the watex package especially"
      " for the prediction modules. It is recommended to install it.")
try : 
    from sklearn.ensemble import RandomForestClassifier
except: 
    from ..exceptions import ScikitLearnImportError 
    import_optional_dependency ("sklearn", extra = msg , min_version="0.18", 
                                exception= ScikitLearnImportError 
                                )
else :
    _HAS_ENSEMBLE_=True
    
from sklearn.ensemble import  (  
    AdaBoostClassifier, 
    VotingClassifier, 
    BaggingClassifier,
    StackingClassifier , 
    ExtraTreesClassifier, 
    )

skl_ensemble_= [
    RandomForestClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    BaggingClassifier, 
    StackingClassifier, 
    ExtraTreesClassifier, 
    ]
        
from sklearn.base import(
    BaseEstimator,
    TransformerMixin, 
    ClassifierMixin, 
    clone 
)
from sklearn.cluster import KMeans
from sklearn.compose import ( 
    make_column_transformer, 
    make_column_selector , 
    ColumnTransformer
)
from sklearn.covariance import ( 
    ShrunkCovariance, 
    LedoitWolf
    )
from sklearn.decomposition import (
    PCA ,
    IncrementalPCA,
    KernelPCA, 
    FactorAnalysis
) 
from sklearn.dummy import DummyClassifier 
from sklearn.feature_selection import ( 
    SelectKBest, 
    f_classif, 
    SelectFromModel 
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
    accuracy_score,
    precision_recall_curve, 
    precision_score,
    recall_score, 
    roc_auc_score, 
    roc_curve, 
    silhouette_samples, 
    make_scorer,
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
    FeatureUnion, 
    _name_estimators, 
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
    Normalizer
) 
from sklearn.svm import ( 
    SVC, 
    LinearSVC, 
    LinearSVR 
    )  
from sklearn.tree import DecisionTreeClassifier


__all__=[
    "BaseEstimator",
    "TransformerMixin",
    "ClassifierMixin", 
    "clone", 
    "KMeans", 
    "make_column_transformer",
    'make_column_selector' , 
    'ColumnTransformer',
    'ShrunkCovariance', 
    'LedoitWolf', 
    'FactorAnalysis',
    'PCA' ,
    'IncrementalPCA',
    'KernelPCA', 
    'DummyClassifier', 
    'SelectKBest', 
    'f_classif',
    'SelectFromModel', 
    'SimpleImputer',
    'permutation_importance',
    'LogisticRegression', 
    'SGDClassifier',
    'confusion_matrix',
    'classification_report' ,
    'mean_squared_error', 
    'f1_score',
    'accuracy_score',
    'precision_recall_curve', 
    'precision_score',
    'recall_score', 
    'roc_auc_score', 
    'roc_curve',
    'silhouette_samples', 
    'make_scorer',
    'train_test_split' , 
    'validation_curve', 
    'StratifiedShuffleSplit' , 
    'RandomizedSearchCV',
    'GridSearchCV', 
    'learning_curve' , 
    'cross_val_score',
    'cross_val_predict',
    'KNeighborsClassifier',
    'Pipeline', 
    'make_pipeline' ,
    'FeatureUnion', 
    '_name_estimators',
    'OneHotEncoder',
    'PolynomialFeatures', 
    'RobustScaler' ,
    'OrdinalEncoder', 
    'StandardScaler',
    'MinMaxScaler', 
    'LabelBinarizer',
    'Normalizer',
    'LabelEncoder',
    'SVC', 
    'LinearSVC', 
    'LinearSVR', 
    'DecisionTreeClassifier',
    'RandomForestClassifier',
    'AdaBoostClassifier', 
    'VotingClassifier', 
    'BaggingClassifier',
    'StackingClassifier' , 
    'ExtraTreesClassifier', 
    'skl_ensemble_', 
    'sklearndoc', 
    '_HAS_ENSEMBLE_'
    ]


