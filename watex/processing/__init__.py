
import warnings

from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import SGDClassifier 
from sklearn.neighbors import KNeighborsClassifier 

from sklearn.svm import SVC 

from watex.utils._watexlog import watexlog 

_HAS_ENSEMBLE_=False 

try : 
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier 
    from sklearn.ensemble import VotingClassifier, BaggingClassifier, StackingClassifier
except: 
    warnings.warn('Trying importation of :mod:`sklearn.ensemble` failed. ')
    watexlog().get_watex_logger(__name__).debug(
        'Try to import :mod:`sklearn.ensemble` failed!')
else : 
    skl_ensemble__= [RandomForestClassifier, AdaBoostClassifier, VotingClassifier,
               BaggingClassifier, StackingClassifier ]
    
    _HAS_ENSEMBLE_=True 