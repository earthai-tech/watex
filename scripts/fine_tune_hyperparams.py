# -*- coding: utf-8 -*-
"""
.. synopsis: Create your model and fine tune their hyperparameters. 
    with your dataset.
    
Created on Fri Sep 24 21:28:48 2021

@author: @Daniel03
"""

from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.svm import SVC, LinearSVC 

from watex.datasets.data_training import multipleGridSearches 
from watex.datasets import fetch_data 
X_prepared, y_prepared = fetch_data('Bagoue dataset prepared')

#cross validation Kfold 
cv = 7
# type of scores 
scoring ='accuracy'#'neg_mean_squared_error'#'neg_mean_squared_error'#

# random state for estimator s
random_state =42 
# kind of grid search 
kind ='GridSearchCV'

# differnts 
logreg_clf = LogisticRegression(random_state =random_state)
linear_svc_clf = LinearSVC(random_state =random_state)
sgd_clf = SGDClassifier(random_state = random_state)
svc_clf = SVC(random_state =random_state) 
# build estimators 
estimators = (svc_clf, logreg_clf )

# save to joblib 
# once the best model found. save it to job lib
gridParams =([
        {'C':[1e-2, 1e-1, 1, 10, 100], 'gamma':[5, 2, 1, 1e-1, 1e-2, 1e-3],'kernel':['rbf']}, 
        {'kernel':['poly'],'degree':[1, 3,5, 7], 'coef0':[1, 2, 3], 'C': [1e-2, 1e-1, 1, 10, 100]}
        ], 
        # [{'C':[1e-2, 1e-1, 1, 10, 100], 'loss':['hinge']}], 
        [dict()],
        # [dict()]
    )

_clfs, _dclfs, joblib= multipleGridSearches(X= X_prepared,
                                         y= y_prepared,
                                         estimators = estimators, 
                                         grid_params = gridParams ,
                                         cv =cv, 
                                         scoring =scoring,
                                         verbose =1,save_to_joblib =True, 
                                         )