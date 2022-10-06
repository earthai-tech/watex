# -*- coding: utf-8 -*-
# @author: Kouadio K. Laurent alias Daniel03
"""
`Search Grid will be able to  fiddle with the hyperparameters until to 
find the great combination for model predictions. 

Created on Tue Sep 21 19:45:40 2021

@author: @Daniel03
"""
from pprint import pprint  

from sklearn.svm import SVC

from watex.models.validation import GridSearch
# modules below are imported for testing scripts.
# Not usefull to import at least you provided your own dataset.
from watex.datasets import fetch_data
X_prepared, y_prepared =fetch_data('Bagoue data prepared')


# set the SVM grid parameters 
grid_params = [
        {'C':[1e-2, 1e-1, 1, 10, 100],
          'gamma':[5, 2, 1, 1e-1, 1e-2, 1e-3],
         'kernel':['linear', 'sigmoid', 'poly', 'rbf'],
          'degree':[1, 3,5, 7],
          'coef0':[1, 2, 3] 
         }
        ]
#{'C': 100, 'coef0': 1, 'degree': 1, 'gamma': 0.01, 'kernel': 'rbf'}

# forest_clf = RandomForestClassifier(random_state =42)
# grid_search = SearchedGrid(forest_clf, grid_params, kind='RandomizedSearchCV')
# grid_search.fit(X= X_prepared , y = y_prepared)


cv =7

# kind of search : can be `RandomizedSearch CV
kindOfSearch = 'GridSearchCV'

# =============================================================================
#Grid search section. Comment this section if your want to evaluate your 
# best parameters found after search.
#==============================================================================
svc_clf = SVC(random_state=42,
                # C=10, gamma=1e-2, kernel ='poly', degree=7, coef0=2
              )
# grid_ keywords arguments 

grid_kws ={'scoring':'accuracy'} #[-0.26763848]'neg_mean_squared_error'#
grid_searchObj= GridSearch(svc_clf,
                           grid_params,
                           cv =cv, 
                           kind=kindOfSearch, 
                           **grid_kws)

grid_searchObj.fit(X= X_prepared , y = y_prepared)

pprint(grid_searchObj.best_params_ )

# cvres = grid_searchObj.cv_results_ 
# pprint(cvres)
# pprint(cvres['mean_test_score'])


# if your estimator has a `feature_importances_`attributes, call it by 
# uncomment the section below. If return None, mean the estimator doesnt have 
#a `feature_importances_` attributes. 

#pprint(grid_searchObj.feature_importances_ )
# best SVMmodels
#{'C': 100, 'coef0': 1, 'degree': 1, 'gamma': 0.01, 'kernel': 'rbf'}

# =============================================================================
#Uncomment the section  below and evalaute your model with the 
# best configuration after drid search 
# =============================================================================
# from sklearn.model_selection import cross_val_score 

# svc_clf = SVC(random_state=42,
#                 **grid_searchObj.best_params_
#               )
# svc_clf_scores = cross_val_score(svc_clf, X_prepared ,
#                                   y_prepared, cv =cv ,
#                                   scoring = 'accuracy')
# print(svc_clf_scores)
