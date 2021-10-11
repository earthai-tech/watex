# -*- coding: utf-8 -*-
"""
.. synopsis: Create your model and fine tune its hyperparameters. 
    with your dataset.
    
Created on Fri Sep 24 21:28:48 2021

@author: @Daniel03
"""

from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.svm import SVC, LinearSVC 

from watex.viewer.mlplot import MLPlots 
from watex.modeling.validation import multipleGridSearches 
#  Test data
from watex.datasets import fetch_data 
X_prepared, y_prepared = fetch_data('Bagoue dataset prepared')

#cross validation Kfold 
cv = 7
# type of scores 
scoring ='neg_mean_squared_error'#accuracy'#'neg_mean_squared_error'#

# random state for estimator s
random_state =42 
# kind of grid search 
kind ='GridSearchCV'
#save to joblib 
save2joblib =True 
# differnts 
logreg_clf = LogisticRegression(random_state =random_state)
linear_svc_clf = LinearSVC(random_state =random_state)
sgd_clf = SGDClassifier(random_state = random_state)
svc_clf = SVC(random_state =random_state) 
# build estimators 
estimators = (svc_clf,linear_svc_clf, logreg_clf )

# plot fine tuned params: 
plot_fineTune =False
    
# save to joblib 
# once the best model found. save it to job lib
gridParams =([
        {'C':[1e-2, 1e-1, 1, 10, 100], 'gamma':[5, 2, 1, 1e-1, 1e-2, 1e-3],'kernel':['rbf']}, 
        {'kernel':['sigmoid'],'degree':[1, 3,5, 7], 'coef0':[1, 2, 3], 'C': [1e-2, 1e-1, 1, 10, 100]}
        ], 
        [{'C':[1e-2, 1e-1, 1, 10, 100], 'loss':['hinge']}], 
        [dict()],
        # [dict()]
    )

_clfs, _dclfs, joblib= multipleGridSearches(X= X_prepared,
                                         y= y_prepared,
                                         estimators = estimators, 
                                         grid_params = gridParams ,
                                         cv =cv, 
                                         scoring =scoring,
                                         verbose =1,
                                         save_to_joblib =save2joblib  
                                         )

if plot_fineTune : 
    # clfs =[(_clfs[i][1], _clfs[i][3]) for i in range(len(_clfs))]
    scores = [ _clfs[i][3] for i in range(len(_clfs))]
    clfs =['SVM:score mean=75.86%', 'LinearSVC:score mean= ', 'LogisticRegression:score mean=74.16%']
    
    plot_kws = {'fig_size':(12, 8),
        'lc':(.9,0.,.8),
            'lw' :3.,           # line width 
            'font_size':7.,
            'show_grid' :True,        # visualize grid 
           'galpha' :0.2,              # grid alpha 
           'glw':.5,                   # grid line width 
           'gwhich' :'major',          # minor ticks
            # 'fs' :3.,                 # coeff to manage font_size
            'xlabel':'Cross-validation (CV)', 
            'ylabel': 'Scores', 
            # 'ylim':[0.5,1.]
            }
    mlObj =MLPlots(**plot_kws)
    lcs_kws ={'lc':['k', 'k', 'k'], #(.9,0.,.8)
              'ls':['-', ':', '-.']}
    mlObj.plotModelvsCV(clfs =clfs, scores =scores, **lcs_kws)
