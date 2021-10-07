# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 09:46:59 2021
..synopsis: 
    plot receiving operating characteric(ROC) classifiers. 
    To plot multiples classifiers, provide a list of classifiers. 

@author: @Daniel03
"""
# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC ,  LinearSVC 

from watex.viewer.mlplot import MLPlots 
# modules below are imported for testing scripts.
# Not usefull to import since you provided your own dataset.
from watex.datasets import fetch_data 
X_prepared, y_prepared = fetch_data('Bagoue dataset prepared')

random_state =42 
# test dirty classifer "stochastic gradient descent" 
# prediction method =`decision_function`.

# Compared with dirty RANDOM FOREST with decision method 'predict_proba`
# forest_clf =RandomForestClassifier(random_state=random_state)

logreg_clf = LogisticRegression(random_state =random_state)
svc_clf = SVC(C=100, gamma=1e-2, kernel='rbf', random_state =random_state) 

# trainset 
trainset= X_prepared#X_train_2
# y -labels 
y_array = y_prepared
# K-Fold cross validation
cv =7 

# `classe_` argument is provied if y are not binarized. i.e 
# created a binary attribute for each flow classes; one attribute 
#equal to 1 when others categories equal to 0.
classe_category = 1 

# plot_keywords arguments 

plot_kws ={'fig_size':(8,8), 
           'lw':3, 
           'lc':(.9, 0, .8), 
           'font_size':7., 
           'show_grid' :True,          # visualize grid 
           'galpha' :0.2,              # grid alpha 
           'glw':.5,                   # grid line width 
           'gwhich' :'minor',           # minor ticks
            # 'fs' :3.,                   # coeff to manage font_size 
            'leg_kws': {'loc':'lower right', 
                        'fontsize':15.}
            }
# classifiers with their methods are put into tuples. 
# IF classifier name is set to None. Will find name automatically.
clfs =[('SVM', svc_clf, "decision_function" ), 
      ('Logistic regression', logreg_clf, "predict_proba")]
    #  can be 
    #  clfs =[(None, sgd_clf, "decision_function" ), 
    #   (None, forest_clf, "predict_proba")]
# call MLPlots objects
mlObj= MLPlots(**plot_kws)

# roc_curves_kws 
roc_curves_kws =dict()

mlObj.ROC_curve_(clf = clfs, 
                 X= trainset, 
                y = y_array ,
                classe_=classe_category, 
                cv=cv,
                **roc_curves_kws)