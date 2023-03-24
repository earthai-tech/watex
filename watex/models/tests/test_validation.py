# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import numpy as np 
import watex as wx 
from pprint import pprint 
from watex.datasets import load_bagoue 
from watex.models import BaseEvaluation 
from watex.models import GridSearchMultiple , displayFineTunedResults
from watex.exlib import ( 
    LinearSVC, SGDClassifier, SVC, LogisticRegression, RandomForestClassifier)
from watex.models.validation import GridSearch, naive_evaluation
from watex.models.validation  import get_best_kPCA_params

X, y  = wx.fetch_data ('bagoue prepared') 
X
# ... <344x18 sparse matrix of type '<class 'numpy.float64'>'
# ... with 2752 stored elements in Compressed Sparse Row format>
# As example, we can build 04 estimators and provide their 
# grid parameters range for fine-tuning as ::
random_state=42
logreg_clf = LogisticRegression(random_state =random_state)
linear_svc_clf = LinearSVC(random_state =random_state)
sgd_clf = SGDClassifier(random_state = random_state)
svc_clf = SVC(random_state =random_state) 
estimators =(svc_clf,linear_svc_clf, logreg_clf, sgd_clf )
grid_params= ([dict(C=[1e-2, 1e-1, 1, 10, 100], 
                        gamma=[5, 2, 1, 1e-1, 1e-2, 1e-3],kernel=['rbf']), 
                   dict(kernel=['poly'],degree=[1, 3,5, 7], coef0=[1, 2, 3],
                        C= [1e-2, 1e-1, 1, 10, 100])],
                [dict(C=[1e-2, 1e-1, 1, 10, 100], loss=['hinge'])], 
                [dict()], # we just no provided parameter for demo
                [dict()]
                )
#Now  we can call :class:`watex.models.GridSearchMultiple` for
# training and self-validating as:
    
def test_GridSearchMultiple():
    gobj = GridSearchMultiple(estimators = estimators, 
                           grid_params = grid_params ,
                           cv =4, 
                           scoring ='accuracy', 
                           verbose =1,   #> 7 put more verbose 
                           savejob=False ,  # set true to save job in binary disk file.
                           kind='GridSearchCV').fit(X, y)
    # Once the parameters are fined tuned, we can display the fined tuning 
    # results using displayFineTunedResults`` function
    displayFineTunedResults (gobj.models.values_) 
    


def test_BaseEvaluation(): 
    X, y = load_bagoue (as_frame =True ) 
    # categorizing the labels 
    yc = wx.smart_label_classifier (y , values = [1, 3, 10 ], 
                                     # labels =['FR0', 'FR1', 'FR2', 'FR4'] 
                                     ) 
    # drop the subjective columns ['num', 'name'] 
    X = X.drop (columns = ['num', 'name']) 
    # X = wx.cleaner (X , columns = 'num name', mode='drop') 
    X.columns 
    # Index(['shape', 'type', 'geol', 'east', 'north', 'power', 'magnitude', 'sfi',
    #        'ohmS', 'lwi'],
    #       dtype='object')
    X =  wx.naive_imputer ( X, mode ='bi-impute') # impute data 
    # create a pipeline for X 
    pipe = wx.make_naive_pipe (X) 
    Xtrain, Xtest, ytrain, ytest = wx.sklearn.train_test_split(X, yc) 
    b = BaseEvaluation (estimator= wx.sklearn.RandomForestClassifier, 
                            scoring = 'accuracy', pipeline = pipe)
    b.fit(Xtrain, ytrain ) # accepts only array 
    b.cv_scores_ 
    # Out[174]: array([0.75409836, 0.72131148, 0.73333333, 0.78333333])
    ypred = b.predict(Xtest)
    scores = wx.sklearn.accuracy_score (ytest, ypred) 
    print(scores )
    
def test_get_best_kPCA_params(): 
    X, y=wx.fetch_data('Bagoue analysis data')
    param_grid=[dict(
        kpca__gamma=np.linspace(0.03, 0.05, 10),
        kpca__kernel=["rbf", "sigmoid"]
        )]
    clf =wx.sklearn.Pipeline([
        ('kpca', wx.sklearn.KernelPCA(n_components=2)), 
        ('log_reg', wx.sklearn.LogisticRegression())
         ])
    kpca_best_params =get_best_kPCA_params(
                X,y=y,scoring = 'accuracy',
                n_components= 2, 
                clf=clf, 
                param_grid=param_grid)
    
    print ( kpca_best_params) 
# ... {{'kpca__gamma': 0.03, 'kpca__kernel': 'rbf'}}

def test_GridSearch ():
    X_prepared, y_prepared =wx.fetch_data ('bagoue prepared')
    grid_params = [ dict(
            n_estimators=[3, 10, 30], max_features=[2, 4, 6, 8]), 
            dict(bootstrap=[False], n_estimators=[3, 10], 
                                 max_features=[2, 3, 4])
           ]
    forest_clf = RandomForestClassifier()
    grid_search = GridSearch(forest_clf, grid_params)
    grid_search.fit(X= X_prepared,y =  y_prepared,)
    pprint(grid_search.best_params_ )
    # {{'max_features': 8, 'n_estimators': 30}}
    pprint(grid_search.cv_results_)
    
def test_naive_evaluation (): 

    clf = wx.sklearn.DecisionTreeClassifier() 
    scores = naive_evaluation(clf, X, y , cv =4 , display ='on' )
    print(scores )
    
# if __name__=='__main__': 
#     test_naive_evaluation()
#     test_GridSearchMultiple()
#     test_BaseEvaluation() 
#     test_get_best_kPCA_params() 
#     test_GridSearch() 