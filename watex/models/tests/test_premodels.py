# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import watex as wx 
from watex.models.premodels import pModels
from watex.models.premodels import p  
X, y = wx.fetch_data ('bagoue prepared')
# xxxxxxxxxxxxxxxx predictionxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Xtrain, Xtest, ytrain, ytest = wx.sklearn.train_test_split(X, y )

def test_pModels (): 
    # fetch the  the pretrained Adaboost model 
    pm= pModels (model ='ada') 
    pm.fit() 
    pm.AdaBoost.best_estimator_ 
    # ... AdaBoostClassifier(base_estimator=LogisticRegression(), learning_rate=0.09,
    # 				   n_estimators=500)
    pm.model = 'vot' 
    pm.fit() 
    print( pm.Voting.best_estimator_ ) 
    # ... VotingClassifier(estimators=[('lr', LogisticRegression()),
    # ...                             ('knn',
    # ...                              KNeighborsClassifier(metric='manhattan',
    # ...                                                   n_neighbors=9)),
    # ...                             ('dt',
    # ...                              DecisionTreeClassifier(criterion='entropy',
    # ...                                                     max_depth=7)),
    # ...                             ('pSVM',
    # ...                              SVC(C=2.0, coef0=0, degree=1, gamma=0.125))])
    p2 = pModels(model='extree', oob_score= True ).fit()
    print( p2.ExtraTrees.best_estimator_ ) 
    # ... ExtraTreesClassifier(bootstrap=True, criterion='entropy', max_depth=18,
    # 					 max_features='auto', n_estimators=300, oob_score=True)
  
    pm.fit(Xtrain, ytrain ) 
    ypred = pm.predict( Xtest )
    print(wx.sklearn.accuracy_score(ytest , ypred)) 
    # 0.7674418604651163
    p2.fit(Xtrain, ytrain ) 
    ypred = p2.predict( Xtest )
    print(wx.sklearn.accuracy_score(ytest , ypred)) 
    # 0.7790697674418605
    
def test_p (): 
    
    p0= p.SVM.poly.best_estimator_
    print(p0)
    # ... SVC(C=128.0, coef0=7, degree=5, gamma=0.00048828125, kernel='poly', tol=0.01)
    p1= p.XGB.best_estimator_ 
    print(p1)
    # ... XGBClassifier(base_score=None, booster='gbtree', colsample_bylevel=None,
    # 			  colsample_bynode=None, colsample_bytree=None,
    # 			  ... 
    # 			  tree_method=None, validate_parameters=None, verbosity=None)
    p2=  p.RandomForest.best_estimator_ 
    print(p2)
    # ... RandomForestClassifier(criterion='entropy', max_depth=16, n_estimators=350)
    print( p.keys ) 
    # ... ('SVM', 'SVM_', 'LogisticRegression', 'KNeighbors', 'DecisionTree',
    # 	 'Voting', 'RandomForest', 'RandomForest_', 'ExtraTrees', 
    # 	 'ExtraTrees_', 'Bagging', 'AdaBoost', 'XGB', 'Stacking'
    # 	 ) 
    # fetch the pretrained LogisticRegression best parameters 
    p3= p.LogisticRegression.best_estimator_ 
    print(p.LogisticRegression.best_params_ )
    # ... {'penalty': 'l2',
    #      'dual': False,
    #      'tol': 0.0001,
    #      'C': 1.0,
    #      'fit_intercept': True,
    #      'intercept_scaling': 1,
    #      'class_weight': None,
    #      'random_state': None,
    #      'solver': 'lbfgs',
    #      'max_iter': 100,
    #      'multi_class': 'auto',
    #      'verbose': 0,
    #      'warm_start': False,
    #      'n_jobs': None,
    #      'l1_ratio': None
    #  }
    # fetcth the pretrained RandomForest with out-of-bagg equal to True 
    p4=p.RandomForest.best_estimator_  
    print(p4 )
    # ... RandomForestClassifier(max_depth=15, oob_score=True)
    
    # xxxxxxxxxxxxx prediction xxxxxxxxxxxxxxxxxxxxxx
    
    p0.fit(Xtrain, ytrain ) 
    ypred = p0.predict( Xtest )
    print(wx.sklearn.accuracy_score(ytest , ypred)) 
    # 0.7209302325581395
    p1.fit(Xtrain, ytrain ) 
    ypred = p1.predict( Xtest )
    print(wx.sklearn.accuracy_score(ytest , ypred)) 
    # 0.6627906976744186
    p2.fit(Xtrain, ytrain ) 
    ypred = p2.predict( Xtest )
    print(wx.sklearn.accuracy_score(ytest , ypred)) 
    # 0.6744186046511628
    p3.fit(Xtrain, ytrain ) 
    ypred = p3.predict( Xtest )
    print(wx.sklearn.accuracy_score(ytest , ypred)) 
    # 0.6162790697674418
    p4.fit(Xtrain, ytrain ) 
    ypred = p4.predict( Xtest )
    print(wx.sklearn.accuracy_score(ytest , ypred)) 
    # 0.6511627906976745

# if __name__=='__main__': 
    
#     test_pModels () 
        
#     test_p() 