"""
=================================================
Plot learning curves  
=================================================

plots inline multiple models learning curves. 
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause 

#%%
# * Plot metaestimator already cross-validated. 

from watex.models.premodels import p 
from watex.datasets import fetch_data 
from watex.utils.plotutils import plot_learning_curves
X, y = fetch_data ('bagoue prepared') # yields a sparse matrix 
# let collect 04 estimators already cross-validated from SVMs
models = [ p.SVM.linear , p.SVM.rbf , p.SVM.sigmoid , p.SVM.poly ]
plot_learning_curves (models, X, y, cv=4, sns_style = 'darkgrid')
#%%
# * Plot  multiples models not crossvalidated yet. It take a little bit times 

from watex.exlib.sklearn import (
    LogisticRegression, 
    RandomForestClassifier, 
    SVC , KNeighborsClassifier 
    )
models =[LogisticRegression(), RandomForestClassifier(), SVC() ,
             KNeighborsClassifier() 
             ]
plot_learning_curves (models, X, y, cv=4, sns_style = 'darkgrid')