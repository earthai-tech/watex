"""
=================================================
Plot single learning inspection 
=================================================

inspects model from its learning curve.
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

#%%

from watex.datasets import fetch_data
from watex.models.premodels import p 
from watex.view.mlplot import plotLearningInspection 
# import sparse  matrix from Bagoue datasets 
X, y = fetch_data ('bagoue prepared') 
# import the  pretrained Radial Basis Function (RBF) from SVM 
plotLearningInspection (p.SVM.rbf.best_estimator_  , X, y )