"""
=================================================
Plot learning inspections  
=================================================

inspects multiple models from their learning curves. 
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause
 
#%%
from watex.datasets import fetch_data
from watex.models.premodels import p 
from watex.view.mlplot import plotLearningInspections 
# import sparse  matrix from Bagoue dataset 
X, y = fetch_data ('bagoue prepared') 
# import the two pretrained models from SVM 
models = [p.SVM.rbf.best_estimator_ , p.SVM.poly.best_estimator_]
plotLearningInspections (models , X, y, ylim=(0.7, 1.01) )