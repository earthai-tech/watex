"""
=================================================
Plot feature importance with Randomforest 
=================================================

plot the features importance with RandomForest estimator  
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause 

#%%
import matplotlib.pyplot as plt 
plt.style.use ('classic')
from watex.datasets import fetch_data
from watex.exlib.sklearn import RandomForestClassifier 
from watex.utils.plotutils import plot_rf_feature_importances 
X, y = fetch_data ('bagoue analysed' ) 
plot_rf_feature_importances (
    RandomForestClassifier(), X=X, y=y , sns_style=False)