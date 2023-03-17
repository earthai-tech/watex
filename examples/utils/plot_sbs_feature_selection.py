"""
=================================================
Plot feature selection with SBS 
=================================================

selects features using the sequential Backward Selection (SBS) algorithm. 
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause 

#%% 
# * SBS in action with fitted data 

import matplotlib.pyplot as plt 
from watex.exlib.sklearn import KNeighborsClassifier , train_test_split
from watex.datasets import fetch_data
from watex.base import SequentialBackwardSelection
from watex.utils.plotutils import plot_sbs_feature_selection
plt.style.use ('classic')

X, y = fetch_data('bagoue analysed') # data already standardized
Xtrain, Xt, ytrain,  yt = train_test_split(X, y)
knn = KNeighborsClassifier(n_neighbors=5)
sbs= SequentialBackwardSelection (knn)
sbs.fit(Xtrain, ytrain )
plot_sbs_feature_selection(sbs) 

#%%
# * Plot estimator with no prefit SBS  

plot_sbs_feature_selection(knn, Xtrain, ytrain) # yield the same result

# The above pplot indicates that performance is mostly achieved from 
# feature 3 to 4 before droppint around 60% with feature equals to 8 