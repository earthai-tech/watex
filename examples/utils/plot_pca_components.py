"""
======================================================
Plot principal components analysis (PCA) components 
======================================================

visualizes the PCA coefficients  as a heatmap  
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause 

# %% 
# Plot can be made with PCA objects 

from watex.datasets import fetch_data
from watex.utils.plotutils import plot_pca_components
from watex.analysis import nPCA 
X, _= fetch_data('bagoue pca') 
pca = nPCA (X, n_components=2, return_X =False)# to return object 
plot_pca_components (pca)

#%% 
# Plot made using the components and features individually 

components = pca.components_ 
features = pca.feature_names_in_
plot_pca_components (components, feature_names= features, 
                         cmap='jet_r')