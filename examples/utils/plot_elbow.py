"""
======================================================
Plot elbow
======================================================

visualize the elbow method to find the optimal number of cluster for a given data. 
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause 

#%%
from watex.datasets import load_hlogs 
from watex.utils.plotutils import plot_elbow 
# get the only resistivy and gamma-gama values for example
res_gamma = load_hlogs ().frame[['resistivity', 'gamma_gamma']]  
plot_elbow(res_gamma, n_clusters=11)