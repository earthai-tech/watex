"""
===============================================
PCA vs Factor Analysis with scedatic noises
===============================================

computes the PCA and Factor Analysis 
scores from training :math:`X` and compare  
the probabilistic PCA and FA  models errors by adding scedatic 
(homo/hereto) noises. 
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

#%%
# fetch the analysed data 
from watex.analysis import pcavsfa 
from watex.datasets import fetch_data 
X, _=fetch_data('Bagoue analysed data')
pcavsfa (X) 