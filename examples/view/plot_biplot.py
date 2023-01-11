"""
=======================================================
Plot bipolar with Principal component analysis (PCA)
=======================================================

visualizes all-in-one features from PCA analysis.
There is an implementation in R but there is no standard implementation
in Python. 
"""
# Author: L.Kouadio 
# Licence: BS3-clause 

#%%
from watex.analysis import nPCA
from watex.datasets import fetch_data
from watex.view import biPlot, pobj  # pobj is Baseplot instance 
X, y = fetch_data ('bagoue pca' )  # fetch pca data 
pca= nPCA (X, n_components= 2 , return_X= False ) # return PCA object 
components = pca.components_ [:2, :] # for two components 
biPlot (pobj, pca.X, components , y ) # pca.X is the reduced dim X 
# to change for instance line width (lw) or style (ls) 
# just use the baseplotobject (pobj)