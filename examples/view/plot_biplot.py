"""
=======================================================
Plot bipolar with Principal component analysis (PCA)
=======================================================

visualizes all-in-one features from PCA analysis. 
"""
# Author: L.Kouadio 
# Licence: BS3-clause 

# %%
# :func:`~watex.view.biPlot` has an an implementation in R but there is no 
# standard implementation in Python. Here is an example:
import matplotlib.pyplot as plt 
from watex.analysis import nPCA
from watex.datasets import fetch_data
from watex.view import biPlot, pobj  # pobj is Baseplot instance 
X, y = fetch_data ('bagoue pca' )  # fetch pca data 
pca= nPCA (X, n_components= 2 , return_X= False ) # return PCA object 
components = pca.components_ [:2, :] # for two components 
# customize plot 
pobj.xlabel ="Axis 1: PC1"
pobj.ylabel="Axis 2: PC2"
pobj.font_size =20. 
biPlot (pobj, pca.X, components , y ) # pca.X is the reduced dim X 
# to change for instance the line width (lw) or line style (ls),
# just use the baseplot-object *pobj* like:: 
# >>> pobj.ls ='-.'; pobj.lw=3 
plt.show()