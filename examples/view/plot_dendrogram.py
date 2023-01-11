"""
=================================================
Plot dendrogram 
=================================================

visualizes specific features on a dendrogram diagram 
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

#%%
from watex.datasets import load_iris 
from watex.view import plotDendrogram
data = load_iris () 
X =data.data[:, :2] 
plotDendrogram (X, columns =['X1', 'X2' ] ) 