"""
================================================
Plot multiple categorical feature distributions 
================================================

plots a multiple categorical distributions onto a FacetGrid
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

#%%
from watex.view.plot import QuickPlot 
from watex.datasets import load_bagoue 
data = load_bagoue ().frame
qplotObj= QuickPlot(lc='b', tname='flow')
qplotObj.sns_style = 'darkgrid'
qplotObj.mapflow=True # to categorize the flow rate 
qplotObj.fit(data)
fdict={
           'x':['shape', 'type', 'type'], 
           'col':['type', 'geol', 'shape'], 
           'hue':['flow', 'flow', 'geol'],
           } 
qplotObj.multicatdist(**fdict)