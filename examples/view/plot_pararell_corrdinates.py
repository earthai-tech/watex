"""
===============================================
Plot parallel coordinates 
===============================================

visualizes the features parallel coordinates 
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

#%%
from watex.datasets import fetch_data 
from watex.view import ExPlot 
data =fetch_data('original data').get('data=dfy1')
p = ExPlot (tname ='flow', fig_size =(7, 5)).fit(data)
# need yellowbrick to be installed if pkg=yb orherwise used pd instead
p.plotparallelcoords(pkg='yb') 
