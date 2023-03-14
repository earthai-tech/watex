"""
======================================================
Plot ex-heatmap 
======================================================

plots correlation matrix  as a heat map. 
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause 

#%%
# `mlxtend` package needs to be install to take advantages of 
# this plot. Use pip for installation as `pip install mlxtend` 

from watex.datasets import load_hlogs 
from watex.utils.plotutils import plot_mlxtend_heatmap
h=load_hlogs()
features = ['gamma_gamma', 'sp',
            'natural_gamma', 'resistivity']
plot_mlxtend_heatmap (h.frame , columns =features, cmap ='seismic')
