"""
======================================================
Plot ex-matrix
======================================================

Plot correlation matrix array  as a heat map. 
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause 

#%%
# `mlxtend` package needs to be install to take advantages of 
# this plot. Use pip for installation as `pip install mlxtend` 

from watex.datasets import load_hlogs 
from watex.utils.plotutils import plot_mlxtend_matrix
import pandas as pd 
import numpy as np 
h=load_hlogs()
features = ['gamma_gamma', 'natural_gamma', 'resistivity']
data = pd.DataFrame ( np.log10 (h.frame[features]), columns =features )
plot_mlxtend_matrix (data, columns =features)
