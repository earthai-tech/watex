"""
=========================
Plot logging predictor  
=========================

plots the predictor :math:`X` only composed  of 
hydro-geophysical features
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

#%%
import matplotlib.pyplot as plt 
from watex.datasets import load_hlogs 
from watex.methods.hydro import Logging 
# get the logging data 
h = load_hlogs ()
   
log= Logging(kname ='k', zname='depth_top' ).fit(h.frame[h.feature_names])
# Uncomment this see all the predictor predictor X 
# except the categorial feature like strata 
# log.plot ()

# for a clariyy , we will plot only the logging features 
features = ['depth_top',
     'resistivity',
     'gamma_gamma',
     'natural_gamma',
     'sp',
     'short_distance_gamma',
     ] 

log= Logging(kname ='k', zname='depth_top' ).fit(h.frame[features])
log.plot ()
plt.tight_layout()