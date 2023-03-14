"""
========================================
Plot logging Predictor and target 
========================================

plots the logging data by including the target y  
containing the permeability coefficient k 
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

#%%
from watex.datasets import load_hlogs 
from watex.methods.hydro import Logging 
# get the logging data 
h = load_hlogs ()
   
log= Logging(kname ='k', zname='depth_top' ).fit(h.frame[h.feature_names])
# Uncomment the line below to plot all the feature that composed the predictor X 
# except the categorial feature like strata 
# log.plot ()

# for a clariyy , we will plot only the logging features 
features = ['depth_top',
     'resistivity',
     'gamma_gamma',
     'natural_gamma',
     'sp',
     ] 

log= Logging(kname ='k', zname='depth_top' ).fit(h.frame[features])
# put target at the first position 
log.plot (y = h.frame.k, draw_spines =(0, 7), posiy= 0, colors =['r', 'k', 'b', 'm', 'c']) 
