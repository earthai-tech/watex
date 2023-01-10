"""
======================================================
Plot pseudo-fracturing index (sfi)
======================================================

Plot the pseudo-fracturing index known as *sfi* that is used to speculate 
about the apparent resistivity dispersion ratio around the cumulated sum 
of the  resistivity values of the selected anomaly (conductive zone)
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause 

#%%
import numpy as np 
from watex.utils.exmath import sfi 
rang = np.random.RandomState (42) 
condzone = np.abs(rang.randn (7)) 
# no visualization and default value `s` with gloabl minimal rho
sfi_value = sfi (condzone)
print(sfi_value)
# visualize fitting curve 
plotkws  = dict (rlabel = 'Conductive zone (cz)', 
                     label = 'fitting model',
                    leg =['selected conductive zone']  # color=f'{P().frcolortags.get("fr3")}', 
                     )
sfi (condzone, view= True , s= 5, figsize =(7, 7), style ='classic', 
          **plotkws )