"""
====================================
Vertical Electrical Sounding (VES)
====================================

visualizes the fracture zone from synthetic VES data.
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

#%%
# VES is carried out to speculate about the existence of a fracture zone and the 
# layer thicknesses. Commonly, it comes as supplement methods to ERP
# after selecting the best conductive zone. Here we give a synthetic plot and 
# compute the probable fracture zone from parameer search 
# we will generate a synehic data with 50 samples i.e 50 measurements in deeper 
# using the function  :func:`~watex.datasets.gdata.make_ves`. 

from watex.datasets import make_ves 
ves_data = make_ves ( seed=123, iorder =4 ).frame 
from watex.methods import VerticalSounding 
veso = VerticalSounding (search = 45 ).fit(ves_data ) 
veso.plotOhmicArea (fbtw=True , style ='classic') 