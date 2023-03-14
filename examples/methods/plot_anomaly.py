"""
======================================================================
Electrical Resistivity Profiling (ERP)
======================================================================

plots the ERP and selects the best conductive zone for the drilling 
operations.
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

#%%
# For demonstration,  the station is not specified, the algorithm will find the best 
# conductive zone based on the resistivity values and will store the value in 
# attribute `sves_` (position to make a drill). The auto-detection can be used 
# when users need to propose a place to make a drill.  

# Generate a synthetic data with make_erp 
# for 70 stations 

from watex.datasets import make_erp 
erp_data = make_erp ( n_stations =70, as_frame =True , seed= 123 )
# call the approapriate methods and plot the anomaly ( select conductive zone)

from watex.methods import ResistivityProfiling 
ResistivityProfiling (auto=True).fit(erp_data ).plotAnomaly(style ='classic')  

# As you can see the station S32 is proposed 