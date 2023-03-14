"""
================================================
Auto-detect the drilling location
================================================

Auto detect the suitable drilling point for drilling operations using the 
naive auto-detection and the auto-detection with the constraints application.
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause 

#%%
# For this example, we randomly generate 50 stations of synthetic ERP 
# resistivity data with minimum and maximum  resistivity values equal 
# to 1e1 and 1e4 ohm.m respectively as:

import watex as wx 
data = wx.make_erp (n_stations=50, max_rho=1e4, min_rho=10., as_frame =True, seed =42 ) 
#%%
# * Naive auto-detection (NAD)
#
# The NAD automatically proposes a suitable location with NO restrictions 
# (constraints) observed in the survey site during the :term:`GWE`. We may understand 
# by suitable, a location expecting to give a flow rate greater than 1m3/hr at 
# least.

robj=wx.ResistivityProfiling (auto=True ).fit(data ) 
robj.sves_ 

#%%
# The suitable drilling location is proposed at 
# station S25 (stored in the attribute sves_).
#
# * Auto-detection with constraints (ADC)
#
# The constraints refer to the restrictions observed in the survey area during
# the :term:`DWSC`. This is common in real-world exploration. For instance, a station 
# close to a heritage site should be discarded since no drilling operations 
# are authorized at that place. When many restrictions are enumerated in the 
# survey site, they must be listed in a dictionary with a reason and passe to 
# the parameter constraints so these stations should be ignored during the 
# automatic detection. Here is an example of constraints application to our 
# example.

restrictions = {
    'S10': 'Household waste site, avoid contamination',
    'S27': 'Municipality site, no authorization to make a drill',
    'S29': 'Heritage site, drilling prohibited',
    'S42': 'Anthropic polluted place, avoid contamination within a few years',
    'S46': 'Marsh zone, borehole will dry up during the dry season'
 }
robjc=wx.ResistivityProfiling (constraints= restrictions, auto=True ).fit(data ) 
robjc.sves_

#%% 
# * Plot ERP including the restricted stations.
robjc.plotAnomaly(style ='classic') 
#%%
# * plot the selected conductive zone wih No cpnstraints
robj.plotAnomaly(style ='classic')
#%%
# When the constraints is applied, the selected conductive zone is move to the 
# station  S33 by ignoring the station S25 close to the restricted stations S27 
# and S29. Even the user might discarded this station also. Now taking the 
# risk for making a drill or rejected the propose station will depend  to the 
# technician on the site. Commonly when the station is close to the restricted 
# area, it is better to avoid the possibility to make a drill at that station. 
#
# .. note::
# Notice, the station S25 is no longer considered as the suitable location and 
# henceforth, propose S33 as the priority for drilling operations. However, if 
# the station is close to a restricted area, a warning should raise to inform 
# the user to avoid taking a risk to perform a drilling location at that location.
# Note that before the drilling operations commence, make sure to carry out 
# the DC-sounding (VES) at that point. 