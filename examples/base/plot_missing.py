"""
=================================
Plot data with missing features 
=================================

plots the missing features and extracts insight with their correlation
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

#%% 
# :class:`Missing` inherits from :class:`Data`  and use :mod:`missingno`. Install 
# the package :code:`missingno` for taking advantage of many missing plot. 
# The parameter `kind` is passed to :class:`Missing` for selecting the 
# kind of plot for visualisation: 

# %% 
# Plot the missing in the data using the base visualization 
from watex.datasets import fetch_data 
from watex.base import Missing
data= fetch_data("bagoue original").get('data=df') # num flow 
ms= Missing().fit (data) 
ms.plot(figsize = (12, 4 )) 

# %%
# Plot the same missing data using the correlation 
# visualization 

ms.kind='corr'
ms.plot(figsize = (12, 4 )) 