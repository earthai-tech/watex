"""
=================================================
Plot Cut/Quantile features
=================================================

compares the quantile values of ordinal categories 
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

#%%
from watex.utils import naive_imputer
from watex.datasets import fetch_data 
from watex.view import ExPlot 

data = fetch_data ('bagoue original').get('data=dfy1') 
p= ExPlot(tname='flow', fig_size =(7, 5) ).fit(data)
data = naive_imputer(data, mode='bi-impute')
p.plotcutcomparison(xname ='power', yname='sfi') # compare 'power' and 'sfi'

