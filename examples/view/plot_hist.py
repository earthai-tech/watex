"""
=================================================
Plot features vs target on histogram plots 
=================================================

plots a histogram of continuous values against the a binary target.
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

#%%
from watex.datasets import fetch_data  
from watex.view import ExPlot
data= fetch_data("bagoue original").get('data=df') # raw data not encoded flow
p = ExPlot(tname ='flow').fit(data)
p.fig_size = (7, 5)
p.sns_style='ticks'
p.plothistvstarget (xname= 'sfi', c = 0, kind = 'binarize',  kde=True, 
					  posilabel='dried borehole (m3/h)',
					  neglabel = 'accept. boreholes'
					  )