"""
==================================================
Plot two dimensional dimensional filtered tensor
==================================================

plots the filtered tensors by applying the filtered function. 
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

#%%

# Here we plot the ama filtered 'ama' 
# passed to the filter function ;ffilter'='ama'
from watex.view.plot import TPlot 
from watex.datasets import load_edis 
# get some 3 samples of EDI for demo 
edi_data = load_edis (return_data =True, samples =12 )
# customize plot by adding plot_kws 
plot_kws = dict( ylabel = '$Log_{10}Frequency [Hz]$', 
						xlabel = '$Distance(m)$', 
						cb_label = '$Log_{10}Rhoa[\Omega.m$]', 
						fig_size =(6, 3), 
						font_size =7.,
						) 
t= TPlot(**plot_kws ).fit(edi_data)
# plot filtered tensor using the log10 resistivity 
t.plot_ctensor2d (to_log10=True, ffilter='ama')