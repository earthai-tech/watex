"""
================================================
Plot two dimensional phase  tensors
================================================

gives a quick visualization of phase tensors at the 
component 'yx'
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

#%%
from watex.view.plot import TPlot 
from watex.datasets import load_edis 
# get some 12 samples of EDI for demo 
edi_data = load_edis (return_data =True, samples =12)
# customize plot by adding plot_kws 
plot_kws = dict( ylabel = '$Log_{10}Frequency [Hz]$', 
						xlabel = '$Distance(m)$', 
						cb_label= '$Phase [\degree]$' , 
						fig_size =(6, 3), 
						font_size =7.,
						) 
t= TPlot(component='yx', **plot_kws).fit(edi_data)
# plot recovery2d using the log10 resistivity 
t.plot_tensor2d( tensor ='phase', to_log10=True) 