"""
=================================
Plot filtered tensors in 2D 
=================================

visualizes the filtered  tensors in a two-dimensional layout
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

# %% 
# Here will show an example using the Fixed -dipole-length filter (FLMA )
from watex.view.plot import TPlot 
from watex.datasets import load_edis 
e= load_edis (samples =21 , key='*') 
edi_data = e.frame.edi.values 
# get some 3 samples of EDI for demo 
# customize plot by adding plot_kws 
plot_kws = dict( ylabel = '$Log_{10}Frequency [Hz]$', 
						xlabel = '$Distance(m)$', 
						cb_label = '$Log_{10}Rhoa[\Omega.m$]', 
						fig_size =(6, 3), 
						font_size =7., 
						) 
t= TPlot(component='yx', **plot_kws ).fit(edi_data)
# plot recovery2d using the log10 resistivity  
t.plot_ctensor2d (to_log10=True, ffilter='flma')