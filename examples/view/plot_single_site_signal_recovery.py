"""
================================================
Plot single site signal recovery
================================================

visualizes the restored tensor at each site. 
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

# For instance the visualisation of 
# the first site `S03` is given as: 
import matplotlib.pyplot as plt 
plt.style.use ('classic')
from watex.view import TPlot
from watex.datasets import load_edis 
# load 7 samples 
edi_data = load_edis (return_data =True, samples =7) 
plot_kws = dict( ylabel = '$Log_{10}Frequency [Hz]$', 
				xlabel = '$Distance(m)$', 
				cb_label = '$Log_{10}Rhoa[\Omega.m$]', 
				fig_size =(7, 4), 
				font_size =7. 
				) 
t= TPlot(**plot_kws ).fit(edi_data)
# plot recovery of site 'S01'
t.plot_recovery ('S03')