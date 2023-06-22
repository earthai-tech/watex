"""
========================
Plot Strata 
========================

plot stratigraphic log 
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause 

#%%

import watex.utils.geotools as GU   
layers= ['$(i)$', 'granite', '$(i)$', 'granite']
thicknesses= [59.0, 150.0, 590.0, 200.0]
hatch =['//.', '.--', '+++.', 'oo+.']
color =[(0.5019607843137255, 0.0, 1.0), 'b', (0.8, 0.6, 1.), 'lime']
GU.plot_stratalog (thicknesses, layers, hatch =hatch ,
                   color =color, station='S00')
GU.plot_stratalog ( thicknesses,layers,hatch =hatch, 
                   zoom =0.25, color =color, station='S00')