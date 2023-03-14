"""
==============================
Plot filtered tensors in 1D
==============================
 
visualizes filtered tensors in 1D using the adaptative moving average (AMA), the fixed-dipole 
length moving average (FLMA) and the trimming-moving average (TMA) filters 
after signal recovery. 
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

#%% 
# After recovering the NSAMT signal using :meth:`~watex.methods.Processing.zrestore`, 
# the latter could  exhibits a field strength amplitude for the next processing 
# step like filtering. :code:`watex` implements three filtering linked to 
# the :class:`~watex.methods.em.Processing`. 
# Here is an example of filtered data using the three filters

import numpy as np 
import matplotlib.pyplot as plt 
from watex.methods import Processing 
from watex.datasets import fetch_data  # load_edis 
e= fetch_data ('edis', samples =15, key='*' )  #"*" to fetch all columns of edi data 
# the above code is the same as 
# e= load_edis (samples =21 , key='*') 
edi_data = e.frame.edi.values 
# 'srho' for static resistivity correction'
pobj= Processing(window_size =5, component='yx', c= 2, out='srho').fit( edi_data ) 
resyx = pobj.make2d ('resyx') 
res_ama = pobj.ama() 
res_flma = pobj.flma () 
res_tma = pobj.tma () 
x_stations = np.arange (len(e.frame.site )) 
plt.xticks (x_stations , list (e.frame.id)) 
# corrected Edi at station S00
plt.semilogy (x_stations, resyx[0, :]  , 'ok-', label ='raw_data' ) 
plt.semilogy(x_stations, res_ama[0, :] ,'or-',  label ='ama') 
plt.semilogy(x_stations, res_flma [0, :], 'ob--',  label ='flma') 
plt.semilogy(x_stations, res_tma[0, :] , 'oc-.', label ='tma') 
plt.title ("Filtered tensor $Res_{xy}$ at station $S00$") 
plt.legend ()
plt.xlabel ("Sites") ; plt.ylabel ("Resistivity ${xy}[\Omega.m]$")
plt.style.use ('classic')