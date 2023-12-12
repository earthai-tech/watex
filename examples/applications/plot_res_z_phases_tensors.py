"""
===============================
Plot Res, Z and phase tensors 
===============================

Plot impedances/resistivity and phase tensors 
from AMT data after applying essential corrections.   

"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

# %%
# :func:`~watex.utils.plot_tensors` plots the impendances/resistivity  
# and phase tensors. In the following example, we called 
# a sample of  :term:`EDI` collected in Huayuan locality, Hunan province, China,
# stored as inner datasets then we apply the three basic corrections such as: 
#
# - `removing noises`: Drop a possible existences of factories noises, power 
#   lines effects , ...
# - `static shift`: Shift the TE and TM tensors at each station. 
# - `drop frequencies`: Drop frequencies is usefull to control the safety of the 
#   EM data collected in an area with severe interferences. 

#%% 
# We start by importing the required modules as: 
import os 
from watex.datasets import fetch_data 
from watex.utils.plotutils import plot_strike 
from watex.datasets.io import get_data # get edidata stored in cache
from watex.methods import EM, MT 
from watex.utils.plotutils import plot_tensors 
import matplotlib.pyplot as plt 
plt.style.use ("seaborn")
#%% 
# Before we'll make a collection of :term:`EDI` data and call 
# :func:`watex.utils.plot_strike` for plotting as: 
#
fetch_data ( 'huayuan', samples = 20) # store 20 edis in cache 
#
# edipath = r'D:\project-Tayuan\data\1\1one'
edi_fn_lst = [os.path.join(get_data(),f) for f in os.listdir(get_data()) 
         if f.endswith('.edi')] 
# edi_fn_lst = [os.path.join(edipath,f) for f in os.listdir(edipath) 
#         if f.endswith('.edi')] 
# %% 
# * Plot the raw tensors 
# We can plot the raw tensors without applying any corrections at  
# the first station ``S00``. 

emo_r = EM().fit(edi_fn_lst)
plot_tensors(emo_r.ediObjs_, station =0)

# %% 
# Then we can visualize the strike using the raw data as 

plot_strike(edi_fn_lst ) 

#%%
# * Plot corrected tensors 
#
# We applied three essential corrections to EDI objets via a chaining 
# method supplied by the :class:`watex.methods.MT` class. 
# We first drop the bad frequencies with a severity set to `10%`
# tolerance. Then removing the static shift effect. After TM and ME mode data  
# are automatically (``nfreq=auto``) shifted by asserting all stations, 
# the noises such as the power lines or interferences can be removed. Note that 
# the all processing step can be applied differently, it does not need to 
# follow a certain order.  

mo = MT().fit(edi_fn_lst).drop_frequencies (
    tol =0.1).remove_static_shift (nfreq='auto' ).remove_noises (method ='base')

# we then visualized the new corrected tensors at the station 0 
plot_tensors ( mo.ediObjs_, station =0 )

#%%
#
# .. topic:: References 
#
#   .. [1] Weaver J.T, Lilley F.E.M.(2003)  Invariants of rotation of axes and indicators of
#          dimensionality in magnetotellurics, Australian National University,
#          University of Victoria; http://bib.gfz-potsdam.de/emtf/2007/pdf/lilley.pdf
#   .. [2] T. Grant Caldwell, Hugh M. Bibby, Colin Brown, The magnetotelluric phase tensor, 
#          Geophysical Journal International, Volume 158, Issue 2, August 2004, 
#          Pages 457–469, https://doi.org/10.1111/j.1365-246X.2004.02281.x
