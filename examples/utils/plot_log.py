"""
=================================================
Plot log 
=================================================

plots a collection of logging data. 
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause 

#%% 
# Plot the logging data using the default behavior
# use the borehole `h2601`. 
# Note that each logging data composes a 
# column of data collected on the field. Note that can also plot anykind of 
# data related that it contains numerical values. The function does not 
# accept categorical data.   If categorical data are given, they should be 
# discarded. 

from watex.datasets import load_hlogs 
from watex.utils.plotutils import plot_logging
X0, y = load_hlogs (as_frame =True, key='h2601') # get the frame rather than object 

print(X0.columns )
print(y.columns )
#for demonstration we plot the following features 
features = ['depth_top',
     'resistivity',
     'gamma_gamma',
     'sp', # spontaneous polarization 
     ]   
X= X0[features ]  
# plot the default logging with Normalize =True  wih only three features 

plot_logging (X, normalize =True) 
#%% 
# Plot log including the target placed at the first position with 
# parameter `posiy=0`. The predictor :math:`X0` is systematically 
# convert to `log10` is set to 'True'. 
# Note that kp is the categorize k 

plot_logging ( X,  y = y.kp , posiy = 0, 
                  columns_to_skip=['sp'], # does not convert sp to log10 
                  log10 =True, 
                  )
#%% 
# Plot can be customize , for instance, by setting the draw_spines 
# with a depth limit from 0 to 700 m. 

# draw spines and limit plot from (0, 700) m depth 
plot_logging (X[features[:-2]] , y= y.k, draw_spines =(0, 700), 
              colors =["#9EB3DD", "#0A4CEE"] )
