"""
=================================================
Plot model scores
=================================================

visualizes model fined tuned scores vs the cross validation 
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

# (1) -> Score is appended to the model 
from watex.exlib.sklearn import SVC 
from watex.view.mlplot import plot_model_scores
import numpy as np 
svc_model = SVC() 
fake_scores = np.random.permutation (np.arange (0, 1,  .05))
plot_model_scores([(svc_model, fake_scores )])
# uncomment this if 
# (2) -> Use model and score separately 
# plot_model_scores([svc_model],scores =[fake_scores] )
  
# now customize the plot as 
# >>> # customize plot by passing keywords properties 
base_plot_params ={
                    'lw' :3.,   
                    'ls': '-.', 
                    'lc':'m', #(.9, 0, .8), 
                    'ms':7.,                
                    'fig_size':(9, 6),
                    'font_size':15.,
                    'xlabel': 'samples',
                    'ylabel':'scores' ,
                    'marker':'o', 
                    'alpha' :1., 
                    'yp_markeredgewidth':2.,
                    'show_grid' :True,          
                    'galpha' :0.2,              
                    'glw':.5,                   
                    'rotate_xlabel' :90.,
                    'fs' :3.,                   
                    's' :20 ,
                    'sns_style': 'ticks', 
                }
plot_model_scores([svc_model],scores =[fake_scores] , **base_plot_params )