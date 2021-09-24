# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 17:08:09 2021

@author: @Daniel03

.. synopsis:: Method to quick plot the qualitatif and quantitatives parameters. 
"""

from watex.viewer.plot import QuickPlot

# path to dataset 
# data_fn = 'data/geo_fdata/BagoueDataset2.xlsx'
data_fn ='data/geo_fdata/main.bagciv.data2.csv'

#-------------------------------------------------------------------------
# uncomment and edit the sl_analysis arguments for your purpose 
flow_classes =  [0., 1., 3.] # mean 0m3/h , 0-1 m3/h , 1-3 m3/h and >3 m3/h

#---------------------------------------------------------------------------
# target name 
target_name ='flow'
# plot params 
plot_params ='cat'  # can be `cat|qual` for categorial or `num|quan` for numerical 
# set figure title 
fig_title= 'Features Matrix correlation'

#customize plots 
# line color 
lc ='b'
# set the theme 
set_theme = 'darkgrid'

# --> Call object  
qkObj = QuickPlot(
            data_fn =data_fn, 
            flow_classes = flow_classes, 
            target=target_name,
            lc=lc, 
            set_theme =set_theme, 
            fig_title=fig_title
             ) 

sns_kwargs ={'annot': False, 
            'linewidth': .5, 
            'center':0 , 
            # 'cmap':'jet_r', 
            'cbar':True}
qkObj.plot_correlation_matrix(
    plot_params=plot_params, **sns_kwargs,)