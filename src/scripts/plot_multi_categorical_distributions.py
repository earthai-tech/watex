# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 17:45:34 2021

@author: @Daniel03
"""

from watex.view.plot import QuickPlot 

# path to dataset 
# data_fn = 'data/geo_fdata/BagoueDataset2.xlsx'
data_fn ='data/geo_fdata/main.bagciv.data.csv'
#-------------------------------------------------------------------------
# uncomment and edit the sl_analysis arguments for your purpose 
flow_classes =  [0., 1., 3.] # mean 0m3/h , 0-1 m3/h , 1-3 m3/h and >3 m3/h

#---------------------------------------------------------------------------
# target name 
target_name ='flow'
# set feature according to x_axis 
x_features = ['shape', 'type', 'type']
# set features accordingto y-axis 
y_features = ['type', 'geol', 'shape']
# set targets 

targets = ['flow', 'flow', 'geol']
#customize plots 
# line color 
lc ='b'
# set the theme 
sns_style= 'darkgrid'

#--> call Object 
qplotObj = QuickPlot(
            data_fn =data_fn,
            flow_classes = flow_classes, 
            target_name = target_name,
            lc=lc, 
            sns_style =sns_style
            )

catfeatures_dict={
     'x_features':x_features, 
     'y_features':y_features, 
    'targets':targets,
           } 

qplotObj.multi_cat_distribution( **catfeatures_dict )  