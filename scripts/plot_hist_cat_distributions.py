# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 20:12:26 2021

.. synopsis:: Quick plot a distributions of categorized classes 
        according to the percentage of occurence. 
@author: @Daniel03

"""

from watex.view.plot import QuickPlot 

# path to dataset 
data_fn = 'data/geo_fdata/BagoueDataset2.xlsx'

#-------------------------------------------------------------------------
# uncomment and edit the sl_analysis arguments for your purpose 
flow_classes =  [0., 1., 3.] # mean 0m3/h , 0-1 m3/h , 1-3 m3/h and >3 m3/h

#---------------------------------------------------------------------------
# target name 
target_name ='flow'

stacked = True 
# line color 
lc ='b'

# set the style  
sns_style= 'darkgrid'

# --> call QuickPlot Object 
qplotObj = QuickPlot(
            data_fn =data_fn,
            flow_classes = flow_classes, 
            target_name = target_name,
            lc=lc, 
            sns_style =sns_style, 
            stacked =stacked, 
            )

qplotObj.hist_cat_distribution()