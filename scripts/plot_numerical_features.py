# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:53:00 2021

Quickplot module uses the default configuration of features reading. To edit 
the default values like `flow_classes` or `target`, call the 
:class:`watex.analysis.features.sl_analysis` from :mod:`~features` before 
calling the `quickPlot` from :mod:`watex.viewer.plot`.

@author: @Daniel03

.. sypnosis:: 
    Plot qualitative features distribution using correlative aspect. Be 
        sure to provided numerical features arguments. 
"""

from watex.view.plot import QuickPlot 

# path to dataset 
# data_fn = 'data/geo_fdata/BagoueDataset2.xlsx'
data_fn ='data/geodata/main.bagciv.data.csv'
#-------------------------------------------------------------------------
# uncomment and edit the sl_analysis arguments for your purpose 
flow_classes =  [0., 1., 3.] # mean 0m3/h , 0-1 m3/h , 1-3 m3/h and >3 m3/h

#---------------------------------------------------------------------------
# target name 
target_name ='flow'

# set figure title 
fig_title= 'Quantitative features correlation'

#customize plots 
# line color 
lc ='b'
# set the theme 
set_theme = 'darkgrid'

# --> Call object  
qkObj = QuickPlot(
            data_fn =data_fn, 
            flow_classes = flow_classes, 
            target_name=target_name,
            lc=lc, 
            set_theme =set_theme, 
            fig_title=fig_title
             ) 
#, 's']  markers must be a singleton or a list of 
#markers for each level of the hue variable
# for instance for hue =[FR0, FR1,  FR2, FR3]  , number of hue = 4 = marker level
sns_pkws={
        'aspect':2 , 
         "height": 2, 
         # 'markers':['o', 'x', 'D', 'H'],  
           # 'kind':'kde', 
          'corner':False,
        }
marklow = {'level':4,  # number of hue variables
          'color':".2", 
          'diag_kind':'kde'}

qkObj.plot_numerical_features(trigger_map_lower_kws=False, 
                                    map_lower_kws=marklow, 
                                   **sns_pkws)
