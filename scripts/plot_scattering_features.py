# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:20:24 2021

@author: @Daniel03

Quickplot module uses the default configuration of features reading. To edit 
the default values like `flow_classes` or `target`, call the 
:class:`watex.analysis.features.sl_analysis` from :mod:`~features` before 
calling the `quickPlot` from :mod:`watex.viewer.plot`.

.. synopsis:: The goal is to draw a scatter plot with possibility of several 
        semantic features groupings. Indeed `scatteringFeatures`
        analysis is a process of understanding  how features in a 
        dataset relate to each other and how those relationships depend 
        on other features. Visualization can be a core component of this 
        process because, when data are visualized properly,
        the human visual system can see trends and patterns
        that indicate a relationship. 
"""

from watex.view.plot import QuickPlot
# from watex.analysis.features import sl_analysis
 
# path to dataset 
data_fn = 'data/geodata/main.bagciv.data.csv'
#-------------------------------------------------------------------------
# uncomment and edit the sl_analysis arguments for your purpose 
# flow_classes =  [0., 1., 3.] # mean 0 , 0-1 m3/h , 1-3 m3/h and >3 m3/h
# slObj =sl_analysis(data_fn=data_fn , set_index =True,
#                    flow_classes = flow_classes )
#---------------------------------------------------------------------------

# target name 
target_name ='flow'

# set figure title 
fig_title= 'geol vs lewel of water inflow'

#customize plots 
#labels 
xlabel = 'Level of water inflow (lwi)'
ylabel= 'Flow rate in m3/h'
marker_list= ['o','s','P', 'H']
# line color 
lc ='b'
# set the theme 
set_theme = 'darkgrid'

# --> Call object 

qkObj = QuickPlot(
        data_fn =data_fn ,
        lc=lc, 
        target_name = target_name, 
        set_theme =set_theme, 
        fig_title=fig_title,
        xlabel=xlabel, 
        ylabel=ylabel
            )  

markers_dict = {key:mv 
               for key, mv in zip( list (
                       dict(qkObj.df ['geol'].value_counts(
                           normalize=True)).keys()), 
                            marker_list)}

sns_pkws={'markers':markers_dict, 
          'sizes':(20, 200),
          "hue":'geol', 
          'style':'geol',
         "palette":'deep',
          'legend':'full',
          # "hue_norm":(0,7)
            }

regpl_kws = {'col':'flow', 
             'hue':'lwi', 
             'style':'geol',
             'kind':'scatter'
            }

# --> call scatteringObject 
qkObj.scatteringFeatures(features=['lwi', 'flow'],
                         relplot_kws=regpl_kws,
                         **sns_pkws, 
                    )