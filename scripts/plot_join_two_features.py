# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:43:25 2021

@author: @Daniel03
.. sypnosis:: 
        Joint methods allow to visualize correlation of two features. 
        Draw a plot of two features with bivariate and univariate graphs.
        Please refer to :meth:`~.viewer.QuickPlot.join2features` for 
        futher details.

"""

from watex.view.plot import QuickPlot 
# from watex.analysis.features import sl_analysis
 
# path to dataset 
# data_fn = 'data/geo_fdata/BagoueDataset2.xlsx'
data_fn ='data/geo_fdata/main.bagciv.data.csv'

flow_classes =  [0., 1., 3.] # mean 0 , 0-1 m3/h , 1-3 m3/h and >3 m3/h


# target name 
target_name ='flow'

# set figure title 
fig_title= 'Join lwi and ohmS features correlation'

#customize plots 
# line color 
lc ='b'
# set the theme 
set_theme = 'darkgrid'

# --> Call object 

qkObj = QuickPlot(
        data_fn =data_fn, 
         target_name = target_name, 
             lc=lc, 
            set_theme =set_theme, 
            fig_title=fig_title
             )  
sns_pkws={
            'kind':'reg' , #'kde', 'hex'
            # "hue": 'flow', 
               }

joinpl_kws={"color": "r", 
             'zorder':0, 'levels':6}
plmarg_kws={'color':"r", 'height':-.15, 'clip_on':False} 
          
qkObj.joint2features(
            features=['ohmS', 'lwi'], 
            join_kws=joinpl_kws, 
            marginals_kws=plmarg_kws, 
            **sns_pkws, 
            ) 