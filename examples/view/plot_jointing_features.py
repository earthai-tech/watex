"""
===================================
Plot jointing features
===================================

draws a scatter plot with possibility of several semantic features grouping.
"""
# Author: L. Kouadio
# Licence: BSD-3- Clause

#%%
from watex.view.plot import QuickPlot 
from watex.datasets import load_bagoue 
data = load_bagoue ().frame
qkObj = QuickPlot( lc='b', sns_style ='darkgrid', 
             fig_title='Quantitative features correlation'
             ).fit(data)  
qkObj.fig_size =(7, 5)
sns_pkws={
            'kind':'reg' , #'kde', 'hex'
          # "hue": 'flow', 
              }
joinpl_kws={"color": "r", 
			'zorder':0, 'levels':6}
plmarg_kws={'color':"r", 'height':-.15, 'clip_on':False}           
qkObj.joint2features(features=['ohmS', 'lwi'], 
           join_kws=joinpl_kws, marginals_kws=plmarg_kws, 
            **sns_pkws, 
          )