"""
===================================
Plot numerical features
===================================

plots qualitative (numerical) features  distribution using correlative aspect.
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

#%%

from watex.view.plot import QuickPlot 
from watex.datasets import load_bagoue 
from watex.utils import smart_label_classifier 
data = load_bagoue ().frame
demo_features =['ohmS', 'power', 'lwi', 'flow'] 
data_area=data [demo_features] 
# use smart label classifier to encode ohmS features
# categorized the ohmS series into a class labels 'oarea1', 'oarea2' and 'oarea3'
data_area ['ohmS'] = smart_label_classifier (data_area.ohmS, values =[1000, 2000 ])
qkObj = QuickPlot(mapflow =False, tname="ohmS"
							  ).fit(data_area)
qkObj.sns_style ='ticks', 
qkObj.fig_title='Quantitative features correlation'
qkObj.fig_size =(7, 5)
sns_pkws={'aspect':2 , 
	          "height": 2, 
	# ...          'markers':['o', 'x', 'D', 'H', 's',
	#                         '^', '+', 'S'],
	          'diag_kind':'kde', 
	          'corner':False,
	          }
marklow = {'level':4, 
         'color':".2"}
qkObj.numfeatures(coerce=True, map_lower_kws=marklow, **sns_pkws)