# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 09:04:27 2021

@author: @Daniel03
"""

from watex.view.plot import QuickPlot 

# path to dataset 
# data_fn = 'data/geo_fdata/BagoueDataset2.xlsx'
data_fn ='data/geo_fdata/main.bagciv.data.csv'

# set figure title 
fig_title= '`sfi` vs`ohmS|`geol`'

# list of features to discuss 


features2dicuss =['ohmS', 'sfi','geol', 'flow']
qkObj = QuickPlot(  fig_legend_kws={'loc':'upper right'},
          fig_title = fig_title,
            )  

# sns keywords arguments 

sns_pkws={'aspect':2 , 
          "height": 2} 

# marker keywords arguments
map_kws={'edgecolor':"w"}   
qkObj.discussingFeatures(
                            data_fn =data_fn , 
                         features =features2dicuss,
                           map_kws=map_kws,  **sns_pkws
                         )  

