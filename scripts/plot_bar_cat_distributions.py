# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 19:52:53 2021

@author: @Daniel03
..synopsis:: Bar plot distribution. Can plot a distribution according to 
        the occurence of the `target` in the data and other parameters 
"""


from watex.viewer.plot import QuickPlot
from watex.datasets import fetch_data 

# fetch the bagoue test dataframe  dataframe df with text attributes label
data = fetch_data('Bagoue original')['data=dfy2']
# path to dataset 
data_fn = None# 'data/geo_fdata/BagoueDataset2.xlsx'


#-------------------------------------------------------------------------
# uncomment and edit the sl_analysis arguments for your purpose 
flow_classes =  [0., 1., 3.] # mean 0m3/h , 0-1 m3/h , 1-3 m3/h and >3 m3/h

#---------------------------------------------------------------------------
# target name 
target_name ='flow'

#customize 
groupFeaturesby ='shape'      # can be a list ['shape', 'type'] or dict as below
# groupFeaturesby = {'type':{'color':'b',
#                         'width':0.25 , 'sep': 0.},
#                 'shape':{'color':'g', 'width':0.25, 
#                         'sep':0.25}}

# line color 
lc ='b'

# set xlabel 
xlabel = 'Anomaly type '
# set ylabel 
ylabel ='Number of  occurence (%)'

# set the style  
sns_style= 'darkgrid'

# --> call QuickPlot Object 
qplotObj = QuickPlot(
            data_fn =data_fn,
            df = data, 
            flow_classes = flow_classes, 
            target_name = target_name,
            xlabel =xlabel, 
            ylabel=ylabel, 
            lc=lc, 
            sns_style =sns_style
            )

qplotObj.bar_cat_distribution(basic_plot =False, 
                            groupby=groupFeaturesby ,
                            )