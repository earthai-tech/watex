# -*- coding: utf-8 -*-
"""
..sypnosis: Visualize dataset.
            Since there is geographical information(latitude/longitude or
            eating/northing), itis a good idea to create a scatterplot of 
            all instances to visaulize data.
        ...
Created on Tue Sep 21 09:27:53 2021

@author: @Daniel03
"""
from watex.viewer.mlplot import MLPlots 
# modules below are imported for testing scripts.
# Not usefull to import since you provided your own dataset.
from watex.datasets import fetch_data 

X,_ = fetch_data('Bagoue stratified sets')
stratified_test_set ,_= fetch_data('test sets')

# dataset X: X can be a dataframe or numpay ndarray. 
test_dataset =X 

# test set (Optional param). If given , will plot will plot test set data 
test_testdata =stratified_test_set 

# Can regular plot by providing X for x-axis data and y for y-axis data 
# x-axis 
x=None 
# y-axis data 
y=None 

# call Matplotlib plot properties 
plot_props ={'lw' :3.,                  # line width 
             'lc':'k',                  # line color 
             'marker_style' :'o', 
             'fig_size':(8, 12),
            'font_size':15.,
            'xlabel': 'east',
            'ylabel':'north' , 
            'markerfacecolor' :'k', 
            'markeredgecolor':'r', 
            'alpha' :1., 
            'markeredgewidth':2.,
            'show_grid' :True,          # visualize grid 
            'galpha' :0.2,              # grid alpha 
            'glw':.5,                   # grid line width 
            'rotate_xlabel' :90.,
            'fs' :3.,                   # coeff to manage font_size 
            's' :None,                  # manage the size of scatter point.
            'leg_kws': {'loc':'upper left', 
                        'fontsize':15.}
               }

# create MLPlots objects 
mlObj= MLPlots(**plot_props
               )
# additional keywords arguments : 
vis_kws = { 'trainlabel': 'Train set', 
        'testlabel': 'Test set'}

mlObj.visualizingGeographycalData(
    X=test_testdata,
    X_=test_testdata, 
    y=y, 
    x=x,
    **vis_kws)