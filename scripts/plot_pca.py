# -*- coding: utf-8 -*-
# 
"""
..synopsis::Plot PCA component analysis using :class:`~sklearn.decomposition`.
        PCA indentifies the axis that accounts for the largest amount of 
        variance in the trainset `X`. It also finds a second axis orthogonal 
        to the first one, that accounts for the largest amount of remaining 
        variance.
        ...
        
:Notes: 
     Set params `y_replace` to ``True`` and `y_values` as well as 
        `y_classes` is usefull when y_lables are not categorized If your 
        label`y` must be a text attributes.If it's not the case, dont need to 
        set none of these parameters and let their defaults values.
    
    - param `replace_y`: customize the encoded values by providing a new list 
        of categorized values
    - param y_values: Once `replace_y` is set to True, then `y_values` must 
        be given to convert the numerical values into a categorial 
        values contained in the list of `y_values`. NB: values in 
        `y_values` must be self containing in `y`(numerical data.) 

   -param `y_classes`: Can replace the numerical  values encoded thoughout 
        `y_values` to text labels which match each encoded values 
        in `y_values`. For instance::
            y_values =[0, 1, 3]
            y_classes = ['FR0', 'FR1', 'FR2', 'FR3']
        where :
            - ``FR0`` equal to values =0 
            - ``FR1`` equal values between  0-1(0< value<=1)
            - ``FR2`` equal values between  1-1 (1< value<=3)
            - ``FR3`` greather than 3 (>3)
            
        Please refer to :doc:`watex.utils.decorator.catmapflow` and 
        :doc:`watex.analysis.features.categorize_flow` for futher 
        details.
    
Created on Tue Sep 21 10:25:59 2021

@author: @Daniel03
"""


from watex.viewer.mlplot import MLPlots 
# modules below are imported for testing scripts.
# Not usefull to import since you provided your own dataset.
from watex.datasets import fetch_data 

# trainset, y -labels
X, y = fetch_data('Bagoue analyses data')

# param replace_y: Change label from regression to classification problem.
# customize the encoded values by providing a new list of categorized values
replace_y =True 
   
#param y_values: Once `replace_y` is set to True, then `y_values` must 
        # be given so that the numerical values should be converted into a Text 
        # values. Otherwise if `replace_y` is ``True`` and `y_values` is not given,
        # be sure to provide the number of text categories equal to the 
        # number of numerical categories self containing in `y`(numerical data) 
yvalues =None              
        
#param y_classes: Cn replace the numercal  values encodes thought 
yclasses = ['FR0', 'FR1', 'FR2', 'FR3']

# biplot 
# biplot pca features importance (pc1 and pc2) and visualize different 
#  variables according to Serafeim Loukas, serafeim.loukas@epfl.ch 
# 
biplot =False 

# pca additionals keywords 
pca_kws =dict()

# call objects
# number of components axes 
nAxes = 7 
pc1_axis = 'Axis 1'
pc2_axis ='Axis 8'
#plot_key words arguments 
plot_kws = {
            # 'lw' :3.,           # line width 
            # 'font_size':7.,
            'show_grid' :False,        # visualize grid 
           'galpha' :0.2,              # grid alpha 
           'glw':.5,                   # grid line width 
           'gwhich' :'major',          # minor ticks
            # 'fs' :3.,                 # coeff to manage font_size 
            }
pcaObj= MLPlots().PCA_(X= X, 
                       y=y,
                       replace_y=replace_y, 
                       y_values =yvalues, 
                       y_classes =yclasses,
                        biplot =biplot,
                        pc1_label=pc1_axis ,
                        pc2_label=pc2_axis,
                        n_axes=nAxes,
                        **pca_kws)

























