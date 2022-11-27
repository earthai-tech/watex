# -*- coding: utf-8 -*-
"""
..sypnosis: Plot the variance ratio. 
    Please refer to :doc:`watex.processing.tips.pcaVarianceRatio ` for 
    futher details.
    ...
    
Created on Wed Sep 22 22:14:10 2021

@author: @Daniel03
"""

from watex.view.mlplot import MLPlots
from watex.analysis.dimensionality import pcaVarianceRatio 
# module below are imported for testing scripts.
# Not usefull to import since you have your own dataset.
from watex.datasets import fetch_data 

X,_ = fetch_data('bagoue analysis')

# plot variance_ratio
plot_variance_ratio=True
# experiences attributes combinaisons 
combined_attributes =True
# pca Components
pca_n_components  =0.95 # can be n=2 etc. 

# addtributes indexes addeds 
# new features is created from indexes of existing numerical attribues 

new_features =[
            (0, 1),
            # (0,3), 
            # (1,4), 
            # (0, 4),
            # (3,4),

            ] #[(1, 0) , (3,2), (1,3),(0, 3), (0, 2), (2, 3)
               #] # divided values at index 1 to index0 and so on 

# provides attributes index to arrange 
# if X is numpay array , Can combined attributes via indexes in list 
#  numerical indexes 
numIndexes =None
# categorical indexes  
catIndexes =None
# automate the selection of   mumerical features when dataframe is given 
autoNumSelector= 'auto'

# type of Scikit-learn feature scaling 
scaler='StandardScaler'             # can be `MinMaxScaler`
# encode categorical feature using original encoders 
encode_categorical_features =True 
#add plot keywords arguments 

plot_kws = {'fig_size':(8, 12),
    'lc':(.9,0.,.8),
        'lw' :3.,           # line width 
        'font_size':7.,
        'show_grid' :True,        # visualize grid 
       'galpha' :0.2,              # grid alpha 
       'glw':.5,                   # grid line width 
       'gwhich' :'major',          # minor ticks
        # 'fs' :3.,                 # coeff to manage font_size 
        'leg_kws': {'loc':'upper left', 
                        'fontsize':15.}

        }
mlObj =MLPlots(**plot_kws)

pcaVarianceRatio(mlObj,
                 X=X,
                 n_components=pca_n_components , 
                 plot_var_ratio=plot_variance_ratio,
                 add_attributes=combined_attributes, 
                 attributes_ix =new_features, 
                 num_indexes =numIndexes, 
                 cat_indexes =catIndexes, 
                 selector__=autoNumSelector, 
                 scaler__= scaler)
