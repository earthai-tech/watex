# -*- coding: utf-8 -*-
"""
..synopsis::
    Precision/recall Tradeoff computes a score based on the decision 
    function. 
        ...
        
..see also::  For parameter definitions, please refer to
        :doc:`~watex.utils.ml_utils.Metrics.PrecisionRecallTradeoff`
        for further details.
        ...
        
Created on Tue Sep 21 10:09:22 2021

@author: @Daniel03
"""

from sklearn.svm import SVC #, LinearSVC 

from watex.viewer.mlplot import MLPlots 
# modules below are imported for testing scripts.
# Not usefull to import since you provided your own dataset.
from watex.datasets import fetch_data 
X_prepared, y_prepared = fetch_data('Bagoue dataset prepared')

# random state 
random_state =42
# # test dirsty classifer "stochastic gradient descent" 
svc_clf = SVC(C=100, gamma=1e-2, kernel='rbf', random_state =random_state) 
# kind of plots
# precision vs recall  `'vsRecall'`
# or  all (precision-recall VS thershold) --> kind ='vsthresholds'   
kind ='vsRecall'

# K-Fold cross validation
cv =7 

# `classe_` argument is provied if y are not binarized. i.e 
# created a binary attribute for each flow classes; one attribute 
#equal to 1 when others categories equal to 0.
classe_category = 1 

#plot_key words arguments 
plot_kws = {'lw' :3.,           # line width 
            'pc' : 'k',         # precision color 
            'rc':'gray',           # recall color 
            'ps':'-',           # precision style 
            'rs':':',          # recall line style
            'font_size':7.,
            'show_grid' :True,        # visualize grid 
           'galpha' :0.2,              # grid alpha 
           'glw':.5,                   # grid line width 
           'gwhich' :'major',          # minor ticks
            # 'fs' :3.,                 # coeff to manage font_size 
            }
leg_kws ={'loc':'upper right'}
mlObj= MLPlots(leg_kws=leg_kws, **plot_kws
               )
# additional scikit_lean precision recall keywords arguments 
prt_kws =dict()
#call object
mlObj.PrecisionRecall(clf = svc_clf, 
                          X= X_prepared, 
                          y = y_prepared, 
                          classe_=classe_category, 
                          cv=cv,
                          kind=kind,
                         **prt_kws
                         )
