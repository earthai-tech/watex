# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:08:07 2021

@author: @Daniel03
"""

import numpy as np 
from sklearn.svm import SVC

from watex.viewer.mlplot import MLPlots
# modules below are imported for testing scripts.
# Not usefull to import since you privied your own dataset.
from watex.bases import fetch_model 
from watex.datasets import fetch_data 
#--------------Evaluate your model on the test data ------------------------------
# from watex.datasets._m import XT_prepared, yT_prepared
my_model, *_ = fetch_model('SVC__LinearSVC__LogisticRegression.pkl', modname ='SVC') 
#---------------------------------------------------------------------------------
X_prepared,  y_prepared = fetch_data('Bagoue prepared datasets')

# plot scoring 
scoring ='rmse'

# random_state 
random_state =42 
# base estimator. If baseeastimator is set, should replace the default estimator 
baseEstimator = my_model #SVC(random_state=42)
#baseEstimator = SVC(C=100, coef0=1, degree=1, kernel='linear', random_state =random_state) 
plot_kws = {'fig_size':(8, 12),
    'lc':(.9,0.,.8),
        'lw' :3.,           # line width 
        'font_size':7.,
        'show_grid' :True,        # visualize grid 
       'galpha' :0.2,              # grid alpha 
       'glw':.5,                   # grid line width 
       'gwhich' :'major',          # minor ticks
        # 'fs' :3.,                 # coeff to manage font_size
        # 'xlabel':'Training set size', 
        # 'ylabel': 'RMSE'
        'vls':'-',         # validation line style
        # 'vlc':'cyan'        # validation line color
        'leg_kws':{'loc':'upper right', 
                   'fontsize':18.}
        }
mlObj =MLPlots(**plot_kws)
mlObj.plot_learning_curves(clf= baseEstimator,
                           X=X_prepared, 
                           y= y_prepared, 
                           random_state =random_state, 
                           scoring =scoring)

