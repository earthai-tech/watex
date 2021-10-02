# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 20:37:10 2021

@author: @Daniel03
"""

from sklearn.svm import SVC #, LinearSVC 

from watex.viewer.mlplot import MLPlots 
# modules below are imported for testing scripts.
# Not usefull to import since you provided your own dataset.
from watex.datasets import fetch_data 

X,y = fetch_data('Bagoue dataset prepared')


# classifier 
svc_clf = SVC(C=100, gamma=1e-2, kernel='rbf', random_state =42) 
# predict y_true since we send test set 
predict_ypred =True 

#fill_line between the actual classes 
visible_line =False
# add index if not given 
index =None 
# add prefix to index 
prefix ='b'
# the categoriss labels
ylabel =['FR0', 'FR1', 'FR2', 'FR3'] # or None 
               
plot_kws ={'lw' :3.,                  # line width 
             'lc':(.9, 0, .8), 
             'ms':7.,                 # line color 
             'yp_marker_style' :'o', 
             'fig_size':(12, 8),
            'font_size':15.,
            'xlabel': 'Test examples',
            'ylabel':'Flow categories' ,
            'marker_style':'o', 
            'markeredgecolor':'k', 
            'markerfacecolor':'b', 
            'markeredgewidth':3, 
            'yp_markerfacecolor' :'k', 
            'yp_markeredgecolor':'r', 
            'alpha' :1., 
            'yp_markeredgewidth':2.,
            'show_grid' :True,          # visualize grid 
            'galpha' :0.2,              # grid alpha 
            'glw':.5,                   # grid line width 
            'rotate_xlabel' :90.,
            'fs' :3.,                   # coeff to manage font_size 
            's' :20 ,                  # manage the size of scatter point.
            'rotate_xlabel':90
               }

modObj = MLPlots(**plot_kws)
modObj.model(y, X_=X, clf =svc_clf, 
              predict= predict_ypred, 
              prefix =prefix ,
              fill_between =visible_line, 
              ylabel=ylabel)
