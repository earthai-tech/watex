# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 20:37:10 2021

@author: @Daniel03
"""

from sklearn.svm import SVC #, LinearSVC 

from watex.view.mlplot import MLPlots 
# modules below are imported for testing scripts.
# Not usefull to import since you provided your own dataset.
from watex.datasets import fetch_data 

X,y = fetch_data('Bagoue dataset prepared')
XT, _= fetch_data('Bagoue test set ')
index = XT.index 

#-----------------------------------------------------------------------------------
from watex.datasets._m import XT_prepared, yT_prepared
from watex.bases import fetch_model
# my_model, *_ = fetch_model('SVC__LinearSVC__LogisticRegression.pkl',
#                            modname ='SVC'
#                            ) 
X, y = XT_prepared, yT_prepared
#-----------------------------------------------------------------------------------
# classifier 
svc_clf = SVC(C=100, gamma=1e-2, kernel='rbf', random_state =42) 
# predict y_true since we send test set 
predict_ypred =True 

#fill_line between the actual classes 
visible_line =False
# add index if not given 
# index =None 
# add prefix to index 
prefix ='b'
# the categoriss labels
ylabel =['FR0', 'FR1', 'FR2', 'FR3'] # or None 
               
plot_kws ={'lw' :3.,                  # line width 
              'lc':(.9, 0, .8), 
              'ms':7.,                 # line color 
              'yp_marker_style' :'o', 
              'fig_size':(16, 8),
            'font_size':18.,
            'xlabel': 'Test set instances',
            'ylabel':'Flow rate categories' ,
            'marker_style':'o', 
            'markeredgecolor':'b', 
            'markerfacecolor':'b', 
            'markeredgewidth':3, 
            'yp_markerfacecolor' :'r', 
            'yp_markeredgecolor':'r', 
            'alpha' :1., 
            'yp_markeredgewidth':2.,
            'show_grid' :True,          # visualize grid 
            'galpha' :0.2,              # grid alpha 
            'glw':.5,                   # grid line width 
            'gls':':',
            'rotate_xlabel' :90.,
            'fs' :3.,                   # coeff to manage font_size 
            's' :100 ,                  # manage the size of scatter point.
            'rotate_xlabel':45, 
            'tp_axis':'x', 
            'tp_labelsize':7.,
            'leg_kws': {'loc':'lower left', 
                        'fontsize':15.}
                }

modObj = MLPlots(**plot_kws)
modObj.model(y, X_=X, clf =svc_clf, 
              predict= predict_ypred, 
              prefix =prefix ,
              fill_between =visible_line, 
              ylabel=ylabel,
              index = index)
# import pandas as pd
# # print(pd.core.indexes.numeric.Index64Index)
# is_index = isinstance(index, pd.Index)#index.is_object()

# print(is_index)