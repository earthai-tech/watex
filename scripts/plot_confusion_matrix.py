# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:38:25 2021

@author: @Daniel03
"""

from sklearn.svm import SVC #, LinearSVC 

from watex.viewer.mlplot import MLPlots 
# modules below are imported for testing scripts.
# Not usefull to import since you provided your own dataset.
from watex.datasets import fetch_data 

X,y = fetch_data('Bagoue dataset prepared')

# randaom state 
random_state =42 

# classifier 
svc_clf = SVC(C=100, gamma=1e-2, kernel='rbf', random_state =random_state) 

## K-Fold cross validation
cv =7 
# plottype 

plottype='error'      # can be 'map' or 'error'
# provide y label to force the mapshow to plot
# the categoriss labels
ylabel =['FR0', 'FR1', 'FR2', 'FR3'] # or None 
matshow_kwargs ={
        'aspect': 'auto', # 'auto'equal
        # 'alpha':0.5, 
        'interpolation': None, #'nearest', # 'antialiased', 'nearest', 'bilinear',
                                    # 'bicubic', 'spline16', 'spline36', 'hanning
       'cmap':'gray', 
       # 'cbar':True
            }
plot_kws ={'lw':3, 
           'lc':(.9, 0, .8), 
           'font_size':15., 
            # 'fs' :3.,                   # coeff to manage font_size 
            # 'cb_label': 'Error',
            'cb_format':None,
            # 'cb_size':15,
            'xlabel': 'Predicted classes',
            'ylabel': 'Actual classes',
            'font_weight':None,
            'tp_labelbottom':False,
            'tp_labeltop':True,
            'tp_bottom': False
            }
mObj =MLPlots(**plot_kws)
mObj.confusion_matrix(svc_clf,
                        X=X,
                        y=y,
                        cv=cv,
                        ylabel=ylabel, 
                        plottype=plottype,
                        matshow_kws = matshow_kwargs,
                        )

# print(mObj.conf_mx)
