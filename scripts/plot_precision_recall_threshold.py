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

from sklearn.linear_model import SGDClassifier

from watex.viewer.mlplot import MLPlots 
# modules below are imported for testing scripts.
# Not usefull to import since you privied your own dataset.
from watex.datasets.data_preparing import X_train_2
from watex.datasets import  y_prepared

# # test dirsty classifer "stochastic gradient descent" 
sgd_clf = SGDClassifier(random_state= 42)
# kind of plots
# precision vs recall  `'vsRecall'`
# or  all (precision-recall VS thershold) --> kind ='vsthresholds'   
kind ='vsRecall'

# trainset 
trainset= X_train_2
# y -labels 
y_array = y_prepared

# K-Fold cross validation
cv =3 

# `classe_` argument is provied if y are not binarized. i.e 
# created a binary attribute for each flow classes; one attribute 
#equal to 1 when others categories equal to 0.
classe_category = 1 

#plot_key words arguments 
plot_kws = {'lw' :3.,           # line width 
            'pc' : 'k',         # precision color 
            'rc':'b',           # recall color 
            'ps':'-',           # precision style 
            'rs':'--',          # recall line style
            'font_size':7.,
            'show_grid' :False,        # visualize grid 
           'galpha' :0.2,              # grid alpha 
           'glw':.5,                   # grid line width 
           'gwhich' :'major',          # minor ticks
            # 'fs' :3.,                 # coeff to manage font_size 
            }
mlObj= MLPlots(**plot_kws
               )
# additional scikit_lean precision recall keywords arguments 
prt_kws =dict()
#call object
mlObj.PrecisionRecall(clf = sgd_clf, 
                          X= X_train_2, 
                          y = y_prepared, 
                          classe_=classe_category, 
                          cv=cv,
                          kind=kind,
                         **prt_kws
                         )
