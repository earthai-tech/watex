# -*- coding: utf-8 -*-
"""
... synopsis::
    Compute the validation score and plot the validation curve if 
         the argument `turn` of decorator is switched to ``on``. If 
         validation keywords arguments `val_curve_kws` doest not contain a 
         `param_range` key, the default param_range should be the one of 
         decorator.
             ...
             
:Notes: If run the scripts without setting your own processing and pipelines as
keywards arguments in module `Processing`, will use the default processing
step. 
Created on Tue Sep 21 11:09:55 2021

@author: @Daniel03
"""
import numpy as np 
from sklearn.svm import SVC

from watex.bases import Processing 
# modules below are imported for testing scripts.
# Not usefull to import since you privied your own dataset.
from watex.datasets import fetch_data 

X_prepared,  y_prepared = fetch_data('Bagoue data prepared')

#dataset 
datafn =None # r'F:/repositories/watex/data/Bag.main&rawds/drfats/BagoueDataset2.csv'
# dataframe 
df = None
# trainset 
# trainset and ytrain are set, to call the rawf files `data_fn`.
trainset= X_prepared
#replace all nan value 

# y -labels 
y_array =y_prepared
# base estimator. If baseeastimator is set, should replace the default estimator 
baseEstimator =SVC(random_state=42)

# processing module kwargs
processing_kws={
        'categorial_features': None,
        'numerical_features': None,
        'target': 'flow',
        'drop_features':['lwi'],        # drop useless features
        'random_state': 42,
        'default_estimator': 'svc',         # default estimator. If set dont need to import SVM
        'test_size': 0.2,                   # test size, default is 20%
        'col_id': 'name',
    }
processObj = Processing(
            # data_fn = datafn ,
            df=df, 
            **processing_kws)
# estimator hyperparameters configuration : for SVM, the default is `C`
val_curve_kws = {"param_name":'C', 
                "param_range": np.arange(1,100,10), 
                "cv":7      # cross validarion k-Folds(CV=4) default values
                }
# plot properties 
val_kws ={'c':'blue',
          # 'linewidth' :3, 
           'marker':'o',
          'alpha' :1, 
        'label':'Validation curve',
          'edgecolors' : 'b',
          'linewidths' : 7.,
          'linestyles' : '-',
          'facecolors' : 'r',
                   }

train_kws ={'c':'r',
            # 'linewidth':3.,
            'marker':'s', 
            'alpha' :1,
            'label':'Training curve', 
            'edgecolors' : 'k',
              'linewidths' : 5,
              'linestyles' : '-',
              'facecolors' : 'k'
         }

# if None :
#     val_curve_kws = {"param_name":'C', 
#                              "param_range": np.arange(1,210,10), 
#                              "cv":4}
# trigger the decorator for plot 
switch_plot='on'
# automatize the default preprocessing. When data is already prepared, 
# dont need to enable the auto-processing 
autoPreprocessing =False 

#call get_validation method drom processing Object 
processObj.get_validation_curve( 
    estimator =baseEstimator,
    switch_plot=switch_plot,
    preprocess_step=autoPreprocessing ,
    X_train =trainset,
    y_train = y_array,
    val_curve_kws=val_curve_kws,
    val_kws =val_kws ,
    train_kws =train_kws 
    )
















