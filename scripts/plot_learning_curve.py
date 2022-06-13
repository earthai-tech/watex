# -*- coding: utf-8 -*-
# @author: Kouadio K. Laurent alias Daniel03
"""
.. sypnosis:: Compute the train score and validation curve to visualize 
        your learning curve
        
Created on Tue Sep 21 15:29:13 2021

@author: @Daniel03
"""
from sklearn.svm import SVC

from watex.bases.modeling import BaseModel
# modules below are imported for testing scripts.
# Not usefull to import at least  you provide your own dataset.
from watex.datasets import fetch_data 

X_prepared,  y_prepared = fetch_data('Bagoue data prepared')
 
#dataset path to to csv file.
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
baseEstimator =SVC( random_state=42) #(random_state=42, C=10, gamma=1e-2, kernel ='rbf')

# processing module kwargs
modeling_kws={
        'categorial_features': None,
        'numerical_features': None,
        'target': 'flow',
        'drop_features':['lwi'],        # drop useless features
        'random_state': 42,
        'default_estimator': 'svc',         # default estimator. If set dont need to import SVM
        'test_size': 0.2,                   # test size, default is 20%
        'col_id': 'name',
    }
# estimator hyperparameters configuration : for SVM, the default is `C`
learning_curve_kws = dict() 
# {"param_name":'C', 
#                 "param_range": np.arange(1,100,10), 
#                 "cv":4      # cross validarion k-Folds(CV=4) default values
#                 }
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
processObj = BaseModel(
                    data_fn =datafn, 
                    df = df, 
                    **modeling_kws
    )

processObj.get_learning_curve (
    estimator = baseEstimator,
    X_train =trainset,
    y_train = y_array,
    switch_plot=switch_plot, 
    preprocessor=autoPreprocessing, 
    learning_curve_kws= learning_curve_kws
    )