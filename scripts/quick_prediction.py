# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 14:50:36 2021

@author: @Daniel03
"""

#--------------Evaluate your model on the test data ------------------------------
from watex.datasets._m import XT_prepared, yT_prepared
from watex.bases import fetch_model, predict 

model_dumped_file = 'data/my_model/SVC__LinearSVC__LogisticRegression.pkl'
my_model, *_ = fetch_model(model_dumped_file, modname ='SVC') 
#---------------------------------------------------------------------------------
# X_prepared,  y_prepared = fetch_data('Bagoue prepared datasets')

results = predict(y_true = yT_prepared, X_= XT_prepared, clf= my_model)
print(results)