"""
=================================================
Plot decision regions 
=================================================

displays the decision region for the training data reduced to 
two principal component axes. 

"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

#%%
# Fetch the test data and do
# select the numerical features and 
# impute the missing values 

from watex.datasets import fetch_data 
from watex.exlib.sklearn import SimpleImputer, LogisticRegression  
from watex.analysis.decomposition import decision_region 
from watex.utils import selectfeatures
data= fetch_data("bagoue original").get('data=dfy1') # encoded flow categories 
y = data.flow ; X= data.drop(columns='flow') 
# select the numerical features 
X =selectfeatures(X, include ='number')
# imputed the missing data 
X = SimpleImputer().fit_transform(X)
lr_clf = LogisticRegression(multi_class ='ovr', random_state =1, solver ='lbfgs') 
_=decision_region(X, y, clf=lr_clf, split = True, view ='Xt') # test set view
