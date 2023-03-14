"""
==============================================
Linear discriminant analysis (LDA)
==============================================

visualizes the LDA to increase the 
computational efficiency and reduce the degree of overfitting 
due to the curse of dimensionality in non-regularized models.
"""

# Author: L.Kouadio 
# Licence: BSD-3 clause

#%%
from watex.datasets import fetch_data 
from watex.utils import selectfeatures
from watex.exlib.sklearn import SimpleImputer  
from watex.analysis.decomposition import linear_discriminant_analysis 
data= fetch_data("bagoue original").get('data=dfy1') # encoded flow
y = data.flow ; X= data.drop(columns='flow') 
# select the numerical features 
X =selectfeatures(X, include ='number')
# imputed the missing data 
X = SimpleImputer().fit_transform(X)
linear_discriminant_analysis (X, y , view =True)

