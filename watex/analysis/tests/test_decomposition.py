# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from watex.exlib.sklearn import SimpleImputer, LogisticRegression  
from watex.utils import selectfeatures 
from watex.datasets import fetch_data  
from watex.analysis.decomposition import (
    decision_region, linear_discriminant_analysis , 
   feature_transformation,  total_variance_ratio ,  extract_pca 
) 

data= fetch_data("bagoue original").get('data=dfy1') # encoded flow categories 
y = data.flow ; X= data.drop(columns='flow') 
# select the numerical features 
X =selectfeatures(X, include ='number')
# imputed the missing data 
X = SimpleImputer().fit_transform(X)
    
def test_extract_pca(): 
    eigval, eigvecs, _ = extract_pca(X)
    print(eigval) 
    
    # ... array([2.09220756, 1.43940464, 0.20251943, 1.08913226, 0.97512157,
    #        0.85749283, 0.64907948, 0.71364687])

def test_total_variance_ratio(): 
    # Use the X value in the example of `extract_pca` function   
    cum_var = total_variance_ratio(X, view=True)
    print(cum_var) 
    # ... array([0.26091916, 0.44042728, 0.57625294, 0.69786032, 0.80479823,
    #        0.89379712, 0.97474381, 1.        ])

def test_feature_transformation(): 
    # Use the X, y value in the example of `extract_pca` function  
    Xtransf = feature_transformation(X, y=y,  positive_class = 2 , view =True)
    print( Xtransf[0] ) 
    
    # ... array([-1.0168034 ,  2.56417088])


def test_decision_region(): 
    lr_clf = LogisticRegression(multi_class ='ovr', random_state =1, solver ='lbfgs') 
    Xpca= decision_region(X, y, clf=lr_clf, split = True, view ='Xt') # test set view
    print( Xpca[0] ) 

def test_linear_discriminant_analysis(): 

    Xtr= linear_discriminant_analysis (X, y , view =True)
    print(Xtr[0])
    
# if __name__=='__main__': 
#     test_extract_pca() 
#     test_total_variance_ratio() 
#     test_feature_transformation() 
#     test_decision_region() 
#     test_linear_discriminant_analysis() 
    