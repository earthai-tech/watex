# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 17:32:49 2022

@author: Daniel
"""


# import datetime
import  unittest 

import numpy as np 
from watex.exlib.sklearn import SimpleImputer, LogisticRegression 
from watex.utils import selectfeatures 
from watex.datasets import fetch_data 
from watex.analysis import ( 
    extract_pca , feature_transformation,
    total_variance_ratio, decision_region
    )

class TestDecomposition(unittest.TestCase):
    data= fetch_data("bagoue original").get('data=dfy1') # encoded flow categories 
    y = data.flow ; X= data.drop(columns='flow') 
    # select the numerical features 
    X =selectfeatures(X, include ='number')
    # imputed the missing data 
    X = SimpleImputer().fit_transform(X)
    
    def test_extract_pca (self): 
        eigval, eigvecs, _ = extract_pca(self.X)
        self.assertEqual(len(eigval,), 8)
    def test_total_variance_ratio (self): 
        cum_var = total_variance_ratio(self.X, view=True)
        self.assertEqual(len(cum_var), 8)
    def test_feature_transformation (self): 
        Xtransf = feature_transformation(
            self.X, y=self.y,  positive_class = 2 , view =True)
        print(np.around (Xtransf[0].sum(), 1))
        print(round(np.array([-1.01 ,  2.56]).sum(), 1))
        self.assertAlmostEqual(
            abs( np.around (Xtransf[0].sum(), 1))  , 
                               abs( round(np.array([-1.01 ,  2.56]).sum(), 1)) , 
                               places =7) 
    def test_decision_region (self): 
        lr_clf = LogisticRegression(multi_class ='ovr', random_state =1, solver ='lbfgs') 
        Xpca= decision_region(self.X, self.y, clf=lr_clf, split = True, view ='Xt') # test set view
        self.assertAlmostEqual(
            abs ( np.around (Xpca[0].sum(), 1))   ,
                               abs( round (np.array([-1.03,  1.42]).sum(), 1) ), 
                               places = 7)
    
# if __name__=='__main__': 
#     #unittest.main()
#     TestDecomposition().test_decision_region() 
    