# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 11:59:06 2021

@author: @Daniel03
"""

import os
# import datetime
import  unittest 
# import pytest
from pprint import pprint 

import numpy as np 
# import pandas as pd 

from tests.__init__ import ( 
    ANALYSIS_DATA,
    make_temp_dir , TEST_TEMP_DIR , 
    reset_matplotlib
    ) 

from watex.analysis.dimensionality import (
    nPCA, kPCA, LLE, iPCA, 
   # get_best_kPCA_params
    )

X, y=ANALYSIS_DATA
class TestReducers(unittest.TestCase):
    """
    Test differents Reducers: 
        - PCA
        -IncrementalPCA
    """
    X=X 
    y=y
    n_components =None
    n_batches =100
    kernel = 'rbf'
    gamma= 0.01
    closest_neighbors=4
    pre_image=False
   
    
    @classmethod 
    def setUpClass(cls):
        """
        Reset building matplotlib plot and generate tempdir inputfiles 
        
        """
        reset_matplotlib()
        cls._temp_dir = make_temp_dir(cls.__name__)

    def setUp(self): 
        
        if not os.path.isdir(TEST_TEMP_DIR):
            print('--> outdir not exist , set to None !')
            
    def test_incrementalPCA(self, **kws ): 
        """ Test incremental PCA"""
         
        iPCA(X=self.X, n_components=self.n_components, 
                                n_batches=self.n_batches, 
                                store_in_binary_file=False, 
                                **kws)

    def test_kPCA(self, **kws): 
        """ Test kernel PCA """
        
        kPCA(X=self.X, n_components=self.n_components, 
                    kernel=self.kernel,reconstruct_pre_image=self.pre_image, 
                    gamma=self.gamma, **kws)
        
        #plot_projection(self.rObj ,self.rObj.n_components )
    
    # def test_get_best_kPCA_params(self): 
    #     """ Get the kpca hyperparameters using grid SearchCV"""
    #     param_grid =[{
    #             "kpca__gamma":np.linspace(0.03, 0.05, 10),
    #             "kpca__kernel":["rbf", "sigmoid"]
    #             }]
    #     from sklearn.pipeline import Pipeline 
    #     from sklearn.linear_model import LogisticRegression
    #     from sklearn.decomposition import KernelPCA
        
    #     clf =Pipeline([
    #         ('kpca', KernelPCA(n_components=self.n_components)),
    #         ('log_reg', LogisticRegression())
    #         ])
        
    #     kpca_best_param =get_best_kPCA_params(self.X,y=y,scoring = 'accuracy',
    #                                           n_components= 2, clf=clf, 
    #                                           param_grid=param_grid,)
        
    #     pprint(kpca_best_param)
        
    def test_LLE(self):
        """ Test Loccally Linear Embedding with closest neighbors"""
        
        lle_kws ={
                'n_components': 4, 
                  "n_neighbors": self.closest_neighbors}
        
        LLE(self.X, **lle_kws)
       
    def test_PCA (self):
        nPCA (self.X, n_components=self.n_components)
        
if __name__=='__main__': 
    unittest.main()
    #Reducers().incrementalPCA(X, n_batches=100)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    