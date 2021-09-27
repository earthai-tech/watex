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

from tests.__init__ import DATA_ANALYSES
from tests import  make_temp_dir , TEST_TEMP_DIR 
from tests.core.__init__ import reset_matplotlib, watexlog
from watex.analysis.dimensionality import (Reducers,plot_projection, 
                                           get_best_kPCA_params)

X, y=DATA_ANALYSES
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
    rObj = Reducers()
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
            watexlog.get_watex_logger().error('Outdir does not exist !')
            
    def test_incrementalPCA(self, **kws ): 
        """ Test incremental PCA"""
         
        self.rObj.incrementalPCA(X=self.X, n_components=self.n_components, 
                                n_batches=self.n_batches, 
                                store_in_binary_file=False, 
                                **kws)
        
        pprint(self.rObj.feature_importances_)
        
        plot_projection(self.rObj ,self.rObj.n_components )
        
    def test_kPCA(self, **kws): 
        """ Test kernel PCA """
        
        self.rObj.kPCA(X=self.X, n_components=self.n_components, 
                                kernel=self.kernel,reconstruct_pre_image=self.pre_image, 
                                gamma=self.gamma, **kws)
        
        pprint(self.rObj.feature_importances_)
        
        #plot_projection(self.rObj ,self.rObj.n_components )
    
    def test_get_best_kPCA_params(self): 
        """ Get the kpca hyperparameters using grid SearchCV"""
        param_grid =[{
                "kpca__gamma":np.linspace(0.03, 0.05, 10),
                "kpca__kernel":["rbf", "sigmoid"]
                }]
        from sklearn.pipeline import Pipeline 
        from sklearn.linear_model import LogisticRegression
        from sklearn.decomposition import KernelPCA
        
        clf =Pipeline([
            ('kpca', KernelPCA(n_components=self.n_components)),
            ('log_reg', LogisticRegression())
            ])
        
        kpca_best_param =get_best_kPCA_params(self.X,y=y,scoring = 'accuracy',
                                              n_components= 2, clf=clf, 
                                              param_grid=param_grid,)
        
        pprint(kpca_best_param)
        
    def test_LLE(self):
        """ Test Loccally Linear Embedding with closest neighbors"""
        
        lle_kws ={
                'n_components': 4, 
                  "n_neighbors": self.closest_neighbors}
        
        self.rObj.LLE(self.X,# n_components=4,
                      **lle_kws)
        pprint(self.rObj.__dict__)
        
if __name__=='__main__': 
    unittest.main()
    #Reducers().incrementalPCA(X, n_batches=100)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    