# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 11:59:06 2021

@author: @Daniel03
"""

import  unittest 
from tests.__init__ import ( 
    ANALYSIS_DATA,
    make_temp_dir, 
    reset_matplotlib
    ) 

from watex.analysis.dimensionality import (
    nPCA, kPCA, LLE, iPCA, 
    )

X, y=ANALYSIS_DATA

class TestDimensionality(unittest.TestCase):
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

    def test_incrementalPCA(self, **kws ): 
        """ Test incremental PCA"""
         
        iPCA(X=self.X, n_components=self.n_components, 
                                n_batches=self.n_batches, 
                                store_in_binary_file=False, 
                                **kws)

    def test_kPCA(self, **kws): 
        """ Test kernel PCA """
        
        kPCA(X=self.X, n_components=self.n_components, 
                    kernel=self.kernel,
                    reconstruct_pre_image=self.pre_image, 
                    gamma=self.gamma, **kws)
        
        #plot_projection(self.rObj ,self.rObj.n_components )
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
    # TestDimensionality().test_incrementalPCA()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    