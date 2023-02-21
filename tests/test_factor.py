# -*- coding: utf-8 -*-

import  unittest 
from watex.analysis.factor import ( 
    shrunk_cov_score , 
    pcavsfa, 
    make_scedastic_data
    ) 
from watex.datasets import fetch_data 

class TestFactor (unittest.TestCase) : 
    """ Test factor analysis """
    # fecth data for the test 
    X, _=fetch_data('Bagoue analysed data')
    
    def test_shrunk_cov_score(self): 
        self.assertLessEqual(shrunk_cov_score (self.X), -10)

    def test_pcavsfa (self): 

	    self.assertAlmostEqual(len (pcavsfa (self.X) ), 2)
        
    def test_make_make_scedastic_data  (self): 
        X, X_homo, X_hetero , n_components = make_scedastic_data ()  
        
                
if __name__=='__main__': 
    
    # TestFactor().test_make_make_scedastic_data()
    # TestFactor().test_shrunk_cov_score() 
    # TestFactor().test_pcavsfa() 
    unittest.main()