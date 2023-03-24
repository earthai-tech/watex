# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from watex.datasets import fetch_data
from watex.analysis import pcavsfa , LW_score, make_scedastic_data

X, _= fetch_data('Bagoue analysed dataset')

def test_pcavsfa (): 
    pcavsfa(X) 
    
def test_LW_score(): 
    LW_score(X) 
    
def test_make_scedastic_data (): 
    make_scedastic_data (n_samples = 2000, n_features =70, 
                         rank = 20  )
# if __name__=='__main__': 
#     test_pcavsfa() 
#     test_LW_score() 
#     test_make_scedastic_data() 


