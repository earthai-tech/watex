# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from watex.analysis.dimensionality import nPCA, iPCA, kPCA, LLE
from watex.datasets import fetch_data
X, _= fetch_data('Bagoue analysed dataset')
 
def test_nPCA (): 
   
    pca = nPCA(X, 0.95, n_axes =3, return_X=False)
    pca.components_
    print(pca.feature_importances_)
def test_iPCA():
    Xtransf = iPCA(X,n_components=None,n_batches=100, view=True)
    print(Xtransf[:7, :]) 
def test_kPCA (): 
    Xtransf=kPCA(X,n_components=None,kernel='rbf', 
                 gamma=0.04, 
                 )
    print(Xtransf.shape) 
def test_LLE () : 
    lle_kws ={
       'n_components': 4, 
        "n_neighbors": 5}
    Xtransf=LLE(X,**lle_kws)
    print(Xtransf[:7, :]) 
    
    
# if __name__=='__main__': 
#     test_nPCA() 
#     test_iPCA() 
#     test_kPCA() 
#     test_LLE() 
    
    