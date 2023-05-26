# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 19:06:11 2022

@author: Daniel
"""

import  unittest 
import  matplotlib.pyplot as plt 
from watex.view.plot import ( 
    ExPlot , 
    )
from watex.datasets import load_bagoue 
from watex.utils import naive_imputer 
from tests import  ( 
    make_temp_dir 
    ) 

from tests.__init__ import ( 
    reset_matplotlib,
    ) 
class TestExPlot(unittest.TestCase):
    """
    Test electrical resistivity profile  and compute geo-lectrical features 
    as followings : 
        - type 
        - shape 
        - sfi 
        - power 
        - magnitude
        - anr
        - select_best_point
        - select_best_value
    """
    data = naive_imputer ( load_bagoue().frame , mode ='bi-impute') 
    p = ExPlot(tname ='flow').fit(data)
    p.fig_size = (12, 4)
    
    @classmethod 
    def setUpClass(cls):
        """
        Reset building matplotlib plot and generate tempdir inputfiles 
        
        """
        reset_matplotlib()
        cls._temp_dir = make_temp_dir(cls.__name__)
        
    def test_plotparallelcoords (self): 
        self.p.plotparallelcoords(pkg ='yb')
        
    def test_plotpairwisecomparizon (self): 
        self.p.plotpairwisecomparison(fmt='.2f', corr ='spearman', pkg ='yb',  
                                     annot=True, 
                                     cmap='RdBu_r', 
                                     vmin=-1, 
                                     vmax=1 
                                     )
        plt.close() 
    def test_plotcutcomparison (self): 
        self.p.plotcutcomparison(xname ='sfi', yname='ohmS')
        
    def test_plotradviz (self): 
        self.p.plotradviz(classes= None, pkg='pd' )
        plt.close() 

    def test_plotbv (self): 
        for k in ('box', 'boxen', 'violen'): 
            self.p.plotbv(xname='flow', yname='sfi', kind=k)
            plt.close() 
            
    def test_plotpairgrid (self): 
        self.p.plotpairgrid (vars = ['magnitude', 'power', 'ohmS'] )
        plt.close () 
        
    def test_plotjoint (self): 
        for lib in ('sns', 'yb'): 
            self.p.plotjoint(xname ='magnitude' , pkg =lib)

        plt.close () 
    def test_plotscatter (self):
         
        self.p.plotscatter (
            xname ='sfi', yname='ohmS')
        plt.close () 
    def test_plothistvstarget (self ): 
        """ test histogram plot against target """
    
        #p.savefig ='bbox.png'
        self.p.plothistvstarget (xname= 'sfi', c = 0, kind = 'binarize',  kde=True, 
                          posilabel='dried borehole (m3/h)',
                          neglabel = 'accept. boreholes'
                          )
        plt.close () 
    def test_plothist(self ):
        """ test plot histogram """
        
        self.p.plothist('sfi', kind='hist')
        plt.close () 
    def test_plotmissing (self, ):
        """ test plot missing data """
        for k in('mbar', 'bar', 'corr', 'dendro', 'mpat'): 
            self.p.plotmissing(kind =k, sample =300 )
        plt.close () 
        
# if __name__=='__main__': 
#     unittest.main()  
        
        
    