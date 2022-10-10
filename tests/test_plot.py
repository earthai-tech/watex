# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 19:06:11 2022

@author: Daniel
"""

import os
# import datetime
import  unittest 
import pytest
import pandas as pd 
from watex.view.plot import ( 
    ExPlot , 
    QuickPlot 
    )
from tests import  ( 
    ERP_PATH, TEST_TEMP_DIR,  
    make_temp_dir 
    ) 

from tests.__init__ import ( 
    reset_matplotlib, watexlog 
    ) 
class TestExplot(unittest.TestCase):
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
    data = pd.read_csv ( 'data/geodata/main.bagciv.data.csv' ) 
    p = ExPlot(tname ='flow').fit(data)
    
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
            
    def test_plothistvstarget (self ): 
        """ test histogram plot against target """
    
        self.p.fig_size = (12, 4)
        #p.savefig ='bbox.png'
        self.p.plothistvstarget (name= 'sfi', c = 0, kind = 'binarize',  kde=True, 
                          posilabel='dried borehole (m3/h)',
                          neglabel = 'accept. boreholes'
                          )
    def test_plothistogram (self ):
        """ test plot histogram """
        
        self.p.plothistogram('sfi', kind='hist')
        
    def test_plotmissing (self, ):
        """ test plot missing data """
        for k in('mbar', 'bar', 'corr', 'dendro', 'mpat'): 
            self.p.plotmissing(kind =k, sample =300 )

        
        
if __name__=='__main__': 
    
   unittest.main()  
        
        
    