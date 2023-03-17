# -*- coding: utf-8 -*-
"""
    Module to test all utilities 
Created on Wed Apr 27 13:18:23 2022

@author: @Daniel03
"""
import os
# import datetime
import  unittest 
# import pytest
import copy 
import numpy as np 
import pandas as pd 

from tests import (
    ORIGINAL_DATA, 
    DATA_UNSAFE_XLS, 
    DATA_UNSAFE, 
    DATA_SAFE, 
    DATA_SAFE_XLS,
    DATA_EXTRA ,
    PREFIX, 
)
from tests.random_sample import (
    array1D , 
    array2D, 
    array2DX, 
    extraarray2D, 
)
from watex.utils.coreutils import  ( 
    erpSelector
    ) 

from watex.utils.mlutils import ( 
    exporttarget , 
    correlatedfeatures
    ) 
from watex.utils.funcutils import ( 
    shrunkformat )


class TestTools(unittest.TestCase):
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
    data_collections = ( DATA_UNSAFE_XLS, DATA_UNSAFE, DATA_SAFE,
                        DATA_SAFE_XLS, DATA_EXTRA, 
        array1D , array2D, array2D [:, :2],  array2DX, extraarray2D
        
        )
    df = copy.deepcopy(ORIGINAL_DATA.get ('data=dfy1')) 
    
    def test_skrunkformat (self): 
        """ shortcut text by adding  ellipsis """
        arr = np.arange(30)
        text =" I'm a long text and I will be shrunked and replace by ellipsis."
        
        ftext = shrunkformat (text, insert_at ='end')
        # print(ftext)
        self.assertEqual('Im a long ... ', ftext)
        ftext2 = shrunkformat (arr, insert_at ='begin')
        self.assertEqual(' ...  26 27 28 29', ftext2)
        
    def test_correlated_columns (self): 
        """ assert the correlated columns """
        try: 
            df_corr = correlatedfeatures (self.df , corr='spearman', 
                                         fmt=None, threshold='80')
        except ValueError : 
            self.assertRaises(ValueError)
        df_corr = correlatedfeatures (self.df , corr='spearman', 
                                     fmt=None, threshold='80%')
        self.assertEqual(0, len(df_corr))
        
    def test_exporttarget (self): 
        """ Assert the target exportation, modified or not the retur values """

        #df = copy.deepcopy(ORIGINAL_DATA.get ('data=dfy1')) 
        data0 = copy.deepcopy(self.df) 
        
        # no modification 
        # tname = 'flow'
        target, data_no = exporttarget (data0 , 'flow', False )
        self.assertEqual(len(data_no.columns ) , len(data0.columns ) )
        explen = len(self.df.columns)  -1   
        # modified in place 
        target, data= exporttarget (data0 , 'flow')
        self.assertEqual ( len(data.columns ) , len(data0.columns ) ) 
        # expect to get a length minum 01 ( target columns )
        self.assertEqual(explen, len(data.columns ))
   

    def test_erpSelector  (self) :
        """ Test the capability of the  func to  read and fetch data 
        straigthly from `csv` and `xlsx` and sanitize data to fit the 
        corresponding ``PREFIX``. """

        for i, f in enumerate(self.data_collections[:4]):
            print('i=', i)
            df =  erpSelector( f, force=True)
            col = list(df.columns) if isinstance(
                df, pd.DataFrame) else [df.name] # for Series
            print(df.head(2))
            if os.path.isfile (f): 
                print(os.path.basename(
                    os.path.splitext(f)[0].lower()) )
                if os.path.basename(
                        os.path.splitext(f)[0].lower()) in (
                        'testunsafedata', 'testunsafedata_extra'): 
                    print('PASSED')
                    print('col ==', col)
                    self.assertListEqual(col , PREFIX)
                    
                elif os.path.basename(
                        os.path.splitext(f)[0].lower()) =='testsafedata': 
                    self.assertEqual(len(col), len(PREFIX),
                        f'The length of data columns={col}  is '
                        f' different from the expected length ={len(PREFIX)}.')
   
            elif isinstance(f, pd.Series): 
                self.assertListEqual (col , ['resistivity'], 
                                     'Expected a sery of "resistivity" by got'
                                     f'{f.name}')
            elif isinstance(f, pd.DataFrame): 
                self.assertListEqual (col , ['station', 'resistivity'], 
                        'Expected a sery of "[station , resistivity]" by got'
                        f'{col}')
                
# if __name__=='__main__': 

#     unittest.main()

    












































