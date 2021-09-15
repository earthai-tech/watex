# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 19:49:36 2021
Description:
        Test to transformers module . Containers of modules  
        Test the functionalities of custums transformers 

    References:
        .. _module-core::`watex.utils.transformers`
@author: @Daniel03
"""
# import datetime
import  unittest 
import pytest
from pprint import pprint 

import numpy as np 
import pandas as pd 

from watex.utils.transformers import FrameUnion
from watex.datasets import X_

class TestTransformers(unittest.TestCase):
    """
    Test differents transformers 
    """

    @pytest.mark.skip(reason="skip  to avoid AttributeError occurs! Indeed"
                      "Unitest version doesn't Recognize the method"
                      " toarray()` use in :class:~transformers.FrameUnion"
                      " to concatenate aray when encoder is `OneHotEncoder`")  
    def test_frameUnion(self, 
                        X=X_, 
                        num_attributes =None , 
                        cat_attributes =None,
                        scale =True,
                        imput_data=True,
                        encode =True, 
                        param_search ='auto', 
                        strategy ='median', 
                        scale_mode ='StandardScaler', 
                        encode_mode ='OrdinalEncoder', **kws): 
        
        """ Unified a categorial features and numerical features after 
        applying scaling on numerical features and encoding categorical
        features  """
        f= 0
        display =kws.pop('display', True)
        
        frameObj = FrameUnion(num_attributes =num_attributes , 
                              cat_attributes =cat_attributes,
                              scale =scale,
                              imput_data=imput_data,
                              encode =encode, 
                              param_search =param_search, 
                              strategy =strategy, 
                              scale_mode =scale_mode, 
                              encode_mode =encode_mode)
        X= X.astype({
                    'power':np.int32, 
                    'magnitude': np.float64, 
                    'sfi':np.float64, 
                    'lwi':np.float64, 
                    # 'east':np.float64, 
                    # 'north':np.float64, 
                    'ohmS':np.float64,
                    })
        self.X = frameObj.fit_transform(X)
        
        self.num_attributes = frameObj.num_attributes 
        self.cat_attributes = frameObj.cat_attributes 
        self.param_search = frameObj.param_search 
        self.imput_data = frameObj.imput_data 
        self.strategy =frameObj.strategy 
        self.scale = frameObj.scale
        self.encode =frameObj.encode 
        self.scale_mode = frameObj.scale_mode
        self.encode_mode = frameObj.encode_mode

        self.X_=frameObj.X_
        self.X_num_= frameObj.X_num_
        self.X_cat_ =frameObj.X_cat_
        self.num_attributes_=frameObj.num_attributes_
        self.cat_attributes_=frameObj.cat_attributes_
        self.attributes_=frameObj.attributes_ 
        
        if self.encode_mode != 'OneHotEncoder': 
            self.df_X = pd.DataFrame(data = self.X, columns = self.attributes_)
            f=1
            
        if display : 
            pprint('List of attributes in dataset:')
            for name , value in zip(['num. attributes', 
                                     'cat. attributes', 
                                     'Total attributes'],
                                    [self.num_attributes_, 
                                    self.cat_attributes_, 
                                     self.attributes_]):
                                                     
                print(name,':',  value)
            pprint('X shape values:')
            for shape , svalue in zip(['X tranformed', 'Raw X', 
                                      'num. X', 'cat. X'], 
                                     [self.X.shape, 
                                    self.X_.shape, 
                                    self.X_num_.shape, 
                                    self.X_cat_.shape]): 
                                                
            
                print(shape, ':', svalue)
        
            if f ==1: 
                pprint('Headvalues of new dataframe X:')
                print(self.df_X.head())
        
        
if __name__=='__main__': 
    unittest.main()
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    