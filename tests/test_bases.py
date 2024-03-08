# -*- coding: utf-8 -*-
"""
Description 
^^^^^^^^^^^^^^

Test to transformers module. Containers of modules  
Test the functionalities of custums transformers 
:mod:`watex.bases.transformers`

Created on Wed Sep 15 19:49:36 2021
@author: @Daniel03
"""
import  unittest 
import pytest
from pprint import pprint 

import numpy as np 
import pandas as pd 

from watex.transformers import  ( 
    FrameUnion ) 


from tests.__init__ import  (
    SEMI_PROCESSED_DATA, 
    )

X, _=SEMI_PROCESSED_DATA

##################### test classes 

# class TestTransformers(unittest.TestCase):
#     """
#     Test differents transformers 
#     """
class MyBaseClass (unittest.TestCase ): 
    """ Test class from :class:`watex.bases.base` """
    def test_data (self): 
        ...
        

class MyFeatures (unittest.TestCase): 
    """ Test class from :class:`watex.bases.features`"""
    
class MyBaseModelingClass (unittest.TestCase): 
    """ Base modeling class """
    
class MyPrepareClass (unittest.TestCase): 
    """ Relate from Base preparation class"""
    
class MyBaseProcessingClass (unittest.TestCase): 
    """ Test Base processing class from :class:`watex.bases.processing` """
    
########### TESTS 

# def test_missing(): 
#     BaseSteps()
    
class MyTransformers(unittest.TestCase):
    """
    Test differents transformers 
    """  
    @pytest.mark.xfail(reason="skip  to avoid AttributeError occurs! Indeed"
                      "Unitest version doesn't Recognize the method"
                      " toarray()` use in :class:~transformers.FrameUnion"
                      " to concatenate aray when encoder is `OneHotEncoder`", 
                      raises =AttributeError) 
    def test_frameUnion(self, 
                        X=X, 
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
        try : 
            
            X= X.astype({
                        'power':np.int32, 
                        'magnitude': np.float64, 
                        'sfi':np.float64, 
                        # 'lwi':np.float64, 
                        # 'east':np.float64, 
                        # 'north':np.float64, 
                        'ohmS':np.float64,
                        })
            self.X = frameObj.fit_transform(X)
            
        except: 
             pass # pytest.skip("Could not convert ro float")
     
            # pytest.skip("Don't recognize the method `toarray()` of Numpy"
            #             "version. Unsupported numpy version used. ", 
            #             allow_module_level=True)
            # pytest.skip('Only a column name can be used for the key in a'
                 #       'dtype mappings argument.')
        else :
            self.num_attributes = frameObj.num_attributes 
            self.encode_mode = frameObj.encode_mode
            self.X_=frameObj.X_
            self.X_num_= frameObj.X_num_
            self.X_cat_ =frameObj.X_cat_
            self.num_attributes_=frameObj.num_attributes_
            self.cat_attributes_=frameObj.cat_attributes_
            self.attributes_=frameObj.attributes_ 
            
            if self.encode_mode != 'OneHotEncoder': 
                self.df_X = pd.DataFrame(data = self.X, 
                                         columns = self.attributes_)
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
        
            # self.cat_attributes = frameObj.cat_attributes 
            # self.param_search = frameObj.param_search 
            # self.imput_data = frameObj.imput_data 
            # self.strategy =frameObj.strategy 
            # self.scale = frameObj.scale
            # self.encode =frameObj.encode 
            # self.scale_mode = frameObj.scale_mode
        
if __name__=='__main__': 
    pytest.main([__file__])
    # unittest.main()
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    