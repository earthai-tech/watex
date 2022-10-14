# -*- coding: utf-8 -*-
#   Created date: Wed Sep 15 11:39:43 2021 
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>.
""" 
Dataset 
==========
Fetch data from the local machine. If data does not exist, retrieve it 
from online (repository or zenodo record ) 

"""
try:
    from ._config import _fetch_data
    from .dload import (
        load_bagoue , 
        load_gbalo, 
        load_iris, 
        load_semien, 
        load_tankesse , 
        load_boundiali, 
        )
    
    __all__=["fetch_data", "load_bagoue" , "load_gbalo", 
             "load_iris", "load_semien", "load_tankesse", 
             "load_boundiali"
             ]

except ImportError : 
    from warnings import warn 
    from .._watexlog import watexlog
    
    m= ("None config file detected. Auto-data preparation process is aborted."
            "Be aware, the basics examples won't be implemented. Fetch data "
            " manually from repository or zenodo record using the module"
            " 'load' via < :mod:`watex.datasets.load` >"
            )
    watexlog().get_watex_logger(__name__).debug(m); warn(m, UserWarning)


def fetch_data (tag, **kws): 
    # if tag=='bagoue': return load_bagoue()
    # if tag=='tankesse': return load_tankesse 
    func= _fetch_data 
    funcs= (load_bagoue , load_gbalo, load_iris, load_semien, load_tankesse , 
            load_boundiali)
    funcns = tuple (map(lambda f: f.__name__.replace('load_', ''), funcs))
    if tag in (funcns): 
        func = funcs[funcns.index (tag)] 
    return func (tag=tag, **kws)


fetch_data.__doc__ ="""\
Fetch dataset from 'tag'. A tag correspond to each level of data 
processing. 

An example of retrieving Bagoue dataset can be experimented.

Parameters 
------------
tag: str,  
    stage of data processing. Tthere are different options to retrieve data
    Could be:
        
    * ['original'] => original or raw data -& returns a dict of details 
        contex combine with get method to get the dataframe like::
            
            >>> fetch_data ('bagoue original').get ('data=df')
    * ['stratified'] => stratification data
    * ['mid' |'semi'|'preprocess'|'fit']=> data cleaned with 
        attributes experience combinaisons.
    * ['pipe']=>  default pipeline created during the data preparing.
    * ['analyses'|'pca'|'reduce dimension']=> data with text attributes
        only encoded using the ordinal encoder +  attributes  combinaisons. 
    * ['test'] => stratified test set data

       
Returns
-------
    `data`: Original data 
    `X`, `y` : Stratified train set and training target 
    `X0`, `y0`: data cleaned after dropping useless features and combined 
        numerical attributes combinaisons if ``True``
    `X_prepared`, `y_prepared`: Data prepared after applying  all the 
       transformation via the transformer (pipeline). 
    `XT`, `yT` : stratified test set and test label  
    `_X`: Stratified training set for data analysis. So None sparse
        matrix is contained. The text attributes (categorical) are converted 
        using Ordianal Encoder.  
    `_pipeline`: the default pipeline.     
 
"""    





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    