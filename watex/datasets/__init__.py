# -*- coding: utf-8 -*-
#   Created date: Wed Sep 15 11:39:43 2021 
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>.
""" 
Dataset 
==========
Fetch data set from the local machine. It data does not exist, retrieve it 
from online (repository or zenodo record ) 

"""

try:
    from watex.datasets.config import fetch_data
except : 
    from warnings import warn 
    from .._watexlog import watexlog
    
    m= ("None config file detected. Auto-data preparation process is aborted."
            "Be aware, the basics examples won't be implemented. Fetch data "
            " manually from repository or zenodo record using the module"
            " 'load' via < :mod:`watex.datasets.load` >"
            )
    watexlog().get_watex_logger(__name__).debug(m); warn(m, UserWarning)

     
    





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    