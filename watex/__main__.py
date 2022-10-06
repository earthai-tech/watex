# -*- coding: utf-8 -*-

import os 
import sys 
import logging 

__version__='0.1.2' ; __author__= 'LKouadio'


# set the package name 
# for consistency ckecker 
sys.path.insert(0, os.path.dirname(__file__))  
for p in ('.','..' ,'./watex'): 
    sys.path.insert(0,  os.path.abspath(p)) 
    
# assert packages 
if  __package__ is None: 
    sys.path.append( os.path.dirname(__file__))
    __package__ ='watex'


# configure the logger 
from ._watexlog import watexlog

try: 
    watexlog.load_configure(os.path.join(
        os.path.dirname(__file__),  "wlog.yml"))
except: 
    watexlog.load_configure(os.path.join(
        os.path.abspath('.'), "wlog.yml"))

# set loging Level
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# ================
# import required modules 
    
from . import ( 
    analysis, 
    bases, 
    datasets, 
    geology, 
    methods, 
    models, 
    tools, 
    view,
    )
