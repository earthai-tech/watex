# -*- coding: utf-8 -*-
# licence : MIT @copyright 2021 <etanoyau@gmail.com>
"""
💧 Machine Learning Research Module for Hydrogeophysic 
======================================================

Originally called **WAT-er EX-ploration using AI learning methods*, `WATex`_ is
an open-source package entirely written in Python to bring a piece of solution 
in the field of groundwater exploration (GWE) via the use of ML learning methods. 
Currently, it deals with the geophysical methods (Electrical and Electromagnetic
methods). And, Modules and packages are written to solve real-engineering 
problems in the field of GWE. Later, it expects to add other methods such as the
induced polarisation and the near surface refraction-seismic for environmental
purposes (especially, for cavities detection to preserve the productive aquifers) 
as well as including pure Hydrogeology methods. 

.. _WATex: https://github.com/WEgeophysics/watex/

"""
import os 
import sys 
import logging 

__version__='0.1.2' ; __author__= 'LKouadio'


# set the package name 
if __name__ =='__main__' or __package__ is None: 
    sys.path.append( os.path.dirname(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.dirname(__file__))
    __package__ ='watex'


# configure the logger 
from ._watexlog import watexlog
try: 
    watexlog.load_configure(os.path.join(
        os.path.abspath('.'),'watex', "wlog.yml"))
except: 
    watexlog.load_configure(os.path.join(
        os.path.abspath('.'), "wlog.yml"))

# set loging Level
logging.getLogger('matplotlib').setLevel(logging.WARNING)

#import required modules 
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
