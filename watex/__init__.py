# -*- coding: utf-8 -*-
# Licence:BSD 3-Clause
# author: @Daniel<etanoyau@gmail.com>
"""
💧 Machine Learning Research Package for Hydrogeophysic 
=========================================================

:code:`watex` stands fpor *WAT-er EX-ploration, it is an open-source package 
entirely written in Python to bring a piece of solution in the field of 
groundwater exploration (GWE) of the wide program of WATER4ALL and for the 
Sustanaible Development Goals N6 achievement( `SDGn6`_ ). And, packages and 
modules are written to solve real-engineering problems in the field of GWE. 
Currently, it deals with the differents methods: 
    
    * `geophysical (from DC-Electrical to Electromagnetic)` 
    * `hydrogeology (from drilling to parameters calculation)`
    * `geology (for stratigraphic model generation)`
    * `predicting permeability coefficient (k), flow rate and else` 
    
All methods mainly focus on GWE field. One of the main advantage using `WATex`_ 
is the application of machine learning methods in the hydrogeophysic parameter 
predictions. It contributes to minimize the risk of unsucessfull drillings and 
the hugely reduce the cost of the hydrogeology parameter collections.

.. _WATex: https://github.com/WEgeophysics/watex/
.. _SDGn6: https://www.un.org/sustainabledevelopment/development-agenda/

"""
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
    conffile = os.path.join(
        os.path.dirname(__file__),  "watex/wlog.yml")
    if not os.path.isfile (conffile ): 
        raise 
except: 
    conffile = os.path.join(
        os.path.dirname(__file__), "wlog.yml")

watexlog.load_configure(conffile)
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


