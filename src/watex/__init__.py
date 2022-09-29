# -*- coding: utf-8 -*-
# licence : MIT @copyright 2021 <etanoyau@gmail.com>
"""
💧 Machine Learning Research Package for Hydrogeophysic 
=========================================================

Originally called *WAT-er EX-ploration using AI learning methods*, `WATex`_ is
an open-source package entirely written in Python to bring a piece of solution 
in the field of groundwater exploration (GWE) of the wide program of WATER4ALL and
for the Sustanaible Development Goals N6 achievement( `SDGn6`_ ). And, packages 
and modules are written to solve real-engineering problems in the field of GWE. 
Currently, it deals with the differents methods: 
    
    * `geophysical (from DC-Electrical to Electromagnetic)` 
    * `hydrogeology (from drilling to  parameters calculation)`
    * `geology (for stratigraphic model generation)`
    
All methods mainly focus on the  field of groundwater explorations. One of 
the main advantage using `WATex`_ is the use of machine learning methods in the 
hydrogeophysic parameter predictions to minimize the risk of unsucessfull 
drillings and the hugely reduce the cost of the hydrogeology parameter 
collections.

.. _WATex: https://github.com/WEgeophysics/watex/
.. _SDGn6: https://www.un.org/sustainabledevelopment/development-agenda/

"""
import os 
import sys 
import logging 

__version__='0.1.2' ; __author__= 'LKouadio'


# set the package name 
if __name__ =='__main__' or __package__ is None: 
    sys.path.append( os.path.dirname(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.abspath('..'))
    __package__ ='watex'


# configure the logger 
from watex._watexlog import watexlog
try: 
    watexlog.load_configure(os.path.join(
        os.path.abspath('.'), "watex", "wlog.yml"))
except: 
    watexlog.load_configure(os.path.join(
        os.path.abspath('.'),'src', "wlog.yml"))

# set loging Level
logging.getLogger('matplotlib').setLevel(logging.WARNING)


    
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
