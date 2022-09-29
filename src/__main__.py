# -*- coding: utf-8 -*-
"""
💧 Machine Learning Research Package for Hydrogeophysic 
======================================================

Originally called *WAT-er EX-ploration using AI learning methods*, `WATex`_ is
an open-source package entirely written in Python to bring a piece of solution 
in the field of groundwater exploration (GWE) in a wide program of WATER4ALL and
for the Sustanaible Development Goals N6 achievement( `SDGn6`_ ). And, packages 
and modules are written to solve real-engineering problems in the field of GWE. 
Currently, it deals with the differents methods: 
    
    * geophysical (from DC-Electrical to Electromagnetic) 
    * hydrogeology (from drilling to  parameters calculation)
    * geology (for stratigraphic model generation)
    
All methods mainly focus on the  field of groundwater explorations. One of 
the main advantage using `WATex`_ is the use of machine learning methods in the 
hydrogeophysic parameter predictions to minimize the risk of unsucessfull 
drillings and the hugely reduce the cost of the hydrogeology parameter 
collections.

.. _WATex: https://github.com/WEgeophysics/watex/
.. _SDGn6: https://www.un.org/sustainabledevelopment/development-agenda/
"""

#import required modules
# set the package name 
import os 
import sys 
if  __package__ is None: 
    sys.path.insert(0, os.path.abspath('.'))
    sys.path.insert(0, os.path.abspath('..'))
    __package__ ='watex'
    
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

