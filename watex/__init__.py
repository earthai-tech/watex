# -*- coding: utf-8 -*-
import os 
import sys 

__version__='0.1'

from . import ( 
    analysis, 
    bases, 
    datasets, 
    geology, 
    methods, 
    modeling, 
    tools, 
    view,
    _watexlog, 
    decorators, 
    decorators, 
    documentation, 
    exceptions, 
    property, 
    sklearn,
    typing, 
    __main__, 
    
    )

if __name__ =='__main__' or __package__ is None: 
    sys.path.append( os.path.dirname(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.dirname(__file__))
    __package__ ='watex'
    # another way to say the sys.path so to force use the relative import: 
    #   sys.path(os.path.dirname(os.path.dirname(__file__)))
    #   root_folder = r'{}'.format(pathlib.Path(
    #       pathlib.Path(__file__).parent.absolute().parent))

