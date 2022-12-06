# -*- coding: utf-8 -*-
"""
Cythonize  "_openmp_helpers" 
=============================
we have choice to use 'pyximport' (Cython Compilation for Developers) or setup 
configuration. the latter one as recommended so. For further details 
refer to  http://docs.cython.org/en/latest/src/tutorial/cython_tutorial.html

Created on Wed Oct 12 21:14:16 2022

@author: Daniel
"""
#import pyximport; pyximport.install(pyimport=True)
from numpy.distutils.misc_util import Configuration

def configuration(parent_package="", top_path=None):
    """ Cythonize _openmp_helpers """
    config = Configuration("utils", parent_package, top_path)
    
    libraries=[]
    config.add_extension(
      "_openmp_helpers", sources=["_openmp_helpers.pyx"], libraries=libraries
      )
    config.add_subpackage("tests")
    
    return config 

if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration(top_path="").todict())