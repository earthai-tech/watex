"""
Geology sub-package is dedicated for structural informations and strata layer 
building. It uses some modeling data exported as two-dimensional matrices block
of resistivity (e.g `ModEM`_, `MTpy`_,  `pycsamt`_, `OCCAM2D`_) and combined 
with the geological informations on the survey area to build the stratigraphy 
logs. It also give an alternative way to draw some drilling logs after 
the drilling operations.

.. _OCCAM2D: https://marineemlab.ucsd.edu/Projects/Occam/index.html
.. _ModEM: https://sites.google.com/site/modularem/download
.. _pycsamt: https://github.com/WEgeophysics/pycsamt
.. _MTpy: https://github.com/MTgeophysics/mtpy
"""

from .geology import ( 
    Geology ,  
    Structures,
    Structural, 
    )
from .drilling import ( 
    Borehole 
    )
from .stratigraphic import  GeoStrataModel

__all__=[ 
    'Geology' ,  
    'Structures',
    'Structural', 
    'Borehole', 
    'GeoStrataModel', 
    
    ]