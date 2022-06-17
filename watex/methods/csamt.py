# -*- coding: utf-8 -*-
# Created on Tue May 17 11:30:51 2022

import warnings 

from .._watexlog import watexlog
from ..exceptions import ( 
    TopModuleError 
) 
from ..tools.funcutils import ( 
    is_installing  
    ) 


HAS_MOD=False 

try : 
    import pycsamt 
except ImportError: 
    try : 
        HAS_MOD=is_installing (
                'pycsamt'
                )
    except : pass 
else : 
    HAS_MOD=True 
    
if HAS_MOD : 
    from pycsamt.ff.core.avg import (
        Avg as AVG 
        )
    from pycsamt.ff.core.edi import (
        Edi_collection as EDI
        ) 
    from pycsamt.ff.core.j import  (
        J_collection as J 
        )
    from pycsamt.geodrill.geocore import ( 
        GeoStratigraphy, 
        Geodrill, 
        Geosurface, 
        )
    from pycsamt.geodrill.geodatabase import (
        GeoDataBase,
        Geo_formation
        )

_logger = watexlog.get_watex_logger(__name__)

class CSAMT :
    """ Deal with control source audio frequency magnetotelluric """
    
    def __new__(cls) : 
        if not HAS_MOD  :
            
            warnings.warn('Prior install the `pycsamt` module as '
                          '< pip install pycsamt>', 
                         UserWarning )
            raise TopModuleError( 'Module Not found. Prior install'
                                 'the module `pycsamt` instead.')
            
        return super().__new__(cls )
    
    def __init__ (self, **kwargs): 
        pass 
    