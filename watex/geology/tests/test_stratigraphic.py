# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import numpy as np 
from watex.geology import GeoStrataModel

seed = 12
 
def test_GeoStrataModel (): 
    # use the a to examples 
    np.random.seed (seed)
    layers = ['granites', 'gneiss', 'sedim.']
    crm = np.abs( np.random.randn (21, 7) *100 )
    tres = np.linspace (crm.min() , crm.max(), 7)  
    gs= GeoStrataModel ( to_log10 =True, max_depth = 300 )
    gs.fit(crm, tres = tres, layers =layers ).buildNM()
    gs.nm_.shape 
    gs.strataModel () 
    gs.strataModel (kind ='crm') 
    gs.strataModel (kind ='nm', misfit_G =True) 
    gs.plotStrata ('s02') 
    
    # test while files 
    np.random.seed (seed)
    crm = np.abs( np.random.randn (215 , 70 ) *1000 )
    tres = np.linspace (crm.min() +1  , crm.max() +1 , 12 )  
    layers = ['permafrost', 'gneiss', 'dolomite', 'clay', 'limestone', 'shale']
    gs= GeoStrataModel (to_log10 =True, max_depth = 120  )
    gs.fit(crm, tres = tres, layers =layers ).buildNM(display_infos =True )
    print( gs.nm_.shape) 
    gs.strataModel () 
    gs.plotStrata ('s33') 
    
    
# if __name__=='__main__': 
#     test_GeoStrataModel()
