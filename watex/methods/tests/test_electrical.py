# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import pytest 
import os 
from watex.methods.electrical import ( 
    ResistivityProfiling, 
    DCProfiling  , VerticalSounding , 
    DCSounding
    )

from watex.utils import vesSelector 
ERPPATH= r'data/erp'
# @pytest.mark.skip(reason="no need of this testing")
@pytest.mark.skipif(os.path.isdir ('data/erp') is False ,
                    reason = 'DC data path does not exist')
def test_ResistivityProfiling(): 
    #XXX TEST ELECTRICAL 
    robj1= ResistivityProfiling(auto=True, force =True ) # auto detection 
    robj1.utm_zone = '50N'
    robj1.fit('data/erp/testsafedata.xlsx') 
    
    robj1.utm_zone = '50N'
    robj1.fit('data/erp/testsafedata.xlsx') 
    robj1.sves_
    # ... 'S036'
    robj2= ResistivityProfiling(auto=True, utm_zone='40S', force =True ) 
    robj2.fit('data/erp/l11_gbalo.xlsx') 
    robj2.sves_ 
    # ... 'S006'
     # read the both objects
     
    return robj1, robj2

@pytest.mark.skipif(os.path.isdir ('data/erp') is False ,
                    reason = 'DC data path does not exist')
def test_DCProfiling(): 
    robj1= ResistivityProfiling(auto=True, force =True ) # auto detection 
    robj1.fit('data/erp/testsafedata.xlsx') 

    robj2= ResistivityProfiling(stations='S03', utm_zone='40S',
                                force =True ) 
    robj2.fit('data/erp/l11_gbalo.xlsx') 
    robj1, robj2= test_ResistivityProfiling() 
    dcobjs = DCProfiling()
    dcobjs.fit(robj1, robj2 ) 
    dcobjs.sves_ 
    # ... array(['S036', 'S006'], dtype=object)
    dcobjs.line1.sves_ # => robj1.sves_
    dcobjs.line2.sves_ # => robj2.sves_ 
    
    # (2) -> Read from a collection of excell data 
    
    
    dcobjs.read_sheets=True 
    dcobjs.fit(ERPPATH) 
    dcobjs.nlines_  # getting the number of survey lines 
    # ... 9
    dcobjs.sves_ # stations of the best conductive zone 
    # ... array(['S017', 'S006', 'S000', 'S036', 'S036', 'S036', 'S036', 'S036',
    #        'S001'], dtype='<U33')
    dcobjs.sves_resistivities_ # the lower conductive resistivities 
    # ... array([  80,   50, 1101,  500,  500,  500,  500,  500,   93], dtype=int64)
    dcobjs.powers_ 
    # ... array([ 50,  60,  30,  60,  60, 180, 180, 180,  40])
    dcobjs.sves_ # stations of the best conductive zone 
    # ... array(['S017', 'S006', 'S000', 'S036', 'S036', 'S036', 'S036', 'S036',
    #        'S001'], dtype='<U33')
    
    # (3) -> Read data and all sheets, assumes all data are arranged in a sheets
    
    dcobjs.read_sheets=True
    dcobjs.fit(ERPPATH) 
    dcobjs.nlines_ # here it assumes all the data are in single worksheets.
    # ... 4 
    dcobjs.line4.conductive_zone_ # conductive zone of the line 4 
    # ... array([1460, 1450,  950,  500, 1300, 1630, 1400], dtype=int64)
    dcobjs.sfis_
    # array([1.05085691, 0.07639077, 0.03592814, 0.07639077, 0.07639077,
    #        0.07639077, 0.07639077, 0.07639077, 1.08655919])
    dcobjs.line3.sfi_ # => robj1.sfi_
    # ... array([0.03592814]) # for line 3 

@pytest.mark.skipif(os.path.isdir ('data/ves') is False ,
                    reason = 'DC data path does not exist')
def test_DCSounding () : 
    #(1) -> read a single DC Electrical Sounding file 
    
    
    dsobj = DCSounding ()  
    dsobj.search = 30. # start detecting the fracture zone from 30m depth.
    dsobj.fit('data/ves/ves_gbalo.xlsx')
    dsobj.ohmic_areas_
    # ...  array([523.25458506])
    dsobj.site1.fractured_zone_ # show the positions of the fracture zone 
    # ... array([ 28.,  32.,  36.,  40.,  45.,  50.,  55.,  60.,  70.,  80.,  90.,
    # 	   100.])
    dsobj.site1.fractured_zone_resistivity_
    # ... array([ 68.74273843,  71.57116555,  74.39959268,  77.2280198 ,
    # 			80.76355371,  84.29908761,  87.83462152,  91.37015543,
    # 			98.44122324, 105.51229105, 112.58335886, 119.65442667])
    
    # (2) -> read multiple sounding files 
    dsobj2 = DCSounding ()  
    dsobj2.fit('data/ves')
    dsobj2.ohmic_areas_  
    # ... array([ 523.25458506,  523.25458506, 1207.41759558]) 
    dsobj2.nareas_ 
    # ... array([2., 2., 3.]) 
    dsobj2.survey_names_
    # ... ['ves_gbalo', 'ves_gbalo', 'ves_gbalo_unique']
    dsobj2.nsites_ 
    # ... 3 
    dsobj2.site1.ohmic_area_
    # ... 523.2545850558677  # => dsobj.ohmic_areas_ -> line 1:'ves_gbalo'
        
        
    dsobj3 = DCSounding (search =10 )  
    dsobj3.fit('data/ves/ves_gbalo.xlsx')
    dsobj3.ohmic_areas_
    # ...  array([523.25458506])
    dsobj3.site1.fractured_zone_ # show the positions of the fracture zone 
    # ... array([ 28.,  32.,  36.,  40.,  45.,  50.,  55.,  60.,  70.,  80.,  90.,
    # 	   100.])
    dsobj3.site1.fractured_zone_resistivity_
    # ... array([ 68.74273843,  71.57116555,  74.39959268,  77.2280198 ,
    # 			80.76355371,  84.29908761,  87.83462152,  91.37015543,
    # 			98.44122324, 105.51229105, 112.58335886, 119.65442667])
    
    
@pytest.mark.skipif(os.path.isdir ('data/ves') is False ,
                    reason = 'DC data path does not exist')
def test_VerticalSounding(): 
    vobj = VerticalSounding(search= 45, vesorder= 3)
    vobj.fit('data/ves/ves_gbalo.xlsx')
    vobj.ohmic_area_ # in ohm.m^2
    # ... 349.6432550517697
    vobj.nareas_ # number of areas computed 
    # ... 2
    vobj.area1_, vobj.area2_ # value of each area in ohm.m^2 
    # ... (254.28891096053943, 95.35434409123027) 
    vobj.roots_ # different boundaries in pairs 
    # ... [array([45.        , 57.55255255]), array([ 96.91691692, 100.        ])]
    data = vesSelector ('data/ves/ves_gbalo.csv', index_rhoa=3)
    vObj = VerticalSounding().fit(data)
    vObj.fractured_zone_ # AB/2 position from 45 to 100 m depth.
    # ... array([ 45.,  50.,  55.,  60.,  70.,  80.,  90., 100.])
    vObj.fractured_zone_resistivity_
    # ...array([57.67588974, 61.21142365, 64.74695755, 68.28249146, 75.35355927,
    # 	   82.42462708, 89.4956949 , 96.56676271])
    vObj.nareas_ 
    # ... 2
    vObj.ohmic_area_
    # ... 349.6432550517697
    
if __name__=='__main__': 
    
    test_DCProfiling
    test_VerticalSounding() 
    test_DCSounding() 
    test_ResistivityProfiling() 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    