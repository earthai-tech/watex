# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import pytest 
import os 
import watex as wx 
from watex.methods.electrical import ( 
    ResistivityProfiling, 
    DCProfiling  , VerticalSounding , 
    DCSounding
    )
from watex.methods.erp import DCMagic
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
     # test with constraints 
     
    constraints_0 =['site2', 'station2', 'site', 'site10']
    robjc1= ResistivityProfiling(constraints= constraints_0 , 
                                auto=True) 
    data1 = wx.make_erp (n_stations =9  ).frame 
    data2 = wx.make_erp (n_stations = 52 , min_rhoa= 1e1 , max_rhoa =1e3, 
                         ).frame 
    robjc1.fit(data1)
    print(robjc1.sves_ )
     
    robjc1.plotAnomaly()
    constraints_1 ={'s0': 'building close to the drilling site ',
                    "s25": 'heritage site, drilling prohibited',
                    "s12": 'polluted area',
                    " s35": 'RAS, in masrh area.'
                    }
    robjc1.constraints = constraints_1 
    robjc1.utm_zone ='49R'
    robjc1.coerce=True 
    robjc1.fit(data2)
    print(robjc1.sves_ )
    robjc1.plotAnomaly()
    
    return robj1, robj2

# @pytest.mark.skipif(os.path.isdir ('data/erp') is False ,
#                     reason = 'DC data path does not exist')
@pytest.mark.skip ("Configure excel sheet seems weird.. ")
def test_DCProfiling(): 
    robj1= ResistivityProfiling(auto=True, force =True ) # auto detection 
    robj1.fit('data/erp/testsafedata.xlsx') 

    robj2= ResistivityProfiling(station='S03', utm_zone='40S',
                                force =True ) 
    robj2.fit('data/erp/l11_gbalo.xlsx') 
    robj1, robj2= test_ResistivityProfiling() 
    dcobjs = DCProfiling(force =True )
    dcobjs.fit(robj1, robj2 ) 
    dcobjs.sves_ 
    # ... array(['S036', 'S006'], dtype=object)
    dcobjs.line1.sves_ # => robj1.sves_
    dcobjs.line2.sves_ # => robj2.sves_ 
    
    # (2) -> Read from a collection of excell data 
    #dcobjs = DCProfiling()
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
    # (3) -> Read data and all sheets, assumes all data are arranged in a sheets
    #dcobjs = DCProfiling()
    dcobjs.read_sheets=True
    dcobjs.fit(ERPPATH) 
    dcobjs.nlines_ # here it assumes all the data are in single worksheets.
    # ... 4 
    dcobjs.line4.conductive_zone_ # conductive zone of the line 4 
    dcobjs.sfis_
    dcobjs.line3.sfi_ # => robj1.sfi_

@pytest.mark.skipif(os.path.isdir ('data/ves') is False ,
                    reason = 'DC data path does not exist')
def test_DCSounding () : 
    #(1) -> read a single DC Electrical Sounding file 

    dsobj = DCSounding ()  
    dsobj.search = 30. # start detecting the fracture zone from 30m depth.
    dsobj.fit('data/ves/ves_gbalo.xlsx')
    dsobj.ohmic_areas_
    dsobj.site1.fractured_zone_ # show the positions of the fracture zone 
    dsobj.site1.fractured_zone_resistivity_
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

    dsobj3 = DCSounding (search =10 )  
    dsobj3.fit('data/ves/ves_gbalo.xlsx')
    dsobj3.ohmic_areas_
    dsobj3.site1.fractured_zone_ # show the positions of the fracture zone 
    dsobj3.site1.fractured_zone_resistivity_

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
    vobj.roots_ # different boundaries in pairs 
    print( vobj.summary(keep_params= True , return_table= True )) 
    vobj.plotOhmicArea()
    data = vesSelector ('data/ves/ves_gbalo.csv', index_rhoa=3)
    vObj = VerticalSounding().fit(data)
    vObj.fractured_zone_ # AB/2 position from 45 to 100 m depth.
    vObj.fractured_zone_resistivity_
    vObj.nareas_ 
    # ... 2
    vObj.ohmic_area_
    
    vObj.plotOhmicArea(fbtw=True )
    
def test_DCMagic (): 
    # test 
    erp_data = wx.make_erp ( seed =33 ).frame  
    ves_data = wx.make_ves (seed =42).frame 
    v = DCSounding ().fit(wx.make_ves (seed =10, as_frame =True, add_xy =True))
    r = DCProfiling().fit( wx.make_erp ( seed =77 , as_frame =True))
    res= ResistivityProfiling(station='S4').fit(erp_data) 
    ves= VerticalSounding(search=60).fit(ves_data)
    m = DCMagic().fit(erp_data, ves_data, v, r, ves, res ) 

    m.summary(keep_params =True)

    data = wx.make_erp (seed =42 , n_stations =12, as_frame =True ) 
    ro= DCProfiling ().fit(data) 
    print(ro.summary()) 

    data_no_xy = wx.make_ves ( seed=0 , as_frame =True) 
    vo = VerticalSounding (
        xycoords = (110.486111,   26.05174)).fit(data_no_xy).summary()
    print(vo.table_) 
    dm = DCMagic ().fit(vo, ro ) 
    print(dm.summary (like = ...)) 
    print(dm.summary (keep_params =True, like = ... )) 
    print(list( dm.table_.columns )) 
    
@pytest.mark.skipif(os.path.isdir ('data/ves') is False  or os.path.isdir ('data/erp') is False,
                    reason = 'DC data path does not exist')    
def test_DCMagic_ (): 
    # test Magic method when aggreate multiple files from Path-like objects
    m0 = DCMagic(verbose =True ).fit('data/erp', 'data/ves')

    m0.summary(keep_params =True, force =True )

    tab= m0.summary(keep_params =True, coerce =True )
    print(tab)
    
    m1 = DCMagic(read_sheets=True , verbose =True).fit('data/erp', 'data/ves')
    tab1= m1.summary(keep_params =True, coerce =True )
    
    print(tab1)
    
if __name__=='__main__': 
    pytest.main([__file__])

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
