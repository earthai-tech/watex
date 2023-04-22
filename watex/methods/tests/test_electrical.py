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
     # read the both objects
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

@pytest.mark.skipif(os.path.isdir ('data/erp') is False ,
                    reason = 'DC data path does not exist')
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
    # ... array(['S017', 'S006', 'S000', 'S036', 'S036', 'S036', 'S036', 'S036',
    #        'S001'], dtype='<U33')
    
    # (3) -> Read data and all sheets, assumes all data are arranged in a sheets
    #dcobjs = DCProfiling()
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
    print( vobj.summary(keep_params= True , return_table= True )) 
    vobj.plotOhmicArea()
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
    
    vObj.plotOhmicArea(fbtw=True )
    
def test_DCMagic (): 
    # test 
    erp_data = wx.make_erp ( seed =33 ).frame  
    ves_data = wx.make_ves (seed =42).frame 
    v = wx.DCSounding ().fit(wx.make_ves (seed =10, as_frame =True, add_xy =True))
    r = wx.DCProfiling().fit( wx.make_erp ( seed =77 , as_frame =True))
    res= wx.methods.ResistivityProfiling(station='S4').fit(erp_data) 
    ves= wx.methods.VerticalSounding(search=60).fit(ves_data)
    # dc-ves  : 100%|################################| 1/1 [00:00<00:00, 111.13B/s]
    # dc-erp  : 100%|################################| 1/1 [00:00<00:00, 196.77B/s]
    m = DCMagic().fit(erp_data, ves_data, v, r, ves, res ) 
    # dc-erp  : 100%|################################| 2/2 [00:00<00:00, 307.40B/s]
    # dc-o:erp: 100%|################################| 1/1 [00:00<00:00, 499.74B/s]
    # dc-ves  : 100%|################################| 2/2 [00:00<00:00, 222.16B/s]
    # dc-o:ves: 100%|################################| 1/1 [00:00<00:00, 997.46B/s]
    m.summary(keep_params =True)
    #     longitude  latitude shape  ...       sfi  sves_resistivity  ohmic_area
    # 0         NaN       NaN     W  ...  1.310417        707.609756  263.213572
    # 1         NaN       NaN     K  ...  1.300024          1.000000  964.034554
    # 2  109.332932  28.41193     U  ...  1.184614          1.000000  276.340744

    data = wx.make_erp (seed =42 , n_stations =12, as_frame =True ) 
    ro= wx.DCProfiling ().fit(data) 
    print(ro.summary()) 

    data_no_xy = wx.make_ves ( seed=0 , as_frame =True) 
    vo = wx.methods.VerticalSounding (
        xycoords = (110.486111,   26.05174)).fit(data_no_xy).summary()
    print(vo.table_) 
    dm = wx.methods.DCMagic ().fit(vo, ro ) 
    print(dm.summary (like = ...)) 
    #    dipole  longitude  latitude  ...  max_depth  ohmic_area  nareas
    # 0      10  110.48611  26.05174  ...      109.0  690.063003       1
    print(dm.summary (keep_params =True, like = ... )) 
    #    longitude  latitude shape  ...       sfi  sves_resistivity  ohmic_area
    # 0  110.48611  26.05174     C  ...  1.141844               1.0  690.063003
    print(list( dm.table_.columns )) 
    
@pytest.mark.skipif(os.path.isdir ('data/ves'
                                   ) is False  or os.path.isdir (
                                       'data/erp') is False  ,
                    reason = 'DC data path does not exist')    
def test_DCMagic_ (): 
    # test Magic method when aggreate multiple files from Path-like objects
    m0 = DCMagic(verbose =True ).fit('data/erp', 'data/ves')
    # dc-ves  : 100%|################################| 1/1 [00:00<00:00, 111.11B/s]
    # dc-erp  : 100%|################################| 1/1 [00:00<00:00, 166.67B/s]
    # dc-erp  : 100%|################################| 2/2 [00:00<00:00, 231.29B/s]
    # dc-o:erp: 100%|################################| 1/1 [00:00<00:00, 498.79B/s]
    # dc-ves  : 100%|################################| 2/2 [00:00<00:00, 203.20B/s]
    # dc-o:ves: 100%|################################| 1/1 [00:00<00:00, 500.04B/s]
    # dc-erp  : 100%|################################| 9/9 [00:00<00:00, 219.63B/s]
    # dc-ves  : 100%|################################| 3/3 [00:00<00:00, 163.37B/s]
    m0.summary(keep_params =True, force =True )
    #     longitude  latitude shape  ...       sfi  sves_resistivity   ohmic_area
    # 0         0.0       0.0     C  ...  1.050857              80.0          NaN
    # 1         0.0       0.0     V  ...  0.076391              50.0          NaN
    # 2         0.0       0.0     C  ...  0.035928            1101.0          NaN
    # 3         0.0       0.0     V  ...  0.076391             500.0          NaN
    # 4         0.0       0.0     V  ...  0.076391             500.0          NaN
    # 5         0.0       0.0     V  ...  0.076391             500.0          NaN
    # 6         0.0       0.0     V  ...  0.076391             500.0          NaN
    # 7         0.0       0.0     V  ...  0.076391             500.0          NaN
    # 8         0.0       0.0     V  ...  1.086559              93.0          NaN
    # 9         NaN       NaN   NaN  ...       NaN               NaN   268.087715
    # 10        NaN       NaN   NaN  ...       NaN               NaN   268.087715
    # 11        NaN       NaN   NaN  ...       NaN               NaN  1183.364102
    tab= m0.summary(keep_params =True, coerce =True )

     #    longitude  latitude shape  ...       sfi  sves_resistivity   ohmic_area
     # 0        0.0       0.0     C  ...  1.050857                80   268.087715
     # 1        0.0       0.0     V  ...  0.076391                50   268.087715
     # 2        0.0       0.0     C  ...  0.035928              1101  1183.364102
    
     # [3 rows x 9 columns]
    print(tab)
    
# if __name__=='__main__': 
    
#     test_DCProfiling()
#     test_VerticalSounding() 
#     test_DCSounding() 
#     test_ResistivityProfiling() 
#     test_DCMagic() 
#     test_DCMagic_ () 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    