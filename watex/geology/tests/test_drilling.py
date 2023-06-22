# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import pytest 
import os 
import watex as wx 

from watex.geology import ( 
    DSBorehole, 
    DSBoreholes, 
    DSDrill
    )
# @pytest.mark.skip(reason="no need of this testing")
@pytest.mark.skipif(os.path.isdir ('data/drill') is False ,
                    reason = 'Drill data path does not exist')
def test_DSBoreholes (): 
    # read borehole data from Nasnhang 
    nsh_data = wx.fetch_data ('nlogs', key='hydrogeological', as_frame=True)
    bs = DSBoreholes().fit(nsh_data)
    print(bs.data_.shape)
    # interpolate coordiantes 
    bs2 = DSBoreholes(projection='dms', holeid='hole_id', 
                      verbose =1, interp_coords=True, 
                      lon='lon', lat='lat', 
                      ).fit( nsh_data)
    print(bs2.data_.columns)
    # print the first borehole object data 
    print(hasattr(bs2.data_, 'hole_id')) 
    ex_holes = bs2.data_.hole_id  [:1 ] 
    
    for hole in ex_holes : 
        print(getattr ( bs2.hole, hole))
    print(bs2.lat_ ) 
    print(bs2.lon_ )
    
    # one more test 
    bs_data = wx.fetch_data ('nlogs', key='hydro', samples=12 ,
                                 as_frame=True )
    bs=DSBoreholes ().fit(bs_data)
    bs.holeid
    # Out[61]: 'hole'
    # when the default object hole is set as:
    bs.hole # outputs a Boxspace object each borehole can be retrieved 
    # as hole object count from 0. to number or rows -1. Here is an 
    # example of fetching the hole 11. 
    print(bs.hole.hole10) 
    # Out[62]:
    # {'hole_id': 'B0103',
    #  'uniform_number': 1.1343e+16,
    #  'original_number': 'Guangzhou multi-element urban geological survey drilling 19ZXXSW11',
    #  'lon': '113:43:00.99',
    #  'lat': '23:16:17.23',
    #  'longitude': 113.71694166666668,
    #  'latitude': 23.271452777777775,
    #  'east': 2577207.0,
    #  'north': 19778060.0,
    #  'easting': 2577207.276,
    #  'northing': 19778177.29,
    #  'coordinate_system': 'Xian 90',
    #  'elevation': 22.0,
    #  'final_hole_depth': 60.1,
    #  'quaternary_thickness': 45.8,
    #  'aquifer_thickness': 18.1,
    #  'top_section_depth': 42.0,
    #  'bottom_section_depth': 60.1,
    #  'groundwater_type': 'igneous rock fissure water',
    #  'static_water_level': 2.36,
    #  'drawdown': 28.84,
    #  'water_inflow': 0.08,
    #  'unit_water_inflow': 0.003,
    #  'filter_pipe_diameter': 0.16,
    #  'water_inflow_in_m3_d': 2.94}
    # when we specified the hole ID to the column that compose the ID like: 
    bs=DSBoreholes (holeid ='hole_id').fit(bs_data)
    print(bs.hole.B0103) 
    # Out[63]:
    # {'hole_id': 'B0103',
    #  'uniform_number': 1.1343e+16,
    #  'original_number': 'Guangzhou multi-element urban geological survey drilling 19ZXXSW11',
    #  'lon': '113:43:00.99',
    #  'lat': '23:16:17.23',
    #  'longitude': 113.71694166666668,
    #  'latitude': 23.271452777777775,
    #  'east': 2577207.0,
    #  'north': 19778060.0,
    #  'easting': 2577207.276,
    #  'northing': 19778177.29,
    #  'coordinate_system': 'Xian 90',
    #  'elevation': 22.0,
    #  'final_hole_depth': 60.1,
    #  'quaternary_thickness': 45.8,
    #  'aquifer_thickness': 18.1,
    #  'top_section_depth': 42.0,
    #  'bottom_section_depth': 60.1,
    #  'groundwater_type': 'igneous rock fissure water',
    #  'static_water_level': 2.36,
    #  'drawdown': 28.84,
    #  'water_inflow': 0.08,
    #  'unit_water_inflow': 0.003,
    #  'filter_pipe_diameter': 0.16,
    #  'water_inflow_in_m3_d': 2.94}
    # each columns can be fetched as 
    print(bs.quaternary_thickness) 
    # Out[64]: 
    # 0     40.5
    # 1     12.3
    # 2     25.5
    # 3     40.0
    # 4     35.0
    # 5     47.0
    # 6     34.0
    # 7     40.4
    # 8     15.1
    # 9     17.2
    # 10    45.8
    # 11    47.0
    # Name: quaternary_thickness, dtype: float64
def test_DSBorehole (): 
    # read Boreholes data fetch from hlogs 
    hdata = wx.fetch_data ('hlogs', keys ='h2601').frame
    b= DSBorehole(hole= 'HLogs', dname ='depth_top', verbose =1).fit(hdata )
    print( b.data_.columns ) 
    print(b.feature_names_in_ )
    data = wx.make_erp ().frame 
    b= DSBorehole ().fit(data)
    b.set_strata () 
    b.data_.columns 
    b.set_thickness()
    b.set_depth(reset_depth= True )
    # one more test 
    b = DSBorehole (hole='H502').fit(hdata)
    b.feature_names_in_
    b.strata_name
    # Out[78]: 
    # 0                       topsoil
    # 1                        gravel
    # 2                      mudstone
    # 3                     siltstone
    # 4                      mudstone
              
    # 176                        coal
    # 177                   siltstone
    # 178    coarse-grained sandstone
    # 179      fine-grained sandstone
    # 180    coarse-grained sandstone
    # Name: strata_name, Length: 181, dtype: object
    b.set_depth () 
    print(b.depth_) 
    # Out[82]: 
    # 0        0.000000
    # 1        3.888889
    # 2        7.777778
    # 3       11.666667
    # 4       15.555556
       
    # 176    684.444444
    # 177    688.333333
    # 178    692.222222
    # 179    696.111111
    # 180    700.000000
    # Name: depth, Length: 181, dtype: float64
    b.set_depth (max_depth = 900, reset_depth= True )
    print(b.depth_) 
    # Out[85]: 
    # 0        0.0
    # 1        5.0
    # 2       10.0
    # 3       15.0
    # 4       20.0
     
    # 176    880.0
    # 177    885.0
    # 178    890.0
    # 179    895.0
    # 180    900.0
    # Name: depth, Length: 181, dtype: float64
    # generate random thickness 
    b.set_thickness () 
    b.layer_thickness_ 
    b.set_thickness (dirichlet_dist=True, reset_layer_thickness=True 
                         ).layer_thickness_
    # Out[89]: 
    # 0       0.681640
    # 1       1.986043
    # 2       6.413090
    # 3       5.305284
    # 4       0.000144
       
    # 176     4.119242
    # 177    12.161252
    # 178     1.809102
    # 179     0.408810
    # 180     4.281848
    # Name: layer_thickness, Length: 181, dtype: float64
    hdata= wx.fetch_data ('hlogs', key='h803').frame
    b = DSBorehole (hole='H803').fit(hdata)
    b.set_strata () 
    b.strata_
    b.set_strata (add_electrical_properties= True, reset_strata= True)
    print(b.strata_) 
    # Out[123]: 
    # 0              phyllite
    # 1               syenite
    # 2              laterite
    # 3             saprolite
    # 4          psammopelite
          
    # 129               chert
    # 130           granulite
    # 131    pyroclastic rock
    # 132         lamprophyre
    # 133          ignimbrite
    # Name: strata, Length: 134, dtype: object
    print(b.strata_electrical_properties_)
    # Out[124]: 
    # 0        0.0
    # 1        0.0
    # 2        0.0
    # 3      330.6
    # 4        0.0
     
    # 129      0.0
    # 130      0.0
    # 131      0.0
    # 132      0.0
    # 133      0.0
    # Name: strata_electrical_properties, Length: 134, dtype: float64
    
    # test nlogs 
    ndata = wx.fetch_data ('nlogs', as_frame =True, key='ns', samples =12 ) 
    bo = DSBorehole ().fit(ndata) 
    # compute layer thickness 
    bo.set_thickness ()
    print(bo.depth_) 
    bo.set_strata(add_electrical_properties= True , random_state=42 ) 
    print(bo.strata_)
    print(bo.strata_electrical_properties_)
    
@pytest.mark.skipif(os.path.isdir ('data/drill') is False ,
                    reason = 'Drill data path does not exist')
def test_DSDrill(): 
    # read drill data 
    
    dr = DSDrill().fit('data/drill/nbleDH.xlsx')
    dr.geology_ 
    dr.collar_ 
    dr.samples_ 
    # read on csv 
    dr2 = DSDrill().fit('data/drill/nbleDH.csv') 
    print(dr2.data_)
    # read data 
    dr3=DSDrill().fit(data = dr.collar_ )
    print(dr3.collar_)
    
    dr = DSDrill(holeid="DH_Hole (ID)") .fit('data/drill/nbleDH.xlsx')
    dr2 = dr.get_collar (dr.collar_, reset_collar= True )
    dr2.collar_
    #   DH_Hole (ID)      DH_East     DH_North  ...  DH_PlanDepth  DH_Decr  Mask 
    # 0          S01  477205.6935  2830978.218  ...           NaN      NaN    NaN
    # 1          S02  477261.7258  2830944.879  ...           NaN      NaN    NaN
    
    dr.holeid # id hole is autodetected if not given
    # 'DH_Hole (ID)'
    # >>> # retreive the holeID S01 
    print(dr.collar.S01) 
    # {'DH_Hole (ID)': 'S01',
    #  'DH_East': 477205.6935,
    #  'DH_North': 2830978.218,
    #  'DH_Dip': -90.0,
    #  'Elevation ': 0.0,
    #  'DH_Azimuth': 0.0,
    #  'DH_Top': 0.0,
    #  'DH_Bottom': 968.83,
    #  'DH_PlanDepth': nan,
    #  'DH_Decr': nan,
    #  'Mask ': nan}
    dr = DSDrill(holeid="DH_Hole").fit('data/drill/nbleDH.xlsx')
    dr.get_geology (dr.geology_, reset_geology=True ).geology_
    #   DH_Hole     Thick01  ...                    Rock03  Rock04
    # 0     S01    0.200000  ...  carbonate iron formation    ROCK
    # 1     S02  174.429396  ...                       GRT    ROCK
    print(dr.holeid) # id hole is autodetected if not given
    # Out[62]: 'DH_Hole'
    # >>> # retreive the hole ID S01 
    print( dr.geology.S01) 
    # {'DH_Hole': 'S01',
    #  'Thick01': 0.2,
    #  'Thick02': 98.62776918,
    #  'Thick03': 204.7500461,
    #  'Thick04': 420.0266651,
    #  'Rock01': 'clast supported breccia',
    #  'Rock02': 'sulphide-rich material',
    #  'Rock03': 'carbonate iron formation',
    #  'Rock04': 'ROCK'}
    dr.get_geosamples (dr.samples_, reset_samples= True ).samples_
    #   DH_Hole  Thick01     Thick02  ...             sample02  sample03     sample04
    # 0     S01     10.0   98.627769  ...                  prt       pup  Boudin Axis
    # 1     S02     17.4  313.904388  ...  Banding/gneissosity       pup          pzs
    dr.holeid # id hole is autodetected if not given
    # 'DH_Hole'
    # >>> # retreive the holeID geosamples S02 
    print( dr.samples.S02) 
    # {'DH_Hole': 'S02',
    #  'Thick01': 17.4,
    #  'Thick02': 313.9043882,
    #  'Thick03': 400.12,
    #  'Thick04': 515.3,
    #  'sample01': 'pup',
    #  'sample02': 'Banding/gneissosity',
    #  'sample03': 'pup',
    #  'sample04': 'pzs'}

# if __name__=='__main__': 
#     test_DSBoreholes()
#     test_DSBorehole()
#     test_DSDrill()