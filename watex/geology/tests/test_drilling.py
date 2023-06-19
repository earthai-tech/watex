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
    nsh_data = wx.read_data ( 'data/drill/nhlogs.csv')
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

if __name__=='__main__': 
    
    test_DSBoreholes()
    test_DSBorehole()
    test_DSDrill()