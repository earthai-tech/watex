# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import os 
import pytest
import numpy as np 
import watex as wx 
from watex.methods.em import EM
import matplotlib.pyplot as plt 
from watex.methods.em import EMAP 
from watex.methods.em import MT 

EDIPATH ='data/edis'
savepath = '/Users/Daniel/Desktop/ediout'

@pytest.mark.skipif(os.path.isdir ('data/edis') is False ,
                    reason = 'EDI path does not exist')
def test_EM() : 

    emObjs = EM().fit(EDIPATH)
    # emObjs.rewrite(by='id', edi_prefix ='b1',savepath =savepath)
    # # 
    # second example to write 7 samples of edi from 
    # Edi objects inner datasets 
    #
    edi_sample = wx.fetch_data ('edis', key ='edi', samples =7, 
                                return_data =True ) 
    emobj = EM ().fit(edi_sample)
    print(emobj.ediObjs_ )
    # uncomment this to output EDI 
    # emobj.rewrite(by='station', prefix='PS')
    
    
    edi_sample = wx.fetch_data ('edis', return_data=True, samples = 12 )
    wx.EM().fit(edi_sample).getfullfrequency(to_log10 =True )
    
    
    emObjs= EM().fit(edi_sample)
    phyx = emObjs.make2d ('phaseyx')
    print( phyx.shape ) 
    
    # get the real number of the yy componet of tensor z 
    zyy_r = emObjs.make2d ('zyx', kind ='real')
    # ... array([[ 4165.6   ,  8665.64  ,  5285.47  ],
    # [ 7072.81  , 11663.1   ,  6900.33  ],
    # ...
    # [   90.7099,   119.505 ,   122.343 ],
    # [       nan,        nan,    88.0624]])
    # get the resistivity error of component 'xy'
    resxy_err = emObjs.make2d ('resxy_err')
    resxy_err 
    # ... array([[0.01329037, 0.02942557, 0.0176034 ],
    # [0.0335909 , 0.05238863, 0.03111475],
    # ...
    # [3.33359942, 4.14684926, 4.38562271],
    # [       nan,        nan, 4.35605603]])
    print( phyx.shape ,zyy_r.shape, resxy_err.shape  ) 
    # ... ((55, 3), (55, 3), (55, 3))
        
    ref = EM().fit(edi_sample).getreferencefrequency(to_log10=True) 
    print(ref )


@pytest.mark.skipif(os.path.isdir ('data/edis') is False ,
                    reason = 'EDI path does not exist')
def test_EMAP(): 

    # xxxxx Test filter xxxxxxxxxxxxxxxxxxxxxxxx
    edi_sample12 = wx.fetch_data ('edis', return_data=True, samples = 24 )
    p = EMAP().fit(edi_sample12) 
    
    p.window_size =2 
    p.component ='yx'
    rc= p.tma()
    rc2= p.flma()
    rc3 =p.ama() 
    # get the resistivy value of the third frequency  at all stations 
    print( p.res2d_[3, :]  ) 
    # ... array([ 447.05423001, 1016.54352954, 1415.90992189,  536.54293994,
    # 	   1307.84456036,   65.44806698,   86.66817791,  241.76592273,
    # 	   ...
    # 		248.29077039,  247.71452712,   17.03888414])
      # get the resistivity value corrected at the third frequency 
    print(rc [3, :]) 
    # ... array([ 447.05423001,  763.92416768,  929.33837349,  881.49992091,
    # 		404.93382163,  190.58264151,  160.71917654,  163.30034875,
    # 		394.2727092 ,  679.71542811,  953.2796567 , 1212.42883944,
    # 		...
    # 		164.58282866,   96.60082159,   17.03888414])
    plt.semilogy (np.arange (p.res2d_.shape[1] ), p.res2d_[7, :], '--', 
                  np.arange (p.res2d_.shape[1] ), rc[7, :], 'ok--', 
                  np.arange (p.res2d_.shape[1] ), rc2[7, :], '*r--',
                  np.arange (p.res2d_.shape[1] ), rc3[7, :], 'xg--',
                  )
    
    # xxxxx Test interpolation  xxxxxxxxxxxxxxxxxxxxxxxx
    p = EMAP().fit(EDIPATH) 
    sk,_ = p.skew()
    print( sk[:7, ]) 
     # ... array([0.45475527, 0.7876896 , 0.44986397])

    pObjs= EMAP().fit(EDIPATH)
    # One can specify the frequency buffer like the example below, However 
    # it is not necessaray at least there is a a specific reason to fix the frequencies 
    buffer = [1.45000e+04, 1.11500e+02]
    zobjs_b =  pObjs.zrestore(buffer = buffer) # with buffer 
    print(zobjs_b [0].resistivity[:, 1, 0 ][:3])
    pobj = EMAP().fit(EDIPATH)
    f = pobj.getfullfrequency ()
    buffer = [5.86000e+04, 1.6300e+02]
    print(f)  
    new_f = pobj.freqInterpolation(f, buffer = buffer)
    print( new_f) 
    
    
    freq_ = np.linspace(7e7, 1e0, 20) # 20 frequencies as reference
    buffer = EMAP.controlFrequencyBuffer(freq_, buffer =[5.70e7, 2e1])
    freq_ 
    # ... array([7.00000000e+07, 6.63157895e+07, 6.26315791e+07, 5.89473686e+07,
    #        5.52631581e+07, 5.15789476e+07, 4.78947372e+07, 4.42105267e+07,
    #        4.05263162e+07, 3.68421057e+07, 3.31578953e+07, 2.94736848e+07,
    #        2.57894743e+07, 2.21052638e+07, 1.84210534e+07, 1.47368429e+07,
    #        1.10526324e+07, 7.36842195e+06, 3.68421147e+06, 1.00000000e+00])
    print(buffer ) 
    # ... array([5.52631581e+07, 1.00000000e+00])
    
    # xxxxx Test QC  xxxxxxxxxxxxxxxxxxxxxxxx
    pobj = EMAP().fit(EDIPATH)
    f = pobj.getfullfrequency ()
    # len(f)
    # ... 55 # 55 frequencies 
    c,_ = pobj.qc ( tol = .4 ) # mean 60% to consider the data as
    # representatives 
    print( c ) # the representative rate in the whole EDI- collection
    # ... 0.95 # the whole data at all stations is safe to 95%. 
    # now check the interpolated frequency 
    c, freq_new  = pobj.qc ( tol=.6 , return_freq =True)
    print(c, freq_new)
    
    # xxxxx Test tensor validity  xxxxxxxxxxxxxxxxxxxxxxxx
    
    pObj = EMAP ().fit(EDIPATH)
    f= pObj.freqs_
    len(f) 
    # ... 55
    pObj.getValidTensors (tol= 0.3 ) # None doesn't export EDI-file
    len(pObj.new_Z_[0]._freq) # suppress 3 tensor data 
    # ... 52 
    pObj.getValidTensors(tol = 0.6 , 
                                     # option ='write'
                                     )
    len(pObj.new_Z_[0]._freq)  # suppress only two 
    
    
    # xxxxx Test z interpolation  xxxxxxxxxxxxxxxxxxxxxxxx
    sedis = wx.fetch_data ('huayuan', samples = 12 , return_data =True , key='raw')
    p = wx.EMAP ().fit(sedis) 
    ff = [ len(ediobj.Z._freq)  for ediobj in p.ediObjs_] 
    # [53, 52, 53, 55, 54, 55, 56, 51, 51, 53, 55, 53]
    Zcol = p.interpolate_z (sedis)
    ffi = [ len(z.freq) for z in Zcol ]
    print(ffi)
    # [56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56]
    # visualize seven Z values at the first site component xy 
    print( p.ediObjs_[0].Z.z[:, 0, 1][:7]) 
    # array([ 4165.6 +2070.13j,  7072.81+6892.41j,  8725.84+5874.15j,
    # 	   14771.8 -2831.28j, 21243.7 -6802.36j,  6381.48+3411.65j,
    # 		5927.85+5074.27j])
    print( Zcol [0].z[:, 0, 1 ][:7]) 
    
    # xxxxx Test frquency removal  xxxxxxxxxxxxxxxxxxxxxxxx
    
    sedis = wx.fetch_data ('huayuan', samples = 12 , 
    						   return_data =True , key='raw')
    p = EMAP ().fit(sedis) 
    ff = [ len(ediobj.Z._freq)  for ediobj in p.ediObjs_] 
    print(ff) 
    # [53, 52, 53, 55, 54, 55, 56, 51, 51, 53, 55, 53]
    p.ediObjs_[0].Z.z[:, 0, 1][:7]
    # array([ 4165.6 +2070.13j,  7072.81+6892.41j,  8725.84+5874.15j,
    # 	   14771.8 -2831.28j, 21243.7 -6802.36j,  6381.48+3411.65j,
    # 		5927.85+5074.27j])
    Zcol = p.drop_frequencies (tol =.2 )
    print( Zcol [0].z[:, 0, 1 ][:7]) 
    # array([ 4165.6 +2070.13j,  7072.81+6892.41j,  8725.84+5874.15j,
    # 	   14771.8 -2831.28j, 21243.7 -6802.36j,  6381.48+3411.65j,
    # 		5927.85+5074.27j])
    print( [ len(z.freq) for z in Zcol ]) 
    # [53, 52, 52, 53, 53, 53, 53, 50, 49, 53, 53, 52]
    p.verbose =True 
    Zcol = p.drop_frequencies (tol =.2 , interpolate= True )
    # Frequencies:     1- 81920.0    2- 48.5294    3- 5.625  Hz have been dropped.
    print( [ len(z.freq) for z in Zcol ] ) # all are interpolated to 53 frequencies
    # [53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53]
    Zcol = p.drop_frequencies (tol =.2 , interpolate= True , 
                               # export =True
                               )
    # drop a specific frequencies 
    # let visualize the 7 frequencies of our ediObjs 
    p.freqs_ [:7]
    # array([81920., 70000., 58800., 49500., 41600., 35000., 29400.])
    # let try to drop 49500 and 29400 frequencies explicitly. 
    Zcol = p.drop_frequencies (freqs = [49500 , 29400] )
    # let check whether this frequencies still available in the data 
    Zcol [5].freq[:7] 
    # array([81920., 70000., 58800., 41600., 35000., 24700., 20800.])
    # frequencies do not need to match exactly the value in frequeny 
    # range. Here is an example 
    Zcol = p.drop_frequencies (freqs = [49800 , 29700] )
    # Frequencies:     1- 49500.0    2- 29400.0  Hz have been dropped.
    # explicitly it drops the 49500 and 29400 Hz the closest. 

def test_MT(): 
    
    # xxxxx Test ZC  xxxxxxxxxxxxxxxxxxxxxxxx 
    edi_sample = wx.fetch_data ('edis', samples =17, return_data =True) 
    zo = MT ().fit(edi_sample) 
    print( zo.ediObjs_[0].Z.resistivity[:, 0, 1][:10]) # for xy components 
    # array([ 427.43690401,  524.87391142,  732.85475419, 1554.3189371 ,
    # 	   3078.87621649, 1550.62680093,  482.64709443,  605.3153687 ,
    # 		499.49191936,  468.88692879])
    zo.remove_static_shift(ss_fx =0.7 , ss_fy =0.85 )
    
    print( zo.ediObjs_[0].Z.resistivity[:, 0, 1][:10])  # corrected xy components 
    # array([ 278.96395263,  319.11187959,  366.43170231,  672.24446295,
    # 	   1344.20120487,  691.49270688,  260.25625996,  360.02452498,
    # 		305.97381587,  273.46251961])
    
    # xxxxx Test SSEMAP filter removal  xxxxxxxxxxxxxxxxxxxxxxxx
    zo = MT ().fit(edi_sample)
    print( zo.ediObjs_[0].Z.z[:, 0, 1][:7]) 
    # array([10002.46 +9747.34j , 11679.44 +8714.329j, 15896.45 +3186.737j,
    #        21763.01 -4539.405j, 28209.36 -8494.808j, 19538.68 -2400.844j,
    #         8908.448+5251.157j])
    zo.remove_ss_emap() 
    print( zo.ediObjs_[0].Z.z[:, 0, 1] [:7]) 
    
    # xxxxx Test static schift correction  xxxxxxxxxxxxxxxxxxxxxxxx
    zo = MT ().fit(edi_sample).remove_static_shift () 
    print( zo.ediObjs_[0].Z.z[:, 0, 1] [:7]) 
    # array([ 8028.46578676+7823.69394148j,  9374.49231974+6994.54856416j,
    #        12759.27171475+2557.831671j  , 17468.06097719-3643.54946031j,
    #        22642.21817697-6818.35022516j, 15682.70444455-1927.03534064j,
    #         7150.35801004+4214.83658174j])
    
    zo = MT ().fit(edi_sample)
    
    distortion = np.array([[1.2, .5],[.35, 2.1]])
    zo.remove_distortion (distortion)
    print( zo.ediObjs_[0].Z.z[:, 0, 1] [:7]) 
    # ([ 9724.52643923+9439.96503198j, 11159.25927505+8431.1101919j ,
    #         14785.52643923+3145.38324094j, 19864.708742  -4265.80166311j,
    #         25632.53518124-8304.88093817j, 17889.15373134-2484.60144989j,
    #          8413.19671642+4925.46660981j])


# if __name__=='__main__': 
    
#     test_MT() 
#     test_EMAP() 
#     test_EM()








