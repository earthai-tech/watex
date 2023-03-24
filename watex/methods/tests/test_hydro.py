# -*- coding: utf-8 -*-

from watex.datasets import load_hlogs 
from watex.methods.hydro import MXS  
from watex.methods.hydro import Logging 
from watex.methods.hydro import AqGroup 

hdata= load_hlogs ().frame  

def test_MXS():
    # drop the 'remark' columns since there is no valid data 
    hdata.drop (columns ='remark', inplace =True)
    mxs = MXS (kname ='k').fit(hdata)
    # predict the default NGA 
    mxs.predictNGA() # default prediction with n_groups =3 
    # make MXS labels using the default 'k' categorization 
    ymxs=mxs.makeyMXS(categorize_k=True, default_func=True)
    mxs.yNGA_ [62:74] 
    # Out[43]: array([1, 2, 2, 2, 3, 1, 2, 1, 2, 2, 1, 2])
    ymxs[62:74] 
    # Out[44]: array([ 1, 22, 22, 22,  3,  1, 22,  1, 22, 22,  1, 22]) 
    # to get the label similariry , need to provide the 
    # the column name of aquifer group and fit again like 
    mxs = MXS (kname ='k', aqname ='aquifer_group').fit(hdata)
    sim = mxs.labelSimilarity() 
    sim 
    # Out[47]: [(0, 'II')] # group II and label 0 are very similar 


def test_Logging():
    # get the logging data 
    h = load_hlogs ()
    h.feature_names
    # Out[29]: 
    # ['hole_id',
    #  'depth_top',
    #  'depth_bottom',
    #  'strata_name',
    #  'rock_name',
    #  'layer_thickness',
    #  'resistivity',
    #  'gamma_gamma',
    #  'natural_gamma',
    #  'sp',
    #  'short_distance_gamma',
    #  'well_diameter']
    # we can fit to collect the valid logging data
    log= Logging(kname ='k', zname='depth_top' ).fit(h.frame[h.feature_names])
    log.feature_names_in_ # categorical features should be discarded.
    # Out[33]: 
    # ['depth_top',
    #  'depth_bottom',
    #  'layer_thickness',
    #  'resistivity',
    #  'gamma_gamma',
    #  'natural_gamma',
    #  'sp',
    #  'short_distance_gamma',
    #  'well_diameter']
    log.plot ()
    # Out[34]: Logging(zname= depth_top, kname= k, verbose= 0)
    # plot log including the target y 
    log.plot (y = h.frame.k , posiy =0 )# first position 
    # Logging(zname= depth_top, kname= k, verbose= 0)


def test_AqGroup (): 
    
    hg = AqGroup (kname ='k', aqname='aquifer_group').fit(hdata ) 
    hg.findGroups () 


def test_makeyMXS(): 
    hdata = load_hlogs ().frame 
    # drop the 'remark' columns since there is no valid data 
    hdata.drop (columns ='remark', inplace=True) 
    mxs =MXS (kname ='k').fit(hdata) # specify the 'k'columns 
    # we can predict the NGA labels and yMXS with single line 
    # of code snippet using the default 'k' classification.
    ymxs = mxs.predictNGA().makeyMXS(categorize_k=True, default_func=True)
    # mxs.yNGA_[:7] 
    # ... array([2, 2, 2, 2, 2, 2, 2])
    ymxs[:7]
    # Out[40]: array([22, 22, 22, 22, 22, 22, 22])
    mxs.mxs_group_classes_
    # Out[56]: {1: 1, 2: 22, 3: 3} # transform classes 
    mxs.mxs_group_labels_ 
    # Out[57]: (2,)
    # **comment: 
    	# # only the label '2' is tranformed to '22' since 
    	# it is the only one that has similariry with the true label 2 

def test_predictNGA():
    hdata = load_hlogs ().frame 
    # drop the 'remark' columns since there is no valid data 
    hdata.drop (columns ='remark', inplace=True) 
    mxs =MXS (kname ='k').fit(hdata) # specify the 'k' column  
    y_pred = mxs.predictNGA(return_label=True )
    y_pred [-12:] 
    # Out[52]: array([1, 3, 1, 3, 3, 3, 3, 1, 3, 3, 3, 3])
    
if __name__=='__main__':    
    
    test_Logging() 
    # test_MXS ()
    
    # test_AqGroup()
    # test_makeyMXS()
    # test_makeyMXS()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    