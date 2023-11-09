# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:01:53 2023

@author: Daniel
"""
import copy 
import os 
from watex.utils import read_data 
from watex.methods import EM   
from watex.site import Profile, Location 
from watex.methods import MTProcess   
from watex.utils import plot_tensors    
from watex.utils.plotutils import plot_strike  
#%%
def get_ll ( data,  /, coordinate_system ='utm', epsg =15921, utm_zone ='49N'): 
    """ Get the coordinates """
    data = read_data(data, sanitize =True )
    
    p = Profile ( utm_zone = utm_zone, coordinate_system= coordinate_system, 
                 epsg = epsg ).fit(x='longitude', y='latitude', elev ='elev', 
                                   data = data )
                                   
    lat, lon = Location.to_latlon_in(p.y, p.x , utm_zone=utm_zone, 
                                     epsg =epsg, datum ='WGS84')
    return lon, lat , p.elev 

def set_ll_and_export_edis ( 
        edipath, /, coord_file,   savepath , dataid =None, **kws) : 
    emo = EM().fit(edipath )
    # set longitude and latitude from Profile 
    lon, lat, elev = get_ll (coord_file , **kws) 
    # read EM obj and reset attribute 
    emo.longitude = lon 
    emo.latitude =lat 
    emo.elevation = elev 
    emo.rewrite (by ='id', dataid = dataid, savepath =savepath, edi_prefix=''  ) 
    
    return emo 

#%%
    # 
edipath =r'D:\project-Tayuan\data\2\2HX'
coord_file=r'D:\project-Tayuan\data\2\2.csv' 
savepath =r'D:\project-Tayuan\data\2\2EDI' 
# set EMobj so use it to set rewrite the dataID 
emo = EM().fit(edipath )
set_ll_and_export_edis ( 
    emo.ediObjs_, 
    coord_file=coord_file, 
    savepath =savepath, 
    dataid =['S{:02}'.format(ix) for ix in range ( len(emo.ediObjs_))] #[::-1]
    )

#%% 
# Process data and out data 
new_edipath =savepath 
outpath =r'D:\project-Tayuan\data\2\2EDIP' # path to save new process EDI
em0 = EM().fit(edipath )

emc = copy.deepcopy(em0) # make a copy to be safe.

zc = MTProcess(verbose =True ).fit( emc.ediObjs_ )
zc.remove_static_shift (nfreq=7 , r = 100).remove_noises (
    method='base').drop_frequencies (tol = .1 ).out(savepath =outpath) 

#%% 
# Plot EDI 
# plot row 
plot_tensors ( em0.ediObjs_, station =0 ) 
#%% 
# plot new .
plot_tensors (emc.ediObjs_, station = 0 )
#%% 
# plot_strike raw 
edipath = new_edipath # outpath 
edi_fn_lst = [os.path.join(edipath,f) for f in os.listdir(edipath) 
        if f.endswith('.edi')] 

plot_strike(edi_fn_lst ) 
plot_strike(edi_fn_lst , kind = 1 ) 

#%% 
# rename EDI 

from watex.utils.funcutils import rename_files 

src_path =outpath 
dst_path =r'D:\project-Tayuan\data\2\renamedEDIs'

rename_files(src_path , dst_files= dst_path , basename ='T2.', trailer='')
