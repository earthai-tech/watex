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
from watex.utils.funcutils import rename_files 
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
EDIPATH = r'D:\project-Tayuan\data\2'
edipath =os.path.join( EDIPATH, '2HX')
coord_file=os.path.join( EDIPATH, '2.csv' )
savepath =os.path.join( EDIPATH, '2EDI') 
# set EMobj so use it to set rewrite the dataID 
# emo = EM().fit(edipath )
# set_ll_and_export_edis ( 
#     emo.ediObjs_, 
#     coord_file=coord_file, 
#     savepath =savepath, 
#     dataid =['S{:02}'.format(ix) for ix in range ( len(emo.ediObjs_))][::-1]
#     )

#%% 
# Process data and out data 
new_edipath =savepath 
outpath =os.path.join( EDIPATH, '2EDIP') # path to save new process EDI
em0 = EM().fit(new_edipath )

emc = copy.deepcopy(em0) # make a copy to be safe.

zc = MTProcess(verbose =True ).fit( emc.ediObjs_ )
zc.remove_static_shift (nfreq=21 , r = 1000).remove_noises (
     method='base').drop_frequencies (tol = .5 ).out(savepath =outpath) 
     #method='base').out(savepath =outpath) 

#%% 
# Plot EDI 
# plot row 
plot_tensors ( em0.ediObjs_, station =0 ) 
#%% 
# plot new .
plot_tensors (emc.ediObjs_, station = 0 )
#%% 
# plot_strike raw 
edipath =  outpath #new_edipath #
edi_fn_lst = [os.path.join(edipath,f) for f in os.listdir(edipath) 
        if f.endswith('.edi')] 

plot_strike(edi_fn_lst ) 
#plot_strike(edi_fn_lst , kind = 1 ) 

#%% 
# rename EDI 
src_path =outpath 
dst_path =os.path.join( EDIPATH, 'renamedEDIs')

rename_files(src_path , dst_files= dst_path , basename ='T3.', trailer='')
