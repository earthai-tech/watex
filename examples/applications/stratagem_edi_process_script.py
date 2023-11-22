"""
====================================================
Fast AMT data processing from Stratagem hardware
====================================================

Short explanation about the process of AMT
data collected in a specific area with a Stratagem 
hardware.
"""

# License: BSD-3-Clause 
# Author: Kouadio Laurent 

#%% 
# * Context 
# This is a scheme for fast processing EDI data collected from 
# Stratagem hardware where data are not included the coordinates 
# in :term:`EDI` -files. The objective is to insert the coordinates 
# into the :term:`EDI` -file throught the :func:`~get_ll` then rewrite 
# the :term:`EDI` -file with :func:`~set_ll_and_export_edis`.
# New saved dta can be used to remove interferences noises, static shift effect 
# and dropped bad frequencies. 

#%%% Import required module. 
import copy 
import os 
from watex.utils import read_data 
from watex.methods import EM   
from watex.site import Profile, Location 
from watex.methods import MT   
from watex.utils import plot_tensors    
from watex.utils.plotutils import plot_strike  
from watex.utils.funcutils import rename_files 
#%% 
# Set usefull functions for incorporating coordinates 
# into :term:`EDI` files. The coordinates were collected using 
# the Beijing projection. 

def get_ll ( data,  /, coordinate_system ='utm', epsg =15921, utm_zone ='49N'): 
    """ Get the coordinates """
    data = read_data(data, sanitize =True )
    p = Profile ( utm_zone = utm_zone, coordinate_system= coordinate_system, 
                 epsg = epsg ).fit(x='easting', y='northing', elev ='elev', 
                                   data = data )
    lat, lon = Location.to_latlon_in(p.y, p.x , utm_zone=utm_zone, 
                                    epsg =epsg, datum ='WGS84')
    return lon, lat , p.elev, data 
 
# Export new EDI-files 
def set_ll_and_export_edis ( 
        edipath, /, coord_file,   savepath , dataid =None, **kws) : 
    emo = EM().fit(edipath )
    # set longitude and latitude from Profile 
    lon, lat, elev, _ = get_ll (coord_file , **kws) 
    # read EM obj and reset attributes then rewrite object. 
    emo.longitude = lon 
    emo.latitude =lat 
    emo.elevation = elev 
    emo.rewrite (by ='id', dataid = dataid, savepath =savepath, edi_prefix=''  ) 
    return emo 
 
# * Update coordinates 
#
# Old coordinates wre reprocessed then transformed  to lon, lat which can be
# be saved to a new csv sheet. 
#
# lon, lat , _, data  = get_ll(r'D:\project-Tayuan\data\all_coordinates.csv')
# data ['longitude'] = lon 
# data ['latitude'] =lat 
# data.to_csv (r'D:\project-Tayuan\data\coordinates_end.csv', index =False )
#
#
# EDIPATH = r'D:\project-Tayuan\data\1'
# edipath =os.path.join( EDIPATH, '6HX')
# coord_file=os.path.join( EDIPATH, '1.csv' )
# savepath =os.path.join( EDIPATH, '1EDI') 
#set EMobj so use it to set rewrite the dataID 
# emo = EM().fit(edipath )
#
# export new EDI files .
# set_ll_and_export_edis ( 
#     emo.ediObjs_, 
#     coord_file=coord_file, 
#     savepath =savepath, 
#     # [::-1] is to used to reverse coordinates when the profile and the 
#     # the hardware numbering are opposite. 
#     dataid =['S{:02}'.format(ix) for ix in range ( len(emo.ediObjs_))][::-1]
#     )
# * Process AMT data and export the results into new EDI files. 
#
# new_edipath =savepath 
# outpath =os.path.join( EDIPATH, '1EDIP') # path to save new process EDI
# em0 = EM().fit(new_edipath )
# emc = copy.deepcopy(em0) # make a copy for safety
# zc = MT(verbose =True ).fit( emc.ediObjs_ )
# zc.remove_static_shift (nfreq="auto" , r = 1000,
#                          ).remove_noises (method='base', # smooothed method .
#     ).drop_frequencies (tol = .1).out(savepath =outpath) 
#
# * Visualization 
# Plot the raw resistivity and phases tensors at the first station.  
# plot_tensors ( em0.ediObjs_, station =0 ) 
#
# * plot adjusted resistivity and phases at station 10.
#
# plot_tensors (emc.ediObjs_, station = 10 )
#
# * Visualize the strike
#
# edipath =  outpath # new_edipath #
# edi_fn_lst = [os.path.join(edipath,f) for f in os.listdir(edipath) 
#         if f.endswith('.edi')] 
# plot_strike(edi_fn_lst ) 
# plot_strike(edi_fn_lst , kind = 1 ) # plot second type of plot.
#
#
# * Optionally, rename EDI  and save it.
# 
# src_path =outpath 
# dst_path =os.path.join( EDIPATH, 'renamedEDIs')
# rename_files(src_path , dst_files= dst_path , basename ='T7.', trailer='')
