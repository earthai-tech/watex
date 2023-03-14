# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio 

from __future__ import annotations 
import numpy as np 
import pandas as pd 
from ..utils.coreutils import  makeCoords 
from ..utils.gistools import ll_to_utm
from ..utils.exmath import fitfunc
from ..utils.box import Boxspace 

__all__ =["make_erp", "make_ves"] 

def make_erp (
    *, 
    n_stations:int= 42, 
    max_rho:float= 1e3 , 
    min_rho:float= 1e0, 
    step:float=20., 
    reflong:float|str='110:29:09.00', 
    reflat:float|str='26:03:05.00' , 
    utm_zone:str='29N', 
    order:str='+',
    full_coordinates:bool=True, 
    raise_warning:bool=False,
    as_frame:bool=False, 
    seed:int=None, 
    **coord_kws
    ): 
    r""" Generate Electrical Resistivity Profiling (ERP) data from stations 
    and coordinates points. 
    
    To generate samples from specific area, it is better to provide both 
    latitude and longitude values from a single station of this area as arguments 
    passed to parameters `reflat` and `reflong` respectively. Also specify the 
    `utm_zone` for the lat/lon coordinates conversion into UTM if necessary. 
    If not useful, can turn off the parameter `full_coordinates` to ``False``.
    
    Parameters 
    -----------
    n_stations: int, default=42 
        number of measurements stations 
    max_rho: float, default=1e3 
        maximum resistivity value on the survey area in :math:`\Omega.m`
        
    min_rho: float, default=1e0 
        minimum resistivity value on the survey area  in :math:`\Omega.m`
        
    reflong: float or string or list of [start, stop], default='110:29:09.00'
        Reference longitude  in degree decimal or in DD:MM:SS for the first 
        station considered as the origin of the landmark.
        
    reflat: float or string or list of [start, stop], default='26:03:05.00' 
        Reference latitude in degree decimal or in DD:MM:SS for the reference  
        site considered as the landmark origin. If value is given in a list, 
        it can contain the start point and the stop point. 
        
    step: float or str , default=20 
        Offset or the distance of seperation between different sites in meters. 
        If the value is given as string type, except the ``km``, it should be 
        considered as a ``m`` value. Only meters and kilometers are accepables.
        
    order: str , default='-'
        Direction of the projection line. By default the projected line is 
        in ascending order i.e. from SW to NE with angle `r` set to ``45``
        degrees. Could be ``-`` for descending order. Any other value should 
        be in ascending order.  
        
    utm_zone: string (##N or ##S), default='29N'
        utm zone in the form of number and North or South hemisphere, 10S or 03N
        Must be given if `utm2deg` is set to ``True``. 

    full_coordinates: bool, default=True, 
        Convert latitude and longitude to approximate UTM values. Easting and 
        northing are gotten using the reference ellipsoid =23 with WGS84. 
        If ``False``, easting and northing are not computed and set to null. 
    
    raise_warning: bool, default=True, 
        Raises warnings if :term:`GDAL` is not set or the coordinates 
        accurately status.
        
    as_frame: bool, default=False, 
         if ``True``, outputs the data into as a pandas dataframe, 
         :class:`~watex.utils.box.Boxspace` object otherwise. 
         
    seed: int, Optional,
        It allows reproducing the same data. If value is passed, it reproduces 
        the same data at that sample points.
    coord_kws: dict, 
        Additional keywords passed to :func:`~watex.utils.coreutils.makeCoords`. 
        
    Returns 
    -------- 
    (pd.Dataframe | :class:`~watex.utils.box.Boxspace` ) 
    
    
    Examples 
    ----------
    >>> from watex.datasets.gdata import make_erp 
    >>> erp_data = make_erp (n_stations =50 , step =30  , as_frame =True)
    >>> erp_data.head(3)
    Out[256]: 
       station  longitude  latitude        easting    northing  resistivity
    0        0 -13.488511  0.000997  668210.580864  110.183287   225.265306
    1       30 -13.488511  0.000997  668210.581109  110.183482   327.204082
    2       60 -13.488510  0.000997  668210.581355  110.183676   204.877551
    """
    
    stations = np.arange (0 , n_stations * step  , step ) 
    resistivity  =np.abs(np.linspace(min_rho, max_rho , n_stations) ) 
    
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(resistivity) 
    
    rlons, rlats = makeCoords(reflong, reflat, nsites=n_stations, is_utm=True, 
                              utm_zone =utm_zone, step = step,   
                              raise_warning=raise_warning, **coord_kws) 
    
    easting = np.zeros_like(rlons) ; northing = np.zeros_like(rlats) 
    
    if full_coordinates: 
        en = [ll_to_utm (23 , lat , lon ) 
         for lat, lon in zip (rlats, rlons)]
        
        _, easting , northing = zip (*en) 
 
    d = {
        "station": stations, 
        "longitude":rlons, 
        "latitude": rlats, 
        "easting": easting, 
        "northing": northing, 
        "resistivity": resistivity
        } 
    dx ={"frame": pd.DataFrame (d), "data":  pd.DataFrame (d).values }
        
    data = Boxspace(**d, **dx )

    return  data.frame if as_frame else data 


def make_ves (
    *, 
    samples:int= 31, 
    min_rho:float=1e1, 
    max_rho:float= 1e3, 
    max_depth:float= 100., 
    order:str='-', 
    as_frame:bool=False, 
    seed:int=None, 
    iorder:float=3, 
    ): 
    r""" Generate Vertical Electrical Sounding (VES) data from 
    pseudo-depth measurements. 
    
    For a large pseudo-depth measurements, one can change the number of samples 
    to a large values. The default samples presumed collected  is ``samples=31`` 
    measurements in deeper. 
    
    Parameters 
    -----------
    samples: int, default=42 
        number of measurements depth AB/2   in meters. 
        
    max_rho: float, default=1e3 
        maximum resistivity value expected in deeeper  on the survey area  
        in :math:`\\Omega.m` 
        
    min_rho: float, default=1e1 
        minimum resistivity value expected in deeper on the survey area 
        in :math:`\\Omega.m` 
        
    order: str , default='-'
        Direction of the projection line. By default the projected line is 
        in ascending order i.e. from SW to NE with angle `r` set to ``45``
        degrees. Could be ``-`` for descending order. Any other value should 
        be in ascending order.  
    max_depth: float, default=100 
        Value of the measurement in deeper expected to reach by AB/2 in meters. 
        
    as_frame: bool, default=False, 
        if ``True``, outputs the data into as a pandas dataframe, 
        :class:`~watex.utils.box.Boxspace` object otherwise. 
    seed: int, Optional,
        It allows reproducing the same data. If value is passed, it reproduces 
        the same data at that sample points. 
    iorder: float, default=3 
        Inflexion order. If ``None`` should compute using the length of  
        extrema (local + global). Must be lower as possible to let the 
        fitting VES curve more realistic. 
        
    Returns
    ---------
    (pd.Dataframe | :class:`~watex.utils.box.Boxspace` )
        
    Notes
    -------
    when returning the :class:`~watex.utils.box.Boxspace` object, each columns
    of 'VES' data can be retrieved as an attributes. Check the examples below
    
    Examples 
    ---------
    >>> from watex.datasets.gdata import make_ves 
    >>> b = make_ves (samples =50 , order ='+') # 50 measurements in deeper 
    >>> b.resistivity [:-7]
    Out[314]: 
    array([429.873 , 434.255 , 438.5707, 442.8203, 447.0042, 451.1228,
           457.5775])
    >>> b.frame.head(3)  
    Out[315]: 
        AB   MN  resistivity
    0  1.0  0.6   429.872999
    1  2.0  0.6   434.255018
    2  3.0  0.6   438.570675
    
    """ 
    
    ix , mnv= [ 4, 16 , 26 , 31 ], [.4 , 1., 5., 10.]
    abv = [1, 2,  4, 10 ] 
    ix = np.array ( np.array (ix)  * samples / 31 , dtype =int ) 
    mnv = np.array(mnv, dtype =float ) * samples / 31 
    abv =  np.array(abv, dtype =float ) * samples / 31  

    MN_max = max_depth //10  # 
    MN = _generate_ABMN_samples(indexes= ix, fixed_values=mnv,
                                threshold_multiplicator= MN_max )
    AB = _generate_ABMN_samples(indexes = ix , fixed_values=abv,
                 threshold_multiplicator=MN_max, kind ="AB")

    # make resistivity 
    g = np.linspace ( max_rho, min_rho , samples 
                     ) if order =='-' else np.linspace (min_rho, max_rho , samples )
    
    if seed is not None: 
        np.random.seed(seed )
        
    np.random.shuffle(g) 
    
    f, *_ = fitfunc (AB, g, deg= iorder )
    resistivity = np.abs(f (AB ))

    d= {"AB":AB, "MN": MN, "resistivity":resistivity}
    
    dx ={"frame": pd.DataFrame (d), "data":  pd.DataFrame (d).values }
        
    data = Boxspace(**d, **dx )

    return  data.frame if as_frame else data 

def _generate_ABMN_samples (
        indexes, fixed_values, threshold_multiplicator, kind ='MN'): 
    """ Isolated part for 'VES' data generating. """
    # use index MN for test 
    MN=list() 
    nsam = np.array (np.diff (indexes ), dtype = int )
    for ii, (ind, v)  in enumerate (zip (indexes, fixed_values)): 
        if ii==0: 
            s = [ v * threshold_multiplicator / 10 for i in range (
                ind ) ] if kind =="MN" else  list(
                    np.linspace (1, indexes [0], indexes [0]))
        else: 
            s = [v* threshold_multiplicator / 10 for i in range (
                indexes[ii -1], indexes[ii])] if kind=="MN" else  list( 
                    np.arange (0, nsam[ii-1] * v, v ) + s[-1] + fixed_values[ii-1]) 
        
        MN.extend (s)
        
    return np.around (MN, 1 )   
    
    
