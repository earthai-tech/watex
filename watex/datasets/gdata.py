# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio 

from __future__ import annotations 
import warnings 
import numpy as np 
import pandas as pd 
from ..utils.coreutils import  makeCoords 
from ..utils.funcutils import _assert_all_types,  is_iterable
from ..utils.gistools import ( 
    ll_to_utm, project_points_ll2utm, project_point_utm2ll) 
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
    is_utm:bool=False, 
    epsg: int =None, 
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
        
    is_utm: bool, default=False
       Type of coordinates passed to `reflat` and reflong` params for generating 
       `longitude-latitude` coordinates. If `is_utm` is explicity set to 
       ``True``, that means values `reflong` and `reflat` arein UTM coordinates. 
       Then the conversion to `longitude-latitude` should be operated. However 
       if `is_utm` is ``False`` when `reflat` and `reflong` values are greater 
       than ``90`` and ``180`` degrees respectively, an errors should raise.
       
       .. versionadded:: 0.2.1 
       
    epsg: int, str, Optional 
       EPSG number defining projection. See http://spatialreference.org/ref/ 
       for moreinfo. Overrides utm_zone if both are provided

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
    
    rlons, rlats = makeCoords(reflong, reflat, nsites=n_stations,
                              utm_zone =utm_zone, step = step, 
                              is_utm= is_utm,   
                              raise_warning=raise_warning, epsg = epsg , 
                              **coord_kws) 
    
    easting = np.zeros_like(rlons) ; northing = np.zeros_like(rlats) 
    
    if full_coordinates: 
        try :
            # projection is more consistent than using the
            # the reference ellipsoid. If error occurs then use  
            # the default computation.
            en = [project_points_ll2utm(lat= lat, lon= lon, 
                                        utm_zone= utm_zone, 
                                        epsg=epsg )   
                  for lat, lon in zip (rlats, rlons)]
        except: 
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
    iorder:int=3, 
    xy = None, 
    is_utm=False, 
    add_xy:bool =False,  
    utm_zone=None, 
    epsg: int=None 
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
        
    iorder: int, default=3 
        Inflexion order. It is a positive value greater than 0. If ``None``, it 
        should be computed using the length of  extrema (local + global). 
        It also might be lower as possible to let the fitting VES curve more 
        realistic. 
        
    xy: tuple, optional  
       Coordinates point  ( easting, northing ) or (lon, lat) corresponding 
       to the :term:`VES` points ``sves``. If coordinates values are not given 
       coordinates are randomly generated into (lon, lat) and stored into the 
       attribute `xy`. 
       To returns the xy auto-coordinates when ``as_frame=True`` set `add_xy`
       to ``True``. 
       
       .. versionadded:: 0.2.1 
      
    is_utm: bool, default=False 
       In principle, `xy` expects to be in `longitude-latitude` coordinates. 
       However if coordinates are passed into a UTM such as `easting-northing`, 
       user can specify the `utm_zone` to convert the `xy` values into 
       a valid longitude and latitude coordinates. 
       
    add_xy: bool, default=False 
       Add `xy` coordinates to the :term:`VES` dataframe. 
     
    utm_zone: str, Optional 
      To generate coordinates `xy` from a specific zone, `utm_zone` can be 
      specified, otherwise ``29N`` is used instead. 
      
    epsg: int, str, Optional 
       EPSG number defining projection. See http://spatialreference.org/ref/ 
       for moreinfo. Overrides utm_zone if both are provided

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
    >>> ves_data = make_ves (samples =50 , min_rho =10, max_rho =1e5 , 
                             as_frame =True, add_xy= True , 
                             xy = ( 3143965.855 , 336704.455) , 
                             is_utm = True , utm_zone = 
                             '49N', epsg =None)
    >>> ves_data.head(2) 
    Out[316]: 
        AB   MN   resistivity   longitude   latitude
    0  1.0  0.6  51544.426685  107.901553 -61.802165
    1  2.0  0.6  51420.739513  107.901553 -61.802165
    """ 
    
    ix , mnv= [ 4, 16 , 26 , 31 ], [.4 , 1., 5., 10.]
    abv = [1, 2,  4, 10 ] 
    
    samples = int (_assert_all_types(samples, int, float, objname="Samples")) 
    
    if samples < 8: 
        warnings.warn("Expects at least samples be greater than 7."
                      f" Got '{samples}'")
        samples = 8 
        
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
    iorder = int ( _assert_all_types(iorder, int, float, 
                                     objname ="Inflexion order 'iorder'")
                  )
    
    if iorder <=0: 
        raise ValueError ("Inflexion order 'iorder' expects a positive"
                          f" number greater than 0. Got '{iorder}'.")
        
    f, *_ = fitfunc (AB, g, deg= iorder )
    resistivity = np.abs(f (AB ))

    d= {"AB":AB, "MN": MN, "resistivity":resistivity}
    
    dx ={"frame": pd.DataFrame (d), "data":  pd.DataFrame (d).values }
        
    # # fetch_random coordinate for sves or manage xy coordinates 
    xy, is_utm  = _manage_xy_coordinates(
        xy , is_utm = is_utm , utm_zone = utm_zone, epsg = epsg )
         
    # now add_xy  
    if add_xy : 
        xname = "easting" if is_utm else 'longitude'
        yname = "northing" if is_utm else 'latitude'
        
        d [xname] = xy if np.isnan (xy).any()  else xy [0] 
        d [yname]= xy if np.isnan (xy).any()  else xy [-1] 
        # insert longitude and latitude in dataframe 
        dx['frame'][xname] = d [xname] 
        dx['frame'][yname] = d [yname]

    dx ['xy']= xy     
  
    data = Boxspace(**d, **dx ,  utm_zone = utm_zone  or '29N')
 
    
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
    
    
def _manage_xy_coordinates ( 
        xy= None,  *, is_utm =False, utm_zone = None,  epsg=None  ):
    """ Manage the coordinates or generate random coordinates.
    Isolated part of `make_ves`. 
    """
    
    # fetch_random coordinate for sves
    if xy is not None: 
        xy_init = xy # make a copy
        if isinstance ( xy, str): 
            xy = is_iterable (xy, parse_string= True, transform = True  ) 
            
        xy = is_iterable(xy, exclude_string= True, transform = True )  
        
        if len(xy )!=2 : 
            # in the case long value is given 
            warnings.warn(
                "Unexpected xy coordinates. xy should be a tuple of"
                " (easting, northing) or (longitude, latitude)."
                f" Got {xy_init}. Can't use it. Set to NaN instead.") 
            
            xy = np.nan

        if is_utm and not np.isnan(xy).any() : 
            if ( 
                    utm_zone is None 
                    and epsg is None 
                    ): 
                warnings.warn ("GISError: Need to input either UTM zone or "
                               "EPSG number for an accurate conversion. Use "
                               "default `49R` instead.")
                
            # is given into longitude latitude 
            # so reverse it 
            yx = project_point_utm2ll(*xy[::-1] , utm_zone = utm_zone or '29N' , 
                                      epsg= epsg 
                                      ) 
            xy = yx [::-1] # reverse back to longitude latitude 
            
            is_utm=False # conversion is done 
            
    if xy is None: 
        # collect 7 sites and fetch randomly
        # one site # turn off warnings about 
        # using the 
        x, y  = makeCoords (3143965.855 , 336704.455 , 
                            utm_zone= utm_zone or '49R', 
                            nsites= 7 , is_utm=True , 
                            raise_warning= False )
        sves_ix = np.random.choice ( np.arange ( len(x)))   
        xy  = ( x [sves_ix], y [sves_ix])
        # block to is_utm equals to False  
        # since values are alread be converted to lonlat 
        is_utm =False  
    
    return xy , is_utm 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
