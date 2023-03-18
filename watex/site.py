# -*- coding: utf-8 -*-
#   Licence:BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Manage site data.
"""
from __future__ import annotations 
import copy 
import re

import warnings 
import numpy as np 

from ._watexlog import watexlog 
from ._typing import ( 
    Optional, 
    ArrayLike,
    F
    )
from .exceptions import ( 
    SiteError, 
    ProfileError,
    NotFittedError, 
    )
from .property import ( 
    UTM_DESIGNATOR
    )
from .utils.coreutils import makeCoords 
from .utils.exmath import ( 
    get_bearing, 
    scalePosition, 
    )
from .utils.funcutils import (
    _assert_all_types, 
    to_numeric_dtypes,
    _validate_name_in, 
    interpolate_grid,
    reshape,
    ) 
from .utils.gistools import (
    assert_elevation_value, 
    assert_lat_value, 
    assert_lon_value , 
    ll_to_utm, 
    utm_to_ll, 
    project_points_ll2utm, 
    project_point_utm2ll,
    assert_xy_coordinate_system,
    convert_position_str2float, 
    convert_position_float2str, 
    )
from .utils.validator import ( 
    _check_consistency_size , 
    _is_numeric_dtype , 
    assert_xy_in, 
    ) 

__all__= ['Location', 'Profile']

class Profile: 
    
    def __init__(
        self , 
        *, 
        utm_zone =None, 
        coordinate_system=None, 
        datum = 'WGS84', 
        epsg = None, 
        reference_ellipsoid =23, 
        ): 
        self.epsg=epsg 
        self.datum =datum 
        self.reference_ellipsoid= reference_ellipsoid 
        self.utm_zone=utm_zone
        self.coordinate_system= coordinate_system

    def fit(
        self, 
        x:ArrayLike |str , 
        y:ArrayLike | str ,
        elev: Optional [ArrayLike] =None, 
        **fit_params
        ): 
        """ Populate profile attributes from x and y. 
        
        By default if the coordinate system is given as latitude/longitude 
        x, y are latitude and longitude respectively. 
 
        Parameters
        ------------
        x, y: ArrayLike 1d /str  
           One dimensional arrays. `x` can be consider as the abscissa of the  
           landmark and `y` as ordinates array.  If `x` or `y` is passed as 
           string argument, `data` must be passed as `fit_params` keyword 
           arguments and the name of `x` and `y` must be a column name of 
           the `data`. 
           By default `x` and `y` are considered as `longitude` and `latitude` 
           when ``dms`` or ``ll`` coordinate system is passed. 
  
        elev: ArrayLike 1d/str 
            Arraylike 1d of elevation at each positions. If not supplied 
            should be set to null at each points. If given, it must be  
            consistent with `x` and `y`. 
            
        data: pd.DataFrame 
           Data containing `x` and `y` values as series. Then if `x` and `y`
           are given as string argument, their names must be included in the 
           data columns. Otherwise an error will raise. 
        
        Return 
        ---------
        self :`watex.site.Profile` 
           Object of Profile. 
           
        Note 
        -------
        When `data` is supplied and `x` and `y` are given by their names 
        existing in the dataframe columns, by default, the non-numerical 
        data are removed. However, if `y` and `x` are given in DD:MM:SS in 
        the dataframe, the coordinate system must explicitly set to ``dms`
        to keep the non-numerical values  in the data. 
        
        """
        emsg =("'DD:MM:SS.ms' coordinate system is detected.")
        data = fit_params.pop('data', None) 
  
        self.coordinate_system = str(
            self.coordinate_system).lower().strip() 
        if  ( 
                self.coordinate_system.find ('dms')>=0 
            or 'dd:mm' in self.coordinate_system ) : 
            self.coordinate_system='dms'
    
        if data is not None: 
            # when coordinate_system is explicity passed 
            # and data is given, don't suppress the non-numerical 
            # data. Keep it untouch and try to convert x, y to 
            # float type later. 
            suppress_cf = True 
            if self.coordinate_system=='dms': 
               suppress_cf= False 
            # suppress non numerical values 
            data = to_numeric_dtypes(data, pop_cat_features= suppress_cf)
            # get elevation if string is given 
            if isinstance (elev , str): 
                elev = elev if elev in data.columns else None 
                if elev is None: 
                    warnings.warn(
                        f"Elevation {elev!r} not found in the dataframe")
                if elev is not None: 
                    elev = np.array (data[elev]) 
        
        self.x, self.y = assert_xy_in( x , y, data = data )
        
       
        if self.coordinate_system in ('none', 'auto'): 
            self.coordinate_system = assert_xy_coordinate_system (
                self.x, self.y )
        
        if self.coordinate_system =='dms' : 
            try : 
                self.x, self.y = Profile.dms2ll(self.x, self.y) 
            except ValueError as e: 
                raise ( emsg + str(e))
            else: 
                # initialize the system to ll if 
                # conversion succeeded. 
                self.coordinate_system =='ll' 
        # set 0. to elevation 
        self.elev = elev if elev is not None else np.zeros_like(
            self.x, dtype = np.float32 ) 
        self.elev = np.array (self.elev , dtype = np.float32 )
        
        if not _check_consistency_size (self.x , np.array( self.elev),
                                        error ='ignore'  ): 
            raise ProfileError ("Elevation and x or y must be consistent."
                              f" Got {len(x)} and {len(elev)}") 

        return self 
        
    def bearing (self, *, to_degree:bool = True ): 
        """
        Compute the bearing between calculate bearing between two coordinates.
        
        A bearing is a direction of one point relative to another point, 
        usually given as an angle measured clockwise from north. 
        In navigation, bearings are often used to determine the direction 
        to a destination or to plot a course on a map. There are two main 
        types of bearings: absolute bearing and relative bearing.
        
        - Relative bearing 
          refers to the angle between the forward direction of a craft 
          (heading) and the location of another object. 
        - Absolute bearing 
          refers to the angle between the magnetic north (magnetic bearing) 
          or true north (true bearing) and an object
          
        The  bearing (:math:`\beta`) between two coordinates points 1(lon1, lat1) 
        and 2 (lon2, lat2) can becalculated as:
        
        .. math:: 
            
            \beta = atan2(sin(lon2-lon1)*cos(lat2), cos(lat1)*sin(lat2) – \
                          sin(lat1)*cos(lat2)*cos(lon2-lon1))
                
        By default, the first and last coordinates for points 1 and 2 are 
        used as `latlon1` and `latlon2` respectively. 
        
        Parameters 
        ------------
        to_degree: bool, default=True 
          Convert the bearing from radians to degree. 
          
        Returns 
        --------
        :math:`\beta` : float, 
           The value of bearing in degree ( default). 
           
        """
        msg =(
            "x, y are not in longitude/latitude format"
            " while 'utm_zone' is not set. The bearing"
            " should be less accurate. Provide the UTM"
            " zone to improve the accuracy.")
        
        self.inspect 
        
        if self.coordinate_system =='ll': 
            xs = np.array(copy.deepcopy(self.x)) 
            ys = np.array(copy.deepcopy(self.y))
        
        if ( 
                self.coordinate_system !='ll' 
                and self.utm_zone is None) : 
            warnings.warn(msg ) 
            
        self.utm_zone = self.utm_zone or '49R'    
        if self.coordinate_system !='ll': 
            # recompute value to lat/lon 
            # from easting/northing
            ys , xs = Location.to_latlon_in(
                self.x, self.y, utm_zone= self.utm_zone) 
        # compute bearing.     
        return get_bearing((ys[0], xs[0]) , ( ys[-1], xs[-1] ), 
                           to_deg= to_degree 
                           ) 
    
    def distance (self, *, average_distance: bool = True): 
        """Compute the distance between profile coordinates points.
        
        Preferably, it uses the UTM coordinates positions. By default 
        the coordinate system is automatically detected. 
        
        Parameters 
        -----------
        average_distance: bool, default =True, 
           Returns the average value of the distance between different points. 
           
        is_latlon: bool, default=False, 
            Convert `x` and `y` latitude  and longitude coordinates values 
            into UTM before computing the distance. `x`, `y` should be considered 
            as ``easting`` and ``northing`` respectively. 
            
        kws: dict, 
           Keyword arguments passed to :meth:`watex.exmath.get_distance`
           
        Returns 
        ---------
        d: Arraylike of shape (N-1) 
          Is the distance between points or the average of all distances.  
        
        Examples 
        --------- 
        >>> import numpy as np 
        >>> from watex.site import Profile 
        >>> posx = np.random.rand (7) *10 
        >>> posx = np.abs ( np.random.randn (7) * 12 ) 
        >>> po= Profile().fit(posx, posy )
        >>> # convert data to UTM and compute distance becuase 
        >>> # our toy example has value less than 90 and 180.
        >>> po.distance () 
        251053.3287093233
        >>> po.coordinate_system = 'UTM' 
        >>> 6.451210308544236
        """
        self.inspect 
        
        x, y = self.x , self.y 
        
        if self.coordinate_system =='ll': 
            x , y = Location.to_utm_in(
                lats= self.y, lons =self.y, 
                epsg = self.epsg , 
                datum = self.datum , 
                reference_ellipsoid= self.reference_ellipsoid
            ) 
    
        d = np.sqrt( np.diff (x) **2 + np.diff (y)**2 ) 
        
        return d.mean()  if average_distance else d 
    
    def scale_positions (self, **sp_kws): 
        """
        Scale the position coordinates along x and y.
        
        Parameters
        -----------
        sp_kws: dict 
           Keyword arguments passed to :func:`~watex.utils.scalePosition`. 
        Returns 
        --------
        x, y : Arraylikes 
           Scaled positions 
           
        See also 
        ---------
        watex.utils.scalePositions: 
            Scale positions using the `scipy` curve fit. 
        watex.utils.exmath.scale_positions: 
            Scale and shift positions using bearing. 
            
        """
        self.inspect 
        x, *_ = scalePosition(self.x , **sp_kws) 
        y, *_ = scalePosition(self.y , **sp_kws)
        
        return x, y 
    
    def shift_positions (
        self, 
        *, 
        step:float= None, 
        use_average_dist:bool=False,
        angle:Optional[float]=None,
        ): 
        """
        Shift the x and y  position coordinates from the step and angle. 
         
        By default, it consider `x` and `y` as easting/latitude and 
        northing/longitude coordinates respectively. It latitude and longitude 
        are given, specify the parameter `is_latlon` to ``True``. 
        
        Parameters
        ------------
        step: float, Optional 
           The positions separation. If not given, the average distance between 
           all positions should be used instead. 
        use_average_dist: bool, default=False, 
           Use the distance computed between positions for the correction. 
           
        angle: float, Optional 
           Bearing angle in degree to shift the profile line . If ``None``, 
           the ``bearing`` of `x` and `y` is used instead.
           
        Returns 
        --------
        xx, yy: Arraylike 1d, 
           The arrays of position correction from `x` and `y` using the 
           bearing. 
           
        See Also 
        ---------
        watex.utils.exmath.get_bearing: 
            Compute the  direction of one point relative to another point. 
          
        Examples
        ---------
        >>> from watex.utils.exmath import scale_positions 
        >>> east = [336698.731, 336714.574, 336730.305] 
        >>> north = [3143970.128, 3143957.934, 3143945.76]
        >>> p= Profile().fit (east, north)
        >>> east_c , north_c= p.scale_positions (east, north, step =20) 
        >>> east_c , north_c
        (array([336686.69198337, 336702.53498337, 336718.26598337]),
         array([3143986.09866306, 3143973.90466306, 3143961.73066306]))
        """
        cs ={'ll': 'longitude/latitude', 
             'dms': 'dd:mm:ss'}
        self.inspect 
        
        if self.coordinate_system !='utm': 
            cse = cs.get('ll').title () if self.coordinate_system =='ll' else (
                cs.get('dms').upper()  if self.coordinate_system =='dms' else
              self.coordinate_system .upper() )
            warnings.warn((
                "{0!r} coordinates system is detected. Shifting positions" 
                " expects UTM coordinates. It is recommended to convert"
                " positions data to UTM (ref:`watex.site.to_utm_in`) before"
                " processing. The following with {0} coordinates"
                " might lead to invalid results. Use at your own risk." 
                ).format(cse)
                          )
            
        if ( not use_average_dist
            and step is None 
            ): 
            warnings.warn("Step is not given. The mean-distance of"
                          " positions should be used instead.")
            use_average_dist =True 
            
        if use_average_dist: 
            step = self.distance (
                average_distance= use_average_dist) 
 
        step  = float (_assert_all_types(step, float, int, objname ='Step'))

        if angle is not None: 
            angle  = float (_assert_all_types(
                step, float, int, objname ='Bearing angle (in degree)'))
            
            angle = np.deg2rad (angle %360)  
            
        angle = angle or self.bearing(to_degree =False ) # return bearing in rad.
     
        xx = self.x + ( step * np.cos (angle))
        yy = self.y +  (step  * np.sin(angle))

        return xx, yy 
    
    @staticmethod 
    def f_ (ar , /,  func: str | F  = 'dms->ll'): 
        """
        Converter position function from dms to longitude/latitude degree 
        decimal or vice versa.
        
        Convert position from str (DD:MM:SS) to float (latitude/longitude)
        and vice versa.
        
        Parameters 
        -----------
        ar: ArrayLike 1d, 
           Array containing the position coordinates for conversion. 
        func: Callable or str, default ='dms->ll'
            Converter functions. They can be: 
               
            - :func:`~watex.utils.gistools.convert_position_str2float` 
              for ``:dd:mm:ss``  to foat(long, lat) coordinates. If 
              string is passed it should be ['dms2ll'|'dmstoll'|'dms-<ll']. 
            - :func:`~watex.utils.gistools.convert_position_float2str` 
              from float (long, lat) in decimal degree coordinates 
              to ``dd:mm:ss``. When string is passed, it should be 
              ['ll2dms'|'lltodms'|'ll->dms']
               
        Returns
        --------
        generator obj. 
           Map object composed of value converted. 
        
        """
        if isinstance (func, str): 
            f = _validate_name_in(func, defaults = (
                'll2dms', 'lltodms', 'll->dms'), 
                expect_name= convert_position_float2str 
                ) 
        if not callable (f): 
            f = convert_position_str2float 
            
        # faster than 
        # x = np.array ( [ convert_position_str2float(i) for i in x ], 
        #               dtype = np.float64)
        # while apply_along not possible due to the string dtype during 
        # the loop
        # x = np.apply_along_axis(convert_position_str2float, 0, 
        #                         np.array (x, dtype =str ))
        return map ( lambda i: f(i) , ar)
    
    @staticmethod 
    def dms2ll (x:ArrayLike  , y:ArrayLike ): 
        """ Convert array x and y from DD:MM:SS to degree decimal -longitude 
        (x) and latitude (y). 
        
        Parameters
        -----------
        x, y: ArrayLike containing the degree-minutes-seconds (DMS) coordinates
           positions.
        Returns 
        --------
        x, y: Arraylike 
           ArrayLike in degree decimal coordinates format. By default `x` and 
           `y` are longitude and latitude respectively. 
           
        Examples
        ---------
        >>> 
        >>> from watex.site import Profile 
        >>> x=['20:15:35'] ; y =['7:45:8.5'] 
        >>> Profile.dms2ll (x, y)
        Out[83]: (array([20.25972222]), array([7.75236111]))
        """
 
        if not _is_numeric_dtype(x , to_array =True ): 
           # reconvert object to str for consistency  
           x = np.array ( list ( Profile.f_(x)), dtype = np.float64 )
        if not _is_numeric_dtype(y , to_array =True): 
           y = np.array ( list (Profile.f_ (y)), dtype = np.float64 )
          
        return x, y
    
    @staticmethod 
    def ll2dms (x:ArrayLike  , y:ArrayLike ): 
        """
        Convert array x and y from degree decimal  to 
        degree-minutes-seconds (DMS)
        
        Parameters 
        -----------
        x, y: ArrayLike containing the degree decimal position 
           coordinates. 
        
        Returns 
        --------
        x, y: Arraylike 
           ArrayLike in DD:MM:SS coordinates format.
           
        Examples
        ---------
        >>> from watex.site import Profile 
        >>> x =[15.18 ] ; y =[19.60]
        >>> Profile.ll2dms (x, y)
        Out[84]: (array(['15:10:48.00'], dtype='<U11'), 
                  array(['19:36:00.00'], dtype='<U11'))
        """
        if _is_numeric_dtype(x , to_array =True ):  
           x = np.array ( list (Profile.f_ (x, 'll->dms')), dtype = str )
        if _is_numeric_dtype(y , to_array =True): 
           y = np.array ( list (Profile.f_ (y, 'll->dms')), dtype = str)
          
        return x, y
    
    def make_xy_coordinates (
        self, 
        *, 
        step =None,
        r = None, 
        todms:bool =False, 
        **kws 
        ): 
        """
        Generate synthetic coordinates from references latitude and 
        longitude from x and y. 
        
        Parameters 
        -----------
        step: float or str 
            Offset or the distance of seperation between different sites 
            in meters. If the value is given as string type, except 
            the ``km``, it should be considered as a ``m`` value. Only 
            meters and kilometers are accepables.
            
        r: float or int 
            The rotate angle in degrees. Rotate the angle features 
            toward the direction of the projection profile. 
            Default value use the :meth:`~.bearing` value in degrees.   
            
        utm_zone: string (##N or ##S)
            utm zone in the form of number and North or South hemisphere, 
            10S or 03N  Must be given if coordinate system is ``UTM``. 
                          
        todms: bool, Default=False
            Reconvert the longitude/latitude degree decimal values into 
            the DD:MM:SS. 
     
        kws: dict, 
           Additional keywords of :func:`~watex.utils.exmath.makeCoords`.   
           
        Returns 
        ----------
        lon, lat : ArrayLike 
           ArrayLike of synthetic coordinates latitude and longitude. 
          
        See Also 
        --------
        watex.utils.exmath: 
            Create mutiple coordinates with different kinds 
            
        Examples 
        ----------
        >>> from watex.utils.coreutils import makeCoords 
        >>> rlons, rlats = makeCoords('110:29:09.00', '26:03:05.00', 
        ...                                     nsites = 7, todms=True)
        
        """
        def _auto ( v): 
            if str (v).lower().strip() in ('none', 'auto'): 
                return None 
            
        self.inspect 
        step = _auto (step ); r = _auto (r )
        # use distance by default and bearing as r angle 
        step = step or self.distance (average_distance= True )
        r = r or self.bearing() 
  
        reflat = ( self.y[0], self.y[-1]) 
        reflong = (self.x[0], self.x[-1])
  
        isutm = False if self.coordinate_system =='ll' else True 
        return makeCoords(
            reflong, 
            reflat, 
            nsites = len(self.x ), 
            r= r ,  
            step =step , 
            todms=todms, 
            utm_zone= self.utm_zone, 
            is_utm= isutm, 
            **kws
            ) 
    def interpolate(
        self, 
        method ='linear', 
        inplace =True, 
        **kws
        ): 
        """
        Interpolate x, y and elev ( if applicable).
        
        Parameters 
        -----------
        inplace: bool, default=True 
           Erase existing value of x , y and elev with the interpolated one. 
           If ``False`` , its return interpolated x, y and elev. 

        method: bool, default='nearest', 
           Method of interpolation. One of ['nearest'|'linear'|'cubic'] 
         
        kws: dict,
           Additional keywords arguments passed to
           :func:`~watex.utils.funcutils.interpolate_grid`. 
           
        Returns
        --------
        self|x, y, elev: :class:`~watex.site.Profile` or Arraylikes 
           `:class:`.Profile` objects if `inplace` is ``True`` or 
           interpolated x, y and elev. 
           
        See Also 
        --------
        watex.utils.funcutils.interpolate_grid: 
            Interpolate two dimensional array. 
            
        Examples 
        --------
        >>> import numpy as np 
        >>> from watex.site import Profile 
        >>> x = [ 28, np.nan, 50, 60 ]
        >>> y =[ np.nan, 1000, 2000, 3000]
        >>> elev=[ 0, 1 , np.nan, np.nan]
        >>> po = Profile().fit(x, y, elev )
        >>> po.interpolate () 
        >>> po.x 
        array([28., 39., 50., 60.])
        >>> po.y 
        array([1000., 1000., 2000., 3000.])
        >>> po.elev 
        array([0., 1., 1., 1.])

        """ 
        emsg = ("Interpolation expects x, y and elev to have a consistent"
                " size. sizes x, y and elev are {}, {} and {}.")
        
        self.inspect 
        
        try :
            _check_consistency_size(self.x, self.y) 
            _check_consistency_size(self.y, self.elev)
        except: 
            raise ProfileError(emsg.format(
                len(self.x), len(self.y), len(self.elev )) )
            
        # make a grid data along axis =0 
        ar = np.vstack ((self.x, self.y, self.elev ))
        # interpolate is done along axis =0 so 
        # for x, y and elev we may transpose 
        # the data first and do the reverse 
        # processing back for new x, y and z 
        ar = interpolate_grid(ar, method = method , **kws) 
        
        x , y, elev = np.vsplit (ar  , 3 )
        x, y, elev = reshape (x) , reshape (y), reshape (elev)
        if inplace : 
            self.x, self.y, self.elev = x, y, elev 
            
            return self 
        
        return x, y, elev 

    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        if not hasattr (self, 'x'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 
    
    def __repr__(self): 
        """ Represent the output class format """
        t_=("utm_zone" ,"coordinate_system","datum" ,
            "epsg" ,"reference_ellipsoid" ) 
        
        outm = ( '<{!r}:' + ', '.join(
            [f"{k}={getattr(self, k)!r}" for k in t_]) + '>' 
            ) 
        return  outm.format(self.__class__.__name__)
    
    
Profile.__doc__="""\
Profile class deals with the positions collected in the survey area. 

In principle, there is no exact solution  between longitude/latitude to 
x/y coordinates. Indeed, there is no isometric map from the sphere to the 
plane. Indeed, when you convert lon/lat coordinates from the sphere to x/y 
coordinates in the plane, we cannot hope that all lengths will be preserved 
by this operation. Therefore, we have to accept some kind of deformation. 
For smallish parts of earth's surface, transverse Mercator is quite common.

By default, we use ``x`` for `longitude` and ``y`` for `latitude`. This is 
useful when data is passed as longitude-latitude (``ll``) or degree-minutes-
seconds (``dms``) in the `fit` method.

Parameters 
----------
utm_zone: Optional, string
   zone number and 'S' or 'N' e.g. '55S'. Default to the centre point
   of coordinates points in the survey area. It should be a string (##N or ##S)
   in the form of number and North or South hemisphere, 10S or 03N
coordinate_system: str, ['utm'|'dms'|'ll'] 
   The coordinate system in which the data points for the profile is collected. 
   If not given, the auto-detection will be triggered and find the  suitable 
   coordinate system. However, it is recommended to provide for consistency. 
   Note that if `x` and `y` are composed of value less than 180 degrees 
   for longitude and 90 degrees for latitude should be considered as ``ll`` 
   longitude-latitude (``ll``) coordinates system. If `x` and `y` are 
   degree-minutes-second (``dms`` or ``dd:mm:ss``) data, they must be 
   specify as coordinate system in order to accept the non-numerical data 
   before transforming to ``ll``. If ``data`` is passed to the :meth:`.fit`
   method and ``dms`` is not specify, `x` and `y` values should be discarded.
   
datum: string, default = 'WGS84'
   well known datum ex. WGS84, NAD27, NAD83, etc.

epsg: Optional, int
   epsg number defining projection (
        see http://spatialreference.org/ref/ for moreinfo)
   Overrides utm_zone if both are provided. 

reference_ellipsoid: int, default=23 
   reference ellipsoids is derived from Peter H. Dana's website-
   http://www.utexas.edu/depts/grg/gcraft/notes/datum/elist.html
   Department of Geography, University of Texas at Austin
   Internet: pdana@mail.utexas.edu . Default is ``23`` constrained to 
   WGS84. 
    
Examples 
--------- 
>>> from watex.datasets import load_edis
>>> from watex.site import Profile  
>>> xy = load_edis (samples =17 , as_frame =True , key ='latitude longitude') 
>>> xy.head (2)
    longitude   latitude
0  110.485833  26.051390
1  110.486153  26.051794 
>>> pro= Profile ().fit( xy.longitude, xy.latitude) 
>>> pro.distance ()
62.890276656978244
>>> pro.bearing () 
35.4252016495945
>>> pro.make_xy_coordinates( ) 
(array([110.48583316, 110.48615319, 110.48647322, 110.48679325,
       110.48711328, 110.48743331, 110.48775334]), 
 array([26.05138954, 26.05179371, 26.05219788, 26.05260205, 26.05300622,
       26.05341038, 26.05381455]))
"""
    
class Location (object): 

    def __init__(
        self, 
        lat= None, 
        lon=None, 
        **kwds
        ) :
        self._logging = watexlog.get_watex_logger(
            self.__class__.__name__)
        self._lat = lat 
        self._lon = lon 
        
        self.datum = kwds.pop('datum', 'WGS84') 
        self.epsg = kwds.pop('epsg' , None) 
        self.reference_ellipsoid = kwds.pop('reference_ellipsoid', 23)
        self._utm_zone =kwds.pop('utm_zone', None) 
        self._elev = kwds.pop('elev', None)
        self._east= None 
        self._north =None 
    @property 
    def utm_zone (self): 
        return self._utm_zone
        
    @property 
    def lat(self): 
        return self._lat
    @property 
    def lon(self) : 
        return self._lon
    @property 
    def east(self ): 
        return self._east
    @property 
    def north(self):
        return self._north
    @property 
    def elev(self): 
        return self._elev

    @utm_zone.setter 
    def utm_zone (self, utm_zone): 
        utm_xone = copy.deepcopy(utm_zone) 
        utm_zone = str(utm_zone).upper().strip() 
        str_ = f"{'|'.join([ i for i in UTM_DESIGNATOR.keys()]).lower()}"
        regex= re.compile(rf'{str_}', flags=re.IGNORECASE)
        if regex.search(utm_zone) is None: 
           raise SiteError (f"Wrong UTM zone value!: {utm_xone!r} ")
        self._utm_zone =utm_zone.upper() 
    
    @lat.setter 
    def lat (self, lat): 
        self._lat = assert_lat_value(lat)
        
        
    @lon.setter 
    def lon(self, lon) : 
        self._lon = assert_lon_value(lon)
        
    @elev.setter 
    def elev(self, elev) : 
        self._elev = assert_elevation_value(elev ) 
        
    @east.setter 
    def east (self, east): 
        self._east = np.array(east, dtype = float )

    @north.setter 
    def north (self, north): 
        self._north = np.array(north, dtype =float)
        
    def to_utm (
        self, 
        lat:float=None, 
        lon:float=None, 
        datum:str= None ,
        utm_zone:str=None, 
        epsg:int=None, 
        reference_ellipsoid:int= None
        ): 
        """
        Project coordinates to utm if coordinates are in degrees at  
        given reference ellipsoid constrained to WGS84 by default. 
         
        Parameters 
        -----------
        lat: float or string (DD:MM:SS.ms)
            latitude of point
                  
        lon: float or string (DD:MM:SS.ms)
            longitude of point
        
        datum: string, default='WGS84'
            well known datum ex. WGS84, NAD27, NAD83, etc.

        utm_zone: Optional, string
            zone number and 'S' or 'N' e.g. '55S'. Defaults to the centre point
            of the provided points
                       
        epsg: Optional, int
            epsg number defining projection (see http://spatialreference.org/ref/ for moreinfo)
            Overrides utm_zone if both are provided

        reference_ellipsoid: Optional, int 
            reference ellipsoids is derived from Peter H. Dana's website-
            http://www.utexas.edu/depts/grg/gcraft/notes/datum/elist.html
            Department of Geography, University of Texas at Austin
            Internet: pdana@mail.utexas.edu . Default is ``23`` constrained to 
            WGS84. 
             
        Returns
        --------
        proj_point: tuple(easting, northing, zone)
            projected point in UTM in Datum
            
        """
        
        if lat is not None : 
            self.lat = lat
        if lon is not None : 
            self.lon =lon
            
        if (self.lat or self.lon) is None : 
            raise SiteError (" Latitude and longitude must not be None")
            
        self.epsg = epsg or self.epsg 
        self.datum = datum  or self.datum 
        if not self.datum:  
            self.datum = 'WGS84'

        self.reference_ellipsoid = reference_ellipsoid or \
            self.reference_ellipsoid
     
        try : 
            self.utm_zone, self.east, self.north= ll_to_utm(
                reference_ellipsoid=self.reference_ellipsoid,
                lat = self.lat, 
                lon= self.lon)
        
        except :
            self.east, self.north, self.utm_zone= project_points_ll2utm(
                lat = self.lat ,
                lon= self.lon,
                datum = self.datum , 
                utm_zone = self.utm_zone ,
                epsg= self.epsg 
                ) 
            
        return float(self.east), float(self.north)
    
        
    def to_latlon(
        self, 
        east:float=None, 
        north:float= None, 
        utm_zone:str=None, 
        reference_ellipsoid:int=None , 
        datum:str = None 
        ): 
        """
        Project coodinate on longitude latitude once  data are utm at  given
        reference ellispoid constrained to WGS-84 by default. 
        
        Parameters 
        -----------
        east: float
            easting coordinate in meters
                    
        north: float
            northing coordinate in meters
        
        utm_zone: Optional, string (##N or ##S)
            utm zone in the form of number and North or South hemisphere, 10S or 03N
        
        datum: string, default ='WGS84'
            well known datum ex. WGS84, NAD27, etc.
            
        reference_ellipsoid: Optional, int 
            reference ellipsoids is derived from Peter H. Dana's website-
            http://www.utexas.edu/depts/grg/gcraft/notes/datum/elist.html
            Department of Geography, University of Texas at Austin
            Internet: pdana@mail.utexas.edu . Default is ``23`` constrained to 
            WGS84.
            
        Returns
        -------
        proj_point: tuple(lat, lon)
            projected point in lat and lon in Datum, as decimal degrees.
                        
        """
        if east is not None: 
            self.east = east 
        if north is not None: 
            self.north = north 
        
        if (self.east or self.north) is None : 
            raise SiteError(" Easting and northing must not be None")
            
        if utm_zone is not None : 
            self._utm_zone =utm_zone 
            
        self.datum = 'WGS84'
        self.datum = datum or self.datum 
        self.reference_ellipsoid = reference_ellipsoid or \
            self.reference_ellipsoid
                          
        try : 
            self.lat, self.lon  = utm_to_ll(
                reference_ellipsoid=self.reference_ellipsoid, 
                northing = self.north , 
                easting= self.east, 
                zone= self.utm_zone 
                                  ) 
        except : 
            self.lat , self.lon = project_point_utm2ll(
                easting = self.east,
                northing= self.north, 
                utm_zone = self.utm_zone, 
                datum= self.datum, 
                epsg= self.epsg 
                ) 
            
        return self.lat, self.lon 
        
    @staticmethod 
    def to_utm_in(
        lats:ArrayLike, 
        lons:ArrayLike, 
        *, 
        utm_zone:str =None, 
        datum: str=None, 
        **kws 
        ):
        """ Convert array of longitude and latitude to array of X, y UTM 
        coordinates. 
        
        Parameters 
        ------------
        lats: arraylike 1d, 
           Array composed of latitude values 
        lons: ArrayLike 1d, 
           Array composed of longitude values. 
 
        utm_zone: Optional, string
            zone number and 'S' or 'N' e.g. '55S'. Defaults to the centre point
            of the provided points, 
        datum: string
            well known datum ex. WGS84, NAD27, NAD83, etc.
            
        kws: dict, 
           Keywords argument passed to :meth:`~watex.site.Location.to_utm`. 
           
        Returns
        --------
        (easts, norths): Iterable object composed of easting and northing 
           coordinates. 
           
        .. versionadded:: 0.1.8 
        
        See Also
        ----------
        watex.site.Location.to_utm: 
            Convert longitude and latitude value to easting and northing 
            coordinates. 
            
        """
        
        emsg = ("longitude" if lons is None else 'latitude') if (
            lats is None or lons is None) else "Both longitude and latitude"
        
        if lats is None or lons is None: 
            raise TypeError (emsg) 
            
        _check_consistency_size(lats, lons)
        lats = np.array(lats ) ; lons = np.array (lons )
        easts = norths = np.zeros_like (lats , dtype = np.float64)
        for ii, (lat, lon) in enumerate (zip (lats, lons )) : 
            Loc = Location()
            x, y  = Loc.to_utm (lat, lon , utm_zone= utm_zone , **kws )
            easts[ii] = x ; norths [ii] =y 
    
        return  easts, norths 
    
            
    @staticmethod 
    def to_latlon_in(
        easts:ArrayLike, 
        norths:ArrayLike, 
        *, 
        utm_zone:str, 
        datum: str=None, 
        **kws 
        ):
        """ 
        Convert array of longitude and latitude to array of X, y UTM 
        coordinates. 
        
        Parameters 
        ------------
        lats: arraylike 1d, 
           Array composed of latitude values 
        lons: ArrayLike 1d, 
           Array composed of longitude values. 
 
        utm_zone: Optional, string
           zone number and 'S' or 'N' e.g. '55S'. Defaults to the centre point
           of the provided points, 
        datum: string
           well known datum ex. WGS84, NAD27, NAD83, etc.
            
        kws: dict, 
           Keywords argument passed to :meth:`~watex.site.Location.to_latlon`. 
           
        Returns 
        -------
        (lats, lons): Iterable object composed of latitude and longitude 
           coordinates. 
          
        .. versionadded:: 0.1.8 
        
        See Also
        ------------
        watex.site.Location.to_latlon: 
           Convert easting and northing value to latitude and  longitude 
           coordinates. 
            
        """
        
        emsg = ("easting" if easts is None else 'northing') if (
            easts is None or norths is None) else "Both easting and northing"
        
        if easts is None or norths is None: 
            raise TypeError (emsg) 
            
        _check_consistency_size(easts, norths)
        easts = np.array(easts ) ; norths = np.array (norths )
        lats_lons =[]
        for east, north in zip (easts, norths ) : 
           Loc = Location()
           lats_lons.append (
               Loc.to_latlon(east, north , utm_zone= utm_zone , **kws )
               )
           
        return tuple (zip ( *lats_lons))  
                   
Location.__doc__="""\
Location class

Assert, convert coordinates lat/lon , east/north  
to appropriate formats. 

Location does not follow the `scikit-learn` API in order to encompass the 
the syntax of :term:`pycsamt` and :term:`mtpy`. The latter follows the 
Generci Mapping Tool (`GMT <ttps://www.generic-mapping-tools.org>`__)  API 
format.   
  
Parameters  
--------------
lat: float or string (DD:MM:SS.ms)
    latitude of point
          
lon: float or string (DD:MM:SS.ms)
    longitude of point
    
east: float
    easting coordinate in meters
            
north: float
    northing coordinate in meters
     
datum: string
    well known datum ex. WGS84, NAD27, NAD83, etc.

utm_zone: Optional, string
    zone number and 'S' or 'N' e.g. '55S'. Defaults to the centre point
    of the provided points
               
epsg: Optional, int
    epsg number defining projection (see http://spatialreference.org/ref/ for moreinfo)
    Overrides utm_zone if both are provided

reference_ellipsoid: Optional, int 
    reference ellipsoids is derived from Peter H. Dana's website-
    http://www.utexas.edu/depts/grg/gcraft/notes/datum/elist.html
    Department of Geography, University of Texas at Austin
    Internet: pdana@mail.utexas.edu . Default is ``23`` constrained to 
    WGS84. 
"""     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        