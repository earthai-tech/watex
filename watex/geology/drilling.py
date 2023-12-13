# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Created on Thu Sep 29 08:30:12 2022 

from __future__ import print_function , annotations
import os 
import functools 
from warnings import warn 
import numpy as np 
import pandas as pd 

from .core import GeoBase 
from .geology import Geology
from .._typing import Any, NDArray, DataFrame
from ..decorators import export_data 
from ..exceptions import NotFittedError, DrillError
from ..site import Profile, Location 
from ..utils.box import Boxspace, data2Box 
from ..utils.coreutils import _is_readable, makeCoords 
from ..utils.exmath import get_azimuth 
from ..utils.funcutils import ( 
    _assert_all_types, 
    to_numeric_dtypes , 
    smart_strobj_recognition, 
    convert_value_in, 
    read_worksheets, 
    sanitize_frame_cols, 
    str2columns,
    key_search, 
    ellipsis2false, 
    exist_features, 
    smart_format, 
    )
from ..utils.geotools import ( 
    build_random_thickness, 
    smart_thickness_ranker
    )
from ..utils.validator import check_array 

__all__=[
    "DSBoreholes",
    "DSBorehole" ,
    "DSDrill", 
    "Drill", 
    "Borehole", 
    "PumpingTest"
]

class DSBoreholes:
    """
    Class deals with many boreholes dataset. 
    
    DSBoreholes works with the data set composed of multiple borehole data. 
    The data columns are the all attributes of the object and any 
    non-alphateic character is by ``_``. For instance, a column name 
    ``layer thickness`` should have an attribute named ``layer_thickness``. 
    Each borehole (row) data become its own object which encompasses all 
    columns as attributes. To have full control of how data must be 
    retrieved, ``holeid`` parameter must be set. For instance, to retrieve 
    the borehole with ID equals to `bx02`, after fitting the class with 
    appropriate parameters, attibute `hole depth` ( if exist in the data) can  
    be retrieved as ``self.hole.bx02.hole_depth``. 
    
    By default if the projection is given as latitude/longitude 

    Parameters
    ------------
    area: str
       Name of area where the data collection is made. 
      
    holeid: str, optional 
       The name of column of the boreholes collections ID. Note that if 
       given, it must exist in the borehole datasets. Note that if 
       hole ID is not specified, each borehole can be fetched from a attribute
       hole count from 0 to n_samples. For instance, the borehole number 12
       can be collected using:: 
           
           >>> b = DSBoreholes ().fit(<borehole_data>)
           >>> b.hole.hole11 
           
      where 12 is the 12em position as Python index starts with 0. 
      However when holeid is specified, the `hole` attribute is replaced 
      by each value of the `hole_id` column as: 
          
          >>> borehole_data['hole_id'][:3]
          0    B0092
          1    B0093
          2    B0094
          Name: hole_id, dtype: object
          >>> b.hole.B0092 
          >>> b.hole.B0094 
          
      where ``B0092`` or ``B0094`` are the borehole in the columns ``hole_id``. 
      Note that ``hole_id`` can be any other names at least it is explicitly 
      specified as a argument of the ``holeid` parameter. 
      
    lon, lat: ArrayLike 1d /str  , optional 
       One dimensional arrays. `xlon` can be consider as the abscissa of   
       the landmark and `ylat` as ordinates array.  If `xlon` or `ylat` is  
       passed as string argument, `data` must be passed as `fit_params` 
       keyword arguments and the name of `xlon` and `y` must be a column 
       name of the `data`. 
       By default `xlon` and `ylat` are considered as `longitude` and 
       `latitude` when ``dms`` or ``ll`` coordinate system is passed.
       
    utm_zone: Optional, string
       zone number and 'S' or 'N' e.g. '55S'. Default to the centre point
       of coordinates points in the survey area. It should be a string (##N or ##S)
       in the form of number and North or South hemisphere, 10S or 03N
       
    projection: str, ['utm'|'dms'|'ll'] 
       The coordinate system in which the data points for the profile is collected. 
       If not given, the auto-detection will be triggered and find the  suitable 
       coordinate system. However, it is recommended to provide it for consistency. 
       Note that if `x` and `y` are composed of value less than 180 degrees 
       for longitude and 90 degrees for latitude, it should be considered as  
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

    encoding: str, default ='utf8'
       Default encoding for parsing data. Can also be ['utf-16-be'] for 
       reading bytes characters. 
       
    interp_coords: bool, default=False 
       Interpolate position coordinates.
      
    reference_ellipsoid: int, default=23 
       reference ellipsoids is derived from Peter H. Dana's website-
       http://www.utexas.edu/depts/grg/gcraft/notes/datum/elist.html
       Department of Geography, University of Texas at Austin
       Internet: pdana@mail.utexas.edu . Default is ``23`` constrained to 
       WGS84. 
       
    verbose: int, default=0 
       Output messages. 
      
    Attributes 
    ----------
    lon_, lat_: Arraylike, 
       longitude/latitude of coordinates arrays. 
       
    `hole.<holeid>.<data_column>`: :class:`~watex.utils.box.Boxspace` 
       Each borehole, commonly which ID correspond to  each row. Each row
       can be fetched as 'holeID'. If `holeid` is nt specified, the string 
       literal `hole+index of data` composed the borehole object. 
       
    Notes 
    ------
    When `data` is supplied and `lon` and `lat` are given by their names 
    existing in the dataframe columns, by default, the non-numerical 
    data are removed. However, if `y` and `x` are given in DD:MM:SS in 
    the dataframe, the coordinate system must explicitly set to ``dms`
    to keep the non-numerical values in the data. 
    
    Examples
    ---------
    >>> import watex as wx 
    >>> from watex.geology import DSBoreholes 
    >>> bs_data = wx.fetch_data ('nlogs', key='hydro', samples=12 ,
                                 as_frame=True )
    >>> bs=DSBoreholes ().fit(bs_data)
    >>> bs.holeid
    Out[61]: 'hole'
    >>> # when the default object hole is set as:
    >>> bs.hole # outputs a Boxspace object each borehole can be retrieved 
    >>> # as hole object count from 0. to number or rows -1. Here is an 
    >>> example of fetching the hole 11. 
    >>> bs.hole.hole10
    Out[62]:
    {'hole_id': 'B0103',
     'uniform_number': 1.1343e+16,
     'original_number': 'Guangzhou multi-element urban geological survey drilling 19ZXXSW11',
     'lon': '113:43:00.99',
     'lat': '23:16:17.23',
     'longitude': 113.71694166666668,
     'latitude': 23.271452777777775,
     'east': 2577207.0,
     'north': 19778060.0,
     'easting': 2577207.276,
     'northing': 19778177.29,
     'coordinate_system': 'Xian 90',
     'elevation': 22.0,
     'final_hole_depth': 60.1,
     'quaternary_thickness': 45.8,
     'aquifer_thickness': 18.1,
     'top_section_depth': 42.0,
     'bottom_section_depth': 60.1,
     'groundwater_type': 'igneous rock fissure water',
     'static_water_level': 2.36,
     'drawdown': 28.84,
     'water_inflow': 0.08,
     'unit_water_inflow': 0.003,
     'filter_pipe_diameter': 0.16,
     'water_inflow_in_m3_d': 2.94}
    >>> # when we specified the hole ID to the column that compose the ID like: 
    >>> bs=DSBoreholes (holeid ='hole_id').fit(bs_data)
    >>> bs.hole.B0103
    Out[63]:
    {'hole_id': 'B0103',
     'uniform_number': 1.1343e+16,
     'original_number': 'Guangzhou multi-element urban geological survey drilling 19ZXXSW11',
     'lon': '113:43:00.99',
     'lat': '23:16:17.23',
     'longitude': 113.71694166666668,
     'latitude': 23.271452777777775,
     'east': 2577207.0,
     'north': 19778060.0,
     'easting': 2577207.276,
     'northing': 19778177.29,
     'coordinate_system': 'Xian 90',
     'elevation': 22.0,
     'final_hole_depth': 60.1,
     'quaternary_thickness': 45.8,
     'aquifer_thickness': 18.1,
     'top_section_depth': 42.0,
     'bottom_section_depth': 60.1,
     'groundwater_type': 'igneous rock fissure water',
     'static_water_level': 2.36,
     'drawdown': 28.84,
     'water_inflow': 0.08,
     'unit_water_inflow': 0.003,
     'filter_pipe_diameter': 0.16,
     'water_inflow_in_m3_d': 2.94}
    >>> # each columns can be fetched as 
    >>> bs.quaternary_thickness
    Out[64]: 
    0     40.5
    1     12.3
    2     25.5
    3     40.0
    4     35.0
    5     47.0
    6     34.0
    7     40.4
    8     15.1
    9     17.2
    10    45.8
    11    47.0
    Name: quaternary_thickness, dtype: float64
    """
    def __init__(
        self, 
        area:str=None,
        holeid:str=None,
        lat:str=None, 
        lon:str=None, 
        projection:str ='ll', 
        utm_zone:str=None, 
        datum:str='WGS84', 
        epsg:int=None, 
        encoding:str='utf-8', 
        interp_coords:bool=False, 
        reference_ellipsoid:int=23, 
        verbose:bool=False 
        ): 
        
        self.area =area 
        self.holeid=holeid  
        self.projection= projection 
        self.utm_zone=utm_zone 
        self.reference_ellipsoid= reference_ellipsoid 
        self.datum=datum 
        self.encoding= encoding 
        self.epsg =epsg 
        self.interp_coords=interp_coords
        self.lon=lon 
        self.lat=lat 
        self.verbose= verbose 
            
    def fit ( self, data, **fit_params): 
        """ Fit Hole data set and populate attributes. 
        
        Parameters 
        ----------
        data: Path-like Object or DataFrame 
          Hole data. 
          
        fit_params: dict,
          Keyword arguments passed to :func:`watex.to_numeric_dtypes` to 
          sanitize the data. 
          
        Return 
        ------
        self: :class:`DSBoreholes`
          Instanced object for chaining methods. 
          
        """
        columns = fit_params.pop ("columns", None  )
        data = _is_readable(data, as_frame =True, 
                            input_name= 'b', 
                            columns = columns, 
                            encoding =self.encoding 
                            )
        
        data = check_array (
            data, 
            force_all_finite= "allow-nan", 
            dtype =object , 
            input_name="Boreholes data", 
            to_frame=True, 
            )
        self.lon_=None; self.lat_=None 
        
        if ( self.lon is not None 
            and self.lat is not None
            ): 
            p = Profile (utm_zone = self.utm_zone , 
                         coordinate_system= self.projection, 
                         datum= self.datum , 
                         epsg= self.epsg, 
                         reference_ellipsoid=self.reference_ellipsoid 
                         ) 
            p.fit (x = self.lon, y = self.lat, data = data ) 
  
            if self.interp_coords: 
               p.interpolate ()
               
            self.lon_= p.x 
            self.lat_= p.y 
            
        # For consistency, Check the datatype, sanitize columns 
        # and drop all NaN columns and row values
        data, nf, cf = to_numeric_dtypes(
            data , 
            return_feature_types= True, 
            verbose =self.verbose, 
            sanitize_columns= True, 
            fill_pattern='_', 
            **fit_params 
            )

        self.feature_names_in_ = nf + cf 
        
        if len(cf )!=0:
            # sanitize the categorical values 
            for c in cf : 
                data[c] = data[c].str.strip() 
            
        for name in data.columns : 
            setattr (self, name, data[name])
            
        # set depth attributes 
        if 'depth'  in self.feature_names_in_: 
            self.depth_= data['depth']
            
        self.data_ = data.copy() 
        
        use_col =False 
        if self.holeid is not None: 
        # Manage the key search to find it in the data frame 
        # columns the the corresponding key in data columns 
            use_col = True 
        else: self.holeid ='hole'
            
        self.hole = data2Box ( 
            self.data_ , 
            name =self.holeid, 
            use_colname= use_col
                      )
        
        return self  
    
    def set_coordinates (
        self, 
        reflong, 
        reflat,  
        step ='5m', 
        todms=False, 
        r= 45, 
        **kws
         ): 
        """ Generate longitude and latitude coordinates for boreholes. 
        
        It assumes boreholes are aligned along the same axis. 
     
        Parameters 
        -----------
        reflong: float or string or list of [start, stop]
            Reference longitude  in degree decimal or in DD:MM:SS for 
            the first site considered as the origin of the landmark.
            
        reflat: float or string or list of [start, stop]
            Reference latitude in degree decimal or in DD:MM:SS for the 
            reference site considered as the landmark origin. If value is 
            given in a list, it can containt the start point and the 
            stop point. 
            
        step: float or str 
            Offset or the distance of seperation between different sites 
            in meters. If the value is given as string type, except 
            the ``km``, it should be considered as a ``m`` value. Only 
            meters and kilometers are accepables.
            
        r: float or int 
            The rotate angle in degrees. Rotate the angle features 
            toward the direction of the projection profile. 
            Default value use the :meth:`~.bearing` value in degrees. 
               
        todms: bool, Default=False
            Reconvert the longitude/latitude degree decimal values into 
            the DD:MM:SS. 
     
        kws: dict, 
           Additional keywords of :func:`~watex.utils.exmath.makeCoords`.   
           
        Returns 
        --------
        self: Instanced object 
        
          Instanced object for method chaining.
          
        Examples
        ---------
        >>> bs_data = wx.fetch_data ('nlogs', key='ns', samples=7,
                                     as_frame=True )
        >>> bs=DSBoreholes ().fit(bs_data)
        >>> bs.set_coordinates(reflong= 113.4, reflat=22.56, step ='10m')
        >>> bs.set_coordinates(reflong= 113.4, reflat=22.56, step ='10m')
        >>> bs.lat_
        Out[71]: 
        array([22.56      , 22.56009391, 22.56018782, 22.56028174, 22.56037565,
               22.56046956, 22.56056347])
        >>> bs.lon_
        Out[72]: 
        array([113.4       , 113.40007436, 113.40014871, 113.40022307,
               113.40029742, 113.40037178, 113.40044614])
        """
        self.inspect
        
        nsites = len(self.data_ )
        isutm = False if self.projection =='ll' else True 
        utm_zone =  kws.pop ('utm_zone', None ) or self.utm_zone 
        
        self.lon_, self.lat_= makeCoords(
            reflong, 
            reflat, 
            nsites =nsites, 
            r= r ,  
            step =step , 
            todms=todms, 
            utm_zone= utm_zone, 
            is_utm= isutm, 
            datum=self.datum, 
            espg=self.epsg,
            **kws
            ) 
        
        return self    
        
    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        _t = ("area", "holeid", "lat", "lon", "projection", "utm_zone", 
              "encoding", "datum", "epsg", "reference_ellipsoid" ,
              "interp_coords", "verbose")

        outm = ( '<{!r}:' + ', '.join(
            [f"{k}={ False if getattr(self, k)==... else  getattr(self, k)!r}" 
             for k in _t]) + '>' 
            ) 
        return  outm.format(self.__class__.__name__)
       
    
    def __getattr__(self, name):
       rv = smart_strobj_recognition(name, self.__dict__, deep =True)
       appender  = "" if rv is None else f'. Do you mean {rv!r}'
       
       err_msg =  f'{appender}{"" if rv is None else "?"}' 
       
       raise AttributeError (
           f'{self.__class__.__name__!r} object has no attribute {name!r}'
           f'{err_msg}'
           )

    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'hole'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1  
    
class DSBorehole: 
    """ Class deals with Borehole datasets. 
    
    :class:`watex.geology.drilling.DSBorehole` works with data collected in 
    a single borehole. For instance, it could follow the arrangement of 
    ``h502`` data in :func:`watex.datasets.load_hlogs`
    
    Parameters
    ------------
    hole: str
       Name or ID of the borehole. 
      
    dname: str, optional 
       Depth column name. If `depth` is specify an attribute `depth_` should 
       be created. Depth specification is usefull for log plotting of machine
       training.
       
    utm_zone: Optional, string
       zone number and 'S' or 'N' e.g. '55S'. Default to the centre point
       of coordinates points in the survey area. It should be a string (##N or ##S)
       in the form of number and North or South hemisphere, 10S or 03N
       
    projection: str, ['utm'|'dms'|'ll'] 
       The coordinate system in which the data points for the profile is collected. 
       If not given, the auto-detection will be triggered and find the  suitable 
       coordinate system. However, it is recommended to provide it for consistency. 
       Note that if `x` and `y` are composed of value less than 180 degrees 
       for longitude and 90 degrees for latitude, it should be considered as  
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
        
    encoding: str, default ='utf8'
       Default encoding for parsing data. Can also be ['utf-16-be'] for 
       reading bytes characters. 
       
    lonlat: Tuple, Optional 
       longitude/latitude for borehole coordinates. The location where the 
       borehole is performed. 
      
    verbose: int, default=0 
       Output messages. 
      
    Attributes
    -----------
    depth_: Series 
       Depth array if `dname` is specified. 
    data_: Pandas DataFrame
       Sanitized dataframe. 
    feature_names_in_: list
      Name of features that composes the dataset. 
      
    Note 
    ------
    Each columns of the dataframe is an attribute. Note that all the non-
    alphabetic letters is removed and replace by '_'. 
    
    Examples
    ----------
    >>> import watex as wx 
    >>> from watex.geology import DSBorehole 
    >>> hdata= wx.fetch_data ('hlogs',samples = 12 ).frame
    >>> b = DSBorehole (hole='H502').fit(hdata)
    >>> b.feature_names_in_
    Out[77]: 
    ['depth_top',
     'depth_bottom',
     'layer_thickness',
     'resistivity',
     'gamma_gamma',
     'natural_gamma',
     'sp',
     'short_distance_gamma',
     'well_diameter',
     'aquifer_thickness',
     'hole_depth_before_pumping',
     'hole_depth_after_pumping',
     'hole_depth_loss',
     'depth_starting_pumping',
     'pumping_depth_at_the_end',
     'pumping_depth',
     'section_aperture',
     'k',
     'kp',
     'r',
     'rp',
     'hole_id',
     'strata_name',
     'rock_name',
     'aquifer_group',
     'pumping_level']
    >>> b.strata_name
    Out[78]: 
    0                       topsoil
    1                        gravel
    2                      mudstone
    3                     siltstone
    4                      mudstone
              
    176                        coal
    177                   siltstone
    178    coarse-grained sandstone
    179      fine-grained sandstone
    180    coarse-grained sandstone
    Name: strata_name, Length: 181, dtype: object
    """
    def __init__ (
        self,
        hole:str=None, 
        dname: str=None,
        projection:str='ll', 
        utm_zone:str=None, 
        datum:str ='WGS84', 
        epsg:int=None, 
        reference_ellipsoid:int=23,
        encoding:str ='utf-8', 
        lonlat:tuple =None, 
        verbose:int= 0, 
        ): 
        
        self.hole=hole
        self.dname=dname 
        self.projection= projection 
        self.utm_zone=utm_zone 
        self.reference_ellipsoid= reference_ellipsoid 
        self.datum=datum 
        self.encoding= encoding
        self.epsg =epsg 
        self.verbose= verbose 
   
    def fit(self, data,  **fit_params ):
        """ Fit Borehole data and populate attribute data. 
        
        By default if the projection is given as latitude/longitude 
        xlon, ylat are longitude and latitude respectively. 
 
        Parameters
        ------------
        data: pd.DataFrame or Path-like object. 
           Data containing the borehole informations. 
           
        fit_params: dict, 
           Keyword arguments passed to :func:`watex.utils.to_numeric_dtypes`
           for data management. 
           
        Return 
        ---------
        self : Instanced object 
            Instanced object for chaining method. 
            
        """
        _fit( self, data = data, drop_nan_columns=False,
             ** fit_params  )
        
        return self 
    
    def set_depth ( self, z0=0. , max_depth =None, reset_depth: bool =...): 
        """Set the a random depth if depth is not supplied in the Borehole 
        data
        
        To fetch the depth, use attribute `depth_`. Note that if the depth 
        exists, calling `set_depth` will erase the former depth value. Use 
        in cautioness. 
        
        Parameters 
        -----------
        z0: float, default=0.
         The surface reference. Preferably, it is set to null. 
         
        max_depth: float, default=700. 
          The maximum depth. Depth size must fit the length of the data in 
          meters. Default depth is fixed to 700 meters. 
          
        reset_depth: bool, default =False, 
          An alternative way to controle whether to keep the previous and 
          new computed depth in the borehole data. The parameter erases the
          previous depth if exists the borehole data. If ``True`` a new depth 
          is set in replacement to the previous depth and both are kept in the 
          data otherwise. 
          
        Return
        -------
        self: Instanced object 
            Instanced object for chaining method. 
            
        Examples 
        -------- 
        >>> import watex as wx 
        >>> from watex.geology import DSBorehole 
        >>> hdata= wx.fetch_data ('hlogs').frame
        >>> b = DSBorehole (hole='H502').fit(hdata)
        >>> b.set_depth () 
        >>> b.depth_
        Out[82]: 
        0        0.000000
        1        3.888889
        2        7.777778
        3       11.666667
        4       15.555556
           
        176    684.444444
        177    688.333333
        178    692.222222
        179    696.111111
        180    700.000000
        Name: depth, Length: 181, dtype: float64
        >>> b.set_depth (max_depth = 900, reset_depth= True )
        >>> b.depth_
        Out[85]: 
        0        0.0
        1        5.0
        2       10.0
        3       15.0
        4       20.0
         
        176    880.0
        177    885.0
        178    890.0
        179    895.0
        180    900.0
        Name: depth, Length: 181, dtype: float64
        """
        if reset_depth is ...: reset_depth =False 
        
        check_results = self._check_object_in(
            'depth', reset_depth )
            
        if check_results =="objectexists": 
            return self 
        
        self._set_depth ( z0=z0 , max_depth = max_depth )
        return self 

    def _set_depth ( 
        self ,z0=0.,  max_depth =None, 
        ): 
        """ Set the a random depth if depth is not given.
        
        An Isolated part of the :meth:`set_depth`

        Parameters 
        -----------
        z0: float, default=0.
         The surface reference. Preferably, it is set to null. 
         
        max_depth: float, default=700. 
          The maximum depth. Depth size must fit the length of the data in 
          meters. Default depth is fixed to 700 meters. 
          
        Return
        -------
        self: Instanced object 
            Instanced object for chaining method. 
        
        """
        self.inspect
   
        z0 = convert_value_in (z0 )
        max_depth =  max_depth or 700.
        max_depth = float( _assert_all_types ( max_depth, int, float, 
                                              objname = 'Maximum-depth')) 
 
        self.depth_ = pd.Series ( np.linspace ( z0, max_depth, len(self.data_) 
                                               ), name ='depth')
        # append depth data 
        # self.data_.insert (0 , 'depth', self.depth_, allow_duplicates =True)
        d= pd.concat ([ self.depth_, self.data_ ], axis = 1, 
                                ignore_index =True )
        # for consistency reset columns names
        d.columns = [self.depth_.name] + list(self.data_.columns)
        self.data_ =d.copy() 
        
        return self 
    
    def _check_object_in ( self, name, reset_obj:bool= ... , warn_msg:bool=...
                          ): 
        """ Check  object in the Borehole data and remove if object exists 
        provided that `reset_obj` is set to ``True``. """ 
        
        reset_obj, warn_msg = ellipsis2false(reset_obj, warn_msg)
 
        if ( hasattr ( self, name + '_' ) 
            and name in self.data_.columns 
            ) : 
            if not reset_obj: 
                # obj_name = name[:-1] if name.endswith ('_') else name 
                
                msg = (
                    f"{name.title()!r} object already exists in borehole"
                    f" data. To set a new {name}, turn `reset_{name}`"
                     " to ``True``.")
                warn(msg) if not warn_msg else warn(warn_msg)
                
                # for consistency reset value if None 
                if  getattr (self, name + '_') is None: 
                    setattr  (self, name + '_', self.data_[name] ) 
                    
                return "objectexists" 
        
            try: 
                self.data_.drop (columns = name , inplace =True, axis =1 )
            except KeyError: 
                warn(f"{name!r} does no longer exist in the borehole data."
                     " Check the data column names.")

    def set_thickness ( 
        self,
        h0= 1 , 
        shuffle: bool = True,
        dirichlet_dist: bool=...,
        reset_layer_thickness: bool=...,
        reset_depth: bool=..., 
        **kws
         ): 
        """
        Generate a random layer thickness from borehole refering to the 
        depth.
        
        To fetch the thickness, use attribute `layer_thickness_`. Use
        `reset_layer_thickness` to set new strata thicknesses. 

        Parameters 
        -----------
        h0: int, default='1m' 
          Thickness of the first layer. 
          
        shuffle: bool, default=True 
          Shuffle the random generated thicknesses. 
          
        dirichlet_dis: bool, default=False 
          Draw samples from the Dirichlet distribution. A Dirichlet-distributed 
          random variable can be seen as a multivariate generalization of a 
          Beta distribution. The Dirichlet distribution is a conjugate prior 
          of a multinomial distribution in Bayesian inference.
   
        reset_layer_thickness: bool, default=False, 
          Set new layer thicknesses to the existing stratum. If ``True`` and 
          the data included layer thicknesses, the latter should be dropped in 
          replacement to the new ones. However, if False, no action is performed 
          and both are kept in the data. 
          
        reset_depth: bool, default=False 
          Note that thickness generating works with the depth. So, if
          the `reset_depth` is set to ``True``, a new depth is computed and 
          drop the former ones. From this new depth, the thickness generating 
          is creating. 
          
        random_state: int, array-like, BitGenerator, np.random.RandomState, \
             np.random.Generator, optional
          If int, array-like, or BitGenerator, seed for random number generator. 
          If np.random.RandomState or np.random.Generator, use as given.
          
        unit: str, default='m' 
          The reference unit for generated layer thicknesses. Default is 
          ``meters``
        
        Return
        -------
        self: Instanced object 
            Instanced object for chaining method. 
            
        Examples 
        ----------
        >>> import watex as wx 
        >>> from watex.geology import DSBorehole 
        >>> hdata= wx.fetch_data ('hlogs').frame
        >>> b = DSBorehole (hole='H502').fit(hdata)
        >>> b.set_thickness () 
        >>> b.layer_thickness_ 
        0      5.410380
        1      2.068812
        2      0.398028
        3      6.352873
        4      6.395714
          
        176    3.396871
        177    0.012463
        178    7.124004
        179    7.038323
        180    3.439711
        Name: layer_thickness, Length: 181, dtype: float64
        >>> b.set_thickness (dirichlet_dist=True, reset_layer_thickness=True 
                             ).layer_thickness_
        Out[89]: 
        0       0.681640
        1       1.986043
        2       6.413090
        3       5.305284
        4       0.000144
           
        176     4.119242
        177    12.161252
        178     1.809102
        179     0.408810
        180     4.281848
        Name: layer_thickness, Length: 181, dtype: float64
        """
        dirichlet_dist, reset_layer_thickness,reset_depth = ellipsis2false(
            dirichlet_dist, reset_layer_thickness,reset_depth )
        
        check_results = self._check_object_in(
            'layer_thickness', reset_layer_thickness )
            
        if check_results =="objectexists": 
            return self 
        
        self._set_thickness ( h0= h0 , 
        dirichlet_dist=dirichlet_dist,
        shuffle = shuffle,
        reset_depth=reset_depth, 
        **kws
        )

        return self 
    
    def _set_thickness(
        self, 
        h0= 1 , 
        dirichlet_dist=False,
        shuffle = True,
        reset_depth: bool=..., 
        **kws 
        ): 
        """ Set a random layer thickness from borehole refering to the depth.
        
        An isolated part of :meth:`set_thickness`. 

        Parameters 
        -----------
        h0: int, default='1m' 
          Thickness of the first layer. 
          
        shuffle: bool, default=True 
          Shuffle the random generated thicknesses. 

        dirichlet_dis: bool, default=False 
          Draw samples from the Dirichlet distribution. A Dirichlet-distributed 
          random variable can be seen as a multivariate generalization of a 
          Beta distribution. The Dirichlet distribution is a conjugate prior 
          of a multinomial distribution in Bayesian inference.
          
        random_state: int, array-like, BitGenerator, np.random.RandomState, \
             np.random.Generator, optional
          If int, array-like, or BitGenerator, seed for random number generator. 
          If np.random.RandomState or np.random.Generator, use as given.
          
        z0: float, default=0.
         The surface reference. Preferably, it is set to null. 
         
        depth: float, default=700. 
          The maximum depth. Depth size must fit the length of the data in 
          meters. Default depth is fixed to 700 meters.
          
        unit: str, default='m' 
          The reference unit for generated layer thicknesses. Default is 
          ``meters``
        
        Return
        -------
        self: Instanced object 
            Instanced object for chaining method. 
        
        """
        self.inspect 
        
        if self.depth_ is None:
            # construct depath 
            self.set_depth (reset_depth = reset_depth ) 
        
        
        thickness = build_random_thickness  (
            self.depth_ , h0= h0 , 
            dirichlet_dist= dirichlet_dist, 
            shuffle = shuffle, 
            **kws
            )
        
        self.layer_thickness_= pd.Series (thickness, name='layer_thickness' )
        
        d= pd.concat ([ self.data_,  self.layer_thickness_], 
                                axis = 1, ignore_index =True )
        # for consistency reset columns names
        d.columns =  list(self.data_.columns) + [self.layer_thickness_.name] 
        self.data_ =d.copy() 
        
        return self 
    
    def set_strata (self, add_electrical_properties :bool=False, 
                    random_state =None, shuffle :bool=True , 
                    reset_strata :bool=... , 
                    reset_electrical_properties :bool=... 
                    ): 
        """ Generate a pseudo strata associated to each depth in the borehole
        data. 
        
        Parameters 
        ----------
        add_electrical_properties: bool, default=False, 
          Add electrical resistivty values associated to each generated stratum
          
        random_state: int, array-like, BitGenerator, np.random.RandomState, \
             np.random.Generator, optional
          If int, array-like, or BitGenerator, seed for random number generator. 
          If np.random.RandomState or np.random.Generator, use as given.
          
        shuffle: bool, default=True 
          Shuffle the random generated thicknesses. 
          
        reset_strata: bool, default=False, 
          generate new strata at each depth. If ``True`` and the name `strata` 
          is valid in the borehole data colum name. Layer names that composes 
          each stratum should be erased. 

        reset_electrical_properties: bool, default=False 
          Erase the former electrical values and replace by new names that 
          fit each strata. 

        Return
        -------
        self: Instanced object 
            Instanced object for chaining method. 
            
        Examples 
        --------
        >>> import watex as wx 
        >>> from watex.geology import DSBorehole 
        >>> hdata= wx.fetch_data ('hlogs', key='h803').frame
        >>> b = DSBorehole (hole='H803').fit(hdata)
        >>> b.set_strata () 
        >>> b.strata_
        Out[122]: 
        0                tourmalinite
        1                        silt
        2                         mud
        3         volcaniclastic rock
        4                ore minerals
                 
        129    sulphide-rich material
        130                 argillite
        131                  graphite
        132            high-Mg basalt
        133                     shale
        Name: strata, Length: 134, dtype: object
        >>> b.set_strata (add_electrical_properties= True, reset_strata= True)
        >>> b.strata_
        Out[123]: 
        0              phyllite
        1               syenite
        2              laterite
        3             saprolite
        4          psammopelite
              
        129               chert
        130           granulite
        131    pyroclastic rock
        132         lamprophyre
        133          ignimbrite
        Name: strata, Length: 134, dtype: object

        b.strata_electrical_properties_
        Out[124]: 
        0        0.0
        1        0.0
        2        0.0
        3      330.6
        4        0.0
         
        129      0.0
        130      0.0
        131      0.0
        132      0.0
        133      0.0
        Name: strata_electrical_properties, Length: 134, dtype: float64
        >>> 
        """
        self.inspect 
        
        reset_strata, reset_electrical_properties= ellipsis2false(
            reset_strata, reset_electrical_properties)
        
        for attr, action  in zip ( ('strata', 'strata_electrical_properties'), 
                                  ( reset_strata, reset_electrical_properties)
                                  ): 
            check_results = self._check_object_in(
                attr, action )
            if check_results =="objectexists": 
                return self 
            
        self._set_strata (add_electrical_properties = add_electrical_properties, 
                      random_state = random_state, shuffle =shuffle , 
                      )
        
        return self 
    
    def _set_strata (self, add_electrical_properties =False, 
                    random_state =None, shuffle =True 
                    ): 
        """ Create strata associated to each depth.  
        An isolated part of :meth:`set_strata`. 
        """
        self.inspect 
        # use default columns [electrical, _description] properties 
        e_props, strata = GeoBase.getProperties() 
        # compute the mean with electrical properties 
        if add_electrical_properties: 
            e_props = list (map ( lambda x : np.mean ( x ) if hasattr (
                x, '__iter__') else x , e_props ))
    
            e_props = np.array(e_props ) 
        strata= np.array(strata )
        if shuffle: 
            ixs = np.random.permutation (
                np.arange ( len(e_props)))
            if add_electrical_properties: 
                e_props = e_props [ixs ]
                
            strata  = strata [ixs ]
        # get the selected part  
        if random_state: 
                np.random.seed (random_state )
        #shuffle again 
        if shuffle: 
            ix = np.random.permutation (
                np.arange ( len(self.data_)))
        else: ix = np.arange ( len(self.data_))
        strata= strata[ix ]
        if add_electrical_properties: 
            e_props = e_props [ix ]
            
        self._set_info_in(name= 'strata', values= strata)
        if add_electrical_properties: 
            self._set_info_in(name= 'strata_electrical_properties',
                              values= e_props)
    
        return self 

    def _set_info_in (self,  name , values , insert_index =None ): 
        """ Setup new information as an attribute and data into the data """
        # if series is given 
        if not hasattr ( values, 'name'): 
            values = pd.Series ( values, name= name  )
        setattr (self, name + '_', values )
        
        # add new attribute to the data 
        if insert_index is None: 
            d = pd.concat ( [self.data_, getattr ( self, name + '_')], 
                           axis =1 , ignore_index =True)
            # for consistency 
            d.columns = list(self.data_.columns) + [name] 
            self.data_ = d.copy() 
        else: 
            self.data_.insert (insert_index, 
                               column =name, value =values.values )
        return self 

    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        _t = ("hole", "dname", "projection", "utm_zone", "encoding", "datum", 
              "epsg", "reference_ellipsoid" , "verbose")
        outm = ( '<{!r}:' + ', '.join(
            [f"{k}={ False if getattr(self, k)==... else  getattr(self, k)!r}" 
             for k in _t]) + '>' 
            ) 
        return  outm.format(self.__class__.__name__)
       
    
    def __getattr__(self, name):
       rv = smart_strobj_recognition(name, self.__dict__, deep =True)
       appender  = "" if rv is None else f'. Did you mean {rv!r}'
       
       err_msg =  f'{appender}{"" if rv is None else "?"}' 
       
       raise AttributeError (
           f'{self.__class__.__name__!r} object has no attribute {name!r}'
           f'{err_msg}'
           )

    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        if not hasattr (self, 'data_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1

class _drill_builder: 
    """Decorator to handle the drillhole properties. 
    
    Decorator builds the collar, geology, geochemistry samples as well as the 
    elevation. When drillholes property is unknwow, set the ``mode=get`` 
    applied to a wrapper that accepts only `kind` as parameter to get the 
    valid properties. For instance, fetching the geochemistry property 
    attributes should be::
        
        >>> from watex.geology.drilling import _drill_builder
        >>> def prop_wrapper (kind): 
               return kind 
        >>> _drill_builder (mode ='get')(prop_wrapper) ('samples')
        Out[1]: 
        ('DH_Hole',
         'DH_East',
         'DH_North',
         'DH_RH',
         'DH_From',
         'DH_To',
         'Sample',
         'Mask')
    
    Parameters 
    -----------
    kind: str, ['collar'| 'geology'| 'samples'|'elevation']
       Kind of drillhole data to build 
    
    mode: str , ['build'|'fit'|'get'], default='build'
       Action to perform.  
       - `build` constructs drillhole manually 
       - `fit` contructs drillhole from the fitted data 
       - `get` outputs a drillhole property attributes. 
       
    Returns
    --------
    well/hole data: pd.DataFrame
       Well/hole, geology and samples data set constructed referring to the 
       Oasis montaj property drillhole codes.  
       
    Note 
    ------
    The drillhole property follows the `Oasis montaj` software module
    `drillhole`. See more about the Oasis Montaj from Seequent here: 
        https://www.seequent.com/products-solutions/geosoft-oasis-montaj/extensions/
        
    Examples 
    ----------
    >>> from watex.geology.drilling import _drill_builder
    >>> from watex.datasets import load_nlogs 
    >>> sample_data = load_nlogs( key='ns', as_frame=True ) 
    >>> def get_props (kind ): 
           return kind 
    >>> props= _drill_builder (mode ='get')( get_props )
    >>> props ('collar')
    ('DH_Hole',
     'DH_East',
     'DH_North',
     'DH_RH',
     'DH_Top',
     'DH_Bottom',
     'DH_Dip',
     'Elevation',
     'DH_Azimuth',
     'DH_PlanDepth',
     'DH_Decr',
     'Mask') 
    >>> props('geology') 
    Out[2]: ('DH_Hole', 'DH_East', 'DH_North', 'DH_RH', 'DH_From', 'DH_To', 'Rock', 'Mask')
    >>> # view the data columns  
    >>> sample_data.columns 
    Out[3]: 
    Index(['hole_id', 'year', 'drilling_type', 'longitude', 'latitude', 'easting',
           'northing', 'ground_height_distance', 'depth', 'opening_date',
           'end_date'],
          dtype='object')
    >>> sample_data [['hole_id', 'easting', 'northing', 'depth']].head(2) 
    Out[4]: 
      hole_id    easting    northing   depth
    0  NSGC01  2519305.0  19752788.0   58.75
    1  NSGC03  2517234.0  19755621.0  200.71
    >>> # construct wrapper function and fed to the decorator
    >>> def make_dh ( *dh_params) : 
           return dh_params   
    >>> # construct attributes data to make the properties. Note that `DH_` 
    >>> # prefixed to drillhole attribute can be skipped as: 
    >>> props = {"hole_id": "hole","easting":"east", "northing":"north",
                 "depth": "depth"}
    >>> fit_props= _drill_builder (mode='fit')( make_dh)
    >>> obj = type ( "do_nothing", (), {}) # make a fake object 
    >>> dh_data = fit_props (obj, sample_data,True, props, 'collar')
    >>> dh_data.head(2) 
    Out[5]: 
      DH_Hole    DH_East    DH_North DH_RH  ... DH_Azimuth DH_PlanDepth DH_Decr Mask
    0  NSGC01  2519305.0  19752788.0   NaN  ...        NaN        58.75     NaN  NaN
    1  NSGC03  2517234.0  19755621.0   NaN  ...        NaN       200.71     NaN  NaN
    [2 rows x 12 columns]
    
    >>> # Now turn to 'geology' the kind parameter to see.
    >>> dh_data = fit_props (obj, sample_data,True, props, 'geology')
    >>> dh_data.head(2) 
    Out[6]: 
      DH_Hole    DH_East    DH_North DH_RH DH_From DH_To Rock Mask
    0  NSGC01  2519305.0  19752788.0   NaN     NaN   NaN  NaN  NaN
    1  NSGC03  2517234.0  19755621.0   NaN     NaN   NaN  NaN  NaN
    """
    init_code= ("DH_Hole",	"DH_East",	"DH_North", "DH_RH" ) 
    code = Boxspace ( 
        # for collar
        collar= (
            "DH_Top", 
            "DH_Bottom",
            "DH_Dip", 
            "Elevation", 
            "DH_Azimuth",
            "DH_PlanDepth",	
            "DH_Decr",	
            "Mask"
            ),
        # for geology
        geology= (
            "DH_From",	
            "DH_To",
            "Rock",	
            "Mask"
            ),
        # for geochemistry sampling
        samples = (
            "DH_From",	
            "DH_To",	
            "Sample",		
            "Mask"
            ),
        # elevation 
        elevation=(
         'Elevation',
         'DH_RL',
         'DH_Dip'
            )
        )
    
    def __init__(
        self, 
        kind: str=None, 
        mode: str='build', 
        **kws ):

        self.kind=kind 
        self.mode=mode 

    def __call__(self, func  ): 
        self._func = func 
        
        @functools.wraps(self._func ) 
        def new_func (*args, **kwargs ):
            """Builder function. 
            Fetch data from the output of former function. 
            """
            self.kind_values = dict ( collar = self.code.collar , 
                                geology= self.code.geology , 
                                samples = self.code.samples, 
                                elevation= self.code.elevation 
                                )
            if str(self.mode).lower()=='build': 
                if self.kind not in ('collar', 'geology', 'samples', 
                                     'elevation'): 
                    raise DrillError (
                        "Wrong argument of kind. Expect ('collar', 'geology',"
                        f" 'samples','elevation'). Got {self.kind!r}") 
                    
                out = self._buildDH (*args, **kwargs)
            elif str(self.mode).lower()=='fit': 
                out= self._fitDH(*args, **kwargs)
                
            elif str(self.mode).lower() =='get': 
                self.kind = self._func (*args, **kwargs)  
                out= self.init_code +  self.kind_values.get(self.kind)
            
            return out 
        
        return new_func
    
    def _buildDH (self, *args, **kwargs): 
        """ Step by step construct drilling collar, geology and samples.
        
        Parameters
        -----------
        args: tuple, 
          Positional arguments of wrapped function. 
        kwargs: dict, 
           Keyword arguments of wrapped function. 
        
        Returns
        --------
        obj or data: Instanced object or pd.dataframe 
           returns Instanced object or a dataframe.
        """
        obj , values, return_data  = self._func (*args, **kwargs)  
        self._columns =self.init_code +  self.kind_values.get(self.kind)
        data = to_numeric_dtypes(
            values, columns =self._columns, 
            drop_nan_columns=False 
            )
        if ( 
            hasattr(obj, '_compute_azimuth') 
            and self.kind=='collar'
            and len(data)>1 
            ) : 
            try: 
                east, north = key_search(
                   'east north', 
                    default_keys=data.columns, 
                    deep=True, 
                    # ignore underscore '_' 
                    pattern ='[#&@!+,;\s-]\s*' 
                    )
                azim_value = get_azimuth(
                    data[east], 
                    data[north],
                    projection ='utm', 
                    extrapolate=True, 
                    )
                azimuth = key_search(
                   'azim', 
                    default_keys= data.columns, 
                    deep=True , 
                    pattern ='[#&@!+,;\s-]\s*'
                    )[0]
                data[azimuth]= azim_value 
            except BaseException as e: 
                if obj.verbose: warn( str(e))

        setattr (obj , self.kind +'_', data )
        return  getattr (obj, self.kind +'_') if return_data else obj 
        
    def _fitDH(self, *args, **kwargs): 
        """ Fit data and set the drillhole property values. 
        
        The property names must be included in the fitted data.
        
        Parameters
        -----------
        args: tuple, 
          Positional arguments of wrapped function. 
        kwargs: dict, 
           Keyword arguments of wrapped function. 
        
        Returns
        --------
        obj or data: Instanced object or pd.dataframe 
           returns Instanced object or a dataframe. 
        """
        
        obj, data, return_data, props, self.kind  = self._func (*args, **kwargs)
        self._columns =self.init_code +  self.kind_values.get(self.kind)
        # construct template for coll 
        d = pd.DataFrame (data = np.full ( ( len(data), len(self._columns)), 
                                          fill_value= np.nan, dtype=object),
                          columns = self._columns )
        if props is None: 
            warn("Missing properties. Can't fit the data values"
                 f" to the {self.kind} properties.")
            
        else: # get shrunked data with valid
            try: 
                exist_features(data, list (props.keys())  )
            except KeyError as key_error: 
                raise KeyError(str(key_error) + 
                               ". Missing property names in the data.")
                
            data_shrunked = data [ list(props.keys() )].copy() 
            # re-verify key to match the exact default _keys 
            pnames =dict()
            for key , value in props.items(): 
                valid_value = key_search(value, default_keys = self._columns, 
                                   parse_keys= False , deep =True )
                if valid_value is not None: 
                    pnames [key] = valid_value [0]
                    
            # rename data_skhrunked columns 
            data_shrunked.rename ( columns= pnames, inplace =True )
            for key in list(d.columns): 
                for pkey in  list(data_shrunked.columns): 
                    if key ==pkey:
                        d[key] = np.array(data_shrunked[pkey])
                        break 

        setattr (obj , self.kind +'_', d )
        return ( getattr (obj, self.kind +'_') if return_data 
                else obj 
                )

class DSDrill (GeoBase) : 
    """ Drill data set class. 
    
    :class:`DSDrill` reads, constructs the well/hole (drillhole:DH), geology  
    and geochemistry samples into a data set for transforming geophysics, 
    geology, GIS, and geochemistry data collecting in a survey area into a 
    three dimensional representation with `Oasis montaj`_. 
    Deal with drillhole menu of Oasis Montaj software. Build data and 
    contruct three dimensional data with `Oasis montaj`_ in Seequent. 

    Parameters 
    -----------
    area: str
       Area where the drilling operation is performed. 
      
    holeid: str,  
      the column name in the data where the well/hole ID is stored. 
      
    dname: str, optional 
       Depth column name. If `depth` is specify an attribute `depth_` should 
       be created. Depth specification is usefull for log plotting of machine
       training.
       
    utm_zone: Optional, string
       zone number and 'S' or 'N' e.g. '55S'. Default to the centre point
       of coordinates points in the survey area. It should be a string (##N or ##S)
       in the form of number and North or South hemisphere, 10S or 03N
       
    projection: str, ['utm'|'dms'|'ll'] 
       The coordinate system in which the data points for the profile is collected. 
       If not given, the auto-detection will be triggered and find the suitable 
       coordinate system. However, it is recommended to provide it for consistency. 
       Note that if `x` and `y` are composed of value less than 180 degrees 
       for longitude and 90 degrees for latitude, it should be considered as  
       longitude-latitude (``ll``) coordinates system. If `x` and `y` are 
       degree-minutes-second (``dms`` or ``dd:mm:ss``) data, they must be 
       specify as coordinate system in order to accept the non-numerical data 
       before transforming to ``ll``. If ``data`` is passed to the :meth:`.fit`
       method and ``dms`` is not specify, `x` and `y` values should be discarded.
       
    datum: string, default = 'WGS84'
       well known datum ex. WGS84, NAD27, NAD83, etc.
      
    encoding: str, default ='utf8'
       Default encoding for parsing data. Can also be ['utf-16-be'] for 
       reading bytes characters. 
          
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
        
   properties: dict, 
      Data column can not fit the Drilling property columns. In that case. 
      Mapping the property names is usefull to specify the columns in the 
      original data that fits the :class:`DSDrill` property codes. For 
      instance:: 
          
          properties ={"hole name": 'DH_Hole', 
                           "easting": 'DH_East', 
                           "northing": 'DH_Northing'
                           }
      
       where "hole name", "easting" and "northing" are the column names in 
       the former data set. These names should be replaced by the "DH_Hole", 
       "DH_East" and "DH_North" respectively. The mapping can be used  to 
       specify any other property names. Note that the valuable properties 
       are: 
           
       - "DH_Hole": Well/hole ID  
       - "DH_East": Easting coordinates of the well/hole
       - "DH_North":  Northing coordinates of the well/hole. 
       - "Mask": Any comment about the well/hole ID 
       - "DH_RH": Radius value of the well/hole ID  
       - 'DH_From': Top(roof) of any stratum/sample or rock in the well/hole.
       - "DH_To" : Wall( bottom) of any stratum/sample or rock in the well/hole. 
       - "Rock" : Rock or strata/layer in the well/hole.  
       - "DH_Azimuth": Azimuth value in the well/hole. 
       - 'DH_Top': Surface-level of the well/hole compared to the level of sea. 
       - 'DH_Bottom': Maximum depth of the well/hole. 
       - 'DH_PlanDepth': Any section performed in the well/hole can be inputed.
       - 'DH_Decr': Description of the well/hole. 
       - 'Sample': Sample collected in the well/hole at different depth.
       - 'DH_Dip': Dip of the well/hole.  
       - 'Elevation': Elevation of the well/hole. 
       - 'DH_RL': Level of piezometric value if exists in the well/hole. 
       
    verbose: int, default=0 
       Output messages. 
      
    Attributes
    -----------
    depth_: Series 
       Depth array if `dname` is specified. 
    data_: Pandas DataFrame
       Sanitized dataframe.
       
    collar_:Pandas DataFrame
       Collar data composed of each borehole description. 
    geology_: Pandas DataFrame
       Geology data that compose each geological rocks 
       
    samples_: Pandas DataFrame
       Each Geochemistry samples data compose each sample collected 
       in the survey area.
    """
    code=(
        "DH_Hole", 
        "DH_East", 
        "DH_North", 
        "Mask",
        "DH_RH", 
        'DH_From' , 
        "DH_To" , 
        "Rock" , 
        "DH_Azimuth" , 
        'DH_Top', 
        'DH_Bottom',
        'DH_PlanDepth', 
        'DH_Decr', 
        'Sample',
        'DH_Dip', 
        'Elevation',
        'DH_RL',
    )       

    def __init__(
        self,  
        area =None, 
        holeid=None, 
        dname = None, 
        projection ='ll', 
        utm_zone = None, 
        datum ='WGS84', 
        encoding ='utf-8', 
        epsg=None, 
        reference_ellipsoid =23, 
        properties =None, 
        **kwd
        ): 
        super().__init__( **kwd)
        
        self.holeid=holeid
        self.area=area 
        self.projection=projection 
        self.utm_zone=utm_zone 
        self.dname=dname 
        self.encoding=encoding 
        self.datum=datum 
        self.epsg=epsg 
        self.reference_ellipsoid=reference_ellipsoid
        self.properties=properties


    def fit (self, data =None, **fit_params ): 
        """ Fit drill data to build 
        
        Parameters
        ------------
        data: pd.DataFrame or Path-like object. 
           Data containing the drilling informations. 
           
        fit_params: dict, 
           Keyword arguments passed to :func:`watex.utils.to_numeric_dtypes`
           for data management. 
           
        Return 
        ---------
        self : Instanced object 
            Instanced object for chaining method. 
            
        Examples 
        -----------
        >>> from watex.geology.drilling import Drill 
        >>> csv_data ='data/drill/nbleDH.csv'
        >>> xlsx_data= 'data/drill/nbleDH.xlsx'
        >>> dr0 = Drill().fit(csv_data) 
        >>> dr0.data_.head(2) 
          DH_Hole__ID_      DH_East     DH_North  ... DH_Top    DH_Bottom DH_PlanDepth
        0          S01  477205.6935  2830978.218  ...      0       968.83          NaN
        1          S02  477261.7258  2830944.879  ...      0  974.8945704          NaN
        >>> dr1 = Drill().fit(xlsx_data) 
        >>> dr1.data_.head(2) 
            DH_Hole (ID)      DH_East  ...  sample03     sample04
          0          S01  477205.6935  ...       pup  Boudin Axis
          1          S02  477261.7258  ...       pup          pzs
        """
        # initialize Drill properties 
        self.collar_=None ; self.geology_=None; self.samples_=None 
        if ( isinstance ( data, str ) 
            and os.path.isfile ( data)
            and str(data).endswith ('.xlsx')
            ): 
            return self._fit_sheets(data )
        
        # NaN columns here can be useful especially 
        # when column name is one of drillimg 
        # property
        _fit( self, data = data, drop_nan_columns=False,
             input_name ='DH', ** fit_params  )
        
        if self.properties is not None: 
            if not isinstance ( self.properties, dict ): 
                if self.verbose:
                    warn(f"Properties {self.properties} are ignored."
                         " Expect a dictionnary of key/values in pair."
                         f" Got {type(self.properties).__name__!r}")
                
            else: self.data_.rename (columns = self.properties, inplace =True ) 
                
        return self 
    
    def _fit_sheets( self, d ): 
        """ Read sheets and populate attributes. If collar, geology 
        and samples are in the files, attributes should be set instead. """
        
        dprops = {'samp': 'samples', 'geol': 'geology', 'coll': 'collar'
         }
        data , names = read_worksheets( d ) 
        names = sanitize_frame_cols(names, fill_pattern='_')
        
        # renames 
        for ii, name in enumerate( names): 
            for k, v in dprops.items (): 
                if str(name).lower().find ( k )>=0: 
                    names[ii] = v 
                    break 
                
        for kk , name in enumerate(names): 
            setattr ( self, name +'_', to_numeric_dtypes(
                data[kk], drop_nan_columns=False ))
            
        # by default concat data and set as a new data 
        self.data_ = pd.concat(data , axis =1 )
        
        return self  
    
    def get_collar (self, data =None, reset_collar =False , **kws ): 
        """ Get or set the collar  data. 
        
        Parameters 
        -----------
        data: str, pd.Dataframe 
          Path-like object of dataframe containing the collar data 
          
        reset_collar: bool, defult=False, 
          If collar is provided, resetting the collar data will replace the 
          previous collar data in the original data. 
          
        kws: dict, 
           Keyword arguments passed to :func:`watex.utils.coreutils._is_readable`
           
        Returns
        --------
        self: Instanced object 
           Object instanced  for method chaining 
           
        Examples
        ---------
        >>> from watex.geology.drilling import DSDrill 
        >>> xlsx_data= 'data/drill/nbleDH.xlsx'
        >>> dr = DSDrill().fit(xlsx_data)
        >>> dr2 = dr.get_collar (dr.collar_, reset_collar= True )
        >>> dr2.collar_
          DH_Hole (ID)      DH_East     DH_North  ...  DH_PlanDepth  DH_Decr  Mask 
        0          S01  477205.6935  2830978.218  ...           NaN      NaN    NaN
        1          S02  477261.7258  2830944.879  ...           NaN      NaN    NaN
        
        >>> dr.holeid # id hole is autodetected if not given
        'DH_Hole (ID)'
        >>> # retreive the holeID S01 
        >>> dr.collar.S01
        {'DH_Hole (ID)': 'S01',
         'DH_East': 477205.6935,
         'DH_North': 2830978.218,
         'DH_Dip': -90.0,
         'Elevation ': 0.0,
         'DH_Azimuth': 0.0,
         'DH_Top': 0.0,
         'DH_Bottom': 968.83,
         'DH_PlanDepth': nan,
         'DH_Decr': nan,
         'Mask ': nan}
        
        """
        self.inspect 
        
        cmsg = ("Collar data already exists."
                " To force resetting collar data,"
                " set `reset_collar=True` instead."
                ) 
        if data is not None: 
            if hasattr(self, 'collar_'): 
                warn(cmsg ) if not reset_collar else None 

            col= _is_readable(data, as_frame =True, 
                                        input_name='DH_', 
                                        **kws)
            if reset_collar: 
                self.collar_= col 
           
        if ( data is None and not hasattr (self, 'collar_')): 
            raise DrillError( "Collar data is missing.")
            
        if self.properties is not None: 
            if not isinstance (self.properties, dict): 
                msg =(
                    "Drilling property names expect a dictionnary. Got"
                    f"{type (self.property_names).__name__!r}. Property"
                    f" names are used for codes {self.code} identification."
                    " In principle, each code must be a component i.e a"
                    " column name of the drill data. For instance, 'DH_hole'"
                    " is used to map 'hole_id' if the property name is set to"
                    " {'DH_hole:'hole_id'}. "
                      )
                warn(msg )
            else: 
                self.collar_.rename (columns = self.properties, 
                                     inplace =True )

        self._create_sub_drill_object (self.collar_, 'collar')
        
        return self 
    
    def get_geology (self, data=None, reset_geology=False, **kws ): 
        """ Get the geological informations that composed each drilling. 
        
        Parameters 
        -----------
        data: str, dataframe
           Path like object composed of layer thickness and geology 
           
        reset_geology: bool, default=False 
           If ``True`` it assumes that the data 
 
        kws: dict, 
           Keyword arguments passed to :func:`watex.utils.coreutils._is_readable`
           
        Returns
        --------
        self: Instanced object 
           Object instanced  for method chaining 
           
        Examples
        ---------
        >>> from watex.geology.drilling import DSDrill
        >>> xlsx_data= 'data/drill/nbleDH.xlsx'
        >>> dr = DSDrill().fit(xlsx_data)
        >>> dr.get_geology (dr.geology_, reset_geology=True ).geology_
          DH_Hole     Thick01  ...                    Rock03  Rock04
        0     S01    0.200000  ...  carbonate iron formation    ROCK
        1     S02  174.429396  ...                       GRT    ROCK
        >>> dr.holeid # id hole is autodetected if not given
        Out[62]: 'DH_Hole'
        >>> # retreive the hole ID of S01  drilling.
        >>> dr.geology.S01
        {'DH_Hole': 'S01',
         'Thick01': 0.2,
         'Thick02': 98.62776918,
         'Thick03': 204.7500461,
         'Thick04': 420.0266651,
         'Rock01': 'clast supported breccia',
         'Rock02': 'sulphide-rich material',
         'Rock03': 'carbonate iron formation',
         'Rock04': 'ROCK'}
        """
        self.inspect 
        
        cmsg = ("Geology data already exists. To force resetting geology data,"
                " set `reset_geology=True` instead."
                ) 
        if data is not None: 
            if hasattr(self, 'geology_'): 
                warn(cmsg ) if not reset_geology else None 

            geol = _is_readable(data, as_frame =True, input_name='Rock_', 
                                **kws)
            if reset_geology: 
                self.geology_= geol 
                
        if ( 
                data is None 
                and not hasattr (self, 'geology_')
                ): 
            raise DrillError( "Geology data is missing.")
            
            
        self._create_sub_drill_object (self.geology_, 'geology')
        
        return self 
    
    def get_geosamples (self, data=None, reset_samples=False, **kws ): 
        """Get or set the geochemistry samples  data. 
        
        Parameters 
        -----------
        data: str, pd.Dataframe 
          Path-like object of dataframe containing the geochemistry  sample 
          data 
          
        reset_samples: bool, default=False, 
          If geochemistry samples is provided, resetting the samples data 
          will replace the  previous samples data in the original data. 
          
        kws: dict, 
           Keyword arguments passed to :func:`watex.utils.coreutils._is_readable`
           
        Returns
        --------
        self: Instanced object 
           Object instanced  for method chaining 
           
        Examples
        ---------
        >>> from watex.geology.drilling import DSDrill 
        >>> xlsx_data= 'data/drill/nbleDH.xlsx'
        >>> dr = DSDrill().fit(xlsx_data)
        >>> dr.get_geosamples (dr.samples_, reset_samples= True ).samples_
          DH_Hole  Thick01     Thick02  ...             sample02  sample03     sample04
        0     S01     10.0   98.627769  ...                  prt       pup  Boudin Axis
        1     S02     17.4  313.904388  ...  Banding/gneissosity       pup          pzs
        >>> dr.holeid # id hole is autodetected if not given
        'DH_Hole'
        >>> # retreive the holeID geosamples S02 
        >>> dr.samples.S02
        {'DH_Hole': 'S02',
         'Thick01': 17.4,
         'Thick02': 313.9043882,
         'Thick03': 400.12,
         'Thick04': 515.3,
         'sample01': 'pup',
         'sample02': 'Banding/gneissosity',
         'sample03': 'pup',
         'sample04': 'pzs'}
        """
        self.inspect 
        
        cmsg = ("Geosamples data already exists. To force resetting sample"
                " from geology data,  set `reset_samples=True` instead."
                ) 
        if data is not None: 
            if hasattr(self, 'samples_'): 
                warn(cmsg ) if not reset_samples else None 
     
            geosamples = _is_readable(
                data, as_frame =True, 
                input_name='Sample_', 
                **kws)
            if reset_samples: 
                self.samples_= geosamples 
        if ( 
                data is None 
                and not hasattr (self, 'samples_')
                ): 
            raise DrillError( "Geosamples data is missing.")
            
        self._create_sub_drill_object (self.samples_ , 'samples')
        
        return self 
    
    def _create_sub_drill_object (self, odata , oname ): 
        """ From object data ( dataframe), set each columns as attribute of 
        of the data.
        
        :param odata: dataframe -data object 
        :param oname: str - name of object to remame as subobject. 
        
        """
        use_col =False 
        # use DH_hole as ID is 
        # column is set
        if self.holeid is None: 
            for name in odata.columns:
               if sanitize_frame_cols([str(name)])[0].lower().find (
                       'hole')>=0:
                   self.holeid= name
                   break 
               
        if self.holeid is not None: 
            if self.holeid not in odata.columns: 
                warn(f"Holeid {self.holeid!r} is missing in the drilling data")
                
            else:use_col = True 
  
        d = data2Box ( 
            odata , 
            name =self.holeid, 
            use_colname= use_col
                      )
        setattr (self, oname, d  )

    def _init_build ( self , kind ='well/hole'): 
        """ Build initial drilling data. 
        
        The geology, geochemistry, collar and elevation follow the common 
        data informations such as: 
            
        - The well/hole or sample ID: ID to identify the object 
        - geographical coordinates ( xlon, ylat): Expect two coordinates. The 
          coordinate system projection is set to ``ll``, so any values greater 
          than 180 and 90 degrees for longitude and latitude respectively 
          sill raise an error. To enter the UTM coordinates, set the attribute 
          ``projection='utm'`` like: 
              
              >>> DSDrill (projection ='utm')
        - radius: Is the hole radius in meters. 
        
        Parameters
        ----------
        kind: str, 
           Indicates the kind of data to build. 
           
        Returns 
        ---------
        dh_hole, dh_xlon, dh_ylat, dh_rh: Tuple of str 
           Parsing data in string value. 
           
        """

        dh_hole = input (f"Enter the {kind} ID:").upper() 
        coords = input ( f"Enter {dh_hole} coordinates (x/lon, y/lat):")
        # accept decimal values. 
        coords= np.array ( str2columns(coords, pattern= r'[_#&-*@!_,;\s\s*]'))
        if len(coords) !=2: 
            raise ValueError (f"Need {dh_hole} both coordinates x/longitude "
                              f"and y/latitude. Got {len(coords)}")
        try : 
            dh_xlon, dh_ylat = list(coords.astype (float)) 
        except: raise TypeError ("Coordinates should be numeric."
                                 f" Got {coords.dtype.name!r}")
        
        if self.projection =='ll': 
            dh_ylat, dh_xlon = Location.to_utm_in(dh_ylat, dh_xlon, 
               utm_zone= self.utm_zone)
            dh_ylat, dh_xlon= dh_ylat[0], dh_xlon[0] 
            
        dh_rh= input ( f"Enter {dh_hole} radius in meters [Optional]:" )
        
        return dh_hole, dh_xlon, dh_ylat, dh_rh
           
    @_drill_builder ( kind ='geology')
    def build_geology(
        self,  *, 
        return_data=False, 
        hole_elevation=0., 
        **kwd 
        ): 
        """Build manually the geology data of collected drilling area.
        
        :meth:`build_geology` collect the rocks or strata names collected 
        during the drilling operations or well contructions.  
        The collection of the information will build a geology dataset 
        which can be used to Oasis Montaj software modeling. Below is an 
        example of geology data set construction. 
        This is some explanation of the prompt: 
            
        - well/hole ID: Is the name or ID of the rocks or strata collected 
          in the area 
        - coordinates (xlon, xlat): is the geographical coordinates where 
          the drilling operation is performed. Expect projection is ``ll``. 
          To enter the UTM coordinates, set the projection in 
          the building 
          object to ``utm`` like:: 
              
              >>> DSDrill (projection='utm').build_geology (return_geology =True )
              
        - radius (m): Is the radius of the hole/well 
        - depth: depth of the well/hole in meters. 
        - layer thickness: The thickness of each strata in the whole. 
          Note that when many thickness are supplied, it may correspond to
          each layer i.e. the number of thicknesses must 
          equal to the number of layers. If not the case, ``NA`` should be 
          used to indicate the missing layer/rock name in the geology dataset. 
          
        - mask/comments: Is any comments aboud the well/hole.  
        
        Parameters 
        -----------
        return_data: bool, default=False, 
          Return the geology dataset rather than object (``False``) after 
          entering the appropriate argumments. Note that even ``True``, 
          the geology data set can be retrieved via the attribute ``geology_``.
          
        hole_elevation: float, default=0, 
          The elevation or the level of surface of the well/hole compared to 
          the level of sea. Note that elevation must be negative value on the 
          top of the air for layer/strat calculation. 
          
        Returns 
        ---------
        self, geology dataset: :class:`DSDrill` or pd.DataFrame. 
           Return object when ``return_geology=False`` and DataFrame otherwise. 
           
        Examples 
        ---------
        >>> DSDrill ().build_geology (return_geology =True )
        Enter the well/hole ID:GEOK4
        Enter GEOK4 coordinates (xlon, ylat):10 15
        Enter GEOK4 radius in meters [Optional]:12.2
        Enter the GEOK4 depth in meters: 75
        Enter each stratum thickness of GEOK4 [top-->bottom] in meters:10 20 11
        Enter the layer/rock names of GEOK4 [top-->bottom]:granite gneiss
        Enter valuable comments about GEOK4 [Optional]:building test
        Tap exit|0 to terminate or enter to continue:0
        Out[17]: 
          DH_Hole       DH_East  DH_North  ...  DH_To     Rock           Mask
        0   GEOK4  1.105412e+06  500000.0  ...   10.0  granite  building test
        1   GEOK4  1.105412e+06  500000.0  ...   30.0   gneiss  building test
        2   GEOK4  1.105412e+06  500000.0  ...   41.0       NA  building test
        3   GEOK4  1.105412e+06  500000.0  ...   75.0       NA  building test
        """
        getter =[]

        while 1: 
            
            init_data = self._init_build () 
            dh_hole = init_data [0] 
            depth= input (f"Enter the {dh_hole} depth in meters: ")
            if str(depth).strip().lower() =='': 
                raise ValueError (f"{dh_hole} total depth is needed for"
                                  " stratum boundaries demarcation.")
                
            try: depth = float(depth)
            except: raise TypeError (
                f"Depth should be numeric. Got {type(depth).__name__!r}")
                
            thick  = input (f"Enter each stratum thickness of {dh_hole}"
                             " [top-->bottom] in meters:"
                             )
            
            if str(thick).strip().lower() =='': 
                raise ValueError ("Missing the layer/stratum thickness."
                                  " Layer are essential for drilling log"
                                  " construction.")
            # compute thickness   
            dh_from , dh_to, thick = smart_thickness_ranker(
                thick , 
                surface_value= hole_elevation, 
                return_thickness= True, 
                verbose=self.verbose , 
                depth=depth, 
                mode='soft', 
                )
      
            dh_rocks = input (f"Enter the layer/rock names of {dh_hole}"
                              " [top-->bottom]:")
            
            dh_rocks = str2columns( dh_rocks ) 
            # for consistency, set to lowercase 
            dh_rocks=[ g.lower() for g in dh_rocks] 
            dh_mask=input (f"Enter valuable comments about {dh_hole}"
                           " [Optional]:")
             
            # add NA for missing rock 
            ad_NA = [ 'NA' for i in range ( len(thick) )]
            dh_rocks +=ad_NA 
            dh_rocks = dh_rocks[: len(dh_from)] 
            
            # Repeat the init data to fit the 
            init_new =[ 
                np.repeat( item, len(dh_from)) for item in init_data ]
            # for item  in init_data: 
            #     init_new .append ( np.repeat( item, len(dh_from)))
            dh_mask = np.repeat ( dh_mask,len(dh_from) )
            
            v= init_new + [dh_from, dh_to, dh_rocks, dh_mask]
            getter.append (np.vstack( v).T)

            end = input('Tap "exit|0" to terminate or "Enter" to continue:') 
            if str(end).lower ().strip() in ( '0', 'exit'): 
                break 
            
        return ( 
            self,  
            np.vstack ( getter ), 
            return_data
            )
    
    @_drill_builder ( kind ='samples')
    def build_geosamples( self, *, return_data=False , **kwd): 
        """Build manually the geochemistry samples of  area.
        
        :meth:`build_geosamples` collect the geochemistry samples in the 
        survey area with their geographical coordinates locations. The name 
        as well as the sample thickness and the radium of the holes can be 
        from the prompt. 
        The collection of the information will build a geosamples dataset 
        which can be used to Oasis Montaj software modeling. Below is an 
        examples of Outputted. This is some indication of the prompts: 
            
        - sample ID: Is the name or ID of the samples collected in the area 
        - coordinates (xlon, xlat): is the geographical coordinates where 
          the sampling is performed. 
        - The radius (m): Is the radius of the hole did for collecting the 
          sample.
        - samples thickness: Ask the thickness of the samples in the whole. 
          Note that when many thickness are supplied, it means the same sample 
          is collected at different depth. There are two kind of data to
          supply: 
              
          - t-value: Compose only with the layer thickness values. For instance 
            ``t= "10 20 7 58"`` indicates four samples with thicknesses 
            equals to 10, 20, 7 and 58 ( meters) respectively. 
          
          - tb-range: compose only with thickness range at each depth. For 
            instance ``t= "0-10 10-30 40-47 101-159"``. Note the character used  
            to separate thickness range is ``'-'``. Here, the top(roof) and 
            bottom(wall) of the sample are 0  (top) and 10 (bottom), 10 and 30, 
            40 and 47 , and 101 and 159 for the same sample. 
            
          - Note that any mixed types is not acceptable and willraises error.
            To verify whether the expected samples values is acceptable or not, 
            use the following :func:`watex.utils.geotools.get_thick_from_range` 
            or :func:`watex.utils.geotools.get_thick_from_values` functions.
          
        - sample name: Is the name of samples collected refereing to the 
          different depth. In principle, the number of samples thickness must 
          equals to the number of samples. If not the case, ``NA`` should be 
          used to indicate the missing samples in the geosamples dataset. 
          
        - mask/comments: Is any comments aboud the sample 
        
        Parameters 
        -----------
        return_data: bool, default=False, 
          Return the samples dataset rather than object (``False``) after 
          prompted the appropriate argumments. Note that even ``True``, 
          the geosample data set can be retrieved via the attribute 
          ``sample_``.
        
        Returns 
        ---------
        self, geosample dataset: :class:`DSDrill` or pd.DataFrame. 
           Return object when ``return_data=False`` and DataFrame otherwise. 
           
        Examples 
        ---------
        >>> dr = DSDrill ().build_geosamples (return_samples =True )
        Enter the sample ID:sx02
        Enter SX02 coordinates (xlat, ylon):12 15
        Enter SX02 radius in meters [Optional]:2.5
        Enter the sample thickness of SX02 in meters:15 15 48 23
        Enter the sample names of SX02:pup sup op
        Enter valuable comments about SX02 [Optional]:t-value
        Tap "exit|0" to terminate or "Enter" to continue:
        Enter the sample ID:sx05
        Enter SX05 coordinates (xlat, ylon):15 56
        Enter SX05 radius in meters [Optional]:2.3
        Enter the sample thickness of SX05 in meters:12-17 26-36 40-57
        Enter the sample names of SX05:benz op 
        Enter valuable comments about SX05 [Optional]:t-range1
        Tap "exit|0" to terminate or "Enter" to continue:
        Enter the sample ID:sx07
        Enter SX07 coordinates (xlat, ylon):10 15
        Enter SX07 radius in meters [Optional]:1.25
        Enter the sample thickness of SX07 in meters:56-76
        Enter the sample names of SX07:gru
        Enter valuable comments about SX07 [Optional]:t-range2
        Tap "exit|0" to terminate or "Enter" to continue:0
        >>> dr
        Out[8]: 
          DH_Hole       DH_East       DH_North  DH_RH  DH_From  DH_To Sample      Mask
        0    SX02  1.326554e+06  500000.000000   2.50      0.0   15.0    pup   t-value
        1    SX02  1.326554e+06  500000.000000   2.50     15.0   30.0    sup   t-value
        2    SX02  1.326554e+06  500000.000000   2.50     30.0   78.0     op   t-value
        3    SX02  1.326554e+06  500000.000000   2.50     78.0  101.0     NA   t-value
        4    SX05  1.658569e+06  392487.772324   2.30     12.0   17.0   benz  t-range1
        5    SX05  1.658569e+06  392487.772324   2.30     26.0   36.0     op  t-range1
        6    SX05  1.658569e+06  392487.772324   2.30     40.0   57.0     NA  t-range1
        7    SX07  1.105412e+06  500000.000000   1.25     56.0   76.0    gru  t-range2
        """
        getter =[]

        while 1: 
            init_data = self._init_build (kind = 'sample') 
            dh_hole = init_data [0] 

            thick  = input (f"Enter the sample thickness of {dh_hole}"
                             " in meters:"
                             )
            if str(thick).strip().lower() =='': 
                raise ValueError ("Missing the layer/stratum thickness."
                                  " Layer are essential for drilling log"
                                  " construction.")
            # compute thickness 
            dh_from , dh_to, thick = smart_thickness_ranker (
                thick, mode ='soft', return_thickness= True)
            
            dh_samples= input (f"Enter the sample names of {dh_hole}:")
            dh_samples = str2columns( dh_samples ) 
            # for consistency, set to lowercase 
            dh_samples=[ g.lower() for g in dh_samples] 
            dh_mask=input (f"Enter valuable comments about {dh_hole}"
                           " [Optional]:") 
            # check_thickness and reset depth if 
            # start always  the layer demarcation from 0 
            # at the surface 
            # append NA to the rocks name 
            ad_NA = [ 'NA' for i in range ( len(thick) )]
            dh_samples +=ad_NA 

            dh_samples = dh_samples[: len(dh_from)] 
            # Repeat the init data to fit the 
            init_new =[ 
                np.repeat( item, len(dh_from)) for item in init_data ]
            # for item  in init_data: 
            #     init_new .append ( np.repeat( item, len(dh_from)))
            dh_mask = np.repeat ( dh_mask,len(dh_from) )
            
            v= init_new + [dh_from, dh_to, dh_samples, dh_mask]
            getter.append (np.vstack( v).T)

            end = input('Tap "exit|0" to terminate or "Enter" to continue:') 
            if str(end).lower ().strip() in ( '0', 'exit'): 
                break 
            
        return ( 
            self, 
            np.vstack ( getter ), 
            return_data 
            )
    
    @_drill_builder ( kind ='collar')
    def build_collar( 
        self, *, 
        return_data=False, 
        compute_azimuth:bool=..., 
        utm_zone: str=..., 
        **kwd
        ): 
        """ Build manually the collar data of collected drilling area.
        
        Collar data is composed of well/hole description and usefull
        informations. 
        
        :meth:`build_collar collect each drillhole informations  during the 
        drilling operations or well contructions.  
        The total of the information will build a collar dataset 
        which can be used to Oasis Montaj software modeling. Below is an 
        example of collar data set construction. 
        This is some explanation of the prompt: 
            
        - well/hole ID: Is the name or ID of the rocks or strata collected 
          in the area 
        - coordinates (xlon, xlat): is the geographical coordinates where 
          the drilling operation is performed. Expect projection is ``ll``. 
          To enter the UTM coordinates, set the projection in 
          the building 
          object to ``utm`` like:: 
              
              >>> DSDrill (projection='utm').build_collar (return_data =True )
              
        - radius (m): Is the radius of the hole/well
        - surface-level: The level of the well/hole compared to the level of 
          the sea. By default it is set to null as the level of the sea. 
        - depth: depth of the well/hole in meters. 
        - dip: dip in degree of the well/hole. 
        - elevation: elevation value of the well/hole in meters. Default is 
          no topography i.e. equal null. 
        - azimuth: azimuth value in degrees. If not given explicitely, it can 
          be calculated using the utm coordinates provided that the parameter
          `compute_azimuth` is set to ``True`` and `utm_zone` is also 
          provided. In this case, we assume that all well/hole collected 
          belongs to the same area. Furthermore, azimuth calculation will 
          cancel if one of the above condition is not met and if only a 
          single borehole is supplied. 
        - plan-depth: Is a litteral string to give at which stage a section
          in a borehole is performed. Note that this is optional parameter  
          and can be skipped. It has no more influence about the collar 
          construction.
        - description: Give a short description of well/hole mostly about the 
          drilling settlment. 
        - mask/comments: Is any comments aboud the well/hole.  
        
        Parameters 
        -----------
        return_data: bool, default=False, 
          Return the collar dataset rather than object (``False``) after 
          entering the appropriate argumments. Note that even ``True``, 
          the collar data set can be retrieved via the attribute ``collar_``.
          
        compute_azimuth: bool, default=False 
          Recompute the azimuth using the UTM coordinates. Note that projection
          need to be turned in 'utm' as well as the 'utm_zone' needs also to 
          be supplied. 
          
        utm_zone:str, 
          zone number and 'S' or 'N' e.g. '55S'. Default to the centre point
          of coordinates points in the survey area. It should be a string 
          (##N or ##S)in the form of number and North or South hemisphere, 
          10S or 03N. if :attr:`~DSdrill.utm_zone` is already set, it is not 
          need to reset again. Resetting new `utm_zone` will erase the value 
          of the former attribute. However for azimuth calculation, utm zone 
          cannot be None otherwise the process is aborted. 
 
        Returns 
        ---------
        self, collar dataset: :class:`DSDrill` or pd.DataFrame. 
           Return object when ``return_data=False`` and DataFrame otherwise. 
           
        Examples 
        ---------
        >>> col_data=DSDrill (projection ='utm').build_collar (return_collar =True )
        Enter the well/hole ID:DXT03
        Enter DXT03 coordinates (xlon, ylat):297856 352145
        Enter DXT03 radius in meters [Optional]:.25
        Enter DXT03 surface-level <top> in meters[0.]:
        Enter DXT03 depth <bottom> in meters[700.]:120
        Enter DXT03 dip in degrees [-90]:
        Enter DXT03 elevation in meters [0.]:
        Enter DXT03 azimuth [Optional]:
        Enter DXT03 plan-depth [Optional]:wandx01
        Enter DXT03 description [Optional]:hole-test
        Enter valuable comments about DXT03 [Optional]:RAS
        Tap "exit|0" to terminate or "Enter" to continue:
        Enter the well/hole ID:toxg
        Enter TOXG coordinates (xlon, ylat):125869 235645
        Enter TOXG radius in meters [Optional]:2.3
        Enter TOXG surface-level <top> in meters[0.]:-10
        Enter TOXG depth <bottom> in meters[700.]:135
        Enter TOXG dip in degrees [-90]:-70
        Enter TOXG elevation in meters [0.]:3
        Enter TOXG azimuth [Optional]:2.3
        Enter TOXG plan-depth [Optional]:wxzu
        Enter TOXG description [Optional]:None
        Enter valuable comments about TOXG [Optional]:RAS
        Tap "exit|0" to terminate or "Enter" to continue:0
        >>> col_data
        Out[3]: 
          DH_Hole   DH_East  DH_North  DH_RH  ...  DH_Azimuth  DH_PlanDepth    DH_Decr  Mask
        0   DXT03  297856.0  352145.0   0.25  ...         NaN       wandx01  hole-test   RAS
        1    TOXG  125869.0  235645.0   2.30  ...         2.3          wxzu       None   RAS
        """
        if compute_azimuth is ...: 
            compute_azimuth=False 
            
        if compute_azimuth: 
            if self.verbose: 
                if self.projection !='utm': 
                    warn("Projection should be set to 'UTM' for azimuth"
                         " calculation.")
            self.projection='utm'
            
            if utm_zone is ...: 
                utm_zone =None 
            self.utm_zone = utm_zone or self.utm_zone 
            
            if self.utm_zone is None: 
                if self.verbose: 
                    warn("Unknow 'utm_zone'. Process for azimuth"
                         " recalculation aborted.")
                compute_azimuth=False 
                
        # Enter data 
        getter =[]
        while 1: 
            init_data = self._init_build () 
            dh_hole = init_data [0] 
            dh_top = input (f"Enter {dh_hole} surface-level <top> in meters[0.]:")
            if str(dh_top).strip().lower() =='': 
                dh_top =0. 
                
            dh_bottom = input (f"Enter {dh_hole} depth <bottom> in meters[700.]:")
            if str(dh_bottom).strip().lower()=='': 
                dh_bottom= 700. 
                
            dh_dip = input (f"Enter {dh_hole} dip in degrees [-90]:")
            if str(dh_dip).strip() =='': 
                dh_dip=90.
            dh_elevation = input (f"Enter {dh_hole} elevation in meters [0.]:")
            if str(dh_elevation).strip().lower()=='': 
                dh_elevation= 0.
            dh_azimuth= input (f"Enter {dh_hole} azimuth [Optional]:")
            
            dh_plan_depth=input (f"Enter {dh_hole} plan-depth [Optional]:")
            dh_plan_descr=input (f"Enter {dh_hole} description [Optional]:")
            dh_mask=input (f"Enter valuable comments about {dh_hole}"
                           " [Optional]:")
            v= init_data + (convert_value_in(dh_top),
                            convert_value_in(dh_bottom),
                            convert_value_in(dh_dip), 
                            convert_value_in(dh_elevation), 
                            dh_azimuth, 
                            dh_plan_depth, 
                            dh_plan_descr,   
                            dh_mask, 
                )
            getter.append (np.array( v) )

            end = input('Tap "exit|0" to terminate or "Enter" to continue:') 
            if str(end).lower ().strip() in ( '0', 'exit'): 
                break 
        setattr ( self, '_compute_azimuth', compute_azimuth )
        return ( self, 
                np.vstack ( getter ), 
                return_data
                )
    @_drill_builder(mode ='get')
    def get_properties (self, kind: str ): 
        """ Get a specific drillhole properties. 
        
        Parameters 
        -----------
        kind: str, {"collar", "geology", "samples"}
          Kind of drillhole properties to fetch.
          
        Returns
        -------
        properties: Tuple 
           Tuple of properties 
           
        Examples 
        -----------
        >>> from watex.geology import DSDrill 
        >>> DSDrill ().get_properties (kind='geology')
        ('DH_Hole', 'DH_East', 'DH_North', 'DH_RH', 'DH_From', 'DH_To', 'Rock', 'Mask')
        """
        return self._assert_kind_value(kind )
    
    def build_drillholes( 
        self, 
        kind: str, 
        return_data: bool= ..., 
        hole_elevation=0., 
        compute_azimuth:bool=..., 
        utm_zone: str=..., 
        ): 
        """Build drillholes ['collar|'geology'|'samples'] mannually through 
        the parameter `kind`.
        
        All the drillholes have the common attributes such as: 
  
        - ``DH_Hole`` - well/hole , rock or sample ID: Is the ID of  the
          geochemistry samples, rocks/strata or well/holes collected in 
          the area 
        - ``"DH_East"`` and ``"DH_North"``- coordinates (xlon, xlat): is the 
          geographical coordinates where the drilling operation is performed. 
          Expect projection is ``ll``. To enter the UTM coordinates, set the 
          projection in the building object to ``utm`` like:: 
              
              >>> DSDrill (projection='utm').build_collar (return_data =True )
              
        - ``"DH_RH"`` - radius (m): Is the radius of the hole/well
        - ``Mask`` - mask/comments: Is any comments aboud the samples/geology 
          or  well/hole.
          
        Besides, there are some other specific attributes related to each 
        drillhole. For instance, the following attributes are required for: 
            
        - collar: 
            
          - ``"DH_Top"`` - surface-level: The level of the well/hole 
            compared to the level of the sea. By default it is set to null 
            as the level of the sea. 
          - ``"DH_Bottom"`` - depth: depth of the well/hole in meters. 
          - ``"DH_Dip"`` - dip: dip in degree of the well/hole. 
          - ``"Elevation"`` - elevation: elevation value of the well/hole 
            in meters. Default is no topography i.e. equal null. 
          - ``"DH_Azimuth"`` - azimuth: azimuth value in degrees. If not 
            given explicitely, it can be calculated using the utm coordinates 
            provided that the parameter `compute_azimuth` is set to ``True`` 
            and `utm_zone` is also provided. In this case, we assume that 
            all well/hole collected belongs to the same area. Furthermore, 
            azimuth calculation will cancel if one of the above condition 
            is not met and if only a single borehole is supplied. 
          - ``"DH_PlanDepth"`` - plan-depth: Is a litteral string to give at 
            which stage a section in a borehole is performed. Note that 
            this is optional parameter  and can be skipped. It has no more 
            influence about the collar construction.
      
          - ``DH_descr`` - description: Give a short description of well/hole 
            mostly about the drilling settlment. 
  
        - geology: 
            
          - ``"DH_From"`` and ``"DH_To"`` is computed from the layer 
            thicknesses i.e. the thickness of each strata in the whole. Note  
            that when many thicknesses are supplied, it may correspond to 
            each layer. Thus, the number of thicknesses must equal to the 
            number of layers. 
          
          - ``"Rock"`` - the rock/layer names. It should be equals to the 
            number of layer thicknesses. If not the case, ``NA`` should be
             used to indicate the missing layer/rock name in the geology 
             dataset. 
    
        - samples: 
        
          - ``"DH_From"`` and ``"DH_To"`` - samples thickness: Ask about the 
            thickness of the samples in the whole. Note that when many 
            thickness are supplied, it means the same sample is collected at 
            different depth. There are two kind of data to supply: 
    
          - t-value: Compose only with the layer thickness values. For 
            instance ``t= "10 20 7 58"`` indicates four samples with 
            thicknesses equals to 10, 20, 7 and 58 ( meters) respectively. 
          - tb-range: compose only with thickness range at each depth. For 
            instance ``t= "0-10 10-30 40-47 101-159"``. Note the character  
            used to separate thickness range is ``'-'``. Here, the top(roof)
            and bottom(wall) of the sample are 0  (top) and 10 (bottom), 
            10 and 30, 40 and 47 , and 101 and 159 for the same sample. 
          - Note that any mixed types is not acceptable and willraises 
            error. To verify whether the expected samples values is  
            acceptable or not, use the following 
            :func:`watex.utils.geotools.get_thick_from_range` 
            or :func:`watex.utils.geotools.get_thick_from_values` functions.
         
          - ``"Sample"`` - sample name: Is the name of samples collected 
            refereing to the different depth. In principle, the number of 
            samples thickness must equals to the number of samples. If not 
            the case, ``NA`` should be used to indicate the missing samples 
            in the geosamples dataset. 
              
        Parameters 
        ------------
        kind: str 
          kind of drillhole to build. It could be ['collar|'geology'|'samples']
          Until now, 'elevation' drillhole is not built yet. 

        return_data: bool, default=False, 
          Return the geology dataset rather than object (``False``) after 
          entering the appropriate argumments. Note that even ``True``, 
          the geology data set can be retrieved via the attribute ``geology_``.
          
        hole_elevation: float, default=0, 
          The elevation or the level of surface of the well/hole compared to 
          the level of sea. Note that elevation must be negative value on the 
          top of the air for layer/strat calculation. 
         
        compute_azimuth: bool, default=False 
          Recompute the azimuth using the UTM coordinates. Note that projection
          need to be turned in 'utm' as well as the 'utm_zone' needs also to 
          be supplied.
            
        utm_zone:str, 
          zone number and 'S' or 'N' e.g. '55S'. Default to the centre point
          of coordinates points in the survey area. It should be a string 
          (##N or ##S)in the form of number and North or South hemisphere, 
          10S or 03N. if :attr:`~DSdrill.utm_zone` is already set, it is not 
          need to reset again. Resetting new `utm_zone` will erase the value 
          of the former attribute. However for azimuth calculation, utm zone 
          cannot be None otherwise the process is aborted. 

        Returns 
        ---------
        self, collar/geology/geochemistry samples dataset:\
            :class:`DSDrill` or pd.DataFrame. 
           Return object when ``return_data=False`` and DataFrame otherwise.

        Examples 
        ----------
        >>> from watex.geology import DSDrill
        >>> DSDrill().build_drillholes ('collar', return_data=True) 
        Enter the well/hole ID:Dh00
        Enter DH00 coordinates (x/lon, y/lat):11 15
        Enter DH00 radius in meters [Optional]:0.2
        Enter DH00 surface-level <top> in meters[0.]:
        Enter DH00 depth <bottom> in meters[700.]:
        Enter DH00 dip in degrees [-90]:
        Enter DH00 elevation in meters [0.]:12
        Enter DH00 azimuth [Optional]:
        Enter DH00 plan-depth [Optional]:
        Enter DH00 description [Optional]:
        Enter valuable comments about DH00 [Optional]:testDh
        Tap "exit|0" to terminate or "Enter" to continue:0
        Out[16]: 
          DH_Hole       DH_East       DH_North  ...  DH_PlanDepth  DH_Decr    Mask
        0    DH00  1.659298e+06  715053.016841  ...           NaN      NaN  testDh
        >>> DSDrill().build_drillholes('geology')
        Enter the well/hole ID:Geo00
        Enter GEO00 coordinates (x/lon, y/lat):11 15
        Enter GEO00 radius in meters [Optional]:2
        Enter the GEO00 depth in meters: 60
        Enter each stratum thickness of GEO00 [top-->bottom] in meters:12
        Enter the layer/rock names of GEO00 [top-->bottom]:sand
        Enter valuable comments about GEO00 [Optional]:test-geo
        Tap "exit|0" to terminate or "Enter" to continue:0
        Out[19]: 
          DH_Hole       DH_East       DH_North  DH_RH  DH_From  DH_To  Rock      Mask
        0   GEO00  1.659298e+06  715053.016841    2.0      0.0   12.0  sand  test-geo
        1   GEO00  1.659298e+06  715053.016841    2.0     12.0   60.0    NA  test-geo
        
        """
        return_data, compute_azimuth = ellipsis2false( 
            return_data, compute_azimuth )
        kind = self._assert_kind_value( kind ) 
        
        func = {"geology": self.build_geology,
                "collar": self.build_collar, 
                "samples": self.build_geosamples 
                }
        return func.get( kind ) ( 
            return_data =return_data, 
            hole_elevation=hole_elevation, 
            compute_azimuth=compute_azimuth,
            utm_zone= utm_zone, 
            )
    
    @_drill_builder(mode ='fit')
    def set_drillholes (
        self, 
        kind:str , 
        properties: dict=None, 
        return_data: bool=False,
        ): 
        """ Transform the data and fit to the corresponding drillholes. 
        
        `kind` is used to specify the type of drillhole to build. The `DH_`
        used to prefix the drillhole properties can be omitted when 
        specifying the properties. 
        
        Parameters
        ------------
        kind: str, {"collar", "geology", "samples"}
           Kind of drillhole to build. 
           
        properties: dict, 
           A mapper that uses to replace the column names from data to 
           the values of the drillholes. If properties is not specified 
           drillhole data returns NaN columns. 
           
        return_data: bool, default=False 
          Out the drillhole data. Otherwise returns the drillhole object 
          :class:`DSDrill`. 
          
        Returns 
        ----------
        self, data : Object instanced and pd.DataFrame 
           :class:`DSDrill` Object instanced or drillholes data set 
           rebuild. Unmatch properties names should be replaced by NaN 
           
        Examples 
        ---------
        >>> import watex as wx
        >>> from watex.geology import DSDrill 
        >>> ndata = wx.fetch_data ('nlogs', samples =12, as_frame = True ) 
        >>> dr = DSDrill ().fit(ndata) 
        >>> dr.data_.head(2) 
        Out[26]: 
          hole_id  uniform_number  ... filter_pipe_diameter water_inflow_in_m3_d
        0   B0092    1.133960e+16  ...                0.120               15.249
        1   B0093    1.133110e+16  ...                0.125              114.474
        [2 rows x 25 columns]
        >>> # try list data column names and find the suitable columns that 
        >>> # could fit the drillholes properies. 
        >>> dr.data_.columns
        Out[27]: 
        Index(['hole_id', 'uniform_number', 'original_number', 'lon', 'lat',
               'longitude', 'latitude', 'east', 'north', 'easting', 'northing',
               'coordinate_system', 'elevation', 'final_hole_depth',
               'quaternary_thickness', 'aquifer_thickness', 'top_section_depth',
               'bottom_section_depth', 'groundwater_type', 'static_water_level',
               'drawdown', 'water_inflow', 'unit_water_inflow', 'filter_pipe_diameter',
               'water_inflow_in_m3_d'],
              dtype='object')
        >>> To get properties, one needs to call get properties to have an idea 
        >>> # before setting the columns to fit the properties as 
        >>> DSDrill ().get_properties (kind='collar')
        Out[29]: 
        ('DH_Hole',
         'DH_East',
         'DH_North',
         'DH_RH',
         'DH_Top',
         'DH_Bottom',
         'DH_Dip',
         'Elevation',
         'DH_Azimuth',
         'DH_PlanDepth',
         'DH_Decr',
         'Mask')
        >>> # Then we can match each data columns to the properties as:
        >>> properties = {"hole_id": "DH_Hole", "easting": 'DH_East',
                          "northing": 'DH_North', "top_section_depth": "DH_Top", 
                          "bottom_section_depth": "DH_Bottom", 
                          "final_hole_depth": 'DH_PlanDepth'}
        >>> dr.set_drillholes (kind='collar', properties = properties)
        >>> dr.collar_ 
        Out[30]: 
           DH_Hole      DH_East     DH_North  ... DH_PlanDepth  DH_Decr  Mask
        0    B0092  2509081.067  19774192.36  ...       200.14      NaN   NaN
        1    B0093  2576045.247  19757910.23  ...        50.00      NaN   NaN
        2    B0094  2579349.264  19763695.25  ...        47.00      NaN   NaN
        3    B0095  2568832.229  19760491.25  ...        59.40      NaN   NaN
        4    B0096  2566131.227  19765367.27  ...        42.00      NaN   NaN
        5    B0097  2568517.240  19770396.28  ...        50.70      NaN   NaN
        6    B0098  2583506.276  19763770.24  ...        50.00      NaN   NaN
        7    B0099  2576326.263  19770189.27  ...        50.00      NaN   NaN
        8    B0101  2567480.246  19777199.30  ...        60.00      NaN   NaN
        9    B0102  2583420.292  19775988.28  ...        50.00      NaN   NaN
        10   B0103  2577207.276  19778177.29  ...        60.10      NaN   NaN
        11   B0106  2519089.079  19758966.30  ...        48.10      NaN   NaN
        [12 rows x 12 columns]
           
        """
        self. inspect
        kind = self._assert_kind_value (kind )
        self.properties = properties or self.properties
        
        return ( self,  
                self.data_ , 
                return_data, 
                self.properties, 
                kind 
                )
    def fit_structures(
        self, kind : str, 
        keep_acronyms: bool=..., 
        fill_value: str =None, 
        inplace: bool=..., 
        ): 
        """ Set the geology or  geochemistry samples to fit the AGSO database 
        structures.
        
        Parameters 
        ------------
        kind: str, { "geology", "samples"}
           Kind of structure to be fitted.
           
        keep_acronym: bool, default=False, 
           Use the acronym of the structural and geological database 
           info to replace the full geological/structural name. 
           
        fill_value: str, optional 
           If None, the undetermined structured are not replaced. They are 
           kept in the data. However, if the fill_value is provided, the 
           missing structure from data is replaced by the fill_value. 
           
        inplace: bool, default=False 
          If ``True``, modifies the geology or samples dataset inplace and 
          return None 
      
        Returns
        --------
        data: pd.DataFrame 
           Geology or Geochemistry fitted data. Data are modified in place 
           if ``True``  and return ``None``. 
           
        Examples 
        ---------
        >>> from watex.geology import DSDrill 
        >>> dr = DSDrill()
        >>> dr.build_drillholes (kind='geology', return_data =True )
        Enter the well/hole ID:GEO02
        Enter GEO02 coordinates (x/lon, y/lat):12 26
        Enter GEO02 radius in meters [Optional]:3.6
        Enter the GEO02 depth in meters: 26
        Enter each stratum thickness of GEO02 [top-->bottom] in meters:7
        Enter the layer/rock names of GEO02 [top-->bottom]:limestone
        Enter valuable comments about GEO02 [Optional]:test
        Tap "exit|0" to terminate or "Enter" to continue:
        Enter the well/hole ID:GEO03
        Enter GEO03 coordinates (x/lon, y/lat):12 15
        Enter GEO03 radius in meters [Optional]:3.10
        Enter the GEO03 depth in meters: 36
        Enter each stratum thickness of GEO03 [top-->bottom] in meters:12 10
        Enter the layer/rock names of GEO03 [top-->bottom]:sand tuff
        Enter valuable comments about GEO03 [Optional]:test2
        Tap "exit|0" to terminate or "Enter" to continue:0
        Out[1]: 
          DH_Hole       DH_East       DH_North  DH_RH  DH_From  DH_To       Rock   Mask
        0   GEO02  2.879133e+06  199681.925199    3.6      0.0    7.0  limestone  test1
        1   GEO02  2.879133e+06  199681.925199    3.6      7.0   26.0         NA  test1
        2   GEO03  1.660514e+06  177349.038212    3.1      0.0   12.0       sand  test2
        3   GEO03  1.660514e+06  177349.038212    3.1     12.0   22.0       tuff  test2
        4   GEO03  1.660514e+06  177349.038212    3.1     22.0   36.0         NA  test2
        >>> dr.fit_structures('geology')
        Out[2]: 
          DH_Hole       DH_East       DH_North  ...  DH_To                      Rock   Mask
        0   GEO02  2.879133e+06  199681.925199  ...    7.0                 limestone  test1
        1   GEO02  2.879133e+06  199681.925199  ...   26.0  carbonate iron formation  test1
        2   GEO03  1.660514e+06  177349.038212  ...   12.0                 sandstone  test2
        3   GEO03  1.660514e+06  177349.038212  ...   22.0                      tuff  test2
        4   GEO03  1.660514e+06  177349.038212  ...   36.0  carbonate iron formation  test2
        """
        keep_acronyms, inplace = ellipsis2false(keep_acronyms, inplace)
        kind= self._assert_kind_value( kind) 
        
        if kind not in {"geology", "samples"}: 
            if self.verbose: 
                warn("Only 'geology' and geochemistry 'samples' are allowed "
                     "to be fitted.")
            return 
        if not hasattr ( self, f"{kind}_"): 
            raise DrillError(f"Missing '{kind}_' dataset. {kind.title()}"
                             f" cannot fitted. Set or build the {kind}"
                             " data first.")
        d = getattr ( self, f"{kind}_").copy()  
 
        if kind =='geology': 
            value = self.find_properties(
                data = d.Rock ,
                constraint=True, 
                keep_acronyms=keep_acronyms, 
                fill_value= fill_value , 
                )
            d.Rock = np.array( value )
        elif kind=='samples': 
            value = self.find_properties(
                data = d.Samples ,
                constraint=True, 
                keep_acronyms=keep_acronyms, 
                fill_value= fill_value , 
                kind='samples', 
                )
            d.Samples = np.array( value )
            
        if inplace: 
            setattr (self, f"{kind}_", d )
            d =None # reset to None 
            
        return d
    
    @export_data(type ='frame', encoding ='utf8')
    def out (
        self, kind: str, 
        fname: str=None,   
        format:str=".xlsx", 
        savepath: str=None, 
        **outkws: Any 
    ): 
        """ Export the drillhole. 
        
        Parameters 
        -------------
        kind: str, {"geology", "collar", "samples", "*", "data"}
          Kind of data to export. If ``"*"``, the geology, collar and 
          samples are exported in single excel sheets or csv formats if 
          `format` is set to ``.xlsx`` or ``.csv`` respectively.
          
         fname: str, 
            Full path to filename.  If `fname` is None, the filename used the 
            `kind` name. 
        format: str, default=".xlsx" 
          Extension to export the file. Note that if the  filename `fname` 
          includes the file extension, it should be used. 
        savepath: str, optional 
          Path to locate the files. 
          
        Outkws: dict, 
          Keywords arguments passed to other pandas exportable readable 
          formats. 
          
        Examples 
        ----------
        >>> from watex.geology import DSDrill 
        >>> from watex.datasets import load_nlogs 
        >>> ndata = load_nlogs(samples =7 , as_frame =True)
        >>> dr =DSDrill(projection='utm').fit(ndata)
        >>> dr.out( kind="data")
        >>> dr.out(fname = 'test_data.csv', kind ="data")
        
        """
       
        if kind !="*": 
            kind = key_search(kind, deep=True, raise_exception=True, 
                              default_keys=('geology', 'collar', 
                                            'samples', 'data'))
        else: kind = ['collar', 'geology', 'samples']
        # name of data  to overwrite the 
        # filename if not given or overwrite 
        # the sheetnames if .xlxs is required 
        nameof= kind.copy() 

        dfs = [getattr (self, f"{ki}_", None) for ki in kind]
        # remove None if exists in dfs 
        dfs = list ( filter (lambda o: o is not None, dfs )) 
        if len(dfs)==0: 
            raise TypeError("NoneType cannot be exported. Please check"
                            " your data.")
        # rename data is needed since the current directory 
        # hold the name data. This prevent moving the software 
        # data directory files 
        _fname = kind[0] if kind[0] !="data" else "data0" 
        fname =fname or (_fname if kind!="*" else "dhdata")
                         
        # check whether there is a dot.
        # remove it and add to the biginning
        format = "." + str(format).lower(
            ).strip().replace (".", "")
        
        return ( dfs, 
                fname, 
                format, 
                savepath,
                nameof, 
                outkws
                )  

    def _assert_kind_value (self, kk ): 
        """Assert value of kind. 
        Function returns valid  value of drillhole kind. Expect 
        ['collar'|'geology'|'samples']"""
        
        valid_keys = {'collar', 'geology', "samples", "elevation" }
        kk = key_search (kk, default_keys= valid_keys, deep=True )
        if kk is not None: 
            kk = kk[0] # take the first valid item 
        assert  str(kk).lower() in valid_keys, (
            f"Wrong value of kind. Expect {smart_format(valid_keys, 'or')}"
            f" Got {kk!r}")
        return str(kk).lower().strip() 

    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        _t = ("area", "holeid", "dname", "projection", "utm_zone", "encoding", 
              "datum", "epsg", "reference_ellipsoid" , "properties", "verbose")
        outm = ", ".join(
            [f"{k}={ False if getattr(self, k)==... else  getattr(self, k)!r}" 
             for k in _t]) + ">" 

        return "<{!r}:".format(self.__class__.__name__ ) + outm 

    def __getattr__(self, name):
       rv = smart_strobj_recognition(name, self.__dict__, deep =True)
       appender  = "" if rv is None else f'. Did you mean {rv!r}'
       
       err_msg =  f'{appender}{"" if rv is None else "?"}' 
       
       raise AttributeError (
           f'{self.__class__.__name__!r} object has no attribute {name!r}'
           f'{err_msg}'
           )

    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'data_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1
    

class Borehole(Geology): 
    """
    Focused on Wells and `Borehole` offered to the population. To use the data
    for prediction purpose, each `Borehole` provided must be referenced on 
    coordinates values or provided the same as the one used on `ves` or `erp` 
    file. 
    
    """
    def __init__(
        self,
        lat:float = None, 
        lon:float = None, 
        area:str = None, 
        status:str =None, 
        depth:float = None, 
        base_depth:float =None, 
        geol:str=None, 
        staticlevel:float =None, 
        airlift:float =None, 
        holeid=None, 
        qmax =None, 
        **kwds
        ): 
       super().__init__(**kwds)
        
       self.lat=lat 
       self.lon=lon 
       self.area=area 
       self.status=status 
       self.depth=depth 
       self.base_depth=base_depth 
       self.geol=geol 
       self.staticlevel=staticlevel 
       self.airlift =airlift 
       self.holeid=holeid 
       self.qmax=qmax 

       for key in list(kwds.keys()): 
           setattr (self, key, kwds[key])

    def fit(self,
            data: str |DataFrame | NDArray 
        )-> object: 
        """ Fit Borehole data and populate the corrsponding attributes"""
        
        self._logging.info ("fit {self.__class__.__name__!r} for corresponding"
                            "attributes. ")
    
        return self

#XXX TODO Next release 
class Drill ( GeoBase): 
    """ 
    Focused on drilling operations
    """
    def __init__(self,  **kwd): 
        super ().__init__(**kwd )
        
        warn("Module `Drill` is not available yet. Wait for the release v1.0")
        
class PumpingTest: 
    """ Focuses on Pumping test operations """
    def __init__( self, verbose: int =0 ):
        self.verbose=verbose
        warn("Module `PumpingTest` is not available yet. Wait for the"
             "release v1.0")
          
#---------------------------------Drilling  utilities -------------------------          
def _fit( o, /, 
    data =None, 
    drop_nan_columns =True, 
    input_name ='b',  
    **fit_params 
)->object: 
    """ Fit borehole and drill data and populate usefull attributes """
    columns = fit_params.pop ("columns", None  )
    
    data = _is_readable(data, as_frame =True, input_name= input_name, 
            columns = columns, encoding =o.encoding )
    data = check_array (
        data, 
        force_all_finite= "allow-nan", 
        dtype =object , 
        input_name="Borehe data", 
        to_frame=True, 
        )
    data, nf, cf = to_numeric_dtypes(
        data , 
        return_feature_types= True, 
        verbose =o.verbose, 
        sanitize_columns= True, 
        fill_pattern='_',
        drop_nan_columns=drop_nan_columns, 
        **fit_params 
        )
    o.feature_names_in_ = nf + cf 
    if len(cf )!=0:
        # sanitize the categorical values 
        for c in cf : 
            data[c] = data[c].str.strip() 
    for name in data.columns : 
        setattr (o, name, data[name])
        
    # set depth attributes 
    o.depth_= None 
    if o.dname is not None: 
        if 'depth' in o.feature_names_in_: 
            o.dname ='depth' 
    if o.dname  in o.feature_names_in_: 
        o.depth_= data[o.dname]
        
    o.data_ = data.copy() 
    
    return o 

