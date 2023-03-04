# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Created on Mon Sep 19 10:29:24 2022
"""
Manage station location data.
"""
import copy 
import re
import warnings  

import numpy as np 

from .utils.gistools import (
    assert_elevation_value, 
    assert_lat_value, 
    assert_lon_value , 
    ll_to_utm, 
    utm_to_ll, 
    project_points_ll2utm, 
    project_point_utm2ll
    
    )
from .exceptions import ( 
    SiteError 
    )
from ._watexlog import watexlog 
from .property import ( 
    UTM_DESIGNATOR
    )
class Location (object): 
    """
    Location class
    
    Assert, convert coordinates lat/lon , east/north  to approprite format 
  
    Arguments 
    ---------
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
    
    def __init__(self, lat = None, lon=None, **kwds) :
        self._logging = watexlog.get_watex_logger(self.__class__.__name__)
        
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
        
    def to_utm (self, lat=None  , lon=None, datum= None ,
                utm_zone=None, epsg=None, reference_ellipsoid= None): 
        """
        Project coordinates to utm if coordinates are in degrees at  
        given reference ellipsoid constrained to WGS84 by default. 
         
        Parameters 
        -----------
        lat: float or string (DD:MM:SS.ms)
            latitude of point
                  
        lon: float or string (DD:MM:SS.ms)
            longitude of point
        
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
            
        if epsg is not None: 
            self.epsg = epsg 
        if datum is not None: 
            self.datum = datum 
        else : 
            warnings.warn("Note that the reference ellipsoid is constrained "
                          "to 'WGS84'.")
            self.datum = 'WGS84'
        if reference_ellipsoid is not None: 
            self.reference_ellipsoid = reference_ellipsoid 
     
        try : 
            self.utm_zone, utm_easting, utm_northing= ll_to_utm(
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
            
        return self.east, self.north
    
        
    def to_latlon(self, east=None, north= None, utm_zone=None, 
                  reference_ellipsoid=None , datum = None ): 
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
        
        datum: Optional, string
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
        
        if datum is not None: 
            self.datum = datum 
        else : 
            warnings.warn("Note that the reference ellipsoid is constrained "
                          "to 'WGS84'.")
            self.datum = 'WGS84'
        if reference_ellipsoid is not None: 
            self.reference_ellipsoid = reference_ellipsoid 
            
                         
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
        
