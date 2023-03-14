# -*- coding: utf-8 -*-

import unittest
from watex.site import  ( 
    Profile , 
    Location 
    ) 
from watex.datasets import load_edis 

 
# Generate coordinates x and y 
xy = load_edis(
    as_frame = True , 
    key ='latitude longitude' , 
    samples =7 
    ) 
# degree-decimals 
x = xy.latitude 
y = xy.longitude 

class TestProfile ( unittest.TestCase ): 
    """ Test site"""
    # make profile object 
    po = Profile ().fit(x ,  y )
    
    def test_distance(self ): 
        """ Test the fit method to chech whether its successfull run"""
        self.po.distance () 
        
    def test_bearing (self) :
        """ Compute bearing between coordinates points. """
        self.po.bearing(to_degree =True ) 
    
    def test_make_coordinates (self ): 
        """ Test how to generate a synthetic coordinates data """
        # use auto computation 
        lat, lon = self.po.make_xy_coordinates() 
        # use manula commputation 
        
        lat2, lon2 = self.po.make_xy_coordinates(
            sep = 20 , r = 45 , to_dms = True  )

        self.assertEqual(':' in lat2[0] , True )  
        self.assertEqual(':' in lon2[0] , True )   
        
    

class TestLocation (unittest.TestCase ): 
    """ Test Location class """ 
    loc = Location () 
    
    def test_to_utm_in (self ) : 
        """ test coordinate arrays conversion to latlon """ 
        xx , yy = self.loc.to_utm_in(x , y )
        
        self.assertEqual (xx, len(x)) 
        self.assertEqual (yy, len(xx ))