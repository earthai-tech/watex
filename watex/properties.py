# -*- coding: utf-8 -*-
#   Copyright (c) 2021  @Daniel03 <etanoyau@gmail.com>
#   Created date: Fri Apr 15 10:46:56 2022
#   Licence: MIT Licence 


class P: 
    """ Self container all the properties values that should not be 
     be modified. """
    
    
    _stations  = ['pk',
                  'sta',
                  'pos'
                  ]
    _easting  =['east',
                'x',
                'long'
                ]
    _northing  = [
        'north',
        'y',
        'lat'
        ]
    _resistivity = [
        'rho',
        'app',
        'res'
        ]
    _names= ['station',
             'easting',
             'northing',
             'resistivity' 
             ] 
    @property 
    def FRctags (self): 
        """ Flow colors tags"""
        return  dict ((f'fr{k}', f'#{v}') for k, v in zip(
                        range(4), ('CED9EF','9EB3DD', '3B70F2', '0A4CEF' )
                        )
        )

    @property 
    def Dtags (self): 
        """ Is the collection of data proertiesproperties.""" 
        return  dict ( (k, v) for k, v in zip(self.SENR,
              [self.STAp, self.EASTp,self.NORTHp , self.RESp ] ))

    @property 
    def STAp (self) : 
        """ Use prefix to identify station locations positions """
        return self._stations 
    
    @STAp .setter 
    def STAp (self, s) : 
        """ Modify the properties stations. Indeed, can onlky be added 
        and not deleted the items in the default list. """
        self._stations = add_prefix(s, self._stations) 
    
    @property 
    def EASTp (self): 
        """ Use prefix to identify easting coordinates if given in the
        dataset. """
        return self._easting 
    
    @EASTp.setter
    def EASTp(self, e): 
        """ Modify the easting prefix. Should top the given list to the 
        existing list."""
        self._easting = add_prefix(e, self._easting )
    
    @property 
    def NORTHp (self): 
        """ Use prefix to identify northing coordinates if given in the
        dataset. """
        return self._northing
    @NORTHp.setter
    def NORTHp (self, n): 
        """ Modify the northing prefix. Should top the given list to the 
        existing list."""
        self._northing= add_prefix(n, self._northing )
    
    @property 
    def RESp(self): 
        """ Use prefix to identify the resistivity values in the dataset"""
        return self._resistivity 
    @RESp.setter
    def RESp (self, e): 
        """ Modify the resistivity prefix. Should top the given list to the 
        existing list."""
        self._resistivity  = add_prefix(e, self._resistivity  )
    
    @property 
    def SENR(self): 
        """ `SENR` is the abbreviation of `S`for ``Stations``, `E` for ``Easting`, 
        `N` for ``Northing`` and `R``for resistivity. `SENR` is the expected 
        columns in Electrical resistivity profiling. Indeed, it keeps the 
        traditional collections sheets during the suyrvey. """
        return self._names 
  
def add_prefix (p, e): 
    """ Add prefix to existing properties values. """
    if isinstance (p, str): 
        e += [p] 
        return e 
    if isinstance(p, (list)): 
        return e + p 
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    