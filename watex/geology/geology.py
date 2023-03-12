# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

import os
import re
import warnings 
import numpy as np 

import pandas as pd
from .core import (
    Base, 
    get_agso_properties 
    )
from ..utils.funcutils import ( 
    repr_callable_obj, 
    smart_strobj_recognition
    )
from ..utils.plotutils import ( 
    get_color_palette 
    )
from ..exceptions import ( 
    StrataError,
    NotFittedError, 
    )

from .._watexlog import watexlog  
_logger =watexlog().get_watex_logger(__name__)


def setstructures(self , configfile:str =None , fillna:int =0  ): 
    """ configure the geological structures as a property object and load 
    attributes.  
    
    :param configfile: is a configure file  from 'AGS0' data 
    :param fillna: fill NaN values in the AGS0 file. The default values to 
        fill is `0` for False. 
    
    :return: ``self`` for method chaining. 
    
    .. note:: 
        Each geological strutures can be retrieved as an attribute. For 
        instance to get the code, the label and the pattern density of the 
        'amphibolite', one can use:: 
            
            >>> from watex.geology import Structures 
            >>> sobj = Structures().fit() 
            >>> sobj.amphibolite.code 
            ... 'AMP'
            >>> sobj.amphibolite.label_
            ... 'AMP'
            >>> sobj.amphibolite.pat_density_
            ... nan # not set 
            >>> # To get all the key (attributes of the structures ), uses: 
            >>> sobj.keys 
            ... 
        
    """
    df= pd.DataFrame (get_agso_properties(configfile) ) 
    df.fillna (fillna, inplace =True)
    # rename columns 
    df.columns = df.columns.str.lower().map (
        lambda c: 'name' if c =='__description' else c )
    # collect name and sanitize attributes 
    regex =re.compile (r'[ -@*#&+/]', flags=re.IGNORECASE)
    keys = df.name.str.lower().map(lambda o: regex.sub('_', o))
    for kk , nn in enumerate (keys): 
        d= {f'{k}_': v for k, v in dict(df.iloc[kk, :]).items()}
        obj = type (nn, (), d)
        setattr(self, nn, obj )

    # set the keys and names as attributes 
    setattr(self,  'names_', tuple (keys.values))
    setattr(self, 'keys_', tuple(df.columns) )
    
    return self 

#XXXTODO
class Geology: 
    """ Geology class deals with all concerns the structures during 
    investigation sites"""
    
    def __init__(self, geofn = None,  **kwargs)-> None:
        self._logging =watexlog().get_watex_logger(self.__class__.__name__)
        pass
 
               
class Structures(Base): 
    """
    This class is an auxilliary class to supplement geodatabase , 
    if the GeodataBase doesnt reply  to SQL request  , then use this class
    to secah information about structures .  If SQL is done as well ,
    program won't call this class as rescure . 
    Containers of more than  150 geological strutures.

    ==================  ============  =========================================
    Attributes          Type           Explanation
    ==================  ============  =========================================
    names               array_like      names of all geological strutures 
    codes               array_like      names of all geological codes 
    **code              str             code of specific geological structure 
    **label             str             label of specific structure
    **name              str             label of specific structure
    **pattern           str             pattern of specific structure  
    **pat_size          str             pattern size  of specific structure
    **pat_density       str             pattern density l of specific structure
    **pat_thickness     str             pttern thickess of specific structure
    **color             str             color of specific structure
    ==================  ============  =========================================
    
    .. note::
        To get the attribute value, merely replace the param "**" by  the
        *name of struture* following by dot ".". See examples: 
    
    Examples 
    ----------
    * To get the names of different strutures, write the script below:: 
    
        >>> from watex.geology.geology import Structures
        >>> geo_structure = Structures().fit()
        >>> geo_structure.names_ # get the list of all geological strutures 
        
    * To extract color  and to get the code of structure  like `tonalite`:: 
    
        >>> from watex.geology.geology import Structures
        >>> sobj = Structures().fit()
        >>> sobj.tonalite.pat_thickness_
        ... 0.  # -> not implemented 
        >>> sobj.tonalite.code_ 
        >>> ...'TNL'
        >>> sobj.tonalite.color_
        ... ''RB128'
    """ 
    codef =['code','label','__description','pattern', 'pat_size',	'pat_density',
            'pat_thickness','color']

    def __init__(self, configfile =None , **kwds) :
        super().__init__(**kwds)
        
        self.configfile = configfile 
 
        for key in list(kwds.keys()): 
            self.__setattr__(key, kwds[key])
            
    
    @property 
    def coerce(self): 
        """ Force configuration if auto getting the property file fails."""
        codef =['code','label','__description','pattern', 'pat_size',	
                'pat_density','pat_thickness','color']
        
        if self.configfile is None : 
             self.configfile = os.path.join(
                 'watex/etc', 'agso'.upper() + '.csv' )
             
        
        if not os.path.isfile (self.configfile ): 
            raise FileNotFoundError("Structure property file not found!")
    
        with open(self.configfile, 'r', encoding ='utf8') as f: 
            data = f.readlines()
            h0 = data[0].strip().split(',')
            h0 =[_code.lower() for _code in h0]
            
            self.data_=data[1:]
            if sorted(h0) != sorted(codef):
                warnings.warn("Unable to decode the geological structures!")
                raise StrataError("Inappropriate structure file:"
                                  f" {os.path.basename (self.configfile)!r}")
     
        self.data_ = [ geos.strip().split(',') for ii, 
                           geos in enumerate(self.data_)]
        self.data_= np.array (self.data_)
        # sanitize the geonames . 
        self.__setattr__('names_',
                         np.array( [name.replace('"', '') 
                                    for name in self.data_[:, 2]] ))
        
        self.__setattr__('codes_', self.data_[:, 0])
        
        # set for all values in geofomations codes 
        # count starting from one .
        for ii, codeff in enumerate(self.codef [1:],1) :  
            if codeff.find('pat') >=0 : 
                for jj, pp in enumerate(self.data_[:, ii]): 
                    if pp == '' or pp ==' ' :       
                        # replace all None value by 0. 
                        # later will filled it 
                        self.data_[:, ii][jj]=0.
                        
                self.__setattr__(codeff, np.array(
                    [float(pp) for pp in self.data_[:, ii]]))
            else : 
               self.__setattr__(codeff,np.array(
                   [name.replace('"', '') for name in self.data_[:, ii]] )) 
        
        DECODE ={}
        # change RGBA color palette into Matplotlib color 
        mpl_colorsp= [ get_color_palette(rgb)
                           for rgb in self.data_[:,-1]]

        for jj,  decode  in enumerate(self.data_):
           for ii, codec  in enumerate( self.codef) : 
               if codec =='__description' : codec = 'name'
               if codec =='color': 
                   DECODE[codec]= mpl_colorsp[jj]
               else : DECODE [codec]= decode[ii]
               self.__setattr__(decode[2].strip(
                   ).replace('"', '').replace(' ', '_'),
                                DECODE)
               self.__setattr__(decode[0].strip(
                   ).replace('"', '').replace(' ', '_'),
                                DECODE)
               
           DECODE ={}  
           
        return self
    
    
    def fit(self, **kwd ): 
        """ Fit and set the geological strutures as object attributes and `kwd`
        arguments refer to the argument below. 
        
        Parameters 
        -----------
        
        configfile: str 
            is a configure file  from 'AGS0' data 
        fillna: float, 
            fill NaN values in the AGS0 file. The default values to fill is
            `0` for False. 
            
        Notes
        -----
        Each geological strutures can be retrieved as an attribute. For 
        instance to get the code, the label and the pattern density of the 
        'amphibolite'. 
        
        Example
        -------- 
        >>> from watex.geology import Structures 
        >>> sobj = Structures().fit() 
        >>> sobj.amphibolite.code 
        ... 'AMP'
        >>> sobj.amphibolite.label_
        ... 'AMP'
        >>> sobj.amphibolite.pat_density_
        ... 0. # not set 
        >>> # To get all the key (attributes of the structures ), uses:: 
        >>> sobj.keys 
                
        """
        try:
            setstructures(self, **kwd)
        except : self.coerce  
        
        return self 
    
    
    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        return repr_callable_obj  (self, skip = self.names_)
       
    def __getattr__(self, name):
        if name.endswith ('_'): 
            if name not in self.__dict__.keys(): 
                if name in (
                    'keys_', 'names_', 'label_', 'code_', 'pattern_', 
                    'pat_size_', 'pat_density_', 'pat_thickness_','color_'): 
                    raise NotFittedError (
                        f'Fit the {self.__class__.__name__!r} object first'
                        )
                
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        appender  = "" if rv is None else f'. Do you mean {rv!r}'
        
        raise AttributeError (
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{appender}{"" if rv is None else "?"}'
            )        
        
class Structural(Base) :
    """
    Geology strutural conventions class.
    
    Note that the given structural objects are quite less than the litterature 
    More structural object can be added as the structures is known. 
    All geological structural informations are geostructral object.
                  
    Holds the following informations:
        
    ==========================  ==========   ==================================
    Attributes                  Type             Explanation
    ==========================  ==========   ==================================
    boudin_axis                 geos_obj      boudin    
    fold_axial_plane            geos_obj      axial plan of structural fold.
    banding_gneissosity         geos_obj      gneissossity of boudin plan  
    s_fabric                    geos_obj      fabric plan
    fault_plane                 geos_obj      fault plan 
    fracture_joint_set          geos_obj      fracture joint 
    undifferentiated_plane      geos_obj      unnamed geological structure 
    sharp_contact               geos_obj      sharp contact `various discrepancy` 
                                              contact `stratigraphy discrepancy`  
                                              fracture or fault discrepancies
    ==========================  =========    ==================================

    More attributes can be added by inputing a key word dictionary

    :Example: 
        
    >>> from watex.geology import Structural 
    >>> s=Structural().fit() 
    >>> s.boudin_axis.code_ 
    ... 'lsb'
    >>> s.boudin_axis.name_
    ... 'Boudin Axis'
    >>> s.boudin_axis.color_
    ... 'R128GB'
        
    """  
    
    
    def __init__(self, configfile =None, **kwds):
        super().__init__(**kwds)
    
        self.configfile = configfile 
 
        for key in list(kwds.keys()): 
            self.__setattr__(key, kwds[key])
            
        pass 
    
    def fit(self, configfile= None , **kwd): 
        """ Configure the structural data and set each object as attributes """
        if self.verbose : 
            self._logging.info('Set structural main geological informations. ')
        is_prop=False 
        if configfile is not None: 
            self.configfile = configfile 
        if self.configfile is None: 
            self.configfile =  self.configfile = os.path.join(
                 'watex/etc', 'agso_stcodes'.upper() + '.csv' )
            # hide configfile if the one of the package is used  
            is_prop =True  
        setstructures(self, configfile =self.configfile,  **kwd)
        
        self.configfile = None if is_prop else self.configfile  
        
        return self 
        
    
    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        return repr_callable_obj  (self, skip = self.names_)
       
    def __getattr__(self, name):
        if name.endswith ('_'): 
            if name not in self.__dict__.keys(): 
                if name in (
                    'keys_', 'names_', 'label_', 'code_', 'pattern_', 
                    'pat_size_', 'pat_density_', 'pat_thickness_','color_'): 
                    raise NotFittedError (
                        f'Fit the {self.__class__.__name__!r} object first'
                        )
                
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        appender  = "" if rv is None else f'. Do you mean {rv!r}'
        
        raise AttributeError (
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{appender}{"" if rv is None else "?"}'
            )        
        
    
    
        
     
 
        


