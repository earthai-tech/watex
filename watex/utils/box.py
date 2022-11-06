# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 14:52:26 2022

@author: Daniel
"""
class Boxspace(dict):  
    """Is a container object exposing keys as attributes.
    
    BowlSpace objects are sometimes used as an output for functions and methods.
    They extend dictionaries by enabling values to be accessed by key,
    `bunch["value_key"]`, or by an attribute, `bunch.value_key`.
    Another option is to use Namespace of collection modules as: 
        
        >>> from collections import namedtuple
        >>> Boxspace = namedtuple ('Boxspace', [< attaributes names >] )
        
    However the explicit class that inhers from build-in dict is easy to 
    handle attributes and to avoid multiple error where the given name 
    in the `names` attributes does not match the expected attributes to fetch. 
    
    Examples
    --------
    >>> from watex.utils.import Boxspace 
    >>> bs = Boxspace(pkg='watex',  objective ='give water', version ='0.1.dev')
    >>> bs['pkg']
    ... 'watex'
    >>> bs.pkg
    ... 'watex'
    >>> bs.objective 
    ... 'give water'
    >>> bs.version
    ... '0.1.dev'
    """

    def __init__(self, **kws):
        super().__init__(kws)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        # Overriding __setstate__ to be a noop has the effect of
        # ignoring the pickled __dict__
        pass