# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Is a set of utilities that deal with geological rocks, strata and 
stratigraphic details for log construction. 
"""
from __future__ import annotations 
import os
import re 
import itertools
import warnings
import copy 
import numpy as np
import pandas as pd 

from .._watexlog import watexlog 
from .._typing import ( 
    List, 
    Tuple, 
    ArrayLike, 
    Any
    )
from ..exceptions import ( 
    StrataError, DepthError )
from ..property import Config 
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
from .funcutils import ( 
    _assert_all_types, 
    smart_format, 
    station_id, 
    convert_value_in, 
    str2columns, 
    is_iterable, 
    ellipsis2false, 
    )
from .exmath import find_closest 
_logger = watexlog().get_watex_logger(__name__ )


def find_similar_structures(
        *resistivities,  return_values: bool=...):
    """
    Find similar geological structures from electrical rock properties 
    stored in configure data. 
    
    Parameters 
    ------------
    resistivities: float 
       Array of geological rocks resistivities 
       
    return_values: bool, default=False, 
       return the closest resistivities of the close structures with 
       similar resistivities. 
       
    Returns 
    --------
    structures , str_res: list 
      List of similar structures that fits each resistivities. 
      returns also the closest resistivities if `return_values` is set 
      to ``True``.
    
    Examples
    ---------
    >>> from watex.utils.geotools import find_similar_structures
    >>> find_similar_structures (2 , 10, 100 , return_values =True)
    Out[206]: 
    (['sedimentary rocks', 'metamorphic rocks', 'clay'], [1.0, 10.0, 100.0])
    """
    if return_values is ...: 
        return_values =False 
    
    def get_single_structure ( res): 
        """ get structure name and it correspoinding values """
        n, v = [], []
        for names, rows in zip ( stc_names, stc_values):
            # sort values 
            rs = sorted (rows)
            if rs[0] <= res and res <= rs[1]: 
            #if rows[0] <= res <= rows[1]: 
                n.append ( names ); v.append (rows )
                
        if len(n)==0: 
            return "*Struture not found", np.nan  
        
        v_values = np.array ( v , dtype = float ) 
        
        close_val  = find_closest ( v_values , res )
        close_index, _= np.where ( v_values ==close_val) 
        # take the first close values 
        close_index = close_index [0] if len(close_index ) >1 else close_index 
        close_name = str ( n[int (close_index )]) 
        # remove the / and take the first  structure 
        close_name = close_name.split("/")[0] if close_name.find(
            "/")>=0 else close_name 
        
        return close_name, close_val 
    
    #---------------------------
    # make array of names structures names and values 
    dict_conf =  Config().geo_rocks_properties 
    stc_values = np.array (list( dict_conf.values()))
    stc_names = np.array ( list(dict_conf.keys() ))

    try : 
        np.array (resistivities) .astype (float)
    except : 
        raise TypeError("Resistivities array expects numeric values."
                        f" Got {np.array (resistivities).dtype.name!r}") 
        
    structures = [] ;  str_res =[]
    for res in resistivities: 
        struct, value = get_single_structure(res )
        structures.append ( struct) ; str_res.append (float(value) )

    return ( structures , str_res ) if  return_values else structures 


def smart_thickness_ranker ( 
    t: str | List[Any, ...], /,  
    depth:float=None, 
    regex:re= None, 
    sep:str= ':-', 
    surface_value:float=0.,  
    mode:str='strict', 
    return_thickness:bool=...,
    verbose:bool=..., 
    )-> Tuple[ArrayLike]: 
    """Compute the layer thicknesses and rank strata accordingly.
    
    Grub from the litteral string the layer depth range to find the ranking 
    of layer thickness. 
    
    Parameters 
    -----------
    t: str, or List of Any  
       Litteral string that containing the data arrangement. The kind of data
       to provide for thickness arrangement are:
           
      - t-value: Compose only with the layer thickness values. For instance 
        ``t= "10 20 7 58"`` indicates four layers with layer thicknesses 
        equals to 10, 20, 7 and 58 ( meters) respectively. 
        
      - tb-range: compose only with thickness range at each depth. For instance
        ``t= "0-10 10-30 40-47 101-159"``. Note the character used to separate 
        thickness range is ``'-'``. Any other character must be specified using 
        the parameter `sep`. Here, the top(roof) and bottom(wall) of the layers
        are 0  (top) and 10 (bottom), 10 and 30, 40 and 47 , and 101 and 159 
        for stratum 1, 2, 3 and 4 respectively.
        
      - mixed: Mixed data kind is composed of the both `t-value` and `tb-range`.
        When this kind of data is provied, to smartly parse the data, user 
        must set the operation `mode` to ``soft``. However, to avoid any 
        unexpected result, it is suggested to used either `t-value` or 
        `tb-range` layer thickness naming.
        
    depth: float, optional 
        Depth is mostly used when `t-value` thickness arrangement is provided. 
        It add additional layer at the bottom the  given thickness 
        and recompute the last layer thickness. Howewer for a sampling as 
        geochemistry sampling, depth specification is not necessary. 
       
    regex: `re` object,
       Regular expresion object used to parse the litteral string `v`. If not 
       given, the default is:: 
            
       >>> import re 
       >>> re.compile (r'[_#&.)(*@!,;\s-]\s*', flags=re.IGNORECASE)
          
    sep:str, default= ':-'
       The character used to separate two layer thickness ranged from top to 
       bottom. Any other character must be specified. Here is an example:: 
           
           >>> sep ='10-35' or sep='10:35'
       
    surface_value: float, default=0. 
      The top value of the first layer. The default is the sea level. For 
      instance, if the first layer `l0` is ``10m`` thick, the top (roof) and 
      the bottom(wall) of `l0` should be ``0-10`` for ``surface_value=0.``. 
      
    return_thickness: bool, default=False 
      If ``True``, return the calculated thickness of each stratum. 
      
    mode: str, default='strict' 
      Control the layer thickness ranking. It can be ['soft'|'strict']. Any 
      other value should be in 'soft' mode. Indeed, the mode is used to 
      retrieve, arrange and compute the layer thicknesses. For instance, 
      in ``strict`` mode, any bad arrangement or misimputed layer thicknesses 
      should raise an error. However, in 'soft', the bad arrangements
      are systematically dropped especially when top and bottom values of 
      the layers are null.
      
   verbose: bool, default=False 
      Warn user about the layer ranking and thickness calculation. 
  
    Returns 
    --------
    dh_from, dh_to| thickness: Tuple of Arraylike 
      - dh_from: Arraylike of each layer roof ( top) 
      - dh_to: Arraylike of each layer wall ( bottom) 
      - thickness: Arraylike of composed of each stratum thickness. Values 
        are returned if ``returun_thickness=True``. 
        
    Examples 
    --------
    >>> from watex.utils.geotools import smart_thickness_ranker
    >>> smart_thickness_ranker ("10 15 70 125")
    (array([ 0., 10., 25., 95.]), array([ 10.,  25.,  95., 220.]))
    >>> smart_thickness_ranker ("10 15 70 125", depth =300, 
                                return_thickness= True)
    (array([  0.,  10.,  25.,  95., 220.]),
     array([ 10.,  25.,  95., 220., 300.]),
     array([ 10.,  15.,  70., 125.,  80.]))
    >>> smart_thickness_ranker ("10-15 70-125")
    (array([10., 70.]), array([ 15., 125.]))
    >>> smart_thickness_ranker ("10-15 70-125", depth =300)
    (array([ 10.,  70., 125.]), array([ 15., 125., 300.]))
    >>> smart_thickness_ranker ("7 10-15 13 70-125 ",mode='soft')
    (array([ 0., 10., 15., 70.]), array([  7.,  15.,  28., 125.]))
    >>> smart_thickness_ranker ("7 10-15 13 70-125 ",depth =300, mode='soft',
                                return_thickness=True)
    (array([  0.,  10.,  15.,  70., 125.]),
     array([  7.,  15.,  28., 125., 300.]),
     array([  7.,   5.,  13.,  55., 175.]))
    """
    # set ellipsis to false 
    return_thickness, verbose = ellipsis2false(return_thickness, verbose)

    mode=str(mode).lower().strip() 
    # check iterbale object 
    # convert items to strings 
    if is_iterable ( t, exclude_string= True ): 
        t=  ' '.join ( [str(it) for it in t ])
    # for consistency reconvert to string 
    t = str(t) 
    
    # check whether there is a mixture types.
    # whether the thickness separator is included 
    # in the default parsing characters then remove
    # it before transforming the litteral string to 
    # columns.
    tr = str2columns(t, regex =re.compile (
        re.sub(rf"[{sep}]", "", rf'[#&*@!,;\s{sep}]\s*'), flags=re.IGNORECASE))

    # check whether all values are passed 
    # as layer thickness t-values or tb-range
    count =[]
    data_kind ="t-value" 
    
    for it in tr : 
        try: float ( it)
        except:continue 
        else:count.append(it ) 
        
    if len(count)==0: 
        # assume all values are given as 
        # a layer top-bottom range 
        data_kind="tb-range" 
    elif len(count) != len(tr): 
        data_kind="mixed"
        
    if data_kind=="mixed": 
        # if mode is strict, mixing is not 
        # tolerable
        msg = ("Mixed thickness entries is detected. The ranking may yield" 
               " to unexpected results." ) 
        if mode=='strict': 
            raise StrataError(msg + " It is recommended to use for each"
                " stratum either the top-bottom naming <tb-range> or only"
                " thickness value <t-value>.")
            
        if verbose: 
            warnings.warn(msg + 
            " In soft mode, the smart parser will be used to rank the layer"
            " thicknesses however it cannot handle any bad arrangement."
            " Use at your own risk.") 
        
        t = _make_thick_range (t, sep =sep  )
        # go to top-bottom range. 
        data_kind="tb-range"
        
    if data_kind=="tb-range": 
        from_to, thickness = get_thick_from_range(t, 
                              sep = sep , 
                              mode=mode ,
                              raise_warn=verbose, 
                              )
        
        from_to  = np.vstack (from_to) 
        dh_from , dh_to =from_to [:, 0 ], from_to [:, 1 ]
        
        if depth is not None: 
            depth =_assert_all_types(
                depth, int, float , objname='Well/hole depth')
            
            if dh_to[-1] >= depth: 
                if mode=='strict': 
                    raise StrataError (
                        f"Depth {depth} cannot be less than the last layer"
                        f" bottom depth {dh_to[-1]}. Please check your the"
                        " range of values used to specify the strata"
                        " thickness 'top-bottom'. "
                        )  
            else: 
                # add depth and compute 
                # the thickness at the bottom 
                dh_from = np.append( dh_from , dh_to[-1]) 
                dh_to = np.append (dh_to, depth )
                thickness = np.append ( thickness , depth - dh_from[-1])
            
    else: 
        dh_from, dh_to , thickness = get_thick_from_values(
                                    tr, depth=depth, raise_warn=verbose, 
                                    surface_value= surface_value, 
                                    mode= mode, 
                                 )
    
    return ( dh_from, dh_to, thickness
            ) if return_thickness else (dh_from, dh_to)

def get_thick_from_values(
    thick:List[float], 
    depth:float=None,
    surface_value:float =0., 
    mode:str='strict', 
    raise_warn:bool =... , 
    )->Tuple[ArrayLike]: 
    """ Compute thickness when only thick is given. 
    
    Here it is respectful of tthe <t-value> data kind. For t-range data, 
    refer to :func:`get_thick_from_range`. 

    Parameters 
    ------------
    thick: List,
      The list of layer thickness. 
      
    depth: float, optional 
       The maximum depth of the borehole. Useful when it comes to describes 
       the stratigraphic log in the borehole. However, it is not necessary 
       when it comes for geochemistry sampling. 
       
    surface_value: float, default=0. 
      The level of the the sea by default. It correspond to the depth of the 
      roof (top) of the first layers.
      
    mode: str, default='strict' 
      Control the layer thickness ranking. It can be ['soft'|'strict']. Any 
      other value should be in 'soft' mode. Here the ``strict`` mode yields 
      an error when the total sum of the thickness is greater than the 
      given depth. However in ``soft`` mode, the depth is merely replaced 
      by the total thick  depth. 
      
    raise_warn: bool, default=False 
      warn user when the given depth is less than the total layer thicknesses. 
      
    Returns 
    --------
    dh_from, dh_to , thickness: Tuple of Arraylike 
      - dh_from: the array of layer roof (top) depth values 
      - dh_to: The array of layer wall ( bottom) depth values 
      - thickness: The calculated thickness of each layer. 
      
    Examples
    ---------
    >>> from watex.utils.geotools import _parse_only_thick 
    >>> _parse_only_thick ([12 , 20, 26, 50 ], depth = 205 )
    Out[63]: 
    (array([  0.,  12.,  32.,  58., 108.]),
     array([ 12.,  32.,  58., 108., 205.]),
     array([12., 20., 26., 50., 97.]))
    
    """
    if raise_warn is ...: 
        raise_warn=False 
    
    if isinstance( thick, str): 
        thick= str2columns(thick)
    
    try: thick = np.array ([ convert_value_in (t)  for t in thick] 
                           ).astype (float)
    except: raise TypeError ("Value for thickness ranking should be numeric."
                             f" Got {np.array(thick).dtype.name!r}")
    
    if depth is not None: 
        depth = convert_value_in(depth )
        if thick.sum() > depth: 
            msg=(f"Expect maximum depth equal to {thick.sum()}. Got {depth}.") 
            if mode=='strict': 
                raise DepthError(msg)
            if raise_warn: 
                warnings.warn( msg )
            
            depth =None 
            # depth = thick.sum() 
    # check_thickness and reset depth if 
    # start always  the layer demarcation from 0 
    # at the surface 
    cum_and_depth = list(np.cumsum (thick ))  + [depth] if depth is not None\
        else np.cumsum (thick ) 
        
    from_to = np.array ( [surface_value] + list( cum_and_depth) )    
    
    dh_from, dh_to  = from_to [: -1 ], from_to [ 1: ]
    # append NA to the rocks name 
    # ad_NA = [ 'NA' for i in range ( len(from_to) )]
    # dh_samples +=ad_NA 
    thickness = np.diff (dh_from ) 
    if depth is not None: 
        # rather than using np.nan 
        thickness = np.append ( thickness, depth - dh_from.max() )
    # dh_samples = dh_samples[: len(dh_from)] 

    return dh_from, dh_to , thickness

def get_thick_from_range( 
    rthick:str, / , 
    sep:str=":-" , 
    flatten_range:bool=False , 
    mode:str='strict', 
    raise_warn: bool=True, 
    )-> Tuple[List[ArrayLike], ArrayLike]: 
    """Computes the thickness from depth range passed in litteral string.
    
    Collect values where thickness range is explicitly specified then  
    compute the thickness. Note that if `sep` is not supplied or empty, value 
    can yield an expected bad thickness calculation. 
    Note that when the layer thicknesses are given as numeric values only, 
    uses :func:`get_thick_from_values` instead.

    Parameters 
    -----------
    rthick: str 
      Text value of thick range composed of layer thickness declaration. 
      Each layer depth range must be separated with spaces. For instance 
      the depth range of layer A and B should be:: 
           
          v='42-52 63-85'
          
      where ``'42-52'`` is the layer A thickness range with 42 (m) as the top 
      and 52(m) the bottom i.e. the thickness of layer A equals to 10m. Idem, 
      the thickness of layer B equals to 22(m) with top starts at 
      63 (m) and end at 85 (m) (bottom). 
      
    sep: str, default=':-' 
      The separator used to differenciate the top and bottom values of 
      the layer. For example:: 
           
          sep =':-' --> '25:50' or '25-50' 
           
      where 25 and 50 corresponds to top (roof) and bottom ( wall) of the 
      layer/stratum respectively. Both ``'':-'`` are valid to make this 
      distinction. Any other character can be used provided that it is specified
      as an argument of `sep`parameter. 
      
    flatten: bool, default=False, 
      If ``True`` return the value of layer top and bottom into a single 
      list. For example:: 
          
         ['20-33', '58:125']-> [['20', '33'], ['58', '125']] -->\
              ['20', '33', '58', '125']
      
    mode: str, bool ='strict'
      Mode to retrieve, arrange and compute the layer thicknesses.
      In ``strict`` mode, any bad arrangement of misimputed of thickness 
      values should raise an error. However, in 'soft', the bad arrangement
      is systematically dropped especially when top and bottom values of 
      the layers are null. 
       
    raise_warn: bool, default=True 
      Warn user about the layer ranking and thickness calculation. 
      
    Returns 
    -------- 
    thick_range : List of layer thickness flattened or not 
    thickness: list - The calculated thickness of each stratum. 
    
    Examples
    ----------
    >>> from watex.utils.geotools import get_thick_from_range 
    >>> get_thick_from_range ('20-33 58:125', sep =':-') 
    Out[88]:
    ([array([20., 33.]), array([ 58., 125.])], [13.0, 67.0])
    >>> # when mixed values  are given 
    >>> get_thick_from ( "99 0-15 15.2-18.8 40.0-70.7", mode='soft')
    Out[89]: 
    ([array([ 0., 15.]), array([15.2, 18.8]), array([40. , 70.7])],
     [15.0, 3.6000000000000014, 30.700000000000003])
    """ 
    rthick = str(rthick) # for consistency 
    # we assume that value given here is compose of separator 
    # mean each layer is given from top bottom. 
    # any other values should not be considers.
    # e.g. layer A : 25-35 --> Top ( 25m), bottom (35m)-> thick = 10 meters. 

    thick_range_str = re.findall(rf"\d+(?:\.\d+)?[{sep}]\d+(?:\.\d+)?",
                             rthick, flags= re.IGNORECASE )
    # then break each of them and compute thickness 
    if len(thick_range_str)==0: 
        raise StrataError ("Stratum thicknesses expect numerical values."
                          f" Got {rthick!r}")
    sep_thick_range= [ str2columns ( it, regex = re.compile (
        rf'[{sep}]', flags = re.IGNORECASE )) for it in thick_range_str]

    # use lambda to transform value in float 
    try : 
        thickness = list ( map ( lambda x : np.diff ( np.array ( x).astype (
            float))[0] , sep_thick_range )
            ) 
    except BaseException as e : 
        raise TypeError (str(e) + ". Value range should be numeric separated"
                         " by a non-alphanumeric character which is explicitly"
                         " specify through the `sep` parameter.")

    thick_range, thickness =_thick_range_asserter ( 
        thickness= thickness , 
        depth_range= sep_thick_range, 
        litteral_string= thick_range_str, 
        mode=mode, 
        raise_warn= raise_warn, 
        )
    if flatten_range : 
        thick_range = list( itertools.chain ( *thick_range )) 
        
    return thick_range, thickness 

def _make_thick_range ( v ,/,  sep='-:' , tostr=True )-> str| list : 
    """ Make thick range from mixed thickness types. 
    
    Parameters 
    ------------
    v: str, 
       Value of mixed thickness types. The mixed types is composed of 
       layer thickness trend ( bottom - top) and the thick range (top to bottom)
       separated by the thickness separator `sep`. 
       
    sep: str, default='-:'
      The character used to separated layer top (roof) and bottom (wall) 
      
    tostr: bool, default=True 
      returns range value as a litteral string separated by a space 
      otherwise keep it in a list.
  
    Return
    --------
    thcik_range: list 
      List of string composed of layer thickness ranges. 
      
    Examples
    ---------
    >>> from watex.utils.geotools import _make_thick_range 
    >>> _make_thick_range ("99 0-15 15.2-18.8 55 40.0-70.7")
    Out[97]: '0-99.0 0-15 15.2-18.8 18.8-73.8 40.0-70.7'
    >>> _make_thick_range ("99 0-15 15.2-18.8 55 40.0-70.7", tostr=False)
    Out[98]: ['0-99.0', '0-15', '15.2-18.8', '18.8-73.8', '40.0-70.7']  
    >>> 
    
    """
    v= str(v)
    default_pattern = rf'[#&*@!,;\s{sep}]\s*'  
    # remove the pattern use to identify top-bottom 
    # layer into the default pattern.
    default_pattern = re.sub(rf"[{sep}]", "", default_pattern) 
    thick_range = str2columns(
        v, regex =re.compile ( default_pattern, flags=re.IGNORECASE) )

    # Now construct thick ranges 
    for k, item   in enumerate( thick_range): 
        try : 
            item = float( item)
        except: 
            continue 
        else: 
            if k==0: 
                thick_range[0] = f'0-{item}'
            else: 
                # get the previous values
                # split it and take second value 
                # convert it to float and add to the given one
                split_value = str2columns(
                    thick_range[k-1], regex = re.compile (
                        rf'[{sep}]', flags=re.IGNORECASE))
                # compute the layer bottom and set 
                # the thick range. 
                bv= float( split_value[-1]) + item
                thick_range[k]= f"{split_value[-1]}-{bv}"
  
    return ' '.join (thick_range ) if tostr else thick_range 
      
def _thick_range_asserter (
        thickness, 
        depth_range, 
        litteral_string, 
        mode='strict', 
        raise_warn=True 
        ): 
    """ Assert whether the thick range is correctly prompted. 
    
    An isolated part of :func:`get_thick_from`. Note that depth range 
    expected to be separated by non-alphanumeric character. 
    
    Parameters 
    -----------
    thickness: list , 
       Thickness of the depth range. Length is equal to the number of layer 
       depth ranges. 
    depth_range: list of list
       The depth range ( without the thickness separator) in a sub-list. For 
       instance the depth range of two layers can be:: 
           
           [ [ 0, 10], [10, 20 ]]
           
    litteral_string: list of str, 
       The list of the string that compose the depth range including the 
       thickness separator. For instance, the litteral string of two 
       layers with a separator ('-')should be::
           
           [ [0-10 ], [10-20 ]]
           
    mode: str, bool ='strict'
       Mode to retrieve, arrange and compute the layer thicknesses.
       In ``strict`` mode, any bad arrangement of misimputed of thickness 
       values should raise an error. However, in 'soft', some a bad arrangement
       is dropped especially when top and bottom values of the layers are 
       null. 
       
    raise_warn: bool, default=True
      Warn user about the layer arrangement and thickness calculation. 
      
    Returns 
    -------- 
    depth_range, thickness: List of array, list of float 
        Valid depth ranges and layer thickness computed from depth range. 

    Examples 
    ----------
    >>> from watex.utils.geotools import _thick_range_asserter
    >>> _thick_range_asserter ( thickness=[10 , 20 ], 
                               depth_range=[ [ 0, 10], [10, 30 ]], 
                               litteral_string=[ "0-10" , "10-30" ] 
                               )
    >>> Out[9]: ([array([ 0., 10.]), array([10., 30.])], [10, 20])
    >>> _thick_range_asserter ( thickness=[10 , -20 ], 
                               depth_range=[ [ 0, 10], [30, 10 ]], 
                               litteral_string=[ "0-10" , "30-10" ] 
                               )
    StrataError: (...)Please check the following stratum: '30-10' 
    >>> _thick_range_asserter ( thickness=[10 , 0 ], 
                               depth_range=[ [ 0, 10], [10, 10 ]], 
                               litteral_string=[ "0-10" , "10-10" ] , 
                               mode='soft'
                               )
    UserWarning: (...) layer should be dropped.
    Out[16]: ([array([ 0., 10.])], [10])
    """
    # first check whether thickness is equal to zero 
    # that means the top and bottom is a sampe 

    idx0, = np.where  ( np.array ( thickness )==0. )
    idx_neg, = np.where  ( np.array ( thickness ) < 0. ) 
    
    if len(idx_neg)!=0: 
        neg_thick = [litteral_string[ii] for ii in idx_neg  ]
        sname="strata" if len(idx_neg) > 1 else 'stratum'
        raise StrataError("Bad arrangement of stratum thicknesses. The Layer"
                          " roof(top) cannot be greater than the wall(bottom)."
                          f" Please check the following {sname}:"
                          f" {smart_format(neg_thick)} ")
    if len(idx0 ) !=0 : 
        bad_thick = [litteral_string[ii] for ii in idx0  ]
        if mode=='strict': 
            raise StrataError("Layer thickness from top to bottom cannot be equal"
                              " to null. Please check the following thickness"
                              f" range: {smart_format(bad_thick)}")
        
        if raise_warn: 
            warnings.warn ("A layer with thickness equals to null is detected at"
                           f" at {smart_format(bad_thick)}. In soft mode, layer"
                           " should be dropped.")
        
        [ thickness.pop (i) for i in idx0 ] 
        [ depth_range.pop (i) for i in idx0] 
        
    # now check whether thcinkess are ordered given. 
    # for consisteny convert values to float 
    depth_range = list ( map ( lambda x: np.array(x).astype (float),
                              depth_range)) 
    #then ravel depth range 
    dr =  list( itertools.chain ( *depth_range ))

    if ''.join( np.array(dr).astype(str) ) != ''.join(
            [str(d) for d in sorted(dr)]) : 
        # then iterate to find the bad 
        bad_arrangement = []
        msg= "Bad layer arrangement is observed at thickness ranges: {}"
        for k  in range ( len(dr)-1): 
            if dr[k] > dr [k +1]: 
                bad_arrangement.append ( str(dr[k]) +'-'+ str(dr[k+1]))
        
        if mode=='strict': 
            raise StrataError ( msg.format( smart_format ( bad_arrangement)) )
        else: 
            if raise_warn: 
                warnings.warn( msg.format( smart_format ( bad_arrangement)) + 
                              ". The automatic-arrangement is used instead.")
            dr = sorted ( dr ) 
            # splitback since it work in pair 
            depth_range = [ list(ar) for ar in np.split(
                np.array(dr), len(dr )//2 )] 
            
            # then recompute the thickness 
            thickness = list ( map ( lambda x : np.diff ( np.array ( x).astype (
                float))[0] , depth_range )
                ) 
    # round thickness 
    return depth_range, list(map ( lambda x: round (x, 3), thickness ))  

    
def build_random_thickness(
    depth, / , 
    n_layers=None, 
    h0= 1 , 
    shuffle = True , 
    dirichlet_dist=False, 
    random_state= None, 
    unit ='m'
): 
    """ Generate a random thickness value for number of layers 
    in deeper. 
    
    Parameters 
    -----------
    depth: ArrayLike, float 
       Depth data. If ``float`` the number of layers `n_layers` must 
       be specified. Otherwise an error occurs. 
    n_layers: int, Optional 
       Number of layers that fit the samples in depth. If depth is passed 
       as an ArrayLike, `n_layers` is ignored instead. 
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
      
    unit: str, default='m' 
      The reference unit for generated layer thicknesses. Default is 
      ``meters``
      
    Return 
    ------ 
    thickness: Arraylike of shape (n_layers, )
      ArrayLike of shape equals to the number of layers.
      
    Examples
    ---------
    >>> from watex.utils.geotools import build_random_thickness 
    >>> build_random_thickness (7, 10, random_state =42  )
    array([0.41865079, 0.31785714, 1.0234127 , 1.12420635, 0.51944444,
           0.92261905, 0.6202381 , 0.8218254 , 0.72103175, 1.225     ])
    >>> build_random_thickness (7, 10, random_state =42 , dirichlet_dist=True )
    array([1.31628992, 0.83342521, 1.16073915, 1.03137592, 0.79986286,
           0.8967135 , 0.97709521, 1.34502617, 1.01632075, 0.62315132])
    """

    if hasattr (depth , '__array__'): 
        max_depth = max( depth )
        n_layers = len(depth )
        
    else: 
        try: 
            max_depth = float( depth )
        except: 
            raise DepthError("Depth must be a numeric or arraylike of float."
                             f" Got {type (depth).__name__!r}")

    if n_layers is None: 
        raise DepthError ("'n_layers' is needed when depth is not an arraylike.")

    layer0 = copy.deepcopy(h0)

    try: 
        h0= convert_value_in (h0 , unit=unit)
    except : 
        raise TypeError(f"Invalid thickness {layer0}. The thickness for each"
                        f" stratum should be numeric.Got {type(layer0).__name__!r}")

    thickness = np.linspace  (h0 , max_depth, n_layers) 
    thickness /= max_depth 
    # add remain data value to depth. 
    if  round ( max_depth - thickness.sum(), 2)!=0: 
        
        thickness +=  np.linspace (h0, abs (max_depth - thickness.sum()),
                                   n_layers )/thickness.sum()
    if dirichlet_dist: 
        if random_state: 
            np.random.seed (random_state )
        if n_layers < 32: 
            thickness= np.random.dirichlet (
                np.ones ( n_layers), size =n_layers) 
            thickness= np.sum (thickness, axis = 0 )
        else: 
            thickness= np.random.dirichlet (thickness) 
            thickness *= max_depth  
    
    if shuffle: 
        ix = np.random.permutation (
            np.arange ( len(thickness)))
        thickness= thickness[ix ]
  
    return thickness 


def lns_and_tres_split(ix,  lns, tres):
    """ Indeed lns and tres from `GeoStratigraphy` model are updated. 
    
    Then splitting the `lns` and `tres` from the topped up values is necessary.
    Kind to resetting `tres` and `ln `back to original and return each split
    of inputting layers and TRES and the automatic rocks topped up
    during the NM construction.
    
    :param ix: int 
        Number of autorocks added 
    :param lns: list 
        List of input layers
    :param tres: list 
        List of input resistivities values.
    """
    if ix ==0: return  lns, tres,[], []
    return  lns[:-ix], tres[:-ix], lns[-ix:], tres[-ix:]   
   

def get_closest_gap (value, iter_obj, status ='isin', 
                          condition_status =False, skip_value =0 ):
    """ Get the value from the minimum gap found between iterable values.
    
    :param value: float 
        Value to find its corresponding in the `iter_obj`
    :param iter_obj: iterable obj 
        Object to iterate in oder to find the index and the value that match 
        the best `value`. 
    :param condition_status:bool 
        If there is a condition to skip an existing value in the `iter_obj`, 
        it should be set to ``True`` and mention the `ship_value`. 
    :param skip_value: float or obj 
        Value to skip existing in the `iter_obj`. 
        
    :param status:str 
        If layer is in the databse, then get the electrical property and 
        from that properties find the closest value in TRES 
        If layer not in the database, then loop the database from the TRES 
        and find the auto rock name from resistivity values in the TRES
        
    :return: 
        - ix_close_res: close value with its index found in` iter_obj`
    :rtype:tuple 
    
    """
    minbuff= np.inf 
    ix_close_res =None
    in_database_args = ['isin' , 'in', 'on', 'yes', 'inside']
    out_database_args= ['outoff' , 'out', 'no', 'isoff']
    if status.lower() in in_database_args:
        status ='isin'
    elif status.lower() in out_database_args: 
        status ='isoff'
    else: 
        raise ValueError(f"Given argument `status` ={status!r} is wrong."
                         f" Use arguments {in_database_args} to specify "
                         "whether rock name exists in the database, "
                         f"otherwise use arguments {out_database_args}.")

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', category=RuntimeWarning)
        for i, v in enumerate(iter_obj):
            if condition_status : 
                if v==skip_value:continue # skip 
            if status=='isin': 
                try: iter(value)
                except :e_min = abs(v - value)
                else : e_min = np.abs(v - np.array(value)).min() 
            # reverse option: loop all the database 
            elif status=='isoff':
                try: iter(v)
                except :e_min = abs(value - v)
                else :e_min = np.abs(value - np.array(v)).min() 
                    
            if e_min <= minbuff : 
                if status =='isoff':
                    ix_close_res = (i,  value) # index and value in database  
                else:ix_close_res = (i,  v)  # index and value in TRES 
                
                minbuff = e_min 
        
    return ix_close_res

def fit_rocks(logS_array, lns_, tres_):
    """ Find the pseudo rock name at each station from the pseudovalue intres. 
    
    :param logS_array: array_like of under the station resistivity value 
    :param lns_: array_like of the rocks or the pseudolayers (automatick)
    :param tres_: array_like of the TRES or the pseudo value of the TRES 
    
    :returns: list of the unik corresponding  resistivity value at each 
            station  and its fitting rock names.
            
    :Example: 
        
        >>> import watex.utils.geotools as GU
        >>> import watex.geology.core as GC 
        >>> obj= GC.quick_read_geomodel()
        >>> pslns , pstres,  ps_lnstres= GU.make_strata(obj)
        >>> logS1 =obj.nmSites[0] # station S0
        >>> fit_rock(logS1, lns_= pslns, tres_= pstres)
    """
    # get the log of each stations 
    # now find the corresponding layer name from close value in 
    #pseudotres 
    unik_l= np.unique(logS_array)
    unik_fitted_rocks=list()
    for k in range(len(unik_l)): 
        ix_best,_= get_closest_gap ( value = unik_l[k], 
                                       iter_obj =tres_ )
        unik_fitted_rocks.append(lns_[ix_best])
    # now build log blocks 
    fitted_rocks =list()
    for value in logS_array : 
        ix_value, =np.where(unik_l==value) # if index found
        fitted_rocks.append(unik_fitted_rocks[int(ix_value)])
        
    return fitted_rocks 

def assert_station(id, nm =None):
    """ Assert station according to the number of stations investigated.
    
    :param id: int or str, station number. The station counter start from 01 
        as litteral count except whn provided value in string format 
        following the letter `S`. For instance : `S00` =1
    :param nm: matrix of new stratiraphy model built. 
    :return: Index at specific station
    :Example:
        >>> import watex.utils.geotools as GU
        >>> import watex.geology.core as GC 
        >>> obj= GC.quick_read_geomodel()
        >>> GU.assert_station(id=47, nm=geoObj.nmSites)
        ...46
        
    """
    nstations = nm.shape[1]
    id_= station_id(id)
    
    if id_> nstations : 
        msg ='Site `S{0:02}` is out of the range. Max site= `S{1:02}`.'
        msg=msg.format(id_, nstations-1)
        msg+='The last station `S{0:02}` shoud be used'\
            ' instead.'.format(nstations-1)
        warnings.warn(msg, UserWarning)
        _logger.debug(msg)
        id_= nstations 
        
    return id_
        
def find_distinct_items_and_indexes(items, cumsum =False ):
    """ Find distincts times and their indexes.
    
    :param items: list of items to get the distincts values 
    :param cumsum: bool, cummulative sum when items is a numerical values
    :returns:  
        - distinct _indexes unique indexes of distinct items 
        - distinct_items: unik items in the list 
        - cumsum: cumulative sum of numerical items
        
    :Example: 
        >>> import watex.utils.geotools as GU
        >>> test_values = [2,2, 5, 8, 8, 8, 10, 12, 1, 1, 2, 3, 3,4, 4, 6]
        >>> ditems, dindexes, cumsum = GU.find_distinct_items_and_indexes(
            test_values, cumsum =True)
        >>> cumsum 
    """
    if isinstance(items, (tuple, np.ndarray, pd.Series)):
        items =list(items)
     
    if cumsum : 
        try : 
            np.array(items).astype(float)
        except: 
            warnings.warn('Cumulative sum is possible only with numerical '
                          f'values not {np.array(items).dtype} type.')
            cumsum_=None
        else: cumsum_= np.cumsum(items)
        
    else: cumsum_=None 

    s, init = items[0], 0
    distinct_items= list()
    ix_b=[]
    for i, value in enumerate(items):
        if value ==s : 
            if i ==len(items)-1: 
                ix_b.append(init)
                distinct_items.append(s)
            continue 
        elif value != s: 
            distinct_items.append(s)
            ix_b.append(init)
            s= value 
            init= i
           
    return  ix_b, distinct_items, cumsum_
        
def grouped_items( items, dindexes, force =True ):   
    """ Grouped items with the same value from their corresponding
    indexes.
    
    :param items: list of items for grouping.
    :param dindexes: list of distinct indexes 
    :param force: bool, force the last value to broken into two lists.
                Forcing value to be broke is usefull when the items are string.
                Otherwise, `force`  param should be ``False`` when dealing 
                numerical values.
    :return: distinct items grouped 
    
    :Example:  
        >>> import watex.utils.geotools as GU
        >>> test_values = [2,2, 5, 8, 8, 8, 10, 12, 1, 1, 2, 3, 3,4, 4, 6]
        >>> dindexes,* _ = GU.find_distinct_items_and_indexes(
            test_values, cumsum =False)
        >>> GU.grouped_items( test_values, dindexes)
        ...  [[2, 2], [5], [8, 8, 8], [10], [12], [1, 1],
        ...      [2], [3, 3], [4, 4], [6]]
        >>> GU.grouped_items( test_values, dindexes, force =False)
        ... [[2, 2], [5], [8, 8, 8], [10], [12], [1, 1],
            [2], [3, 3], [4, 4, 6]]
    """
    gitems =list() 
    
    def split_l(list0): 
       """ split list to two when distinct values is found
       for instance list [3, 3, 4, 4]--> [3, 3], [4,4]"""
       for i, v in enumerate(list0): 
           if i ==0: continue 
           if v != list0[i-1]: 
               return [list0[:i], list0[i:]] 

    for k , val in enumerate(dindexes): 
        if k== len(dindexes)-1:
            # check the last value and compare it to the new 
            gitems.append(items[val:])
            # get the last list and check values 
            l= gitems[-1]
            if force:
                if len(set(l)) !=1: 
                    try:
                        gitems = gitems[:-1] + split_l(l)
                    except: 
                        raise TypeError(
                            'can only concatenate list (not "NoneType") to list.'
                            ' Please check your `items` argument.')
            break 
        gitems.append(items[val:dindexes[k+1]])
    # if there a empty list a the end then remove it     
    if len(gitems[-1]) ==0: 
        gitems = gitems [1:]
        
    return gitems

def fit_stratum_property (fittedrocks , z , site_tres):
    """ Separated whole blocks into different stratum and fit their
    corresponding property like depth and value of resistivities
    
    :param fittedrocks: array_like of layers fitted from the TRES 
    :param z: array like of the depth 
    :param site_tres: array like of the station TRES 
    
    :returns: 
        - s_grouped: Each stratum grouped from s_tres 
        - site_tres_grouped: The site resistivity value `site_tres` grouped 
        - z_grouped: The depth grouped (from the top to bottom )
        - z_cumsum_grouped: The cumulative sum grouped from the top to bottom
        
    :Example: 
        
        >>> import watex.geology.core as GC
        >>> obj= GC.quick_read_geomodel()
        >>> logS1 = obj.nmSites[:, 0] 
        >>> sg, stg, zg, zcg= fit_stratum_property (
            obj.fitted_rocks, obj.z, obj.logS)
    """
    
    # loop the fitted rocks and find for each stratum its depth and it values
    # find indexes of distinct rocks 
    dindexes,* _ = find_distinct_items_and_indexes(fittedrocks)
    strata_grouped = grouped_items( fittedrocks , dindexes )
    # do it for tres 
    site_tres_grouped = grouped_items(site_tres, dindexes)
    # do it for depth 
    cumsumz = np.cumsum(z)
    z_grouped = grouped_items(z, dindexes, force =False)
    zcumsum_grouped = grouped_items( cumsumz, dindexes, force=False)
    
    return strata_grouped, site_tres_grouped, z_grouped , zcumsum_grouped

def get_s_thicknesses(grouped_z, grouped_s, display_s =True, station=None):
    """ Compute the thickness of each stratum from the grouped strata from 
    the top to the bottom.
    
    :param grouped_z: depth grouped according its TRES 
    :param grouped_s: strata grouped according to its TRES 
    :param s_display: bool, display infos in stdout 
    
    :returns: 
        - thick : The thickness of each layers 
        - strata: name of layers 
        - status: check whether the total thickness is equal to the 
            depth of investigation(doi). Iftrue: `coverall= 100%
            otherwise coverall is less which mean there is a missing layer 
            which was not probably taking account.
            
    :Example: 
        
        >>> import watex.geology.core as GC
        >>> obj= GC.quick_read_geomodel()
        >>> sg, _, zg, _= fit_stratum_property (obj.fitted_rocks,
        ...                                    obj.z, obj.nmSites[:, 0]  )
        >>> get_s_thicknesses( zg, sg)
        ... ([13.0, 16.0, 260.0, 240.0, 470.0],
        ...     ['*i', 'igneous rocks', 'granite', 'igneous rocks', 'granite'],
        ...     'coverall =100%')
    """
    # get the distincs layers 
    
    pstrata = [stratum[0] if stratum[0] != '$(i)$' else '*i' 
               for stratum in grouped_s ]
    # pstrata =[stratum for s in pstrata else '']
    b= grouped_z[0][0] #take the first values 
    thick =[]
    thick_range =[]
    for k , zg in enumerate(grouped_z): 
        th = round (abs(max(zg) - b), 5)  # for consistency 
        thick.append( th) 
        str_=f"{b:^7} ----- "
        b= round (max(zg), 3) 
        thick_range.append(str_ + f"{b:^7}")
    
    doi = grouped_z[-1][-1]
    if sum(thick) == doi: 
        status = "coverall =100%"
    else : 
        status = f"coverall= {round((sum(thick)/doi)*100, 2)}%"
    
    if display_s: 
        display_s_infos(pstrata , thick_range, thick, 
                        station =station )
    # keep back the stamp 
    pstrata =['$(i)$' if s== '*i' else s for s in  pstrata]
    return thick , pstrata,  status 

def display_s_infos( s_list, s_range, s_thick, **kws):
    """ Display strata infos at the requested station.
    
    :param s_list: the pseudostratigraphic details list 
    :param s_range: the pseudostratigraphic strata range 
    :param s_thick: the pseudostratigraphic  thicknesses
    :param kws:additional keywords arguments.
    """
    linestyle =kws.pop('linestyle', '-')
    mullines = kws.pop('linelength', 102)
    station = kws.pop('station', None)
    
    if station is not None: 
        if isinstance(station, (float,int)): 
            station = 'S{0:02}'.format(station)
        ts_= '{'+':~^'+f'{mullines}'+'}'
        print(ts_.format('[ PseudoStratigraphic Details: '
                         'Station = {0} ]'.format(station)))
        
    print(linestyle *mullines)
    headtemp = "|{0:>10} | {1:^30} | {2:^30} | {3:^20} |"
    if '*i' in  s_list :
        str_i ='(*i=unknow)'
    else: str_i =''
    headinfos= headtemp.format(
        'Rank', f'Stratum {str_i}', 'Thick-range(m)', 'Thickness(m)')
    print(headinfos)
    print(linestyle *mullines)
    for i, (sn, sthr, sth) in enumerate(zip(s_list, s_range, s_thick)):
        print(headtemp.format(f"{i+1}.", sn, sthr, sth ))
        
    print(linestyle *mullines)
    
def assert_len_lns_tres(lns, tres): 
    """ Assert the length of LN and Tres"""
    msg= "Input resistivity values <TRES> and input layers <LN> MUST" +\
          " have the same length. But {0} and {1} were given respectively. "
    is_the_same = len(tres) == len(lns)
    return is_the_same , msg.format(len(tres), len(lns))   

 
def _sanitize_db_items (value, force =True ): 
    """ Sanitize Database properties by removing the parenthesis and 
    convert numerical data to float. 
    
    :param value: float of list of values to sanitize.
    :param force: If `force` is ``True`` will return value without 
        parenthesis but not convert the inside values
        
    :return: A list of sanitized items 
    
    :Example:
        
        >>> import watex.utils.geotools as GU
        >>> test=['(1.0, 0.5019607843137255, 1.0)','(+o++.)',
        ...          '(0.25, .0, 0.98)', '(0.23, .0, 1.)']
        >>> GU._sanitize_db_items (test)
        ...[(1.0, 0.5019607843137255, 1.0),
        ...    '+o++.', (0.25, 0.0, 0.98), (0.23, 0.0, 1.0)]
        >>> GU._sanitize_db_items (test, force =False)
        ... [(1.0, 0.5019607843137255, 1.0), 
             '(+o++.)', (0.25, 0.0, 0.98), (0.23, 0.0, 1.0)]
    """

    if isinstance(value, str): 
        value=[value]
    def sf_(v):
        """Sanitise only a single value"""
        if '(' and ')' not in  v:
            try : float(v) 
            except : return v 
            else:return float(v)
        try : 
            v = tuple([float (ss) for ss in 
                 v.replace('(', '').replace(')', '').split(',')])
        except : 
            if force:
                if '(' and ')' in v:
                    v=v.replace('(', '').replace(')', '')
        return v

    return list(map(lambda x:sf_(x), value))


def base_log( ax, thick, layers, *, ylims=None, hatch=None, color=None )  : 
    """ Plot pseudo-stratigraphy basemap and return axis. 
    
    :param ax: obj, Matplotlib axis 
    :param thick: list of the thicknesses of the layers 
    :param layers: list of the name of layers
    :param hatch: list of the layer patterns
    :param color: list of the layer colors
    
    :return: ax- matplotlib axis properties 
    """
    if ylims is None: 
        ylims=[0, int(np.cumsum(thick).max())]
    ax.set_ylim(ylims)
    th_data = np.array([np.array([i]) for i in thick ]) 

    for ii, data in enumerate(th_data ): 
        next_bottom = sum(th_data [:ii]) +  ylims[0]
        ax.bar(1,
               data,
               bottom =next_bottom, 
               hatch = hatch[ii], 
               color = color[ii],
               width = .3)

    ax.set_ylabel('Depth(m)', fontsize = 16 , fontweight='bold')
    
    pg = [ylims[0]] + list (np.cumsum(thick) + ylims[0])

    ax.set_yticks(pg)
    ax.set_yticklabels([f'${int(i)}$' for i in  pg] )
    
    ax.tick_params(axis='y', 
                   labelsize= 12., 
                        )
    # inverse axes 
    plt.gca().invert_yaxis()
    return ax 

def annotate_log (ax, thick, layers,*, ylims=None, colors=None, 
                    set_nul='*unknow', bbox_kws=None, 
                    set_nul_bbox_kws=None, **an_kws): 
    """ Draw annotate stratigraphic map. 
    
    :param ax: obj, Matplotlib axis 
    :param thick: list of the thicknesses of the layers 
    :param layers: list of the name of layers
    :param set_nul: str 
        `set the Name of the unknow layers. Default is `*unknow`. Can be 
        changed with any other layer name. 
    :param bbox_kws:dict,  Additional keywords arguments of Fancy boxstyle 
        arguments
    :param set_nul_bbox_kws: dict, customize the boxstyle of the `set_nul` 
        param. If you want the bbox to be the same like `bbox_kws`, we need 
        just to write ``idem`` or `same`` or ``origin``.
        
    :return: ax: matplotlib axis properties 
    """
    
    xinf , xsup =-1, +1
    xpad =  .1* abs(xinf)/2
    ax.set_xlim([xinf, xsup])
    if ylims is None: 
        ax.set_ylim([0, int(np.cumsum(thick).max())]) #### 1 check this part 
        ylim0=0.
    else :
        ax.set_ylim(ylims)
        ylim0=ylims[0]
    # inverse axes 
    plt.gca().invert_yaxis()
    # if ylims is None:
    pg = np.cumsum([ylim0] + thick)# add 0. to thick to set the origin  #### 2
    # take values except the last y from 0 to 799 
    v_arrow_bases = [(xinf + xpad, y) for y in  pg ]
    v_xy = v_arrow_bases[:-1]
    v_xytext = v_arrow_bases[1:]
    # build the pseudo _thickness distance between axes 
    for k, (x, y) in enumerate(v_xy):
        ax.annotate('', xy=(x, y), xytext =v_xytext[k],
                    xycoords ='data',
                    # textcoords ='offset points',
                    arrowprops=dict(arrowstyle = "<|-|>", 
                                  ),  
                horizontalalignment='center',
                verticalalignment='top',                         
                )
    # ------------make horizontal arraow_[properties]
    # build the mid point where starting annotations 
    mid_loc = np.array(thick)/2 
    # if ylims is None: 
    center_positions =  pg[:-1] + mid_loc
    # else :center_positions =  pg + mid_loc
    h_xy = [ (xinf + xpad, cp) for cp in center_positions]
    h_xytext = [(0, mk ) for mk in center_positions ]
    
    # control the color 
    if colors is not None: 
        if isinstance(colors, (tuple, list, np.ndarray)):
            if len(colors) != len(thick): colors =None 
        else : colors =None 
    # build the pseudo _thickness distance between axes 
    if bbox_kws is None:  
         bbox0=  dict(boxstyle ="round", fc ="0.8", ec='k')
    if set_nul_bbox_kws in ['idem', 'same', 'origin']: 
        bbox_i = bbox0
        
    if not isinstance (set_nul_bbox_kws, dict): 
         set_nul_bbox =  None     
    if set_nul_bbox is None:
        bbox_i= dict (boxstyle='round', 
                     fc=(.9, 0, .8), ec=(1, 0.5, 1, 0.5))

    layers=[f"${set_nul}$" 
            if s.find("(i)")>=0 else s for s in layers ]
    
    for k, (x, y) in enumerate(h_xy):
        if layers[k] ==f"${set_nul}$" : 
            bbox = bbox_i
        else: bbox = bbox0
        if colors is not None:
            bbox ['fc']= colors[k]
        ax.annotate( f"{layers[k]}",  
                    xy= (x, y) ,
                    xytext = h_xytext[k],
                    xycoords='data', 
                    arrowprops= dict(arrowstyle='-|>', lw = 2.), 
                    va='center',
                    ha='center',
                    bbox = bbox, **an_kws
                 )

    return ax 


def plot_stratalog(
    thick, 
    layers, 
    station, *,
    zoom =None, 
    hatch=None, 
    color=None, 
    fig_size=(10, 4),
    **annot_kws
): 
    """ Make the stratalog log with annotate figure.
    
    Parameters 
    ------------
    thick, layer, hatch, colors: list, 
       list of the layers thicknesses , names, patterns and colors. 
       
    zoom: float, list 
       If float value is given, it considered as a 
       zoom ratio and it should be ranged between 0 and 1. 
       For isntance: 
           
       - 0.25 --> 25% plot start from 0. to max depth * 0.25 m.
            
       Otherwise if values given are in the list, they should be
       composed of two items which are the `top` and `bottom` of
       the plot.  For instance: 
           
       - [10, 120] --> top =10m and bottom = 120 m.
            
       Note that if the length of `zoom` list is greater than 2, 
       the function will return all the plot and 
       no errors should raised. 
             
      fig_size: tuple, default=(10, 4)
        Figure size 
        
    Examples 
    ---------
    >>> import watex.utils.geotools as GU   
    >>> layers= ['$(i)$', 'granite', '$(i)$', 'granite']
    >>> thicknesses= [59.0, 150.0, 590.0, 200.0]
    >>> hatch =['//.', '.--', '+++.', 'oo+.']
    >>> color =[(0.5019607843137255, 0.0, 1.0), 'b', (0.8, 0.6, 1.), 'lime']
    >>> GU.plot_stratalog (thicknesses, layers, hatch =hatch ,
                       color =color, station='S00')
    >>> GU.plot_stratalog ( thicknesses,layers,hatch =hatch, 
                            zoom =0.25, color =color, station='S00')
    """

    is_the_same, typea_status, typeb_status= _assert_list_len_and_item_type(
        thick, layers,typea =(int, float, np.ndarray),typeb =str)
    if not is_the_same: 
        # try to shrunk values 
        diff_thick = len(thick) - len(layers) 
        if abs (diff_thick)>2: 
            raise TypeError("Layer thicknesses and layer names must be consistent."
                            f". Got {len(thick)} and {len(layers)} respectively.")
        else: 
            # tolerate one layer 
            # by shruunking data 
            if diff_thick < 0: 
                layers = layers [: len(thick)]
            else: 
                thick = thick [: len(layers)]
                
    if not typea_status: # try to convert to float
        try : 
            thick =[float(f) for f in thick ]
        except :raise TypeError(
                "Layer thickness expect numeric value."
                f" Got {np.array(thick).dtype.name!r}")
        
    if not  typeb_status: 
        layers =[str(s) for s in layers] 
    
    # get the dfault layers properties hatch and colors 
    # change the `none` values if exists to the default values
    #for hatch and colors
    # print(color)
    hatch , color = set_default_hatch_color_values(hatch, color)
    #####INSERT ZOOM TIP HERE############
    ylims =None
    if zoom is not None: 
        ylims, thick, layers, hatch, color = zoom_processing(zoom=zoom, 
             data= thick, layers =layers, hatches =hatch,colors =color)
    #####################################
    fig = plt.figure( f"Log of Station ={station.upper()}",
                     figsize = fig_size, 
                      # dpi =300 
                     )
    plt.clf()
    gs = GridSpec.GridSpec(1,2,
                       wspace=0.05,
                       left=.08,
                       top=.85,
                       bottom=0.1,
                       right=.98,
                       hspace=.0,
                       ) 
    doi = sum(thick) 
    axis_base = fig.add_subplot(gs[0, 0],
                           ylim = [0, int(doi)] 
                           )
                
    axis_annot= fig.add_subplot(gs[0, 1],
                                sharey=axis_base) 
    axis_base = base_log(ax = axis_base, 
                         thick=thick, 
                         ylims=ylims, 
                         layers=layers,
                         hatch=hatch,
                         color=color)
    
    axis_annot = annotate_log(ax= axis_annot,
                              ylims=ylims, 
                             thick=thick,
                             layers=layers,
                             colors =color,
                             **annot_kws)
    
    for axis in (axis_base, axis_annot):
        for ax_ in ['top','bottom','left','right']:
            if ax_ =='left': 
                if ylims is None:
                    axis.spines[ax_].set_bounds(0, doi)
                else :  axis.spines[ax_].set_bounds(ylims[0], ylims[1])
                axis.spines[ax_].set_linewidth(3)
            else : axis.spines[ax_ ].set_visible(False)
  
        axis.set_xticks([]) 
        
    fig.suptitle( f"Strata log of Station ={station.upper()}",
                ha='center',
        fontsize= 7* 2., 
        verticalalignment='center', 
        style ='italic',
        bbox =dict(boxstyle='round',facecolor ='moccasin'), 
        y=0.90)
    
    plt.show()
    

def _assert_list_len_and_item_type(lista, listb, typea=None, typeb=None):
    """ Assert two lists and items type composed the lists 
    
    :param lista: List A to check the length and the items type
    :param listb: list B to check the length and the items type
    :param typea: The type which all items in `lista` might be
    :param typeb: The type which all items in `listb` might be
    
    :returns: 
        - the status of the length of the two lists ``True`` or ``False``
        - the status of the type of `lista` ``True`` if all items are the 
            same type otherwise ``False``
        - idem of `listb`
        
    :Example: 
        >>> import watex.utils.geotools as GU
        >>> thicknesses= [59.0, 150.0, 590.0, 200.0]
        >>> hatch =['//.', '.--', '+++.', 'oo+.']
        >>> GU._assert_list_len_and_item_type(thicknesses, hatch,
        ...                                   typea =(int, float, np.ndarray),
        ...                                    typeb =str))
        ... (True, True, True)
    """
    try: import __builtin__ as b
    except ImportError: import builtins as b
    
    def control_global_type(typ):
        """ Check the given type """
        builtin_types= [t for t in b.__dict__.values()
                     if isinstance(t, type)] 
        conv_type = builtin_types+ [np.ndarray, pd.Series,pd.DataFrame]
        if not isinstance( typ, (tuple, list)):
            typ =[typ]
        # Now loop the type and check whether one given type is true
        for ityp in typ:
            if ityp not in conv_type: 
                raise TypeError(f"The given type= {ityp} is unacceptable!"
                                " Need a least the following types "
                                f" {smart_format([str(i) for i in conv_type])}"
                                " for checking.")
        return True
    
    is_the_same_length  = len(lista) ==len (listb)
    lista_items_are_the_same, listb_items_are_the_same =False, False
    
    def check_items_type(list0, type0): 
        """ Verify whether all items  in the list are the same type"""
        all_items_type = False
        is_true = control_global_type(type0)
        if is_true:
             s0= [True if isinstance(i0, type0) else False for i0 in list0 ]
             if len(set(s0)) ==1: 
                 all_items_type = s0[0]
        return  all_items_type
        
    if typea is not None :
        lista_items_are_the_same = check_items_type (lista, typea)
    if typeb is not None :
        listb_items_are_the_same = check_items_type (listb, typeb)
        
    return (is_the_same_length , lista_items_are_the_same,
            listb_items_are_the_same )
    

def set_default_hatch_color_values(hatch, color, dhatch='.--', 
                                   dcolor=(0.5019607843137255, 0.0, 1.0),
                                   force =False): 
    """ Set the none hatch or color to their default values. 
    
    :param hatch: str or list of layer patterns 
    :param color: str or list of layers colors
    :param dhatch: default hatch 
    :param dcolor: default color 
    :param force: Return only single tuple values otherwise put the RGB tuple
        values  in the list. For instance::
            -if ``False`` color =[(0.5019607843137255, 0.0, 1.0)]
            - if ``True`` color = (0.5019607843137255, 0.0, 1.0)
    :Example: 
        >>> from watex.utils.geotools as  GU.
        >>> hatch =['//.', 'none', '+++.', None]
        >>> color =[(0.5019607843137255, 0.0, 1.0), None, (0.8, 0.6, 1.),'lime']
        >>> GU.set_default_hatch_color_values(hatch, color))
    """
    fs=0 # flag to reconvert the single RGB color in tuple 
    def set_up_(hc, dhc):
        if isinstance(hc, (list, tuple, set, np.ndarray)): 
            hc =[dhc if h in (None, 'none') else h for h in hc ]
        return hc 
    
    if isinstance(hatch, str): hatch=[hatch]
    if isinstance(color, str): color=[color]
    elif len(color)==3 : 
        try:iter(color[0]) # check iterable value in tuple
        except : 
            try : color =[float(c) for c in color]
            except : raise ValueError("wrong color values.")
            else: fs=1
    hatch = set_up_(hatch, dhatch)
    color = set_up_(color, dcolor) 
    if force:
        if len(color) ==1: color = color[0]
    if len(hatch) ==1 : hatch = hatch[0]
    if fs==1: color = tuple(color)
    return hatch, color 

def print_running_line_prop(obj, inversion_software='modelling softw.') :
    """ print the file  in stdout which is currently used
    " for pseudostratigraphic  plot when extracting station for the plot. """
    
    print('{:~^108}'.format(
        f' Survey Line: {inversion_software} files properties '))
    print('|' + ''.join([ "{0:<5} = {1:<17}|".format(
        i, os.path.basename( str(getattr(obj, f'{i}_fn', 'NA'))))  
     for i in ['model', 'iter', 'mesh', 'data']]) )
    print('~'*108)

def map_bottom (bottom, data, origin=None): 
    """Reduce the plot map from the top assumes to start at 0. to the
    bottom value.
    
    :param bottom: float, is the bottom value from
        which the plot must be end 
    :param data: the list of layers thicknesses in meters
    :param origin: The top value for plotting.by the default 
        the `origin` takes 0. as the starting point
        
    :return: the mapping depth from the top to the maximum depth.
            - the index indicated the endpoint of number of layer 
                for visualizing.
            - the list of pairs <top-bottom>, ex: [0, bottom]>
            - the value of thicknesses ranged from  0. to the bottom 
            - the coverall, which is the cumul sum of the value of
                the thicknesses reduced compared to the total depth.
     Note that to avoid raising errors, if coverall not equal to 100%,
     will return safety default values (original values).
     
    :Example: 
        >>> ex= [ 59.0, 150.0, 590.0, 200.0]
        >>> map_bottom(60, ex)
        ... ((2, [0.0, 60], [59.0, 1.0]), 'coverall = 100.0 %')
    """
    
    cumsum_origin = list(itertools.accumulate(data)) 
    if origin is None: origin = 0.
    end = max(cumsum_origin)
    wf =False # warning flag
    coverall, index =0., 0
    wmsg = ''.join([ "Bottom value ={0} m might be less than or ",
                          "equal to the maximum depth ={1} m."])
    t_to_b = list(itertools.takewhile(lambda x: x<= bottom,
                                      cumsum_origin))
    
    if bottom > end :bottom , wf = end, True 
    elif bottom ==0 or bottom < 0: 
        bottom , wf = data[0], True 
        to_bottom=([origin , bottom], [bottom])
    elif bottom < data[0] : 
        to_bottom = ([origin ,bottom], [bottom]) 
    elif len(t_to_b) !=0 :
        # add remain extent bottom values
        if max(t_to_b) < bottom : 
            add_bottom = [abs (bottom - max(t_to_b))] 
            to_bottom = ([origin, bottom], data[:len(t_to_b)] + add_bottom )
        elif max(t_to_b) ==bottom :
            to_bottom= ([origin, sum(t_to_b)],  t_to_b)
        index =len(to_bottom[1])   # get the endpoint of view layers 
    if bottom ==end : # force to take the bottom value
        to_bottom= ([origin, bottom], data)
        index = len(data)
        
    if wf:
        warnings.warn(wmsg.format(bottom, sum(data)), UserWarning)
        wf =False # shut down the flag
    coverall=  round(sum(to_bottom[1])/ bottom, 2)
    cov = f"coverall = {coverall *100} %"
    if coverall !=1.: 
        to_bottom = (len(data), [0., sum(data)], data)
    else : to_bottom = get_index_for_mapping(index, to_bottom )
    
    return to_bottom, cov 
    
def get_index_for_mapping (ix, tp): 
    """ get the index and set the stratpoint of the top or the endpoint 
    of bottom from tuple list. The index should be used to map the o
    ther properties like `color` or `hatches`"""
    tp=list(tp)
    # insert index from which to reduce other property
    tp.insert(0, ix)
    return  tuple(tp )
    
    
def map_top (top, data, end=None): 
    """ Reduce the plot map from the top value to the bottom assumed to 
    be the maximum of investigation depth. 
    
    :param top: float, is the top value from which the plot must be starts 
    :param data: the list of layers thicknesses in meters
    :param end: The maximum depth for plotting. Might be reduced, 
        but the default takes the maximum depth investigation depth 
    
    :return: 
         the mapping depth from the top to the maximum depth.
        - the index indicated the number of layer for visualizing to 
                from the top to the max depth.
        - the list of pairs <top-bottom>, ex: [top, max depth]>
        - the value of thicknesses ranged from 0. to the bottom 
        - the coverall, which is the cumul sum of the value of
            the thicknesses reduced compared to the total depth.
            It allows to track a bug issue.
            
        Note that if coverall is different 100%, will return the 
        default values giving values. 
        
    :Example:
        >>> import watex.utils.geotools as GU
        >>> ex= [ 59.0, 150.0, 590.0, 200.0] # layers thicknesses 
        >>> GU.map_top(60, ex)
        ... ((3, [60, 999.0], [149.0, 590.0, 200.0]), 'coverall = 100.0 %')
    """
    wmsg = ''.join([ "Top value ={0} m might be less than ",
                    "the bottom value ={1} m."])
    cumsum_origin = list(itertools.accumulate(data)) 
    if end is None: end = max(cumsum_origin)
    # filter list and keep value in cumsum 
    #greater  or equal to top values 
    data_= copy.deepcopy(data)
    if top < 0: top =0.
    elif top > end : 
        warnings.warn(wmsg.format(top, sum(data)), UserWarning)
        top = sum(data[:-1])
    t_to_b = list(filter(lambda x: x > top, cumsum_origin)) 
    index =0
    coverall =0.
    if len(t_to_b) !=0:
        if min (t_to_b)> top : # top = 60  --> [209.0, 799.0, 999.0]
            #get index of the min value from the cums_origin 
            # find 209 in [59.0, 209.0, 799.0, 999.0] --> index = 1
            index= cumsum_origin.index(min(t_to_b))
            #  find the  value at index =1 in data 
            #[ 59.0, 150.0, 590.0, 200.0]=> 150
             # reduce the downtop abs(59 - 60) = 1
            r_= abs(sum(data[:index]) - top )
            # reduced the data at index  1 with r_= 1
            reduce_top = abs(data [index] - r_)  # data[1]=150-1: 149 m 
            # replace the top value `150` in data with the reduced value
            data[index] = reduce_top  # 150==149
            from_top= ([top, end],data [index:] )# [ 149, 590.0, 200.0]
        elif min(t_to_b) == top: 
            index = cumsum_origin.index(min(t_to_b))
            from_top = ([top, end], data[index:])
            r_ = abs(sum(data[:index]) - top )
        
        coverall = round((sum(data[index :] +data[:index ])
                          + r_)/ sum(data_),2)
        
    cov = f"coverall = {coverall *100} %"
    if coverall !=1.:
        from_top = (index, [0., sum(data_)], data_)
    else:from_top= get_index_for_mapping(index, from_top )
        
    return from_top, cov 

def smart_zoom(v):
    """ select the ratio or value for zooming. Don't raise any error, just 
    return the original size. No shrunk need to be apply when error occurs.

    :param v: str, float or iterable for zoom
            - str: 0.25% ---> mean 25% view 
                    1/4 ---> means 25% view 
            - iterable: [0, 120]--> the top starts a 0.m  and bottom at 120.m 
            note: terable should contains only the top value and the bottom 
                value.
    :return: ratio float value of iteration list value including the 
        the value range (top and the bottom values).
    :Example: 
        >>> import watex.utils.geotools as GU
        >>> GU.smart_zoom ('1/4')
        ... 0.25
        >>> GU.smart_zoom ([60, 20])
        ... [20, 60]
    """
    def str_c (s):
        try:
            s=float(s)
        except:
            if '/' in s: # get the ratio to zoom 
                s= list(map(float, sorted(s.split('/'))))
                s=round(min(s)/max(s),2)
            elif '%' in s: # remove % and get the ratio for zoom
                s= float(s.replace('%', ''))*1e-2
                if s >1.:s=1.
            else: s=1.
            if s ==0.:s=1.
        return s
    
    msg="Ratio value `{}` might be greater than 0 and less than 1."
    emsg = f" Wong given value. Could not convert {v} to float "
    is_iterable =False 
    try:iter(v)
    except :pass 
    else : 
        if isinstance(v, str): v= str_c(v)
        else: is_iterable = True
        
    if not is_iterable: 
        try:  float(v)
        except ValueError:
            s=False 
            try:v = str_c(v)
            except ValueError :warnings.warn(emsg)
            else :s=True # conversion to zoom ratio was accepted
            if not s:
                warnings.warn(emsg)
                v=1.
        else:
            if v > 1. or v ==0.: 
                warnings.warn(msg.format(v)) 
                v=1.
                    
    elif is_iterable : 
        if len(v)>2:
            warnings.warn(f" Expect to get size =2 <top and bottom values>. "
                          f"Size of `{v}` should not  be greater than 2,"
                          f" but {len(v)} were given.", UserWarning)
            v=1.
        try : v= list(map(float, sorted(v)))
        except:  
            warnings.warn(emsg)
            v=1.
    return v

def frame_top_to_bottom (top, bottom, data ): 
    """ Compute the value range between the top and the bottom.
    
    Find the main range value for plot ranged between the top model and 
        the bottom. 
    :param top: is the top value where plot might be started 
    :param bottom: is the bottom value where plot must end 
    :param data: is the list of thicknesses of each layers 
    :param cumsum_origin: is the list of cumul sum of the `data` 
    
    :returns: 
        - the index for other properties mapping. It indicates the
            index of layer for the top and the index of layer for the 
            bottom for visualizing.
        - the list of pairs top-bottom , ex: [10, 999.0] 
                where tp -> 10 and bottom ->999. m
        - the value of thicknesses ranged from the top to the bottom 
            For instance:  [49.0, 150.0, 590.0, 200.0] where 
            - 49 is the thockness of the first layer 
            - 200 m is the thickness of the 
        - the coverall allows to track bug issues.The thickness of layer 
            for visualizing should be the same that shrank. Otherwise, 
            the mapping was not successfully done. Therefore coverall 
            will be different to 100% and function will return the raw data
            instead of raising errors. 
            
    :Example: 
        >>> import watex.utils.geotools as GU
        >>> layer_thicknesses = [ 59.0, 150.0, 590.0, 200.0]
        >>> top , bottom = 10, 120 # in meters 
        >>> GU.frame_top_to_bottom( top = top, bottom =bottom,
                                data = layer_thicknesses)
        ...(([0, 2], [10, 120], [49.0, 61.0]), 'coverall = 100.0 %')
 
    """
    if top > bottom :
        warnings.warn( f"Top value ={top} should be less than"
                      f" the bottom ={bottom} ")
        top=0.
    if top ==bottom :top , bottom = 0.,  sum(data) 
    if top <0 : top =0.
    if bottom > sum(data): 
        warnings.warn( f"Bottom value {bottom} should be less than"
                      f" or equal to {sum(data)}")
        bottom =sum(data)
        
    # test data = [ 59.0, 150.0, 590.0, 200.0]
    data_safe = copy.deepcopy(data)
    data_ = copy.deepcopy(data)
    # get the value from the to to the bottom 
    tm,*_ = map_top (top, data = data )
    ixt, _, tm = tm # [149.0, 150.0, 590.0, 200.0]
    bm, *_= map_bottom(bottom, data = data_ )
    ixb, _, bm = bm  # [59.0, 150.0, 391.0])
    #remove the startpoint and the endpoint from top and bottom 
    sp = [tm[0]]  # 149.
    del tm[0]       #--> [150.0, 590.0, 200.0]
    ep = [bm[-1]]         # 391.
    del bm[-1]      # --> [59.0, 150.0]
    # compute the intersection of the two lists 
    inter_set_map_tb = set(tm).intersection(set(bm))
    # set obj classification is sometimes messy, so let loop 
    # to keep the layer disposal the same like the safe data value
    inter_set_map_tb=[v for v in data_safe if v in inter_set_map_tb]
    top_bottom = sp + inter_set_map_tb + ep 
    # compute coverall to track bug issues 
    coverall = round(sum(top_bottom)/ abs(top-bottom ), 2)
    cov = f"coverall = {coverall *100} %"
    top_bottom = ([ixt, ixb], [top, bottom], top_bottom)
    if coverall !=1.:
        top_bottom = ([0, len(data_safe) ],
                      [0., sum(data_safe )], data_safe )
 
    return top_bottom, cov

def zoom_processing(zoom, data, layers =None, 
                    hatches=None, colors =None): 
    """ Zoom the necessary part of the plot. 
    
    If some optionals data are given such as `hatches`, `colors`, `layers`,
    they must be the same size like the data.
    
    :param zoom: float, list. If float value is given, it's cnsidered as a 
            zoom ratio than it should be ranged between 0 and 1. 
            For isntance: 
                - 0.25 --> 25% plot start from 0. to max depth * 0.25 m.
                
            Otherwise if values given are in the list, they should be
            composed of two items which are the `top` and `bottom` of
            the plot.  For instance: 
                - [10, 120] --> top =10m and bottom = 120 m.
                
            Note that if the length of list  is greater than 2, the function 
            will return the entire plot and  no errors should be raised.
    :param data: list of composed data. It should be the thickness from 
        the top to the bottom of the plot.
        
    :param layers: optional, list of layers that fits the `data`
    :param hatches: optional, list of hatches that correspond to the `data` 
    :param colors: optional, list of colors that fits the `data`
    
    :returns: 
        - top-botom pairs: list composed of top bottom values
        - new_thicknesses: new layers thicknesses computed from top to bottom
        - other optional arguments shrunk to match the number of layers and 
            the name of exact layers at the depth.
    
    :Example: 
        >>> import watex.utils.geotools as GU
        >>> layers= ['$(i)$', 'granite', '$(i)$', 'granite']
        >>> thicknesses= [59.0, 150.0, 590.0, 200.0]
        >>> hatch =['//.', 'none', '+++.', None]
        >>> color =[(0.5019607843137255, 0.0, 1.0), None, (0.8, 0.6, 1.),'lime'] 
        >>> GU.zoom_processing(zoom=0.5 , data= thicknesses, layers =layers, 
                              hatches =hatch, colors =color) 
        ... ([0.0, 499.5],
        ...     [59.0, 150.0, 290.5],
        ...     ['$(i)$', 'granite', '$(i)$'],
        ...     ['//.', 'none', '+++.'], 
        ...     [(0.5019607843137255, 0.0, 1.0), None, (0.8, 0.6, 1.0)])
        
    """
    def control_iterable_obj( l, size= len(data)): 
        """ Control obj size compared with the data size."""
        if len(l) != size : 
            warnings.mess(f"The iterable object {l}  and data "
                        f" must be same size ={size}. But {len(l)}"
                        f" {'were' if len(l)>1 else 'was'} given.")
            l=None 
        return l 
   
    #**********************************************************************
    # In the case, only the depth is given.
    # [120] and float `value` is given. 
    # If float value is greater than one, value should be consider as a 
    # max_depth for visualization 
    l1ratio , l1trange= round(data[0] /sum(data), 2), [0. , data[0]]
    
    raise_msg , set_l1 =False , False 
    if isinstance(zoom, list): # e.g., [120] 
        if len(zoom)==1: 
            try : float(zoom [0])
            except: 
                if ('%' or '/') in zoom [0]: zoom =zoom [0]
                else:raise_msg =True # e.g. [120%]
                
            else:  zoom = zoom[0]
 
    try: zoom = float(zoom)
    except: pass

    if isinstance(zoom, float):#e.g. 0.12 < 0.25 of ratio fist layer .
        if zoom < l1ratio: # map only the first layer from ratio 
            zoom =l1ratio
        elif 1. < zoom < data[0]: # 1 < 120 < 129.
            set_l1 =True 
        else: zoom =[0. , zoom]
    
    if set_l1: # map the first layer from the list 
        zoom = l1trange 
         
    if raise_msg :    
        raise ValueError(f'The given `zoom` value =`{zoom}` is wrong.'
            ' `zoom` param expects a ratio (e.g. `25%` or `1/4`) or a'
            ' list of [top, bottom] values, not {zoom!r}.'
            )
    #**********************************************************************
    zoom = smart_zoom(zoom) # return ratio or iterable obj 
    y_low, y_up = 0., sum(data)
    ix_inf=0
    try : 
        iter(zoom)
    except :
        if zoom ==1.:# straightforwardly return the raw values (zoom =100%)
            return [y_low, y_up], data, layers,  hatches, colors
 
    if isinstance(zoom, (int, float)): #ratio value ex:zoom= 0.25
        # by default get the bootom value and start the top at 0.
        y_up = zoom * sum(data)  # to get the bottom value  as the max depth 
        bm, *_=map_bottom(y_up, data = data )
        ix_sup, _, maptopbottom = bm  # [59.0, 150.0, 391.0])
        y = [y_low, y_up ]

    if isinstance(zoom , (list, tuple, np.ndarray)):
        top, bottom = zoom 
        tb, *_= frame_top_to_bottom (top, bottom, data )
        ixtb, y, maptopbottom= tb 
        ix_inf, ix_sup = ixtb  # get indexes range
        
    if hatches is not None:
        hatches = control_iterable_obj(hatches)
        hatches =hatches [ix_inf:ix_sup]
    if colors is not None:
        colors= control_iterable_obj(colors)
        colors = colors[ix_inf:ix_sup]
    if layers is not None:
        layers= control_iterable_obj(layers)
        layers = layers [ix_inf:ix_sup]
        
    return y, maptopbottom, layers, hatches , colors 

def _assert_model_type(kind):
    """ Assert stratigraphic model argument parameter.
    :param param: str, can be : 
        -'nm', 'strata', 'geomodel', 'logS', '2' for 
            the stratigraphic model
        - 'crm', 'resmodel', 'occam', 'rawmodel', '1'
    """
    kind =str(kind)
    for v in [ 'nm', 'strata', 'geomodel', 'logS', 'rr','2']: 
        if kind.lower().find(v)>=0: 
            kind = 'nm'
    if kind  in ['crm', 'resmodel', 'occam', 'rawmodel', '1']: 
        kind= 'crm'
    if kind not in ('nm', 'crm'): 
        raise StrataError(
            f"Argument kind={kind!r} is wrong! Should be `nm`"
            "for stratigraphyic model and `crm` for occam2d model. ")
    return kind 
    
def display_ginfos(
    infos,
    inline='-', 
    size =70,  
    header ='Automatic rocks',  
    **kws):
    """ Display unique element on list of array infos.
    
    :param infos: Iterable object to display. 
    :param header: Change the `header` to other names. 
    :Example: 
        >>> from watex.geology.stratigraphic import display_infos
        >>> ipts= ['river water', 'fracture zone', 'granite', 'gravel',
             'sedimentary rocks', 'massive sulphide', 'igneous rocks', 
             'gravel', 'sedimentary rocks']
        >>> display_infos('infos= ipts,header='TestAutoRocks', 
                          size =77, inline='~')
    """

    if isinstance(infos, str ): 
        infos =[infos]
        
    infos = list(set(infos))
    print(inline * size )
    mes= '{0}({1:02})'.format(header.capitalize(),
                                  len(infos))
    mes = '{0:^70}'.format(mes)
    print(mes)
    print(inline * size )
    am=''
    for ii in range(len(infos)): 
        if (ii+1) %2 ==0: 
            am = am + '{0:>4}.{1:<30}'.format(ii+1, infos[ii].capitalize())
            print(am)
            am=''
        else: 
            am ='{0:>4}.{1:<30}'.format(ii+1, infos[ii].capitalize())
            if ii ==len(infos)-1: 
                print(am)
    print(inline * size )
     
