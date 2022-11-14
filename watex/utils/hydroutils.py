# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Hydrogeological utilities 
================================
Hydrogeological parameters of aquifer are the essential and crucial basic data 
in the designing and construction progress of geotechnical engineering and 
groundwater dewatering, which are directly related to the reliability of these 
parameters.

"""
from __future__ import annotations 
import random
import copy 
import math
import itertools
from collections import Counter 
import inspect
import warnings 
import numpy as np
import pandas as pd 
from .._docstring import ( 
    _core_docs, 
    DocstringComponents 
    )
from  .._typing import (
    List, 
    Tuple, 
    Optional, 
    Union, T,
    Series, 
    DataFrame, 
    ArrayLike, 
    F
    ) 
from ..decorators import ( 
    catmapflow2, 
    writef  
    )
from ..exceptions import ( 
    FileHandlingError, 
    DepthError, 
    DatasetError, 
    StrataError
    )
from .funcutils import  (
    _assert_all_types, 
    is_iterable,
    is_in_if , 
    smart_format, 
    savepath_ , 
    is_depth_in, 
    reshape , 
    listing_items_format, 
    to_numeric_dtypes, 
    _isin 
    
    )
from ..exlib.sklearn import SimpleImputer 

from .validator import _is_arraylike_1d
#-----------------------

_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"], 
    )


def select_base_stratum (
    d: Series | ArrayLike | DataFrame , 
    /, 
    sname:str = None, 
    stratum:str= None,
    return_rate:bool=False, 
    return_counts:bool= False, 
    ):
    """ Select base stratum in the strata data contain. 
    
    Find the most recurrent stratum in the data and compute the rate of 
    occurrence. 
    
    Parameters 
    ------------
    d: array-like 1D , pandas.Series or DataFrame
        Valid data containing the strata. If dataframe is passed, 'sname' is 
        needed to fetch strata values. 
    sname: str, optional 
        Name of column in the dataframe that contains the strata values. 
        Dont confuse 'sname' with 'stratum' which is the name of the valid 
        layer/rock in the array/Series of strata. 
    stratum: str, optional 
        Name of the base stratum. Must be self contain as an item of the 
        strata data. Note that if `stratum` is passed, the auto-detection of 
        base stratum is not triggered. It returns the same stratum , however
        it can gives the rate and occurence of this stratum if `return_rate` 
        or `return_counts` is set to ``True``. 
    return_rate: bool,default=False, 
        Returns the rate of occurence of the base stratum in the data. 
    return_counts: bool, default=False, 
        Returns each stratum name and the occurences (count) in the data. 
    
    Returns 
    ---------
    bs: str 
        - base stratum , self contain in the data 
    r: float 
        rate of occurence in base stratum in the data 
    c: tuple (str, int)
        Tuple of each stratum whith their occurrence in the data. 
        
    Example 
    --------
    >>> from watex.datasets import load_hlogs 
    >>> from watex.utils.hydroutils import select_base_stratum 
    >>> data = load_hlogs().frame # get only the frame 
    >>> select_base_stratum(data, sname ='strata_name')
    ... 'siltstone'
    >>> select_base_stratum(data, sname ='strata_name', return_rate =True)
    ... 0.287292817679558
    >>> select_base_stratum(data, sname ='strata_name', return_counts=True)
    ... [('siltstone', 52),
         ('fine-grained sandstone', 40),
         ('mudstone', 37),
         ('coal', 24),
         ('Coarse-grained sandstone', 15),
         ('carbonaceous mudstone', 9),
         ('medium-grained sandstone', 2),
         ('topsoil', 1),
         ('gravel layer', 1)]
    """
    _assert_all_types(d, pd.DataFrame, pd.Series, np.ndarray )
    
    if hasattr(d, 'columns'): 
        if sname is None :
            raise TypeError ("'sname' ( strata column name )  can not be "
                              "None when a dataframe is passed.")
        sn= copy.deepcopy(sname)
        sname = _assert_all_types(sname, str, objname ='Column') 
        sname = is_in_if(d.columns, sname, error ='ignore')
        if sname is None: 
            raise ValueError ( f"Name {sn!r} is not a valid column strata name."
                              " Please, check your data.") 
        sname =sname [0] if isinstance(sname, list) else sname 
        sdata = d[sname ]    

    elif hasattr (d, '__array__') and not hasattr (d, 'name'):
        if not _is_arraylike_1d(d): 
            raise StrataError("Strata data supports only one-dimensional array."
                             )
        sdata = d
        
    if stratum is not None: 
        if not stratum in set (sdata):
            out= listing_items_format(set(sdata), begintext = 'strata', 
                                      verbose = False )
            raise StrataError (f"Stratum {stratum!r} not found in the data."
                              f" Expects {out}")
    #compute the occurence of the stratum in the data: 
    bs,  r , c  = _get_s_occurence(sdata , stratum )
        
    return ( ( r , c )  if ( return_rate and return_counts) else  ( 
            r if return_rate else c ) if return_rate or return_counts else bs 
            ) 

def _get_s_occurence (
        sd, /,  bs = None ) -> Tuple [str, float, List ]: 
    """ Returns the occurence of the object in the data. 
    :param sd: array-like 1d of  data 
    :param bs: str - base name of the object. If 'bs' if given the auto 
        search  will not be used. 
    :param return_counts: return each object with their occurence 
    :returns: bs, c, r
        return the base object, counts or rate.
    """
    # sorted strata in ascending occurence 
    s=dict ( Counter(sd ) ) 
    sm = dict (
        sorted (s.items () , key= lambda x:x[1], reverse =True )
        )
    bs = list(sm) [0]  if bs is None else bs 
    r= sm[bs] / sum (sm.values ()) # ratio
    c = list(zip (sm.keys(), sm.values ())) 
    
    return  bs,  r , c

            
def get_compressed_vector(
    d, /, 
    sname,  
    stratum =None , 
    strategy ="average", 
    as_frame = False, 
    random_state = None, 
    )-> Series :
    """ Compress base stratum data into a singular vector composes of all 
    feature names in the targetted data `d`. 
    
    Parameters 
    ------------
    d: pandas DataFrame
        Valid data containing the strata. If dataframe is passed, 'sname' is 
        needed to fetch strata values. 
    sname: str, optional 
        Name of column in the dataframe that contains the strata values. 
        Dont confuse 'sname' with 'stratum' which is the name of the valid 
        layer/rock in the array/Series of strata. 
    stratum: str, optional 
        Name of the base stratum. Must be self contain as an item of the 
        strata data. Note that if `stratum` is passed, the auto-detection of 
        base stratum is not triggered. It returns the same stratum , however
        it can gives the rate and occurence of this stratum if `return_rate` 
        or `return_counts` is set to ``True``. 
    
    strategy: str , default='average' or 'mean', 
        strategy used to select or compute the numerical data into a 
        singular series. It can be ['naive']. In that case , a single serie 
        if randomly picked up into the base strata data.
    as_frame: bool, default='False'
        Returns compressed vector into a dataframe rather that keeping in 
        series. 
    random_state: int, optional, 
        State for randomly selected a compressed vector when ``naive`` is 
        passed as strategy.
    
    Returns 
    --------
    ms: pandas series/dataframe 
        returns a compressed vector in pandas series compose of all features. 
        Note , the vector here does not refer as math vector compose of 
        numerical values only. A compressed vector here is a series that is 
        the result of averaging the numerical features of the base stratum and 
        incluing its corresponding categorical values. Note there, the  `ms`
        can contain categorical values and has the same number and features as 
        the original frame `d`. 
    
    Example
    -------
    >>> from watex.datasets import load_hlogs 
    >>> from watex.utils.hydroutils import get_compressed_vector 
    >>> data = load_hlogs().frame # get only the frame  
    >>> get_compressed_vector (data, sname='strata_name')[:4]
    ... hole_number           H502
        strata_name      siltstone
        aquifer_group           II
        pumping_level       ZFSAII
        dtype: object
    >>> get_compressed_vector (data, sname='strata_name', as_frame=True )
    ...   hole_number strata_name aquifer_group  ...        r     rp remark
        0        H502   siltstone            II  ...  41.7075  59.23    NaN
        [1 rows x 23 columns]
    >>> get_compressed_vector (data, sname='strata_name', strategy='naive')
    ... hole_number          H502
        depth_top          379.15
        depth_bottom        379.7
        strata_name     siltstone
        Name: 39, dtype: object
    """
    _assert_all_types(d, pd.DataFrame, objname = "Data for samples compressing")
    sname = _assert_all_types(sname, str , "'sname' ( strata column name )")
    
    assert strategy in {'mean', 'average', 'naive'}, "Supports only strategy "\
        f"'mean', 'average' or 'naive'; got {strategy!r}"
    if stratum is None: 
        stratum = select_base_stratum(d, sname= sname, stratum= stratum )
    stratum = _assert_all_types(stratum, str , objname = 'Base stratum ')
    #group y and get only the base stratum data 
    pieces = dict(list(d.groupby (sname))) 
    bs_d  = pd.DataFrame( pieces [ stratum ]) 
    # get the numerical features only before  applying operation 
    _, numf , catf  = to_numeric_dtypes(bs_d , return_feature_types= True )
    
    if strategy  in ('mean', 'average') :
        ms = bs_d[ numf ].mean() 
        if len(catf)!=0:
            # Impute data and fill the gap if exists
            #  by the most frequent categorial features.
            sim = SimpleImputer(strategy = 'most_frequent') 
            xt = sim.fit_transform(bs_d[catf]) 
            bs_dc = pd.DataFrame(xt , columns = sim.feature_names_in_ ) 
            # get only single value of the first row 
            bs_init = bs_dc .iloc [0 , : ] 
            #ms.reset_index (inplace =True ) 
            ms = pd.concat ( [ bs_init, ms  ], axis = 0 ) 
    elif strategy =='naive':
        random_state= random_state or 42 
        # randomly pick up one index 
        rand = np.random.RandomState (random_state )
        # if use sample , -> return a list and must 
        # specify the k number of sequence , 
        # while here , only a single is is expected: like 
        # random.sample (list(rand.permutation (X0.index )) , 1 )
        ix = random.choice (rand.permutation (bs_d.index )) 
        ms = bs_d.loc [ix ] 
        
    return  ms  if not as_frame  else pd.DataFrame(
        dict(ms) , index = range (1))

def _assert_reduce_indexes (*ixs ) : 
    """ Assert reducing indexing and return a list of valids indexes `ixs`"""
    ixs = list(ixs )
    for ii, ix in enumerate (ixs): 
        if not is_iterable( ix) : 
            raise IndexError ("Expects a pair tuple or list i.e.[start, stop]'"
                              f" for reducing indexing; got {ix}") 
        if len(ix) !=2 : 
            raise IndexError(f"Index must be a pair [start, top]: got {ix}")
        try:
            ix = [int (i) for i in ix ]
        except : 
            raise IndexError("Index should be a pair tuple/list of integers;"
                             f" check {ix}")
        else: ixs[ii] = ix 
        
    return ixs 

def get_sections_from_depth  (z, z_range, return_indexes =False ) :
    """ Get aquifer section indexes in data 'z' from the depth range.
    
    This might be usefull to compute the thickness of the aquifer. 
    
    Parameters 
    ----------
    z: array-like 1d or pd.Series 
        Array or pandas series contaning the depth values 
    z_range: tuple (float), 
        Section ['upper', 'lower'] of the aquifer at differnt depth.
        The range of the depth must a pair values and  could not be
         greater than the maximum depth of the well. 
    return_indexes: bool, default=False 
        returns the indexes of the sections ['upper', 'lower'] 
        of the aquifer and non-valid sections data. 
        
    Returns 
    ----------
    sections: Tuple (float, float)
       Real values of the  upper and lower sections of the aquifer. 
    If ``return_indexes`` is 'True', function returns: 
      (upix, lowix): Tuple (int, int )
          indices of upper and lower sections in the depth array `z`
      (invix): list of Tuple (int, int) 
          list of indices of invalid sections
    Example
    --------
    >>> from watex.datasets import load_hlogs 
    >>> from watex.utils.hydroutils import get_sections_from_depth
    >>> data= load_hlogs().frame  
    >>> # get real sections from depth 16.25 to 125.83 m
    >>> get_sections_from_depth ( data.depth_top, ( 16.25, 125.83))
    ...  (22.46, 128.23)
    >>> # aquifer depth from 16.25 m to the end 
    >>> get_sections_from_depth ( data.depth_top, ( 16.25,))
    ... (22.46, 693.37)
    >>> get_sections_from_depth ( data.depth_top, ( 16.25, 125.83),
                                 return_indexes =True )
    ... ((3, 11), [(0, 3), (11, 180)])
    >>> get_sections_from_depth ( data.depth_top, ( 16.25,), 
                                 return_indexes =True )
    ... ((3, 181), [(0, 3)])
 
    """
    z = _assert_all_types(z, pd.Series, np.ndarray , "Depth")
    
    if not _is_arraylike_1d (z) : 
        raise DepthError( "Depth expects one-dimensional array.")
    
    if not is_iterable(z_range): 
        return TypeError ("Depth range must be an iterable object,"
                          f" not {type (z_range).__name__!r}")
    z_range= sorted ( list(z_range ) ) 
    if max(z_range ) > max(z): 
        raise DepthError("Depth value can not be greater than the maximum "
                         f"depth in the well= {max(z)}; got {max(z_range)}")
    if len(z_range)==1: 
        warnings.warn("Single value is passed. Remember, it may correspond "
                      "to the depth value of the upper section thin the end.")
        z_range = z_range + [max (z )]
    elif len(z_range) > 2: 
        raise DepthError( "Too many values for the depth section range."
                         "Expects a pair values [ upper, lower] sections."
                         )
    # get the indices from depth 
    upix  = np.argmin  ( np.abs ( 
        (np.array(z) - z_range [0] ) ) ) 
    lowix = np.argmin  ( np.abs (
        (np.array(z) - z_range [-1] ) ) ) 
    # for consistency , reset_zrange with 
    # true values from depth z 
    sections = ( z [upix ], z[lowix ] )  
    z_range =  np.array ( ( upix , lowix ) , dtype = np.int32 ) 

    # compute the difference between adjacent depths
    diff = np.diff (z) 
    # when depth 
    if set (sections )==1: 
        raise DepthError("Upper and lower sections must have different depths.")
    
    if ( float( np.diff (sections)) <=diff.min() ): 
        # thickness to pass to another layers 
        raise DepthError(f"Depth {z_range} are too close that probably "
                         "figure out the same layer. Difference between "
                         "adjacent depth must be greater than"
                        f" {round ( float(diff.min()), 2) }")
    # not get the index from non valid data
    # +1 for Python indexing
    invix = _get_invalid_indexes (z, z_range )
    
    return  sections if not  return_indexes else ( 
        ( upix , lowix + 1 ),  invix ) 


def get_unique_sections (
        *data, zname, kname,  return_index=False, return_data =False, 
        error='raise', **kws ) : 

    sect, dat = get_aquifers_sections(*data, zname=zname, kname=kname, 
                                 return_indexes =return_index, 
                                 return_data= True,
                                 error = error , **kws)
    sect = np.array (list(itertools.chain(*sect)))
    si = np.array ([sect.min(), sect.max()], 
                   dtype = np.int32 if return_index else np.float32 )
    return si if not return_data else  ( si, dat ) 

get_unique_sections.__doc__="""\
Get the section to consider unique in multiple aquifers. 

The unique section 'upper' and 'lower' is the valid range of the whole 
data to consider as a  valid data. 
The use of the index is  necessary to shrunk the data of the whole 
boreholes. Mosly the data from the section is consided the valid data as the 
predictor Xr. Out of the range of aquifers ection, data can be discarded or 
compressed to top Xr. 

Returns valid section indexes if 'return_index' is set to ``True``.    
    
d: list of pandas dataframe 
    Data that contains mainly the aquifer values. It needs to specify the 
    name of the depth column `zname` as well as the name of permeabiliy 
    `kname` column.  
{params.core.zname}
{params.core.kname}
{params.core.z}

return_index: bool, default =False , 
    Returns the positions (indexes) of the upper and lower sections of the
    shallower  and deep aquifers found in the whole  dataframes.
return_data: bool, default=False, 
    Return valid data. It is usefull when 'error' is set to 'ignore'
    to collect the valid data. 
error: str, default='raise' 
    Raise errors if trouble occurs when computing the section of each aquifer. 
    If 'ignore', a UserWarning is displayed when invalid data is found. Any 
    other value of `error` will set error to `raise`. 
kws: dict, 
    Additional keywords arguments passed  to  
    :func:`~watex.utils.hydroutils.get_aquifer_sections`.
    
Returns 
--------
up, low :list of upper and lower section values of aquifer.
    - (upix, lowix ): Tuple of indexes of lower and upper sections  
    - (up, low): Tuple of aquifer sections (upper and lower)  
    - (upix, lowix), (up, low) : positions and sections values of aquifers 
        if `return_indexes` and return_sections` are ``True``.  

See Also 
----------
- compute multiple sections: :func:`~watex.utils.hydroutils.get_aquifers_sections`. 
- compute single secion:  :func:`~watex.utils.hydroutils.get_aquifer_sections`. 

Example
-------   
>>> from watex.datasets import load_hlogs 
>>> data = load_hlogs ().frame 
>>> get_unique_sections (data.copy() , zname ='depth', kname ='k', ) 
... array([197.12, 369.71], dtype=float32)
>>> get_unique_sections (data.copy() , zname ='depth', kname ='k', 
                                return_index =True)
... array([16, 29])

""".format(
    params=_param_docs,
    )
    
def get_aquifers_sections (
    *d ,  
    zname, 
    kname, 
    return_indexes =False, 
    return_data=False,
    error = 'ignore',  
    **kws 
    ): 

    errors = []
    is_valid_dfs = [] ; is_not_valid =[]
    section_indexes ,sections =[] , []
    
    error ='raise' if error !='ignore' else 'ignore'

    for ii, df in enumerate ( d) : 
        try : 
            ix, sec = get_aquifer_sections(
                df , 
                zname = zname , 
                kname = kname , 
                return_indexes= True, 
                return_sections=True, 
                **kws
                )
            is_valid_dfs .append (df )
        except Exception as err :
            # if error =='raise':
            #     raise err
            errors.append(str(err))
            is_not_valid.append (ii + 1 )
            continue 
        section_indexes.append(ix); sections.append(sec )
        
    if len(is_not_valid)!=0 : 
        msg = "Unsupports data at position{0} {1}.".format(
            f"{'s' if len(is_not_valid)>1 else''}", smart_format(is_not_valid))
                     
        if error =='raise':
            btext = "\nReasons"
            entext = "Sections can not be computed. Please check your data."
            mess = msg +  listing_items_format(
                errors, begintext=btext, endtext=entext , verbose =False )
            raise DatasetError(mess) 
            
        warnings.warn(msg + " Data {} discarded.".format( 
            "is" if len(is_not_valid)<2 else "are")
                      )        
    r= section_indexes if return_indexes else sections 
    
    return  r  if not return_data else ( r , is_valid_dfs) 

get_aquifers_sections.__doc__="""\
Get the section of each aquifer form multiple dataframes. 
 
The unique section 'upper' and 'lower' is the valid range of the whole 
data to consider as a  valid data. 
The use of the index is  necessary to shrunk the data of the whole 
boreholes. Mosly the data from the section is consided the valid data as the 
predictor Xr. Out of the range of aquifers ection, data can be discarded or 
compressed to top Xr. 

Returns valid section indexes if 'return_index' is set to ``True``.    
    
d: list of pandas dataframe 
    Data that contains mainly the aquifer values. It needs to specify the 
    name of the depth column `zname` as well as the name of permeabiliy 
    `kname` column.  
{params.core.zname}
{params.core.kname}
{params.core.z}

return_indexes: bool, default =False , 
    Returns the positions (indexes) of the upper and lower sections of the
   each aquifer found in each dataframe.

error: str, default='ignore' 
    Raise errors if trouble occurs when computing the section of each aquifer. 
    If 'ignore', a UserWarning is displayed if invalid data is found. Any 
    other value of `error` will set error to `raise`. 
return_data: bool, default=False, 
    Return valid data. It is usefull when 'error' is set to 'ignore'
    to collect the valid data. 
       
kws: dict, 
    Additional keywords arguments passed  to  
    :func:`~watex.utils.hydroutils.get_aquifer_sections`.
    
Returns 
--------
up, low :list of upper and lower section values of aquifer.
    - (upix, lowix ): Tuple of indexes of lower and upper sections  
    - (up, low): Tuple of aquifer sections (upper and lower)  
    - (upix, lowix), (up, low) : positions and sections values of aquifers 
        if `return_indexes` and return_sections` are ``True``.  

See Also 
----------
- compute single secion:  :func:`~watex.utils.hydroutils.get_aquifer_sections`. 

Example
-------   
>>> from watex.datasets import load_hlogs 
>>> data = load_hlogs ().frame 
>>> get_aquifers_sections (data, data , zname ='depth', kname ='k' ) 
... [[197.12, 369.71], [197.12, 369.71]]
>>> get_aquifers_sections (data, data , zname ='depth', kname ='k' , 
                           return_indexes =True ) 
...  [[16, 29], [16, 29]]

""".format(
    params=_param_docs,
    )
def _get_invalid_indexes  ( d, /, valid_indexes, in_arange =False ): 
    """ Get non valid indexes from valid section indexes 
    
    :param d: array_like 1d 
        array-like data for recover the section range indexes 
    :param section_ix: Tuple (int, int) 
        Index of upper and lower sections
    :param in_arange: bool, 
        List all index values. 
    :returns: 
        invix: List(Tuple(int))
        Returns invalid indexes onto a list 
    Example 
    -----------
    >>> from watex.utils.hydroutils import _get_invalid_indexes
    >>> import numpy as np 
    >>> idx = np.arange (50) 
    >>> _get_invalid_indexes (idx , (3, 11 ))
    ... [(0, 3), (12, 50)]
    
    """
    
    if in_arange : 
        valid_indexes = np.array (  list( 
            range ( * [  valid_indexes [0] , valid_indexes [-1] +1 ] )))  
        mask = _isin(range(len(d)), valid_indexes, return_mask=True )
        invix = np.arange (len(d))[~mask ]
    else :
        # +1 for Python indexing
        invix =  (np.arange (len(d))[:valid_indexes [0] + 1 ],
                  np.arange (len(d) + 1 )[valid_indexes[1]+1 : ]) 
        invix=  [ ( min(ix) , max(ix))  for ix in invix  if  ( 
            len(ix )!=0 and len(set(ix))>1)  ] # (181, 181 )
    
    return invix 

    
def get_xs_xr_splits (
    df, 
    /,
    z_range = None, 
    zname = None, 
    section_indexes:Tuple[int, int]=None, 
    )-> Tuple [DataFrame ]:
    """Split data into matrix :math:`X_s` with sample :math:`ms` (unwanted data ) 
    and :math:`X_r` of samples :math:`m_r`( valid aquifer data )
    
    Parameters 
    -----------
    df: pandas dataframe 
        Dataframe for compressing. 
    zname: str,int , 
        the name of depth column. 'name' needs to be supplied 
        when `section_indexes` is not provided. 
    z_range: tuple (float), 
        Section ['upper', 'lower'] of the aquifer at different depth.
        The range of the depth must a pair values and  could not be
        greater than the maximum depth of the well.
    section_indexes: tuple or list of int 
        list of a pair tuple or list of integers. It is be the the valid 
        sections( upper and lower ) indexes of  of the aquifer. If 
        the depth range `z_range` and `zname` are supplied, `section_indexes`
        can be None.  Note that the last indix is considered as the last 
        position, the bottom of the section therefore, its value is 
        included in the data.
        
    Returns
    --------
    - xs : list of pandas dataframe 
        - shrinking part of data for compressing. Note that it is on list 
        because if dataframe corresponds to the non-valid dataframe sections. 
    - xr: pandas dataframe  
        - valid data reflecting to the aquifer part or including the 
        aquifer data. 
        
    Example
    --------
    >>> from watex.datasets import load_hlogs 
    >>> from watex.utils.hydroutils import get_xs_xr_splits 
    >>> data = load_hlogs ().frame 
    >>> xs, xr = get_xs_xr_splits (data, 3.11, section_indexes = (17, 20 ) )
    """
    xs, xr = None, None
    
    if section_indexes is not None: 
        section_indexes = _assert_reduce_indexes (section_indexes) [0] 
        xr = df.iloc [range (*section_indexes)]
        invalid_indexes = _get_invalid_indexes(
            np.arange (len(df)), section_indexes)  

    # valid section index of aquifer
    elif z_range is not None : 
        z = is_valid_depth (df, zname = zname , return_z = True)
        section_indexes, invalid_indexes = get_sections_from_depth(
            z, z_range, return_indexes=True )

    # +1 for Python index 
    xr = df.iloc [range (*[section_indexes[0], section_indexes[-1] +1])]

    invalid_indexes = _assert_reduce_indexes(*invalid_indexes )
    max_ix = max (list(itertools.chain(*invalid_indexes)))
    
    if  max_ix > len(df) :
        raise IndexError(f"Wrong index! Index {max_ix} is out of range "
                         f"of data with length = {len(df)}")
 
    xs = [ df.iloc[ range (* ind)] for ind in invalid_indexes]

    return xs, xr 

def samples_reducing (
    *data , 
    sname, 
    zname=None, 
    kname= None,
    section_indexes=None,  
    error='raise', 
    strategy= 'average',  
    verify_integrity=False, 
    ignore_index=False, 
    **kws
    )->List[DataFrame] : 
    
    msg = ("'Soft' mode is triggered for samples reducing."
           " {0} number{1} of data passed are not valid."
           " Remember that data must contain the 'depth' and"
           " aquifer  values. Should be discarded during the"
           " computing of aquifer sections. This might lead to"
           " breaking code or invalid results. Use at your own "
           " risk." 
        )

    df0 = copy.deepcopy(data) # make a copy of frame 
    dfs = _validate_samples( *df0 )  
    
    dfs=[df.reset_index() for df in dfs] # reset index 
    # get the aquifer sections firts 
    if section_indexes is None: 
        section_indexes, dfs = get_unique_sections(
            *dfs, zname=zname, kname=kname, error= error, 
            return_data =True, return_index=True 
            )
        
        if len(df0)!=len(dfs): 
            warnings.warn ( msg.format(len(section_indexes), 
                        "s" if len(section_indexes)>1 else ""))
        
    Xs, Xr =[], []
    for df in dfs : 
        xs, xr = get_xs_xr_splits (df, section_indexes= section_indexes)
        Xs.append(xs) ; Xr.append(xr)
        
    d_new=[]
    for  df_xs , df_xr in zip ( Xs , Xr ): 
        # # compute the base stratum for 
        # each each reduce sections 
        bases_s = [ select_base_stratum(d, sname=sname )
                    for i, d in enumerate (df_xs) ] 
        # reduce sample for each invalid section with 
        # missing k 
        comp_vecs = [ get_compressed_vector( d, sname=sname , stratum = st,  
                     as_frame =True , strategy=strategy, 
            ) for i, (st , d)  in enumerate ( zip (bases_s , df_xs))  ]
        # get the index to stack the compresed sample with 
        # the valid part of aquifer data. 
        xs_indexes = [( min( df.index), max(df.index)) for df in df_xs ]
        # concat the compress with xr 
        df_= _concat_compressed_xs_xr(
            xs_indexes =xs_indexes ,xr_indexes = section_indexes, 
                compressed_frames = comp_vecs, 
                xr= df_xr )
        d_new.append (df_)

    if not ignore_index: 
        # got back inial data. 
        d_new = [ df.drop ( columns = 'index') 
                  if 'index' in df.columns else df 
                  for df in d_new 
                  ]
    # verify integrity first
    # before reset index 
    if verify_integrity: 
        d_new = [  df.drop_duplicates(subset=None, keep='first',  
            ignore_index=ignore_index ) for df in d_new ] 
        
    if ignore_index : 
        # reset the index of the new data frame
        d_new = [df.reset_index () for df in d_new ]
        d_new = [ df.drop (columns = 'level_0' or 'index') if
                 ('level_0' or 'index')  in df.columns else df 
                 for df in d_new  ]
    
    return d_new 

samples_reducing.__doc__ ="""\
Create a new dataframe with reducing/crompressing the non valid data. 

The m-samples reduction is necessary for the dataset with a lot of 
missing k-values. The technique of shrinking the number of k0 –values 
(k-missing values ) seems a relevant idea. It consists to compressed the 
values of the missing :math:`k -values from the top ( depth equals 0 ) 
thin the upper section of the first aquifer with lower depth into 
a single vector :math:`x_r` with dimension (1×n ) i.e. contains 
the n-features.  
 
Parameters 
-----------
data: list of dataframes
    Data that contains mainly the aquifer values. It must contains the 
    depth values refering at the column_name passed at `zname`  and 
    the permeability coefficient `k` passed to `kname` . Both argument need 
    t supplied when datafame as passes as positional arguments.
    
sname: str, optional 
    Name of column in the dataframe that contains the strata values. 
    Dont confuse 'sname' with 'stratum' which is the name of the valid 
    layer/rock in the array/Series of strata. 

{params.core.zname}
{params.core.kname}
{params.core.z}

strategy: str , default='average' or 'mean', 
    strategy used to select or compute the numerical data into a 
    singular series. It can be ['naive']. In that case , a single serie 
    if randomly picked up into the base strata data.
    
section_indexes: tuple or list of int 
    list of a pair tuple or list of integers. It is be the the valid 
    sections( upper and lower ) indexes of  of the aquifer. If 
    the depth range `z_range` and `zname` are supplied, `section_indexes`
    can be None.  Note that the last indix is considered as the last 
    position, the bottom of the section therefore, its value is 
    included in the data.
        
error: str, default='raise' 
    Raise errors if trouble occurs when computing the section of each aquifer. 
    If 'ignore', a UserWarning is displayed when invalid data is found. Any 
    other value of `error` will set error to `raise`. 

verify_integrity: bool, default=False
    Check the new index for duplicates. Otherwise defer the check until 
    necessary. Setting to False will improve the performance of 
    this method.
    if 'True', remove the duplicate rows from a DataFrame.
    
        subset: By default, if the rows have the same values in all the 
        columns, they are considered duplicates. This parameter is used 
        to specify the columns that only need to be considered for 
        identifying duplicates.
        keep: Determines which duplicates (if any) to keep. It takes inputs as,
        first – Drop duplicates except for the first occurrence. 
        This is the default behavior.
        last – Drop duplicates except for the last occurrence.
        False – Drop all duplicates.
        inplace: It is used to specify whether to return a new DataFrame or 
        update an existing one. It is a boolean flag with default False.
ignore_index: bool, default=False, 
    It is a boolean flag to indicate if row index should 
    be reset after dropping duplicate rows. False: It keeps the original 
    row index. True: It reset the index, and the resulting rows will be 
    labeled 0, 1, …, n – 1. 
    
Returns 
----------
df_new: List of pandas.dataframes
    new dataframes with reducing samples. 
    
Example 
--------
>>> from watex.datasets import load_hlogs 
>>> data = load_hlogs ().frame # get the frames 
>>> # add explicitly the aquifer indi
>>> dfnew= samples_reducing (data.copy(), sname='strata_name', # data, zname='depth', kname='k', 
                      section_indexes = (16, 29 ),)
>>> dfnew[0]
...    hole_number               strata_name     rock_name  ...      r     rp  remark
0         H502                  mudstone           J2z  ...    NaN    NaN     NaN
16        H502                 siltstone           NaN  ...  35.74  59.23     NaN
17        H502    fine-grained sandstone           NaN  ...  35.74  59.23     NaN
18        H502                 siltstone           NaN  ...  35.74  59.23     NaN
19        H502    fine-grained sandstone           NaN  ...  35.74  59.23     NaN
20        H502                  mudstone           NaN  ...  35.74  59.23     NaN
21        H502                 siltstone           NaN  ...  35.74  59.23     NaN
22        H502    fine-grained sandstone           NaN  ...  59.61  59.23     NaN
23        H502                 siltstone           NaN  ...  59.61  59.23     NaN
24        H502    fine-grained sandstone           NaN  ...  59.61  59.23     NaN
25        H502  Coarse-grained sandstone           NaN  ...  59.61  59.23     NaN
26        H502                  mudstone           NaN  ...  82.33  59.23     NaN
27        H502    fine-grained sandstone           NaN  ...  82.33  59.23     NaN
28        H502  Coarse-grained sandstone           J2z  ...  82.33  59.23     NaN
29        H502                      coal  (J2y)  2coal  ...  82.33  59.23     NaN
0         H502                 siltstone           NaN  ...    NaN    NaN     NaN

[16 rows x 23 columns]
>>> # specify the column name and knames without section indexes 
>>> dfnew= samples_reducing (
    data.copy(), sname='strata_name', data, zname='depth', kname='k', 
    ignore_index= True )[0]
... dfnew[0].index # index is reset 
.. RangeIndex(start=0, stop=16, step=1)

""".format(
    params=_param_docs,
    )
def _concat_compressed_xs_xr (
        xs_indexes:List[int], 
        xr_indexes: List[int], 
        compressed_frames:List[DataFrame], 
        xr:DataFrame  ):
    """ Concat the compressed frames from `xs` with the valid frames.
    
    Use the index of different frames to merge the frame by respecting the 
    depth positions. For instance, if the valid secion of aquifer is framed 
    between two invalid sections composed of missing 'k' values, the both
    sections are shrank and their compressed frames are also framed the 
    section of valid data. This keep the position of the 
    aquifer intact. This is usefull for prediction purpose. 
    
    :param xs_indexes: list of int 
        indices of invalid sections 
    :param xr_indexes: list of int ,
        indices of valid section of aquifer. valid data 
    :param compressed_frames: pandas dataframe 
        the compressed frames from `xs`. 
    :param xr: dataframe 
        valid data ( contain the aquifer sections )
    """
    pos = [ np.array(k).mean() for k in xs_indexes ]
    dics = dict ( zip ( pos , compressed_frames))
    
    dics [np.array(xr_indexes).mean()]= xr 
    # sorted strata in ascending occurence 
    sm = dict (
        sorted (dics.items () , key= lambda x:x[0])
        )
    c= list(sm.values ())
    return  pd.concat (c )

    
def is_valid_depth (z, /, zname =None , return_z = False): 
    """ Assert whether depth is valid in dataframe of two-dimensional 
    array passed to `z` argument. 
    
    Parameters 
    ------------
    z: ndarray, pandas series or dataframe 
        If Dataframe is given, 'zname' must be supplied to fetch or assert 
        the depth existence of the depth in `z`. 
    zname: str,int , 
        the name of depth column. 'name' needs to be supplied when `z` is 
        given whereas index is needed when `z` is an ndarray with two 
        dimensional. 
        
    return_X_z: bool, default =False
        returns z series or array  if set to ``True``. 
    
    Returns 
    ---------
    z0, is_z: array /bool, 
        An array-like 1d of `z` or 'True/False' whether z exists or not. 
        
    Example 
    --------
    >>> from watex.datasets import load_hlogs 
    >>> from watex.utils.hydroutils import is_valid_depth 
    >>> d= load_hlogs () 
    >>> X= d.frame 
    >>> is_valid_depth(X, zname='depth') # is dataframe , need to pass 'zname'
    ... True
    >>> is_valid_depth (X, zname = 'depth', return_z = True)
    ... 0        0.00
        1        2.30
        2        8.24
        3       22.46
        4       44.76
         
        176    674.02
        177    680.18
        178    681.68
        179    692.97
        180    693.37
        Name: depth_top, Length: 181, dtype: float64
    """
    is_z =True 
    z = _assert_all_types(z, np.ndarray , pd.Series, pd.DataFrame, 
                          objname ='Depth') 
    zname = _assert_all_types(zname, str, objname ="'zname"
                              ) if zname is not None else None  
    if hasattr(z, '__array__') and hasattr (z, 'name'): 
        zname = z.name 
        
    elif hasattr (z ,'columns' ): 
        # assert whether depth 
        # mape a copy to not corrupt X since the function 
        # remove the depth in columns 
        z_copy = z.copy() 
        if zname is None: 
            raise ValueError ("'zname' ( Depth column name ) can not be None"
                              " when a dataframe is given.")
        # --> deals with depth 
        # in the case depth is given while 
        # dataframe is given. 
        # if z is not None: 
        #     zname =None # set None 
        if zname is not None : 
            # erased the depth and name
            try: 
                _, z0 = is_depth_in(
                z_copy, name = zname, error = 'raise') 
            except Exception as err:
                if return_z: 
                    raise DepthError("Depth name 'zname' " + str(
                        err).replace ('E', 'e') )
                    
                else: is_z= False  
                
        zname= z0.name 
    elif hasattr (z, '__array__'): 
        if not _is_arraylike_1d (z): 
            raise ValueError ("Multidimensional 'k' array is not allowed"
                              " Expect one-dimensional array.")
        z0= pd.Series (z, name =zname) if zname is not None else z 

    return z0 if return_z else is_z  

def get_aquifer_sections (
        arr_k, /, zname=None, kname = None,  z= None, 
        return_indexes = False, return_sections = True 
        ) : 
    _assert_all_types( arr_k, pd.DataFrame, np.ndarray)
    
    if z is not None: 
        ms = (f"Depth {type(z).__name__} size must be consistent with"
             f" {type (arr_k).__name__!r};got {len(z)} and {len(arr_k)}."
             )
        _assert_all_types(z, np.ndarray, pd.Series)
        
        if not _is_arraylike_1d(z): 
            raise DepthError ("Depth supports only one-dimensional array,"
                             f" not {type(z).__name__!r}.")
        if len(z)!= len(arr_k): 
            raise DepthError (ms)
                
    if (z is None and zname is not None ): 
        z = is_valid_depth ( arr_k , zname = zname , return_z = True )
        zname = z.name 
        
    elif ( z is None and zname is None ): 
           raise TypeError ("Expects an array of depth 'z' or  depth column"
                            " name 'zname' in the dataframe.")    
        
    if hasattr (arr_k ,'columns' ):
        # deal with arr_k 
        if kname is None: 
            raise ValueError ("Permeability coefficient 'k' name can not "
                              "be None when a dataframe is given.") 
        else: 
            _assert_all_types(kname, str , int , float,  objname="'kname'") 
            
        if isinstance (kname , (int, float)): 
            kname = int (kname) 
            if kname > len(arr_k.columns): 
                raise IndexError (f"'kname' at index {kname} is out of the "
                                  f"dataframe column size={len(arr_k.columns)}")
                
            kname = arr_k.columns[kname]
            
        if kname not in arr_k.columns:
            raise ValueError (f"'kname' {kname!r} not found in dataframe.")
        
        arr_k = arr_k[kname] 
        arr_k= arr_k.values 
        
    elif hasattr (arr_k, '__array__'): 
        if not _is_arraylike_1d (arr_k): 
            raise ValueError ("Multidimensional 'k' array is not allowed"
                              " Expect one-dimensional array.")

    # for consistency, set all to 1d array 
    z = reshape (z) ; arr_k = reshape (arr_k)

    indexes,  = np.where (~np.isnan (arr_k)) 
    if hasattr (indexes, '__len__'): 
        # +1 for Python indexing
        indexes =[ indexes [0 ] , indexes [-1]] 
        
    sections = z[indexes ]
    
    return ( [* indexes ], [* sections ])   if ( 
        return_indexes and return_sections ) else  ( 
            [*indexes ] if return_indexes else  [*sections])

get_aquifer_sections.__doc__="""\
Detect aquifer sections (upper and lower) sections 

Detects the section of aquifer in depth. 

Parameters 
-----------
arr_k: ndarray or dataframe 
    Data that contains mainly the aquifer values. It can also contains the 
    depth values. If the depth is included in the `arr_k`, `zname` needs to 
    be supplied for recovering and depth. 
    
{params.core.zname}
{params.core.kname}
{params.core.z}

return_indexes: bool, default =False , 
    Returns the positions (indexes) of the upper and lower sections of the
     aquifer found in the dataframe `arr_k`. 
return_sections: bool, default=True, 
    Returns the sections (upper and lower) of the aquifers. 

Returns 
--------
up, low :list of upper and lower section values of aquifer.
    - (upix, lowix ): Tuple of indexes of lower and upper sections  
    - (up, low): Tuple of aquifer sections (upper and lower)  
    - (upix, lowix), (up, low) : positions and sections values of aquifers 
        if `return_indexes` and return_sections` are ``True``.  

Example
-------
>>> from watex.datasets import load_hlogs 
>>> from watex.utils.hydroutils import get_aquifer_sections 
>>> data = load_hlogs ().frame # return all data including the 'depth' values 
>>> get_aquifer_sections (data , zname ='depth', kname ='k')
... [197.12, 369.71] # section starts from 197.12 -> 369.71 m 
>>> get_aquifer_sections (data , zname ='depth', kname ='k', return_indexes=True) 
... ([16, 29], [197.12, 369.71]) # upper and lower-> position 16 and 29.


""".format(
    params=_param_docs,
    )
    
def _kp (k, /,  kr= (.01 , .07 ), string = False ) :
    """ Default permeability 'k' mapping using dict to validate the continue 
    value 'k' 
    :param k: float, 
        continue value of the permeability coefficient 
    :param kr: Tuple, 
        range of permeability coefficient to categorize 
    :param string: bool, str 
        label to prefix the the categorial value. 
    :return: float/str - new categorical value . 

    """
    d = {0: k <=0 , 1: 0 < k <= kr[0], 2: kr[0] < k <=kr[1], 3: k > kr[1] 
         }
    label = 'k' if str(string).lower()=='true' else str(string )
    for v, value in d.items () :
        if value: return v if not string else  ( 
                label + str(v) if not math.isnan (v) else np.nan ) 
        
def map_k (
        o:DataFrame| Series | ArrayLike, /,  ufunc: callable|F= None , 
        kname:str=None, inplace:bool =False, string:str =False, 
        default_ufunc:bool=False  
        ):
    """ Categorize the permeability coefficient 'k'
    
    Map the continuous 'k' into categorial classes. 
    
    Parameters 
    ----------
    o: ndarray of pd.Series or Dataframe
        data containing the permeability coefficient k columnns 
    unfunc: callable 
        Function to specifically map the permeability coefficient column 
        in the dataframe of serie. If not given, the default function can be 
        enabled instead from param `use_default_ufunc`. 
    inplace: bool, default=False 
        Modified object inplace and return None 
    string: bool, 
        If set to "True", categorized map from 'k'  should be prefixed by "k". 
        However is string value is given , the prefix is changed according 
        to this label. 
    default_ufunc: bool, 
        Default function for mapping k is setting to ``True``. Note that, this 
        could probably not fitted your own data. So  it is recommended to 
        provide your own function for mapping 'k'. However the default 'k' 
        mapping is given as follow: 
            
        - k0 {0}: k = 0 
        - k1 {1}: 0 < k <= .01 
        - k2 {2}: .01 < k <= .07 
        - k3 {3}: k> .07 
    Returns
    --------
    o: None,  ndarray, Series or Dataframe 
        return None only if dataframe is given and `inplace` is set 
        to ``True`` i.e modified object inplace. 
        
    Examples 
    --------
    >>> import numpy as np 
    >>> from watex.datasets import load_hlogs 
    >>> from watex.utils.hydroutils import map_k 
    >>> _, y0 = load_hlogs (as_frame =True) 
    >>> # let visualize four nonzeros values in y0 
    >>> y0.k.values [ ~np.isnan (y0.k ) ][:4]
    ...  array([0.054, 0.054, 0.054, 0.054])
    >>> map_k (y0 , kname ='k', inplace =True, use_default_ufunc=True )
    >>> # let see again the same four value in the dataframe 
    >>> y0.k.values [ ~np.isnan (y0.k ) ][:4]
    ... array([2., 2., 2., 2.]) 
    
    """
    _assert_all_types(o, pd.Series, pd.DataFrame, np.ndarray)
    
    dfunc = lambda k : _kp (k, string = string ) # default 
    ufunc = ufunc or   ( dfunc if default_ufunc else None ) 
    if ufunc is None: 
        raise TypeError ("'ufunc' can not be None when the default"
                         " 'k' mapping function is not triggered.")
    oo= copy.deepcopy (o )
    if hasattr (o, 'columns'):
        if kname is None: 
            raise ValueError ("kname' is not set while dataframe is given. "
                              "Please specify the name of permeability column.")
        is_in_if( o, kname )
  
        if inplace : 
            o[kname] = o[kname].map (ufunc) 
            return 
        oo[kname] = oo[kname].map (ufunc) 
        
    elif hasattr(o, 'name'): 
        oo= oo.map(ufunc ) 
  
    elif hasattr(o, '__array__'): 
        oo = np.array (list(map (ufunc, o )))
        
    return oo 

        
#XXXTODO compute t parameters 
def transmissibility (s, d, time, ): 
    """Transmissibility T represents the ability of aquifer's water conductivity.
    
    It is the numeric equivalent of the product of hydraulic conductivity times
    aquifer's thickness (T = KM), which means it is the seepage flow under the
    condition of unit hydraulic gradient, unit time, and unit width
    
    """
      
def check_flow_objectivity ( y ,/,  values, classes  ) :
    """ Function checks the flow rate objectivity
    
    If objective is set to `flow` i.e the prediction focuses on the flow
    rate, there are some conditions that the target `y` needs to meet when 
    values are passed for classes categorization. 
    
    :param values: list of values to encoding the numerical target `y`. 
        for instance ``values=[0, 1, 2]`` 
    :param objective: str, relate to the flow rate prediction. Set to 
        ``None`` for any other predictions. 
    :param prefix: the prefix to add to the class labels. For instance, if 
        the `prefix` equals to ``FR``, class labels will become:: 
            
            [0, 1, 2] => [FR0, FR1, FR2]
            
    :classes: list of classes names to replace the default `FR` that is 
        used to specify the flow rate. For instance, it can be:: 
            
            [0, 1, 2] => [sf0, sf1, sf2]
    """
    msg= ("Objective is 'flow' whereas the target value is set to {0}."
          " Target is defaultly encoded to hold integers {1}. If"
          " the auto-categorization does not fit the real values"
          " of flow ranges, please set the range of the real flow values"
          " via param `values` or `label_values`."
          ) 
    if values is None:
        msg = ("Missing values for categorizing 'y'; the number of"
                " occurence in the target is henceforth not allowed."
                )
        warnings.warn("Values are not set. The new version does not" 
                      " tolerate the number of occurrence to be used."
                      "Provided the list of flow values instead.",
                      DeprecationWarning )
        raise TypeError (msg)
        
    elif values is not None: 
        if isinstance(values,  (int, float)): 
           y =  categorize_target(y , labels = int(values) )
           warnings.warn(msg.format(values, np.unique (y) ))
           values = np.unique (y)
        
        elif isinstance(values, (list, tuple, np.ndarray)):
            y = np.unique(y) 
            if len(values)!=len(y): 
                warnings.warn("Size of unique identifier class labels"
                              " and the given values might be consistent."
                              f" Idenfier sizes = {len(y)} whereas given "
                              f" values length are ={len(values)}. Will"
                              " use the unique identifier labels instead.")
                values = y 
                
            y = categorize_flow(y, values, classes=classes  )
        else : 
            raise ValueError("{type (values).__name__!r} is not allow"
                             " Expect a list of integers.")
            
    classes = classes or values 
    return y, classes 
 
@catmapflow2(cat_classes=['FR0', 'FR1', 'FR2', 'FR3'])#, 'FR4'] )
def categorize_flow(
        target: Series | ArrayLike[T] ,
        flow_values: List [float],
        **kwargs
    ) -> Tuple[ List[float], T, List[str]]: 
    """ 
    Categorize `flow` into different classes. If the optional
    `flow_classes`  argument is given, it should be erased the
    `cat_classes` argument of decororator `deco.catmapflow`.
    
    Parameters 
    ------------
    target: array-like, pandas.Series, 
        Flow array to be categorized
    
    flow_values: list of str 
        Values for flow categorization; it distributes the flow values as
        numerical values. For instance can be ranged as a tuple of bounds 
        as below :: 
    
            flow_values= [0.0, [0.0, 3.0], [3.0, 6.0], [6.0, 10.0], 10.0] (1)
            
        or it can also accept the list of integer label identifiers as::
            
            flow_values =[0. , 3., 6., 10.] (2)
        
        For instance runing the step (2) shoud convert the flow rate bounds to 
        reach the step (1). The arrangement of the flow rate obeys some criteria 
        which depend of the types of hydraulic system required according to the
        number of inhabitants living on a survey locality/villages or town.
        The common request flow rate during the campaigns for drinling 
        water supply can be  organized as follow: 
            
            flow_values =[0,  1,  3 , 10  ]
            classes = ['FR0', 'FR1', 'FR2', 'FR3']
    
        where :
            - ``FR0`` equals to values =0  -> dry boreholes 
            - ``FR1`` equals to values between  0-1(0< value<=1) for Village 
                hydraulic systems (VH)
            - ``FR2`` equals to values between  1-1 (1< value<=3) for improved  
                village hydraulic system (IVH)
            - ``FR3`` greather than 3 (>3) for urban hydraulic system (UH)
            
            Refer to [1]_ for more details. 
        
    classes: list of str , 
        literal labels of categorized flow rates. If given, should be 
        consistent with the size of `flow_values`'
    
        
    Returns 
    ---------
    (new_flow_values, target, classes)
        - ``new_flow_values``: Iterable object as type (2) 
        - ``target``: Raw flow iterable object to be categorized
        - ``classes``: If given , see ``classes`` params. 
            
    References 
    -------------
    .. [1] Kouadio, K.L., Kouame, L.N., Drissa, C., Mi, B., Kouamelan, K.S., 
        Gnoleba, S.P.D., Zhang, H., et al. (2022) Groundwater Flow Rate 
        Prediction from Geo‐Electrical Features using Support Vector Machines. 
        Water Resour. Res. :doi:`10.1029/2021wr031623`
        
    .. [2] Kra, K.J., Koffi, Y.S.K., Alla, K.A. & Kouadio, A.F. (2016) Projets 
        d’émergence post-crise et disparité territoriale en Côte d’Ivoire. 
        Les Cah. du CELHTO, 2, 608–624.
        
        
    """
    classes =  kwargs.pop('classes', None)

    new_flow_values = []
    inside_inter_flag= False
    
    if isinstance(flow_values, (tuple, np.ndarray)): 
        flow_values =list(flow_values)
    # Loop and find 
    for jj, _iter in enumerate(flow_values) : 
        if isinstance(_iter, (list, tuple, np.ndarray)): 
            inside_inter_flag = True 
            flow_values[jj]= list(_iter)
 
    if inside_inter_flag: 
        new_flow_values =flow_values 
    
    if inside_inter_flag is False: 
        flow_values= sorted(flow_values)
        # if 0. in flow_values : 
        #     new_flow_values.append(0.) 
        for ss, val in enumerate(flow_values) : 
            if ss ==0 : 
                #append always the first values. 
                 new_flow_values.append(val) 
            # if val !=0. : 
            else:
                if val ==flow_values[-1]: 
                    new_flow_values.append([flow_values[ss-1], val])
                    new_flow_values.append(val)
                else: 
                   new_flow_values.append([flow_values[ss-1], val])
 
    return new_flow_values, target, classes        

@writef(reason='write', from_='df')
def exportdf (
    df : DataFrame =None,
    refout: Optional [str] =None, 
    to: Optional [str] =None, 
    savepath:Optional [str] =None,
    modname: str  ='_wexported_', 
    reset_index: bool =True
) -> Tuple [DataFrame, Union[str], bool ]: 
    """ 
    Export dataframe ``df``  to `refout` files. 
    
    `refout` file can be Excell sheet file or '.json' file. To get more details 
    about the `writef` decorator , see :doc:`watex.utils.decorator.writef`. 
    
    :param refout: 
        Output filename. If not given will be created refering to the 
        exported date. 
        
    :param to: Export type; Can be `.xlsx` , `.csv`, `.json` and else.
       
    :param savepath: 
        Path to save the `refout` filename. If not given
        will be created.
    :param modname: Folder to hold the `refout` file. Change it accordingly.
        
    :returns: 
        - `df_`: new dataframe to be exported. 
        
    """
    if df is None :
        warnings.warn(
            'Once ``df`` arguments in decorator :`class:~decorator.writef`'
            ' is selected. The main type of file ready to be written MUST be '
            'a pd.DataFrame format. If not an error raises. Please refer to '
            ':doc:`~.utils.decorator.writef` for more details.')
        
        raise FileHandlingError(
            'No dataframe detected. Please provided your dataFrame.')

    df_ =df.copy(deep=True)
    if reset_index is True : 
        df_.reset_index(inplace =True)
    if savepath is None :
        savepath = savepath_(modname)
        
    return df_, to,  refout, savepath, reset_index   

def categorize_target(
        arr :ArrayLike |Series , /, 
        func: F = None,  
        labels: int | List[int] = None, 
        rename_labels: Optional[str] = None, 
        coerce:bool=False,
        order:str='strict',
        ): 
    """ Categorize array to hold the given identifier labels. 
    
    Classifier numerical values according to the given label values. Labels 
    are a list of integers where each integer is a group of unique identifier  
    of a sample in the dataset. 
    
    Parameters 
    -----------
    arr: array-like |pandas.Series 
        array or series containing numerical values. If a non-numerical values 
        is given , an errors will raises. 
    func: Callable, 
        Function to categorize the target y.  
    labels: int, list of int, 
        if an integer value is given, it should be considered as the number 
        of category to split 'y'. For instance ``label=3`` and applied on 
        the first ten number, the labels values should be ``[0, 1, 2]``. 
        If labels are given as a list, items must be self-contain in the 
        target 'y'.
    rename_labels: list of str; 
        list of string or values to replace the label integer identifier. 
    coerce: bool, default =False, 
        force the new label names passed to `rename_labels` to appear in the 
        target including or not some integer identifier class label. If 
        `coerce` is ``True``, the target array holds the dtype of new_array. 

    Return
    --------
    arr: Arraylike |pandas.Series
        The category array with unique identifer labels 
        
    Examples 
    --------

    >>> from watex.utils.mlutils import cattarget 
    >>> def binfunc(v): 
            if v < 3 : return 0 
            else : return 1 
    >>> arr = np.arange (10 )
    >>> arr 
    ... array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> target = cattarget(arr, func =binfunc)
    ... array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=int64)
    >>> cattarget(arr, labels =3 )
    ... array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    >>> array([2, 2, 2, 2, 1, 1, 1, 0, 0, 0]) 
    >>> cattarget(arr, labels =3 , order =None )
    ... array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    >>> cattarget(arr[::-1], labels =3 , order =None )
    ... array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]) # reverse does not change
    >>> cattarget(arr, labels =[0 , 2,  4]  )
    ... array([0, 0, 0, 2, 2, 4, 4, 4, 4, 4])

    """
    arr = _assert_all_types(arr, np.ndarray, pd.Series) 
    is_arr =False 
    if isinstance (arr, np.ndarray ) :
        arr = pd.Series (arr  , name = 'none') 
        is_arr =True 
        
    if func is not None: 
        if not  inspect.isfunction (func): 
            raise TypeError (
                f'Expect a function but got {type(func).__name__!r}')
            
        arr= arr.apply (func )
        
        return  arr.values  if is_arr else arr   
    
    name = arr.name 
    arr = arr.values 

    if labels is not None: 
        arr = _cattarget (arr , labels, order =order)
        if rename_labels is not None: 
            arr = rename_labels_in( arr , rename_labels , coerce =coerce ) 

    return arr  if is_arr else pd.Series (arr, name =name  )

def rename_labels_in (arr, new_names, coerce = False): 
    """ Rename label by a new names 
    
    :param arr: arr: array-like |pandas.Series 
         array or series containing numerical values. If a non-numerical values 
         is given , an errors will raises. 
    :param new_names: list of str; 
        list of string or values to replace the label integer identifier. 
    :param coerce: bool, default =False, 
        force the 'new_names' to appear in the target including or not some 
        integer identifier class label. `coerce` is ``True``, the target array 
        hold the dtype of new_array; coercing the label names will not yield 
        error. Consequently can introduce an unexpected results.
    :return: array-like, 
        An array-like with full new label names. 
    """
    
    if not is_iterable(new_names): 
        new_names= [new_names]
    true_labels = np.unique (arr) 
    
    if labels_validator(arr, new_names, return_bool= True): 
        return arr 

    if len(true_labels) != len(new_names):
        if not coerce: 
            raise ValueError(
                "Can't rename labels; the new names and unique label" 
                " identifiers size must be consistent; expect {}, got " 
                "{} label(s).".format(len(true_labels), len(new_names))
                             )
        if len(true_labels) < len(new_names) : 
            new_names = new_names [: len(new_names)]
        else: 
            new_names = list(new_names)  + list(
                true_labels)[len(new_names):]
            warnings.warn("Number of the given labels '{}' and values '{}'"
                          " are not consistent. Be aware that this could "
                          "yield an expected results.".format(
                              len(new_names), len(true_labels)))
            
    new_names = np.array(new_names)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # hold the type of arr to operate the 
    # element wise comparaison if not a 
    # ValueError:' invalid literal for int() with base 10' 
    # will appear. 
    if not np.issubdtype(np.array(new_names).dtype, np.number): 
        arr= arr.astype (np.array(new_names).dtype)
        true_labels = true_labels.astype (np.array(new_names).dtype)

    for el , nel in zip (true_labels, new_names ): 
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # element comparison throws a future warning here 
        # because of a disagreement between Numpy and native python 
        # Numpy version ='1.22.4' while python version = 3.9.12
        # this code is brittle and requires these versions above. 
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # suppress element wise comparison warning locally 
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            arr [arr == el ] = nel 
            
    return arr 

    
def _cattarget (ar , labels , order=None): 
    """ A shadow function of :func:`watex.utils.funcutils.cattarget`. 
    
    :param ar: array-like of numerical values 
    :param labels: int or list of int, 
        the number of category to split 'ar'into. 
    :param order: str, optional, 
        the order of label to ne categorized. If None or any other values, 
        the categorization of labels considers only the leangth of array. 
        For instance a reverse array and non-reverse array yield the same 
        categorization samples. When order is set to ``strict``, the 
        categorization  strictly consider the value of each element. 
        
    :return: array-like of int , array of categorized values.  
    """
    # assert labels
    if is_iterable (labels):
        labels =[int (_assert_all_types(lab, int, float)) 
                 for lab in labels ]
        labels = np.array (labels , dtype = np.int32 ) 
        cc = labels 
        # assert whether element is on the array 
        s = set (ar).intersection(labels) 
        if len(s) != len(labels): 
            mv = set(labels).difference (s) 
            
            fmt = [f"{'s' if len(mv) >1 else''} ", mv,
                   f"{'is' if len(mv) <=1 else'are'}"]
            warnings.warn("Label values must be array self-contain item. "
                           "Label{0} {1} {2} missing in the array.".format(
                               *fmt)
                          )
            raise ValueError (
                "label value{0} {1} {2} missing in the array.".format(*fmt))
    else : 
        labels = int (_assert_all_types(labels , int, float))
        labels = np.linspace ( min(ar), max (ar), labels + 1 ) #+ .00000001 
        #array([ 0.,  6., 12., 18.])
        # split arr and get the range of with max bound 
        cc = np.arange (len(labels)) #[0, 1, 3]
        # we expect three classes [ 0, 1, 3 ] while maximum 
        # value is 18 . we want the value value to be >= 12 which 
        # include 18 , so remove the 18 in the list 
        labels = labels [:-1] # remove the last items a
        # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2]) # 3 classes 
        #  array([ 0.        ,  3.33333333,  6.66666667, 10. ]) + 
    # to avoid the index bound error 
    # append nan value to lengthen arr 
    r = np.append (labels , np.nan ) 
    new_arr = np.zeros_like(ar) 
    # print(labels)
    ar = ar.astype (np.float32)

    if order =='strict': 
        for i in range (len(r)):
            if i == len(r) -2 : 
                ix = np.argwhere ( (ar >= r[i]) & (ar != np.inf ))
                new_arr[ix ]= cc[i]
                break 
            
            if i ==0 : 
                ix = np.argwhere (ar < r[i +1])
                new_arr [ix] == cc[i] 
                ar [ix ] = np.inf # replace by a big number than it was 
                # rather than delete it 
            else :
                ix = np.argwhere( (r[i] <= ar) & (ar < r[i +1]) )
                new_arr [ix ]= cc[i] 
                ar [ix ] = np.inf 
    else: 
        l= list() 
        for i in range (len(r)): 
            if i == len(r) -2 : 
                l.append (np.repeat ( cc[i], len(ar))) 
                
                break
            ix = np.argwhere ( (ar < r [ i + 1 ] ))
            l.append (np.repeat (cc[i], len (ar[ix ])))  
            # remove the value ready for i label 
            # categorization 
            ar = np.delete (ar, ix  )
            
        new_arr= np.hstack (l).astype (np.int32)  
        
    return new_arr.astype (np.int32)  


def labels_validator (t, /, labels, return_bool = False): 
    """ Assert the validity of the label in the target  and return the label 
    or the boolean whether all items of label are in the target. 
    
    :param t: array-like, target that is expected to contain the labels. 
    :param labels: int, str or list of (str or int) that is supposed to be in 
        the target `t`. 
    :param return_bool: bool, default=False; returns 'True' or 'False' rather 
        the labels if set to ``True``. 
    :returns: bool or labels; 'True' or 'False' if `return_bool` is set to 
        ``True`` and labels otherwise. 
        
    :example: 
    >>> from watex.datasets import fetch_data 
    >>> from watex.utils.mlutils import cattarget, labels_validator 
    >>> _, y = fetch_data ('bagoue', return_X_y=True, as_frame=True) 
    >>> # binarize target y into [0 , 1]
    >>> ybin = cattarget(y, labels=2 )
    >>> labels_validator (ybin, [0, 1])
    ... [0, 1] # all labels exist. 
    >>> labels_validator (y, [0, 1, 3])
    ... ValueError: Value '3' is missing in the target.
    >>> labels_validator (ybin, 0 )
    ... [0]
    >>> labels_validator (ybin, [0, 5], return_bool=True ) # no raise error
    ... False
        
    """
    
    if not is_iterable(labels):
        labels =[labels] 
        
    t = np.array(t)
    mask = np.isin(t, labels) 
    true_labels = np.unique (t[mask]) 
    # set the difference to know 
    # whether all labels are valid 
    remainder = list(set(labels).difference (true_labels))
    
    isvalid = True 
    if len(remainder)!=0 : 
        if not return_bool: 
            # raise error  
            raise ValueError (
                "Label value{0} {1} {2} missing in the target 'y'.".format ( 
                f"{'s' if len(remainder)>1 else ''}", 
                f"{smart_format(remainder)}",
                f"{'are' if len(remainder)> 1 else 'is'}")
                )
        isvalid= False 
        
    return isvalid if return_bool else  labels 

def _validate_samples (*dfs , error:str ='raise'): 
    """ Validate data . 
     check shapes and the columns items in the data.
     
    :param dfs: list of dataframes or array-like 
        Dataframe must have the same size along axis 1. If error is 'ignore'
        error is muted if the length ( along axis 0) of data does not fit 
        each other. 
    :param error: str, default='raise' 
        Raise absolutely error if data has not the same shape, size and items 
        in columns. 
    :return: 
        valid_dfs: List of valida data. If 'error' is 'ignore' , It still 
        returns the list of valid data and excludes the invalid all times 
        leaving an userwarnmimg.
        
    """
    shape_init = dfs[0].shape[1]
    [ _assert_all_types(df, np.ndarray, pd.DataFrame) for df in dfs ]
    diff_shape , shapes  , cols = [], [],[]
    
    col_init = dfs[0].columns if hasattr (dfs[0] , 'columns') else [] 
    valid_dfs =[]
    for k , df in enumerate (dfs) : 
        if df.shape[1] != shape_init :
            diff_shape.append(k) 
        else: valid_dfs.append (df )
        
        shapes.append (df.shape)
        if hasattr (df, 'columns'): 
            cols.append (list(df.columns ))
            
    countshapes = list(Counter (shapes )) # iterable object 
    occshapes = countshapes [0] # the most occurence shape
    if len(diff_shape )!=0 : 
        v=f"{'s' if len(diff_shape)>1 else ''}"
        mess = ("Shapes for all data must be consistent; got " 
                f"at the position{v} {smart_format(diff_shape)}.")
        
        if error =='raise': 
            raise ValueError (mess + f" Expects {occshapes}")

        warnings.warn(mess + f"The most frequent shape is {occshapes}"
                      " Please check or reverify your data. This might lead to"
                      " breaking code or invalid results. Use at your own risk."
                      )
        shape1 = list(map (lambda k:k[1],  countshapes))
        
        if set (shape1) !=1 : 
            raise ValueError ("Shape along axis 1 must be consistent. "
                              f"Got {smart_format (countshapes)}. Check the "
                              f"data at position{v} {smart_format(diff_shape)} "
                ) 
            
    colsset = set ( list(itertools.chain (*cols ) ) ) 
 
    if len(colsset ) != len(col_init) : 
        raise DatasetError ("Expect identical columns for all data"
                            " Please check your data.") 
    
    return valid_dfs 


