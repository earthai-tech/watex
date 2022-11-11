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
    DatasetError
    )
from .funcutils import  (
    _assert_all_types, 
    is_iterable,
    is_in_if , 
    smart_format, 
    savepath_ , 
    is_depth_in, 
    reshape 
    
    )
from ..exlib.sklearn import SimpleImputer 

from .validator import _is_arraylike_1d 
#-----------------------

_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"], 
    )

#XXTODO 
# from sections indexes , get the 

def _validate_samples (*dfs , error ='raise'): 
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
        mess = ("Shapes for all data must be consistent; got different" 
                f" shape{v} at the position{v} {smart_format(diff_shape)}.")
        
        if error =='raise': 
            raise ValueError (mess + f" Expects {occshapes}")

        warnings.warn(mess + f"The most frequent shape is {occshapes}"
                      " Please check or reverify your data to avoid."
                      "a misunderstanding or unpleasant data arrangement.")
        shape1 = list(map (lambda k:k[1],  countshapes))
        
        if set (shape1) !=1 : 
            raise ValueError ("shape along axis 1 must be consistent. "
                              f"Got {smart_format (countshapes)}. Check the "
                              f"data at position{v} {smart_format(diff_shape)} "
                ) 
            
    colsset = set ( list(itertools.chain (*cols ) ) ) 
 
    if len(colsset ) != len(col_init) : 
        raise DatasetError ("Expect identical columns for all data"
                            " Please check your data.") 
    
    return valid_dfs 

        
def compute_compressed_vector (d, /, start= 0 , stop = None, 
                               strategy ='most_frequent') :
    
    pass 

def samples_reducing (*dfs , start=0 , stop =None, zname =None, kname = None,
                      z=None, add_compressed_vector =True, **kws) : 
    
    """ Reduced samples and """
    
    msg = ("Data position {0} passed to arguments {1} not valid."
        " Should be discarded during the computing of aquifer sections."
        " Please check your data and run the script again to have full"
        "  control of the sampling compressing. Remember that data must "
        "contain the 'depth' and aquifer  values."
        )
    is_valid_dfs = [] ; is_not_valid =[]
    
    dfs = _validate_samples( dfs )  
    
    #start, stop = [int (_assert_all_types(o, int )) for o in [start, stop ]]
    
    indexes ,sections =[] , []
    for ii, df in enumerate ( dfs) : 
        try : 
            ix, sec = get_aquifer_sections(
                df , 
                zname = zname , 
                kname = kname , 
                z = z, 
                return_indexes= True 
                )
            is_valid_dfs .append (df )
        except :
            is_not_valid.append (ii)
            
            continue 
        
        indexes.append(ix); sections.append(sec ) 
        
    if len(is_not_valid) !=0 : 
        warnings.warn ( msg.format(smart_format(is_not_valid)))
        
    #==== compte the compressed vectors . 
        
        
    if stop is None: 
        stop = -1 
        
    fupper = lambda s :s[start] ; flower = lambda s :s[stop] 
    ixss, ixst =  list(map ( fupper , indexes )) , list(map ( flower , indexes )) 
    secss, secst= list(map ( fupper, sections )) , list(map ( flower, sections )) 
    min_ix = min(ixss) ; max_ixs = max (ixst)
    up_sec= min(secss) ;  low_sec =max (secst)
    
    
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
            raise ValueError ("Depth name 'zname' can not be None "
                              "when a dataframe is given.")
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
        
def map_kp (o:DataFrame| Series | ArrayLike, /,  ufunc: callable|F= None , 
            kname:str=None, inplace:bool =False, 
            string:str =False, use_default_ufunc:bool=False  
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
    use_default_ufunc: bool, 
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
    >>> from watex.utils.hydroutils import map_kp 
    >>> _, y0 = load_hlogs (as_frame =True) 
    >>> # let visualize four nonzeros values in y0 
    >>> y0.k.values [ ~np.isnan (y0.k ) ][:4]
    ...  array([0.054, 0.054, 0.054, 0.054])
    >>> map_kp (y0 , kname ='k', inplace =True, use_default_ufunc=True )
    >>> # let see again the same four value in the dataframe 
    >>> y0.k.values [ ~np.isnan (y0.k ) ][:4]
    ... array([2., 2., 2., 2.]) 
    
    """
    _assert_all_types(o, pd.Series, pd.DataFrame, np.ndarray)
    
    dfunc = lambda k : _kp (k, string = string ) # default 
    ufunc = ufunc or   ( dfunc if use_default_ufunc else None ) 
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