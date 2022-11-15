# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Created date: Fri Apr 15 10:46:56 2022

"""
core utilities 
========================

Encompasses the main functionalities for class and methods to sucessfully 
run. Somes modules are written and shortcutted for the users to do some 
singular tasks before feeding to the main algorithms. 

"""
from __future__ import  annotations 
import os
import re 
import pathlib
import warnings 
import copy 
import itertools  

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
 
from .._docstring import refglossary 
from .._typing import (
    Any, 
    List ,  
    Union, 
    Tuple,
    Dict,
    Optional,
    NDArray,
    DataFrame, 
    Series,
    ArrayLike, 
    DType, 
    Sub, 
    SP
)
from .._watexlog import watexlog
from ..decorators import refAppender, docSanitizer
from ..property import P , Config
from ..exceptions import ( 
    StationError, 
    HeaderError, 
    ResistivityError,
    ERPError,
    VESError, 
    FileHandlingError, 
)
from .funcutils import (
    smart_format as smft,
    _isin , 
    _assert_all_types,
    accept_types,
    read_from_excelsheets,
    reshape
    ) 
from .gistools import (
    assert_lat_value,
    assert_lon_value,
    convert_position_str2float,
    convert_position_float2str,
    utm_to_ll, 
    project_point_ll2utm, 
    project_point_utm2ll, 
    HAS_GDAL, 
    )

_logger = watexlog.get_watex_logger(__name__)


def _is_readable (
        f:str, 
        **kws
 ) -> DataFrame: 
    """ Assert and read specific files and url allowed by the package
    
    Readable files are systematically convert to a frame.  
    
    :param f: Path-like object -Should be a readable files or url  
    :param kws: Pandas readableformats additional keywords arguments. 
    :return: dataframe - A dataframe with head contents... 
    
    """
    cpObj= Config().parsers 
    
    if isinstance (f, pd.DataFrame): 
        return f 
    
    elif not os.path.isfile :# or ('http' not in f) : 
        try : 
            if 'http' in f: # force pandas read html etc 
                pass 
        except:
            raise TypeError (
            f'Expected a Path-like object or url, got : {type(f).__name__!r}')
        
    _, ex = os.path.splitext(f.strip()) 
    if ex.lower() not in tuple (cpObj.keys()):
        raise TypeError(f"Can only parse the {smft(cpObj.keys(), 'or')} files"
                        )
    try : 
        f = cpObj[ex](f, **kws)
        
    except FileNotFoundError:
        raise FileNotFoundError (
            f"No such file in directory: {os.path.basename (f)!r}")
    except: 
        raise FileHandlingError (
            f" Can not parse the file : {os.path.basename (f)!r}")

    return f 
    

@refAppender(refglossary.__doc__)
def vesSelector( 
    data:str | DataFrame[DType[float|int]] = None, 
    *, 
    rhoa: ArrayLike |Series | List [float] = None, 
    AB :ArrayLike |Series = None, 
    MN: ArrayLike|Series | List[float] =None, 
    index_rhoa: Optional[int]  = None, 
    **kws
) -> DataFrame : 
    """ Assert the validity of |VES| data and return a sanitize dataframe. 
    
    :param rhoa: array-like - Apparent resistivities collected during the 
        sounding. 
        
    :param AB: array-like - Investigation distance between the current 
        electrodes. Note that the `AB` is by convention equals to `AB/2`. 
        It's taken as half-space of the investigation depth.
        
    :param MN: array-like - Potential electrodes distances at each investigation 
        depth. Note by convention the values are half-space and equals to 
        `MN/2`. 
        
    :param f: Path-like object or sounding dataframe. If given, the 
        others parameters could keep the ``None` values. 
        
    :param index_rhoa: int - The index to retrieve the resistivity data of a 
        specific sounding point. Sometimes the sounding data are composed of
        the different sounding values collected in the same survey area into 
        different |ERP| line. For instance:
            
            +------+------+----+----+----+----+----+
            | AB/2 | MN/2 |SE1 | SE2| SE3| ...|SEn |
            +------+------+----+----+----+----+----+
            
        Where `SE` are the electrical sounding data values  and `n` is the 
        number of the sounding points selected. `SE1`, `SE2` and `SE3` are 
        three  points selected for |VES| i.e. 3 sounding points carried out 
        either in the same |ERP| or somewhere else. These sounding data are 
        the resistivity data with a  specific numbers. Commonly the number 
        are randomly chosen. It does not refer to the expected best fracture
        zone selected after the prior-interpretation. After transformation 
        via the function `ves_selector`, the header of the data should hold 
        the `resistivity`. For instance, refering to the table above, the 
        data should be:
            
            +----+----+-------------+-------------+-------------+-----+
            | AB | MN |resistivity  | resistivity | resistivity | ... |
            +----+----+-------------+-------------+-------------+-----+
        
        Therefore, the `index_rhoa` is used to select the specific resistivity
        values i.e. select the corresponding sounding number  of the |VES| 
        expecting to locate the drilling operations or for computation. For 
        esample, ``index_rhoa=1`` should figure out: 
            
            +------+------+----+--------+-----+----+------------+
            | AB/2 | MN/2 |SE2 |  -->   | AB  | MN |resistivity |
            +------+------+----+--------+-----+----+------------+
        
        If `index_rhoa` is ``None`` and the number of sounding curves are more 
        than one, by default the first sounding curve is selected ie 
        `index_rhoa` equals to ``0``.
        
    :param kws: dict - Pandas dataframe reading additionals
        keywords arguments.
        
    :return: -dataframe -Sanitize |VES| dataframe with ` AB`, `MN` and
        `resistivity` as the column headers. 
    
    :Example: 
        
        >>> from watex.utils.coreutils import vesSelector 
        >>> df = vesSelector (data='data/ves/ves_gbalo.csv')
        >>> df.head(3)
        ...    AB   MN  resistivity
            0   1  0.4          943
            1   2  0.4         1179
            2   3  0.4         1103
        >>> df = vesSelector ('data/ves/ves_gbalo.csv', index_rhoa=3 )
        >>> df.head(3) 
        ...    AB   MN  resistivity
            0   1  0.4          457
            1   2  0.4          582
            2   3  0.4          558
    """
    
    for arr in (AB , MN, rhoa): 
        if arr is not None: 
            _assert_all_types(arr, list, tuple, np.ndarray, pd.Series) 
            
    try: 
        index_rhoa =  index_rhoa if index_rhoa is None else int(index_rhoa) 
    except: 
        raise TypeError (
            f'Index is an integer, not {type(index_rhoa).__name__!r}')
        
    if data is not None: 
        if isinstance(data, (str,  pathlib.PurePath)): 
            try : 
                data = _is_readable(data, **kws)
            except TypeError as typError: 
                raise VESError (str(typError))
 
        data = _assert_all_types(data, pd.DataFrame )
  
        # sanitize the dataframe 

        pObj =P() ; ncols = pObj(hl = list(data.columns), kind ='ves')
        if ncols is None:
            raise HeaderError (f"Columns {smft(pObj.icpr)} are missing in "
                               "the given dataset.")
        data.columns = ncols 
        try : 
            rhoa= data.resistivity 
        except : 
            raise ResistivityError(
                "Data validation aborted! Missing resistivity values.")
        else : 
            # In the case, we got a multiple resistivity values 
            # corresponding to the different sounding values 
            if rhoa.ndim > 1 :
                if index_rhoa is None: 
                    index_rhoa = 0 
                elif (index_rhoa  >= len(rhoa.columns)) or index_rhoa <0 : 
                    warnings.warn(f'The index `{index_rhoa}` is out of the range' 
                                  f'`{len(rhoa.columns)-1}` for selecting the'
                                  ' specific resistivity data. By default, we '
                                  'only keep the data at the index 0.'
                        )
                    index_rhoa= 0 
                    
            rhoa = rhoa.iloc[:, index_rhoa] if rhoa.ndim > 1 else rhoa 
            
        if 'MN' in data.columns: 
            MN = data.MN 
        try: 
            AB= data.AB 
        except: 
            raise VESError("Data validation aborted! Current electrodes values"
                " are missing. Specify the deep measurement!")
            
    if rhoa is None: 
        raise ResistivityError(
            "Data validation aborted! Missing resistivity values.")
    if AB is None: 
        raise VESError("Data validation aborted! Current electrodes values"
            " are missing. Specify the deep measurement!")

    AB = np.array(AB) ; MN = np.array(MN) ; rhoa = np.array(rhoa) 
    
    if len(AB) !=len(rhoa): 
        raise VESError(" Deep measurement from the current electrodes `AB` and"
                       " the resistiviy values `rhoa` must have the same length"
                       f'. But `{len(AB)}` and `{len(rhoa)}` were given.')
        
    sdata =pd.DataFrame(
        {'AB': AB, 'MN': MN, 'resistivity':rhoa},index =range(len(AB)))
    
    return sdata
 
@docSanitizer()
def fill_coordinates(
    data: DataFrame =None, 
    lon: ArrayLike = None,
    lat: ArrayLike = None,
    east: ArrayLike = None,
    north: ArrayLike = None, 
    epsg: Optional[int] = None , 
    utm_zone: Optional [str]  = None,
    datum: str  = 'WGS84', 
    verbose:int =0, 
) -> Tuple [DataFrame, str] : 
    """ Recompute coordinates values  
    
    Compute the couples (easting, northing) or (longitude, latitude ) 
    and set the new calculated values into a dataframe.
    
    Parameters 
    -----------
    
    data : dataframe, 
        Dataframe contains the `lat`, `lon` or `east` and `north`. All data 
        don't need to  be provided. If ('lat', 'lon') and (`east`, `north`) 
        are given, ('`easting`, `northing`') should be overwritten.
        
    lat: array-like float or string (DD:MM:SS.ms)
        Values composing the `longitude`  of point

    lon: array-like float or string (DD:MM:SS.ms)
        Values composing the `longitude`  of point
              
    east : array-like float
        Values composing the northing coordinate in meters
                 
    north : array-like float
        Values composing the northing coordinate in meters

    datum: string
        well known datum ex. WGS84, NAD27, etc.
                
    projection: string
        projected point in lat and lon in Datum `latlon`, as decimal degrees 
        or 'UTM'.
                
    epsg: int
        epsg number defining projection (see http://spatialreference.org/ref/ 
        for moreinfo). Overrides utm_zone if both are provided
        
    utm_zone : string
            zone number and 'S' or 'N' e.g. '55S'. Defaults to the
            centre point of the provided points
                    
    Returns 
    ------- 
        - `data`: Dataframe with new coodinates values computed 
        - `utm_zone`: zone number and 'S' or 'N'  
        
        
    """
    def _get_coordcomps (str_, df):
        """ Retrieve coordinate values and assert whether values are given. 
        If ``True``, returns `array` of `given item` and valid type of the 
        data. Note that if data equals to ``0``, we assume values are not 
        provided. 
        
        :param str_: str - item in the `df` columns 
        :param df: DataFrame - dataframe expected containing the `str_` item. 
        """
        
        if str_ in df.columns: 
            return df[str_] , np.all(df[str_])!=0 
        return None, None 
    
    def _set_coordinate_values (x, y, *, func ): 
        """ Iterate `x` and `y` and output new coordinates values computed 
        from `func` . 
        param x: iterable values 
        :param y: iterabel values 
        :param func: function F 
            can be: 
                - ``project_point_utm2ll`` for `UTM` to `latlon`` or 
                - `` project_point_ll2utm`` for `latlon`` to `UTM` 
        :retuns: 
            - xx new calculated 
            - yy new calculated 
            - utm zone 
        """
        xx = np.zeros_like(x); 
        yy = np.zeros_like(xx)
        for ii, (la, lo) in enumerate (zip(x, y)):
            e , n, uz  = func (
                la, lo, utm_zone = utm_zone, datum = datum, epsg =epsg 
                ) 
            xx [ii] = e ; yy[ii] = n  
                
        return xx, yy , uz  
    
    if data is None:  
        data = pd.DataFrame (
            dict ( 
                longitude = lon ,
                latitude = lat ,
                easting = east,
                northing=north
                ), 
            #pass index If using all scalar values 
            index = range(4)  
            )

    if data is not None : 
        data = _assert_all_types(data, pd.DataFrame)

    lon , lon_isvalid  = _get_coordcomps(
        'longitude', data )
    lat , lat_isvalid = _get_coordcomps(
        'latitude', data )
    east , e_isvalid = _get_coordcomps(
        'easting', data )
    north, n_isvalid  = _get_coordcomps(
        'northing', data )

    if lon_isvalid and lat_isvalid: 
        try : 
            east , north , uz = _set_coordinate_values(
                lat.values, lon.values,
                project_point_ll2utm,
                )
        except :# pass if an error occurs 
            pass 
        else : data['easting'] = east ; data['northing'] = north 
            
    elif e_isvalid and n_isvalid: 
        if utm_zone is None: 
            if verbose > 0: 
                warnings.warn(
                    'Should provide the `UTM` for `latitute` and `longitude`'
                    ' calculus. `NoneType` can not be used as UTM zone number.'
                    ' Refer to the documentation.')
        try : 
            lat , lon, utm_zone = _set_coordinate_values(
                east.values, north.values,
                func = project_point_utm2ll,
                )
        except : pass 
        else : data['longitude'] = lon ;  data['latitude'] = lat 
        
    
    return data, utm_zone 

    
def _assert_data (data :DataFrame  ): 
    """ Assert  the data and return the property dataframe """
    data = _assert_all_types(
        data, list, tuple, np.ndarray, pd.Series, pd.DataFrame) 
    
    if isinstance(data, pd.DataFrame): 
        cold , ixc =list(), list()
        for i , ckey in enumerate(data.columns): 
            for kp in P().isrll : 
                if ckey.lower() .find(kp) >=0 : 
                    cold.append (kp); ixc.append(i)
                    break 
                    
        if len (cold) ==0: 
            raise ValueError (f'Expected {smft(P().isrll)} '
                ' columns, but not found in the given dataframe.'
                )
                
        dup = cold.copy() 
        # filter and remove one by one duplicate columns.
        list(filter (lambda x: dup.remove(x), set(cold)))
        dup = set(dup)
        if len(dup) !=0 :
            raise HeaderError(
                f'Duplicate column{"s" if len(dup)>1 else ""}'
                f' {smft(dup)} found. It seems to be {smft(dup)}'
                f'column{"s" if len(dup)>1 else ""}. Please provide'
                '  the right column name in the dataset.'
                )
        data_ = data [cold] 
  
        col = list(data_.columns)
        for i, vc in enumerate (col): 
            for k in P().isrll : 
                if vc.lower().find(k) >=0 : 
                    col[i] = k ; break 
                
    return data_
 
def is_erp_series (
        data : Series ,
        dipolelength : Optional [float] = None 
        ) -> DataFrame : 
    """ Validate the series.  
    
    The `data` should be the resistivity values with the one of the following 
    property index names ``resistivity`` or ``rho``. Will raises error 
    if not detected. If a`dipolelength` is given, a data should include 
    each station positions values. 
    
    Parameters 
    -----------
    
    data : pandas Series object 
        Object of resistivity values 
    
    dipolelength: float
        Distance of dipole during the whole survey line. If it is
        is not given , the station location should be computed and
        filled using the default value of the dipole. The *default* 
        value is set to ``10 meters``. 
        
    Returns 
    --------
    A dataframe of the property indexes such as
    ``['station', 'easting','northing', 'resistivity']``. 
    
    Raises 
    ------ 
    ResistivityError
    If name does not match the `resistivity` column name. 
    
    Examples 
    --------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from watex.utils.coreutils imprt is_erp_series 
    >>> data = pd.Series (np.abs (np.random.rand (42)), name ='res') 
    >>> data = is_erp_series (data)
    >>> data.columns 
    ... Index(['station', 'easting', 'northing', 'resistivity'], dtype='object')
    >>> data = pd.Series (np.abs (np.random.rand (42)), name ='NAN') 
    >>> data = _is_erp_series (data)
    ... ResistivityError: Unable to detect the resistivity column: 'NAN'.
    
    """
    
    data = _assert_all_types(data, pd.Series) 
    is_valid = False 
    for p in P().iresistivity : 
        if data.name.lower().find(p) >=0 :
            data.name = p ; is_valid = True ; break 
    
    if not is_valid : 
        raise ResistivityError(
            f"Unable to detect the resistivity column: {data.name!r}."
            )
    
    if is_valid: 
        df = is_erp_dataframe  (pd.DataFrame (
            {
                data.name : data , 
                'NAN' : np.zeros_like(data ) 
                }
            ),
                dipolelength = dipolelength,
            )
    return df 

def is_erp_dataframe (
        data :DataFrame ,
        dipolelength : Optional[float] = None 
        ) -> DataFrame:
    """ Ckeck whether the dataframe contains the electrical resistivity 
    profiling (ERP) index properties. 
    
    DataFrame should be reordered to fit the order of index properties. 
    Anyway it should he dataframe filled by ``0.`` where the property is
    missing. However, if `station` property is not given. station` property 
    should be set by using the dipolelength default value equals to ``10.``.
    
    Parameters 
    ----------
    
    data : Dataframe object 
        Dataframe object. The columns dataframe should match the property 
        ERP property object such as ``['station','resistivity', 
                                       'longitude','latitude']`` 
        or ``['station','resistivity', 'easting','northing']``.
            
    dipolelength: float
        Distance of dipole during the whole survey line. If the station 
        is not given as  `data` columns, the station location should be 
        computed and filled the station columns using the default value 
        of the dipole. The *default* value is set to ``10 meters``. 
        
    Returns
    --------
    A new data with index properties.
        
    Raises 
    ------
    - None of the column matches the property indexes.  
    - Find duplicated values in the given data header.
    
    Examples
    --------
    >>> import numpy as np 
    >>> from watex.utils.coreutils import is_erp_dataframe 
    >>> df = pd.read_csv ('data/erp/testunsafedata.csv')
    >>> df.columns 
    ... Index(['x', 'stations', 'resapprho', 'NORTH'], dtype='object')
    >>> df = _is_erp_dataframe (df) 
    >>> df.columns 
    ... Index(['station', 'easting', 'northing', 'resistivity'], dtype='object')
    
    """
    
    data = _assert_all_types(data, pd.DataFrame)
    datac= data.copy() 
    
    def _is_in_properties (h ):
        """ check whether the item header `h` is in the property values. 
        Return `h` and it correspondence `key` in the property values. """
        for key, values in P().idicttags.items() : 
            for v in values : 
                if h.lower().find (v)>=0 :
                    return h, key 
        return None, None 
    
    def _check_correspondence (pl, dl): 
        """ collect the duplicated name in the data columns """
        return [ l for l in pl for d  in dl if d.lower().find(l)>=0 ]
        
    cold , c = list(), list()
    # create property object
    pObj = P(data.columns)
    for i , ckey in enumerate(list(datac.columns)): 
        h , k = _is_in_properties(ckey)
        cold.append (h) if h is not None  else h 
        c.append(k) if k is not None else k
        
    if len (cold) ==0: 
        raise HeaderError (
            f'Wrong column headers {list(data.columns)}.'
            f' Unable to find the expected {smft(pObj.isrll)}'
            ' column properties.'
                           )

    dup = cold.copy() 
    # filter and remove one by one duplicate columns.
    list(filter (lambda x: dup.remove(x), set(cold)))

    dup = set(dup) ; ress = _check_correspondence(
        pObj() or pObj.idicttags.keys(), dup)
    
    if len(dup) !=0 :
        raise HeaderError(
            f'Duplicate column{"s" if len(dup)>1 else ""}' 
            f' {smft(dup)} {"are" if len(dup)>1 else "is"} '
            f'found. It seems correspond to {smft(ress)}. '
            'Please ckeck your data column names. '
            )
            
    # fetch the property column names and 
    # replace by 0. the non existence column
    # reorder the column to match 
    # ['station','resistivity', 'easting','northing', ]
    
    data_ = data[cold] 
    data_.columns = c  
    
    msg = ERPError("Unknown DC-ERP data. ERP data must contain"
                   f" {smft(pObj.idicttags.keys())}")
    try : 
        data_= data_.reindex (columns =pObj.idicttags.keys(), fill_value =0.
                              ) 
    except : 
        raise msg 
        
    dipolelength = _assert_all_types(
        dipolelength , float, int) if dipolelength is not None else None 
    
    if (np.all (data_.station) ==0. 
        and dipolelength is None 
        ): 
        dipolelength = 10.
        data_.station = np.arange (
            0 , data_.shape[0] * dipolelength  , dipolelength ) 
        
    return data_


def erpSelector (
        f: str | NDArray | Series | DataFrame ,
        columns: str | List[str] = ..., 
        **kws:Any 
) -> DataFrame  : 
    """ Read and sanitize the data collected from the survey. 
    
    `data` should be an array, a dataframe, series, or  arranged in ``.csv`` 
    or ``.xlsx`` formats. Be sure to provide the header of each columns in'
    the worksheet. In a file is given, header columns should be aranged as  
    ``['station','resistivity' ,'longitude', 'latitude']``. Note that 
    coordinates columns (`longitude` and `latitude`) are not  compulsory. 
    
    Parameters 
    ----------
    
    f: Path-like object, ndarray, Series or Dataframe, 
        If a path-like object is given, can only parse `.csv` and `.xlsx` 
        file formats. However, if ndarray is given and shape along axis 1 
        is greater than 4, the ndarray should be shrunked. 
        
    columns: list 
        list of the valuable columns. It can be used to fix along the axis 1 
        of the array the specific values. It should contain the prefix or 
        the whole name of each item in 
        ``['station','resistivity' ,'longitude', 'latitude']``.
        
    kws: dict
        Additional pandas `pd.read_csv` and `pd.read_excel` 
        methods keyword arguments. Be sure to provide the right argument. 
        when reading `f`. For instance, provide ``sep= ','`` argument when 
        the file to read is ``xlsx`` format will raise an error. Indeed, 
        `sep` parameter is acceptable for parsing the `.csv` file format
        only.
        
         
    Returns 
    -------
    DataFrame with valuable column(s). 
    
    Notes
    ------
    The length of acceptable columns is ``4``. If the size of the columns is 
    higher than `4`, the data should be shrunked to match the expected columns.
    Futhermore, if the header is not specified in `f` , the defaut column
    arrangement should be used. Therefore, the second column should be 
    considered as the ``resistivity`` column. 
     
    Examples
    ---------
    >>> import numpy as np 
    >>> from watex.utils.coreutils import erpSelector
    >>> df = erpSelector ('data/erp/testsafedata.csv')
    >>> df.shape 
    ... (45, 4)
    >>> list(df.columns) 
    ... ['station','resistivity', 'longitude', 'latitude']
    >>> df = erp_selector('data/erp/testunsafedata.xlsx') 
    >>> list(df.columns)
    ... ['easting', 'station', 'resistivity', 'northing']
    >>> df = erpSelector(np.random.randn(7, 7)) 
    >>> df.shape 
    ... (7, 4)
    >>> list(df.columns) 
    ... ['station', 'resistivity', 'longitude', 'latitude']
    
    """
    
    if columns is ...: columns=None 
    if columns is not None: 
        if isinstance(columns, str):
            columns =columns.replace(':', ',').replace(';', ',')
            if ',' in columns: columns =columns.split(',')
            
    if isinstance(f, (str,  pathlib.PurePath)):
        try : 
            f = _is_readable(f, **kws)
        except TypeError as typError: 
            raise ERPError (str(typError))
            
    if isinstance( f, np.ndarray): 
        name = copy.deepcopy(columns)
        columns = P().isrll if columns is None else columns 
        colnum = 1 if f.ndim ==1 else f.shape[1]
     
        if colnum==1: 
            if isinstance (name, list) : 
                if len(name) ==1: name = name[0]
            f = is_erp_series (
                pd.Series (f, name = name or columns[1] 
                           )
                ) 
    
        elif colnum==2 : 
            f= pd.DataFrame (f, columns = columns
                             if columns is None  
                             else columns[:2]
                             ) 
      
        elif colnum==3: 
            warnings.warn("One missing column `longitude|latitude` value."
                          "If the `longitude` and `latitude` data are"
                          f" not available. Use {smft(P().isrll[:2])} "
                          "columns instead.", UserWarning)
            columns = name or columns [:colnum]
            f= pd.DataFrame (f[:, :len(columns)],
                              columns =columns )

        elif f.shape[1]==4:
            f =pd.DataFrame (f, columns =columns 
                )
        elif colnum > 4: 
            # add 'none' columns for the remaining columns.
                f =pd.DataFrame (
                    f, columns = columns  + [
                        'none' for i in range(colnum-4)]
                    )
                
    if isinstance(f, pd.DataFrame): 
        f = is_erp_dataframe( f)
    elif isinstance(f , pd.Series ): 
        f = is_erp_series(f)
    else : 
        amsg = smft(accept_types (
            pd.Series, pd.DataFrame, np.ndarray) + ['*.xls', '*.csv'])
        raise ValueError (f" Unsupports data. Expects only {amsg}."
                          )  
    if np.all(f.resistivity)==0: 
        raise ResistivityError('Resistivity values need to be supply.')

    return f 

def _fetch_prefix_index (
    arr:NDArray [DType[float]] = None,
    col: List[str]  = None,
    df : DataFrame = None, 
    prefixs: List [str ]  =None
) -> Tuple [int | int]: 
    """ Retrieve index at specific column. 
    
    Use the given station positions collected on the field to 
    compute the dipole length during the whole survey. 
    
    :param arr: array. Ndarray of data where one colum must the 
            positions values. 
    :param col: list. The list should be considered as the head of array. Each 
        position in the list sould fit the column data in the array. It raises 
        an error if the number of item in the list is different to the size 
        of array in axis=1. 
    :param df: dataframe. When supply, the `arr` and `col` is not 
        compulsory. 
        
    :param prefixs: list. Contains specific column prefixs to 
        fetch the corresponding data. For instance::
            
            - Station prefix : ['pk','sta','pos']
            - Easting prefix : ['east', 'x', 'long'] 
            - Northing prefix: ['north', 'y', 'lat']
   :returns: 
       - index of the position columns in the data 
       - station position array-like. 
       
    :Example: 
        >>> from numpy as np 
        >>> from watex.utils.coreutils import _assert_positions
        >>> array1 = np.c_[np.arange(0, 70, 10), np.random.randn (7,3)]
        >>> col = ['pk', 'x', 'y', 'rho']
        >>> index, = _fetch_prefix_index (array1 , col = ['pk', 'x', 'y', 'rho'], 
        ...                         prefixs = EASTPREFIX)
        ... 1
        >>> index, _fetch_prefix_index (array1 , col = ['pk', 'x', 'y', 'rho'], 
        ...                         prefixs = NOTHPREFIX )
        ... 2
    """
    if prefixs is None: 
        raise ValueError('Please specify the list of items to compose the '
                         'prefix to fetch the columns data. For instance'
                         f' `station prefix` can  be `{P().istation}`.')

    if arr is None and df is None :
        raise TypeError ( 'Expected and array or a dataframe not'
                         ' a Nonetype object.'
                        )
    elif df is None and col is None: 
        raise StationError( 'Column list is missing.'
                         ' Could not detect the position index.') 
        
    if isinstance( df, pd.DataFrame): 
        # collect the resistivity from the index 
        # if a dataFrame is given 
        arr, col = df.values, df.columns 

    if arr.ndim ==1 : 
        # Here return 0 as colIndex
        return  0, arr 
    if isinstance(col, str): col =[col] 
    if len(col) != arr.shape[1]: 
        raise ValueError (
            f'Column should match the array shape in axis =1 <{arr.shape[1]}>.'
            f' But {"was" if len(col)==1 else "were"} given')
        
    # convert item in column in lowercase 
    comsg = col.copy()
    col = list(map(lambda x: x.lower(), col)) 
    colIndex = [col.index (item) for item in col 
             for pp in prefixs if item.find(pp) >=0]   

    if len(colIndex) is None or len(colIndex) ==0: 
        raise ValueError (f'Unable to detect the position in `{smft(comsg)}`'
                          ' columns. Columns must contain at least'
                          f' `{smft(prefixs)}`.')
 
    return colIndex[0], arr 


def _assert_station_positions(
    arr: SP = None,
    prefixs: List [str] =...,
    **kws
) -> Tuple [int, float]: 
    """ Assert positions and compute dipole length. 
    
    Use the given station positions collected on the field to 
    detect the dipole length during the whole survey. 
    
    :param arr: array. Ndarray of data where one column must the 
            positions values. 
    :param col: list. The list should be considered as the head of array. Each 
        position in the list sould fit the column data in the array. It raises 
        an error if the number of item in the list is different to the size 
        of array in axis=1. 
    :param df: dataframe. When supply, the `arr` and `col` are not needed.

    :param prefixs: list. Contains all the station column names prefixs to 
        fetch the corresponding data.
    :returns: 
        - positions: new positions numbering from station `S00` to ...    
        - dipolelength:  recomputed dipole value
    :Example: 
        
        >>> from numpy as np 
        >>> from watex.utils.coreutils import _assert_station_positions
        >>> array1 = np.c_[np.arange(0, 70, 10), np.random.randn (7,3)]
        >>> col = ['pk', 'x', 'y', 'rho']
        >>> _assert_positions(array1, col)
        ... (array([ 0, 10, 20, 30, 40, 50, 60]), 10)
        >>> array1 = np.c_[np.arange(30, 240, 30), np.random.randn (7,3)]
        ... (array([  0,  30,  60,  90, 120, 150, 180]), 30)
    
    """
    if prefixs is (None or ...): prefixs = P().istation 
    
    colIndex, arr =_fetch_prefix_index( arr=arr, prefixs = prefixs, **kws )
    positions = arr[:, colIndex]
    # assert the position is aranged from lower to higher 
    # if there is not wrong numbering. 
    fsta = np.argmin(positions) 
    lsta = np.argmax (positions)
    if int(fsta) !=0 or int(lsta) != len(positions)-1: 
        raise StationError(
            'Wrong numbering! Please number the position from first station '
            'to the last station. Check your array positionning numbers.')
    
    dipoleLength = int(np.abs (positions.min() - positions.max ()
                           ) / (len(positions)-1)) 
    # renamed positions  
    positions = np.arange(0 , len(positions) *dipoleLength ,
                          dipoleLength ) 
    
    return  positions, dipoleLength 

@refAppender(refglossary.__doc__)
def plotAnomaly(
    erp: ArrayLike | List[float],
    cz: Optional [Sub[ArrayLike], List[float]] = None, 
    s: Optional [str] = None, 
    figsize: Tuple [int, int] = (10, 4),
    fig_dpi: int = 300 ,
    savefig: str | None = None, 
    show_fig_title: bool = True,
    style: str = 'seaborn', 
    fig_title_kws: Dict[str, str|Any] = ...,
    czkws: Dict [str , str|Any] = ... , 
    legkws: Dict [Any , str|Any] = ... , 
    **kws, 
) -> None: 

    """ Plot the whole |ERP| line and selected conductive zone. 
    
    Conductive zone can be supplied nannualy as a subset of the `erp` or by 
    specifying the station expected for drilling location. For instance 
    ``S07`` for the seventh station. Futhermore, for automatic detection, one 
    should set the station argument `s` to ``auto``. However, it 's recommended 
    to provide the `cz` or the `s` to have full control. The conductive zone 
    is juxtaposed to the whole |ERP| survey. One can customize the `cz` plot by 
    filling with `Matplotlib pyplot`_ additional keywords araguments thought 
    the kewords argument `czkws`. 

    :param sample: array_like - the |ERP| survey line. The line is an array of
        resistivity values.  
        
    :param cz: array_like - the selected conductive zone. If ``None``, only 
        the `erp` should be displayed. Note that `cz` is an subset of `erp` 
        array. 
        
    :param s: str - The station location given as string (e.g. ``s= "S10"``) 
        or as a station number (indexing; e.g ``s =10``). If value is set to 
        ``"auto"``, `s` should be find automatically and fetching `cz` as well. 
        
    :param figsize: tuple- Tuple value of figure size. Refer to the 
        web resources `Matplotlib figure`_. 
        
    :param fig_dpi: int - figure resolution "dot per inch". Refer to 
            `Matplotlib figure`_.
        
    :param savefig: str -  save figure. Refer  to `Matplotlib figure`_.
    
    :param show_fig_tile: bool - display the title of the figure. 
    
    :param fig_title_kws: dict - Keywords arguments of figure suptile. Refer to 
        `Matplotlib figsuptitle`_.
        
    :param style: str - the style for customizing visualization. For instance to 
        get the first seven available styles in pyplot, one can run 
        the script below:: 
        
            plt.style.available[:7]
        Futher details can be foud in Webresources below or click on 
        `GeekforGeeks`_. 
        
    :param czkws: dict - keywords `Matplotlib pyplot`_ additional arguments to 
        customize the `cz` plot.
        
    :param legkws: dict - keywords Matplotlib legend additional keywords
        arguments. 
        
    :param kws: dict - additional keywords argument for `Matplotlib pyplot`_ to 
        customize the `erp` plot.
        
   
    :Example: 
        >>> import numpy as np 
        >>> from watex.utils.coreutils import ( 
        ...    plot_anomaly, _define_conductive_zone)
        >>> test_array = np.random.randn (10)
        >>> selected_cz ,*_ = _define_conductive_zone(test_array, 7) 
        >>> plot_anomaly(test_array, selected_cz )
        >>> plot_anomaly(tes_array, selected_cz , s= 5)
        >>> plot_anomaly(tes_array, s= 's02')
        >>> plot_anomaly(tes_array)
        
    .. note::
        
        If `cz` is given, one does not need to worry about the station `s`. 
        `s` can stay with it default value``None``. 
        
     
    References   
    -----------
    See Matplotlib Axes: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html
    GeekforGeeks: https://www.geeksforgeeks.org/style-plots-using-matplotlib/#:~:text=Matplotlib%20is%20the%20most%20popular,without%20using%20any%20other%20GUIs.
    
    """
    
    def format_ticks (value, tick_number):
        """ Format thick parameter with 'FuncFormatter(func)'
        rather than using:: 
            
        axi.xaxis.set_major_locator (plt.MaxNLocator(3))
        
        ax.xaxis.set_major_formatter (plt.FuncFormatter(format_thicks))
        """
        if value % 7 ==0: 
            return 'S{:02}'.format(int(value)+ 1)
        else: None 
        
    
    erp = _assert_all_types( 
        erp, tuple, list , np.ndarray , pd.Series)
    if cz is not None: 
        cz = _assert_all_types(
            cz, tuple, list , np.ndarray , pd.Series)
        cz = np.array (cz)
        
    erp =np.array (erp) 
    
    plt.style.use (style)

    kws =dict (
        color=P().frcolortags.get('fr1') if kws.get(
            'color') is None else kws.get('color'), 
        linestyle='-' if kws.get('ls') is None else kws.get('ls'),
        linewidth=2. if kws.get('lw') is None else kws.get('lw'),
        label = 'Electrical resistivity profiling' if kws.get(
            'label') is None else kws.get('label')
                  )

    if czkws is ( None or ...) :
        czkws =dict (color=P().frcolortags.get('fr3'), 
                      linestyle='-',
                      linewidth=3,
                      label = 'Conductive zone'
                      )
    
    if czkws.get('color') is None: 
        czkws['color']= P().frcolortags.get(czkws['color'])
      
    if (xlabel := kws.get('xlabel')) is not None : 
        del kws['xlabel']
    if (ylabel := kws.get('ylabel')) is not None : 
        del kws['ylabel']
        
    if (rotate:= kws.get ('rotate')) is not None: 
        del kws ['rotate']
        
    fig, ax = plt.subplots(1,1, figsize =figsize)
    
    leg =[]

    zl, = ax.plot(np.arange(len(erp)), erp, 
                  **kws 
                  )
    leg.append(zl)
    
    if s =='' : s= None  # for consistency 
    if s is not None:
        auto =False ; keepindex =True 
        if isinstance (s , str): 
            auto = True if s.lower()=='auto' else s 
            if 's' or 'pk' in s.upper(): 
                # if provide the station. 
                keepindex =False 
                
            if s.lower() =='auto': s=None  # reset s 
        cz , _ , _, ix = defineConductiveZone(
           erp, s = s , auto = auto, keepindex=keepindex 
           )
        
        s = "S{:02}".format(ix +1) if s is not None else s 

    if cz is not None: 
        # construct a mask array with np.isin to check whether
        if not _isin (erp, cz ): 
            raise ValueError ('Expected a conductive zone to be a subset of '
                              ' the resistivity profiling line.')
        # `cz` is subset array
        z = np.ma.masked_values (erp, np.isin(erp, cz ))
        # a masked value is constructed so we need 
        # to get the attribute fill_value as a mask 
        # However, we need to use np.invert or the tilde operator  
        # to specify that other value except the `CZ` values mus be 
        # masked. Note that the dtype must be changed to boolean
        sample_masked = np.ma.array(
            erp, mask = ~z.fill_value.astype('bool') )

        czl, = ax.plot(
            np.arange(len(erp)), sample_masked, 'o',
            **czkws)
        leg.append(czl)
        
        
    ax.tick_params (labelrotation = 0. if rotate is None else rotate)
    ax.set_xticks(range(len(erp)),
                  )
    if len(erp ) >= 14 : 
        ax.xaxis.set_major_formatter (plt.FuncFormatter(format_ticks))
    else : 
        
        ax.set_xticklabels(
            ['S{:02}'.format(int(i)+1) for i in range(len(erp))],
            rotation =0. if rotate is None else rotate ) 
   

    if legkws is( None or ...): 
        legkws =dict() 
    
    ax.set_xlabel ('Stations') if xlabel is  None  else ax.set_xlabel (xlabel)
    ax.set_ylabel ('Resistivity (Ω.m)'
                ) if ylabel is None else ax.set_ylabel (ylabel)

    ax.legend( handles = leg, 
              **legkws )
    

    if show_fig_title: 
        title = 'Plot ERP line with SVES = {0}'.format(s if s is not None else '')
        if fig_title_kws is ( None or ...): 
            fig_title_kws = dict (
                t = title if s is not None else title.replace (
                    'with SVES =', ''), 
                style ='italic', 
                bbox =dict(boxstyle='round',facecolor ='lightgrey'))
            
        plt.tight_layout()
        fig.suptitle(**fig_title_kws, 
                      )
    if savefig is not None :
        plt.savefig(savefig,
                    dpi=fig_dpi,
                    )
        
    plt.show()
        
#XXX OPTIMIZE 
def defineConductiveZone(
    erp:ArrayLike| pd.Series | List[float] ,
    s: Optional [str|int] = None, 
    p: SP = None,  
    auto: bool = False, 
    **kws,
) -> Tuple [ArrayLike, int] :
    """ Define conductive zone as subset of the erp line.
    
    Indeed the conductive zone is a specific zone expected to hold the 
    drilling location `s`. If drilling location is not provided, it would be 
    by default the very low resistivity values found in the `erp` line. 
    
    Parameters 
    -----------
    erp : array_like,
        the array contains the apparent resistivity values 
    s: str or int, 
        is the station position. 
    auto: bool
        If ``True``, the station position should be the position of the lower 
        resistivity value in |ERP|. 
    
    Returns 
    -------- 
        - conductive zone of resistivity values 
        - conductive zone positionning 
        - station position index in the conductive zone
        - station position index in the whole |ERP| line 
    
    :Example: 
        >>> import numpy as np 
        >>> from watex.utils.coreutils import defineConductiveZone
        >>> test_array = np.random.randn (10)
        >>> selected_cz ,*_ = defineConductiveZone(test_array, 's20') 
        >>> shortPlot(test_array, selected_cz )
    """
    if isinstance(erp, pd.DataFrame): 
        try: erp = erp.resistivity  
        except AttributeError: 
            raise ResistivityError (" Resistivity data is missing ")
            
    if isinstance(erp, pd.Series):
        erp = erp.values 
    
    # conductive zone positioning
    pcz : Optional [ArrayLike]  = None  
    
    if s is None and auto is False: 
        raise StationError("Expect a station position or trigger the 'auto'"
                        "to 'True'. NoneType is given.")
        
    elif  ( s is None 
           and auto is True 
           ): 
        s= np.argwhere (erp ==erp.min())
        s= int(s) if len(s) ==1 else int(s[0])
        # s, = np.where (erp == erp.min()) 
        # s=int(s)
    s, pos = _assert_stations(s, **kws )
    
    # takes the last position if the position is outside 
    # the number of stations. 
    pos = len(erp) -1  if pos >= len(erp) else pos 
    # frame the `sves` (drilling position) and define the conductive zone 
    ir = erp[:pos][-3:] ;  il = erp[pos:pos +3 +1 ]
    cz = np.concatenate((ir, il))

    if p is not None: 
        if len(p) != len(erp): 
            raise StationError (
                'Array of position and conductive zone must have the same '
                f'length: `{len(p)}` and `{len(cz)}` were given.')
            
        sr = p[:pos][-3:] ;  sl = p[pos:pos +3 +1 ]
        pcz = np.concatenate((sr, sl))
        
    # Get the new position in the selected conductive zone 
    # from the of the whole erp 
    pix= np.argwhere (cz == erp[pos])
    pix = pix [0] if len(pix) > 1 else pix 

    return cz , pcz, int(pix), pos

#XXX OPTIMIZE 
def _assert_stations(
    s:Any , 
    dipole:Any = None,
    keepindex:bool = False
) -> Tuple[str, int]:
    """ Sanitize stations and returns station name and index.
    
    ``pk`` and ``S`` can be used as prefix to define the station `s`. For 
    instance ``S01`` and ``PK01`` means the first station. 
    
    :param s: Station name
    :type s: str, int 
    
    :param dipole: dipole_length in meters.  
    :type dipole: float 
    
    :param keepindex: bool - Stands for keeping the Python indexing. If set to 
        ``True`` so the station should start by `S00` and so on. 
    
    :returns: 
        - station name 
        - index of the station.
        
    .. note:: 
        
        The defaut station numbering is from 1. SO if ``S00` is given, and 
        the argument `keepindex` is still on its default value i.e ``False``,
        the station name should be set to ``S01``. Moreover, if `dipole`
        value is given, the station should  named according to the 
        value of the dipole. For instance for `dipole` equals to ``10m``, 
        the first station should be ``S00``, the second ``S10`` , 
        the third ``S30`` and so on. However, it is recommend to name the 
        station using counting numbers rather than using the dipole 
        position.
            
    :Example: 
        >>> from watex.utils.coreutils import _assert_stations
        >>> _assert_stations('pk01')
        ... ('S01', 0)
        >>> _assert_stations('S1')
        ... ('S01', 0)
        >>> _assert_stations('S1', keepindex =True)
        ... ('S01', 1) # station here starts from 0 i.e `S00` 
        >>> _assert_stations('S00')
        ... ('S00', 0)
        >>> _assert_stations('S1000',dipole ='1km')
        ... ('S02', 1) # by default it does not keep the Python indexing 
        >>> _assert_stations('S10', dipole ='10m')
        ... ('S02', 1)
        >>> _assert_stations(1000,dipole =1000)
        ... ('S02', 1)
    """
    # in the case s is string: eg. "00", "pk01", "S001"
    ix = 0
    stnl =P().istation 
    s = _assert_all_types(s, str, int, float)

    s = str(s).strip() 
    regex = re.compile (r'\d+', flags= re.IGNORECASE)
    s = regex.findall (s)
    if len(s)==0: 
        raise StationError (f"Wrong station name {s!r}. Station must be "
                            f"prefixed by {smft(stnl +['S'], 'or')} e.g. "
                            "'S00' for the first station")
    else : s = int(s[0])
    
    if s ==0 : 
        # set index to 0 , is station `S00` is found for instance.
        keepindex =True 

    st = copy.deepcopy(s)
    
    if isinstance(s, int):  
        msg = 'Station numbering must start'\
            ' from {0!r} or set `keepindex` argument to {1!r}.'
        msg = msg.format('0', 'False') if keepindex else msg.format(
            '1', 'True')
        if not keepindex: # station starts from 1
            if s <=0: 
                raise ValueError (msg )
            s , ix  = "S{:02}".format(s), s - 1
        
        elif keepindex: 
            
            if s < 0: raise ValueError (msg) # for consistency
            s, ix =  "S{:02}".format(s ), s  
    # Recompute the station position if the dipole value are given
    if dipole is not None: 
        if isinstance(dipole, str): #'10m'
            if dipole.find('km')>=0: 
           
                dipole = dipole.lower().replace('km', '000') 
                
            dipole = dipole.lower().replace('m', '')
            try : 
                dipole = float(dipole) 
            except : 
                raise StationError(f'Invalid literal value for dipole: {dipole!r}')
        # since the renamed from dipole starts at 0 
        # e.g. 0(S1)---10(S2)---20(S3) ---30(S4)etc ..
        ix = int(st//dipole)  ; s= "S{:02}".format(ix +1)
    
    return s, ix 

def _parse_args (
    args:Union[List | str ]
)-> Tuple [ pd.DataFrame, List[str|Any]]: 
    """ `Parse_args` function returns array of rho and coordinates 
    values (X, Y).
    
    Arguments can be a list of data, a dataframe or a Path like object. If 
    a Path-like object is set, it should be the priority of reading. 
    
    :param args: arguments 
    
    :return: ndarray or array-like  arranged with apparent 
        resistivity at the first index 
        
    .. note:: If a list of arrays is given or numpy.ndarray is given, 
            we assume that the columns at the first index fits the
            apparent resistivity values. 
            
    :Example: 
        >>> import numpy as np 
        >>> from watex.utils.coreutils import _parse_args
        >>> a, b = np.arange (1, 10 , 0.5), np.random.randn(9).reshape(3, 3)
        >>> _parse_args ([a, 'data/erp/l2_gbalo.xlsx', b])
        ... array([[1.1010000e+03, 0.0000000e+00, 7.9075200e+05, 1.0927500e+06],
                   [1.1470000e+03, 1.0000000e+01, 7.9074700e+05, 1.0927580e+06],
                   [1.3450000e+03, 2.0000000e+01, 7.9074300e+05, 1.0927630e+06],
                   [1.3690000e+03, 3.0000000e+01, 7.9073800e+05, 1.0927700e+06],
                   [1.4060000e+03, 4.0000000e+01, 7.9073300e+05, 1.0927765e+06],
                   [1.5430000e+03, 5.0000000e+01, 7.9072900e+05, 1.0927830e+06],
                   [1.4800000e+03, 6.0000000e+01, 7.9072400e+05, 1.0927895e+06],
                   [1.5170000e+03, 7.0000000e+01, 7.9072000e+05, 1.0927960e+06],
                   [1.7540000e+03, 8.0000000e+01, 7.9071500e+05, 1.0928025e+06],
                   [1.5910000e+03, 9.0000000e+01, 7.9071100e+05, 1.0928090e+06]])
    
    """
    
    keys= ['res', 'rho', 'app.res', 'appres', 'rhoa']
    
    col=None 
    if isinstance(args, list): 
        args, isfile  = _assert_file(args) # file to datafame 
        if not isfile:                     # list of values 
        # _assert _list of array_length 
            args = np.array(args, dtype =np.float64).T
            
    if isinstance(args, pd.DataFrame):
        # firt drop all untitled items 
        # if data is from xlsx sheets
        args.drop([ c for c in args.columns if c.find('untitle')>=0 ],
                  axis =1, inplace =True) 

        # get the index of items `resistivity`
        ixs = [ii for ii, name in enumerate(args.columns ) 
               for item in keys if name.lower().find(item)>=0]
        if len(set(ixs))==0: 
            raise ValueError(
                f"Column name `resistivity` not found in {list(args.columns)}"
                " Please provide the resistivity column.")
        elif len(set(ixs))>1: 
            raise ValueError (
                f"Expected 1 but got {len(ixs)} resistivity columns "
                f"{tuple([list(args.columns)[i] for i in ixs])}.")

        rc= args.pop(args.columns[ixs[0]]) 
        args.insert(0, 'app.res', rc)
        col =list(args.columns )  
        args = args.values

    if isinstance(args, pd.Series): 
        col =args.name 
        args = args.values

    return args, col

def _assert_file (
        args: List[str, Any]
)-> Tuple [List [str , pd.DataFrame] | Any , bool]: 
    """ Check whether the data is gathering into a Excel sheet workbook file.
    
    If the workbook is detected, will read the data and grab all into a 
    dataframe. 
    
    :param args: argument into a list 
    :returns: 
        - dataframe  
        - assert whether workbook was successful read. 
        
    :Example: 
        >>> import numpy as np 
        >>> from watex.utils.coreutils import  _assert_file
        >>> a, b = np.arange (1, 10 , 0.5), np.random.randn(9).reshape(3, 3)
        >>> data = [a, 'data/erp/l2_gbalo', b] # collection of 03 objects 
        >>>  # but read only the Path-Like object 
        >>> _assert_file([a, 'data/erp/l2_gbalo.xlsx', b])
        ... 
        ['l2_gbalo',
            pk       x          y   rho
         0   0  790752  1092750.0  1101
         1  10  790747  1092758.0  1147
         2  20  790743  1092763.0  1345
         3  30  790738  1092770.0  1369
         4  40  790733  1092776.5  1406
         5  50  790729  1092783.0  1543
         6  60  790724  1092789.5  1480
         7  70  790720  1092796.0  1517
         8  80  790715  1092802.5  1754
         9  90  790711  1092809.0  1591]
    """
    
    isfile =False 
    file = [ item for item in args if isinstance(item, str)
                    if os.path.isfile (item)]

    if len(file) > 1: 
        raise ValueError (
            f"Expected a single file but got {len(file)}. "
            "Please select the right file expected to contain the data.")
    if len(file) ==1 : 
        _, args = read_from_excelsheets(file[0])
        isfile =True 
        
    return args , isfile 
 

def makeCoords(
        reflong: str | Tuple[float], 
        reflat: str | Tuple[float], 
        nsites: int ,  
        *,  
        r: int =45.,
        utm_zone: Optional[str] =None,   
        step: Optional[str|float] ='1km', 
        order: str = '+', 
        todms: bool =False, 
        is_utm: bool  =False,
        **kws
  )-> Tuple[ArrayLike[DType[float]]]: 
    """ Generate multiples stations coordinates (longitudes, latitudes)
    from a reference station/site.
    
    One degree of latitude equals approximately 364,000 feet (69 miles), 
    one minute equals 6,068 feet (1.15 miles), and one-second equals 101 feet.
    One-degree of longitude equals 288,200 feet (54.6 miles), one minute equals
    4,800 feet (0.91 mile) , and one second equals 80 feet. Illustration showing
    longitude convergence. (1 feet ~=0.3048 meter)
    
    Parameters 
    ----------
    reflong: float or string or list of [start, stop]
        Reference longitude  in degree decimal or in DD:MM:SS for the first 
        site considered as the origin of the landmark.
        
    reflat: float or string or list of [start, stop]
        Reference latitude in degree decimal or in DD:MM:SS for the reference  
        site considered as the landmark origin. If value is given in a list, 
        it can containt the start point and the stop point. 
        
    nsites: int or float 
        Number of site to generate the coordinates onto. 
        
    r: float or int 
        The rotate angle in degrees. Rotate the angle features the direction
        of the projection line. Default value is ``45`` degrees. 
        
    step: float or str 
        Offset or the distance of seperation between different sites in meters. 
        If the value is given as string type, except the ``km``, it should be 
        considered as a ``m`` value. Only meters and kilometers are accepables.
        
    order: str 
        Direction of the projection line. By default the projected line is 
        in ascending order i.e. from SW to NE with angle `r` set to ``45``
        degrees. Could be ``-`` for descending order. Any other value should 
        be in ascending order. 
    
    is_utm: bool, 
        Consider the first two positional arguments as UTM coordinate values. 
        This is an alternative way to assume `reflong` and `reflat` are UTM 
        coordinates 'easting'and 'northing` by default. If `utm2deg` is ``False``, 
        any value greater than 180 degrees for longitude and 90 degrees for 
        latitude will raise an error. Default is ``False``.
        
    utm_zone: string (##N or ##S)
        utm zone in the form of number and North or South hemisphere, 10S or 03N
        Must be given if `utm2deg` is set to ``True``. 
                      
    todms: bool 
        Convert the degree decimal values into the DD:MM:SS. Default is ``False``. 
        
    kws: dict, 
        Additional keywords of :func:`.gistools.project_point_utm2ll`. 
        
    Returns 
    -------
        Tuple of  generated projected coordinates longitudes and latitudes
        either in degree decimals or DD:MM:SS
        
    Notes 
    ------
    The distances vary. A degree, minute, or second of latitude remains 
    fairly constant from the equator to the poles; however a degree, minute,
    or second of longitude can vary greatly as one approaches the poles
    and the meridians converge.
        
    References 
    ----------
    https://math.answers.com/Q/How_do_you_convert_degrees_to_meters
    
    Examples 
    --------
    >>> from watex.utils.coreutils import makeCoords 
    >>> rlons, rlats = makeCoords('110:29:09.00', '26:03:05.00', 
    ...                                     nsites = 7, todms=True)
    >>> rlons
    ... array(['110:29:09.00', '110:29:35.77', '110:30:02.54', '110:30:29.30',
           '110:30:56.07', '110:31:22.84', '110:31:49.61'], dtype='<U12')
    >>> rlats 
    ... array(['26:03:05.00', '26:03:38.81', '26:04:12.62', '26:04:46.43',
           '26:05:20.23', '26:05:54.04', '26:06:27.85'], dtype='<U11')
    >>> rlons, rlats = makeCoords ((116.7, 119.90) , (44.2 , 40.95),
                                            nsites = 238, step =20. ,
                                            order = '-', r= 125)
    >>> rlons 
    ... array(['119:54:00.00', '119:53:11.39', '119:52:22.78', '119:51:34.18',
           '119:50:45.57', '119:49:56.96', '119:49:08.35', '119:48:19.75',
           ...
           '116:46:03.04', '116:45:14.43', '116:44:25.82', '116:43:37.22',
           '116:42:48.61', '116:42:00.00'], dtype='<U12')
    >>> rlats 
    ... array(['40:57:00.00', '40:57:49.37', '40:58:38.73', '40:59:28.10',
           '41:00:17.47', '41:01:06.84', '41:01:56.20', '41:02:45.57',
           ...
       '44:07:53.16', '44:08:42.53', '44:09:31.90', '44:10:21.27',
       '44:11:10.63', '44:12:00.00'], dtype='<U11')
    
    """  
    def assert_ll(coord):
        """ Assert coordinate when the type of the value is string."""
        try: coord= float(coord)
        except ValueError: 
            if ':' not in coord: 
                raise ValueError(f'Could not convert value to float: {coord!r}')
            else : 
                coord = convert_position_str2float(coord)
        return coord
    
    xinf, yinf = None, None 
    
    nsites = int(_assert_all_types(nsites,int, float)) 
    if isinstance (reflong, (list, tuple, np.ndarray)): 
        reflong , xinf, *_ = reflong 
    if isinstance (reflat, (list, tuple, np.ndarray)): 
        reflat , yinf, *_ = reflat 
    step=str(step).lower() 
    if step.find('km')>=0: # convert to meter 
        step = float(step.replace('km', '')) *1e3 
    elif step.find('m')>=0: step = float(step.replace('m', '')) 
    step = float(step) # for consistency 
    
    if str(order).lower() in ('descending', 'down', '-'): order = '-'
    else: order ='+'
    # compute length of line using the reflong and reflat
    # the origin of the landmark is x0, y0= reflong, reflat
    x0= assert_ll(reflong) if is_utm else assert_ll(
        assert_lon_value(reflong))
    y0= assert_ll(reflat) if is_utm else assert_ll(
        assert_lat_value(reflat))
    
    xinf = xinf or x0  + (np.sin(np.deg2rad(r)) * step * nsites
                          ) / (364e3 *.3048) 
    yinf = yinf or y0 + (np.cos(np.deg2rad(r)) * step * nsites
                         ) /(2882e2 *.3048)
    
    reflon_ar = np.linspace(x0 , xinf, nsites ) 
    reflat_ar = np.linspace(y0, yinf, nsites)
    #--------------------------------------------------------------------------
    # r0 = np.sqrt(((x0-xinf)*364e3 *.3048)**2 + ((y0 -yinf)*2882e2 *.3048)**2)
    # print('recover distance = ', r0/nsites )
    #--------------------------------------------------------------------------
    if is_utm : 
        if utm_zone is None: 
            raise TypeError("Please provide your UTM zone e.g.'10S' or '03N' !")
        lon = np.zeros_like(reflon_ar) 
        lat = lon.copy() 
        
        for kk , (lo, la) in enumerate (zip(reflat_ar, reflon_ar)): 
            try : 
                with warnings.catch_warnings(): # ignore multiple warnings 
                    warnings.simplefilter('ignore')
                    lat[kk], lon[kk] = project_point_utm2ll(
                        easting= lo, northing=la, utm_zone=utm_zone, **kws)
            except : 
                lat[kk], lon[kk] = utm_to_ll(
                    23, northing=la, easting=lo, zone=utm_zone)
                
        if not HAS_GDAL : 
            warnings.warn("It seems GDAL is not set! will use the equations"
                          "from USGS Bulletin 1532. Be aware, the positionning" 
                          " is less accurate than using GDAL.")
            
        warnings.warn("By default 'easting' and 'northing' are presumed to match"
                      " the first and second argument respectively.") 
        
        reflat_ar, reflon_ar = lat , lon 
    
    if todms:
       reflat_ar = np.array(list(
           map(lambda l: convert_position_float2str(float(l)), reflat_ar)))
       reflon_ar = np.array(list(
           map(lambda l: convert_position_float2str(float(l)), reflon_ar)))
       
    return (reflon_ar , reflat_ar ) if order =='+' else (
        reflon_ar[::-1] , reflat_ar[::-1] )  

#XXX OPTIMIZE 
def parseDCArgs(fn :str , 
                delimiter:Optional[str]=None,
                 arg='stations'
                 )-> ArrayLike [str]: 
    """ Parse DC `stations` and `fromS` arguments from file and output to 
    array accordingly.
    
    The `froms` argument is the depth in meters from which one expects to find  
    a fracture zone outside of pollutions. Indeed, the `fromS` parameter is
    used to  speculate about the expected groundwater in the fractured rocks 
    under the average level of water inrush in a specific area. For more details
    refer to :attr:`watex.methods.electrical.VerticalSounding.fromS` 
    documentation. 
    
    :param fn: path-like object, full path to DC station or fromS file. 
        if data is considered as a station file, it must be composed  
        the station names. Commonly it can be used to specify the selected 
        station of all DC-resistity line where one expects
        to locate the drilling. 
        Conversly, the fromS file should not include any letter so if given, 
        ot sould be removed.  
        
    :param arg: str of the attribute of the DC methods.Any other value except 
        ``station`` should considered as ``fromS`` value and will parse the 
        file accordingly. 
        
    :param delimiter: str , delimiter to separate the different stations 
        or 'fromS' value. For instance, use use < delimiter=' '> when all 
        values are separated with space and be arranged in the same line like::
            
            >>> 'S02 S12 S12 S15 S28 S30' #  line of the file.
    
    :return: 
        array: array of station name. 
        
    :note: if all station prefixes belong to the module station property object 
        i.e :class:`watex.property.P.istation`, the prefix should be overwritten 
        to only keep the `S`. For instance 'pk25'-> 'S25'
    
    :Example: 
        >>> from watex.utils.coreutils import parseDCArgs 
        >>> sf='data/sfn.txt' # use delimiter if values are in the same line. 
        >>> sdata= parseDCArgs(sf)
        >>> sdata 
        ...
        >>> # considered that the digits in the file correspond to the depths 
        >>> fdata= parseDCArgs(sf, arg='froms') 
        >>> fdata 
        ...
    """
    if not os.path.isfile (fn): 
        raise FileNotFoundError("No file found:")
    arg= str(arg).lower().strip() 
    if arg.find('station')>=0 : 
        arg ='station'
    with open(fn, 'r', encoding ='utf8') as f : 
        sdata = f.readlines () 
    if delimiter is not None: 
        # flatter list into a list 
        sdata = list(map (lambda l: l.split(delimiter), sdata ))
        sdata = list(itertools.chain(*sdata))

    regex =re.compile (rf"{'|'.join([a for a in (P().istation+['S'])])}", 
                       flags =re.IGNORECASE
                       ) if arg =='station' else re.compile (
                           r'\d+', flags=re.IGNORECASE ) 
    
    sdata = list(map(lambda o:  regex.sub('S', o.strip()), 
                     sdata )
                 ) if arg =='station' else list(map(
                     lambda o:  regex.findall(o.strip()), sdata )
                              )
    # for consitency delte all empty string in the list 
    sdata = list(filter (None, sdata ))
    
    return np.array(sdata )if arg=='station' else reshape (np.array(
        sdata ).astype(float))


def read_data (
        f:str, 
        **kws
 ) -> DataFrame: 
    """ Assert and read specific files and url allowed by the package
    
    Readable files are systematically convert to a frame.  
    
    :param f: Path-like object -Should be a readable files or url  
    :param kws: Pandas readableformats additional keywords arguments. 
    :return: dataframe - A dataframe with head contents... 
    
    """
    cpObj= Config().parsers 
    
    if isinstance (f, pd.DataFrame): 
        return f 
    
    elif not os.path.isfile :# or ('http' not in f) : 
        try : 
            if 'http' in f: # force pandas read html etc 
                pass 
        except:
            raise TypeError (
            f'Expected a Path-like object or url, got : {type(f).__name__!r}')
        
    _, ex = os.path.splitext(f.strip()) 
    if ex.lower() not in tuple (cpObj.keys()):
        raise TypeError(f"Can only parse the {smft(cpObj.keys(), 'or')} files"
                        )
    try : 
        f = cpObj[ex](f, **kws)
        
    except FileNotFoundError:
        raise FileNotFoundError (
            f"No such file in directory: {os.path.basename (f)!r}")
    except: 
        raise FileHandlingError (
            f" Can not parse the file : {os.path.basename (f)!r}")

    return f 
    




















        