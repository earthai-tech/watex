# -*- coding: utf-8 -*-
#   Copyright (c) 2021  @Daniel03 <etanoyau@gmail.com>
#   Created date: Fri Apr 15 10:46:56 2022
#   Licence: MIT Licence 
# 
from __future__ import  annotations 

import os 
import warnings 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from .._typing import (
    Any, 
    List ,  
    Union, 
    Tuple,
    Dict,
    NDArray,
    Array, 
    DType, 
    Sub, 
    SP
)
from .func_utils import smart_format 
from .ml_utils import read_from_excelsheets
from ..properties import P 
from ..exceptions import ( 
    WATexError_station, 
    WATexError_parameter_number
)

def _data_sanitizer (
        f: str | NDArray,
        **kws:Any 
) -> pd.DataFrame : 
    """ Sanitize the raw data collected from the survey. 
    
    `data` should be arranged in ``.csv`` or ``.xlsx`` formats. Be sure to 
    provide the header of each columns in the worksheet. In the given file
    data should be aranged as:: 
        
        ['station','easting', 'northing', 'resistivity' ]
        
    If it is not the case, user may provide at least the prefix composed of 
    the first four letters of each column. 
    
    :param f: str. Path to the data location. Can parse `.csv` and `.xlsx` 
        file formats. Can also accept arrays and the output is one of the 
        given result::
            
            - array-like shape (M,): --> pd.Series with name ='resistivity'
            - array with shape (M, 2) --> ['station', 'resistivity'] 
            - array with shape (M, 3) --> Raise a ValueError 
            - array with shape (M, 4) --> columns above 
            - array with shape (M , N ) --> shrunked to fit the colum above. 
            
            
    :param kws: dict. Additional pandas `~.read_csv` and `~.read_excel` 
        methods keyword arguments. Be sure to provide the right argument . 
        when reading `f`. For instance, provide `sep=','` argument when 
        the file to read is ``xlsx`` format will raise an error. Indeed, 
        `sep` parameter is acceptable for parsing the `.csv` file format
        only.
        
        
    :return: DataFrame or Series with valuable column(s). 
    
    .. note:: The length of acceptable columns is ``4``. If the size of the 
            columns is higher than `4`, the data should be shrunked to match
            the expected columns. Futhermore, if the header is not specified in 
            `f`, the defaut column arrangement should be used. Therefore, the 
            last column should be considered as the ``resistivity` column. 
     
    .. Example:: 
        >>> import numpy as np 
        >>> from watex.utils.coreutils import _data_sanitizer 
        >>> df = _data_sanitizer ('data/erp/testsafedata.csv')
        >>> df.shape 
        ... (45, 4)
        >>> list(df.columns) 
        ... ['station', 'easting', 'northing', 'resistivity']
        >>> df = _data_sanitizer ('data/erp/testunsafedata.xlsx') 
        >>> list(df.columns)
        ... ['easting', 'station', 'resistivity', 'northing']
        >>> df = _data_sanitizer(np.random.randn(7)) 
        >>> df.name 
        ... 'resistivity'
        >>> df = _data_sanitizer(np.random.randn(7, 7)) 
        >>> df.shape 
        ... (7, 4)
        >>> list(df.columns) 
        ... ['station', 'easting', 'northing', 'resistivity']
    
        >>> df = _data_sanitizer(np.random.randn(7, 3)) 
        ... ValueError: ...

    """
    if os.path.isfile(f): 
        if os.path.splitext(f)[1].lower() not in ('.csv', '.xlsx'):
            raise ValueError('Can only read the file `.csv and `.xlsx`'
                            ' file. Please provide the right file.')
    
        if f.endswith ('.csv'): 
            f = pd.read_csv (f,**kws) 
        elif f.endswith ('.xlsx'): 
            f = pd.read_excel(f, **kws )

    if isinstance(f, pd.DataFrame): 
        rawcol = f.columns 
        temp = list(map(lambda x: x.lower().strip(), f.columns))  
        for i, item  in enumerate (temp): 
            for key, values in P().Dtags.items (): 
                for v in values: 
                    if item.find(v)>=0: 
                        temp[i] = key 
                        break 
        # check the existence of  duplicate element 
        # into the dataframe column
        if len(set (temp)) != len(temp): 
            # search for duplicated items by making  
            # a copy of original list. Thus by using 
            # filter, we remove all item found in 
            # the iterated set
            duplicate =temp.copy() 
            list(filter (lambda x: duplicate.remove(x), set(temp)))
            # Find the corresponding prefix values in DTAGS 
            # then use the values for searching the raw 
            # columns name. 
            tv = list(P().Dtags.get(key) for key in duplicate)
            for val in tv: 
                ls = set([ it for it in rawcol for e in val if it.find(e)>=0])
                if len(ls)!=0: 
                    break 
            raise WATexError_parameter_number(
                f'Duplicate column{"s" if len(ls)>1 else ""}'
                f' {smart_format(ls)} found. It seems to be'
                f' {smart_format(duplicate)} '
                f'column{"s" if len(duplicate)>1 else ""}.'
                ' Please provide the right column name in'
                ' the dataset.'
                              )
        # rename dataframe columns 
        f.columns= temp 
    # If an array is given instead of dataframe.
    # will check whether the shape of array match
    elif isinstance( f, np.ndarray): 
        msg ='Please create a dataframe to exactly fit {} columns.'
        appendmsg =''.join([
            ' If `easting` and `northing` data are not available, ', 
            "use ['station', 'resistivity'] columns instead."])
        
        if len(f.shape) ==1 : # for array-like 
            warnings.warn('1D array is found. Values should be considered'
                          ' as the resistivity values')
            f= pd.Series (f, name ='resistivity')
        elif f.shape[1] ==2: 
            warnings.warn('2D dimensional array is found. Head columns should'
                          " match `{}` by default.".format(
                              ['station','resistivity']))
            f = pd.DataFrame( f, columns =['station', 'resistivity']) 
            
        elif f.shape [1] ==3: 
            raise ValueError (msg.format(P().SENR) + appendmsg ) 
        elif f.shape[1]==4:
            f =pd.DataFrame (
                f, columns = P().SENR 
                )
        elif f.shape [1] > 4: 
            # add 'none' columns for the remaining columns.
                f =pd.DataFrame (
                    f, columns = P().SENR + [
                        'none' for i in range(f.shape[1]-4)]
                    )
    else : 
        raise ValueError ('Unaccepatable data. Can only parse'
                          ' `pandas.DataFrame`, `.xlsx` and '
                          '`.csv` file format.')        
    # shrunk the dataframe out of 4 columns . 
    if len(f.shape ) !=1 and f.shape[1]> 4 : 
        warnings.warn(f'Expected four columns = `{P().SENR}`, '
                      f'but `{f.shape[1]}` are given. Data is shrunked'
                      ' to match the fitting columns.')
        f = f.iloc[::, :4]

    return f 

def _fetch_prefix_index (
    arr:NDArray |None = None,
    col: List[str] | None = None,
    df :pd.DataFrame | None = None, 
    prefixs: List [str ] | None =None
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
   
    :Example: 
        >>> from numpy as np 
        >>> from watex.utils.coreutils import _assert_positions
        >>> array1 = np.c_[np.arange(0, 70, 10), np.random.randn (7,3)]
        >>> col = ['pk', 'x', 'y', 'rho']
        >>> _fetch_prefix_index (array1 , col = ['pk', 'x', 'y', 'rho'], 
        ...                         prefixs = EASTPREFIX)
        ... [1]
        >>> _fetch_prefix_index (array1 , col = ['pk', 'x', 'y', 'rho'], 
        ...                         prefixs = NOTHPREFIX )
        ... [2]
    """
    if prefixs is None: 
        raise ValueError('Please specify the list of items to compose the '
                         'prefix to fetch the columns data. For instance'
                         f' `station prefix` can  be `{P().STAp}`.')

    if arr is None and df is None :
        raise TypeError ( 'Expected and array or a dataframe not'
                         ' a Nonetype object.'
                        )
    elif df is None and col is None: 
        raise WATexError_station( 'Column list is missing.'
                         ' Could not detect the position index') 
        
    if isinstance( df, pd.DataFrame): 
        # collect the resistivity from the index 
        # if a dataFrame is given 
        arr, col = df.values, df.columns 

    if arr.ndim ==1 : 
        # Here return 0 as colIndex
        return arr , 0
    if isinstance(col, str): col =[col] 
    if len(col) != arr.shape[1]: 
        raise ValueError (
            'Column should match the array shape in axis =1 <{arr.shape[1]}>.'
            f' But {"was" if len(col)==1 else "were"} given')
        
    # convert item in column in lowercase 
    comsg = col.copy()
    col = list(map(lambda x: x.lower(), col)) 
    colIndex = [col.index (item) for item in col 
             for pp in prefixs if item.find(pp) >=0]   

    if len(colIndex) is None: 
        raise ValueError ( 'Unable to detect the position'
                          f' in `{smart_format(comsg)}` columns'
                          '. Columns must contain at least'
                          f' `{smart_format(prefixs)}`.')
    return colIndex 


def _assert_station_positions(
    arr: NDArray | None =None,
    prefixs: List [str] |None =P().STAp,
    **kws
) -> Tuple [int, float]: 
    """ Assert positions and compute dipole length. 
    
    Use the given station postions collected on the field to 
    detect the dipole length during the whole survey. 
    
    :param arr: array. Ndarray of data where one colum must the 
            positions values. 
    :param col: list. The list should be considered as the head of array. Each 
        position in the list sould fit the column data in the array. It raises 
        an error if the number of item in the list is different to the size 
        of array in axis=1. 
    :param df: dataframe. When supply, the `arr` and `col` is not needed.

    :param prefixs: list. Contains all the station column names prefixs to 
        fetch the corresponding data.
   
    :Example: 
        
        >>> from numpy as np 
        >>> from watex.utils.coreutils import _assert_positions
        >>> array1 = np.c_[np.arange(0, 70, 10), np.random.randn (7,3)]
        >>> col = ['pk', 'x', 'y', 'rho']
        >>> _assert_positions(array1, col)
        ... (array([ 0, 10, 20, 30, 40, 50, 60]), 10)
        >>> array1 = np.c_[np.arange(30, 240, 30), np.random.randn (7,3)]
        ... (array([  0,  30,  60,  90, 120, 150, 180]), 30)
    
    """

    colIndex =_fetch_prefix_index(arr=arr, prefixs = prefixs, **kws )
    positions= arr[:, colIndex[0]]
    # assert the position is aranged from lower to higher 
    # if there is not wrong numbering. 
    fsta = np.argmin(positions) 
    lsta = np.argmax (positions)
    if int(fsta) !=0 or int(lsta) != len(positions)-1: 
        raise WATexError_station(
            'Wrong numbering! Please number the position from first station '
            'to the last station. Check your array positionning numbers.')
    
    dipoleLength = int(np.abs (positions.min() - positions.max ()
                           ) / (len(positions)-1)) 
    # renamed positions  
    positions = np.arange(0 , len(positions) *dipoleLength ,
                          dipoleLength ) 
    
    return  positions, dipoleLength 

def plot_anomaly(
    sample:NDArray | List[float],
    cz:NDArray | List[float], 
    station: str =None, 
    figsize:Tuple [int, int] =(10, 4), 
    show_fig_title:bool =True,
    fig_title_kws:str |None  = None,
    show_grid: bool =True,
    grid_which: str ='major',
    erpkws :Dict [str , str | Any ] =None, 
    mkws: Dict [str , str | Any ]=None,
    czkws : Dict [str , str | Any ]=None , 
    legkws: Dict [Any , str | Any ] =None, 
    show_markers =True,
    xlabel :str |None = None,
    ylabel :str |None =None,
    fig_dpi :int =300 ,
    savefig :str |None =None
) -> None: 

    """ Quick plot to visualize the  selected conductive zone overlained  
    to the whole electrical resistivity profiling.
    
    :param sample: array_like - the electrical profiling array. 
    :param cz: array_like - the selected conductive zone. If ``None``, `cz` 
        should be plotted only.
    
    :Example: 
        >>> import numpy as np 
        >>> from watex.utils.coreutils import ( 
        ...    plot_anomaly, _define_conductive_zone)
        >>> test_array = np.random.randn (10)
        >>> selected_cz ,*_ = _define_conductive_zone(test_array, 7) 
        >>> plot_anomaly(test_array, selected_cz )
        
    """
    if mkws is None:
        mkws = {'marker':'o', 'edgecolors' :'white', 'c':P().FRctags.get('fr3'), 
                   'alpha' :.9, 's':60.}
        
    if erpkws is None:
        erpkws =dict (color=P.FRctags.get('fr1'), 
                      linestyle='-',
                      linewidth=3,
                      label = 'Electrical resistivity profiling'
                      )

    if czkws is None:
        czkws =dict (color='fr3', 
                      linestyle='-',
                      linewidth=2,
                      # markersize=12,
                      label = 'Conductive zone'
                      )
    
    if czkws.get('color') is not None: 
        if str(czkws.get('color')).lower().find('fr')>=0: 
            try : 
                czkws['color']= P().FRctags.get(czkws['color'])
            except: 
                czkws['color']= P().FRctags.get('fr3')

    fig, ax = plt.subplots(1,1, figsize =(10, 4))
    leg =[]

    zl, = ax.plot(np.arange(len(sample)), sample, 
                  **erpkws 
                  )
    leg.append(zl)
    if cz is not None: 
        # construct a mask array with np.isin to check whether 
        # `cz` is subset array
        z = np.ma.masked_values (sample, np.isin(sample, cz ))
        # a masked value is constructed so we need 
        # to get the attribute fill_value as a mask 
        # However, we need to use np.invert or the tilde operator  
        # to specify that other value except the `CZ` values mus be 
        # masked. Note that the dtype must be changed to boolean
        sample_masked = np.ma.array(
            sample, mask = ~z.fill_value.astype('bool') )

        czl, = ax.plot(
            np.arange(len(sample)), sample_masked, 
            **czkws)
        leg.append(czl)
        
    if show_markers: 
        ax.scatter (np.arange(len(sample)), sample,
                        **mkws if mkws is not None else mkws,  
                        )
    
    ax.set_xticks(range(len(sample)))
    ax.set_xticklabels(
        ['S{0:02}'.format(i+1) for i in range(len(sample))])
    
    if xlabel is None:
        xlabel ='Stations'
    if ylabel is None: 
        ylabel ='Resistivity (Ω.m)'
    if legkws is None: 
        legkws =dict() 
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend( handles = leg, 
              **legkws )
    
    if show_grid is True : 
        if grid_which=='minor': 
              ax.minorticks_on() 
        ax.grid(show_grid,
                axis='both',
                which =  grid_which, 
                color = 'k',
                linestyle='--',
                linewidth=1., 
                alpha = .2 
                )
       
    if show_fig_title: 
        if fig_title_kws is None: 
            fig_title_kws = dict (
                t = 'Plot ERP line with SVES= {0}'.format(station), 
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
        


def _define_conductive_zone(
    erp: List[float| int ] | pd.Series,
    s: str | int = None, 
    auto: bool = False, 
    **kws,
) -> Tuple [NDArray, int] :
    """ Define conductive zone as subset of the erp line.
    
    Indeed the conductive zone is a specific zone expected to hold the 
    drilling location `s`. If drilling location is not provided, it would be 
    by default the very low resistivity values found in the `erp` line. 
    
    
    :param erp: array_like, the array contains the apparent resistivity values 
    :param s: str or int, is the station position. 
    :param auto: bool. The station position should be the position of 
    the lower resistivity value in `erp`. 
    
    :returns: 
        - conductive zone 
        - station position 
    
    :Example: 
        >>> import numpy as np 
        >>> from watex.utils.coreutils import  _define_conductive_zone
        >>> test_array = np.random.randn (10)
        >>> selected_cz ,*_ = _define_conductive_zone(test_array, 's20') 
        >>> shortPlot(test_array, selected_cz )
    """
    if isinstance(erp, pd.Series): erp = erp.values 
    
    if s is None and auto is False: 
        raise TypeError ('Expected the station position. NoneType is given.')
    elif s is None and auto: 
        s = int(np.argwhere (erp == erp.min())) 
    s , pos = _assert_stations(s, **kws )
    # takes the last position if the position is outside 
    # the number of stations. 
    pos = len(erp)-1 if pos >= len(erp) else pos 
    # frame the `sves` (drilling position) and define the conductive zone 
    ir = erp[:pos][-3:]
    il = erp[pos:pos +3 +1 ]
    cz = np.concatenate((ir, il))
    
    return cz , pos 



def _assert_stations(
    s:Any , 
    dipole:Any=None,
    keepnums:bool=False
) -> Tuple[str, int]:
    """ Sanitize stations and returns station name and index.
    
    ``pk`` and ``S`` can be used as prefix to define the station `s`. For 
    instance ``S01`` and ``PK01`` means the first station. 
    
    :param s: Station name
    :type s: str, int 
    
    :param dipole: dipole_length in meters.  
    :type dipole: float 
    
    :returns: 
        - station name 
        - index of the station.
        
    .. note:: The station should numbered from 1 not 0. SO if ``S00` is given
            the station name should be set to ``S01``. Moreover, if `dipole`
            value is set, i.e. the station is  named according to the 
            value of the dipole. For instance for `dipole` equals to ``10m``, 
            the first station should be ``S00``, the second ``S10`` , 
            the third ``S30`` and so on. However, it is recommend to name the 
            station using counting numbers rather than using the dipole 
            position.
            
    :Example: 
        >>> from watex.utils.coreutils import _assert_stations
        >>> _assert_stations('pk01')
        >>> _assert_stations('S1')
        >>> _assert_stations('S00')
        >>> _assert_stations('S1000',dipole ='1km')
        >>> _assert_stations('S10', dipole ='10m')
        >>> _assert_stations(1000,dipole =1000)
        ... ('S01', 0) 
    """
    # in the case s is string: eg. "00", "pk01", "S001"
    ix = 0.
    if isinstance(s, str): 
        s =s.lower().replace('pk', '').replace('s', '')
    
    if dipole is not None: 
        if isinstance(dipole, str): #'10m'
            if dipole.find('km')>=0: 
                dipole = dipole.lower().replace('km', '000')
            dipole = float(dipole.lower().replace('m', ''))
        # since the renamed from dipole starts at 0 
        # e.g. 0(S1)---10(S2)---20(S3) ---30(S4)etc ..
        s= int(s)//dipole +1 
    
    ix = int(s) if int(s) ==0 else int(s) -1 
    s = "S{:02}".format(ix +1) 

    return s, ix 


def _parse_args (
    args:Union[List | str ]
)-> Tuple [ pd.DataFrame, List[str|Any]]: 
    """ `Parse_args` function returns array of rho and coordinates 
    values (X, Y).
    
    Arguments can be a list of data, a dataframe or a Path like object. If 
    a Path-like object is set, it should be the priority of reading. 
    
    :param args: arguments 
    
    :return: ndarrayor array-like  arranged with apparent 
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
    









































        
        