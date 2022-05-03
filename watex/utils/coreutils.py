# -*- coding: utf-8 -*-
#   Copyright (c) 2021  @Daniel03 <etanoyau@gmail.com>
#   Created date: Fri Apr 15 10:46:56 2022
#   Licence: MIT Licence 
# 
from __future__ import  annotations 

import os 
import warnings 
import copy 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
 

from .._property import P 
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
    Array, 
    DType, 
    Sub, 
    SP
)
from .exmath import __isin , __assert_all_types 
from .func_utils import smart_format 
from .ml_utils import read_from_excelsheets

from ..exceptions import ( 
    WATexError_station, 
    WATexError_parameter_number
)

def data_sanitizer (
        f: str | NDArray | Series | DataFrame ,
        **kws:Any 
) -> Union [Series, DataFrame]  : 
    """ Sanitize the raw data collected from the survey. 
    
    `data` should be arranged in ``.csv`` or ``.xlsx`` formats. Be sure to 
    provide the header of each columns in the worksheet. In the given file
    data should be aranged as:: 
        
        ['station','easting', 'northing', 'resistivity' ]
        
    If it is not the case, user may provide at least the prefix composed of 
    the first four letters of each column. 
    
    :param f: str. Path to the data location. Can parse `.csv` and `.xlsx` 
        file formats. Can also accept arrays and the output is one of the 
        given results::
            
            - array-like shape (M,): --> pd.Series with name ='resistivity'
            - array with shape (M, 2) --> ['station', 'resistivity'] 
            - array with shape (M, 3) --> Raise an Error 
            - array with shape (M, 4) --> columns above 
            - array with shape (M , N ) --> shrunked to fit the column above. 
            
            
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
            for key, values in P().idictags.items (): 
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
            tv = list(P().idictags.get(key) for key in duplicate)
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
            raise ValueError (msg.format(P().isenr) + appendmsg ) 
        elif f.shape[1]==4:
            f =pd.DataFrame (
                f, columns = P().isenr 
                )
        elif f.shape [1] > 4: 
            # add 'none' columns for the remaining columns.
                f =pd.DataFrame (
                    f, columns = P().isenr + [
                        'none' for i in range(f.shape[1]-4)]
                    )
    else : 
        raise ValueError ('Unaccepatable data. Can only parse'
                          ' `pandas.DataFrame`, `.xlsx` and '
                          '`.csv` file format.')        
    # shrunk the dataframe out of 4 columns . 
    if len(f.shape ) !=1 and f.shape[1]> 4 : 
        warnings.warn(f'Expected four columns = `{P().isenr}`, '
                      f'but `{f.shape[1]}` are given. Data is shrunked'
                      ' to match the fitting columns.')
        f = f.iloc[::, :4]

    return f 

def _fetch_prefix_index (
    arr:NDArray [DType[float]] | None = None,
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
                         f' `station prefix` can  be `{P().istation}`.')

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
    prefixs: List [str] |None =P().istation,
    **kws
) -> Tuple [int, float]: 
    """ Assert positions and compute dipole length. 
    
    Use the given station postions collected on the field to 
    detect the dipole length during the whole survey. 
    
    :param arr: array. Ndarray of data where one column must the 
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
    erp:Array | List[float],
    cz:Optional [Sub[Array], List[float]] = None, 
    s: Optional [str] = None, 
    figsize:Tuple [int, int] =(10, 4),
    fig_dpi :int =300 ,
    savefig :str | None =None, 
    show_fig_title:bool =True,
    style : str = 'seaborn', 
    fig_title_kws:Dict[str, str | Any] = ...,
    czkws : Dict [str , str | Any ] = ... , 
    legkws: Dict [Any , str | Any ] = ... , 
    **kws, 
) -> None: 

    """ Plot the whole |ERP| line and selected conductive zone. 
    
    Conductive zone can be supplied nannualy as a subset of the `erp` or by 
    specifyting the station expected for drilling location. For instance 
    ``S07`` for the seventh station. Futhermore, for automatic detection, one 
    should set the station argument `s`  to ``auto``. However, it 's recommended 
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
    
    :param show_fig_tile: bool - display the title of the figure 
    
    :param fig_title_kws: dict - Keywords arguments of figure suptile. Refer to 
        `Matplotlib figsuptitle`_
        
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
        
     
    Web resources  
    --------------
    
    See Matplotlib Axes: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html
    GeekforGeeks: https://www.geeksforgeeks.org/style-plots-using-matplotlib/#:~:text=Matplotlib%20is%20the%20most%20popular,without%20using%20any%20other%20GUIs.
    """
    
    def format_thicks (value, tick_number):
        """ Format thick parameter with 'FuncFormatter(func)'
        rather than using:: 
            
        axi.xaxis.set_major_locator (plt.MaxNLocator(3))
        
        ax.xaxis.set_major_formatter (plt.FuncFormatter(format_thicks))
        """
        if value % 7 ==0: 
            return 'S{:02}'.format(int(value)+ 1)
        else: None 
        
    
    erp = __assert_all_types( 
        erp, tuple, list , np.ndarray , pd.Series)
    if cz is not None: 
        cz = __assert_all_types(
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
        cz , ix = _define_conductive_zone(
            erp, s = s , auto = auto, keepindex=keepindex )
        
        s = "S{:02}".format(ix +1) if s is not None else s 

    if cz is not None: 
        # construct a mask array with np.isin to check whether
        if not __isin (erp, cz ): 
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
        # for axi in ax.flat() : 
            # axi.xaxis.set_major_locator (plt.MaxNLocator(3))
        ax.xaxis.set_major_formatter (plt.FuncFormatter(format_thicks))
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
        

def _define_conductive_zone(
    erp:Array| pd.Series | List[float] ,
    s: Optional [str ,  int] = None, 
    auto: bool = False, 
    **kws,
) -> Tuple [Array, int] :
    """ Define conductive zone as subset of the erp line.
    
    Indeed the conductive zone is a specific zone expected to hold the 
    drilling location `s`. If drilling location is not provided, it would be 
    by default the very low resistivity values found in the `erp` line. 
    
    
    :param erp: array_like, the array contains the apparent resistivity values 
    :param s: str or int, is the station position. 
    :param auto: bool. If ``True``, the station position should be 
            the position of the lower resistivity value in |ERP|. 
    
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
    s, pos = _assert_stations(s, **kws )
    # takes the last position if the position is outside 
    # the number of stations. 
    pos = len(erp) -1  if pos >= len(erp) else pos 
    # frame the `sves` (drilling position) and define the conductive zone 
    ir = erp[:pos][-3:] ;  il = erp[pos:pos +3 +1 ]

    cz = np.concatenate((ir, il))

    return cz , pos 



def _assert_stations(
    s:Any , 
    dipole:Any=None,
    keepindex:bool=False
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

    s = __assert_all_types(s, str, int, float)
    
    if isinstance(s, str): 
        s =s.lower().replace('pk', '').replace('s', '').replace('ta', '')
        try : 
            s = int(s )
        except : 
            raise TypeError ('Unable to convert str to float.')
        else : 
            # set index to 0 , is station `S00` is found for instance.
            if s ==0 : 
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
            
    
    if dipole is not None: 
        if isinstance(dipole, str): #'10m'
            if dipole.find('km')>=0: 
           
                dipole = dipole.lower().replace('km', '000') 
                
            dipole = dipole.lower().replace('m', '')
            try : 
                dipole = float(dipole) 
            except : 
                raise WATexError_station( 'invalid literal value for'
                                         f' dipole : {dipole!r}')
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
 
   
"""
.. |ERP| replace: Electrical resistivity profiling 

.. |VES| replace: Vertical electrical sounding 

.. _Matplotlib pyplot: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.plot.html

.. _Matplotlib figure: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.figure.html

.. _Matplotlib figsuptitle: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.suptitle.html

.. _GeekforGeeks: https://www.geeksforgeeks.org/style-plots-using-matplotlib/#:~:text=Matplotlib%20is%20the%20most%20popular,without%20using%20any%20other%20GUIs.

"""








































        
        