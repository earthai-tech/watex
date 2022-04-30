# -*- coding: utf-8 -*-
#   Copyright (c) 2021 Kouadio K. Laurent, 
#   Created datae: on Fri Sep 17 11:25:15 2021
#   Licence: MIT Licence 

from __future__ import annotations 

import copy 
import numpy as np
import pandas as pd 
import  matplotlib.pyplot as plt 
from scipy.signal import argrelextrema 
from  .. properties import P
from ..utils.decorator import deprecated 
from ..utils._watexlog import watexlog
from ..utils import exceptions as Wex 
from .._typing import (
    T, 
    List, 
    Tuple,
    Dict,
    Any,
    Union,
    Array,
    DType,
    Optional,
    Sub, 
    SP
)

_logger =watexlog.get_watex_logger(__name__)

def __isin (arr: Array | List [float] ,
              subarr: Sub [Array] |Sub[List[float]] | float ): 
    """ Check whether the subset array `subcz` is in  `cz` array. 
    
    :param arr: Array-like - Array of item elements 
    :param subarr: Array-like, float - Subset array containing a subset items.
    :return: True if items in  test array `subarr` are in array `arr`. 
    
    """
    arr , subarr  = np.array (arr ), np.array(subarr )
    return True if True in np.isin (arr, subarr) else False 

def __sves__ (
        s_index: int  , 
        cz: Array | List[float], 
        rhoap : Optional[float] =None, 
) -> Tuple[Array, Array]: 
    """ Divided the conductive zone in lefzone and righzone from 
    the drilling location index . 
    
    :param s_index - station location index expected for dilling location. 
        It refers to the position of |VES|. 
        
    :param cz: array-like - Conductive zone . 
    
    :returns: 
        - <--Sves: Left side of conductive zone from |VES| location. 
        - --> Sves: Right side of conductive zone from |VES| location. 
        
    .. note:: Both sides included the `Sves` |VES| position.
    
    """
    s_index = __assert_all_type( s_index , int)
    cz = __assert_all_type(cz, np.ndarray, pd.Series, list, tuple )
    ls , rs = max(cz[:s_index  + 1])  , max(cz[s_index  :])   
    
    if rhoap is not None : 
        
        if not __isin(rhoap, cz): 
            raise ValueError (
                f'A value `{rhoap}` should be an argument `cz` items. ')
        # now find with positions 
        for v, side  in zip((ls, rs ) , ('leftside', 'rightside')) : 
            if __isin(rhoap, v): 
                return v, side 
        
    return ls , rs 


def __assert_all_type (
        obj: object , 
        *expected_objtype: type 
 ) -> object: 
    """ Quick assertion of object type. Raise an `TypeError` if 
    wrong type is given."""

    if not isinstance( obj, expected_objtype): 
        return TypeError (
            f'Expected `{expected_objtype}` type but `{type(obj)}` is given.')
    return obj 

def detect_station_position (
        s : Union[str, int] ,
        p: Array|List [float] , 
) -> Tuple [int, float]: 
    """ Detect station position and return the index in positions
    
    :param s: str, int - Station location  in the position array. It should 
        be the positionning of the drilling location. If the value given
         is type string. It should be match the exact position to 
         locate the drilling. Otherwize, if the value given is in float or 
         integer type, it should be match the index of the position array. 
         
    :param p: Array-like - Should be the  conductive zone as array of 
        resistivity values. 
            
    :returns: 
        - `s_index`- the position index location in the conductive zone.  
        - `s`- the station position in distance. 
        
    :Example: 
        
        >>> import numpy as np 
        >>> from watex.utils.exmath import detect_station_position 
        >>> pos = np.arange(0 , 50 , 10 )
        >>> detect_station_position (s ='S30', p = pos)
        ... (3, 30.0)
        >>> detect_station_position (s ='40', p = pos)
        ... (4, 40.0)
        >>> detect_station_position (s =2, p = pos)
        ... (2, 20)
        >>> detect_station_position (s ='sta200', p = pos)
        ... WATexError_station: Station sta200 \
            is out of the range; max position = 40
    """
    s = __assert_all_type( s, float, int, str)
    p = __assert_all_type( p, tuple, list, np.ndarray, pd.Series) 
    
    S=copy.deepcopy(s)
    if isinstance(s, str): 
        s =s.lower().replace('s', '').replace('pk', '').replace('ta', '')
        try : 
            s=int(s)
        except : 
            raise ValueError (f'could not convert string to float: {S}')
    p = np.array(p, dtype = np.int32)
    dl = (p.max() - p.min() ) / (len(p) -1) 
    if isinstance(s, (int, float)): 
        if s > len(p): # consider this as the dipole length position: 
            # now let check whether the given value is module of the station 
            if s % dl !=0 : 
                raise Wex.WATexError_station  (
                    f'Unable to detect the station position {S}')
            elif s % dl == 0 and s <= p.max(): 
                # take the index 
                s_index = s//dl
                return int(s_index), s_index * dl 
            else : 
                raise Wex.WATexError_station (
                    f'Station {S} is out of the range; max position = {max(p)}'
                )
        else : 
            if s >= len(p): 
                raise Wex.WATexError_station (
                    'Location index must be less than the number of'
                    f' stations = {len(p)}. {s} is gotten.')
            # consider it as integer index 
            # erase the last variable
            # s_index = s 
            # s = S * dl   # find 
            return s , p[s ]
       
    # check whether the s value is in the p 
    if True in np.isin (p, s): 
        s_index ,  = np.where (p ==s ) 
        s = p [s_index]
        
    return int(s_index) , s 
    
def _sfi (
        cz: Sub[Array[T, DType[T]]] | List[float] ,
        p: Sub[SP[Array, DType [int]]] | List [int]= None, 
        s: Optional [str] =None, 
        dipolelength: Optional [float] = None, 
        plot: bool = False,
        raw : bool = False, 
        sckws: Dict [str, Any] = None, 
        plotstyle : str ='seaborn', 
        **plotkws
) -> float: 
    """ Compute  the pseudo-fracturing Index knows as *sfi*. 
    
    The sfi parameter does not indicate the rock fracturing degree in 
    the underground but it is used to speculate about the apparent resistivity 
    dispersion ratio around the cumulated sum of the  resistivity values of 
    the selected anomaly. It uses a similar approach of  IF parameter proposed 
    by `Dieng et al`_ (2004).  Furthermore, its threshold is set to
    :math:`\sqrt(2)`  for symmetrical anomaly characterized by a perfect 
    distribution of resistivity in a homogenous medium. The formula is
    given by:
    
    .. math::
        
        sfi=\sqrt((P_a/(P_a^\ast\ ))^2+(M_a/(M_a^\ast\ ))^2\ \ )
    
    where P_a and M_a are the anomaly power and the magnitude respectively. 
    :math:`\P_a^\ast\`  is and :math:`\M_a^\ast\` are the projected power and 
    magnitude of the lower point of the selected anomaly.
    
    :param cz: array-like. Selected conductive zone 
    :param p: array-like. Station positions of the conductive zone.
    :param dipolelength: float. If `p` is not given, it will be set 
        automatically using the default value to match the ``cz`` size. 
        The **default** value is ``10.``.
    :param plot: bool. Visualize the fitting curve.
    :param raw: bool. Overlaining the fitting curve with the raw curve from `cz`. 
    :param sckws: dict. `Matplotlib scatter`_ keyword additional arguments.
    :param plotkws:dict. `Matplotlib plot`_ keyword arguments. 
    
    
    :Example:
        
        >>> from numpy as np 
        >>> _sfi (condzone , plot =True , raw =True )
        ... Out[598]: (array([ 0., 10., 20., 30.]), 1)
        
    """
 
    
    
    # Determine the number of curve inflection 
    # to find the number of degree to compose 
    # cz fonction 
    if p is None :
        dipolelength = 10. if dipolelength is  None else dipolelength  
        p = np.arange (0, len(cz) * dipolelength, dipolelength)
    minl, = argrelextrema(cz, np.less)
    maxl, = argrelextrema(cz,np.greater)
    ixf = len(minl) + len(maxl)
    
    # create the polyfit function f from coefficents (coefs)
    coefs  = np.polyfit(x=p, y=cz, deg =ixf + 1 ) 
    f = np.poly1d(coefs )
    # generate a sample of values to cover the fit function 
    # for degree 2: eq => f(x) =ax2 +bx + c or c + bx + ax2 as 
    # the coefs are aranged. See https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html
    # coefs are ranged for index0  =c, index1 =b and index 2=a 
    # for instance for degree =2 
    # model (f)= [coefs[2] + coefs[1] * x  +   coefs [0]* x**2  for x in xmod]
    # where x_new(xn ) = 100 points generated 
    # thus compute ynew (yn) from the poly function f
    xn  = np.linspace (min(p), max(p), 1000) 
    yn = f(xn)
    # https://stackoverflow.com/questions/10457240/solving-polynomial-equations-in-python
    # solve the system to find the different root 
    # from the min resistivity value bound. 
    # -> Get the bound with the min values 
    
    if s is not None : 
        # explicity giving s 
        s_ix , spos = detect_station_position(s , p)
        # once index 
        maxlh , maxrh = __sves__(s_ix , cz )
        print(' maxlh , maxrh',  maxlh , maxrh)
        rhoamin = maxlh if maxlh < maxrh else maxrh 
    if s is None: 
        # take the index of min value of p 
        s_ix  = np.argmin(p)
        maxlh , maxrh = __sves__(s_ix , cz )
        rhoamin = maxlh if maxlh < maxrh else maxrh 
    # find the roots from rhoa min 
    # f (x) = rhoamin
    fn = f  - rhoamin 
    roots = fn.r 
    
    print('roots FN', roots)
    print('absolute roots FN', np.abs(roots))
    print('seek roots in f = ', f (roots))
    print('seek from abs roots in f = ', f (np.abs(roots)))
    print('--' *10)
    print('rhoamin= ', rhoamin )
    print('f(0) and f(18)=', f(0), f(18))
    print('fn(0) and fn(18)=', fn(0), fn(18))


    if plot: 
        plt.style.use('seaborn')
        fig = plt.figure(1, figsize =(7, 7))
        plt.plot (p, cz,'o', xn, yn, label= 'fitting model',  **plotkws)
        if raw: 
            plt.plot (p, cz, 
                      c = f'{P().FRctags.get("fr1")}',
                      label ='Conductive zone (cz)'
                      )
        sckws = dict() if sckws is None else sckws 
        plt.scatter (p, cz,
                     marker ='o', 
                     c=f'{P().FRctags.get("fr3")}',
                     edgecolor ='r',
                     **sckws 
                     )
        plt.xticks (p,
                    labels = [f'S{int(i):02}' for i in p],
                    rotation =45.
                    )
        plt.xlabel ('Stations')
        plt.ylabel ('Resistivity (Ω.m)')
        
        fig_title_kws = dict (
            t = 'Plot fit model', 
            style ='italic', 
            bbox =dict(boxstyle='round',facecolor ='lightgrey'))
            
        plt.tight_layout()
        fig.suptitle(**fig_title_kws, 
                      )
        plt.legend ()
        plt.show ()
    # plt.figure (1) 
    # plt.scatter (p, cz, marker ='o', c='blue', label ='Positions')
    # plt.plot (xmod, model,  c='r',
    #           label =f'Modelisation: y ={coefs[0]:.2f}x2 +{coefs[1]:.2f}x + {coefs[2]:.2f}')
    
    return p, ixf 

    
def quickplot (arr, dl =10): 
    """Quick plot to see the anomaly"""
    
    plt.plot(np.arange(0, len(arr) * dl, dl), arr , ls ='-', c='k')
    plt.show() 
    
    

def _magnitude (cz:Sub[Array[float, DType[float]]] ) -> float: 
    """ Compute the magnitude of selected conductive zone. 
    
    The magnitude parameter is the absolute resistivity value between
    the minimum :math:`\rho_(a_min\ )\` and maximum :math:`\rho_(a_max\ )` 
    value of selected anomaly:
    
    .. math::
    
        magnitude=|\begin\rho_a〗_min-ρ_(a_max ) |

    :param cz: array-like. Array of apparent resistivity values composing 
        the conductive zone. 
    
    :return: Absolute value of anomaly magnitude in ohm.meters.
    """
    return np.abs (cz.max()- cz.min()) 

def _power (p:Sub[SP[Array, DType [int]]] | List[int] ) -> float : 
    """ Compute the power of the selected conductive zone. Anomaly `power` 
    is closely referred to the width of the conductive zone.
    
    The power parameter implicitly defines the width of the conductive zone
    and is evaluated from the difference between the abscissa 
    :math:`\begin(X〗_LB)` and the end :math:`\left(X_{UB}\right)` points of 
    the selected anomaly:
    
    .. math::
        
        power=|X_LB-X_UB\ |
    
    :param p: array-like. Station position of conductive zone.
    
    :return: Absolute value of the width of conductive zone in meters. 
    """
    return np.abs(p.min()- p.max()) 


def _find_cz_bound_indexes (
    erp: Union[Array[float, DType[float]], List[float], pd.Series],
    cz: Union [Sub[Array], List[float]] 
)-> Tuple[int, int]: 
    """ Fetch the limits 'LB' and 'UB' of the selected conductive zone.
    
    Indeed the 'LB' and 'UB' fit the lower and upper boundaries of the 
    conductive zone respectively. 
    
    :param erp: array-like. Apparent resistivities collected during the survey. 
    :param cz: array-like. Array of apparent resistivies composing the  
        conductive zone. 
    
    :return: The index of boundaries 'LB' and 'UB'. 
    
    .. note::`cz` must be self-containing of `erp`. If ``False`` should  
            raise and error. 
    """
    # assert whether cz is a subset of erp. 
    if isinstance( erp, pd.Series): erp = erp.values 

    if not np.isin(True,  (np.isin (erp, cz))):
        raise ValueError ('Expected the conductive zone array being a '
                          'subset of the resistivity array')
    # find the indexes using np.argwhere  
    cz_indexes = np.argwhere(np.isin(erp, cz)).ravel()
    
    return cz_indexes [0] , cz_indexes [-1] 


def convert_distance_to_m(
        value:T ,
        converter:float =1e3,
        unit:str ='km'
)-> float: 
    """ Convert distance from `km` to `m` or vice versa even a string 
    value is given.
    
    :param value: value to convert. 
    "paramm converter: Equivalent if given in 'km' rather than 'meters'.
    :param unit: unit to convert to."""
    if isinstance(value, str): 
        try:
            value = float(value.replace(unit, '')
                              )*converter if value.find(
                'km')>=0 else float(value.replace('m', ''))
        except: 
            raise TypeError(f"Expected float not {type(value)!r}."
               )
            
    return value
    
    
def get_station_number (
        dipole:float,
        distance:float , 
        from0:bool = False,
        **kws
)-> float: 
    """ Get the station number from dipole length and 
    the distance to the station.
    
    :param distance: Is the distance from the first station to `s` in 
        meter (m). If value is given, please specify the dipole length in 
        the same unit as `distance`.
    :param dipole: Is the distance of the dipole measurement. 
        By default the dipole length is in meter.
    :param kws: :func:`convert_distance_to_m` additional arguments
    
    """
    dipole=convert_distance_to_m(dipole, **kws)
    distance =convert_distance_to_m(distance, **kws)

    return  distance/dipole  if from0 else distance/dipole + 1 

# @deprecated('Deprecated function. Replaced by '
#             '`:func: ~watex.utils.coreutils._define_conductive_zone`'
#             'more efficient.')
def define_conductive_zone (
        erp: Array | List[float],
        stn: Optional [int] = None,
        sres:Optional [float] = None,
        *, 
        distance:float | None = None , 
        dipole_length:float | None = None,
        extent:int =7): 
    """ Detect the conductive zone from `s`ves point.
    
    :param erp: Resistivity values of electrical resistivity profiling(ERP)
    :param stn: Station number expected for VES and/or drilling location.
    :param sres: Resistivity value at station number `stn`. 
                If `sres` is given, the auto-search will be triggered to 
                find the station number that fits the resistivity value. 
            
    :param distance: Distance from the first station to `stn`. If given, 
                    be sure to provide the `dipole_length`
    :param dipole_length: Length of the dipole. Comonly the distante between 
                two close stations. Since we use config AB/2 
    :param extent: Is the width to depict the anomaly. If provide, need to be 
                consistent along all ERP line. Should keep unchanged for other 
                parameters definitions. Default is ``7``.
    :returns: 
        - CZ:Conductive zone including the station position 
        - sres: Resistivity value of the station number
        - ix_stn: Station position in the CZ
            
    .. note:: 
        If many stations got the same `sres` value, the first station 
        is flagged. This may not correspond to the station number that is 
        searching. Use `sres` only if you are sure that the 
        resistivity value is unique on the whole ERP. Otherwise it's 
        not recommended.
        
    :Example: 
        
        >>> import numpy as np
        >>> from watex.utils.exmath import define_conductive_zone 
        >>> sample = np.random.randn(9)
        >>> cz, stn_res = define_conductive_zone(sample, 4, extent = 7)
        ... (array([ 0.32208638,  1.48349508,  0.6871188 , -0.96007639,
                    -1.08735204,0.79811492, -0.31216716]),
             -0.9600763919368086, 
             3)
    """
    try : 
        iter(erp)
    except : raise Wex.WATexError_inputarguments(
            f'`erp` must be a sequence of values not {type(erp)!r}')
    finally: erp = np.array(erp)
  
    # check the distance 
    if stn is None: 
        if (dipole_length and distance) is not None: 
            stn = get_station_number(dipole_length, distance)
        elif sres is not None: 
            snix, = np.where(erp==sres)
            if len(snix)==0: 
                raise Wex.WATexError_parameter_number(
                    "Could not  find the resistivity value of the VES "
                    "station. Please provide the right value instead.") 
                
            elif len(snix)==2: 
                stn = int(snix[0]) + 1
        else :
            raise Wex.WATexError_inputarguments(
                '`stn` is needed or at least provide the survey '
                'dipole length and the distance from the first '
                'station to the VES station. ')
            
    if erp.size < stn : 
        raise Wex.WATexError_parameter_number(
            f"Wrong station number =`{stn}`. Is larger than the "
            f" number of ERP stations = `{erp.size}` ")
    
    # now defined the anomaly boundaries from sn
    stn =  1 if stn == 0 else stn  
    stn -=1 # start counting from 0.
    if extent %2 ==0: 
        if len(erp[:stn]) > len(erp[stn:])-1:
           ub = erp[stn:][:extent//2 +1]
           lb = erp[:stn][len(ub)-int(extent):]
        elif len(erp[:stn]) < len(erp[stn:])-1:
            lb = erp[:stn][stn-extent//2 +1:stn]
            ub= erp[stn:][:int(extent)- len(lb)]
     
    else : 
        lb = erp[:stn][-extent//2:] 
        ub = erp[stn:][:int(extent//2)+ 1]
    
    # read this part if extent anomaly is not reached
    if len(ub) +len(lb) < extent: 
        if len(erp[:stn]) > len(erp[stn:])-1:
            add = abs(len(ub)-len(lb)) # remain value to add 
            lb = erp[:stn][-add -len(lb) - 1:]
        elif len(erp[:stn]) < len(erp[stn:])-1:
            add = abs(len(ub)-len(lb)) # remain value to add 
            ub = erp[stn:][:len(ub)+ add -1] 
          
    conductive_zone = np.concatenate((lb, ub))
    # get the index of station number from the conductive zone.
    ix_stn, = np.where (conductive_zone == conductive_zone[stn])
    ix_stn = int(ix_stn[0]) if len(ix_stn)> 1 else  int(ix_stn)
    
    return  conductive_zone, conductive_zone[stn], ix_stn 
    

def W (cz, stn_pos=None ): 
    """Validate the shape `w`"""
    # get anomaly boundaries 
    # anomaly M: 
    # UB and LB  > than Lmin > 1 and exists  Lmax >1 at least 
    
    lb , ub = cz [0], cz[-1]
    
    lmin, = argrelextrema(cz, np.less)
    lmax, = argrelextrema(cz, np.greater)
               
    return lmin, lmax 
    # try: 

    #     minlocals_ix, = argrelextrema(rhoa_range, np.less)
    # except : 
 
    #     minlocals_ix = argrelextrema(rhoa_range, np.less)
    # try : 

    #     maxlocals_ix, = argrelextrema(rhoa_range, np.greater)
    # except : maxlocals_ix = argrelextrema(rhoa_range, np.greater)

#FR0: CED9EF
#FR1: 9EB3DD
#FR2: 9EB3DD
#FR3: 0A4CEE
def shortPlot (sample, cz=None): 
    """ Quick plot to visualize the `sample` line as well as the  selected 
    conductive zone if given.
    
    :param sample: array_like, the electrical profiling array 
    :param cz: array_like, the selected conductive zone. If ``None``, `cz` 
        should be plotted.
    
    :Example: 
        >>> import numpy as np 
        >>> from watex.utils.exmath import shortPlot, define_conductive_zone 
        >>> test_array = np.random.randn (10)
        >>> selected_cz ,*_ = define_conductive_zone(test_array, 7) 
        >>> shortPlot(test_array, selected_cz )
        
    """
    import matplotlib.pyplot as plt 
    fig, ax = plt.subplots(1,1, figsize =(10, 4))
    leg =[]
    ax.scatter (np.arange(len(sample)), sample, marker ='.', c='b')
    zl, = ax.plot(np.arange(len(sample)), sample, 
                  c='r', 
                  label ='Electrical resistivity profiling')
    leg.append(zl)
    if cz is not None: 
        # construct a mask array with np.isin to check whether 
        # `cz` is subset array
        z = np.ma.masked_values (sample, np.isin(sample, cz ))
        # a masked value is constructed so we need 
        # to get the attribute fill_value as a mask 
        # However, we need to use np.invert or tilde operator  
        # to specify that other value except the `CZ` values mus be 
        # masked. Note that the dtype must be changed to boolean
        sample_masked = np.ma.array(
            sample, mask = ~z.fill_value.astype('bool') )
    
        czl, = ax.plot(
            np.arange(len(sample)), sample_masked, 
            ls='-',
            c='#0A4CEE',
            lw =2, 
            label ='Conductive zone')
        leg.append(czl)

    ax.set_xticks(range(len(sample)))
    ax.set_xticklabels(
        ['S{0:02}'.format(i+1) for i in range(len(sample))])
    
    ax.set_xlabel('Stations')
    ax.set_ylabel('app.resistivity (ohm.m)')
    ax.legend( handles = leg, 
              loc ='best')
        
    plt.show()
    

def compute_sfi (pk_min, pk_max, rhoa_min,
                 rhoa_max,  rhoa, pk)  : 
    """
    SFI is introduced to evaluate the ratio of presumed existing fracture
    from anomaly extent. We use a similar approach as IF computation
    proposed by Dieng et al. (2004) to evaluate each selected anomaly 
    extent and the normal distribution of resistivity values along the 
    survey line. The SFI threshold is set at :math:`$\sqrt\2$`  for 
    symmetrical anomaly characterized by a perfect distribution of 
    resistivity in a homogenous medium. 
    
    :param pk_min: see :doc:`compute_power` 
    :param pk_max: see :doc:`compute_power` 
    
    :param rhoa_max: see :doc:`compute_magnitude` 
    :param rhoa_min: see :doc:`compute_manitude`
    
    :param pk: 
        
        Station position of the selected anomaly in ``float`` value. 
        
    :param rhoa: 
        
        Selected anomaly apparent resistivity value in ohm.m 
        
    :return: standard fracture index (SFI)
    :rtype: float 
    
    :Example: 
        
        >>> from watex.utils.exmath import compute_sfi 
        >>> sfi = compute_sfi(pk_min = 90,
        ...                      pk_max=130,
        ...                      rhoa_min=175,
        ...                      rhoa_max=170,
        ...                      rhoa=132,
        ...                      pk=110)
        >>> sfi
    
    """  
    def deprecated_sfi_computation () : 
        """ Deprecated way for `sfi` computation"""
        try : 
            if  pk_min -pk  < pk_max - pk  : 
                sfi= np.sqrt((((rhoa_max -rhoa) / 
                                  (rhoa_min- rhoa)) **2 + 
                                 ((pk_max - pk)/(pk_min -pk))**2 ))
            elif pk_max -pk  < pk_min - pk : 
                sfi= np.sqrt((((rhoa_max -rhoa) / 
                                  (rhoa_min- rhoa)) **2 + 
                                 ((pk_min - pk)/(pk_max -pk))**2 ))
        except : 
            if sfi ==np.nan : 
                sfi = - np.sqrt(2)
            else :
                sfi = - np.sqrt(2)
       
    try : 
        
        if (rhoa == rhoa_min and pk == pk_min) or\
            (rhoa==rhoa_max and pk == pk_max): 
            ma= max([rhoa_min, rhoa_max])
            ma_star = min([rhoa_min, rhoa_max])
            pa= max([pk_min, pk_max])
            pa_star = min([pk_min, pk_max])
    
        else : 
       
            if  rhoa_min >= rhoa_max : 
                max_rho = rhoa_min
                min_rho = rhoa_max 
            elif rhoa_min < rhoa_max: 
                max_rho = rhoa_max 
                min_rho = rhoa_min 
            
            ma_star = abs(min_rho - rhoa)
            ma = abs(max_rho- rhoa )
            
            ratio = ma_star / ma 
            pa = abs(pk_min - pk_max)
            pa_star = ratio *pa
            
        sfi = np.sqrt((pa_star/ pa)**2 + (ma_star/ma)**2)
        
        if sfi ==np.nan : 
                sfi = - np.sqrt(2)
    except : 

        sfi = - np.sqrt(2)
  
    
    return sfi
    
def compute_anr (sfi , rhoa_array, pos_bound_indexes):
    """
    Compute the select anomaly ratio (ANR) along with the
    whole profile from SFI. The standardized resistivity values
    `rhoa`  of is averaged from   X_begin to  X_end .
    The ANR is a positive value. 
    
    :param sfi: 
        
        Is standard fracturation index. please refer to :doc: `compute_sfi`
        
    :param rhoa_array: Resistivity values of :ref:`erp` line 
    :type rhoa_array: array_like 
    
    :param pos_bound_indexes: 
        
        Select anomaly station location boundaries indexes. Refer to 
        :doc:`compute_power` of ``pos_bounds``. 
        
    :return: Anomaly ratio 
    :rtype:float 
    
    :Example: 
        
        >>> from watex.utils.exmath import compute_anr 
        >>> import pandas as pd
        >>> anr = compute_anr(sfi=sfi, 
        ...                  rhoa_array=data = pd.read_excel(
        ...                  'data/l10_gbalo.xlsx').to_numpy()[:, -1],
        ...              pk_bound_indexes  = [9, 13])
        >>> anr
    """
    stand = (rhoa_array - rhoa_array.mean())/np.std(rhoa_array)
    try: 
        stand_rhoa =stand[int(min(pos_bound_indexes)): 
                          int(max(pos_bound_indexes))+1]
    except: 
        stand_rhoa = np.array([0])
        
    return sfi * np.abs(stand_rhoa.mean())


@deprecated('Deprecated function to `:func:`watex.core.erp.get_type`'
            ' more efficient using median and index computation. It will '
            'probably deprecate soon for neural network pattern recognition.')
def get_type (erp_array, posMinMax, pk, pos_array, dl): 
    """
    Find anomaly type from app. resistivity values and positions locations 
    
    :param erp_array: App.resistivty values of all `erp` lines 
    :type erp_array: array_like 
    
    :param posMinMax: Selected anomaly positions from startpoint and endpoint 
    :type posMinMax: list or tuple or nd.array(1,2)
    
    :param pk: Position of selected anomaly in meters 
    :type pk: float or int 
    
    :param pos_array: Stations locations or measurements positions 
    :type pos_array: array_like 
    
    :param dl: 
        
        Distance between two receiver electrodes measurement. The same 
        as dipole length in meters. 
    
    :returns: 
        - ``EC`` for Extensive conductive. 
        - ``NC`` for narrow conductive. 
        - ``CP`` for conductive plane 
        - ``CB2P`` for contact between two planes. 
        
    :Example: 
        
        >>> from watex.utils.exmath import get_type 
        >>> x = [60, 61, 62, 63, 68, 65, 80,  90, 100, 80, 100, 80]
        >>> pos= np.arange(0, len(x)*10, 10)
        >>> ano_type= get_type(erp_array= np.array(x),
        ...            posMinMax=(10,90), pk=50, pos_array=pos, dl=10)
        >>> ano_type
        ...CB2P

    """
    # Get position index 
    anom_type ='CP'
    index_pos = int(np.where(pos_array ==pk)[0])
    # if erp_array [:index_pos +1].mean() < np.median(erp_array) or\
    #     erp_array[index_pos:].mean() < np.median(erp_array) : 
    #         anom_type ='CB2P'
    if erp_array [:index_pos+1].mean() < np.median(erp_array) and \
        erp_array[index_pos:].mean() < np.median(erp_array) : 
            anom_type ='CB2P'
            
    elif erp_array [:index_pos +1].mean() >= np.median(erp_array) and \
        erp_array[index_pos:].mean() >= np.median(erp_array) : 
                
        if  dl <= (max(posMinMax)- min(posMinMax)) <= 5* dl: 
            anom_type = 'NC'

        elif (max(posMinMax)- min(posMinMax))> 5 *dl: 
            anom_type = 'EC'

    return anom_type   
    
@deprecated('`Deprecated function. Replaced by :meth:~core.erp.get_shape` ' 
            'more convenient to recognize anomaly shape using ``median line``'
            'rather than ``mean line`` below.')   
def get_shape (rhoa_range): 
    """
    Find anomaly `shape`  from apparent resistivity values framed to
    the best points using the mean line. 
 
    :param rhoa_range: The apparent resistivity from selected anomaly bounds
                        :attr:`~core.erp.ERP.anom_boundaries`
    :type rhoa_range: array_like or list 
    
    :returns: 
        - V
        - W
        - K 
        - C
        - M
        - U
    
    :Example: 
        
        >>> from watex.utils.exmath import get_shape 
        >>> x = [60, 70, 65, 40, 30, 31, 34, 40, 38, 50, 61, 90]
        >>> shape = get_shape (rhoa_range= np.array(x))
        ... U

    """
    minlocals = argrelextrema(rhoa_range, np.less)
    shape ='V'
    average_curve = rhoa_range.mean()
    if len (minlocals[0]) >1 : 
        shape ='W'
        average_curve = rhoa_range.mean()
        minlocals_slices = rhoa_range[
            int(minlocals[0][0]):int(minlocals[0][-1])+1]
        average_minlocals_slices  = minlocals_slices .mean()

        if average_curve >= 1.2 * average_minlocals_slices: 
            shape = 'U'
            if rhoa_range [-1] < average_curve and\
                rhoa_range [-1]> minlocals_slices[
                    int(argrelextrema(minlocals_slices, np.greater)[0][0])]: 
                shape ='K'
        elif rhoa_range [0] < average_curve and \
            rhoa_range [-1] < average_curve :
            shape ='M'
    elif len (minlocals[0]) ==1 : 
        if rhoa_range [0] < average_curve and \
            rhoa_range [-1] < average_curve :
            shape ='M'
        elif rhoa_range [-1] <= average_curve : 
            shape = 'C'
            
    return shape 



def compute_power (posMinMax=None, pk_min=None , pk_max=None, ):
    """ 
    Compute the power Pa of anomaly.
    
    :param pk_min: 
        Min boundary value of anomaly. `pk_min` is min value (lower) 
        of measurement point. It's the position of the site in meter. 
    :type pk_min: float 
    
    :param pk_max: 
        Max boundary of the select anomaly. `pk_max` is the maximum value 
        the measurement point in meter. It's  the upper boundary position of 
        the anomaly in the site in m. 
    :type pk_max: float 
    
    :return: The absolute value between the `pk_min` and `pk_max`. 
    :rtype: float 
    
    :Example: 
        
        >>> from wmathandtricks import compute_power 
        >>> power= compute_power(80, 130)
    
    
    """
    if posMinMax is not None: 
        pk_min = np.array(posMinMax).min()     
        pk_max= np.array(posMinMax).max()
    
    if posMinMax is None and (pk_min is None or pk_max is None) : 
        raise Wex.WATexError_parameter_number(
            'Could not compute the anomaly power. Provide at least'
             'the anomaly position boundaries or the left(`pk_min`) '
             'and the right(`pk_max`) boundaries.')
    
    return np.abs(pk_max - pk_min)
    
def compute_magnitude(rhoa_max=None , rhoa_min=None, rhoaMinMax=None):
    """
    Compute the magnitude ``Ma`` of  selected anomaly expressed in Ω.m.
    ano
    :param rhoa_min: resistivity value of selected anomaly 
    :type rhoa_min: float 
    
    :param rhoa_max: Max boundary of the resistivity value of select anomaly. 
    :type rhoa_max: float 
    
    :return: The absolute value between the `rhoa_min` and `rhoa_max`. 
    :rtype: float 
    
    :Example: 
        
        >>> from watex.utils.exmath import compute_power 
        >>> power= compute_power(80, 130)
    
    """
    if rhoaMinMax is not None : 
        rhoa_min = np.array(rhoaMinMax).min()     
        rhoa_max= np.array(rhoaMinMax).max()
        
    if rhoaMinMax is None and (rhoa_min  is None or rhoa_min is None) : 
        raise Wex.WATexError_parameter_number(
            'Could not compute the anomaly magnitude. Provide at least'
            'the anomaly resistivy value boundaries or the buttom(`rhoa_min`)'
             'and the top(`rhoa_max`) boundaries.')

    return np.abs(rhoa_max -rhoa_min)

"""
.. _Dieng et al: http://documents.irevues.inist.fr/bitstream/handle/2042/36362/2IE_2004_12_21.pdf?sequence=1

.. _Matplotlib scatter: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.scatter.html

.. _Matplotlib plot: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.plot.html

.. |VES| replace: Vertical Electrical Sounding 

.. |ERP| replace: Electrical resistivity profiling 

"""

