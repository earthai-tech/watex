# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Sep 17 11:25:15 2021
# This module is a WATex-AI calculator released under MIT Licence 
"""
Created on Fri Sep 17 11:25:15 2021

@author: @Daniel03
"""

import numpy as np
import pandas as pd 
 
from scipy.signal import argrelextrema 
from ..utils.decorator import deprecated 
from ..utils._watexlog import watexlog
from ..utils import exceptions as Wex 
from .._typing import (
    Array, 
    T, 
    List, 
    Tuple, 
    
)

_logger =watexlog.get_watex_logger(__name__)


def _magnitude (cz:List[float] | Array) -> float: 
    """ Compute the magnitude of selected conductive zone. 
    
    :param cz: array-like. Array of apparent resistivity values composing 
        the conductive zone. 
    
    :return: Absolute value of anomaly magnitude.
    """
    return np.abs (cz.max()- cz.min()) 

def _power (czp:List[int] | Array ) -> float : 
    """ Compute the power of the selected conductive zone. 
    
    Anomaly `power` is closely referred to the width of the conductive zone.
    
    :param czp: array-like. Array  station position of conductive zones.
    
    :return: Absolute value of the width of conductive zone. 
    """
    return np.abs(czp.min()- czp.max()) 


def _find_cz_bound_indexes (
        erp: List[float |int] | pd.Series |Array ,
        cz: List[float|int] 
)-> Tuple[int, int]: 
    """ Fetch the limits 'LB' and 'UB' of the selected conductive zone.
    
    Indeed the 'LB' and 'UB' fit the lower and upper boundaries of the 
    conductive zone respectively. 
    
    :param erp: array-like. Apparent resistivities collected during the survey. 
    :param cz: array-like. Array of apparent resistivies composing the  
        conductive zone. 
    
    :return: The index of boundaries  'LB' and 'UB' 
    
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
        dipole:float | int ,
        distance:float | int, 
        from0:bool =False,
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

@deprecated('Deprecated function. Replaced by '
            '`:func: ~watex.utils.coreutils._define_conductive_zone`'
            'more efficient.')
def define_conductive_zone (
        erp:Array | List[float],
        stn: int |None  =None,
        sres:float | None =None,
        distance:float | None =None , 
        dipole_length:float | None =None,
        *, 
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



