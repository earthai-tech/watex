# -*- coding: utf-8 -*-
"""
===============================================================================
Copyright (c) 2021 Kouadio K. Laurent

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
===============================================================================

.. synopsis:: 'watex.utils.wmathandtricks'
            Module for computing

Created on Mon Jun 21 14:43:25 2021

.. _electrical-resistivity-profile::`erp`

.. _vertical-electrical-sounding::`ves`

.. _station-position::`pk`

.._anomaly-boundaries:`anBounds`

@author: @Daniel03

"""

import os 
import warnings
import numpy as np 
import pandas as pd
from scipy.signal import argrelextrema 
from scipy.interpolate import interp1d as sp1d
import watex.utils.exceptions as Wex
from watex.utils._watexlog import watexlog 
from watex.utils.decorator import deprecated  

_logger =watexlog.get_watex_logger(__name__)



def compute_lower_anomaly(erp_array, station_position=None, 
                          step=None, **kws): 
    """
    Function to get the minimum value on the ERP array. 
    If `pk` is provided wil give the index of pk
    
    :param erp_array: array of apparent resistivity profile 
    :type erp_array: array_like
    
    :param station position: array of station position (survey) , if not given 
                    and `step` is known , set the step value and 
                    `station_position` will compute automatically 
    :type station_position: array_like 
    
    :param step: The distance between measurement im meter. If given will 
        recompute the `station_position`
    
    :returns: * `bestSelectedDict`: dict containing best anomalies  
                with the anomaly resistivities range.
              * `anpks`: Main positions of best select anomaly 
              * `collectanlyBounds`: list of arrays of select anomaly values
              * `min_pks`: list of tuples (pk, 
                                           minVal of best anomalies points.)
    :rtype: tuple 
    
    :Example: 
        
        >>> from watex.utils.wmathandtricks import compute_lower_anolamy 
        >>> import pandas as pd 
        >>> path_to_= 'data/l10_gbalo.xlsx'
        >>> dataRes=pd.read_excel(erp_data).to_numpy()[:,-1]
        >>> anomaly, *_ =  compute_lower_anomaly(erp_array=data, step =10)
        >>> anomaly
                
    """
    display_infos= kws.pop('diplay_infos', False)
    # got minumum of erp data 
    collectanlyBounds=[]
    if step is not None: 
        station_position = np.arange(0, step * len(erp_array), step)

    min_pks= get_minVal(erp_array) # three min anomaly values 

    # compute new_pjk 
    # find differents anomlies boundaries 
    for ii, (rho, index) in enumerate(min_pks) :
        _, _, anlyBounds= drawn_anomaly_boundaries(erp_data = erp_array,
                                 appRes = rho, index=index)
        
        collectanlyBounds.append(anlyBounds)

    if station_position is None :
        pks =np.array(['?' for ii in range(len(erp_array))])
    else : pks =station_position

    if pks.dtype in ['int', 'float']: 
        anpks =np.array([pks[skanIndex ] for
                         (_, skanIndex) in min_pks ])
    else : anpks ='?'
    
    bestSelectedDICT={}
    for ii, (pk, anb) in enumerate(zip(anpks, collectanlyBounds)): 
        bestSelectedDICT['{0}_pk{1}'.format(ii+1, pk)] = anb
    
    if display_infos:
        print('{0:+^100}'.format(
            ' *Best Conductive anomaly points (BCPts)* '))
        fmtAnText(anFeatures=bestSelectedDICT)
    
    return bestSelectedDICT, anpks, collectanlyBounds, min_pks
        

def get_minVal(array): 
    """
    Function to find the three minimum values on array and their 
    corresponding indexes 
    
    :param array: array  of values 
    :type array: array_like 
    
    :returns: Three minimum values of rho, index in rho_array
    :rtype: tuple
    
    """

    holdList =[]
    if not isinstance(array, (list, tuple, np.ndarray)):
        if isinstance(array, float): 
            array=np.array([array])
        else : 
            try : 
                array =np.array([float(array)])
            except: 
                raise Wex.WATexError_float('Could not convert %s to float!')
    try : 
        # first option:find minimals locals values 
        minlocals = argrelextrema(array, np.less)[0]
        temp_array =np.array([array[int(index)] for index in minlocals])
        if len(minlocals) ==0: 
            ix = np.where(array ==array.min())
            if len(ix)>1: 
                ix =ix[0]
            temp_array = array[int(ix)]
            
    except : 
        # second option: use archaic computation.
        temp_array =np.sort(array)
    else : 
        temp_array= np.sort(temp_array)
        
    ss=0

    for ii, tem_ar in enumerate(temp_array) : 
        if ss >=3 : 
            holdList=holdList[:3]
            break 
        min_index = np.where(array==tem_ar)[0]
  
        if len(min_index)==1 : 
            holdList.append((array[int(min_index)], 
                             int(min_index)))
            ss +=ii
        elif len(min_index) > 1 :
            # loop the index array and find the min for consistency 
            for jj, indx in enumerate(min_index):  
                holdList.append((array[int(indx)], 
                                 int(indx)))
        ss =len(holdList)
        
    # for consistency keep the 3 best min values 
    if len(holdList)>3 : 
        holdList = holdList[:3]

    return holdList 

def drawn_anomaly_boundaries(erp_data, appRes, index):
    """
    Function to drawn anomaly boundary 
    and return the anomaly with its boundaries
    
    :param erp_data: erp profile 
    :type erp_data: array_like or list 
    
    :param appRes: resistivity value of minimum pk anomaly 
    :type appRes: float 
    
    :param index: index of minimum pk anomaly 
    :type index: int 
    
    :return: anomaly boundary 
    :rtype: list of array_like 

    """
    f = 0 # flag to mention which part must be calculated 
    if index ==0 : 
        f = 1 # compute only right part 
    elif appRes ==erp_data[-1]: 
        f=2 # compute left part 
    
    def loop_sideBound(term):
        """
        loop side bar from anomaly and find the term side 
        
        :param term: is array of left or right side of anomaly.
        :type term: array 
        
        :return: side bar 
        :type: array_like 
        """
        tem_drawn =[]
        maxT=0 

        for ii, tem_rho in enumerate(term) : 

            diffRes_betw_2pts= tem_rho - appRes 
            if diffRes_betw_2pts > maxT : 
                maxT = diffRes_betw_2pts
                tem_drawn.append(tem_rho)
            elif diffRes_betw_2pts < maxT : 
                # rho_limit = tem_rho 
                break 
        return np.array(tem_drawn)
    # first broke erp profile from the anomalies 
    if f ==0 or f==2 : 
        left_term = erp_data[:index][::-1] # flip left term  for looping
        # flip again to keep the order 
        left_limit = loop_sideBound(term=left_term)[::-1] 

    if f==0 or f ==1 : 
        right_term= erp_data[index :]
        right_limit=loop_sideBound(right_term)
    # concat right and left to get the complete anomaly 
    if f==2: 
        anomalyBounds = np.append(left_limit,appRes)
                                   
    elif f ==1 : 
        anomalyBounds = np.array([appRes]+ right_limit.tolist())
    else: 
        left_limit = np.append(left_limit, appRes)
        anomalyBounds = np.concatenate((left_limit, right_limit))
    
    return appRes, index, anomalyBounds 

def defineAnomaly(erp_data, station_position=None, pks=None, 
                  dipole_length=10., **kwargs):
    """
    Function will select the different anomalies. If pk is not given, 
    the best three anomalies on the survey lines will be
    computed automatically
    
    :param erp_data: Electrical resistivity profiling 
    :type erp_data: array_like 
    
    :param pks: station positions anomaly boundaries (pk_begin, pk_end)
                If selected anomalies is more than one, set `pks` into dict
                where number of anomaly =keys and pks = values 
    :type pks: list or dict
    
    :param dipole_length: Distance between two measurements in meters
                        Change the `dipole lengh
    :type dipole_length: float
    
    :param station_position: station position array 
    :type statiion_position: array_like 
    
    :return: list of anomalies bounds 
    
    """
    selectedPk =kwargs.pop('selectedPk', None)
    bestSelectedDICT={}
    if station_position is not None : 
        dipole_length = (station_position.max()-
               station_position.min())/(len(station_position -1))
    if station_position is None : 
        station_position =np.arange(0, dipole_length * len(erp_data), 
                                    dipole_length)
                                        
  
    def getBound(pksbounds): 
        """
        Get the bound from the given `pks`
        :param pksbounds: Anomaly boundaries
        :type pksbounds: list of array_like 
        
        :returns: * anomBounds- array of appRes values of anomaly
        :rtype: array_like 
        """
        # check if bound is on station positions
        for spk in pksbounds : 
            if not pksbounds.min() <= spk <= pksbounds.max(): 
                raise Wex.WATexError_AnomalyBounds(
                    'Bound <{0}> provided is out of range !'
                   'Dipole length is set to = {1} m.'
                   ' Please set a new bounds.')
            
        pkinf = np.where(station_position==pksbounds.min())[0]
        pksup = np.where(station_position==pksbounds.max())[0]
        anomBounds = erp_data[int(pkinf):int(pksup)+1]
        return anomBounds
    
    if pks is None : 
        bestSelectedDICT, *_= compute_lower_anomaly(
            erp_array=erp_data, step=dipole_length, 
            station_position =station_position)
        
    elif isinstance(pks, list):
        pks =np.array(sorted(pks))
        collectanlyBounds = getBound(pksbounds= pks)
        # get the length of selected anomalies and computed the station 
        # location wich composed the bounds (Xbegin and Xend)
        pkb, *_= find_pk_from_selectedAn(
            an_res_range=collectanlyBounds, pos=pks, 
            selectedPk=selectedPk)
        bestSelectedDICT={ '1_{}'.format(pkb):collectanlyBounds}

    elif isinstance(pks, dict):
        for ii, (keys, values) in enumerate(pks.items()):
            if isinstance(values, list): 
                values =np.array(values)
            collectanlyBounds=  getBound(pksbounds=values) 
            pkb, *_= find_pk_from_selectedAn(
            an_res_range=collectanlyBounds, pos=pks, 
            selectedPk=selectedPk)
            bestSelectedDICT['{0}_{1}'.format(ii+1, pkb)]=collectanlyBounds
           
    return bestSelectedDICT
       
def find_pk_from_selectedAn(an_res_range,  pos=None, selectedPk=None): 
    """
    Function to select the main :ref:`pk` from both :ref:`anBounds`. 
    
    :paran an_res_range: anomaly resistivity range on :ref:`erp` line. 
    :type an_res_range: array_like 
    
    :param pos: position of anomaly boundaries (inf and sup):
                anBounds = [90, 130]
                - 130 is max boundary and 90 the  min boundary 
    :type pos: list 
    
    :param selectedPk: 
        
        User can set its own position of the right anomaly. Be sure that 
        the value provided is right position . 
        Could not compute again provided that `pos`
        is not `None`.
                
    :return: anomaly station position. 
    :rtype: str 'pk{position}'
    
    :Example:
        
        >>> from watex.utils.wmathandtricks import find_pk_from_selectedAn
        >>> resan = np.array([168,130, 93,146,145])
        >>> pk= find_pk_from_selectedAn(
        ...    resan, pos=[90, 13], selectedPk= 'str20')
        >>> pk
    
    
    """
    #compute dipole length from pos
    if pos is not None : 
        if isinstance(pos, list): 
            pos =np.array(pos)
    if pos is None and selectedPk is None : 
        raise Wex.WATexError_parameter_number(
            'Give at least the anomaly boundaries'
            ' before computing the selected anomaly position.')
        
    if selectedPk is not None :  # mean is given
        if isinstance(selectedPk, str):
            if selectedPk.isdigit() :
                sPk= int(selectedPk)
            elif selectedPk.isalnum(): 
                oss = ''.join([s for s in selectedPk
                    if s.isdigit()])
                sPk =int(oss)
        else : 
            try : 
                sPk = int(selectedPk)
                
            except : pass 

        if pos is not None : # then compare the sPk and ps value
            try :
                if not pos.min()<= sPk<=pos.max(): 
                    warnings.warn('Wrong position given <{}>.'
                                  ' Should compute new positions.'.
                                  format(selectedPk))
                    _logger.debug('Wrong position given <{}>.'
                                  'Should compute new positions.'.
                                  format(selectedPk))
                    
            except UnboundLocalError:
                print("local variable 'sPk' referenced before assignment")
            else : 
                
                return 'pk{}'.format(sPk ), an_res_range
                
            
        else : 
            selectedPk='pk{}'.format(sPk )
            
            return selectedPk , an_res_range
    

    if isinstance(pos, list):
        pos =np.array(pos)  
    if isinstance(an_res_range, list): 
        an_res_range =np.array(an_res_range)
    dipole_length = (pos.max()-pos.min())/(len(an_res_range)-1)

    tem_loc = np.arange(pos.min(), pos.max()+dipole_length, dipole_length)
    
    # find min value of  collected anomalies values 
    locmin = np.where (an_res_range==an_res_range.min())[0]
    if len(locmin) >1 : locmin =locmin[0]
    pk_= int(tem_loc[int(locmin)]) # find the min pk 

    selectedPk='pk{}'.format(pk_)
    
    return selectedPk , an_res_range
 
def fmtAnText(anFeatures=None, title=['Ranking', 'rho(Ω.m)', 
                                    'position pk(m)',
                                    'rho range(Ω.m)'],
                                    **kwargs) :
    """
    Function format text from anomaly features 
    
    :param anFeatures: Anomaly features 
    :type anFeatures: list or dict
    
    :param title: head lines 
    :type title: list
    
    :Example: 
        
        >>> from watex.utils.wmathandtricks import fmtAnText
        >>> fmtAnText(anFeatures =[1,130, 93,(146,145, 125)])
    
    """
    inline =kwargs.pop('inline', '-')
    mlabel =kwargs.pop('mlabels', 100)
    line = inline * int(mlabel)
    
    #--------------------header ----------------------------------------
    print(line)
    tem_head ='|'.join(['{:^15}'.format(i) for i in title[:-1]])
    tem_head +='|{:^45}'.format(title[-1])
    print(tem_head)
    print(line)
    #-----------------------end header----------------------------------
    newF =[]
    if isinstance(anFeatures, dict):
        for keys, items in anFeatures.items(): 
            rrpos=keys.replace('_pk', '')
            rank=rrpos[0]
            pos =rrpos[1:]
            newF.append([rank, min(items), pos, items])
            
    elif isinstance(anFeatures, list): 
        newF =[anFeatures]
    
    
    for anFeatures in newF: 
        strfeatures ='|'.join(['{:^15}'.format(str(i)) \
                               for i in anFeatures[:-1]])
        try : 
            iter(anFeatures[-1])
        except : 
            strfeatures +='|{:^45}'.format(str(anFeatures[-1]))
        else : 
            strfeatures += '|{:^45}'.format(
                ''.join(['{} '.format(str(i)) for i in anFeatures[-1]]))
            
        print(strfeatures)
        print(line)
    
def compute_power (posMinMax=None, pk_min=None , pk_max=None, ):
    """ 
    Compute the power Pa of anomaly.
    
    :param pk_min: 
        
        Min boundary value of anomaly. `pk_min` is min value (lower) 
        of measurement point. It's the position of the site in meter 
        
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
        
        >>> from wmathandtricks import compute_power 
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

    
def select_anomaly ( rhoa_array, pos_array=None, auto=True,
                    dipole_length =10., **kws ) :
    """
    Select the anomaly value from `rhoa_array` and find its boundaries if 
    ``auto` is set to ``True``. If `auto` is ``False``, it's usefull to 
    provide the anomaly boundaries from station position. Change  the argument 
    `dipole_length`  i.e. the distance between measurement electrode is not
     equal to ``10``m else give the `pos_array`. If the `pos_array` is given,
     the `dipole_length` will be recomputed.
     
    :note:  If the `auto` param is ``True``, the automatic computation will
             give at most three best animalies ranking according 
             to the resitivity value. 
     
     :param rhoa_array: The apparent resistivity value of :ref:`erp` 
     :type rho_array: array_like 
     
     :param pos_array: The array of station position in meters 
     :type pos_array: array_like 
     
     :param auto: 
         
         Automaticaly of manual computation to select the best anomaly point. 
         Be sure if `auto` is set to ``False`` to provide the anomaly boundary
         by setting `pos_bounds` : 
             
             pos_bounds=(90, 130)
             
        where :math:`90` is the `pk_min` and :math:`130` is the `pk_max` 
        If `pos_bounds` is not given an station error will probably occurs 
        from :class:`~utils.exceptions.WATexError_station`. 
    
    :param dipole_length: 
        
        Is the distance between two closest measurement. If the value is known 
        it's better to provide it and don't need to provied a `pos_array`
        value. 
    :type dipole_length: float 

    :param pos_bounds: 
        
        Is the tuple value of anomaly boundaries  composed of `pk_min` and 
        `pk_max`. Please refer to :doc:`compute_power`. When provided 
        the `pos_bounds` value, please set `the dipole_length` to accurate 
        the computation of :func:`compute_power`.
        
    :return: 
        
        - *rhoa*: The app. resistivity value of the selected anomaly 
        - `pk_min` and the `pk_max`: refer to :doc:`compute_power`. 
        - `rhoa_max` and `rhoa_min`: refer to :doc:`compute_magnitude`
        - 
          
    """
    
    pos_bounds =kws.pop("pos_bounds", (None, None))
    anom_pos = kws.pop('pos_anomaly', None)
    display_infos =kws.pop('display', False)
    
    if auto is False : 
        if None in pos_bounds  or pos_bounds is None : 
            raise Wex.WATexError_site('One position is missed' 
                                'Plase provided it!')
        
        pos_bounds = np.array(pos_bounds)
        pos_min, pos_max  = pos_bounds.min(), pos_bounds.max()
        
        # get the res from array 
        dl_station_loc = np.arange(0, dipole_length * len(rhoa_array), 
                                   dipole_length)
        # then select rho range 
        ind_pk_min = int(np.where(dl_station_loc==pos_min)[0])
        ind_pk_max = int(np.where(dl_station_loc==pos_max)[0]) 
        rhoa_range = rhoa_array [ind_pk_min:ind_pk_max +1]
        pk, res= find_pk_from_selectedAn(an_res_range=rhoa_range, 
                                         pos=pos_bounds,
                                selectedPk= anom_pos) 
        pk = int(pk.replace('pk', ''))
        rhoa = rhoa_array[int(np.where(dl_station_loc == pk )[0])]
        rhoa_min = rhoa_array[int(np.where(dl_station_loc == pos_min )[0])]
        rhoa_max = rhoa_array[int(np.where(dl_station_loc == pos_max)[0])]
        
        rhoa_bounds = (rhoa_min, rhoa_max)
        
        return {'1_pk{}'.format(pk): 
                (pk, rhoa, pos_bounds, rhoa_bounds, res)} 
    
    if auto: 
        bestSelectedDICT, anpks, \
            collectanlyBounds, min_pks = compute_lower_anomaly(
                erp_array= rhoa_array, 
                station_position= pos_array, step= dipole_length,
                display_infos=display_infos ) 

            
        return {key: find_pkfeatures (anom_infos= bestSelectedDICT, 
                                      anom_rank= ii+1, pks_rhoa_index=min_pks, 
                                      dl=dipole_length) 
                for ii, (key , rho_r) in enumerate(bestSelectedDICT.items())
                }

     
                
                
        
def find_pkfeatures (anom_infos, anom_rank, pks_rhoa_index, dl): 
    """
    Get the pk bound from ranking of computed best points
    
    :param anom_infos:
        
        Is a dictionnary of best anomaly points computed from 
        :func:`compute_lower_anomaly` when `pk_bounds` is not given.  
        see :doc:`compute_lower_anomaly`
        
    :param anom_rank: Automatic ranking after selecting best points 
        
    :param pk_rhoa_index: 
        
        Is tuple of selected anomaly resistivity value and index in the whole
        :ref:`erp` line. for instance: 
            
            pks_rhoa_index= (80., 17) 
            
        where "80" is the value of selected anomaly in ohm.m and "17" is the 
        index of selected points in the :ref:`erp` array. 
        
    :param dl: 
        
        Is the distance between two measurement as `dipole_length`. Provide 
        the `dl` if the *default* value is not right. 
        
    :returns: 
        
        see :doc:`select_anomaly`
    
    """     
    rank_code = '{}_pk'.format(anom_rank)
    for key in anom_infos.keys(): 
        if rank_code in key: 
            pk = float(key.replace(rank_code, ''))

            rhoa = list(pks_rhoa_index[anom_rank-1])[0]
            codec = key
            break 
         
    ind_rhoa =np.where(anom_infos[codec] ==rhoa)[0]
    if len(ind_rhoa) ==0 : ind_rhoa =0 
    leninf = len(anom_infos[codec][: int(ind_rhoa)])
    
    pk_min = pk - leninf * dl 
    lensup =len(anom_infos[codec][ int(ind_rhoa):])
    pk_max =  pk + (lensup -1) * dl 
    
    pos_bounds = (pk_min, pk_max)
    rhoa_bounds = (anom_infos[codec][0], anom_infos[codec][-1])
    
    return pk, rhoa, pos_bounds, rhoa_bounds, anom_infos[codec]

                 
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
        
        >>> from watex.utils.wmathandtricks import compute_sfi 
        >>> sfi = compute_sfi(pk_min = 90,
        ...                      pk_max=130,
        ...                      rhoa_min=175,
        ...                      rhoa_max=170,
        ...                      rhoa=132,
        ...                      pk=110)
        >>> sfi
    
    """  
    def deprecated_sfi_computation () : 
        """ Decorated way for `sfi` computation"""
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
        
        >>> from watex.utils.wmathandtricks import compute_anr 
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


def find_pkBounds( pk , rhoa, rhoa_range, dl=10.):
    """
    Find station position boundary indexed in :ref:`erp` line. Usefull 
    to get the boundaries indexes `pk_boun_indexes` for :ref:`erp` 
    normalisation  when computing `anr` or else. 
    
    :param pk: Selected anomaly station value 
    :type pk: float 
    
    :param rhoa: Selected anomaly value in ohm.m 
    :type rhoa: float 
    
    :rhoa_range: Selected anomaly values from `pk_min` to `pk_max` 
    :rhoa_range: array_like 
    
    :parm dl: see :doc:`find_pkfeatures`
    
    :Example: 
        
        >>> from from watex.utils.wmathandtricks import find_pkBounds  
        >>> find_pkBounds(pk=110, rhoa=137, 
                          rhoa_range=np.array([175,132,137,139,170]))
    """

    if isinstance(pk, str): 
        pk = float(pk.replace(pk[0], '').replace('_pk', ''))
        
    index_rhoa = np.where(rhoa_range ==rhoa)[0]
    if len(index_rhoa) ==0 : index_rhoa =0 
    
    leftlen = len(rhoa_range[: int(index_rhoa)])
    rightlen = len(rhoa_range[int(index_rhoa):])
    
    pk_min = pk - leftlen * dl 
    pk_max =  pk + (rightlen  -1) * dl 
    
    return pk_min, pk_max 


def wrap_infos (phrase , value ='', underline ='-', unit ='',
                site_number= '', **kws) : 
    """Display info from anomaly details."""
    
    repeat =kws.pop('repeat', 77)
    intermediate =kws.pop('inter+', '')
    begin_phrase_mark= kws.pop('begin_phrase', '--|>')
    on = kws.pop('on', False)
    if not on: return ''
    else : 
        print(underline * repeat)
        print('{0} {1:<50}'.format(begin_phrase_mark, phrase), 
              '{0:<10} {1}'.format(value, unit), 
              '{0}'.format(intermediate), "{}".format(site_number))
        print(underline * repeat )
    
def drawn_anomaly_boundaries2(erp_data, appRes, index):
    """
    Function to drawn anomaly boundary 
    and return the anomaly with its boundaries
    
    :param erp_data: erp profile 
    :type erp_data: array_like or list 
    
    :param appRes: resistivity value of minimum pk anomaly 
    :type appRes: float 
    
    :param index: index of minimum pk anomaly 
    :type index: int 
    
    :return: anomaly boundary 
    :rtype: list of array_like 

    """
    f = 0 # flag to mention which part must be calculated 
    if index ==0 : 
        f = 1 # compute only right part 
    elif appRes ==erp_data[-1]: 
        f=2 # compute left part 
    
    def loop_sideBound(term):
        """
        loop side bar from anomaly and find the term side 
        
        :param term: is array of left or right side of anomaly.
        :type trem: array 
        
        :return: side bar 
        :type: array_like 
        """
        tem_drawn =[]
        maxT=0 

        for ii, tem_rho in enumerate(term) : 

            diffRes_betw_2pts= tem_rho - appRes 
            if diffRes_betw_2pts > maxT : 
                maxT = diffRes_betw_2pts
                tem_drawn.append(tem_rho)
            elif diffRes_betw_2pts < maxT : 
                # rho_limit = tem_rho 
                break 
        # print(tem_drawn)
        return np.array(tem_drawn)
    # first broke erp profile from the anomalies 
    if f==2 : # compute the left part 
        # flip array and start backward counting 
        temp_erp_data = erp_data [::-1] 
        sbeg = appRes   # initialize value 
        for ii, valan in enumerate(temp_erp_data): 
            if valan >= sbeg: 
                sbeg = valan 
            elif valan < sbeg: 
                left_term = erp_data[ii:]
                break 
            
        left_term = erp_data[:index][::-1] # flip left term  for looping
        # flip again to keep the order 
        left_limit = loop_sideBound(term=left_term)[::-1] 

    if f==0 or f ==1 : 
        right_term= erp_data[index :]
        right_limit=loop_sideBound(right_term)
    # concat right and left to get the complete anomaly 
    if f==2: 
        anomalyBounds = np.append(left_limit,appRes)
                                   
    elif f ==1 : 
        anomalyBounds = np.array([[appRes]+ right_limit.tolist()])
    else: 
        left_limit = np.append(left_limit, appRes)
        anomalyBounds = np.concatenate((left_limit, right_limit))
    
    return appRes, index, anomalyBounds  

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
        
        >>> from watex.core.erp import get_shape 
        >>> x = [60, 70, 65, 40, 30, 31, 34, 40, 38, 50, 61, 90]
        >>> shape = get_shape (rhoa_range= np.array(x))
        ...U

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


def getdfAndFindAnomalyBoundaries(df): 
    """
    Define anomaly boundary `upper bound` and `lowerbound` from 
    :ref:`ves` location. 
        
    :param df: Dataframe pandas contained  the columns 
                'pk', 'x', 'y', 'rho', 'dl'. 
    returns: 
        - `autoOption` triggered the automatic Option if nothing is specified 
            into excelsheet.
        - `ves_loc`: Sounding curve location at pk 
        - `posMinMax`: Anomaly boundaries composed of ``lower`` and ``upper``
            bounds.
           Specific names can be used  to define lower and upper bounds:: 
                
               `lower`: 'lower', 'inf', 'min', 'min', '1' or  'low'
               `upper`: 'upper', 'sup', 'maj', 'max', '2, or 'up'
               
        To define the sounding location, can use:: 
            `ves`:'ves', 'se', 'sond','vs', 'loc', '0' or 'dl'
            
    """
    shape_=[ 'V','W', 'U', 'H', 'M', 'C', 'K' ]
    type__= ['EC', 'NC', 'CP', 'CB2P']
        # - ``EC`` for Extensive conductive. 
        # - ``NC`` for narrow conductive. 
        # - ``CP`` for conductive PLANE 
        # - ``CB2P`` for contact between two planes. 
    shape =None 
    type_ =None 
    
    def recoverShapeOrTypefromSheet(listOfAddedArray, param): 
        """ Loop the array and get whether an anomaly shape name is provided
        :param listOfAddedArray: all Added array values except 
         'pk', 'x', 'y', 'rho' are composed of list of addedArray.
        :param param: Can be main description of different `shape_` of `type__` 
        
        :returns: 
            - `shape` : 'V','W', 'U', 'H', 'M', 'C' or  'K' from sheet or 
             `type` : 'EC', 'NC', 'CP', 'CB2P'
            - listOfAddedArray : list of added array 
        """
        param_ =None 
        for jj, colarray in enumerate(listOfAddedArray[::-1]): 
            tem_=[str(ss).upper().strip() for ss in list(colarray)] 
            for ix , elem in enumerate(tem_):
                for param_elm in param: 
                    if elem ==param_elm : 
                        # retrieves the shape and replace by np.nan value 
                        listOfAddedArray[::-1][jj][ix]=np.nan  
                        return param_elm , listOfAddedArray
                    
        return param_, listOfAddedArray
    
    def mergeToOne(listOfColumns, _df):
        """ Get data from other columns annd merge into one array
        
        :param listOfColumns: Columns names 
        :param _df: dataframe to retrieve data to one
        """
        new_array = np.full((_df.shape[0],), np.nan)
        listOfColumnData = [ _df[name].to_numpy() for name in listOfColumns ]
        # loop from backward so we keep the most important to the first row 
        # close the main df that composed `pk`,`x`, `y`, and `rho`.
        # find the shape 
        shape, listOfColumnData = recoverShapeOrTypefromSheet(listOfColumnData, 
                                                        param =shape_)
        type_, listOfColumnData = recoverShapeOrTypefromSheet(listOfColumnData, 
                                                        param =type__)
  
        for colarray in listOfColumnData[::-1]: 
           for ix , val  in enumerate(colarray): 
               try: 
                   if not np.isnan(val) : 
                       new_array[ix]=val
               except :pass  
        
        return shape , type_,  new_array 
    
    
    def retrieve_ix_val(array): 
        """ Retrieve value and index  and build `posMinMax boundaries
        
        :param array: array of main colum contains the anomaly definitions or 
                a souding curve location like :: 
                
                sloc = [NaN, 'low', NaN, NaN, NaN, 'ves', NaN,
                        NaN, 'up', NaN, NaN, NaN]
                `low`, `ves` and `up` are the lower boundary, the electric 
                sounding  and the upper boundary of the selected anomaly 
                respectively.
        For instance, if dipole_length is =`10`m, t he location (`pk`)
            of `low`, `ves` and `up` are 10, 50 and 80 m respectively.
            `posMinMax` =(10, 80)
        """
        
        lower_ix =None 
        upper_ix =None
        ves_ix = None 

        array= array.reshape((array.shape[0],) )
        for ix, val in enumerate(array):
            for low, up, vloc in zip(
                    ['lower', 'inf', 'min', 'min', '1', 'low'],
                    ['upper', 'sup', 'maj', 'max', '2', 'up'], 
                    ['ves', 'se', 'sond','vs', 'loc', '0', 'dl']
                    ): 
                try : 
                    floatNaNor123= np.float(val)
                except: 
                    if val.lower().find(low)>=0: 
                        lower_ix = ix 
                        break
                    elif val.lower().find(up) >=0: 
                        upper_ix = ix 
                        break
                    elif val.lower().find(vloc)>=0: 
                        ves_ix = ix 
                        break 
                else : 
                    if floatNaNor123 ==1: 
                        lower_ix = ix 
                        break
                    elif floatNaNor123 ==2: 
                        upper_ix = ix 
                        break 
                    elif floatNaNor123 ==0: 
                        ves_ix = ix 
                        break 
                           
        return lower_ix, ves_ix, upper_ix 
    
    # set pandas so to consider np.inf as NaN number.
    
    pd.options.mode.use_inf_as_na = True
    
    # unecesseray to specify the colum of sounding location.
    # dl =['drill', 'dl', 'loc', 'dh', 'choi']
    
    _autoOption=False  # set automatic to False one posMinMax  
    # not found as well asthe anomaly location `ves`.
    posMinMax =None 
    #get df columns from the 4-iem index 
    for sl in ['pk', 'sta', 'loc']: 
        for val in df.columns: 
            if val.lower()==sl: 
                pk_series = df[val].to_numpy()
                break 

    listOfAddedColumns= df.iloc[:, 4:].columns
    
    if len(listOfAddedColumns) ==0:
        return True,  shape, type_, None, posMinMax,  df   
 
    df_= df.iloc[:, 4:]
    # check whether all remains dataframe values are `NaN` values
    if len(list(df_.columns[df_.isna().all()])) == len(listOfAddedColumns):
         # If yes , trigger the auto option 
        return True,  shape, type_, None, posMinMax,  df.iloc[:, :4] 
  
    # get the colum name with any nan values 
    sloc_column=list(df_.columns[df_.isna().any()])
    # if colun is one man the sloc colum is found 
    sloc_values = df_[sloc_column].to_numpy()
    
    if len(sloc_column)>1 : #
    # get the value from single array
        shape , type_,  sloc_values =  mergeToOne(sloc_column, df_)


    lower_ix, ves_ix ,upper_ix   = retrieve_ix_val(sloc_values)
    
    # if `lower` and `upper` bounds are not found then start or end limits of
    # selected anomaly  from the position(pk) of the sounding curve. 
    if lower_ix is None : 
        lower_ix =ves_ix 
    if upper_ix is None: 
        upper_ix = ves_ix 
   
    if (lower_ix  and upper_ix ) is None: 
        posMinMax =None
    if posMinMax is None and ves_ix is None: _autoOption =True 
    else : 
        posMinMax =(pk_series[lower_ix], pk_series[upper_ix] )
        
    if ves_ix is None: ves_loc=None
    else : ves_loc = pk_series[ves_ix]
    
    return _autoOption, shape, type_, ves_loc , posMinMax,  df.iloc[:, :4]  
    

    
    

if __name__=='__main__': 
    path = 'data/erp/l10_gbalo.xlsx' # ztepogovogo_0
    path= r'F:\repositories\watex\data\Bag.main&rawds\ert_copy\nt\b1_5.xlsx'
    path = 'data/erp/test_anomaly.xlsx'
    data = pd.read_excel(path).to_numpy()[:, -1]
    df = pd.read_excel(path)

    # autotrig, shape ,type_,  indexanom , posMinMax, newdf = getdfAndFindAnomalyBoundaries(df)
    # print(autotrig, shape,type_,  indexanom , posMinMax, newdf)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    