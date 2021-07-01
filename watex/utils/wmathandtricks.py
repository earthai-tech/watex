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

import watex.utils.exceptions as Wex
from watex.utils._watexlog import watexlog  

_logger =watexlog.get_watex_logger(__name__)



def compute_lower_anomaly(erp_array, station_position=None,  step=None): 
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
              * `min_pks`: list of tuples (pk, minVal of best anomalies points.)
    :rtype: tuple 
    
    :Example: 
        
        >>> from watex.utils.wmathandtricks import compute_lower_anolamy 
        >>> import pandas as pd 
        >>> path_to_= 'data/l10_gbalo.xlsx'
        >>> dataRes=pd.read_excel(erp_data).to_numpy()[:,-1]
        >>> anomaly, *_ =  compute_lower_anomaly(erp_array=data, step =10)
        >>> anomaly
                
    """

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

    print('{0:+^100}'.format(' *Best Conductive anomaly points (BCPts)* '))

    if pks.dtype in ['int', 'float']: 
        anpks =np.array([pks[skanIndex ] for
                         (_, skanIndex) in min_pks ])
    else : anpks ='?'
    
    bestSelectedDICT={}
    for ii, (pk, anb) in enumerate(zip(anpks, collectanlyBounds)): 
        bestSelectedDICT['{0}_pk{1}'.format(ii+1, pk)] = anb
    
    
    fmtAnText(anFeatures=bestSelectedDICT)
    
    
    return bestSelectedDICT, anpks, collectanlyBounds, min_pks, 
        

def get_minVal(array): 
    """
    Function to find the three minimum values on array and their 
    corresponding indexes 
    
    :param array: array  of values 
    :type array: array_like 
    
    :returns: Three minimum values of rho, index in rho_array
    :rtype: tuple
    
    """
    if not isinstance(array, (list, tuple, np.ndarray)):
        if isinstance(array, float): 
            array=np.array([array])
        else : 
            try : 
                array =np.array([float(array)])
            except: 
                Wex.WATexError_float('Could not convert %s to float!')
                
    temp_array =np.sort(array)
    holdList =[]
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
            # loop the index array and find the min 
            for jj, indx in enumerate(min_index):  
                holdList.append((array[int(indx)], 
                                 int(indx)))
        ss =len(holdList)
   
    return holdList 

def drawn_anomaly_boundaries(erp_data , appRes, index):
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
    if appRes ==erp_data[-1]: 
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
    if f ==0 or f==2 : 
        left_term = erp_data[:index][::-1] # flip left term  for looping 
        left_limit = loop_sideBound(term=left_term)[::-1] # flip again to keep the order 

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

def defineAnomaly(erp_data, station_position=None, pks=None, 
                  dipole_length=10., **kwargs):
    """
    Function will select the different anomalies. If pk is not given, 
    the best three anomalies on the survey lines will be computed automatically
    
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
                   'Dipole length is set to = {1} m. Please set a new bounds.')
            
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
    
    :paran an_res_range: anomaly resistivity range on ``erp`` line. 
    :type an_res_range: array_like 
    
    :param pos: position of anomaly boundaries (inf and sup):
                anBounds = [90, 130]
                - 130 is max boundary and 90 the  min boundary 
    :type pos: list 
    
    :param selectedPk: user can set its own position of the right anomaly 
                        Be sure that the value provided is right position . 
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
                print(sPk)
        elif isinstance(selectedPk, (float, int, np.ndarray)): 
            sPk = int(selectedPk)
        
        if pos is not None : # then compare the sPk and ps value 
            if not pos.min()<= sPk<=pos.max(): 
                warnings.warn('Wrong position given <{}>.'
                              ' Should compute new positions.'.format(selectedPk))
                _logger.debug('Wrong position given <{}>.'
                              'Should compute new positions.'.format(selectedPk))
            
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
        strfeatures ='|'.join(['{:^15}'.format(str(i)) for i in anFeatures[:-1]])
        try : 
            iter(anFeatures[-1])
        except : 
            strfeatures +='|{:^45}'.format(str(anFeatures[-1]))
        else : 
            strfeatures += '|{:^45}'.format(
                ''.join(['{} '.format(str(i)) for i in anFeatures[-1]]))
            
        print(strfeatures)
        print(line)
    
 
if __name__=='__main__': 

    erp_data='data/l10_gbalo.xlsx'# 'data/l11_gbalo.csv'
    df=pd.read_excel(erp_data)
    array= df.to_numpy()
    pk=array[:,0]
    data=array[:,-1]
    # print(data)
    # anom =np.array([168,130, 93,146,145,95,50,130,
    #                 163,140,167,154,93,113,138
    #         ])
    # _, _, test_an= drawn_anomaly_boundaries(erp_data=anom, appRes=93, index=12)
    # print(test_an)
    
    # anomaly, *_ =  compute_lower_anomaly(erp_array=data, step =10)
    anomaly = defineAnomaly(erp_data =data , station_position=None,
                            pks=[90, 130], dipole_length=10)
    print(anomaly)
    # pk, res= find_pk_from_selectedAn(an_res_range=[175,132,137,139,170], pos=[90, 130])
    # # fmtAnText(anFeatures =[1,130, 93,(146,145, 125)])
    # print(pk, res)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    