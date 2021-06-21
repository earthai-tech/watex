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

.. synopsis:: 'watex.utils.calculator'
            Module for computing

Created on Mon Jun 21 14:43:25 2021

@author: @Daniel03

"""

import os 
import numpy as np 
import pandas as pd
import watex.utils.exceptions as Wex



def compute_lower_anomaly(erp_array, station_position=None,  step=None): 
    """
    Function to get the minimum value on the ERP array. 
    If `pk` is provided wil give the index of pk
    
    :param erp_array: array of apparent resistivity profile 
    :type erp_array: array_like
    
    :station position: array of station position (survey) , if not given 
                    and `step` is known , set the step value and 
                    `station_position` will compute automatically 
    :type station_position: array_like 
    
    :param step: The distance between measurement im meter. If given will 
        recompute the `station_position`
    
    :returns: * anpks: Main positions of best select anomaly 
              * collectanlyBounds: list of arrays of select anomaly values
              * min_pks: list of tuples (pk, minVal of best anomalies points.)
    :rtype: tuple 
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

    print('{0:-^77}'.format(' *Best Conductive points (BCPts)* '))
    print('** -----|> {0:<17}  = {1} points.'.format('Number of BCPts found',
                                                   len(min_pks)))
    for ii, (skanRes, skanIndex) in enumerate(min_pks): 
        print('* ({0}): pk{1} --> {1} m with rho value = {2} Ω.m'.format(ii+1,
                                                    pks[skanIndex ],
                                                      skanRes))
    if pks.dtype in ['int', 'float']: 
        anpks =np.array([pks[skanIndex ] for
                         (_, skanIndex) in min_pks ])
    else : anpks ='?'
    
    return anpks, collectanlyBounds, min_pks
        

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
            for jj, indx in enumerate(min_index):  # loop the index array and find the min 
                holdList.append((array[int(indx)], 
                                 int(indx)))
        ss =len(holdList)
   
    return holdList 

def drawn_anomaly_boundaries(erp_data , appRes, index):
    """
    function to drawn anomaly boundary 
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
        lopp side bar from anomaly and find the term side 
        :param term: is array of left or right side of anomaly 
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

def defineAnomaly(erp_data, station_position=None, pks=None, dipole_length=10.):
    """
    Function will select the different anomalies. If pk is not given, 
    the best three anomalies on the survey lines will be computed automatically
    
    :param erp_data: Electrical resistivity profiling 
    :type erp_data: array_like 
    
    :param pks: station positions anomaly boundaries (pk_begin, pk_end)
                If selected anomalies is more than one, set `pks` into dict
                where number of anomaly =keys and pks = values 
    :type pks: list of dict
    
    :param dipole_length: Distance between two measurements in meters
                        Change the `dipole lengh
    :type dipole_length: float
    
    :param station_position: station position array 
    :type statiion_position: array_like 
    
    
    """
    collectanlyBounds=[]
    if station_position is not None : 
        dipole_length = (station_position.max()-
               station_position.min())/(len(station_position -1))
    if station_position is None : 
        station_position =np.arange(0, dipole_length * len(erp_data), 
                                    dipole_length)
                                        
  
    def getBound(pksbounds): 
        """
        Get the bound from the given `pks`
        erp
        
        """
        # check if bound is on station positions
        for spk in pksbounds : 
            if not pksbounds.min() <= spk <= pksbounds.max(): 
                Wex.WATexError_AnomalyBounds(
                    'Bound <{0}> provided is out of range !'
                   'Dipole length is set to = {1} m. Please set a new bounds.')
            
        pkinf = np.where(station_position==pksbounds.min())
        pksup = np.where(station_position==pksbounds.max())
        anomBounds = erp_data[int(pkinf):int(pksup)]
        return anomBounds
    
    if pks is None : 
        anpks, collectanlyBounds, min_pks= compute_lower_anomaly(
            erp_array=erp_data, step=dipole_length, 
            station_position =station_position)
        
    elif isinstance(pks, list):
        pks =np.array(sorted(pks))
        collectanlyBounds = getBound(pksbounds= pks)
    elif isinstance(pks, dict):
        for keys, values in pks.items():
            if isinstance(values, list): 
                values =np.array(values)
            collectanlyBounds.append(getBound(pksbounds=values))

    return collectanlyBounds 
       
     
     
if __name__=='__main__': 

    erp_data='data/l10_gbalo.xlsx'# 'data/l11_gbalo.csv'
    df=pd.read_excel(erp_data)
    array= df.to_numpy()
    pk=array[:,0]
    data=array[:,-1]
    #print(data)
    # anom =np.array([168,130, 93,146,145,95,50,130,
    #                 163,140,167,154,93,113,138
    #         ])
    # _, _, test_an= drawn_anomaly_boundaries(erp_data=anom, appRes=93, index=12)
    # print(test_an)
    
    #anomaly =  compute_lower_anomaly(erp_array=data, step =10)
    anomaly = defineAnomaly(erp_data =data , station_position=None,
                            pks=None, dipole_length=10)
    print(anomaly)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    