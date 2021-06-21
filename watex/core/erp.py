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

.. synopsis:: 'watex.core.erp'
            Module to deal with Electrical resistivity profile (ERP)
            exploration tools 


Created on Tue May 18 12:33:15 2021

@author: @Daniel03
"""
import os 
import json 
import numpy as np 
import pandas as pd 
import  watex.utils.exceptions as Wex

import xml.etree.ElementTree as ET

from watex.utils._watexlog import watexlog 

_logger =watexlog.get_watex_logger(__name__)

class Features: 
    """
    Special module wich deal with Electrical Resistivity profile (VES) and
    Vertical electrical Sounding (VES). fet all features values of sites area 
    ERP data is given, Will set the minimum found on the
    array as the best anomaly. For better accuracy. The class of features 
    works after the features as been computed. 

    """
    
    dataType ={
                ".csv":pd.read_csv, 
                 ".xlsx":pd.read_excel,
                 ".json":pd.read_json,
                 ".html":pd.read_json,
                 ".sql" : pd.read_sql
                 }  
    feature_labels = [
                        'boreh', 
                        'x_m',
                        "y_m",
                        'pa',
                        "ma",
                        "shape",
                        "type",
                        "sfi",
                        'ohms',
                        'wi', 
                        'geol',
                        'flow'
            ]
    

    def __init__(self, erpData=None, horizonDis=None , rhoApp=None,**kwargs):
        self._logging = watexlog.get_watex_logger(self.__class__.__name__)
        
        self.erp_data=erpData
        
        self.utmX =kwargs.pop('utmX', None)
        self.utmY =kwargs.pop('utmY', None)
        
        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])
            

    def from_csv(self, erp_fn):
        """
        Method essentially created to read file from csv , collected 
        horizontal distance value and apparent resistivy values. 
        then send to the class for computation purposes. 
        
        :param erp_fn: path_like string of CSV file 
        :type erp_fn: str 
        
        :return: horizontal distance im meters 
        :rtype: np.array of all data
        """
        if not os.path.isfile(erp_fn):
            raise Wex.WATexError_file_handling('{} is not a file. '
                                               'Please provide a right file !'
                                               .format(erp_fn))
        with open(erp_fn, 'r') as fcsv: 
            csvData = fcsv.readlines()
            
        # retrieve ;locations,  coordinates values for each pk, 
        # horizontal distance in meter and apparent resistivity values 
        

    def from_xml (self, xml_fn, columns=None):
        """
        collected data from xml  and build dataFrame 
        :param xxlm_fn: full path to xml file 
        :type xml: str 
        
        :param columns: list of columns of dataset 
        :type columns: list 
        
        """
        tree = ET.parse(xml_fn)

        root = tree.getroot()
        dataframe = pd.DataFrame(columns = columns)
        # loop the node and collect files from loop 
        seriesList =[]
        for ii, node in enumerate(root): 
            seriesList.append(node.find(columns[ii]).text)
            
        # after loop , use series to create pandas dataframes 
        #create dataframe but ignore index 
    
        dataframe=dataframe.append(pd.Series(seriesList, index=columns ), 
                                   ignore_index=True )
        

    def from_json (self, json_fn , indent =4):
        """
        Collected data from json files and retrieve the most insights contents 
        
        :param json_fn: json file 
        :type json_fn: str 
        
        """
    
    def data_to_numpy(self, data_fn): 
        """
        Method to get datatype and set different features into nympy array
        
        """
        if data_fn is not None : 
            self.erp_data =data_fn 
        
        if not os.path.isfile(self.erp_data): 
            raise Wex.WATexError_file_handling('{} is not a file. '
                                               'Please provide a right file !'
                                               .format(self.erp_data))
        ex_file = os.path.splitext(pathfile)[1] 
        ex_file in not self.dataType.keys(): 
            
            
        
    
            

    
    
    
       
if __name__=='__main__'   : 
    pathData =r'data/BagoueDataset.xlsx'
    ex_file =  os.path.splitext(pathData)[1]
    print(ex_file)
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        