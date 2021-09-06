# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Wed Jul 14 20:00:26 2021
# This module is a set of transformers for data preparing. It is  part of 
# the WATex preprocessing module which is released under a MIT- licence.

"""
Created on Sat Aug 28 16:26:04 2021

@author: @Daniel03

"""

from typing import TypeVar, Generic, Iterable , Callable, Text

T= TypeVar('T', float, int, str)
V=TypeVar('V', list, dict, tuple)

import os 

import tarfile 
from six.moves import urllib 
import pandas as pd 

DOWNLOAD_ROOT = 'https://github.com/WEgeophysics/watex'
#'https://zenodo.org/record/4896758#.YTWgKY4zZhE'
DATA_PATH = 'data/geo_fdata'
FILENAME = '/bag.main&rawds.rar'

DATA_URL = DOWNLOAD_ROOT + DATA_PATH  + FILENAME 




def read_from_excelsheets(erp_file: T =None ) -> V[T]: 
    
    """
    Read all Excelsheets and build a list of dataframe of all sheets.
    :param erp_file:
        Excell workbooks containing `erp` profile data.
    :return: A list composed of the name of `erp_file` at index =0 and the 
            datataframes.
    """
    allfls= pd.read_excel(erp_file, sheet_name=None)
    
    list_of_df =[os.path.basename(os.path.splitext(erp_file)[0])]
    for sheets , values in allfls.items(): 
        list_of_df.append(pd.DataFrame(values))

    return list_of_df 

def write_excel(listOfDfs:V[T], csv:bool =False , sep:T =','): 
    """ 
    Rewrite excell workbook with dataframe for :ref:`read_from_excelsheets`. 
    
    Its recover the name of the files and write the data from dataframe 
    associated with the name of the `erp_file`. 
    
    :param listOfDfs: list composed of `erp_file` name at index 0 and the
     remains dataframes. 
    :param csv: output workbood in 'csv' format. If ``False`` will return un 
     `excel` format. 
    :param sep: type of data separation. 'default is ``,``.'
    
    """
    site_name = listOfDfs[0]
    listOfDfs = listOfDfs[1:]
    for ii , df in enumerate(listOfDfs):
        
        if csv:
            df.to_csv(df, sep=sep)
        else :
            with pd.ExcelWriter(f"z{site_name}_{ii}.xlsx") as writer: 
                df.to_excel(writer, index=False)
    
   
def fetch_geo_data (data_url = DATA_URL, data_path =DATA_PATH,
                    filename =FILENAME ) -> Text: 
    """ Fetch data from data repository in zip of 'targz_file. I will create 
     a `datasets/data` directory in your workspace, downloqding the `~.tgz_file
     and extract the `data.csv` from this directory.
     
    :param data_url: url to the datafilename where `tgz` filename is located  
    :param data_path: absolute path to the `tgz` filename 
    :param filename: `tgz` filename. 
    """
    if not os.path.isdir(data_path): 
        os.makedirs(data_path)
    tgz_path = os.path.join(data_url, filename.replace('/', ''))
    
    urllib.request.urlretrieve(data_url, tgz_path)
    data_tgz = tarfile.open(tgz_path)
    data_tgz.extractall(path = data_path )
    data_tgz.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__=="__main__": 
    erp_file ='kouto-gbalo.xlsx'
    # erpPath = os.abspath()
    # print(os.path.abspath('.'))
    list_erp = [os.path.join(os.path.abspath('.'), file) 
                for file in os.listdir('.') 
                if os.path.isfile(file) and file.endswith('.xlsx')]
    # print(list_erp)
    for ffile in list_erp : 
        dictfiles = read_from_excelsheets(ffile)
        write_excel(dictfiles)