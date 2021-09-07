# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Wed Jul 14 20:00:26 2021
# This module is a set of utils for data prepprocessing
# released under a MIT- licence.
"""
Created on Sat Aug 28 16:26:04 2021

@author: @Daniel03

"""

from typing import TypeVar, Generic, Iterable , Callable, Text

T= TypeVar('T')
KT=TypeVar('KT')
VT=TypeVar('VT')


import os 
import hashlib 
import tarfile 
from six.moves import urllib 

import pandas as pd 
import numpy as np 

from sklearn.model_selection import StratifiedShuffleSplit 

DOWNLOAD_ROOT = 'https://github.com/WEgeophysics/watex/master/'
#'https://zenodo.org/record/4896758#.YTWgKY4zZhE'
DATA_PATH = 'data/tar.tgz_files'
TGZ_FILENAME = '/bagoue.main&rawdata.tgz'
CSV_FILENAME = '_bagoue_civ_loc_ves&erpdata3.csv'

DATA_URL = DOWNLOAD_ROOT + DATA_PATH  + TGZ_FILENAME


def read_from_excelsheets(erp_file: T = None ) -> Iterable[VT]: 
    
    """ Read all Excelsheets and build a list of dataframe of all sheets.
   
    :param erp_file:
        Excell workbooks containing `erp` profile data.
    :return: A list composed of the name of `erp_file` at index =0 and the 
            datataframes.
    """
    
    allfls:Generic[KT, VT] = pd.read_excel(erp_file, sheet_name=None)
    
    list_of_df =[os.path.basename(os.path.splitext(erp_file)[0])]
    for sheets , values in allfls.items(): 
        list_of_df.append(pd.DataFrame(values))

    return list_of_df 

def write_excel(listOfDfs: Iterable[VT], csv:bool =False , sep:T =','): 
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
    
   
def fetch_geo_data (data_url:str = DATA_URL, data_path:str =DATA_PATH,
                    tgz_filename =TGZ_FILENAME ) -> Text: 
    """ Fetch data from data repository in zip of 'targz_file. 
    
    I will create a `datasets/data` directory in your workspace, downloading
     the `~.tgz_file and extract the `data.csv` from this directory.
    
    :param data_url: url to the datafilename where `tgz` filename is located  
    :param data_path: absolute path to the `tgz` filename 
    :param filename: `tgz` filename. 
    """
    if not os.path.isdir(data_path): 
        os.makedirs(data_path)
    tgz_path = os.path.join(data_url, tgz_filename.replace('/', ''))
    
    urllib.request.urlretrieve(data_url, tgz_path)
    data_tgz = tarfile.open(tgz_path)
    data_tgz.extractall(path = data_path )
    data_tgz.close()
    
    
def load_data (data_path:str = DATA_PATH,
               filename:str =CSV_FILENAME, sep =',' )-> Generic[VT]:
    """ Load CSV file to pd.dataframe. 
    
    :param data_path: path to data file 
    :param filename: name of file. 
    
    """ 
    csv_path = os.path.join(data_path , filename)
    
    return pd.read_csv(csv_path, sep)


def split_train_test (data:Generic[VT], test_ratio:T)-> Generic[VT]: 
    """ Split dataset into trainset and testset from `test_ratio` 
    and return train set and test set.
        
    ..note: `test_ratio` is ranged between 0 to 1. Default is 20%.
    """
    shuffled_indices =np.random.permutation(len(data)) 
    test_set_size = int(len(data)* test_ratio)
    test_indices = shuffled_indices [:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    return data.iloc[train_indices], data.iloc[test_indices]
    
def test_set_check_id (identifier, test_ratio, hash:Callable[..., T]) -> bool: 
    """ 
    Get the testset id and set the corresponding unique identifier. 
    
    Compute the a hash of each instance identifier, keep only the last byte 
    of the hash and put the instance in the testset if this value is lower 
    or equal to 51(~20% of 256) 
    has.digest()` contains object in size between 0 to 255 bytes.
    
    :param identifier: integer unique value 
    :param ratio: ratio to put in test set. Default is 20%. 
    
    :param hash:  
        Secure hashes and message digests algorithm. Can be 
        SHA1, SHA224, SHA256, SHA384, and SHA512 (defined in FIPS 180-2) 
        as well as RSA’s MD5 algorithm (defined in Internet RFC 1321). 
        
        Please refer to :ref:`<https://docs.python.org/3/library/hashlib.html>` 
        for futher details.
    """
    return hash(np.int64(identifier)).digest()[-1]< 256 * test_ratio

def split_train_test_by_id(data, test_ratio:T, id_column:T=None,
                           hash=hashlib.md5)-> Generic[VT]: 
    """Ensure that data will remain consistent accross multiple runs, even if 
    dataset i refreshed. 
    
    The new testset will contain 20%of the instance, but it will not contain 
    any instance that was previously in the training set.

    :param data: Pandas.core.DataFrame 
    :param test_ratio: ratio of data to put in testset 
    :id_colum: identifier index columns. If `id_column` is None,  reset  
                dataframe `data` index and set `id_column` equal to ``index``
    :param hash: secures hashes algorithms. Refer to 
                :func:`~test_set_check_id`
    :returns: consistency trainset and testset 
    """
    if id_column is None: 
        id_column ='index' 
        data = data.reset_index() # adds an `index` columns
        
    ids = data[id_column]
    in_test_set =ids.apply(lambda id_:test_set_check_id(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

def discretizeCategoriesforStratification(data, in_cat:str =None,
                               new_cat:str=None, **kws) -> Generic[VT]: 
    """ Create a new category attribute to discretize instances. 
    
    A new category in data is better use to stratified the trainset and 
    the dataset to be consistent and rounding using ceil values.
    
    :param in_cat: column name used for stratified dataset 
    :param new_cat: new category name created and inset into the 
                dataframe.
    :return: new dataframe with new column of created category.
    """
    divby = kws.pop('divby', 1.5) # normalize to hold raisonable number 
    combined_cat_into = kws.pop('higherclass', 5) # upper class bound 
    
    data[new_cat]= np.ceil(data[in_cat]) /divby 
    data[new_cat].where(data[in_cat] < combined_cat_into, 
                             float(combined_cat_into), inplace =True )
    return data 

def stratifiedUsingDiscretedCategories(data:VT , cat_name:str , n_splits:int =1, 
                    test_size:float= 0.2, random_state:int = 42)-> Generic[VT]: 
    """ Stratified sampling based on new generated category  from 
    :func:`~DiscretizeCategoriesforStratification`.
    
    :param data: dataframe holding the new column of category 
    :param cat_name: new category name inserted into `data` 
    :param n_splits: number of splits 
    """
    
    split = StratifiedShuffleSplit(n_splits, test_size, random_state)
    for train_index, test_index in split.split(data, data[cat_name]): 
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index] 
        
    return strat_train_set , strat_test_set 

    
if __name__=="__main__": 
    
    df = load_data('data/geo_fdata')
    # df.hist(bins=50, figsize=(20, 15))
    
    data = discretizeCategoriesforStratification(df, in_cat='flow', new_cat='tempf_',
                                          divby =1, higherclass=3)
    
    # f_, t_=[], []
    # for elm in id_columns : 

    #     op = test_set_check_id(elm, test_ratio=0.2, hash=hashlib.md5)
 
    #     if op is False :
    #         f_.append(op)
    #     else :
    #         t_.append(op)
    #     # print(op)
    # print(f_)
    # print(t_)
    # df.hist(bins=50, figsize=(20, 15))
    # create index 
    
    
    # train_set , test_set = split_train_test_by_id(df, test_ratio=0.2)
    # print(train_set)
    # erp_file ='kouto-gbalo.xlsx'
    # # erpPath = os.abspath()
    # # print(os.path.abspath('.'))
    # list_erp = [os.path.join(os.path.abspath('.'), file) 
    #             for file in os.listdir('.') 
    #             if os.path.isfile(file) and file.endswith('.xlsx')]
    # # print(list_erp)
    # for ffile in list_erp : 
    #     dictfiles = read_from_excelsheets(ffile)
    #     write_excel(dictfiles)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        