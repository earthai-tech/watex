# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Sat Aug 28 16:26:04 2021
#       This module is a set of utils for data prepprocessing
#       released under a MIT- licence.
#       @author: @Daniel03 <etanoyau@gmail.com>
import os 
import hashlib 
import tarfile 
import warnings 
import pickle 
import joblib
import datetime 
import shutil
from pprint import pprint  
from six.moves import urllib 
from typing import TypeVar, Generic, Iterable , Callable, Text
import pandas as pd 
import numpy as np 

from sklearn.model_selection import StratifiedShuffleSplit 

from watex.utils.__init__ import savepath as savePath 
from watex.utils._watexlog import watexlog

__logger = watexlog().get_watex_logger(__name__)

T= TypeVar('T')
KT=TypeVar('KT')
VT=TypeVar('VT')

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/WEgeophysics/watex/master/'
# from Zenodo: 'https://zenodo.org/record/5560937#.YWQBOnzithE'
DATA_PATH = 'data/__tar.tgz'  # 'BagoueCIV__dataset__main/__tar.tgz_files__'
TGZ_FILENAME = '/fmain.bagciv.data.tar.gz'
CSV_FILENAME = '/__tar.tgz_files__/___fmain.bagciv.data.csv'

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
    :param csv: output workbook in 'csv' format. If ``False`` will return un 
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
    
   
def fetchGeoDATA (data_url:str = DATA_URL, data_path:str =DATA_PATH,
                    tgz_filename =TGZ_FILENAME) -> Text: 
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
    
def fetchTGZDatafromURL (data_url:str =DATA_URL, data_path:str =DATA_PATH,
                    tgz_file=TGZ_FILENAME, file_to_retreive=None, **kws): 
    """ Fetch data from data repository in zip of 'targz_file. 
    
    I will create a `datasets/data` directory in your workspace, downloading
     the `~.tgz_file and extract the `data.csv` from this directory.
    
    :param data_url: url to the datafilename where `tgz` filename is located  
    :param data_path: absolute path to the `tgz` filename 
    :param filename: `tgz` filename. 
    """
    if data_url is not None: 
        
        tgz_path = os.path.join(data_path, tgz_file.replace('/', ''))
        try: 
            urllib.request.urlretrieve(data_url, tgz_path)
        except urllib.URLError: 
            print("<urlopen error [WinError 10061] No connection could "
                  "be made because the target machine actively refused it>")
        except ConnectionError or ConnectionRefusedError: 
            print("Connection failed!")
        except: 
            print("Unable to fetch {tgz_file} from <{git_url!r}>")
            
        return False 
    
    if file_to_retreive is not None: 
        f= fetchSingleTGZData(filename=file_to_retreive, **kws)
        
    return f

def fetchSingleTGZData(tgz_file, filename='___fmain.bagciv.data.csv',
                         savefile='data/geo_fdata', rename_outfile=None ):
    """ Fetch single file from archived tar file and rename a file if possible.
    
    :param tgz_file: str or Parh-Like obj 
        Full path to tarfile. 
    :param filename:str 
        Tagert  file to fetch from the tarfile.
    :savefile:str or Parh-like obj 
        Destination path to save the retreived file. 
    :param rename_outfile:str or Path-like obj
        Name of of the new file to replace the fetched file.
    :return: Location of the fetched file
    :Example: 
        >>> from watex.utils.ml_utils import fetchSingleTGZData
        >>> fetchSingleTGZData('data/__tar.tgz/fmain.bagciv.data.tar.gz', 
                               rename_outfile='main.bagciv.data.csv')
    """
     # get the extension of the fetched file 
    fetch_ex = os.path.splitext(filename)[1]
    def retreive_main_member (tarObj): 
        """ Retreive only the main member that contain the target filename."""
        for tarmem in tarObj.getmembers():
            if os.path.splitext(tarmem.name)[1]== fetch_ex: #'.csv': 
                return tarmem 
            
    if not os.path.isfile(tgz_file):
        raise FileNotFoundError(f"Source {tgz_file!r} is a wrong file.")
   
    with tarfile.open(tgz_file) as tar_ref:
        tar_ref.extractall(members=[retreive_main_member(tar_ref)])
        tar_name = [ name for name in tar_ref.getnames()
                    if name.find(filename)>=0 ][0]
        shutil.move(tar_name, savefile)
        # for consistency ,tree to check whether the tar info is 
        # different with the collapse file 
        if tar_name != savefile : 
            # print(os.path.join(os.getcwd(),os.path.dirname(tar_name)))
            _fol = tar_name.split('/')[0]
            shutil.rmtree(os.path.join(os.getcwd(),_fol))
        # now rename the file to the 
        if rename_outfile is not None: 
            os.rename(os.path.join(savefile, filename), 
                      os.path.join(savefile, rename_outfile))
        if rename_outfile is None: 
            rename_outfile =os.path.join(savefile, filename)
            
        print(f"{os.path.join(savefile, rename_outfile)!r} was successfully"
              f" decompressed  from {os.path.basename(tgz_file)!r}"
              "and saved to {savefile!r}")
        
    return os.path.join(savefile, rename_outfile)
    
def load_data (data_path:str = DATA_PATH,
               filename:str =CSV_FILENAME, sep =',' )-> Generic[VT]:
    """ Load CSV file to pd.dataframe. 
    
    :param data_path: path to data file 
    :param filename: name of file. 
    
    """ 
    if os.path.isfile(data_path): 
        return pd.read_csv(data_path, sep)
    
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
    dataset is refreshed. 
    
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

def fetch_model(modelfile, modelpath =None, default=True,
                modname =None, verbose =0): 
    """ Fetch your model saved using Python pickle module or 
    joblib module. 
    
    :param modelfile: str or Path-Like object 
        dumped model file name saved using `joblib` or Python `pickle` module.
    :param modelpath: path-Like object , 
        Path to model dumped file =`modelfile`
    :default: bool, 
        Model parameters by default are saved into a dictionary. When default 
        is ``True``, returns a tuple of pair (the model and its best parameters)
        . If False return all values saved from `~.MultipleGridSearch`
       
    :modname: str 
        Is the name of model to retrived from dumped file. If name is given 
        get only the model and its best parameters. 
    :verbose: int, level=0 
        control the verbosity.More message if greater than 0.
    
    :returns:
        - `model_class_params`: if default is ``True``
        - `pickedfname`: model dumped and all parameters if default is `False`
        
    :Example: 
        >>> from watex.bases import fetch_model 
        >>> my_model = fetch_model ('SVC__LinearSVC__LogisticRegression.pkl',
                                    default =False,  modname='SVC')
        >>> my_model
    """
    
    try:
        isdir =os.path.isdir( modelpath)
    except TypeError: 
        #stat: path should be string, bytes, os.PathLike or integer, not NoneType
        isdir =False
        
    if isdir and modelfile is not None: 
        modelfile = os.join.path(modelpath, modelfile)

    isfile = os.path.isfile(modelfile)
    if not isfile: 
        raise FileNotFoundError ("File {modelfile!r} not found!")
        
    from_joblib =False 
    if modelfile.endswith('.pkl'): from_joblib  =True 
    
    if from_joblib:
       __logger.info(f"Loading models `{os.path.basename(modelfile)}`!")
       try : 
           pickedfname = joblib.load(modelfile)
           # and later ....
           # f'{pickfname}._loaded' = joblib.load(f'{pickfname}.pkl')
           dmsg=f"Model {modelfile !r} retreived from~.externals.joblib`!"
       except : 
           dmsg=''.join([f"Nothing to retrived. It's seems model {modelfile !r}", 
                         " not really saved using ~external.joblib module! ", 
                         "Please check your model filename."])
    
    if not from_joblib: 
        __logger.info(f"Loading models `{os.path.basename(modelfile)}`!")
        try: 
           # DeSerializing pickled data 
           with open(modelfile, 'rb') as modf: 
               pickedfname= pickle.load (modf)
           __logger.info(f"Model `{os.path.basename(modelfile)!r} deserialized"
                         "  using Python pickle module.`!")
           
           dmsg=f'Model `{modelfile!r} deserizaled from  {modelfile}`!'
        except: 
            dmsg =''.join([" Unable to deserialized the "
                           f"{os.path.basename(modelfile)!r}"])
           
        else: 
            __logger.info(dmsg)   

    if verbose > 0: 
        pprint(
            dmsg 
            )
           
    if modname is not None: 
        keymess = "{modname!r} not found."
        try : 
            if default:
                model_class_params  =( pickedfname[modname]['best_model'], 
                                   pickedfname[modname]['best_params_'], 
                                   pickedfname[modname]['best_scores'],
                                   )
            if not default: 
                model_class_params=pickedfname[modname]
                
        except KeyError as key_error: 
            warnings.warn(
                f"Model name {modname!r} not found in the list of dumped"
                f" models = {list(pickedfname.keys()) !r}")
            raise KeyError from key_error(keymess + "Shoud try the model's"
                                          "names ={list(pickedfname.keys())!r}")
        
        if verbose > 0: 
            pprint('Should return a tuple of `best model` and the'
                   ' `model best parameters.')
           
        return model_class_params  
            
    if default:
        model_class_params =list()    
        
        for mm in pickedfname.keys(): 
            model_class_params.append((pickedfname[mm]['best_model'], 
                                      pickedfname[mm]['best_params_'],
                                      pickedfname[modname]['best_scores']))
    
        if verbose > 0: 
               pprint('Should return a list of tuple pairs:`best model`and '
                      ' `model best parameters.')
               
        return model_class_params

    return pickedfname 

def dumpOrSerializeData (data , filename=None, savepath =None, to=None): 
    """ Dump and save binary file 
    
    :param data: Object
        Object to dump into a binary file. 
    :param filename: str
        Name of file to serialize. If 'None', should create automatically. 
    :param savepath: str, PathLike object
         Directory to save file. If not exists should automaticallycreate.
    :param to: str 
        Force your data to be written with specific module like ``joblib`` or 
        Python ``pickle` module. Should be ``joblib`` or ``pypickle``.
    :return: str
        dumped or serialized filename.
        
    :Example:
        
        >>> import numpy as np
        >>> from watex.utils.ml_utils import dumpOrSerializeData
        >>>  data=(np.array([0, 1, 3]),np.array([0.2, 4]))
        >>> dumpOrSerializeData(data, filename ='__XTyT.pkl', to='pickle', 
                                savepath='watex/datasets')
    """
    if filename is None: 
        filename ='__mydumpedfile.{}__'.format(datetime.datetime.now())
        filename =filename.replace(' ', '_').replace(':', '-')

    if to is not None: 
        if not isinstance(to, str): 
            raise TypeError(f"Need to be string format not {type(to)}")
        if to.lower().find('joblib')>=0: to ='joblib'
        elif to.lower().find('pickle')>=0:to = 'pypickle'
        
        if to not in ('joblib', 'pypickle'): 
            raise ValueError("Unknown argument `to={to}`."
                             " Should be <joblib> or <pypickle>!")
    # remove extension if exists
    if filename.endswith('.pkl'): 
        filename = filename.replace('.pkl', '')
        
    __logger.info(f'Dumping data to `{filename}`!')    
    try : 
        if to is None or to =='joblib':
            joblib.dump(data, f'{filename}.pkl')
            
            filename +='.pkl'
            __logger.info(f'Data dumped in `{filename} using '
                          'to `~.externals.joblib`!')
        elif to =='pypickle': 
            # force to move pickling data  to exception and write using 
            # Python pickle module
            raise 
    except : 
        # Now try to pickle data Serializing data 
        with open(filename, 'wb') as wfile: 
            pickle.dump( data, wfile)
        __logger.info( 'Data are well serialized  '
                      'using Python pickle module.`')
        
    if savepath is not None:
        try : 
            savepath = savePath (savepath)
        except : 
            savepath = savePath ('_dumpedData_')
        try:
            shutil.move(filename, savepath)
        except :
            print(f"--> It seems destination path {filename!r} already exists.")

    if savepath is None: savepath =os.getcwd()
    print(f"Data are well {'serialized' if to=='pypickle' else 'dumped'}"
          f" to <{os.path.basename(filename)!r}> in {savepath!r} directory.")
   
def loadDumpedOrSerializedData (filename:str): 
    """ Load dumped or serialized data from filename 
    
    :param filename: str or path-like object 
        Name of dumped data file.
    :return: 
        Data loaded from dumped file.
        
    :Example:
        
        >>> from watex.utils.ml_utils import loadDumpedOrSerializedData
        >>> loadDumpedOrSerializedData(filename ='Watex/datasets/__XTyT.pkl')
    """
    
    if not isinstance(filename, str): 
        raise TypeError(f'filename should be a <str> not <{type(filename)}>')
        
    if not os.path.isfile(filename): 
        raise FileExistsError(f"File {filename!r} does not exist.")
        
    dumped_type =''
    _filename = os.path.basename(filename)
    __logger.info(f"Loading data from `{_filename}`!")
   
    data =None 
    try : 
        data= joblib.load(filename)
        __logger.info(''.join([f"Data from {_filename !r} are sucessfully", 
                      " loaded using ~.externals.joblib`!"]))
        
        dumped_type ='joblib'
    except : 
        __logger.info(
            ''.join([f"Nothing to reload. It's seems data from {_filename!r}", 
                      " are not really dumped using ~external.joblib module!"])
            )
        # Try DeSerializing using pickle module
        with open(filename, 'rb') as tod: 
            data= pickle.load (tod)
        __logger.info(f"Data from `{_filename!r} are well"
                      " deserialized using Python pickle module.`!")
        dumped_type ='pickle'
        
    is_none = data is None
    if is_none: 
        print("Unable to deserialize data. Please check your file.")
    
    else : print(f"Data from {_filename} have been reloaded sucessfully  "
          f"using <{dumped_type}> module.")
    
    return data 



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
