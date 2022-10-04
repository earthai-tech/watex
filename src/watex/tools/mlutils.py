# -*- coding: utf-8 -*-
#      Copyright (c) 2021 Kouadio K. Laurent, Sat Aug 28 16:26:04 2021
#      released under a MIT- licence.
#      @author: @Daniel03 <etanoyau@gmail.com>

from __future__ import annotations 
import os 
# import re
import inspect 
import hashlib 
import tarfile 
import warnings 
import pickle 
import joblib
import datetime 
import shutil
from pprint import pprint  
from six.moves import urllib 

import numpy as np 
import pandas as pd 

from .._watexlog import watexlog
from ..exlib  import ( 
    train_test_split , 
    StratifiedShuffleSplit, 
    confusion_matrix, 
    mean_squared_error 
    
)
from ..typing import (
    List,
    Tuple, 
    Any,
    Dict, 
    Optional,
    Union, 
    Iterable ,
    T,
    F, 
    ArrayLike, 
    NDArray,
    DType, 
    DataFrame, 
    Sub                 
)
from ..exceptions import ( 
    ParameterNumberError , 
    EstimatorError
)
from .funcutils import (
    _assert_all_types, 
    savepath_, 
    smart_format, 
    
)

_logger = watexlog().get_watex_logger(__name__)


DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/WEgeophysics/watex/master/'
# from Zenodo: 'https://zenodo.org/record/5560937#.YWQBOnzithE'
DATA_PATH = 'data/__tar.tgz'  # 'BagoueCIV__dataset__main/__tar.tgz_files__'
TGZ_FILENAME = '/fmain.bagciv.data.tar.gz'
CSV_FILENAME = '/__tar.tgz_files__/___fmain.bagciv.data.csv'

DATA_URL = DOWNLOAD_ROOT + DATA_PATH  + TGZ_FILENAME

_estimators ={
        'dtc': ['DecisionTreeClassifier', 'dtc', 'dec', 'dt'],
        'svc': ['SupportVectorClassifier', 'svc', 'sup', 'svm'],
        'sdg': ['SGDClassifier','sdg', 'sd', 'sdg'],
        'knn': ['KNeighborsClassifier','knn', 'kne', 'knr'],
        'rdf': ['RandomForestClassifier', 'rdf', 'rf', 'rfc',],
        'ada': ['AdaBoostClassifier','ada', 'adc', 'adboost'],
        'vtc': ['VotingClassifier','vtc', 'vot', 'voting'],
        'bag': ['BaggingClassifier', 'bag', 'bag', 'bagg'],
        'stc': ['StackingClassifier','stc', 'sta', 'stack'],
    'xgboost': ['ExtremeGradientBoosting', 'xgboost', 'gboost', 'gbdm'], 
     'logit': ['LogisticRegression', 'logit', 'lr', 'logreg'], 
     'extree': ['ExtraTreesClassifier', 'extree', 'xtree', 'xtr']
        }  

def existfeatures (df, features, error='raise'): 
    """Control whether the features exists or not  
    
    :param df: a dataframe for features selections 
    :param features: list of features to select. Lits of features must be in the 
        dataframe otherwise an error occurs. 
    :param error: str - raise if the features don't exist in the dataframe. 
        *default* is ``raise`` and ``ignore`` otherwise. 
        
    :return: bool 
        assert whether the features exists 
    """
    isf = False  
    
    error= 'raise' if error.lower().strip().find('raise')>= 0  else 'ignore' 

    if isinstance(features, str): 
        features =[features]
        
    features = _assert_all_types(features, list, tuple, np.ndarray)
    set_f =  set (features).intersection (set(df.columns))
    if len(set_f)!= len(features): 
        nfeat= len(features) 
        msg = f"Feature{'s' if nfeat >1 else ''}"
        if len(set_f)==0:
            if error =='raise':
                raise ValueError (f"{msg} {smart_format(features)} "
                                  f"{'does not' if nfeat <2 else 'dont'}"
                                  " exist in the dataframe")
            isf = False 
        # get the difference 
        diff = set (features).difference(set_f) if len(
            features)> len(set_f) else set_f.difference (set(features))
        nfeat= len(diff)
        if error =='raise':
            raise ValueError(f"{msg} {smart_format(diff)} not found in"
                             " the dataframe.")
        isf = False  
    else : isf = True 
    
    return isf  
    
def selectfeatures (
        df: DataFrame,
        features: List[str] =None, 
        include = None, 
        exclude = None,
        coerce: bool=False,
        **kwd
        ): 
    """ Select features  and return new dataframe.  
    
    :param df: a dataframe for features selections 
    :param features: list of features to select. Lits of features must be in the 
        dataframe otherwise an error occurs. 
    :param include: the type of data to retrieved in the dataframe `df`. Can  
        be ``number``. 
    :param exclude: type of the data to exclude in the dataframe `df`. Can be 
        ``number`` i.e. only non-digits data will be keep in the data return.
    :param coerce: return the whole dataframe with transforming numeric columns.
        Be aware that no selection is done and no error is raises instead. 
        *default* is ``False``
    :param kwd: additional keywords arguments from `pd.astype` function 
    
    :ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html
    """
    
    if features is not None: 
        existfeatures(df, features, error ='raise')
    # change the dataype 
    df = df.astype (float, errors ='ignore', **kwd) 
    # assert whether the features are in the data columns
    if features is not None: 
        return df [features] 
    # raise ValueError: at least one of include or exclude must be nonempty
    # use coerce to no raise error and return data frame instead.
    return df if coerce else df.select_dtypes (include, exclude) 
    
    
def getGlobalScore (
        cvres : Dict[str, ArrayLike] 
        ) -> Tuple [ Dict[str, ArrayLike] ,  Dict[str, ArrayLike]  ]: 
    """ Retrieve the global mean and standard deviation score  from the 
    cross validation containers. 
    
    :param cvres: cross validation results after training the models of number 
        of parameters equals to N. 
    :type cvres: dict of Array-like, Shape (N, ) """
    return  ( cvres.get('mean_test_score').mean() ,
             cvres.get('std_test_score').mean()) 
    
    
def cfexist(features_to: List[ArrayLike], 
            features: List[str] )-> bool:      
    """
    Control features existence into another list . List or array can be a 
    dataframe columns for pratical examples.  
    
    :param features_to :list of array to be controlled .
    :param features: list of whole features located on array of `pd.DataFrame.columns` 
    
    :returns: 
        -``True``:If the provided list exist in the features colnames 
        - ``False``: if not 

    """
    if isinstance(features_to, str): 
        features_to =[features_to]
    if isinstance(features, str): features =[features]
    
    if sorted(list(features_to))== sorted(list(
            set(features_to).intersection(set(features)))): 
        return True
    else: return False 

def formatGenericObj(generic_obj :Iterable[T])-> T: 
    """
    Format a generic object using the number of composed items. 

    :param generic_obj: Can be a ``list``, ``dict`` or other `TypeVar` 
        classified objects.
    
    :Example: 
        
        >>> from watex.tools.mlutils import formatGenericObj 
        >>> formatGenericObj ({'ohmS', 'lwi', 'power', 'id', 
        ...                         'sfi', 'magnitude'})
        
    """
    
    return ['{0}{1}{2}'.format('`{', ii, '}`') for ii in range(
                    len(generic_obj))]


def findIntersectionGenObject(
        gen_obj1: Iterable[Any], 
        gen_obj2: Iterable[Any]
                              )-> set: 
    """
    Find the intersection of generic object and keep the shortest len 
    object `type` at the be beginning 
  
    :param gen_obj1: Can be a ``list``, ``dict`` or other `TypeVar` 
        classified objects.
    :param gen_obj2: Idem for `gen_obj1`.
    
    :Example: 
        
        >>> from watex.tools.mlutils import findIntersectionGenObject
        >>> findIntersectionGenObject(
        ...    ['ohmS', 'lwi', 'power', 'id', 'sfi', 'magnitude'], 
        ...    {'ohmS', 'lwi', 'power'})
        [out]:
        ...  {'ohmS', 'lwi', 'power'}
    
    """
    if len(gen_obj1) <= len(gen_obj2):
        objType = type(gen_obj1)
    else: objType = type(gen_obj2)

    return objType(set(gen_obj1).intersection(set(gen_obj2)))

def findDifferenceGenObject(gen_obj1: Iterable[Any],
                            gen_obj2: Iterable[Any]
                              )-> None | set: 
    """
    Find the difference of generic object and keep the shortest len 
    object `type` at the be beginning: 
 
    :param gen_obj1: Can be a ``list``, ``dict`` or other `TypeVar` 
        classified objects.
    :param gen_obj2: Idem for `gen_obj1`.
    
    :Example: 
        
        >>> from watex.tools.mlutils import findDifferenceGenObject
        >>> findDifferenceGenObject(
        ...    ['ohmS', 'lwi', 'power', 'id', 'sfi', 'magnitude'], 
        ...    {'ohmS', 'lwi', 'power'})
        [out]:
        ...  {'ohmS', 'lwi', 'power'}
    
    """
    if len(gen_obj1) < len(gen_obj2):
        objType = type(gen_obj1)
        return objType(set(gen_obj2).difference(set(gen_obj1)))
    elif len(gen_obj1) > len(gen_obj2):
        objType = type(gen_obj2)
        return objType(set(gen_obj1).difference(set(gen_obj2)))
    else: return 
   
        
    
    return set(gen_obj1).difference(set(gen_obj2))
    
def featureExistError(superv_features: Iterable[T], 
                      features:Iterable[T]) -> None:
    """
    Catching feature existence errors.
    
    check error. If nothing occurs  then pass 
    
    :param superv_features: 
        list of features presuming to be controlled or supervised
        
    :param features: 
        List of all features composed of pd.core.DataFrame. 
    
    """
    for ii, supff in enumerate([superv_features, features ]): 
        if isinstance(supff, str): 
            if ii==0 : superv_features=[superv_features]
            if ii==1 :features =[superv_features]
            
    try : 
        resH= cfexist(features_to= superv_features,
                           features = features)
    except TypeError: 
        
        print(' Features can not be a NoneType value.'
              'Please set a right features.')
        _logger.error('NoneType can not be a features!')
    except :
        raise ParameterNumberError  (
           f'Parameters number of {features} is  not found in the '
           ' dataframe columns ={0}'.format(list(features)))
    
    else: 
        if not resH:  raise ParameterNumberError  (
            f'Parameters number is ``{features}``. NoneType object is'
            ' not allowed in  dataframe columns ={0}'.
            format(list(features)))
        
def controlExistingEstimator(
        estimator_name: str , raise_err =False ) -> Union [Dict[str, T], None]: 
    """ 
    When estimator name is provided by user , will chech the prefix 
    corresponding

    Catching estimator name and find the corresponding prefix 
        
    :param estimator_name: Name of given estimator 
    
    :Example: 
        
        >>> from watex.tools.mlutils import controlExistingEstimator 
        >>> test_est =controlExistingEstimator('svm')
        ('svc', 'SupportVectorClassifier')
        
    """
    estimator_name = str(estimator_name).lower().strip() 
    e = None ; efx = None 
    for k, v in _estimators.items() : 
        v_ = list(map(lambda o: str(o).lower(), v)) 
        
        if estimator_name in v_ : 
            e, efx = k, v[0]
            break 
    ef = list(map(lambda o: o[0], _estimators.values() ))
    
    if e is None: 
        _logger.error(
            f'Default estimator `{estimator_name}` not found ! Expect:'
            ' {}'.format(formatGenericObj(ef)).format(*ef))
        warnings.warn(f'Default estimator `{estimator_name}` not available.'
            ' Expect: {}'.format(formatGenericObj(ef)).format(*ef))
        if raise_err: 
            raise EstimatorError(f'Unsupport estimator {estimator_name!r}')
            
        return 
    
    return e, efx 

    
def formatModelScore(
        model_score: Union [float, Dict[str, float]] = None,
        select_estimator: str = None ) -> None   : 
    """
    Format the result of `model_score`
        
    :param model_score: Can be float or dict of float where key is 
                        the estimator name 
    :param select_estimator: Estimator name 
    
    :Example: 
        
        >>> from watex.tools.mlutils import formatModelScore 
        >>>  formatModelScore({'DecisionTreeClassifier':0.26, 
                      'BaggingClassifier':0.13}
        )
    """ 
    print('-'*77)
    if isinstance(model_score, dict): 
        for key, val in model_score.items(): 
            print('> {0:<30}:{1:^10}= {2:^10} %'.format( key,' Score', round(
                val *100,3 )))
    else : 
        if select_estimator is None : 
            select_estimator ='___'
        if inspect.isclass(select_estimator): 
            select_estimator =select_estimator.__class__.__name__
        
        try : 
            _, select_estimator = controlExistingEstimator(select_estimator)
        
        except : 
            if select_estimator is None :
                select_estimator =str(select_estimator)
            else: select_estimator = '___'
            
        print('> {0:<30}:{1:^10}= {2:^10} %'.format(select_estimator,
                     ' Score', round(
            model_score *100,3 )))
        
    print('-'*77)
    
def predict(
        y_true: ArrayLike,
        y_pred: ArrayLike =None,
        *, 
        X_: Optional [NDArray]=None, 
        clf:Optional [F[T]]=None,
        verbose:int =0
) -> Tuple[float, float]: 
    """ Make a quick statistic after prediction. 
    
    :param y_true: array-like 
        y value (label) to predict
    :param y_pred: array_like
        y value predicted
    :pram X: ndarray(nexamples, nfeatures)
        Training data sets 
    :param X_: ndarray(nexamples, nfeatures)
        test sets 
    :param clf: callable
        Estimator or classifier object. 
    :param XT_: ndarray
    :param verbose:int, level=0 
        Control the verbosity. More than 1 more message
    :param from_c: str 
        Column to visualize statistic. Be sure the colum exist into the
        test sets. If not raise errors.
    """
    
    clf_name =''
    if y_pred is None: 
        if clf is None: 
            warnings.warn('None estimator found! Could not predict `y` ')
            _logger.error('NoneType `clf` <estimator> could not'
                                ' predict `y`.')
            raise ValueError('None estimator detected!'
                             ' could not predict `y`.') 
        # check whether is 
        is_clf = hasattr(clf, '__call__')
        if is_clf : clf_name = clf.__name__
        if not is_clf :
            # try whether is ABCMeta class 
            try : 
                is_clf = hasattr(clf.__class__, '__call__')
            except : 
                raise TypeError(f"{clf!r} is not a model estimator. "
                                 " Could not use for prediction.")
            clf_name = clf.__class__.__name__
            # check estimator 
        if X_ is None: 
            raise TypeError('NoneType can not used for prediction.'
                            ' Need a test set `X`.')
        clf.fit(X_, y_true)
        y_pred = clf.predict(X_)
        
    if len(y_true) !=len(y_pred): 
        raise TypeError("`y_true` and `y_pred` must have the same length." 
                        f" {len(y_true)!r} and {len(y_pred)!r} were given"
                        " respectively.")
        
    # get the model score apres prediction 
    clf_score = round(sum(y_true ==y_pred)/len(y_true), 4)
    dms = f"Overall model {clf_name!r} score ={clf_score *100 } % "

    conf_mx =confusion_matrix(y_true, y_pred)
    if verbose >1:
        dms +=f"\n Confusion matrix= \n {conf_mx}"
    mse = mean_squared_error(y_true, y_pred )

    dms += f"\n MSE error = {mse }."
    pprint(dms)

    return clf_score, mse 



def write_excel(
        listOfDfs: List[DataFrame],
        csv: bool =False , 
        sep:str =',') -> None: 
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
    
   
def fetchGeoDATA (
    data_url:str = DATA_URL,
    data_path:str =DATA_PATH,
    tgz_filename:str  =TGZ_FILENAME
   ) -> None: 
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
    
def fetchTGZDatafromURL (
    data_url:str =DATA_URL, 
    data_path:str =DATA_PATH,
    tgz_file=TGZ_FILENAME, 
    file_to_retreive=None,
    **kws
    ) -> Union [str, None]: 
    """ Fetch data from data repository in zip of 'targz_file. 
    
    I will create a `datasets/data` directory in your workspace, downloading
     the `~.tgz_file and extract the `data.csv` from this directory.
    
    :param data_url: url to the datafilename where `tgz` filename is located  
    :param data_path: absolute path to the `tgz` filename 
    :param filename: `tgz` filename. 
    """
    f= None
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
            print(f"Unable to fetch {os.path.basename(tgz_file)!r}"
                  f" from <{data_url}>")
            
        return False 
    
    if file_to_retreive is not None: 
        f= fetchSingleTGZData(filename=file_to_retreive, **kws)
        
    return f

def fetchSingleTGZData(
        tgz_file: str , 
        filename: str ='___fmain.bagciv.data.csv',
        savefile: str ='data/geo_fdata',
        rename_outfile: Optional [str]=None 
        ) -> str :
    """ Fetch single file from archived tar file and rename a file if possible.
    
    :param tgz_file: str or Path-Like obj 
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
    if not os.path.isdir(savefile):
        os.makedirs(savefile)
    
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
            
        print(f"---> {os.path.join(savefile, rename_outfile)!r} was "
              f" successfully decompressed from {os.path.basename(tgz_file)!r}"
              f"and saved to {savefile!r}")
        
    return os.path.join(savefile, rename_outfile)
    
def load_data (
        data_path: str = DATA_PATH,
        filename: str =CSV_FILENAME,
        sep: str  =',' )-> DataFrame:
    """ Load CSV file to pd.dataframe. 
    
    :param data_path: path to data file 
    :param filename: name of file. 
    
    """ 
    if os.path.isfile(data_path): 
        return pd.read_csv(data_path, sep)
    
    csv_path = os.path.join(data_path , filename)
    
    return pd.read_csv(csv_path, sep)


def split_train_test (
        df:DataFrame[DType[T]],
        test_ratio:float 
        )-> Tuple [DataFrame[DType[T]]]: 
    """ A naive dataset split into train and test sets from a ratio and return 
    a shuffled train set and test set.
        
    :param df: a dataframe containing features 
    :param test_ratio: a ratio for test set batch. `test_ratio` is ranged 
        between 0 to 1. Default is 20%.
        
    :returns: a tuple of train set and test set. 
    
    """
    if isinstance (test_ratio, str):
        if test_ratio.lower().find('%')>=0: 
            try: test_ratio = float(test_ratio.lower().replace('%', ''))/100.
            except: TypeError (f"Could not convert value to float: {test_ratio!r}")
    if test_ratio <=0: 
        raise ValueError ("Invalid ratio. Must greater than 0.")
    elif test_ratio >=1: 
        raise ValueError("Invalid ratio. Must be less than 1 and greater than 0.")
        
    shuffled_indices =np.random.permutation(len(df)) 
    test_set_size = int(len(df)* test_ratio)
    test_indices = shuffled_indices [:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    return df.iloc[train_indices], df.iloc[test_indices]
    
def test_set_check_id (
        identifier:int, 
        test_ratio: float , 
        hash:F[T]
        ) -> bool: 
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

def split_train_test_by_id(
        data:DataFrame,
        test_ratio:float,
        id_column:Optional[List[int]]=None,
         hash : F =hashlib.md5
         )-> Tuple[ Sub[DataFrame[DType[T]]], Sub[DataFrame[DType[T]]]] : 
    """
    Ensure that data will remain consistent accross multiple runs, even if 
    dataset is refreshed. 
    
    The new testset will contain 20%of the instance, but it will not contain 
    any instance that was previously in the training set.

    :param data: Pandas.core.DataFrame 
    :param test_ratio: ratio of data to put in testset 
    :param id_colum: identifier index columns. If `id_column` is None,  reset  
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

def discretizeCategoriesforStratification(
        data: Union [ArrayLike, DataFrame],
        in_cat:str =None,
        new_cat:Optional [str] = None, 
        **kws
        ) -> DataFrame: 
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

def stratifiedUsingDiscretedCategories(
        data: Union [ArrayLike, DataFrame],
        cat_name:str , 
        n_splits:int =1, 
        test_size:float= 0.2, 
        random_state:int = 42
        )-> Tuple[ Sub[DataFrame[DType[T]]], Sub[DataFrame[DType[T]]]]: 
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

def fetch_model(
        modelfile: str ,
        modelpath:Optional[str] = None,
        default:bool =True,
        modname: Optional[str] =None,
        verbose:int =0): 
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
        raise FileNotFoundError (f"File {modelfile!r} not found!")
        
    from_joblib =False 
    if modelfile.endswith('.pkl'): from_joblib  =True 
    
    if from_joblib:
       _logger.info(f"Loading models `{os.path.basename(modelfile)}`!")
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
        _logger.info(f"Loading models `{os.path.basename(modelfile)}`!")
        try: 
           # DeSerializing pickled data 
           with open(modelfile, 'rb') as modf: 
               pickedfname= pickle.load (modf)
           _logger.info(f"Model `{os.path.basename(modelfile)!r} deserialized"
                         "  using Python pickle module.`!")
           
           dmsg=f'Model `{modelfile!r} deserizaled from  {modelfile}`!'
        except: 
            dmsg =''.join([" Unable to deserialized the "
                           f"{os.path.basename(modelfile)!r}"])
           
        else: 
            _logger.info(dmsg)   

    if verbose > 0: 
        pprint(
            dmsg 
            )
           
    if modname is not None: 
        keymess = f"{modname!r} not found."
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
                                          f"names ={list(pickedfname.keys())!r}")
        
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
        
    _logger.info(f'Dumping data to `{filename}`!')    
    try : 
        if to is None or to =='joblib':
            joblib.dump(data, f'{filename}.pkl')
            
            filename +='.pkl'
            _logger.info(f'Data dumped in `{filename} using '
                          'to `~.externals.joblib`!')
        elif to =='pypickle': 
            # force to move pickling data  to exception and write using 
            # Python pickle module
            raise 
    except : 
        # Now try to pickle data Serializing data 
        # Using HIGHEST_PROTOCOL is almost 2X faster and creates a file that
        # is ~10% smaller.  Load times go down by a factor of about 3X.
        with open(filename, 'wb') as wfile: 
            pickle.dump( data, wfile, protocol=pickle.HIGHEST_PROTOCOL) 
        _logger.info( 'Data are well serialized ')
        
    if savepath is not None:
        try : 
            savepath = savepath_ (savepath)
        except : 
            savepath = savepath_ ('_dumpedData_')
        try:
            shutil.move(filename, savepath)
        except :
            print(f"--> It seems destination path {filename!r} already exists.")

    if savepath is None:
        savepath =os.getcwd()
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

    _filename = os.path.basename(filename)
    _logger.info(f"Loading data from `{_filename}`!")
   
    data =None 
    try : 
        data= joblib.load(filename)
        _logger.info(''.join([f"Data from {_filename !r} are sucessfully", 
                      " loaded using ~.externals.joblib`!"]))
    except : 
        _logger.info(
            ''.join([f"Nothing to reload. It's seems data from {_filename!r}", 
                      " are not really dumped using ~external.joblib module!"])
            )
        # Try DeSerializing using pickle module
        with open(filename, 'rb') as tod: 
            data= pickle.load (tod)
        _logger.info(f"Data from `{_filename!r} are well"
                      " deserialized using Python pickle module.`!")
        
    is_none = data is None
    if is_none: 
        print("Unable to deserialize data. Please check your file.")
    else : print(f"Data from {_filename} have been sucessfully reloaded.")
    
    return data 

def subprocess_module_installation (module, upgrade =True ): 
    """ Install  module using subprocess.
    :param module: str, module name 
    :param upgrade:bool, install the lastest version.
    """
    import sys 
    import subprocess 
    #implement pip as subprocess 
    # refer to https://pythongeeks.org/subprocess-in-python/
    MOD_IMP=False 
    print(f'---> Module {module!r} installation will take a while,'
          ' please be patient...')
    cmd = f'<pip install {module}> | <python -m pip install {module}>'
    try: 

        upgrade ='--upgrade' if upgrade else ''
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
        f'{module}', f'{upgrade}'])
        reqs = subprocess.check_output([sys.executable,'-m', 'pip',
                                        'freeze'])
        [r.decode().split('==')[0] for r in reqs.split()]
        _logger.info(f"Intallation of `{module}` and dependancies"
                     "was successfully done!") 
        MOD_IMP=True
     
    except: 
        _logger.error(f"Fail to install the module =`{module}`.")
        print(f'---> Module {module!r} installation failed, Please use'
           f'  the following command {cmd} to manually install it.')
    return MOD_IMP 
        
                
def _assert_sl_target (target,  df=None, obj=None): 
    """ Check whether the target name into the dataframe for supervised 
    learning.
    
    :param df: dataframe pandas
    :param target: str or index of the supervised learning target name. 
    
    :Example: 
        
        >>> from watex.utils.ml_utils import _assert_sl_target
        >>> from watex.datasets import fetch_data
        >>> data = fetch_data('Bagoue original').get('data=df')  
        >>> _assert_sl_target (target =12, obj=prepareObj, df=data)
        ... 'flow'
    """
    is_dataframe = isinstance(df, pd.DataFrame)
    is_ndarray = isinstance(df, np.ndarray)
    if is_dataframe :
        targets = smart_format(
            df.columns if df.columns is not None else [''])
    else:targets =''
    if target is None:
        nameObj=f'{obj.__class__.__name__}'if obj is not None else 'Base class'
        msg =''.join([
            f"{nameObj!r} {'basically' if obj is not None else ''}"
            " works with surpervised learning algorithms so the",
            " input target is needed. Please specify the target", 
            f" {'name' if is_dataframe else 'index' if is_ndarray else ''}", 
            " to take advantage of the full functionalities."
            ])
        if is_dataframe:
            msg += f" Select the target among {targets}."
        elif is_ndarray : 
            msg += f" Max columns size is {df.shape[1]}"

        warnings.warn(msg, UserWarning)
        _logger.warning(msg)
        
    if target is not None: 
        if is_dataframe: 
            if isinstance(target, str):
                if not target in df.columns: 
                    msg =''.join([
                        f"Wrong target value {target!r}. Please select "
                        f"the right column name: {targets}"])
                    warnings.warn(msg, category= UserWarning)
                    _logger.warning(msg)
                    target =None
            elif isinstance(target, (float, int)): 
                is_ndarray =True 
  
        if is_ndarray : 
            _len = len(df.columns) if is_dataframe else df.shape[1] 
            m_=f"{'less than' if target >= _len  else 'greater than'}" 
            if not isinstance(target, (float,int)): 
                msg =''.join([f"Wrong target value `{target}`!"
                              f" Object type is {type(df)!r}. Target columns", 
                              "  index should be given instead."])
                warnings.warn(msg, category= UserWarning)
                _logger.warning(msg)
                target=None
            elif isinstance(target, (float,int)): 
                target = int(target)
                if not 0 <= target < _len: 
                    msg =f" Wrong target index. Should be {m_} {str(_len-1)!r}."
                    warnings.warn(msg, category= UserWarning)
                    _logger.warning(msg) 
                    target =None
                    
            if df is None: 
                wmsg = ''.join([
                    f"No data found! `{target}` does not fit any data set.", 
                      "Could not fetch the target name.`df` argument is None.", 
                      " Need at least the data `numpy.ndarray|pandas.dataFrame`",
                      ])
                warnings.warn(wmsg, UserWarning)
                _logger.warning(wmsg)
                target =None
                
            target = list(df.columns)[target] if is_dataframe else target
    return target
        
def default_data_splitting(X, y=None, *,  test_size =0.2, target =None,
                           random_state=42, fetch_target =False,
                           **skws): 
    """ Splitting data function. 
    
    Split data into the training set and test set. If target `y` is not
    given and you want to consider a specific array as a target for 
    supervised learning, just turn `fetch_target` argument to ``True`` and 
    set the `target` argument as a numpy columns index or pandas dataframe
    colums name. 
    
    :param X: np.ndarray or pd.DataFrame 
    :param y: array_like 
    :param test_size: If float, should be between 0.0 and 1.0 and represent
        the proportion of the dataset to include in the test split. 
    :param random_state: int, Controls the shuffling applied to the data
        before applying the split. Pass an int for reproducible output across
        multiple function calls
    :param fetch_target: bool, use to retrieve the targetted value from 
        the whole data `X`. 
    :param target: int, str 
        If int itshould be the index of the targetted value otherwise should 
        be the columns name of pandas DataFrame.
    :param skws: additional scikit-lean keywords arguments 
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    
    :returns: list, length -List containing train-test split of inputs.
        
    :Example: 
        
        >>> from watex.datasets import fetch_data 
        >>> data = fetch_data ('Bagoue original').get('data=df')
        >>> X, XT, y, yT= default_data_splitting(data.values,
                                     fetch_target=True,
                                     target =12 )
        >>> X, XT, y, yT= default_data_splitting(data,
                             fetch_target=True,
                             target ='flow' )
        >>> X0= data.copy()
        >>> X0.drop('flow', axis =1, inplace=True)
        >>> y0 = data ['flow']
        >>> X, XT, y, yT= default_data_splitting(X0, y0)
    """

    if fetch_target: 
        target = _assert_sl_target (target, df =X)
        s='could not be ' if target is None else 'was succesffully '
        wmsg = ''.join([
            f"Target {'index' if isinstance(target, int) else 'value'} "
            f"{str(target)!r} {s} used to fetch the `y` value from "
            "the whole data set."])
        if isinstance(target, str): 
            y = X[target]
            X= X.copy()
            X.drop(target, axis =1, inplace=True)
        if isinstance(target, (float, int)): 
            y=X[:, target]
            X = np.delete (X, target, axis =1)
        warnings.warn(wmsg, category =UserWarning)
        
    __V= train_test_split(X, y, random_state=random_state, **skws) \
        if y is not None else train_test_split(
                X,random_state=random_state, **skws)
    if y is None: X, XT , yT = *__V,  None 
    else: X, XT, y, yT= __V
    
    return  X, XT, y, yT

    
def fetchModel(
        modelfile: str,
        modelpath: str = None,
        default: bool = True,
        modname: Optional[str] = None,
        verbose: int = 0
)-> object: 
    """ Fetch your model saved using Python pickle module or joblib module. 
    
    :param modelfile: str or Path-Like object 
        dumped model file name saved using `joblib` or Python `pickle` module.
    :param modelpath: path-Like object , 
        Path to model dumped file =`modelfile`
    :default: bool, 
        Model parameters by default are saved into a dictionary. When default 
        is ``True``, returns a tuple of pair (the model and its best parameters).
        If ``False`` return all values saved from `~.MultipleGridSearch`
       
    :modname: str 
        Is the name of model to retreived from dumped file. If name is given 
        get only the model and its best parameters. 
    :verbose: int, level=0 
        control the verbosity. More messages if greater than 0.
    
    :returns:
        - `model_class_params`: if default is ``True``
        - `pickledmodel`: model dumped and all parameters if default is `False`
        
    :Example: 
        >>> from watex.bases import fetch_model 
        >>> my_model, = fetchModel ('SVC__LinearSVC__LogisticRegression.pkl',
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
        raise FileNotFoundError (f"File {modelfile!r} not found!")
        
    from_joblib =False 
    if modelfile.endswith('.pkl'): from_joblib  =True 
    
    if from_joblib:
       _logger.info(f"Loading models `{os.path.basename(modelfile)}`!")
       try : 
           pickledmodel = joblib.load(modelfile)
           if len(pickledmodel)>=2 : 
               pickledmodel = pickledmodel[0]
           # and later ....
           # f'{pickfname}._loaded' = joblib.load(f'{pickfname}.pkl')
           dmsg=f"Model {modelfile !r} retreived from~.externals.joblib`!"
       except : 
           dmsg=''.join([f"Nothing to retreive. It's seems model {modelfile !r}", 
                         " not really saved using ~external.joblib module! ", 
                         "Please check your model filename."])
    
    if not from_joblib: 
        _logger.info(f"Loading models `{os.path.basename(modelfile)}`!")
        try: 
           # DeSerializing pickled data 
           with open(modelfile, 'rb') as modf: 
               pickledmodel= pickle.load (modf)
           _logger.info(f"Model `{os.path.basename(modelfile)!r} deserialized"
                         "  using Python pickle module.`!")
           
           dmsg=f"Model {modelfile!r} deserizaled from  {modelfile!r}!"
        except: 
            dmsg =''.join([" Unable to deserialized the "
                           f"{os.path.basename(modelfile)!r}"])
           
        else: 
            _logger.info(dmsg)   
           
    if verbose > 0: 
        pprint(
            dmsg 
            )
           
    if modname is not None: 
        keymess = "{modname!r} not found."
        try : 
            if default:
                model_class_params  =( pickledmodel[modname]['best_model'], 
                                   pickledmodel[modname]['best_params_'], 
                                   pickledmodel[modname]['best_scores'],
                                   )
            if not default: 
                model_class_params= pickledmodel.get(modname), 
                
        except KeyError as key_error: 
            warnings.warn(
                f"Model name {modname!r} not found in the list of dumped"
                f" models = {list(pickledmodel.keys()) !r}")
            raise KeyError from key_error(keymess + "Shoud try the model's"
                                          f"names ={list(pickledmodel.keys())!r}")
        
        if verbose > 0: 
            pprint('Should return a tuple of `best model` and the'
                   ' `model best parameters.')
           
        return model_class_params  
            
    if default:
        model_class_params =list()    
        
        for mm in pickledmodel.keys(): 
            try : 
                model_class_params.append((pickledmodel[mm]['best_model'], 
                                          pickledmodel[mm]['best_params_'],
                                          pickledmodel[modname]['best_scores']))
            except KeyError as key_error : 
                raise KeyError (f"Unable to retrieve {key_error.args[0]!r}")
                
        if verbose > 0: 
               pprint('Should return a list of tuple pairs:`best model`and '
                      ' `model best parameters.')
               
        return model_class_params, 

    return pickledmodel,       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
