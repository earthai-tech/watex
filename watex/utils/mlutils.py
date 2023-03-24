# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Learning utilities for data transformation, 
model learning and inspections. 
"""
from __future__ import annotations 
import os 
import copy 
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
from .._typing import (
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
    Series,
    Sub                 
)
from ..exceptions import ( 
    ParameterNumberError , 
    EstimatorError, 
    DatasetError
)
from ..exlib.sklearn import ( 
    train_test_split , 
    StratifiedShuffleSplit, 
    accuracy_score,
    confusion_matrix, 
    mean_squared_error , 
    classification_report ,
    f1_score,
    precision_recall_curve, 
    precision_score,
    recall_score, 
    roc_auc_score, 
    roc_curve, 
    SelectFromModel, 
    StandardScaler, 
    MinMaxScaler, 
    Normalizer, 
    SimpleImputer, 
    LabelBinarizer, 
    LabelEncoder, 
    OrdinalEncoder, 
    Pipeline, 
    FeatureUnion, 
    OneHotEncoder, 
    RobustScaler
)
from .funcutils import (
    _assert_all_types, 
    _isin, 
    savepath_, 
    smart_format, 
    str2columns, 
    is_iterable, 
    is_in_if, 
    to_numeric_dtypes
)
from .validator import ( 
    get_estimator_name , 
    check_array, 
    )

_logger = watexlog().get_watex_logger(__name__)

__all__=[ 
    "evalModel",
    "selectfeatures", 
    "getGlobalScore", 
    "split_train_test", 
    "correlatedfeatures", 
    "findCatandNumFeatures",
    "evalModel", 
    "cattarget", 
    "labels_validator", 
    "projection_validator", 
    "rename_labels_in" , 
    "naive_imputer", 
    "naive_scaler", 
    "select_feature_importances", 
    "make_naive_pipe", 
    "bi_selector", 
    "correlatedfeatures", 
    "exporttarget", 
    "predict", 
    "fetchGeoDATA", 
    "fetchModel", 
    "fetch_model", 
    "load_data", 
    "split_train_test_by_id", 
    "split_train_test", 
    "discretizeCategoriesforStratification", 
    "stratifiedUsingDiscretedCategories", 
    "dumpOrSerializeData", 
    "loadDumpedOrSerializedData", 
    "default_data_splitting", 
    "findCatandNumFeatures", 
    
    ]


_scorers = { 
    "classification_report":classification_report,
    'precision_recall': precision_recall_curve,
    "confusion_matrix":confusion_matrix,
    'precision': precision_score,
    "accuracy": accuracy_score,
    "mse":mean_squared_error, 
    "recall": recall_score, 
    'auc': roc_auc_score, 
    'roc': roc_curve, 
    'f1':f1_score,
    }

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
    'xgboost': ['ExtremeGradientBoosting', 'xgboost', 'gboost', 'gbdm', 'xgb'], 
     'logit': ['LogisticRegression', 'logit', 'lr', 'logreg'], 
     'extree': ['ExtraTreesClassifier', 'extree', 'xtree', 'xtr']
        }  
#------
def evalModel(
        model: F, 
        X:NDArray |DataFrame, 
        y: ArrayLike |Series, 
        Xt:NDArray |DataFrame, 
        yt:ArrayLike |Series=None, 
        scorer:str | F = 'accuracy',
        eval:bool =False,
        **kws
    ): 
    """ Evaluate model and quick test the score with metric scorers. 
    
    Parameters
    --------------
    model: Callable, {'preprocessor + estimator } | estimator,
        the preprocessor is list of step for data handling all encapsulated 
        on the pipeline. model can also be a simple estimator with `fit`,
        
    X: N-d array, shape (N, M) 
       the training set composed of N-columns and the M-samples. The 
        feature set excludes the target `y`. 
    y: arraylike , shape (M)
        the target is composed of M-examples in supervised learning. 
    
    Xt: N-d array, shape (N, M) 
        test set array composed of N-columns and the M-samples. The 
        feature set excludes the target `y`. 
    yt: arraylike , shape (M)
        test label (or test target)  composed of M-examples in 
        supervised learning.
        
    scorer: str, Callable, 
        a scorer is a metric  function for model evaluation. If given as string 
        it should be the prefix of the following metrics: 
            
            * "classification_report"     -> for classification_report,
            * 'precision_recall'          -> for precision_recall_curve,
            * "confusion_matrix"          -> for a confusion_matrix,
            * 'precision'                 -> for  precision_score,
            * "accuracy"                  -> for  accuracy_score
            * "mse"                       -> for mean_squared_error, 
            * "recall"                    -> for  recall_score, 
            * 'auc'                       -> for  roc_auc_score, 
            * 'roc'                       -> for  roc_curve 
            * 'f1'                        -> for f1_score,
            
        Other string prefix values should raises an errors 
        
    kws: dict, 
        Additionnal keywords arguments from scklearn metric function.
        
    Returns 
    ----------
    Tuple : (score, ypred)
        the model score or the predicted y if `predict` is set to ``True``. 
        
    """

    score = None 
    if X.ndim ==1: 
        X = X.reshape(-1, 1) 
    if Xt.ndim ==1: 
        Xt = Xt.reshape(-1, 1)
        
    model.fit(X, y)
    # model.transform(X, y)
    ypred = model.predict(Xt)
    
    if eval : 
        if yt is None: 
            raise TypeError(" NoneType 'yt' cannot be used for model evaluation.")
            
        if scorer is None: 
           scorer =  _scorers['accuracy']
           
        if isinstance (scorer, str): 
            if str(scorer) not in _scorers.keys(): 
                raise ValueError (
                    "Given scorer {scorer!r }is unknown. Accepts "
                    f" only {smart_format(_scorers.keys())}") 
                
            scorer = _scorers.get(scorer)
        elif not hasattr (scorer, '__call__'): 
            raise TypeError ("scorer should be a callable object,"
                             f" got {type(scorer).__name__!r}")
            
        score = scorer (yt, ypred, **kws)
    
    return  ypred, score  

def correlatedfeatures(
        df:DataFrame ,
        corr:str ='pearson', 
        threshold: float=.95 , 
        fmt: bool= False 
        )-> DataFrame: 
    """Find the correlated features/columns in the dataframe. 
    
    Indeed, highly correlated columns don't add value and can throw off 
    features importance and interpretation of regression coefficients. If we  
    had correlated columns, choose to remove either the columns from  
    level_0 or level_1 from the features data is a good choice. 
    
    Parameters 
    -----------
    df: Dataframe or shape (M, N) from :class:`pandas.DataFrame` 
        Dataframe containing samples M  and features N
    corr: str, ['pearson'|'spearman'|'covariance']
        Method of correlation to perform. Note that the 'person' and 
        'covariance' don't support string value. If such kind of data 
        is given, turn the `corr` to `spearman`. *default* is ``pearson``
        
    threshold: int, default is ``0.95``
        the value from which can be considered as a correlated data. Should not 
        be greater than 1. 
        
    fmt: bool, default {``False``}
        format the correlated dataframe values 
        
    Returns 
    ---------
    df: `pandas.DataFrame`
        Dataframe with cilumns equals to [level_0, level_1, pearson]
        
    Examples
    --------
    >>> from watex.utils.mlutils import correlatedcolumns 
    >>> df_corr = correlatedcolumns (data , corr='spearman',
                                     fmt=None, threshold=.95
                                     )
    """
    th= copy.deepcopy(threshold) 
    threshold = str(threshold)  
    try : 
        threshold = float(threshold.replace('%', '')
                          )/1e2  if '%' in threshold else float(threshold)
    except: 
        raise TypeError (
            f"Threshold should be a float value, got: {type(th).__name__!r}")
          
    if threshold >= 1 or threshold <= 0 : 
        raise ValueError (
            f"threshold must be ranged between 0 and 1, got {th!r}")
      
    if corr not in ('pearson', 'covariance', 'spearman'): 
        raise ValueError (
            f"Expect ['pearson'|'spearman'|'covariance'], got{corr!r} ")
    # collect numerical values and exclude cat values 
    df = selectfeatures(df, include ='number')
        
    # use pipe to chain different func applied to df 
    c_df = ( 
        df.corr()
        .pipe(
            lambda df1: pd.DataFrame(
                np.tril (df1, k=-1 ), # low triangle zeroed 
                columns = df.columns, 
                index =df.columns, 
                )
            )
            .stack ()
            .rename(corr)
            .pipe(
                lambda s: s[
                    s.abs()> threshold 
                    ].reset_index()
                )
                .query("level_0 not in level_1")
        )

    return  c_df.style.format({corr :"{:2.f}"}) if fmt else c_df 

                           
def exporttarget (df, tname, inplace = True): 
    """ Extract target and modified data in place or not . 
    
    :param df: A dataframe with features including the target name `tname`
    :param tname: A target name. It should be include in the dataframe columns 
        otherwise an error is raised. 
    :param inplace: modified the dataframe inplace. if ``False`` return the 
        dataframe. the *defaut* is ``True`` 
        
    :returns: Tuple of the target and dataframe (modified or not)
    
    :example: 
    >>> from watex.datasets import fetch_data '
    >>> from watex.utils.mlutils import exporttarget 
    >>> data0 = fetch_data ('bagoue original').get('data=dfy1') 
    >>> # no modification 
    >>> target, data_no = exporttarget (data0 , 'sfi', False )
    >>> len(data_no.columns ) , len(data0.columns ) 
    ... (13, 13)
    >>> # modified in place 
    >>> target, data= exporttarget (data0 , 'sfi')
    >>> len(data.columns ) , len(data0.columns ) 
    ... (12, 12)
        
    """
    df = _assert_all_types(df, pd.DataFrame)
    existfeatures(df, tname) # assert tname 
    if is_iterable(tname, exclude_string=True): 
        tname = list(tname)
        
    t = df [tname ] 
    df.drop (tname, axis =1 , inplace =inplace )
    
    return t, df
    
    
def existfeatures (df, features, error='raise'): 
    """Control whether the features exist or not  
    
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
    :param features: list of features to select. List of features must be in the 
        dataframe otherwise an error occurs. 
    :param include: the type of data to retrieve in the dataframe `df`. Can  
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
    :type cvres: dict of Array-like, Shape (N, ) 
    :returns: tuple 
        ( mean_test_scores', 'std_test_scores') 
         scores on test_dcore and standard deviation scores 
        
    """
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
        
        >>> from watex.utils.mlutils import formatGenericObj 
        >>> formatGenericObj ({'ohmS', 'lwi', 'power', 'id', 
        ...                         'sfi', 'magnitude'})
        
    """
    
    return ['{0}{1}{2}'.format('{', ii, '}') for ii in range(
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
        
        >>> from watex.utils.mlutils import findIntersectionGenObject
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
        
        >>> from watex.utils.mlutils import findDifferenceGenObject
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
        
        >>> from watex.utils.mlutils import controlExistingEstimator 
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

    if e is None: 
        ef = map(lambda o: o[0], _estimators.values() )
        if raise_err: 
            raise EstimatorError(f'Unsupport estimator {estimator_name!r}.'
                                 f' Expect {smart_format(ef)}') 
        ef =list(ef)
        emsg = f"Default estimator {estimator_name!r} not found!" +\
            (" Expect: {}".format(formatGenericObj(ef)
                                  ).format(*ef))

        warnings.warn(emsg)
        
            
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
        
        >>> from watex.utils.mlutils import formatModelScore 
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

    dms += f"\n MSE error = {mse}."
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
    data_url:str ,
    data_path:str ,
    tgz_filename:str 
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
    data_url:str , 
    data_path:str ,
    tgz_file, 
    file_to_retreive=None,
    **kws
    ) -> Union [str, None]: 
    """ Fetch data from data repository in zip of 'targz_file. 
    
    I will create a `datasets/data` directory in your workspace, downloading
     the `~.tgz_file and extract the `data.csv` from this directory.
    
    :param data_url: url to the datafilename where `tgz` filename is located  
    :param data_path: absolute path to the `tgz` filename 
    :param filename: `tgz` filename. 
    
    :example: 
    >>> from watex.utils.mlutils import fetchTGZDatafromURL
    >>> DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/WEgeophysics/watex/master/'
    >>> # from Zenodo: 'https://zenodo.org/record/5560937#.YWQBOnzithE'
    >>> DATA_PATH = 'data/__tar.tgz'  # 'BagoueCIV__dataset__main/__tar.tgz_files__'
    >>> TGZ_FILENAME = '/fmain.bagciv.data.tar.gz'
    >>> CSV_FILENAME = '/__tar.tgz_files__/___fmain.bagciv.data.csv'
    >>> fetchTGZDatafromURL (data_url= DATA_URL,
                            data_path=DATA_PATH,
                            tgz_filename=TGZ_FILENAME
                            ) 
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
        >>> from watex.utils.mlutils import fetchSingleTGZData
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
        data: str = None,
        delimiter: str  =None ,
        **kws
        )-> DataFrame:
    """ Load csv file to a frame. 
    
    :param data_path: path to data csv file 
    :param delimiter: str, item for data  delimitations. 
    :param kws: dict, additional keywords arguments passed to :class:`pandas.read_csv`
    :return: pandas dataframe 
    
    """ 
    if not os.path.isfile(data): 
        raise TypeError("Expect a valid CSV file.")
    if (os.path.splitext(data)[1].replace('.', '')).lower() !='csv': 
        raise ValueError("Read only a csv file.")
        
    return pd.read_csv(data, delimiter=delimiter, **kws) 


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
    Get the test set id and set the corresponding unique identifier. 
    
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
    keep_colindex:bool=True, 
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
    if isinstance(data, np.ndarray) : 
        data = pd.DataFrame(data) 
        if 'index' in data.columns: 
            data.drop (columns='index', inplace=True)
            
    if id_column is None: 
        id_column ='index' 
        data = data.reset_index() # adds an `index` columns
        
    ids = data[id_column]
    in_test_set =ids.apply(lambda id_:test_set_check_id(id_, test_ratio, hash))
    if not keep_colindex: 
        data.drop (columns ='index', inplace =True )
        
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
       if verbose: _logger.info(
               f"Loading models `{os.path.basename(modelfile)}`")
       try : 
           pickedfname = joblib.load(modelfile)
           # and later ....
           # f'{pickfname}._loaded' = joblib.load(f'{pickfname}.pkl')
           dmsg=f"Model {modelfile !r} retreived from~.externals.joblib`"
       except : 
           dmsg=''.join([f"Nothing to retrived. It's seems model {modelfile !r}", 
                         " not really saved using ~external.joblib module! ", 
                         "Please check your model filename."])
    
    if not from_joblib: 
        if verbose: _logger.info(
                f"Loading models `{os.path.basename(modelfile)}`")
        try: 
           # DeSerializing pickled data 
           with open(modelfile, 'rb') as modf: 
               pickedfname= pickle.load (modf)
           if verbose: _logger.info(
                   f"Model `{os.path.basename(modelfile)!r} deserialized"
                         "  using Python pickle module.`!")
           
           dmsg=f'Model `{modelfile!r} deserizaled from  {modelfile}`!'
        except: 
            dmsg =''.join([" Unable to deserialized the "
                           f"{os.path.basename(modelfile)!r}"])
           
        else: 
            if verbose: _logger.info(dmsg)   

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
        
        if verbose: 
            pprint('Should return a tuple of `best model` and the'
                   ' `model best parameters.')
           
        return model_class_params  
            
    if default:
        model_class_params =list()    
        
        for mm in pickedfname.keys(): 
            model_class_params.append((pickedfname[mm]['best_model'], 
                                      pickedfname[mm]['best_params_'],
                                      pickedfname[modname]['best_scores']))
    
        if verbose: 
               pprint('Should return a list of tuple pairs:`best model`and '
                      ' `model best parameters.')
               
        return model_class_params

    return pickedfname 

def dumpOrSerializeData (
        data , 
        filename=None, 
        savepath =None, 
        to=None, 
        verbose=0,
        ): 
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
        >>> from watex.utils.mlutils import dumpOrSerializeData
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
                             " Should be <joblib> or <pypickle>")
    # remove extension if exists
    if filename.endswith('.pkl'): 
        filename = filename.replace('.pkl', '')
        
    if verbose: _logger.info(f'Dumping data to `{filename}`!')    
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
        if verbose: _logger.info( 'Data are well serialized ')
        
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
        
    if verbose: 
        print(f"Data {'serialization' if to=='pypickle' else 'dumping'}"
          f" complete,  save to {savepath!r}")
   
def loadDumpedOrSerializedData (filename:str, verbose=0): 
    """ Load dumped or serialized data from filename 
    
    :param filename: str or path-like object 
        Name of dumped data file.
    :return: 
        Data loaded from dumped file.
        
    :Example:
        
        >>> from watex.utils.mlutils import loadDumpedOrSerializedData
        >>> loadDumpedOrSerializedData(filename ='Watex/datasets/__XTyT.pkl')
    """
    
    if not isinstance(filename, str): 
        raise TypeError(f'filename should be a <str> not <{type(filename)}>')
        
    if not os.path.isfile(filename): 
        raise FileExistsError(f"File {filename!r} does not exist.")

    _filename = os.path.basename(filename)
    if verbose: _logger.info(f"Loading data from `{_filename}`!")
   
    data =None 
    try : 
        data= joblib.load(filename)
        if verbose: _logger.info(
                ''.join([f"Data from {_filename !r} are sucessfully", 
                      " loaded using ~.externals.joblib`!"]))
    except : 
        if verbose: 
            _logger.info(
            ''.join([f"Nothing to reload. It's seems data from {_filename!r}", 
                      " are not really dumped using ~external.joblib module!"])
            )
        # Try DeSerializing using pickle module
        with open(filename, 'rb') as tod: 
            data= pickle.load (tod)
            
        if verbose: 
            _logger.info(f"Data from `{_filename!r}` are well"
                      " deserialized using Python pickle module!")
        
    is_none = data is None
    if is_none: 
        print("Unable to deserialize data. Please check your file.")

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
        
        >>> from watex.utils.mlutils import _assert_sl_target
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
                              " index should be given instead."])
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

def get_target(
    ar, /, 
    tname, 
    drop_target =True , 
    columns =None,
    as_frame=False 
    ): 
    """ Extract target from multidimensional array or dataframe.  
    
    Parameters 
    ------------
    ar: arraylike2d or pd.DataFrame 
      Array that supposed to contain the target value. 
      
    tname: int/str, list of int/str 
       index or the name of the target; if ``int`` is passed it should range 
       ranged less than the columns number of the array i.e. a shape[1] in 
       the case of np.ndarray. If the list of indexes or names are given, 
       the return target should be in two dimensional array. 
       
    drop_target: bool, default=True 
       Remove the target array in the 2D array or dataframe in the case 
       the target exists and returns a data exluding the target array. 
       
    columns: list, default=False. 
       composes the dataframe when the array is given rather than a dataframe. 
       The list of column names must match the number of columns in the 
       two dimensional array, otherwise an error occurs. 
       
    as_frame: bool, default=False, 
       returns dataframe/series or the target rather than array when the array 
       is supplied. This seems useful when column names are supplied. 
       
    Returns
    --------
    t, ar : array-like/pd.Series , array-like/pd.DataFrame 
      Return the targets and the array/dataframe of the target. 
      
    Examples 
    ---------
    >>>> import numpy as np 
    >>> import pandas as pd 
    >>> from watex.utils.mtutils import get_target 
    >>> ar = np.random.randn ( 3,  3 )
    >>> df0 = pd.DataFrame ( ar, columns = ['x1', 'x2', 'tname'])
    >>> df= df0.copy() 
    >>> get_target (df, 'tname', drop_target= False )
    (      tname
     0 -0.542861
     1  0.781198,
              x1        x2     tname
     0 -1.424061 -0.493320 -0.542861
     1  0.416050 -1.156182  0.781198)
    >>> get_target (df, [ 'tname', 'x1']) # drop is True by default
    (      tname        x1
     0 -0.542861 -1.424061
     1  0.781198  0.416050,
              x2
     0 -0.493320
     1 -1.156182)
    >>> df = df0.copy() 
    >>> # when array is passed 
    >>> get_target (df.values , '2', drop_target= False )
    (array([[-0.54286148],
            [ 0.7811981 ]]),
     array([[-1.42406091, -0.49331988, -0.54286148],
            [ 0.41605005, -1.15618243,  0.7811981 ]]))
    >>> get_target (df.values , 'tname') # raise error 
    ValueError: 'tname' ['tname'] is not valid...
    
    """
    emsg =("Array is passed.'tname' must be a list of indexes or column names"
           " that fit the shape[axis=1] of the given array. Expect {}, got {}.")
    emsgc =("'tname' {} {} not valid. Array is passed while columns are not "
            "supplied. Expect 'tname' in the range of numbers betwen 0- {}")
    is_arr=False 
    tname =[ str(i) for i in is_iterable(
        tname, exclude_string =True, transform =True)] 
    
    if isinstance (ar, np.ndarray): 
        columns = columns or [str(i) for i in range(ar.shape[1])]
        if len(columns) < ar.shape [1]: 
            raise ValueError(emsg.format(ar.shape[1], len(tname)))
        ar = pd.DataFrame (ar, columns = columns) 
        if not existfeatures(ar, tname, error='ignore'): 
            raise ValueError(emsgc.format(tname, "is" if len(tname)==1 else "are", 
                                         len(columns)-1)
                             )
        is_arr=True if not as_frame else False 
        
    t, ar =exporttarget(ar, tname , inplace = drop_target ) 

    return (t.values, ar.values ) if is_arr  else (t, ar) 
        
def default_data_splitting(X, y=None, *,  test_size =0.2, target =None,
                           random_state=42, fetch_target =False,
                           **skws): 
    """ Splitting data function naively. 
    
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
        
    V= train_test_split(X, y, random_state=random_state, **skws) \
        if y is not None else train_test_split(
                X,random_state=random_state, **skws)
    if y is None: 
        X, XT , yT = *V,  None 
    else: 
        X, XT, y, yT= V
    
    return  X, XT, y, yT

#XXX FIX IT
def fetchModel(
    file: str,
    *, 
    default: bool = True,
    name: Optional[str] = None,
    storage=None, 
)-> object: 
    """ Fetch your data/model saved using Python pickle or joblib module. 
    
    Parameters 
    ------------
    file: str or Path-Like object 
        dumped model file name saved using `joblib` or Python `pickle` module.
    path: path-Like object , 
        Path to model dumped file =`modelfile`
    default: bool, 
        Model parameters by default are saved into a dictionary. When default 
        is ``True``, returns a tuple of pair (the model and its best parameters).
        If ``False`` return all values saved from `~.MultipleGridSearch`
    storage: str, default='joblib'
        kind of module use to pickling the data
    name: str 
        Is the name of model to retreived from dumped file. If name is given 
        get only the model and its best parameters. 
        
    Returns
    --------
    - `data`: Tuple (Dict, )
        data composed of models, classes and params for 'best_model', 
        'best_params_' and 'best_scores' if default is ``True``,
        and model dumped and all parameters otherwise.

    Example
    ---------
        >>> from watex.bases import fetch_model 
        >>> my_model, = fetchModel ('SVC__LinearSVC__LogisticRegression.pkl',
                                    default =False,  modname='SVC')
        >>> my_model
    """
    
    if not os.path.isfile (file): 
        raise FileNotFoundError (f"File {file!r} not found. Please check"
                                 " your filename.")
    st = storage 
    if storage is None: 
        ex = os.path.splitext (file)[-1] 
        storage = 'joblib' if ex =='.joblib' else 'pickle'

    storage = str(storage).lower().strip() 
    
    assert storage in {"joblib", "pickle"}, (
        "Data pickling supports only the Python's built-in persistence"
        f" model'pickle' or 'joblib' as replacement of pickle: got{st!r}"
        )
    _logger.info(f"Loading models {os.path.basename(file)}")
    
    if storage =='joblib':
        pickledmodel = joblib.load(file)
        if len(pickledmodel)>=2 : 
            pickledmodel = pickledmodel[0]
    elif storage =='pickle': 
        with open(file, 'rb') as modf: 
            pickledmodel= pickle.load (modf)
            
    data= copy.deepcopy(pickledmodel)
    if name is not None: 
        name =_assert_all_types(name, str, objname="Model to pickle ")
        if name not in pickledmodel.keys(): 
            raise KeyError(
                f"Model {name!r} is missing in the dumped models."
                f" Available pickled models: {list(pickledmodel.keys())}"
                         )
        if default: 
            data =[pickledmodel[name][k] for k in (
                "best_model", "best_params_", "best_scores")
                ]
        else:
            # When using storage as joblib
            # trying to unpickle estimator directly other
            # format than dict from version 1.1.1 
            # might lead to breaking code or invalid results. 
            # Use at your own risk. For more info please refer to:
            # https://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations
            
            # pickling all data
            data= pickledmodel.get(name)
        
    return data,       

        
def findCatandNumFeatures( 
        df: DataFrame= None, 
        features: List[str]= None,  
        return_frames: bool= False 
        ) -> Tuple[List[str] | DataFrame, List[str] |DataFrame]: 
    """ 
    Retrieve the categorial or numerical features on whole features 
    of dataset. 
    
    Parameters 
    -----------
    df: Dataframe 
        Dataframe with columns composing the features
        
    features: list of str, 
        list of the column names. If the dataframe is big, can set the only 
        required features. If features are provided, frame should be shrunked 
        to match the only given features before the numerical and categorical 
        features search. Note that an error will raises if any of one features 
        is missing in the dataframe. 
        
    return_frames: bool, 
        if set to ``True``, it returns two separated dataframes (cat & num) 
        otherwise, it only returns the cat and num columns names. 
 
    Returns
    ---------
    Tuple:  `cat_features` and  `num_features` names or frames 
       
    Examples 
    ----------
    >>> from watex.datasets import fetch_data 
    >>>> from watex.tools import findCatandNumFeatures
    >>> data = fetch_data ('bagoue original').get('data=dfy2')
    >>> cat, num = findCatandNumFeatures(data)
    >>> cat, num 
    ... (['type', 'geol', 'shape', 'name', 'flow'],
     ['num', 'east', 'north', 'power', 'magnitude', 'sfi', 'ohmS', 'lwi'])
    >>> cat, num = findCatandNumFeatures(
        data, features = ['geol', 'ohmS', 'sfi'])
    ... (['geol'], ['ohmS', 'sfi'])
        
    """
    
    if features is None: 
        features = list(df.columns) 
        
    existfeatures(df, list(features))
    df = df[features].copy() 
    
    # get num features 
    num = selectfeatures(df, include = 'number')
    catnames = findDifferenceGenObject (df.columns, num.columns ) 

    return ( df[catnames], num) if return_frames else (
        list(catnames), list(num.columns)  )
   
        
def cattarget(
        arr :ArrayLike |Series , /, 
        func: F = None,  
        labels: int | List[int] = None, 
        rename_labels: Optional[str] = None, 
        coerce:bool=False,
        order:str='strict',
        ): 
    """ Categorize array to hold the given identifier labels. 
    
    Classifier numerical values according to the given label values. Labels 
    are a list of integers where each integer is a group of unique identifier  
    of a sample in the dataset. 
    
    Parameters 
    -----------
    arr: array-like |pandas.Series 
        array or series containing numerical values. If a non-numerical values 
        is given , an errors will raises. 
    func: Callable, 
        Function to categorize the target y.  
    labels: int, list of int, 
        if an integer value is given, it should be considered as the number 
        of category to split 'y'. For instance ``label=3`` applied on 
        the first ten number, the labels values should be ``[0, 1, 2]``. 
        If labels are given as a list, items must be self-contain in the 
        target 'y'.
    rename_labels: list of str; 
        list of string or values to replace the label integer identifier. 
    coerce: bool, default =False, 
        force the new label names passed to `rename_labels` to appear in the 
        target including or not some integer identifier class label. If 
        `coerce` is ``True``, the target array holds the dtype of new_array. 

    Return
    --------
    arr: Arraylike |pandas.Series
        The category array with unique identifer labels 
        
    Examples 
    --------

    >>> from watex.utils.mlutils import cattarget 
    >>> def binfunc(v): 
            if v < 3 : return 0 
            else : return 1 
    >>> arr = np.arange (10 )
    >>> arr 
    ... array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> target = cattarget(arr, func =binfunc)
    ... array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=int64)
    >>> cattarget(arr, labels =3 )
    ... array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    >>> array([2, 2, 2, 2, 1, 1, 1, 0, 0, 0]) 
    >>> cattarget(arr, labels =3 , order =None )
    ... array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    >>> cattarget(arr[::-1], labels =3 , order =None )
    ... array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]) # reverse does not change
    >>> cattarget(arr, labels =[0 , 2,  4]  )
    ... array([0, 0, 0, 2, 2, 4, 4, 4, 4, 4])

    """
    arr = _assert_all_types(arr, np.ndarray, pd.Series) 
    is_arr =False 
    if isinstance (arr, np.ndarray ) :
        arr = pd.Series (arr  , name = 'none') 
        is_arr =True 
        
    if func is not None: 
        if not  inspect.isfunction (func): 
            raise TypeError (
                f'Expect a function but got {type(func).__name__!r}')
            
        arr= arr.apply (func )
        
        return  arr.values  if is_arr else arr   
    
    name = arr.name 
    arr = arr.values 

    if labels is not None: 
        arr = _cattarget (arr , labels, order =order)
        if rename_labels is not None: 
            arr = rename_labels_in( arr , rename_labels , coerce =coerce ) 

    return arr  if is_arr else pd.Series (arr, name =name  )

def rename_labels_in (arr, new_names, coerce = False): 
    """ Rename label by a new names 
    
    :param arr: arr: array-like |pandas.Series 
         array or series containing numerical values. If a non-numerical values 
         is given , an errors will raises. 
    :param new_names: list of str; 
        list of string or values to replace the label integer identifier. 
    :param coerce: bool, default =False, 
        force the 'new_names' to appear in the target including or not some 
        integer identifier class label. `coerce` is ``True``, the target array 
        hold the dtype of new_array; coercing the label names will not yield 
        error. Consequently can introduce an unexpected results.
    :return: array-like, 
        An array-like with full new label names. 
    """
    
    if not is_iterable(new_names): 
        new_names= [new_names]
    true_labels = np.unique (arr) 
    
    if labels_validator(arr, new_names, return_bool= True): 
        return arr 

    if len(true_labels) != len(new_names):
        if not coerce: 
            raise ValueError(
                "Can't rename labels; the new names and unique label" 
                " identifiers size must be consistent; expect {}, got " 
                "{} label(s).".format(len(true_labels), len(new_names))
                             )
        if len(true_labels) < len(new_names) : 
            new_names = new_names [: len(new_names)]
        else: 
            new_names = list(new_names)  + list(
                true_labels)[len(new_names):]
            warnings.warn("Number of the given labels '{}' and values '{}'"
                          " are not consistent. Be aware that this could "
                          "yield an expected results.".format(
                              len(new_names), len(true_labels)))
            
    new_names = np.array(new_names)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # hold the type of arr to operate the 
    # element wise comparaison if not a 
    # ValueError:' invalid literal for int() with base 10' 
    # will appear. 
    if not np.issubdtype(np.array(new_names).dtype, np.number): 
        arr= arr.astype (np.array(new_names).dtype)
        true_labels = true_labels.astype (np.array(new_names).dtype)

    for el , nel in zip (true_labels, new_names ): 
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # element comparison throws a future warning here 
        # because of a disagreement between Numpy and native python 
        # Numpy version ='1.22.4' while python version = 3.9.12
        # this code is brittle and requires these versions above. 
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # suppress element wise comparison warning locally 
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            arr [arr == el ] = nel 
            
    return arr 

    
def _cattarget (ar , labels , order=None): 
    """ A shadow function of :func:`watex.utils.mlutils.cattarget`. 
    
    :param ar: array-like of numerical values 
    :param labels: int or list of int, 
        the number of category to split 'ar'into. 
    :param order: str, optional, 
        the order of label to be categorized. If None or any other values, 
        the categorization of labels considers only the length of array. 
        For instance a reverse array and non-reverse array yield the same 
        categorization samples. When order is set to ``strict``, the 
        categorization  strictly considers the value of each element. 
        
    :return: array-like of int , array of categorized values.  
    """
    # assert labels
    if is_iterable (labels):
        labels =[int (_assert_all_types(lab, int, float)) 
                 for lab in labels ]
        labels = np.array (labels , dtype = np.int32 ) 
        cc = labels 
        # assert whether element is on the array 
        s = set (ar).intersection(labels) 
        if len(s) != len(labels): 
            mv = set(labels).difference (s) 
            
            fmt = [f"{'s' if len(mv) >1 else''} ", mv,
                   f"{'is' if len(mv) <=1 else'are'}"]
            warnings.warn("Label values must be array self-contain item. "
                           "Label{0} {1} {2} missing in the array.".format(
                               *fmt)
                          )
            raise ValueError (
                "label value{0} {1} {2} missing in the array.".format(*fmt))
    else : 
        labels = int (_assert_all_types(labels , int, float))
        labels = np.linspace ( min(ar), max (ar), labels + 1 ) #+ .00000001 
        #array([ 0.,  6., 12., 18.])
        # split arr and get the range of with max bound 
        cc = np.arange (len(labels)) #[0, 1, 3]
        # we expect three classes [ 0, 1, 3 ] while maximum 
        # value is 18 . we want the value value to be >= 12 which 
        # include 18 , so remove the 18 in the list 
        labels = labels [:-1] # remove the last items a
        # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2]) # 3 classes 
        #  array([ 0.        ,  3.33333333,  6.66666667, 10. ]) + 
    # to avoid the index bound error 
    # append nan value to lengthen arr 
    r = np.append (labels , np.nan ) 
    new_arr = np.zeros_like(ar) 
    # print(labels)
    ar = ar.astype (np.float32)

    if order =='strict': 
        for i in range (len(r)):
            if i == len(r) -2 : 
                ix = np.argwhere ( (ar >= r[i]) & (ar != np.inf ))
                new_arr[ix ]= cc[i]
                break 
            
            if i ==0 : 
                ix = np.argwhere (ar < r[i +1])
                new_arr [ix] == cc[i] 
                ar [ix ] = np.inf # replace by a big number than it was 
                # rather than delete it 
            else :
                ix = np.argwhere( (r[i] <= ar) & (ar < r[i +1]) )
                new_arr [ix ]= cc[i] 
                ar [ix ] = np.inf 
    else: 
        l= list() 
        for i in range (len(r)): 
            if i == len(r) -2 : 
                l.append (np.repeat ( cc[i], len(ar))) 
                
                break
            ix = np.argwhere ( (ar < r [ i + 1 ] ))
            l.append (np.repeat (cc[i], len (ar[ix ])))  
            # remove the value ready for i label 
            # categorization 
            ar = np.delete (ar, ix  )
            
        new_arr= np.hstack (l).astype (np.int32)  
        
    return new_arr.astype (np.int32)       
        
def projection_validator (X, Xt=None, columns =None ):
    """ Retrieve x, y coordinates of a datraframe ( X, Xt ) from columns 
    names or indexes. 
    
    If X or Xt are given as arrays, `columns` may hold integers from 
    selecting the the coordinates 'x' and 'y'. 
    
    Parameters 
    ---------
    X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        training set; Denotes data that is observed at training and prediction 
        time, used as independent variables in learning. The notation 
        is uppercase to denote that it is ordinarily a matrix. When a matrix, 
        each sample may be represented by a feature vector, or a vector of 
        precomputed (dis)similarity with each training sample. 

    Xt: Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        Shorthand for "test set"; data that is observed at testing and 
        prediction time, used as independent variables in learning. The 
        notation is uppercase to denote that it is ordinarily a matrix.
    columns: list of str or index, optional 
        columns is usefull when a dataframe is given  with a dimension size 
        greater than 2. If such data is passed to `X` or `Xt`, columns must
        hold the name to consider as 'easting', 'northing' when UTM 
        coordinates are given or 'latitude' , 'longitude' when latlon are 
        given. 
        If dimension size is greater than 2 and columns is None , an error 
        will raises to prevent the user to provide the index for 'y' and 'x' 
        coordinated retrieval. 
      
    Returns 
    -------
    ( x, y, xt, yt ), (xname, yname, xtname, ytname), Tuple of coordinate 
        arrays and coordinate labels 
 
    """
    # initialize arrays and names 
    init_none = [None for i in range (4)]
    x,y, xt, yt = init_none
    xname,yname, xtname, ytname = init_none 
    
    m="{0} must be an iterable object, not {1!r}"
    ms= ("{!r} is given while columns are not supplied. set the list of "
        " feature names or indexes to fetch 'x' and 'y' coordinate arrays." )
    
    # args = list(args) + [None for i in range (5)]
    # x, y, xt, yt, *_ = args 
    X =_assert_all_types(X, np.ndarray, pd.DataFrame ) 
    
    if Xt is not None: 
        Xt = _assert_all_types(Xt, np.ndarray, pd.DataFrame)
        
    if columns is not None: 
        if isinstance (columns, str): 
            columns = str2columns(columns )
        
        if not is_iterable(columns): 
            raise ValueError(m.format('columns', type(columns).__name__))
        
        columns = list(columns) + [ None for i in range (5)]
        xname , yname, xtname, ytname , *_= columns 

    if isinstance(X, pd.DataFrame):
      
        x, xname, y, yname = _validate_columns(X, xname, yname)
        
    elif isinstance(X, np.ndarray):
        x, y = _is_valid_coordinate_arrays (X, xname, yname )    
        
        
    if isinstance (Xt, pd.DataFrame) :
        # the test set holds the same feature names
        # as the train set 
        if xtname is None: 
            xtname = xname
        if ytname is None: 
            ytname = yname 
            
        xt, xtname, yt, ytname = _validate_columns(Xt, xname, yname)

    elif isinstance(Xt, np.ndarray):
        
        if xtname is None: 
            xtname = xname
        if ytname is None: 
            ytname = yname 
            
        xt, yt = _is_valid_coordinate_arrays (Xt, xtname, ytname , 'test')
        
    if (x is None) or (y is None): 
        raise ValueError (ms.format('X'))
    if Xt is not None: 
        if (xt is None) or (yt is None): 
            warnings.warn (ms.format('Xt'))

    return  (x, y , xt, yt ) , (
        xname, yname, xtname, ytname ) 
    

def _validate_columns (df, xni, yni ): 
    """ Validate the feature name  in the dataframe using either the 
    string litteral name of the index position in the columns.
    
    :param df: pandas.DataFrame- Dataframe with feature names as columns. 
    :param xni: str, int- feature name  or position index in the columns for 
        x-coordinate 
    :param yni: str, int- feature name  or position index in the columns for 
        y-coordinate 
    
    :returns: (x, ni) Tuple of (pandas.Series, and names) for x and y 
        coordinates respectively.
    
    """
    def _r (ni): 
        if isinstance(ni, str): # feature name
            existfeatures(df, ni ) 
            s = df[ni]  
        elif isinstance (ni, (int, float)):# feature index
            s= df.iloc[:, int(ni)] 
            ni = s.name 
        return s, ni 
        
    xs , ys = [None, None ]
    if df.ndim ==1: 
        raise ValueError ("Expect a dataframe of two dimensions, got '1'")
        
    elif df.shape[1]==2: 
       warnings.warn("columns are not specify while array has dimension"
                     "equals to 2. Expect indexes 0 and 1 for (x, y)"
                     "coordinates respectively.")
       xni= df.iloc[:, 0].name 
       yni= df.iloc[:, 1].name 
    else: 
        ms = ("The matrix of features is greater than 2. Need column names or"
              " indexes to  retrieve the 'x' and 'y' coordinate arrays." ) 
        e =' Only {!r} is given.' 
        me=''
        if xni is not None: 
            me =e.format(xni)
        if yni is not None: 
            me=e.format(yni)
           
        if (xni is None) or (yni is None ): 
            raise ValueError (ms + me)
            
    xs, xni = _r (xni) ;  ys, yni = _r (yni)
  
    return xs, xni , ys, yni 


def _validate_array_indexer (arr, index): 
    """ Select the appropriate coordinates (x,y) arrays from indexes.  
    
    Index is used  to retrieve the array of (x, y) coordinates if dimension 
    of `arr` is greater than 2. Since we expect x, y coordinate for projecting 
    coordinates, 1-d  array `X` is not acceptable. 
    
    :param arr: ndarray (n_samples, n_features) - if nfeatures is greater than 
        2 , indexes is needed to fetch the x, y coordinates . 
    :param index: int, index to fetch x, and y coordinates in multi-dimension
        arrays. 
    :returns: arr- x or y coordinates arrays. 

    """
    if arr.ndim ==1: 
        raise ValueError ("Expect an array of two dimensions.")
    if not isinstance (index, (float, int)): 
        raise ValueError("index is needed to coordinate array with "
                         "dimension greater than 2.")
        
    return arr[:, int (index) ]

def _is_valid_coordinate_arrays (arr, xind, yind, ptype ='train'): 
    """ Check whether array is suitable for projecting i.e. whether 
    x and y (both coordinates) can be retrived from `arr`.
    
    :param arr: ndarray (n_samples, n_features) - if nfeatures is greater than 
        2 , indexes is needed to fetch the x, y coordinates . 
        
    :param xind: int, index to fetch x-coordinate in multi-dimension
        arrays. 
    :param yind: int, index to fetch y-coordinate in multi-dimension
        arrays
    :param ptype: str, default='train', specify whether the array passed is 
        training or test sets. 
    :returns: (x, y)- array-like of x and y coordinates. 
    
    """
    xn, yn =('x', 'y') if ptype =='train' else ('xt', 'yt') 
    if arr.ndim ==1: 
        raise ValueError ("Expect an array of two dimensions.")
        
    elif arr.shape[1] ==2 : 
        x, y = arr[:, 0], arr[:, 1]
        
    else :
        msg=("The matrix of features is greater than 2; Need index to  "
             " retrieve the {!r} coordinate array in param 'column'.")
        
        if xind is None: 
            raise ValueError(msg.format(xn))
        else : x = _validate_array_indexer(arr, xind)
        if yind is None : 
            raise ValueError(msg.format(yn))
        else : y = _validate_array_indexer(arr, yind)
        
    return x, y         
        
def labels_validator (t, /, labels, return_bool = False): 
    """ Assert the validity of the label in the target  and return the label 
    or the boolean whether all items of label are in the target. 
    
    :param t: array-like, target that is expected to contain the labels. 
    :param labels: int, str or list of (str or int) that is supposed to be in 
        the target `t`. 
    :param return_bool: bool, default=False; returns 'True' or 'False' rather 
        the labels if set to ``True``. 
    :returns: bool or labels; 'True' or 'False' if `return_bool` is set to 
        ``True`` and labels otherwise. 
        
    :example: 
    >>> from watex.datasets import fetch_data 
    >>> from watex.utils.mlutils import cattarget, labels_validator 
    >>> _, y = fetch_data ('bagoue', return_X_y=True, as_frame=True) 
    >>> # binarize target y into [0 , 1]
    >>> ybin = cattarget(y, labels=2 )
    >>> labels_validator (ybin, [0, 1])
    ... [0, 1] # all labels exist. 
    >>> labels_validator (y, [0, 1, 3])
    ... ValueError: Value '3' is missing in the target.
    >>> labels_validator (ybin, 0 )
    ... [0]
    >>> labels_validator (ybin, [0, 5], return_bool=True ) # no raise error
    ... False
        
    """
    
    if not is_iterable(labels):
        labels =[labels] 
        
    t = np.array(t)
    mask = _isin(t, labels, return_mask=True ) 
    true_labels = np.unique (t[mask]) 
    # set the difference to know 
    # whether all labels are valid 
    remainder = list(set(labels).difference (true_labels))
    
    isvalid = True 
    if len(remainder)!=0 : 
        if not return_bool: 
            # raise error  
            raise ValueError (
                "Label value{0} {1} {2} missing in the target 'y'.".format ( 
                f"{'s' if len(remainder)>1 else ''}", 
                f"{smart_format(remainder)}",
                f"{'are' if len(remainder)> 1 else 'is'}")
                )
        isvalid= False 
        
    return isvalid if return_bool else  labels 
        
def bi_selector (d, /,  features =None, return_frames = False ):
    """ Auto-differentiates the numerical from categorical attributes.
    
    This is usefull to select the categorial features from the numerical 
    features and vice-versa when we are a lot of features. Enter features 
    individually become tiedous and a mistake could probably happenned. 
    
    Parameters 
    ------------
    d: pandas dataframe 
        Dataframe pandas 
    features : list of str
        List of features in the dataframe columns. Raise error is feature(s) 
        does/do not exist in the frame. 
        Note that if `features` is ``None``, it returns the categorical and 
        numerical features instead. 
        
    return_frames: bool, default =False 
        return the difference columns (features) from the given features  
        as a list. If set to ``True`` returns bi-frames composed of the 
        given features and the remaining features. 
        
    Returns 
    ----------
    - Tuple ( list, list)
        list of features and remaining features 
    - Tuple ( pd.DataFrame, pd.DataFrame )
        List of features and remaing features frames.  
            
    Example 
    --------
    >>> from watex.utils.mlutils import bi_selector 
    >>> from watex.datasets import load_hlogs 
    >>> data = load_hlogs().frame # get the frame 
    >>> data.columns 
    >>> Index(['hole_id', 'depth_top', 'depth_bottom', 'strata_name', 'rock_name',
           'layer_thickness', 'resistivity', 'gamma_gamma', 'natural_gamma', 'sp',
           'short_distance_gamma', 'well_diameter', 'aquifer_group',
           'pumping_level', 'aquifer_thickness', 'hole_depth_before_pumping',
           'hole_depth_after_pumping', 'hole_depth_loss', 'depth_starting_pumping',
           'pumping_depth_at_the_end', 'pumping_depth', 'section_aperture', 'k',
           'kp', 'r', 'rp', 'remark'],
          dtype='object')
    >>> num_features, cat_features = bi_selector (data)
    >>> num_features
    ...['gamma_gamma',
         'depth_top',
         'aquifer_thickness',
         'pumping_depth_at_the_end',
         'section_aperture',
         'remark',
         'depth_starting_pumping',
         'hole_depth_before_pumping',
         'rp',
         'hole_depth_after_pumping',
         'hole_depth_loss',
         'depth_bottom',
         'sp',
         'pumping_depth',
         'kp',
         'resistivity',
         'short_distance_gamma',
         'r',
         'natural_gamma',
         'layer_thickness',
         'k',
         'well_diameter']
    >>> cat_features 
    ... ['hole_id', 'strata_name', 'rock_name', 'aquifer_group', 
         'pumping_level']
    """
    _assert_all_types( d, pd.DataFrame, objname=" unfunc'bi-selector'")
    if features is None: 
        d, diff_features, features = to_numeric_dtypes(
            d,  return_feature_types= True ) 
    if features is not None: 
        diff_features = is_in_if( d.columns, items =features, return_diff= True )
        if diff_features is None: diff_features =[]
    return  ( diff_features, features ) if not return_frames else  (
        d [diff_features] , d [features ] ) 

def make_naive_pipe(
    X, 
    y =None, *,   
    num_features = None, 
    cat_features=None, 
    label_encoding='LabelEncoder', 
    scaler = 'StandardScaler' , 
    missing_values =np.nan, 
    impute_strategy = 'median', 
    sparse_output=True, 
    for_pca =False, 
    transform =False, 
    ): 
    """ make a pipeline to transform data at once. 
    
    make a naive pipeline is usefull to fast preprocess the data at once 
    for quick prediction. 
    
    Work with a pandas dataframe. If `None` features is set, the numerical 
    and categorial features are automatically retrieved. 
    
    Parameters
    ---------
    X : pandas dataframe of shape (n_samples, n_features)
        The input samples. Use ``dtype=np.float32`` for maximum
        efficiency. Sparse matrices are also supported, use sparse
        ``csc_matrix`` for maximum efficiency.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target relative to X for classification or regression;
        None for unsupervised learning.
    num_features: list or str, optional 
        Numerical features put on the list. If `num_features` are given  
        whereas `cat_features` are ``None``, `cat_features` are figured out 
        automatically.
    cat_features: list of str, optional 
        Categorial features put on the list. If `num_features` are given 
        whereas `num_features` are ``None``, `num_features` are figured out 
        automatically.
    label_encoding: callable or str, default='sklearn.preprocessing.LabelEncoder'
        kind of encoding used to encode label. This assumes 'y' is supplied. 
    scaler: callable or str , default='sklearn.preprocessing.StandardScaler'
        kind of scaling used to scaled the numerical data. Note that for 
        the categorical data encoding, 'sklearn.preprocessing.OneHotEncoder' 
        is implemented  under the hood instead. 
    missing_values : int, float, str, np.nan, None or pandas.NA, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        can be set to either `np.nan` or `pd.NA`.
    
    impute_strategy : str, default='mean'
        The imputation strategy.
    
        - If "mean", then replace missing values using the mean along
          each column. Can only be used with numeric data.
        - If "median", then replace missing values using the median along
          each column. Can only be used with numeric data.
        - If "most_frequent", then replace missing using the most frequent
          value along each column. Can be used with strings or numeric data.
          If there is more than one such value, only the smallest is returned.
        - If "constant", then replace missing values with fill_value. Can be
          used with strings or numeric data.
    
           strategy="constant" for fixed value imputation.
           
    sparse_output : bool, default=False
        Is used when label `y` is given. Binarize labels in a one-vs-all 
        fashion. If ``True``, returns array from transform is desired to 
        be in sparse CSR format.
        
    for_pca:bool, default=False, 
        Transform data for principal component ( PCA) analysis. If set to 
        ``True``, :class:`watex.exlib.sklearn.OrdinalEncoder`` is used insted 
        of :class:`watex.exlib.sklearn.OneHotEncoder``. 
        
    transform: bool, default=False, 
        Tranform data inplace rather than returning the naive pipeline. 
        
    Returns
    ---------
    full_pipeline: :class:`watex.exlib.sklearn.FeatureUnion`
        - Full pipeline composed of numerical and categorical pipes 
    (X_transformed &| y_transformed):  {array-like, sparse matrix} of \
        shape (n_samples, n_features)
        - Transformed data. 
        
        
    Examples 
    ---------
    >>> from watex.utils.mlutils import make_naive_pipe 
    >>> from watex.datasets import load_hlogs 
    
    (1) Make a naive simple pipeline  with RobustScaler, StandardScaler 
    >>> from watex.exlib.sklearn import RobustScaler 
    >>> X_, y_ = load_hlogs (as_frame=True )# get all the data  
    >>> pipe = make_naive_pipe(X_, scaler =RobustScaler ) 
    
    (2) Transform X in place with numerical and categorical features with 
    StandardScaler (default). Returned CSR matrix 
    
    >>> make_naive_pipe(X_, transform =True )
    ... <181x40 sparse matrix of type '<class 'numpy.float64'>'
    	with 2172 stored elements in Compressed Sparse Row format>

    """
    
    from ..transformers import DataFrameSelector
    
    sc= {"StandardScaler": StandardScaler ,"MinMaxScaler": MinMaxScaler , 
         "Normalizer":Normalizer , "RobustScaler":RobustScaler}

    if not hasattr (X, '__array__'):
        raise TypeError(f"'make_naive_pipe' not supported {type(X).__name__!r}."
                        " Expects X as 'pandas.core.frame.DataFrame' object.")
    X = check_array (
        X, 
        dtype=object, 
        force_all_finite="allow-nan", 
        to_frame=True, 
        input_name="Array for transforming X or making naive pipeline"
        )
    if not hasattr (X, "columns"):
        # create naive column for 
        # Dataframe selector 
        X = pd.DataFrame (
            X, columns = [f"naive_{i}" for i in range (X.shape[1])]
            )
    #-> Encode y if given
    if y is not None: 
        # if (label_encoding =='labelEncoder'  
        #     or get_estimator_name(label_encoding) =='LabelEncoder'
        #     ): 
        #     enc =LabelEncoder()
        if  ( label_encoding =='LabelBinarizer' 
                or get_estimator_name(label_encoding)=='LabelBinarizer'
               ): 
            enc =LabelBinarizer(sparse_output=sparse_output)
        else: 
            label_encoding =='labelEncoder'
            enc =LabelEncoder()
            
        y= enc.fit_transform(y)
    #set features
    if num_features is not None: 
        cat_features, num_features  = bi_selector(
            X, features= num_features 
            ) 
    elif cat_features is not None: 
        num_features, cat_features  = bi_selector(
            X, features= cat_features 
            )  
    if ( cat_features is None 
        and num_features is None 
        ): 
        num_features , cat_features = bi_selector(X ) 
    # assert scaler value 
    if get_estimator_name (scaler)  in sc.keys(): 
        scaler = sc.get (get_estimator_name(scaler )) 
    elif ( any ( [v.lower().find (str(scaler).lower()) >=0
                  for v in sc.keys()])
          ):  
        for k, v in sc.items () :
            if k.lower().find ( str(scaler).lower() ) >=0: 
                scaler = v ; break 
    else : 
        msg = ( f"Supports {smart_format( sc.keys(), 'or')} or "
                "other scikit-learn scaling objects, got {!r}" 
                )
        if hasattr (scaler, '__module__'): 
            name = getattr (scaler, '__module__')
            if getattr (scaler, '__module__') !='sklearn.preprocessing._data':
                raise ValueError (msg.format(name ))
        else: 
            name = scaler.__name__ if callable (scaler) else (
                scaler.__class__.__name__ ) 
            raise ValueError (msg.format(name ))
    # make pipe 
    npipe = [
            ('imputerObj',SimpleImputer(missing_values=missing_values , 
                                    strategy=impute_strategy)),                
            ('scalerObj', scaler() if callable (scaler) else scaler ), 
            ]
    
    if len(num_features)!=0 : 
       npipe.insert (
            0,  ('selectorObj', DataFrameSelector(attribute_names= num_features))
            )

    num_pipe=Pipeline(npipe)
    
    if for_pca : encoding=  ('OrdinalEncoder',OrdinalEncoder())
    else:  encoding =  (
        'OneHotEncoder', OneHotEncoder())
        
    cpipe = [
        encoding
        ]
    if len(cat_features)!=0: 
        cpipe.insert (
            0, ('selectorObj', DataFrameSelector(attribute_names= cat_features))
            )

    cat_pipe = Pipeline(cpipe)
    # make transformer_list 
    transformer_list = [
        ('num_pipeline', num_pipe),
        ('cat_pipeline', cat_pipe), 
        ]

    #remove num of cat pipe if one of them is 
    # missing in the data 
    if len(cat_features)==0: 
        transformer_list.pop(1) 
    if len(num_features )==0: 
        transformer_list.pop(0)
        
    full_pipeline =FeatureUnion(transformer_list=transformer_list) 
    
    return  ( full_pipeline.fit_transform (X) if y is None else (
        full_pipeline.fit_transform (X), y ) 
             ) if transform else full_pipeline
       
#XXX TODO: terminate func move to the metric module
def _stats (
    X_, 
    y_true,*, 
    y_pred, # noqa
    from_c ='geol', 
    drop_columns =None, 
    columns=None 
    )  : 
    """ Present a short static"""

    if from_c not in X_.columns: 
        raise TypeError(f"{from_c!r} not found in columns "
                        "name ={list(X_.columns)}")
        
    if columns is not None:
        if not isinstance(columns, (tuple, list, np.ndarray)): 
            raise TypeError(f'Columns should be a list not {type(columns)}')
        
    is_dataframe = isinstance(X_, pd.DataFrame)
    if is_dataframe: 
        if drop_columns is not None: 
            X_.drop(drop_columns, axis =1)
            
    if not is_dataframe : 
        len_X = X_.shape[1]
        if columns is not None: 
            if len_X != len(columns):
                raise TypeError(
                    "Columns and test set must have the same length"
                    f" But `{len(columns)}` and `{len_X}` were given "
                    "respectively.")
                
            X_= pd.DataFrame (data = X_, columns =columns)
            
    # get the values counts on the array and convert into a columns 
    if isinstance(y_pred, pd.Series): 
        y_pred = y_pred.values 
        # initialize array with full of zeros
    # get the values counts of the columns to analyse 'geol' for instance
    s=  X_[from_c].value_counts() # getarray of values 
    #s_values = s.values 
    # create a pseudo serie and get the values counts of each elements
    # and get the values counts

    y_actual=pd.Series(y_true, index = X_.index, name ='y_true')
    y_predicted =pd.Series(y_pred, index =X_.index, name ='y_pred')
    pdf = pd.concat([X_[from_c],y_actual,y_predicted ], axis=1)
 
    analysis_array = np.zeros((len(s.index), len(np.unique(y_true))))
    for ii, index in enumerate(s.index): 
        for kk, val in enumerate( np.unique(y_true)): 
            geol = pdf.loc[(pdf[from_c]==index)]
            geols=geol.loc[(geol['y_true']==geol['y_pred'])]
            geolss=geols.loc[(geols['y_pred']==val)]             
            analysis_array [ii, kk]=len(geolss)/s.loc[index]

    return analysis_array     
        

def select_feature_importances (
    clf, 
    X, 
    y=None, *,  
    threshold = .1 , 
    prefit = True , 
    verbose = 0 ,
    return_selector =False, 
    **kws
    ): 
    """
    Select feature importance  based on a user-specified threshold 
    after model fitting, which is useful if one want to use 
    `RandomForestClassifier` as a feature selector and intermediate step in 
    scikit-learn ``Pipeline`` object, which allows us to connect different 
    processing steps  with an estimator. 
  
    Parameters 
    ----------
    clf : estimator object
        The base estimator from which the transformer is built.
        This can be both a fitted (if ``prefit`` is set to True)
        or a non-fitted estimator. The estimator should have a
        ``feature_importances_`` or ``coef_`` attribute after fitting.
        Otherwise, the ``importance_getter`` parameter should be used.
        
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.
        
    y: array-like of shape (n_samples, ) 
        Target vector where `n_samples` is the number of samples. If given, 
        set `prefit=False` for estimator to fit and transform the data for 
        feature importance selecting. If estimator is already fitted  i.e.
        `prefit=True`, 'y' is not needed.

    threshold : str or float, default=None
        The threshold value to use for feature selection. Features whose
        absolute importance value is greater or equal are kept while the others
        are discarded. If "median" (resp. "mean"), then the ``threshold`` value
        is the median (resp. the mean) of the feature importances. A scaling
        factor (e.g., "1.25*mean") may also be used. If None and if the
        estimator has a parameter penalty set to l1, either explicitly
        or implicitly (e.g, Lasso), the threshold used is 1e-5.
        Otherwise, "mean" is used by default.

    prefit : bool, default=False
        Whether a prefit model is expected to be passed into the constructor
        directly or not.
        If `True`, `estimator` must be a fitted estimator.
        If `False`, `estimator` is fitted and updated by calling
        `fit` and `partial_fit`, respectively.

    importance_getter : str or callable, default='auto'
        If 'auto', uses the feature importance either through a ``coef_``
        attribute or ``feature_importances_`` attribute of estimator.

        Also accepts a string that specifies an attribute name/path
        for extracting feature importance (implemented with `attrgetter`).
        For example, give `regressor_.coef_` in case of
        :class:`~sklearn.compose.TransformedTargetRegressor`  or
        `named_steps.clf.feature_importances_` in case of
        :class:`~sklearn.pipeline.Pipeline` with its last step named `clf`.

        If `callable`, overrides the default feature importance getter.
        The callable is passed with the fitted estimator and it should
        return importance for each feature.
    
    norm_order : non-zero int, inf, -inf, default=1
        Order of the norm used to filter the vectors of coefficients below
        ``threshold`` in the case where the ``coef_`` attribute of the
        estimator is of dimension 2.

    max_features : int, callable, default=None
        The maximum number of features to select.

        - If an integer, then it specifies the maximum number of features to
          allow.
        - If a callable, then it specifies how to calculate the maximum number of
          features allowed by using the output of `max_feaures(X)`.
        - If `None`, then all features are kept.

        To only select based on ``max_features``, set ``threshold=-np.inf``.
        
    return_selector: bool, default=False, 
        Returns selector object if ``True``., otherwise returns the transformed
        `X`. 
        
    verbose: int, default=0 
        display the number of features that meet the criterion according to 
        their importance range. 
    
    Returns 
    --------
    Xs or selector : ndarray (n_samples, n_criterion_features), or \
        :class:`sklearn.feature_selection.SelectFromModel`
        Ndarray of number of samples and features that meet the criterion
        according to the importance range or selector object 
        
        
    Examples
    --------
    >>> from watex.utils.mlutils import select_feature_importances
    >>> from watex.exlib.sklearn import LogisticRegression
    >>> X0 = [[ 0.87, -1.34,  0.31 ],
    ...      [-2.79, -0.02, -0.85 ],
    ...      [-1.34, -0.48, -2.55 ],
    ...      [ 1.92,  1.48,  0.65 ]]
    >>> y0 = [0, 1, 0, 1]
    
    (1) use prefit =True and get the Xs importance features 
    >>> Xs = select_feature_importances (
        LogisticRegression().fit(X0, y0), 
        X0 , prefit =True )
    >>> Xs 
    array([[ 0.87, -1.34,  0.31],
           [-2.79, -0.02, -0.85],
           [-1.34, -0.48, -2.55],
           [ 1.92,  1.48,  0.65]])
    
    (2) Set off prefix  and return selector obj 
    
    >>> selector= select_feature_importances (
        LogisticRegression(), X= X0 , 
        y =y0  ,
        prefit =False , return_selector= True 
        )
    >>> selector.estimator_.coef_
    array([[-0.3252302 ,  0.83462377,  0.49750423]])
    >>> selector.threshold_
    0.1
    >>> selector.get_support()
    array([ True,  True,  True])
    
    >>> selector = SelectFromModel(estimator=LogisticRegression()).fit(X, y)
    >>> selector.estimator_.coef_
    array([[-0.3252302 ,  0.83462377,  0.49750423]])
    >>> selector.threshold_
    0.55245...
    >>> selector.get_support()
    array([False,  True, False])
    >>> selector.transform (X0) 
    array([[ 0.87, -1.34,  0.31],
           [-2.79, -0.02, -0.85],
           [-1.34, -0.48, -2.55],
           [ 1.92,  1.48,  0.65]])
    
    """
    if ( hasattr (clf, 'feature_names_in_') 
        or hasattr(clf, "feature_importances_")
        or hasattr (clf, 'coef_')
        ): 
        if not prefit: 
            warnings.warn(f"It seems the estimator {get_estimator_name (clf)!r}"
                          "is fitted. 'prefit' is set to 'True' to call "
                          "transform directly.")
            prefit =True 
            
    selector = SelectFromModel(
        clf, 
        threshold= threshold , 
        prefit= prefit, 
        **kws
        )
    
    if prefit:
        Xs = selector.transform(X) 
    else:
        Xs = selector.fit_transform(X, y =y)
        
    if verbose: 
        print(f"Number of features that meet the 'threshold={threshold}'" 
              " criterion: ", Xs.shape[1]
              ) 
        
    return selector if return_selector else Xs 

 
def naive_imputer (
    X, 
    y=None, 
    strategy = 'mean', 
    mode=None,  
    drop_features =False,  
    missing_values= np.nan ,
    fill_value = None , 
    verbose = "deprecated",
    add_indicator = False,  
    copy = True, 
    keep_empty_features=False, 
    **fit_params 
 ): 
    """ Imput missing values in the data. 
    
    Whatever data contains categorial features, 'bi-impute' argument passed to 
    'kind' parameters has a strategy to both impute the numerical and 
    categorical features rather than raising an error when the 'strategy' is 
    not set to 'most_frequent'.
    
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data used to compute the mean and standard deviation
        used for later scaling along the features axis.
        
    y : None
        Not used, present here for API consistency by convention.
        
    strategy : str, default='mean'
       The imputation strategy.

       - If "mean", then replace missing values using the mean along
         each column. Can only be used with numeric data.
       - If "median", then replace missing values using the median along
         each column. Can only be used with numeric data.
       - If "most_frequent", then replace missing using the most frequent
         value along each column. Can be used with strings or numeric data.
         If there is more than one such value, only the smallest is returned.
       - If "constant", then replace missing values with fill_value. Can be
         used with strings or numeric data.

          strategy="constant" for fixed value imputation.
        
    mode: str, [bi-impute'], default= None
        If mode is set to 'bi-impute', it imputes the both numerical and 
        categorical features and returns a single imputed 
        dataframe.
        
    drop_features: bool or list, default =False, 
        drop a list of features in the dataframe before imputation. 
        If ``True`` and no list of features is supplied, the categorial 
        features are dropped. 
        
    missing_values : int, float, str, np.nan, None or pandas.NA, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        can be set to either `np.nan` or `pd.NA`.

    fill_value : str or numerical value, default=None
        When strategy == "constant", fill_value is used to replace all
        occurrences of missing_values.
        If left to the default, fill_value will be 0 when imputing numerical
        data and "missing_value" for strings or object data types.
        
    keep_empty_features : bool, default=False
        If True, features that consist exclusively of missing values when
        `fit` is called are returned in results when `transform` is called.
        The imputed value is always `0` except when `strategy="constant"`
        in which case `fill_value` will be used instead.

        .. versionadded:: 0.2.0
         
    verbose : int, default=0
        Controls the verbosity of the imputer.

    copy : bool, default=True
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible. Note that, in the following cases,
        a new copy will always be made, even if `copy=False`:

        - If `X` is not an array of floating values;
        - If `X` is encoded as a CSR matrix;
        - If `add_indicator=True`.

    add_indicator : bool, default=False
        If True, a :class:`MissingIndicator` transform will stack onto output
        of the imputer's transform. This allows a predictive estimator
        to account for missingness despite imputation. If a feature has no
        missing values at fit/train time, the feature won't appear on
        the missing indicator even if there are missing values at
        transform/test time.
        
    fit_params: dict, 
        keywords arguments passed to the scikit-learn fitting parameters 
        More details on https://scikit-learn.org/stable/ 
    Returns 
    --------
    Xi: Dataframe, array-like, sparse matrix of shape (n_samples, n_features)
        Data imputed 
        
    Examples 
    --------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from watex.utils.mlutils import naive_imputer 
    >>> X= np.random.randn ( 7, 4 ) 
    >>> X[3, :] =np.nan  ; X[:, 3][-4:]=np.nan 
    >>> naive_imputer  (X)
    ... array([[ 1.34783528,  0.53276798, -1.57704281,  0.43455785],
               [ 0.36843174, -0.27132106, -0.38509441, -0.29371997],
               [-1.68974996,  0.15268509, -2.54446498,  0.18939122],
               [ 0.06013775,  0.36687602, -0.21973368,  0.11007637],
               [-0.27129147,  1.18103398,  1.78985393,  0.11007637],
               [ 1.09223954,  0.12924661,  0.52473794,  0.11007637],
               [-0.48663864,  0.47684353,  0.87360825,  0.11007637]])
    >>> frame = pd.DataFrame (X, columns =['a', 'b', 'c', 'd']  ) 
    >>> # change [bc] types to categorical values.
    >>> frame['b']=['pineaple', '', 'cabbage', 'watermelon', 'onion', 
                    'cabbage', 'onion']
    >>> frame['c']=['lion', '', 'cat', 'cat', 'dog', '', 'mouse']
    >>> naive_imputer(frame, kind ='bi-impute')
    ...             b      c         a         d
        0    pineaple   lion  1.347835  0.434558
        1     cabbage    cat  0.368432 -0.293720
        2     cabbage    cat -1.689750  0.189391
        3  watermelon    cat  0.060138  0.110076
        4       onion    dog -0.271291  0.110076
        5     cabbage    cat  1.092240  0.110076
        6       onion  mouse -0.486639  0.110076
        
    """
    X_cat, _isframe =None , True  
    
    X = check_array (
        X, 
        dtype=object, 
        force_all_finite="allow-nan", 
        to_frame=True, 
        input_name="X"
        )
 
    if drop_features :
        if not hasattr(X, 'columns'): 
            raise ValueError ("Drop feature is possible only if  X is a"
                              f" dataframe. Got {type(X).__name__!r}") 
        
        if ( str(drop_features).lower().find ('cat') >=0 
                or  str(drop_features).lower()=='true' 
                    ) :
            # drop cat features
            X= to_numeric_dtypes(X, pop_cat_features=True, verbose =True )

        else : 
            if not is_iterable(drop_features): 
                raise TypeError ("Expects a list of features to drop;"
                                 " not {type(drop_features).__name__!r}")
        # drop_feature is a list assert whether features exist in X
            existfeatures(X, features = drop_features ) 
            diff_features = is_in_if(X.columns, drop_features, return_diff= True
                                     )
            if diff_features is None:
                raise DatasetError(
                    "It seems all features in X have been dropped. "
                    "Cannot impute a dataset with no features."
                    f" Drop features: '{drop_features}'")
                
            X= X[diff_features ]
            
    # ====> implement bi-impute strategy.  
    # strategy expects at the same time 
    # categorical  and num features 
    err_msg =(". Use 'bi-impute' strategy passed to"
              " the parameter 'mode' to coerce the categorical"
              " besides the numerical features."
    )
    if strategy =="most_frequent": 
       # altered the bi-impute strategy 
       # since most_frequent imputes at 
       # the same time num and cat features 
       
       mode =None 
    if mode is not None: 
        mode = str(mode).lower().strip () 
        if mode.find ('bi')>=0: 
            mode='bi-impute'
            
        assert mode in {'bi-impute'} , (
            f"Strategy passed to 'mode' supports only 'bi-impute', not {mode!r}")

    if mode=='bi-impute':
        if not hasattr (X, 'columns'): 
            # "In pratice, the bi-Imputation is only allowed"
            # " with adataframe so create naive columns rather"
            # than raise error
            X= pd.DataFrame(X, columns =[f"bi_{i}" for i in range(X.shape[1])]
                            )
            _isframe =False 
            
        # recompute the num and cat features
        # since drop features can remove the
        # the cat features 
        X , nf, cf = to_numeric_dtypes(X, return_feature_types= True ) 
        if (len(nf) and len(cf) ) !=0 :
            # keep strategy to bi-impute 
            mode='bi-impute'
            X_cat , X = X [cf] ,  X[nf] 
            
        elif len(nf) ==0 and len(cf)!=0: 
            strategy ='most_frequent'
            mode =None # reset the kind method 
            X = X [cf]
        else: # if numeric 
            mode =None 
            
    # <==== end bi-impute strategy
    imp = SimpleImputer(strategy= strategy , 
                        missing_values= missing_values , 
                        fill_value = fill_value , 
                        verbose = verbose, 
                        add_indicator=False, 
                        copy = copy, 
                        keep_empty_features=keep_empty_features, 
                        )
    try : 
        Xi = imp.fit_transform (X, y =y, **fit_params )
    except Exception as err :
        #improve error msg 
        raise ValueError (str(err) + err_msg)

    if hasattr (imp , 'feature_names_in_'): 
        Xi = pd.DataFrame( Xi , columns = imp.feature_names_in_)  
    # commonly when strategy is most frequent
    # categorical features are also imputed.
    # so dont need to use bi-impute strategy
    if  mode=='bi-impute':
        imp.strategy ='most_frequent'
        Xi_cat  = imp.fit_transform (X_cat, y =y, **fit_params ) 
        Xi_cat = pd.DataFrame( Xi_cat , columns = imp.feature_names_in_)
        Xi = pd.concat ([Xi_cat, Xi], axis =1 )
        
        if not _isframe : 
            Xi = Xi.values 
            
    return Xi

    
def naive_scaler(
    X,
    y =None, *, 
    kind= StandardScaler, 
    copy =True, 
    with_mean = True, 
    with_std= True , 
    feature_range =(0 , 1), 
    clip = False,
    norm ='l2',  
    **fit_params  
    ): 
    """ Quick data scaling using both strategies implemented in scikit-learn 
    with StandardScaler and MinMaxScaler. 
    
    Function returns scaled frame if dataframe is passed or ndarray. For other 
    scaling, call scikit-learn instead. 
    
    Parameters 
    ------------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data used to compute the mean and standard deviation
        used for later scaling along the features axis.

    y : None
        Ignored.
        
    kind: str, default='StandardScaler' 
        Kind of data scaling. Can also be ['MinMaxScaler', 'Normalizer']. The 
        default is 'StandardScaler'
    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

    with_mean : bool, default=True
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.

    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).
        
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    norm : {'l1', 'l2', 'max'}, default='l2'
        The norm to use to normalize each non zero sample. If norm='max'
        is used, values will be rescaled by the maximum of the absolute
        values.

    clip : bool, default=False
        Set to True to clip transformed values of held-out data to
        provided `feature range`.
        
    fit_params: dict, 
        keywords arguments passed to the scikit-learn fitting parameters 
        More details on https://scikit-learn.org/stable/ 
            
    Returns
    -------
    X_sc : {ndarray, sparse matrix} or dataframe of  shape \
        (n_samples, n_features)
        Transformed array.
        
    Examples 
    ----------
    >>> import numpy as np  
    >>> import pandas as pd 
    >>> from watex.utils.mlutils import naive_scaler 
    >>> X= np.random.randn (7 , 3 ) 
    >>> X_std = naive_scaler (X ) 
    ... array([[ 0.17439644,  1.55683005,  0.24115109],
           [-0.59738672,  1.3166854 ,  1.23748004],
           [-1.6815365 , -1.19775838,  0.71381357],
           [-0.1518278 , -0.32063059, -0.47483155],
           [-0.41335886,  0.13880519,  0.69258621],
           [ 1.45221902, -1.03852015, -0.40157981],
           [ 1.21749443, -0.45541153, -2.00861955]])
    >>> # use dataframe 
    >>> Xdf = pd.DataFrame (X, columns =['a', 'c', 'c'])
    >>> naive_scaler (Xdf , kind='Normalizer') # return data frame 
    ...           a         c         c
        0  0.252789  0.967481 -0.008858
        1 -0.265161  0.908862  0.321961
        2 -0.899863 -0.416231  0.130380
        3  0.178203  0.039443 -0.983203
        4 -0.418487  0.800306  0.429394
        5  0.933933 -0.309016 -0.179661
        6  0.795234 -0.051054 -0.604150
    """
    msg =("Supports only the 'standardization','normalization' and  'minmax'"
          " scaling types, not {!r}")
    
    kind = kind or 'standard'
    
    if   ( 
            str(kind).lower().strip().find ('standard')>=0 
            or get_estimator_name(kind) =='StandardScaler'
            ): 
        kind = 'standard'
    elif ( 
            str(kind).lower().strip().find ('minmax')>=0 
            or get_estimator_name (kind) =='MinMaxScaler'
            ): 
        kind = 'minmax'
    elif  ( 
            str(kind).lower().strip().find ('norm')>=0  
            or get_estimator_name(kind)=='Normalizer'
            ):
        kind ='norm'
        
    assert kind in {"standard", 'minmax', 'norm'} , msg.format(kind)
    
    if kind =='standard': 
        sc = StandardScaler(
            copy=copy, with_mean= with_mean , with_std= with_std ) 
    elif kind == 'minmax': 
        sc = MinMaxScaler(feature_range= feature_range, 
                          clip = clip, copy =copy  ) 
    elif kind=='norm': 
        
        sc = Normalizer(copy= copy , norm = norm ) 
        
    X_sc = sc.fit_transform (X, y=y, **fit_params)
    
    if hasattr (sc , 'feature_names_in_'): 
        X_sc = pd.DataFrame( X_sc , columns = sc.feature_names_in_)  
    return X_sc 


    











        
        
        
        
        
        
        

