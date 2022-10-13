# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause, Author: LKouadio 

"""
load different data as a compliant functions 
============================================= 

Inspired with the machine learning popular dataset loading 

Created on Thu Oct 13 16:26:47 2022
@author: Daniel
"""
from warning import warn 
from ._io import csv_data_loader, _to_dataframe , DATA 
from ..tools.coreutils import vesSelector, erpSelector 
from ..tools.mlutils import split_train_test_by_id #(split_train_test , 
from ..tools.box import Boxspace

def load_tankesse (*, as_frame =True, **kws ):
    data_file ="tankesse.csv"
    df =  erpSelector(f = data_file  , **kws)
    return df if as_frame else df.values 

def load_semien (*, as_frame =True , index_rhoa =0, **kws ):
    data_file ="semien_ves.csv"
    df = vesSelector(data= data_file, index_rhoa = index_rhoa, **kws) 
    return df if as_frame else df.values 

def load_gbalo (*, kind ='ves', as_frame=True, index_rhoa = 0 , **kws ): 
    kind =str(kind).lower().strip() 
    
    if kind not in ("erp", "ves"): 
        warn ("{kind!r} is unknow! By default DC-Resistivity profiling is returned.")
        kind="erp"
    data_file = f"dc{kind}_gbalo.csv" 
    if "ves" in data_file: 
        return vesSelector(data =data_file , index_rhoa= index_rhoa , **kws) 
    if "erp" in data_file : 
        return erpSelector(data_file )
    
def load_bagoue(
        *, return_X_y=False, as_frame=True, split_X_y=False, test_size =.3 ,  
 ):
    data_file = "bagoue.csv"
    data, target, target_names, fdescr = csv_data_loader(
        data_file=data_file, descr_file="bagoue.rst"
    )
    data.columns = data.columns.str.lower()
    feature_names = list(data.columns)
    frame = None
    target_columns = [
        "flow",
    ]
    if as_frame:
        frame, data, target = _to_dataframe(
            data, feature_names = feature_names, tnames = target_columns
                                            )
    if split_X_y: 
        X, Xt = split_train_test_by_id (data = frame , test_ratio= test_size )
        y = X.flow ;  X.drop(columns =target_columns, inplace =True)
        yt = Xt.flow , Xt.drop(columns =target_columns, inplace =True)
        
        return  (X, Xt, y, yt ) if as_frame else (
            X.values, Xt.values, y.values , yt.values )
    
    if return_X_y:
        return data, target

    return Boxspace(
        data=data,
        target=target,
        frame=frame,
        target_names=target_names,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file,
        data_module=DATA,
    )

def load_iris(*, return_X_y=False, as_frame=False):
    data_file = "iris.csv"
    data, target, target_names, fdescr = csv_data_loader(
        data_file=data_file, descr_file="iris.rst"
    )
    feature_names = ["sepal length (cm)","sepal width (cm)",
        "petal length (cm)","petal width (cm)",
    ]
    frame = None
    target_columns = [
        "target",
    ]
    if as_frame:
        frame, data, target = _to_dataframe(
            data, feature_names = feature_names, tnames = target_columns)
        # _to(
        #     "load_iris", data, target, feature_names, target_columns
        # )
    if return_X_y:
        return data, target

    return Boxspace(
        data=data,target=target,frame=frame,target_names=target_names,
        DESCR=fdescr,feature_names=feature_names,filename=data_file,
        data_module=DATA,
        )

load_iris.__doc__="""\
Load and return the iris dataset (classification).
The iris dataset is a classic and very easy multi-class classification
dataset.

Parameters
----------
return_X_y : bool, default=False
    If True, returns ``(data, target)`` instead of a BowlSpace object. See
    below for more information about the `data` and `target` object.
    .. versionadded:: 0.18
as_frame : bool, default=False
    If True, the data is a pandas DataFrame including columns with
    appropriate dtypes (numeric). The target is
    a pandas DataFrame or Series depending on the number of target columns.
    If `return_X_y` is True, then (`data`, `target`) will be pandas
    DataFrames or Series as described below.
    .. versionadded:: 0.23
Returns
-------
data : :class:`~sklearn.utils.Bunch`
    Dictionary-like object, with the following attributes.
    data : {ndarray, dataframe} of shape (150, 4)
        The data matrix. If `as_frame=True`, `data` will be a pandas
        DataFrame.
    target: {ndarray, Series} of shape (150,)
        The classification target. If `as_frame=True`, `target` will be
        a pandas Series.
    feature_names: list
        The names of the dataset columns.
    target_names: list
        The names of target classes.
    frame: DataFrame of shape (150, 5)
        Only present when `as_frame=True`. DataFrame with `data` and
        `target`.
        .. versionadded:: 0.23
    DESCR: str
        The full description of the dataset.
    filename: str
        The path to the location of the data.
        .. versionadded:: 0.20
(data, target) : tuple if ``return_X_y`` is True
    A tuple of two ndarray. The first containing a 2D array of shape
    (n_samples, n_features) with each row representing one sample and
    each column representing the features. The second ndarray of shape
    (n_samples,) containing the target samples.
    .. versionadded:: 0.18
Notes
-----
    .. versionchanged:: 0.20
        Fixed two wrong data points according to Fisher's paper.
        The new version is the same as in R, but not as in the UCI
        Machine Learning Repository.
Examples
--------
Let's say you are interested in the samples 10, 25, and 50, and want to
know their class name.
>>> from sklearn.datasets import load_iris
>>> data = load_iris()
>>> data.target[[10, 25, 50]]
array([0, 0, 1])
>>> list(data.target_names)
['setosa', 'versicolor', 'virginica']
"""    
    
    
    
    
    
    
    
    
    
    
    
    
