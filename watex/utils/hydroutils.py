# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Hydrogeological parameters of aquifer are the essential and crucial basic data 
in the designing and construction progress of geotechnical engineering and 
groundwater dewatering, which are directly related to the reliability of these 
parameters.

.. note::
    For strong and clear demonstration as examples in many scripts, we use 
    the data 'hf.csv'. This data is a confident data so it is not available 
    in the package. The idea consists to show how scripts will works if 
    many boreholes data are available. 

"""
from __future__ import annotations 
import random
import copy 
import math
import itertools
from collections import  ( 
    Counter , 
    defaultdict
    )
import inspect
import warnings 
import numpy as np
import pandas as pd 
from .._docstring import ( 
    _core_docs, 
    DocstringComponents 
    )
from .._typing import (
    List, 
    Tuple, 
    Optional, 
    Union, T,
    Series, 
    DataFrame, 
    ArrayLike, 
    F
    ) 
from ..decorators import ( 
    catmapflow2, 
    writef, 
    deprecated
    )
from ..exceptions import ( 
    FileHandlingError, 
    DepthError, 
    DatasetError, 
    StrataError, 
    AquiferGroupError
    )
from .box import ( 
    _Group, 
    Boxspace
    )
from .funcutils import  (
    _assert_all_types, 
    _isin ,
    is_iterable,
    is_in_if , 
    smart_format, 
    savepath_ , 
    is_depth_in, 
    reshape , 
    listing_items_format, 
    to_numeric_dtypes, 
    )
from .validator import ( 
    _is_arraylike_1d,
    _is_numeric_dtype, 
    _check_consistency_size, 
    to_dtype_str,
    check_y, 
    check_array, 
    )

__all__=[
    "select_base_stratum" , 
    "get_aquifer_section" , 
    "get_aquifer_sections", 
    "get_unique_section", 
    "get_compressed_vector", 
    "get_xs_xr_splits", 
    "reduce_samples" , 
    "get_sections_from_depth", 
    "check_flow_objectivity", 
    "make_MXS_labels", 
    "predict_NGA_labels", 
    "find_aquifer_groups", 
    "find_similar_labels", 
    "classify_k", 
    "is_valid_depth", 
    "label_importance", 
    "validate_labels", 
    "rename_labels_in", 
    "transmissibility", 
    "categorize_target", 
    ]

#-----------------------
_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"], 
    )
#------------------------

def make_MXS_labels (
    y_true, 
    y_pred, 
    threshold= None, 
    similar_labels= None, 
    sep =None, 
    prefix =None, 
    method='naive', 
    trailer="*",
    return_obj=False,  
    **kws
   ): 
    """ Create a Mixture Learning Strategy (MXS) labels from true labels 
    'y_true' and the predicted Naive Group of Aquifer (NGA) labels 'y_pred'
    
    Parameters
    -----------
    y_true: array-like 1d, pandas.Series 
        Array composed of valid k-values and possible missing k-values. 
        
    y_pred: Array-like 1d, pandas.Series
        Array composing the valid NGA labels. Note that NGA labels is  a 
        predicted labels mostly using the unsupervising learning. 
         
    threshold: float, default=None 
        The threshold from which, label in 'y_true' can be considered  
        similar than the one in NGA labels 'y_pred'. The default is 'None' which 
        means none rule is considered and the high preponderence or occurence 
        in the data compared to other labels is considered as the most 
        representative  and similar. Setting the rule instead by fixing 
        the threshold is recommended especially in a huge dataset.
        
    similar_labels: list of tuple, optional   
        list of tuple in pair (label and similar group). If given, the similar 
        group must be the label existing in the predicted NGA. If ``None``, 
        the auto-similarity is triggered. 
        
    sep: str, default'' 
        Separator between the true labels 'y_true' and predicted NGA labels.
        Sep is used to rewrite the MXS labels. Mostly the MXS labels is a 
        combinaison with the true label of permeability coefficient 'k' and 
        the label of NGA to compose new similarity labels. For instance 
        
        >>> true_labels=['k1', 'k2', 'k3'] ; NGA_labels =['II', 'I', 'UV']
        >>> # gives 
        >>> MXS_labels= ['k1_II', 'k2_I', 'k3_UV']
    
        where the seperator `sep` is set to ``_``. This happens especially 
        when one of the label (NGA or true_labels) is not a numeric datatype 
        and a similariy is found between 'k1' and 'II', 'k2' and 'I' and so on.
        
    prefix: str, default=''
        prefix is used to rename the true_labels i.e the true valid-k. For
        instance::
            >>> k_valid =[1, 2, ..] -> k_new = [k1, k2, ...]
        where 'k' is the prefix. 
        
    method: str ['naive', 'strict'], default='naive'
        The kind of strategy to compute the representativity of a label 
        in the predicted array 'y_pred'. It can also be 'strict'. Indeed:
        
        - ``naive`` computes the importance of the label by the number of its
            occurence for this specific label in the array 'y_true'. It does not 
            take into account of the occurence of other existing labels. This 
            is usefull for unbalanced class labels in `y_true`.
        - ``strict`` computes the importance of the label by the number of 
            occurence in the whole valid `y_true` i.e. under the total of 
            occurence of all the labels that exist in the whole 'arra_aq'. 
            This can give a suitable anaylse results if the data is not 
            unbalanced for each labels in `y_pred`.
            
    trailer: str, default='*'
        The Mixture strategy marker to differentiate the existing class label  
        in 'y_true' with the predicted labels 'y_pred' especially when  
        the the same class labels are also present the true label with the 
        same label-identifier name. This usefull  to avoid any confusion  for
        both labels  in `y_true` and `y_pred` for better demarcation and 
        distinction. Note that if the `trailer`is set to ``None`` and both 
        `y_true` and `y_pred` are numeric data, the labels in `y_pred` are 
        systematically renamed to be distinct with the ones in the 'y_true'. 
        For instance :: 
            
            >>> true_labels=[1, 2, 3] ; NGA_labels =[0, 1, 2]
            >>> # with trailer , MXS labels should be 
            >>>  MXS_labels= ['0', '1*', '2*', '3'] # 1 and 2 are in true_labels 
            >>> # with no trailer 
            >>> MXS_labels= [0, 4, 5, 3] # 1 and 2 have been changed to [4, 5]
            
    return_obj: :class:`watex.utils.box.Boxspace`
        If ``True``, returns a MXS object with usefull attributes such as: 
            - mxs_classes_ = the MXS class labels 
            - mxs_labels_=  the array-like of MXS labels. It also includes some
                non similar labels from NGA
            mxs_map_classes_= a dict or original class labels of the array
                'k' <'y_true'> and their temporary integer class labels.
                Indeed, if 'y_true' class labels are not a numeric dtype, 
                New labels with integer dtype is created. The dict is used to 
                wrap the true labels (original ones) during the MXS creation. 
                Thus, the original labels are not altered and will be map in 
                turn  at the end to recover their positions as well in 
                new MXS array. It is set to 'None' if 'y_true' has a numeric 
                dtype. 
            mxs_group_classes_: dict of all the similar group labels  with the 
                MXS labels related from the modified existing groups of NGA.
                Note that the non-similar group are modified if their labels 
                are also found in the true_labels to avoid any confusion. Thus
                the dict wrap the non-similar label with their new temporay 
                labels. 
            mxs_similar_groups_= list of the similar labels found in 
                y_true that have a similarity in NGA.  
            mxs_similarity_= Tuple of similarity in pair (label, group) 
                existing between the label class in y_true and NGA. 
            mxs_group_labels_= list of the similar groups found in the 
                predicted NGA that have a similarity in true labels 'y_true'
        
    Returns 
    ---------
    MXS: array-like 1d or :class:`~watex.utils.box.Boxspace`
        array like of MXS labels or MXS object containing the 
        usefull attributes. 
    
    See Also
    ---------
    predict_NGA_labels: Predicts Naive group of Aquifers  labels. 
    
    
    Examples
    ---------
    >>> from watex.datasets import load_hlogs
    >>> from watex.utils import read_data 
    >>> from watex.utils.hydroutils import classify_k, make_MXS_labels
    >>> data = load_hlogs ().frame 
    >>> # map data.k to categorize k values 
    >>> ymap = classify_k(data.k , default_func =True) 
    >>> y_mxs = make_MXS_labels (ymap, data.aquifer_group)
    >>> y_mxs[14:24] 
    ...  array(['I', 'I', 2, 2, 2, 2, 2, 2, 2, 2], dtype=object)
    >>> mxs_obj = make_MXS_labels (ymap, data.aquifer_group, return_obj=True )
    >>> mxs_obj.mxs_labels_[14: 24]
    ... array(['I', 'I', 2, 2, 2, 2, 2, 2, 2, 2], dtype=object)
    >>> # now we did the same task using the private data 'hf.csv'
    >>> # composed of 11 boreholes. For default we alternatively uses 
    >>> # the aquifer groups like a fake NGA 
    >>> data = read_data ('data/boreholes/hf.csv') 
    >>> ymap =  classify_k(data.k , default_func =True)  
    >>> y_mxs= make_MXS_labels (ymap, data.aquifer_group)
    >>> np.unique (y_mxs)
    ... array(['1', '1V', '2', '2III', '3', 'I', 'II', 'III&IV', 'IV'],
          dtype='<U6')
    >>> # *comments: 
        # label '1V' means the group V (expected to be a cluster) 
        # and label 1 (true labels) have a similarity 
        # the same of label '2III' while the remain label 3 does not  
        #  any similarity in the other labels  in the 'y_pred' expected 
        # to be NGA labels. 
        
    """
    CONTEXT_MSG = (
        "Can only process unfunc {0!r} if and only if {1} similarity"
        " is found between true labels in 'y_true' and the predicted NGA"
        " labels in 'y_pred'."
        )
    
    sep = sep or '' 
    prefix = prefix or '' 
    # for consistency
    # check arrays 
    y_true = check_y (
        y_true, 
        allow_nan= True, 
        to_frame =True, 
        input_name="y_true",
        )  

    y_pred = check_y (
        y_pred, 
        to_frame = True, 
        allow_nan= False, 
        input_name ="NGA labels"
        )

    _check_consistency_size(y_true, y_pred ) 
    # check whether the y_true is numerical data 
    # if not rename y_true and keep the classes 
    # for mapping at the end of class transformation 
    #y_true_transf, mxs_map_classes_  = _kmapping( y_true )
    
    if similar_labels is None: 
        similar_labels = find_similar_labels (
            y_true, 
            y_pred, 
            threshold= threshold, 
            method=method, 
            **kws 
            ) 
        
    CONTEXT = 'no' if len(similar_labels)==0  else 'similarity is found' 

    if CONTEXT =='no' : 
        y_mxs, group_classes_, group_labels, sim_groups = _MXS_if_no(
            CONTEXT, 
            y_true, 
            y_pred, 
            cmsg=CONTEXT_MSG , 
            trailer=trailer 
            )
    else : 
        y_mxs, group_classes_, group_labels, sim_groups = _MXS_if_yes(
            CONTEXT , 
            similar_labels, 
            y_pred, 
            y_true, 
            sep =sep,
            prefix= prefix, 
            cmsg= CONTEXT_MSG, 
            trailer= trailer 
        )
    # # save the not_nan indices to not 
    # # altered the k-valid values 
    not_nan_indices,  = np.where ( ~np.isnan (y_true) )
    # # not altered the k-valid data
    try: 
        # try to reconvert class labels to integer
        # if class are numeric values, otherwise  
        # keep the values as they were.
        y_mxs [not_nan_indices] = y_true [not_nan_indices].astype(np.int32)
    except :  
        y_mxs [not_nan_indices] = y_true [not_nan_indices]
    
    #let pandas to find the best dtype since 
    # string value in y_mxs object remain a string 
    # object in data
    y_mxs = pd.Series (y_mxs, name ='mxs').values 

    try : 
        y_mxs = y_mxs .astype (int) 
    except : y_mxs= y_mxs.astype(str )
    
    MXS =y_mxs .copy() 
    
    if return_obj : 
        # create a metatype of mixture object class and 
        # wrapp the importance attributes 
        try : 
            mxs_classes_ = np.unique (y_mxs) 
        except:
            mxs_classes_ = np.unique (y_mxs.astype (str ) ) 
            
        MXS_attributes = dict (
            mxs_classes_ = mxs_classes_, 
            mxs_labels_= y_mxs ,  
            # mxs_map_classes_= mxs_map_classes_, 
            mxs_group_classes_=group_classes_ ,
            mxs_similar_labels_= similar_labels, 
            mxs_similarity_= sim_groups,  
            mxs_group_labels_= group_labels
            )  
        
        MXS = Boxspace(**MXS_attributes)
        
    return MXS 

def predict_NGA_labels( 
        X, / , n_clusters , random_state =0 , keep_label_0 = False, 
        n_init="auto",return_cluster_centers =False,  **kws 
        ): 
    """
    Predict the Naive Group of Aquifer (NGA) labels. 
    
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training instances to cluster. It must be noted that the data
        will be converted to C ordering, which will cause a memory
        copy if the given data is not C-contiguous.
        If a sparse matrix is passed, a copy will be made if it's not in
        CSR format.
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    random_state : int, RandomState instance or None, default=42
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
   
    keep_label_0: bool, default=False
        The prediction already include the label 0. However, including 0 in 
        the predicted label refers to 'k=0' i.e. no permeability coefficient 
        equals to 0, which is not True in principle, because all rocks  have 
        a permeability coefficient 'k'. Here we considered 'k=0' as an undefined 
        permeability coefficient. Therefore, '0' , can be exclude since, it can 
        also considered as a missing 'k'-value. If predicted '0' is in the target 
        it should mean a missing 'k'-value rather than being a concrete label.  
        Therefore, to avoid any confusion, '0' is altered to '1' so the value 
        `+1` is used to move forward all class labels thereby excluding 
        the '0' label. To force include 0 in the label, set `keep_label_0` 
        to ``True``. 
        
    n_init : 'auto' or int, default=10
        Number of times the k-means algorithm is run with different centroid
        seeds. The final results is the best output of `n_init` consecutive runs
        in terms of inertia. Several runs are recommended for sparse
        high-dimensional problems (see :ref:`kmeans_sparse_high_dim`).
    
        When `n_init='auto'`, the number of runs will be 10 if using
        `init='random'`, and 1 if using `init='kmeans++'`.
    
        .. versionadded:: 0.2.0 
           Added 'auto' option for `n_init`.
    
    return_cluster_centers: bool, default=False, 
        export the array of clusters centers if ``True``. 
    kws: dict, 
        Additional keyword arguments passed to :class:`sklearn.clusters.KMeans`.
         
    Returns 
    ---------
    NGA: array_like of  shape (n_samples, n_features)
        Predicted NGA labels. 
    ( NGA , cluster_centers) : Tuple of array-like, 
       MGA and clusters centers if ``return_cluster_centers` is 
       set to ``True``. 
    """
    from ..exlib.sklearn import KMeans 
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ko= KMeans(n_clusters= n_clusters, random_state = random_state , 
                  init="random", n_init=n_init , **kws
                  )
    NGA=ko.fit_predict(X)
    if not keep_label_0:
        if 0 in list(np.unique (NGA)):
            NGA +=1 
            
    return ( NGA , ko.cluster_centers_ ) if return_cluster_centers else NGA 


def find_aquifer_groups (
        arr_k, /, arr_aq=None, kname =None, aqname=None, subjectivity =False,  
         default_arr= None, keep_label_0 = False,  method ='naive', 
  )->'_Group': 
    msg = ("{} cannot be None when a dataframe is given.")
    d = copy.deepcopy(arr_k)
    if hasattr (d, '__array__') and hasattr (d, 'columns'): 
        if arr_aq is None  and aqname is None : 
            raise TypeError (msg.format("Group of aquifer column ('aqname')"))
        if kname is None: 
            raise TypeError (msg.format("Permeability coefficient column ('kname')"))
            
        arr_aq = d[aqname ] ; arr_k = d[kname]
        
    if arr_aq is None and not subjectivity: 
        msg =("In principle, missing aquifer array is not allowed. Turn on "
              "'subjectivity' instead. Make sure, you know what you intend to"
              " solve when turning 'subjectivity' to 'True'. This might lead"
              " to breaking code or unexpected results. Use at your own risk." )
        raise AquiferGroupError (msg)
    if subjectivity: 
        if arr_aq is not None: 
            warnings.warn ("No need to set subjectivity to 'True' while the "
                           "array of the group of aquifer 'arra_aq' is provided.")
        if default_arr is None: 
            raise TypeError ("Default array 'default_arr' must not be None"
                             "An alternatively array is used to use the"
                             "subjectivity case. The default array is used"
                             " to substitute the aquifer groups.")
        arr_aq = default_arr 
        
    # check consistency 
    _check_consistency_size(arr_aq, arr_k)
    if not all ([ _is_arraylike_1d(arr_aq), _is_arraylike_1d(arr_k)]):
        raise AquiferGroupError (
            "Expects one-dimensional arrays for 'k' and aquifer group.")
    
    # check arrays 
    arr_k = check_y (
        arr_k, 
        allow_nan= True, 
        to_frame =True, 
        input_name="Array of Permeability coefficient 'k'",
        )  

    if np.nan in list(arr_aq): 
        raise TypeError ("Missing value(s) is/are not allowed in group of "
                         " aquifer. Please impute the data first.")
    # for consistency recheck 
    arr_aq = check_y (
        arr_aq, 
        to_frame = True, 
        allow_nan= False, 
        input_name ="Array of aquifer group 'arr_aq'"
        )
    
    arr_k_valid , arr_aq_valid = _get_y_from_valid_indexes(
        arr_k, arr_aq, include_label_0= keep_label_0  )
    
    labels , counts = np.unique (arr_k_valid , return_counts= True) 
    labels_rate = counts / sum(counts )
    dict_labels_rate = { k: v for k , v in zip ( labels, labels_rate )} 
    
    groups = defaultdict(list)  
    for label in sorted (labels) : 
        g = label_importance(
            label, arr_k=arr_k_valid , arr_aq= arr_aq_valid, method =method )
        groups[label].append (dict_labels_rate.get(label))
        groups[label].append(g)
        
    return _Group(groups)

find_aquifer_groups.__doc__="""\
Fit the group of aquifer and find the representative of each true label in 
array 'k' in the aquifer group array. 

The idea consists to find the corresponding aquifer group which fits the most 
the true label 'X' in 'y_true'. 

'arr_k' and 'arr_aq' must contain a class label, not continue values. 

Parameters 
-----------
arr_k: array_like, pandas series or dataframe 
    arraylike that contains the permeability coefficients 'k'. If a dataframe 
    is supplied, the permeabitlity coefficient column name 'kname' must be 
    specified. 
arr_aq: array-like , pandas series or dataframe 
    array-like that contains the aquifer groups. If NAN values exists in the 
    aquifer groups, it is suggested to imputed values before feediing to 
    the algorithms. Missing values are not allowed. If dataframe is supplied,
    the aquifer group column name 'aqname' must be specified. 

{params.core.kname}

aqname: str, optional, 
    Name of aquifer group columns. `aqname` allows to retrieve the 
    aquifer group `arr_aq` value in  a specific dataframe. Commonly
   `aqname` needs to be supplied when a dataframe is passed as a positional 
    or keyword argument. 
    
subjectivity: bool, default=False
    Considers each class label as a naive group of aquifer. Subjectivity 
    occurs when no group of aquifer is not found in the data. Therefore, each 
    class label is considered as a naive group of aquifer. It is strongly 
    recommended to provide a default group passes to parameter `default_arr` 
    to substitute the group of aquifers for more pratical reason. For instance
    it can be the layer collected at a specific depth like the 'strata' 
    columns. 
    
default_arr: array-like, pd.Series 
   Array used as deefault for subsitutue the group of aqquifer if the latter 
   is missing. This is an heuristic option because it might lead to breaking 
   code or invalid results.
   
keep_label_0: bool, default=False
    The prediction already include the label 0. However, including 0 in 
    the predicted label refers to 'k=0' i.e. no permeability coefficient 
    equals to 0, which is not True in principle, because all rocks  have 
    a permeability coefficient 'k'. Here we considered 'k=0' as an undefined 
    permeability coefficient. Therefore, '0' , can be exclude since, it can 
    also considered as a missing 'k'-value. If predicted '0' is in the target 
    it should mean a missing 'k'-value rather than being a concrete label.  
    Therefore, to avoid any confusion, '0' is altered to '1' so the value 
    `+1` is used to move forward all class labels thereby excluding 
    the '0' label. To force include 0 in the label, set `keep_label_0` 
    to ``True``. 
        
method: str ['naive', 'strict'], default='naive'
    The kind of strategy to compute the representativity of a label 
    in the predicted array 'array_aq'. It can also be 'strict'. Indeed:
    
    - ``naive`` computes the importance of the label by the number of its
        occurence for this specific label in the array 'k'. It does not 
        take into account of the occurence of other existing labels. This 
        is usefull for unbalanced class labels in `arr_k`.
    - ``strict`` computes the importance of the label by the number of 
        occurence in the whole valid `arr_k` i.e. under the total of 
        occurence of all the labels that exist in the whole 'arra_aq'. 
        This can give a suitable anaylse results if the data is not 
        unbalanced for each labels in `arr_k`.
        
Returns
-------
_Group: :class:`~.box._Group` class object 
    Use attribute `.groups` to find the group values. 

Examples
----------
(1) Use the real aquifer group collected in the area 

>>> from watex.utils import naive_imputer, read_data, reshape 
>>> from watex.datasets import load_hlogs 
>>> from watex.utils.hydroutils import classify_k, find_aquifer_groups 
>>> b= load_hlogs () #just taking the target names
>>> data = read_data ('data/boreholes/hf.csv') # read complete data
>>> y = data [b.target_names]
>>> # impute the missing values found in aquifer group columns
>>> # reshape 1d array along axis 0 for imputation 
>>> agroup_imputed = naive_imputer ( reshape (y.aquifer_group, axis =0 ) , 
...                                    strategy ='most_frequent') 
>>> # reshape back to array_like 1d 
>>> y.aquifer_group =reshape (agroup_imputed) 
>>> # categorize the 'k' continous value in 'y.k' using the default 
>>> # 'k' mapping func 
>>> y.k = classify_k (y.k , default_func =True)
>>> # get the group obj
>>> group_obj = find_aquifer_groups(y.k, y.aquifer_group) 
>>> group_obj 
_Group(Label=[' 1 ', 
             Preponderance( rate = '53.141  %', 
                           [('Groups', {{'V': 0.32, 'IV': 0.266, 'II': 0.236, 
                                        'III': 0.158, 'IV&V': 0.01, 
                                        'II&III': 0.005, 'III&IV': 0.005}}),
                            ('Representativity', ( 'V', 0.32)),
                            ('Similarity', 'V')])],
        Label=[' 2 ', 
              Preponderance( rate = ' 19.11  %', 
                           [('Groups', {{'III': 0.274, 'II': 0.26, 'V': 0.26, 
                                        'IV': 0.178, 'III&IV': 0.027}}),
                            ('Representativity', ( 'III', 0.27)),
                            ('Similarity', 'III')])],
        Label=[' 3 ', 
              Preponderance( rate = '27.749  %', 
                           [('Groups', {{'V': 0.443, 'IV': 0.311, 'III': 0.245}}),
                            ('Representativity', ( 'V', 0.44)),
                            ('Similarity', 'V')])],
             )
(2) Use the subjectivity and set the strata columns as default array 

>>> find_aquifer_groups(y.k, subjectivity=True, default_arr= X.strata_name ) 
_Group(Label=[' 1 ', 
             Preponderance( rate = '53.141  %', 
                           [('Groups', {{'siltstone': 0.35, 'coal': 0.227, 
                                        'fine-grained sandstone': 0.158, 
                                        'medium-grained sandstone': 0.094, 
                                        'mudstone': 0.079, 
                                        'carbonaceous mudstone': 0.054, 
                                        'coarse-grained sandstone': 0.03, 
                                        'coarse': 0.01}}),
                            ('Representativity', ( 'siltstone', 0.35)),
                            ('Similarity', 'siltstone')])],
        Label=[' 2 ', 
              Preponderance( rate = ' 19.11  %', 
                           [('Groups', {{'mudstone': 0.288, 'siltstone': 0.205, 
                                        'coal': 0.192, 
                                        'coarse-grained sandstone': 0.137, 
                                        'fine-grained sandstone': 0.137, 
                                        'carbonaceous mudstone': 0.027, 
                                        'medium-grained sandstone': 0.014}}),
                            ('Representativity', ( 'mudstone', 0.29)),
                            ('Similarity', 'mudstone')])],
        Label=[' 3 ', 
              Preponderance( rate = '27.749  %', 
                           [('Groups', {{'mudstone': 0.245, 'coal': 0.226, 
                                        'siltstone': 0.217, 
                                        'fine-grained sandstone': 0.123, 
                                        'carbonaceous mudstone': 0.066, 
                                        'medium-grained sandstone': 0.066, 
                                        'coarse-grained sandstone': 0.057}}),
                            ('Representativity', ( 'mudstone', 0.24)),
                            ('Similarity', 'mudstone')])],
             )
""".format(params=_param_docs
)
    
def label_importance (
    label: int, 
    arr_k: ArrayLike , 
    arr_aq:ArrayLike, 
    *, 
    method:str='naive' 
    )->dict:
    """Compute the score for the label and its representativity in the valid 
    array 'arr_k' 
    
    Parameters 
    -------------
    label: int, or string  
        class label from the true labels array of  permeability coefficient 'k'.
        If string, be sure to convert the array to hold the dtype str. It is 
        recommnended to provide data with no NaN to have full control the 
        occurence results. 

    arr_k: array-like 1d 
        True labels of array containing the permeability coefficient 'k'.

    arr_aq: array_like 1d 
        True labels of the groups of aquifers or predicted naive group of 
         aquifer (NGA labels). See :func:`~.predict_NGA_labels`.
         
    method: str ['naive', 'strict'], default='naive'
        The kind of strategy to compute the representativity of a label 
        in the predicted array 'array_aq'. It can also be 'strict'. Indeed:
        
        - 'naive' computes the importance of the label by the number of its
            occurence for this specific label in the array 'k'. It does not 
            take into account of the occurence of other existing labels. This 
            is usefull for unbalanced class labels in 'arr_k'
        - 'strict' computes the importance of the label by the number of 
            occurence in the whole valid 'arr_k' i.e. under the total of 
            occurence of all the labels that exist in the whole 'arra_aq'. 
            This can give a suitable anaylse results if the data is not 
            unbalanced for each labels in 'arr_k'.
        
    Returns 
    -----------
    label_dict_group_rate: dict, 
        Dictionnary of the label and its  rate of occurence in the `arr_aq`. 
        Thus each group in `arr_aq` has its rate of representativity of the 
        label in `arr_k`.
        
    Examples 
    -----------
    >>> from watex.datasets import load_hlogs
    >>> from watex.utils.hydroutils import label_importance, classify_k 
    >>> array_k = load_hlogs().frame.k 
    >>> # categorize k_labels using default categorization 
    >>> array_k = classify_k (array_k, default_func =True )
    >>> # for the demo, we used the group of aquifers however in 
    >>> # pratice, NGA should  be prediced labels instead. 
    >>> array_aq = load_hlogs().frame.aquifer_group  
    >>> # get the labels except NaN 
    >>> np.unique (array_k) # give the k label in data; here only k=2 is available
    array([ 2., nan])
    >>> # compute the representativity of label ='2' ( for k=1) 
    >>> label_importance(label = 2, arr_k= array_k, arr_aq= array_aq )
    {' II ': 1.0}
    >>> # let take the example of 11 boreholes, note that the 'hf.csv'
    >>> # data use for demo is not  not avaibale in the package for confidency 
    >>> # just use for demonstration 
    >>> from watex.utils import read_data 
    >>> cdata = read_data ('data/boreholes/hf.csv') 
    >>> array_k = cdata.k ; array_aq= cdata.aquifer_group 
    >>> np.unique (array_k) # give the labels in k
    array([ 1.,  2.,  3., nan])
    >>> array_k = classify_k(array_k, default_func =True)
    >>> # will compute the representativity of each label  using the 
    >>> # the method 'strict'
    >>> for label in [1, 2, 3]: 
            r=label_importance(label , array_k, array_aq , 
                                       method ='strict') 
            print("label k =", label, ':\n' , r)
    label k = 1 :
     {'V': 0.17, 'IV': 0.141, 'II': 0.126, 'III': 0.084, 'IV&V': 0.005, 
      'II&III': 0.003, 'III&IV': 0.003}
    label k = 2 :
     {'III': 0.052, 'II': 0.05, 'V': 0.05, 'IV': 0.034, 'III&IV': 0.005}
    label k = 3 :
     {'V': 0.123, 'IV': 0.086, 'III': 0.068}
    >>> # **comments: 
        # label k=1 is 17% importance for group V, 12.3% for group II whereas
        # label k=2 has a weak rate in the whole dataset ~=0.19% for all groups
        # the most dominate labels are k=1 and k=3 with 53.14% and 27.74 % 
        # respectively in the dataset. 
        # If threshold of representativity is set to 50% , none of the true 
        # label k will fit any aquifer group since the max representativity 
        # score is 17% and is for the group V especially for k=1. 
    """ 
    arr_k = check_y (
        arr_k, 
        allow_nan=True , 
        input_name="Array 'arr_k'",
        )  
    arr_aq = check_y(
        arr_aq, 
        input_name="Array 'arr_aq'", 
        )
    
    _check_consistency_size(arr_k, arr_aq)

    assert str(method).lower().strip()  in {"naive", "strict"}, (
        f"Supports only 'naive' or 'strict'. Got {method!r}")
    method =str(method).lower().strip() 
    # if NaN exists get the non_valid k 
    if np.isnan(arr_k).any() : 
        not_nan_indices , = np.where (~np.isnan(arr_k))
        arr_aq = arr_aq[not_nan_indices] 
        arr_k = arr_k [not_nan_indices] 
        
    if not _is_numeric_dtype(arr_k):
        # therefore convert array_aq too to dtype string 
        arr_aq = to_dtype_str( arr_aq , return_values= True ) 
        label =str (label) # for consistency 
        # this is usefull when using np.unique since 
        # numeric data cannot be coerced  with string dtype 
    if label not in (np.unique (arr_k)): 
        raise ValueError (f"Missing '{label}' in array. {label!r} must be"
                          " a label included in 'arr_k'. Valid labels are:"
                          f" {list(np.unique (arr_k))}"
                          )
    # indices where label k exists in arr_k 
    index, = np.where (arr_k ==label )
    # find its corresponding value from indices in groups arr_aq
    label_in_arr_q = arr_aq[index ]
    # count the labels that fits label k in arr_k
    label_group , group_counts = np.unique (
        label_in_arr_q, return_counts=True ) 
    # compute ratio, compare to its importance 
    # in the whole valid array_K
    tot = sum(group_counts) if method =='naive' else len(arr_k) 
    label_dict_group_rate = { k: round (v, 3) for k , v in zip (
        label_group, group_counts/tot)
        } 
    # sort
    label_dict_group_rate = dict( sorted (
        label_dict_group_rate.items() ,
        key=lambda x:x[1], reverse =True )
        ) 
    
    return label_dict_group_rate
 

def find_similar_labels ( 
    y_true, 
    y_pred,  
    *, 
    categorize_k:bool=False, 
    threshold: float=None, 
    func: callable=None, 
    keep_label_0 :bool=False, 
    method:str='naive', 
    return_groups:bool=False, 
    **kwd
        ):
    """Find similarities between y_true and y_pred and returns rate 
    
    Parameters 
    -----------
    y_true: array-like 1d or pandas.Series 
        Array containing the true labels of 'k' 
    y_pred: array_like, or pandas.Series
        array containing the predicted naive group of aquifers (NGA)  
        
    categorize_k: bool, 
        If set to ``True``, user needs to provide a function `ufunc` to map 
        or categorize the permeability coefficient 'k' into an integer 
        labels. 
        
    func: callable 
       Function to specifically map the permeability coefficient column 
       in the dataframe of serie. If not given, the default function can be 
       enabled instead from param `default_func`.     
        
    threshold: float, default=None 
        The threshold from which, label in 'y_true' can be considered  
        similar than the one in NGA labels 'y_pred'. The default is 'None' which 
        means none rule is considered and the high preponderence or occurence 
        in the data compared to other labels is considered as the most 
        representative  and similar. Setting the rule instead by fixing 
        the threshold is recommended especially in a huge dataset.
        
    keep_label_0: bool, default=0
        Force including 0 in the predicted label if  `include_label_0` is set 
        to ``True``. Mostly label '0' refers to 'k=0' i.e. no permeability 
        coefficient equals to 0, which is not True in principle, because all rocks  
        have a permeability coefficient 'k'. Here we considered 'k=0' as an undefined 
        permeability coefficient. Therefore, '0' , can be exclude since, it can 
        also considered as a missing 'k'-value. If predicted '0' is in the target 
        it should mean a missing 'k'-value rather than being a concrete label.  
        Therefore, to avoid any confusion, '0' is removed by default in the 'k'
        categorization. However, when the prediction 'y_pred' is made from the 
        the unsupervising method, the prediction '0' straigthforwardly includes
         '0' i.e 'k=0' as a first class. So the value `+1` is used to move forward 
        all class labels thereby excluding the '0' label. To force include 0 
        in the label, set `include_label_0` to ``True``. 
        
    method: str ['naive', 'strict'], default='naive'
        The kind of strategy to compute the representativity of a label 
        in the predicted array 'y_pred'. It can also be 'strict'. Indeed:
        
        - ``naive`` computes the importance of the label by the number of its
            occurence for this specific label in the array 'y_true'. It does not 
            take into account of the occurence of other existing labels. This 
            is usefull for unbalanced class labels in `y_true`.
        - ``strict`` computes the importance of the label by the number of 
            occurence in the whole valid `y_true` i.e. under the total of 
            occurence of all the labels that exist in the whole 'arra_aq'. 
            This can give a suitable anaylse results if the data is not 
            unbalanced for each labels in `y_pred`.
            
    return_groups: bool, default=False 
        Returns label groups and their values counts in the predicted 
        labels `y_pred`  where 'k' values are not missing. 
    
    Returns 
    --------- 
    g.similarity : Tuple of  labels found that are considered similar in 
        predicted labels. 
    g.group: Tuple of group that have their similarity in the true labels 
    
    Example 
    ----------
    >>> from watex.utils import read_data 
    >>> from watex.utils.hydroutils import find_similar_labels, classify_k
    >>> data = read_data ('data/boreholes/hf.csv')
    >>> ymap = classify_k(data.k , default_func =True) 
    >>> # Note that for the demo we use the group of aquifer columns, however
    >>> # in pratical example, y_pred must be a predicted NGA labels. This 
    >>> # is possible using the function <predict_NGA_labels> 
    >>> sim = find_similar_labels(y_true= ymap, y_pred=data.aquifer_group)
    >>> sim 
    ... ((1, 'V'), (2, 'III'), (3, 'V'))
    >>> group= find_similar_labels(ymap, data.aquifer_group, return_groups=True) 
    >>> group 
    ... ((1,
      {'V': 0.17,
       'IV': 0.141,
       'II': 0.126,
       'III': 0.084,
       'IV&V': 0.005,
       'II&III': 0.003,
       'III&IV': 0.003}),
     (2, {'III': 0.052, 'II': 0.05, 'V': 0.05, 'IV': 0.034, 'III&IV': 0.005}),
     (3, {'V': 0.123, 'IV': 0.086, 'III': 0.068}))
    >>> find_similar_labels(y_true= ymap, y_pred=data.aquifer_group,
                                  threshold = 0.15) 
    ... [(1, 'V')]
    
    """
    [  _assert_all_types(o, pd.Series, np.ndarray, objname = lab) 
         for lab, o  in zip (
                 ["'y_true'(true labels)", "'y_pred '( predicted labels )'"], 
                 [y_true, y_pred]) 
    ]

    _check_consistency_size(y_true, y_pred) 
    if not all ([ _is_arraylike_1d(ar ) for ar in (y_true, y_pred )] ) :
        raise TypeError ("True and predicted labels supports only "
                         "one-dimensional array.")
    # check arrays for consistency
    y_true = check_y (
        y_true, 
        allow_nan= True, 
        to_frame =True, 
        input_name="y_true",
        )  

    y_pred = check_y (
        y_pred, 
        to_frame = True, 
        allow_nan= False, 
        input_name ="NGA labels"
        )
        
    if categorize_k : 
        #categorize k if func is given.
        y_true = classify_k( y_true ,  func= func ,  **kwd)
    g = find_aquifer_groups(y_true, arr_aq= y_pred,keep_label_0= keep_label_0,
                            method= method, 
                            ) 
    # Fetch similarity according to the  threshold 
    simg = tuple (_similarity_rules ( list(g.groups), threshold = threshold )
                  ) 
    similarities = [] if len(simg)==0 else [
        (label, list(value)[0]) for label, value  in simg ]

    return similarities  if not return_groups else tuple (g.groups )

def _similarity_rules (lg,  threshold =.5 ):
    """ Considers two labels similar from the threshold value. 
    
    :param lg: dict, 
        dictionnary of  tuple pair (true_label, dict of group occurence) 
    :param threshold: float, default =.25 
        The threshold to consider two label similar from the rate of 
        their occurences. 
    :return: 
        - A generator object from :func:`_similarity_rules`
        
    :example:
    >>> from watex.utils.hydroutils import _similarity_rules 
    >>> groups = ((1,{'V': 0.32,'IV': 0.266,'II': 0.236,'III': 0.158,
       'IV&V': 0.01,'II&III': 0.005,'III&IV': 0.005}),
     (2, {'III': 0.274, 'II': 0.26, 'V': 0.26, 'IV': 0.178, 'III&IV': 0.027}),
     (3, {'V': 0.443, 'IV': 0.311, 'III': 0.245}))
    >>> _similarity_rules (groups , threshold = .4 )
    ...  <generator object _similarity_rules.<locals>.<genexpr> at 0x00000255448B4BA0>
    >>> tuple (_similarity_rules (groups , threshold = .4 ))
    ... ((3, {'V': 0.443, 'IV': 0.311, 'III': 0.245}),)
        
    """
   
    threshold = threshold or .0
    if isinstance (threshold, str): 
        try : 
            threshold = float(threshold.replace("%", '')
                              )/1e2 if '%' in threshold else threshold 
        except: 
            raise TypeError ("Threshold must be a number between 0 and "
                             f"1, got: {type(threshold).__name__!r}")
    # the gdict is already sorted 
    threshold = float(
        _assert_all_types(threshold, int, float, objname="Threshold" ))
    
    if threshold < 0. or threshold > 1: 
        raise ValueError ("Threshold expects a value ranged between 0 and 1,"
                          f" got: {threshold}")
    for k , g in lg:
        if g.get (list(g)[0]) >= threshold : 
            yield (k, g )
      
def _get_y_from_valid_indexes (
        y_true, y_pred =None , *,  include_label_0 = False , replace_nan = False 
        ): 
    """From valid indices in true labels 'y_true', get the valid 
    valid y array as as possible the value at the valid indices from 'y_true' 
    in predicted labels' 
    :param y_true: 1d- array-like 
        array composing of true labels 
    :param y_pred: 1d array-like
        array composing of predicted labels 
    :param include_label_0: bool, default=False 
        keep 0 of the predicted label as a particular class label. 

    :returns:  (y_true | ypred) array-like 1d
       - y_true: returns array of valid indices only if 'y_pred' is ``None``
       -y_pred: returns array of valid indices got from true labels 'y_true'
       
    :example: 
        >>> import numpy as np 
        >>> from watex.utils.hydroutils import _get_y_from_valid_indexes 
        >>> y_true = np.array ([ np.nan, 1, 1, 2, 3, 2, 3, 1, 3, np.nan])
        >>> y_pred = np.array ([0, 0, 0, 1, 2, 2, 4, 5, 1, 4])
        >>> # for includ label is set to 'False'
        >>> yt, yp =_get_y_from_valid_indexes (y_true, y_pred)
        >>> yt  
        ... array([1, 1, 2, 3, 2, 3, 1, 3]) # remove indexes where NaN values 
        >>> yp  
        ... array([1, 1, 2, 3, 3, 5, 6, 2])
        >>> # include label to True 
        >>> yt, yp =_get_y_from_valid_indexes (y_true, y_pred)
        >>> yp 
        ... array([0, 0, 1, 2, 2, 4, 5, 1])
        
    """
    msg =("{} supports only one-dimensional array")
    
    if not _is_arraylike_1d(y_true) : 
        raise TypeError (msg.format ("True labels 'y_true'"))
    
    if y_pred is not None: 
        _check_consistency_size(y_true, y_pred) 
        if not _is_arraylike_1d(y_pred) :
            raise TypeError (msg.format("Predicted labels 'y_pred'"))
            
        ## Only replace NaN in y_pred array if there 
        # is no cheaper, heuristic option.    
        if hasattr(y_pred, 'name') and isinstance (y_pred, pd.Series): 
            y_pred = y_pred.values 
       
    indices,  =  np.where (~np.isnan (y_true )) 
    y_true= y_true [ indices ]
    y_true= np.array (y_true).astype (np.int32) 
    
    if y_pred is not None:
        if ( 0 not in list(np.unique (y_pred))): 
            if include_label_0 : 
                warnings.warn("'0' label does not exist "
                              "in the predicted labels.")
            include_label_0 =True 
        y_pred= y_pred[indices ] if include_label_0 else \
            y_pred[indices ] + 1
    
    return  y_true if y_pred is None else (y_true, y_pred )
  
#XXXTODO terminate the label score 
# computation and move it in metric module    
def label_score (y_true , y_pred , metric ="accuracy_score" ):
    """ Compute the score of each true label and its similarity in 
    the predicted label 'y_pred' 
    """
    scores =dict ()
    for label in list(np.unique (y_true) ): 
        indexes, = np.where (y_true ==label ) 
        yp = y_pred[indexes]
        score = metric (y_true [indexes] , yp ) 
        scores[label] = score  
        
    return scores 
 
def select_base_stratum (
    d: Series | ArrayLike | DataFrame , 
    /, 
    sname:str = None, 
    stratum:str= None,
    return_rate:bool=False, 
    return_counts:bool= False, 
    ):
    """ Selects base stratum from the the strata column in the logging data. 
    
    Find the most recurrent stratum in the data and compute the rate of 
    occurrence. 
    
    Parameters 
    ------------
    d: array-like 1D , pandas.Series or DataFrame
        Valid data containing the strata. If dataframe is passed, 'sname' is 
        needed to fetch strata values. 
    sname: str, optional 
        Name of column in the dataframe that contains the strata values. 
        Dont confuse 'sname' with 'stratum' which is the name of the valid 
        layer/rock in the array/Series of strata. 
    stratum: str, optional 
        Name of the base stratum. Must be self contain as an item of the 
        strata data. Note that if `stratum` is passed, the auto-detection of 
        base stratum is not triggered. It returns the same stratum , however
        it can gives the rate and occurence of this stratum if `return_rate` 
        or `return_counts` is set to ``True``. 
    return_rate: bool,default=False, 
        Returns the rate of occurence of the base stratum in the data. 
    return_counts: bool, default=False, 
        Returns each stratum name and the occurences (count) in the data. 
    
    Returns 
    ---------
    bs: str 
        base stratum , self contain in the data 
    r: float 
        rate of occurence in base stratum in the data 
    c: tuple (str, int)
        Tuple of each stratum whith their occurrence in the data. 
        
    Example 
    --------
    >>> from watex.datasets import load_hlogs 
    >>> from watex.utils.hydroutils import select_base_stratum 
    >>> data = load_hlogs().frame # get only the frame 
    >>> select_base_stratum(data, sname ='strata_name')
    ... 'siltstone'
    >>> select_base_stratum(data, sname ='strata_name', return_rate =True)
    ... 0.287292817679558
    >>> select_base_stratum(data, sname ='strata_name', return_counts=True)
    ... [('siltstone', 52),
         ('fine-grained sandstone', 40),
         ('mudstone', 37),
         ('coal', 24),
         ('Coarse-grained sandstone', 15),
         ('carbonaceous mudstone', 9),
         ('medium-grained sandstone', 2),
         ('topsoil', 1),
         ('gravel layer', 1)]
    """
    _assert_all_types(d, pd.DataFrame, pd.Series, np.ndarray )
    
    if hasattr(d, 'columns'): 
        if sname is None :
            raise TypeError ("'sname' ( strata column name )  can not be "
                              "None when a dataframe is passed.")
        sn= copy.deepcopy(sname)
        sname = _assert_all_types(sname, str, objname ='Name of stratum column') 
        sname = is_in_if(d.columns, sname, error ='ignore')
        if sname is None: 
            raise ValueError ( f"Name {sn!r} is not a valid column strata name."
                              " Please, check your data.") 
        sname =sname [0] if isinstance(sname, list) else sname 
        sdata = d[sname ]    

    elif hasattr (d, '__array__') and not hasattr (d, 'name'):
        if not _is_arraylike_1d(d): 
            raise StrataError("Strata data supports only one-dimensional array."
                             )
        sdata = d
        
    if stratum is not None: 
        if not stratum in set (sdata):
            out= listing_items_format(set(sdata), begintext = 'strata', 
                                      verbose = False )
            raise StrataError (f"Stratum {stratum!r} not found in the data."
                              f" Expects {out}")
    #compute the occurence of the stratum in the data: 
    bs,  r , c  = _get_s_occurence(sdata , stratum )
        
    return ( ( r , c )  if ( return_rate and return_counts) else  ( 
            r if return_rate else c ) if return_rate or return_counts else bs 
            ) 

def _get_s_occurence (
        sd, /,  bs = None , reverse= True, key = 1, 
        ) -> Tuple [str, float, List ]: 
    """ Returns the occurence of the object in the data. 
    :param sd: array-like 1d of  data 
    :param bs: str - base name of the object. If 'bs' if given the auto 
        search  will not be used. 
    :param key: int, default=1 
        key of ordered sorted dict. Must be either {0, 1}: `0` for key 
        ordered searcg while `1` is for value search. 
    :param reverse: bool, reverse ordered dictionnary
    :returns: bs, r, c
        return the base object, rate or counts.
    """
    # sorted strata in ascending occurence 
    s=dict ( Counter(sd ) ) 
    sm = dict (
        sorted (s.items () , key= lambda x:x[key], reverse =reverse )
        )
    bs = list(sm) [0]  if bs is None else bs 
    r= sm[bs] / sum (sm.values ()) # ratio
    c = list(zip (sm.keys(), sm.values ())) 
    
    return  bs,  r , c

            
def get_compressed_vector(
    d, /, 
    sname,  
    stratum =None , 
    strategy ="average", 
    as_frame = False, 
    random_state = None, 
    )-> Series :
    """ Compresses base stratum data into a singular vector composed of all 
    feature names in the targetted data `d`. 
    
    Parameters 
    ------------
    d: pandas DataFrame
        Valid data containing the strata. If dataframe is passed, 'sname' is 
        needed to fetch strata values. 
    sname: str, optional 
        Name of column in the dataframe that contains the strata values. 
        Dont confuse 'sname' with 'stratum' which is the name of the valid 
        layer/rock in the array/Series of strata. 
    stratum: str, optional 
        Name of the base stratum. Must be self contain as an item of the 
        strata data. Note that if `stratum` is passed, the auto-detection of 
        base stratum is not triggered. It returns the same stratum , however
        it can gives the rate and occurence of this stratum if `return_rate` 
        or `return_counts` is set to ``True``. 
    
    strategy: str , default='average' or 'mean', 
        strategy used to select or compute the numerical data into a 
        singular series. It can be ['naive']. In that case , a single serie 
        if randomly picked up into the base strata data.
    as_frame: bool, default='False'
        Returns compressed vector into a dataframe rather that keeping in 
        series. 
    random_state: int, optional, 
        State for randomly selected a compressed vector when ``naive`` is 
        passed as strategy.
    
    Returns 
    --------
    ms: pandas series/dataframe 
        returns a compressed vector in pandas series compose of all features. 
        Note , the vector here does not refer as math vector compose of 
        numerical values only. A compressed vector here is a series that is 
        the result of averaging the numerical features of the base stratum and 
        incluing its corresponding categorical values. Note there, the  `ms`
        can contain categorical values and has the same number and features as 
        the original frame `d`. 
    
    Example
    -------
    >>> from watex.datasets import load_hlogs 
    >>> from watex.utils.hydroutils import get_compressed_vector 
    >>> data = load_hlogs().frame # get only the frame  
    >>> get_compressed_vector (data, sname='strata_name')[:4]
    ... hole_number           H502
        strata_name      siltstone
        aquifer_group           II
        pumping_level       ZFSAII
        dtype: object
    >>> get_compressed_vector (data, sname='strata_name', as_frame=True )
    ...   hole_number strata_name aquifer_group  ...        r     rp remark
        0        H502   siltstone            II  ...  41.7075  59.23    NaN
        [1 rows x 23 columns]
    >>> get_compressed_vector (data, sname='strata_name', strategy='naive')
    ... hole_number          H502
        depth_top          379.15
        depth_bottom        379.7
        strata_name     siltstone
        Name: 39, dtype: object
    """
    _assert_all_types(d, pd.DataFrame, objname = "Data for samples compressing")

    d= check_array(
        d, 
        force_all_finite="allow-nan", 
        dtype =object, 
        input_name="Data for squeezing",
        to_frame =True, 
        )
    sname = _assert_all_types(sname, str , "'sname' ( strata column name )")
    
    assert strategy in {'mean', 'average', 'naive'}, "Supports only strategy "\
        f"'mean', 'average' or 'naive'; got {strategy!r}"
    if stratum is None: 
        stratum = select_base_stratum(d, sname= sname, stratum= stratum )
    stratum = _assert_all_types(stratum, str , objname = 'Base stratum ')
    #group y and get only the base stratum data 
    pieces = dict(list(d.groupby (sname))) 
    bs_d  = pd.DataFrame( pieces [ stratum ]) 
    # get the numerical features only before  applying operation 
    _, numf , catf  = to_numeric_dtypes(bs_d , return_feature_types= True )
    
    if strategy  in ('mean', 'average') :
        #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        from ..exlib.sklearn import SimpleImputer 
        #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        ms = bs_d[ numf ].mean() 
        if len(catf)!=0:
            # Impute data and fill the gap if exists
            #  by the most frequent categorial features.
            sim = SimpleImputer(strategy = 'most_frequent') 
            xt = sim.fit_transform(bs_d[catf]) 
            bs_dc = pd.DataFrame(xt , columns = sim.feature_names_in_ ) 
            # get only single value of the first row 
            bs_init = bs_dc .iloc [0 , : ] 
            #ms.reset_index (inplace =True ) 
            ms = pd.concat ( [ bs_init, ms  ], axis = 0 ) 
    elif strategy =='naive':
        random_state= random_state or 42 
        # randomly pick up one index 
        rand = np.random.RandomState (random_state )
        # if use sample , -> return a list and must 
        # specify the k number of sequence , 
        # while here , only a single is is expected: like 
        # random.sample (list(rand.permutation (X0.index )) , 1 )
        ix = random.choice (rand.permutation (bs_d.index )) 
        ms = bs_d.loc [ix ] 
        
    return  ms  if not as_frame  else pd.DataFrame(
        dict(ms) , index = range (1))

def _assert_reduce_indexes (*ixs ) : 
    """ Assert reducing indexing and return a list of valids indexes `ixs`"""
    ixs = list(ixs )
    for ii, ix in enumerate (ixs): 
        if not is_iterable( ix) : 
            raise IndexError ("Expects a pair tuple or list i.e.[start, stop]'"
                              f" for reducing indexing; got {ix}") 
        if len(ix) !=2 : 
            raise IndexError(f"Index must be a pair [start, top]: got {ix}")
        try:
            ix = [int (i) for i in ix ]
        except : 
            raise IndexError("Index should be a pair tuple/list of integers;"
                             f" check {ix}")
        else: ixs[ii] = ix 
        
    return ixs 

def get_sections_from_depth  (z, z_range, return_index =False ) :
    """ Gets aquifer sections ('upper', 'lower') in data 'z' from the 
    depth range.
    
    This might be usefull to compute the thickness of the aquifer. 
    
    Parameters 
    ----------
    z: array-like 1d or pd.Series 
        Array or pandas series contaning the depth values 
    z_range: tuple (float), 
        Section ['upper', 'lower'] of the aquifer at differnt depth.
        The range of the depth must a pair values and  could not be
         greater than the maximum depth of the well. 
    return_index: bool, default=False 
        returns the indices of the sections ['upper', 'lower'] 
        of the aquifer and non-valid sections too. 
        
    Returns 
    ----------
    sections: Tuple (float, float)
       Real values of the  upper and lower sections of the aquifer. 
    If ``return_index`` is 'True', function returns: 
      (upix, lowix): Tuple (int, int )
          indices of upper and lower sections in the depth array `z`
      (invix): list of Tuple (int, int) 
          list of indices of invalid sections
          
    Example
    --------
    >>> from watex.datasets import load_hlogs 
    >>> from watex.utils.hydroutils import get_sections_from_depth
    >>> data= load_hlogs().frame  
    >>> # get real sections from depth 16.25 to 125.83 m
    >>> get_sections_from_depth ( data.depth_top, ( 16.25, 125.83))
    ...  (22.46, 128.23)
    >>> # aquifer depth from 16.25 m to the end 
    >>> get_sections_from_depth ( data.depth_top, ( 16.25,))
    ... (22.46, 693.37)
    >>> get_sections_from_depth ( data.depth_top, ( 16.25, 125.83),
                                 return_index =True )
    ... ((3, 11), [(0, 3), (11, 180)])
    >>> get_sections_from_depth ( data.depth_top, ( 16.25,), 
                                 return_index =True )
    ... ((3, 181), [(0, 3)])
 
    """
    z = _assert_all_types(z, pd.Series, np.ndarray , "Depth")
    if not _is_arraylike_1d (z) : 
        raise DepthError( "Depth expects one-dimensional array.")
        
    # check depth z array 
    z= check_y (
        z, 
        input_name= "Array of depth 'z'", 
        to_frame =True, 
        )
    if not is_iterable(z_range): 
        return TypeError ("Depth range must be an iterable object,"
                          f" not {type (z_range).__name__!r}")
    z_range= sorted ( list(z_range ) ) 
    if max(z_range ) > max(z): 
        raise DepthError("Depth value cannot be greater than the maximum "
                         f"depth in the well= {max(z)}; got {max(z_range)}")
    if len(z_range)==1: 
        warnings.warn("Single value is passed. Remember, it may correspond "
                      "to the depth value of the upper section thin the end.")
        z_range = z_range + [max (z )]
    elif len(z_range) > 2: 
        raise DepthError( "Too many values for the depth section range."
                         "Expects a pair values [ upper, lower] sections."
                         )
    # get the indices from depth 
    upix  = np.argmin  ( np.abs ( 
        (np.array(z) - z_range [0] ) ) ) 
    lowix = np.argmin  ( np.abs (
        (np.array(z) - z_range [-1] ) ) ) 
    # for consistency , reset_zrange with 
    # true values from depth z 
    sections = ( z [upix ], z[lowix ] )  
    z_range =  np.array ( ( upix , lowix ) , dtype = np.int32 ) 

    # compute the difference between adjacent depths
    diff = np.diff (z) 
    # when depth 
    if set (sections )==1: 
        raise DepthError("Upper and lower sections must have different depths.")
    
    if ( float( np.diff (sections)) <=diff.min() ): 
        # thickness to pass to another layers 
        raise DepthError(f"Depth {z_range} are too close that probably "
                         "figure out the same layer. Difference between "
                         "adjacent depths must be greater than"
                        f" {round ( float(diff.min()), 2) }")
    # not get the index from non valid data
    # +1 for Python indexing
    invix = _get_invalid_indexes (z, z_range )
    
    return  sections if not  return_index else ( 
        ( upix , lowix + 1 ),  invix ) 


def get_unique_section (
        *data, zname, kname,  return_index=False, return_data =False, 
        error='raise', **kws ) : 

    sect, dat = get_aquifer_sections(*data, zname=zname, kname=kname, 
                                 return_index =return_index, 
                                 return_data= True,
                                 error = error , **kws)
    sect = np.array (list(itertools.chain(*sect)))
    si = np.array ([sect.min(), sect.max()], 
                   dtype = np.int32 if return_index else np.float32 )
    return si if not return_data else  ( si, dat ) 

get_unique_section.__doc__="""\
Get the section to consider unique in multiple aquifers. 

The unique section 'upper' and 'lower' is the valid range of the whole 
sections of each aquifers. It is  considered as  the main valid section 
from which data can not be compressed and not altered. For instance,  
the use of indexes is  necessary to shrunk the data except this valid 
section. Mosly the data from the section is considered the valid data as the 
predictor Xr. Out of the range of aquifers ection, data can be discarded or 
compressed to top Xr. 

Returns valid section indexes if 'return_index' is set to ``True``.    
    
Parameters
-----------
d: list of pandas dataframe 
    Data that contains mainly the aquifer values. It needs to specify the 
    name of the depth column `zname` as well as the name of permeabiliy 
    `kname` column.  
{params.core.zname}
{params.core.kname}
{params.core.z}

return_index: bool, default =False , 
    Returns the positions (indexes) of the upper and lower sections of the
    shallower  and deep aquifers found in the whole  dataframes.
return_data: bool, default=False, 
    Return valid data. It is usefull when 'error' is set to 'ignore'
    to collect the valid data. 
error: str, default='raise' 
    Raise errors if trouble occurs when computing the section of each aquifer. 
    If 'ignore', a UserWarning is displayed when invalid data is found. Any 
    other value of `error` will set error to `raise`. 
kws: dict, 
    Additional keywords arguments passed  to  
    :func:`~watex.utils.hydroutils.get_aquifer_sections`.
    
Returns 
--------
up, low :list of upper and lower section values of aquifer.
    - (upix, lowix ): Tuple of indexes of lower and upper sections  
    - (up, low): Tuple of aquifer sections (upper and lower)  
    - (upix, lowix), (up, low) : positions and sections values of aquifers 
        if `return_index` and return_sections` are ``True``.  

See Also 
----------
watex.utils.hydroutils.get_aquifer_section: compute single section

watex.utils.hydroutils.get_aquifer_sections: compute multiple sections 
 

Example
-------   
>>> from watex.datasets import load_hlogs 
>>> data = load_hlogs ().frame 
>>> get_unique_section (data.copy() , zname ='depth', kname ='k', ) 
... array([197.12, 369.71], dtype=float32)
>>> get_unique_sections (data.copy() , zname ='depth', kname ='k', 
                                return_index =True)
... array([16, 29])

""".format(
    params=_param_docs,
    )
    
def get_aquifer_sections (
    *data ,  
    zname, 
    kname, 
    return_index =False, 
    return_data=False,
    error = 'ignore',  
    **kws 
    ): 

    errors = []
    is_valid_dfs = [] ; is_not_valid =[]
    section_indexes ,sections =[] , []
    
    error ='raise' if error !='ignore' else 'ignore'

    for ii, df in enumerate ( data) : 
        try : 
            ix, sec = get_aquifer_section(
                df , 
                zname = zname , 
                kname = kname , 
                return_index= True, 
                return_sections=True, 
                **kws
                )
            is_valid_dfs .append (df )
        except Exception as err :
            # if error =='raise':
            #     raise err
            errors.append(str(err))
            is_not_valid.append (ii + 1 )
            continue 
        section_indexes.append(ix); sections.append(sec )
        
    if len(is_not_valid)!=0 : 
        verb = f"{'s' if len(is_not_valid)>1 else''}"
        msg = "Unsupports data at position{0} {1}.".format( verb, 
             smart_format(is_not_valid))
                     
        if error =='raise':
            getr = ("Sections", "computed" 
                      )  if not return_index else  ("Indices", "obtained" )
            btext = "\nReason{}".format(verb)
            entext = "{0} cannot be {1}. Please check your data.".format ( 
                getr[0], getr[-1])
            mess = msg +  listing_items_format(
                errors, begintext=btext, endtext=entext , verbose =False )
            raise DatasetError(mess) 
            
        warnings.warn(msg + " Data {} discarded.".format( 
            "is" if len(is_not_valid)<2 else "are")
                      )        
    r= section_indexes if return_index else sections 
    
    return  r  if not return_data else ( r , is_valid_dfs) 

get_aquifer_sections.__doc__="""\
Get the section of each aquifer form multiple dataframes. 
 
The unique section 'upper' and 'lower' is the valid range of the whole 
data to consider as a  valid data. 
The use of the index is  necessary to shrunk the data of the whole 
boreholes. Mosly the data from the section is consided the valid data as the 
predictor Xr. Out of the range of aquifers ection, data can be discarded or 
compressed to top Xr. 

Returns valid section indexes if 'return_index' is set to ``True``.    
   
Parameters 
------------ 
data: list of pandas dataframe 
    Data that contains mainly the aquifer values. It needs to specify the 
    name of the depth column `zname` as well as the name of permeabiliy 
    `kname` column.  
{params.core.zname}
{params.core.kname}
{params.core.z}

return_index: bool, default =False , 
    Returns the positions (indexes) of the upper and lower sections of the
   each aquifer found in each dataframe.

error: str, default='ignore' 
    Raise errors if trouble occurs when computing the section of each aquifer. 
    If 'ignore', a UserWarning is displayed if invalid data is found. Any 
    other value of `error` will set error to `raise`. 
return_data: bool, default=False, 
    Return valid data. It is usefull when 'error' is set to 'ignore'
    to collect the valid data. 
       
kws: dict, 
    Additional keywords arguments passed  to  
    :func:`~watex.utils.hydroutils.get_aquifer_sections`.
    
Returns 
--------
up, low :list of upper and lower section values of aquifer.
    - (upix, lowix ): Tuple of indexes of lower and upper sections  
    - (up, low): Tuple of aquifer sections (upper and lower)  
    - (upix, lowix), (up, low) : positions and sections values of aquifers 
        if `return_index` and return_sections` are ``True``.  

See Also 
----------
watex.utils.hydroutils.get_aquifer_sections: 
    compute multiples aquifer sections

Example
-------   
>>> from watex.datasets import load_hlogs 
>>> from watex.utils.hydroutils import get_aquifer_sections
>>> data = load_hlogs ().frame 
>>> get_aquifer_sections (data, data , zname ='depth', kname ='k' ) 
... [[197.12, 369.71], [197.12, 369.71]]
>>> get_aquifer_sections (data, data , zname ='depth', kname ='k' , 
                           return_index =True ) 
...  [[16, 29], [16, 29]]

""".format(
    params=_param_docs,
    )
def _get_invalid_indexes  ( d, /, valid_indexes, in_arange =False ): 
    """ Get non valid indexes from valid section indexes 
    
    :param d: array_like 1d 
        array-like data for recover the section range indexes 
    :param section_ix: Tuple (int, int) 
        Index of upper and lower sections
    :param in_arange: bool, 
        List all index values. 
    :returns: 
        invix: List(Tuple(int))
        Returns invalid indexes onto a list 
    Example 
    -----------
    >>> from watex.utils.hydroutils import _get_invalid_indexes
    >>> import numpy as np 
    >>> idx = np.arange (50) 
    >>> _get_invalid_indexes (idx , (3, 11 ))
    ... [(0, 3), (12, 50)]
    
    """
    if in_arange : 
        valid_indexes = np.array (  list( 
            range ( * [  valid_indexes [0] , valid_indexes [-1] +1 ] )))  
        mask = _isin(range(len(d)), valid_indexes, return_mask=True )
        invix = np.arange (len(d))[~mask ]
    else :
        # +1 for Python indexing
        invix =  (np.arange (len(d))[:valid_indexes [0] + 1 ],
                  np.arange (len(d) + 1 )[valid_indexes[1]+1 : ]) 
        invix=  [ ( min(ix) , max(ix))  for ix in invix  if  ( 
            len(ix )!=0 and len(set(ix))>1)  ] # (181, 181 )
    
    return invix 

    
def get_xs_xr_splits (
    data, 
    /,
    z_range = None, 
    zname = None, 
    section_indexes:Tuple[int, int]=None, 
    )-> Tuple [DataFrame ]:
    """Split data into matrix :math:`X_s` with sample :math:`ms` (unwanted data ) 
    and :math:`X_r` of samples :math:`m_r`( valid aquifer data )
    
    Parameters 
    -----------
    data: pandas dataframe 
        Dataframe for compressing. 
    zname: str,int , 
        the name of depth column. 'name' needs to be supplied 
        when `section_indexes` is not provided. 
    z_range: tuple (float), 
        Section ['upper', 'lower'] of the aquifer at different depth.
        The range of the depth must a pair values and  could not be
        greater than the maximum depth of the well.
    section_indexes: tuple or list of int 
        list of a pair tuple or list of integers. It is be the the valid 
        sections( upper and lower ) indexes of  of the aquifer. If 
        the depth range `z_range` and `zname` are supplied, `section_indexes`
        can be None.  Note that the last indix is considered as the last 
        position, the bottom of the section therefore, its value is 
        included in the data.
        
    Returns
    --------
    - xs : list of pandas dataframe 
        - shrinking part of data for compressing. Note that it is on list 
        because if dataframe corresponds to the non-valid dataframe sections. 
    - xr: pandas dataframe  
        - valid data reflecting to the aquifer part or including the 
        aquifer data. 
        
    Example
    --------
    >>> from watex.datasets import load_hlogs 
    >>> from watex.utils.hydroutils import get_xs_xr_splits 
    >>> data = load_hlogs ().frame 
    >>> xs, xr = get_xs_xr_splits (data, 3.11, section_indexes = (17, 20 ) )
    """
    xs, xr = None, None
    
    data= check_array(
        data, 
        force_all_finite="allow-nan", 
        dtype =object, 
        input_name="Data for squeezing",
        to_frame =True, 
    )
    
    if section_indexes is not None: 
        section_indexes = _assert_reduce_indexes (section_indexes) [0] 
        if section_indexes [1] > len(data ): 
            # if index is if wide,take the first index thin the end 
            section_indexes = [section_indexes[0], len(data)]
        invalid_indexes = _get_invalid_indexes(
            np.arange (len(data)), section_indexes)  

    # valid section index of aquifer
    elif z_range is not None : 
        z = is_valid_depth (data, zname = zname , return_z = True)
        section_indexes, invalid_indexes = get_sections_from_depth(
            z, z_range, return_index=True )

    # +1 for Python index 
    try : 
        xr = data.iloc [range (*[section_indexes[0], section_indexes[-1] +1])]
    except IndexError : 
        # break +1 of Python index and take index thin the end. 
        xr = data.iloc [range (*[section_indexes[0], section_indexes[-1]])]
    except Exception as err :
        raise err 
    invalid_indexes = _assert_reduce_indexes(*invalid_indexes )
    max_ix = max (list(itertools.chain(*invalid_indexes)))
    
    if  max_ix > len(data) :
        raise IndexError(f"Wrong index! Index {max_ix} is out of range "
                         f"of data with length = {len(data)}")
 
    xs = [ data.iloc[ range (* ind)] for ind in invalid_indexes]

    return xs, xr 

def reduce_samples (
    *data , 
    sname, 
    zname=None, 
    kname= None,
    section_indexes=None,  
    error='raise', 
    strategy= 'average',  
    verify_integrity=False, 
    ignore_index=False, 
    **kws
    )->List[DataFrame] : 
    
    msg = ("'Soft' mode is triggered for samples reducing."
           " {0} number{1} of data passed are not valid."
           " Remember that data must contain the 'depth' and"
           " aquifer values. Should be discarded during the"
           " computing of aquifer sections. This might lead to"
           " breaking code or invalid results. Use at your own "
           " risk." 
        )

    df0 = copy.deepcopy(data) # make a copy of frame 
    dfs = _validate_samples( *df0 )  
    
    dfs=[df.reset_index() for df in dfs] # reset index 
    # get the aquifer sections firts 
    if section_indexes is None: 
        section_indexes, dfs = get_unique_section(
            *dfs, zname=zname, kname=kname, error= error, 
            return_data =True, return_index=True 
            )
        
        if len(df0)!=len(dfs): 
            warnings.warn ( msg.format(len(section_indexes), 
                        "s" if len(section_indexes)>1 else ""))
        
    Xs, Xr =[], []
    for df in dfs : 
        xs, xr = get_xs_xr_splits (df, section_indexes= section_indexes)
        Xs.append(xs) ; Xr.append(xr)
        
    d_new=[]
    for  df_xs , df_xr in zip ( Xs , Xr ): 
        # # compute the base stratum for 
        # each each reduce sections 
        bases_s = [ select_base_stratum(d, sname=sname )
                    for i, d in enumerate (df_xs) ] 
        # reduce sample for each invalid section with 
        # missing k 
        comp_vecs = [ get_compressed_vector( d, sname=sname , stratum = st,  
                     as_frame =True , strategy=strategy, 
            ) for i, (st , d)  in enumerate ( zip (bases_s , df_xs))  ]
        # get the index to stack the compresed sample with 
        # the valid part of aquifer data. 
        xs_indexes = [( min( df.index), max(df.index)) for df in df_xs ]
        # concat the compress with xr 
        df_= _concat_compressed_xs_xr(
            xs_indexes =xs_indexes ,xr_indexes = section_indexes, 
                compressed_frames = comp_vecs, 
                xr= df_xr )
        d_new.append (df_)

    if not ignore_index: 
        # got back inial data. 
        d_new = [ df.drop ( columns = 'index') 
                  if 'index' in df.columns else df 
                  for df in d_new 
                  ]
    # verify integrity first
    # before reset index 
    if verify_integrity: 
        d_new = [  df.drop_duplicates(subset=None, keep='first',  
            ignore_index=ignore_index ) for df in d_new ] 
        
    if ignore_index : 
        # reset the index of the new data frame
        d_new = [df.reset_index () for df in d_new ]
        d_new = [ df.drop (columns = 'level_0' or 'index') if
                 ('level_0' or 'index')  in df.columns else df 
                 for df in d_new  ]
    
    return d_new 

reduce_samples.__doc__ ="""\
Create a new dataframe by squeezing/compressing the non valid data. 

The m-samples reduction is necessary for the dataset with a lot of 
missing k-values. The technique of shrinking the number of k0 –values 
(k-missing values ) seems a relevant idea. It consists to compressed the 
values of the missing :math:`k -values from the top ( depth equals 0 ) 
thin the upper section of the first aquifer with lower depth into 
a single vector :math:`x_r` with dimension (1×n ) i.e. contains 
the n-features.  
 
Parameters 
-----------
data: list of dataframes
    Data that contains mainly the aquifer values. It must contains the 
    depth values refering at the column_name passed at `zname`  and 
    the permeability coefficient `k` passed to `kname` . Both argument need 
    t supplied when datafame as passes as positional arguments.
    
sname: str, optional 
    Name of column in the dataframe that contains the strata values. 
    Dont confuse 'sname' with 'stratum' which is the name of the valid 
    layer/rock in the array/Series of strata. 

{params.core.zname}
{params.core.kname}
{params.core.z}

strategy: str , default='average' or 'mean', 
    strategy used to select or compute the numerical data into a 
    singular series. It can be ['naive']. In that case , a single serie 
    if randomly picked up into the base strata data.
    
section_indexes: tuple or list of int 
    list of a pair tuple or list of integers. It is be the the valid 
    sections( upper and lower ) indexes of  of the aquifer. If 
    the depth range `z_range` and `zname` are supplied, `section_indexes`
    can be None.  Note that the last indix is considered as the last 
    position, the bottom of the section therefore, its value is 
    included in the data.
        
error: str, default='raise' 
    Raise errors if trouble occurs when computing the section of each aquifer. 
    If 'ignore', a UserWarning is displayed when invalid data is found. Any 
    other value of `error` will set error to `raise`. 

verify_integrity: bool, default=False
    Check the new index for duplicates. Otherwise defer the check until 
    necessary. Setting to False will improve the performance of 
    this method.
    if 'True', remove the duplicate rows from a DataFrame.
    
        subset: By default, if the rows have the same values in all the 
        columns, they are considered duplicates. This parameter is used 
        to specify the columns that only need to be considered for 
        identifying duplicates.
        keep: Determines which duplicates (if any) to keep. It takes inputs as,
        first – Drop duplicates except for the first occurrence. 
        This is the default behavior.
        last – Drop duplicates except for the last occurrence.
        False – Drop all duplicates.
        inplace: It is used to specify whether to return a new DataFrame or 
        update an existing one. It is a boolean flag with default False.
ignore_index: bool, default=False, 
    It is a boolean flag to indicate if row index should 
    be reset after dropping duplicate rows. False: It keeps the original 
    row index. True: It reset the index, and the resulting rows will be 
    labeled 0, 1, …, n – 1. 
    
Returns 
----------
df_new: List of pandas.dataframes
    new dataframes with reducing samples. 
    
Example 
--------
>>> from watex.datasets import load_hlogs
>>> from watex.utils.hydroutils import reduce_samples 
>>> data = load_hlogs ().frame # get the frames 
>>> # add explicitly the aquifer section indices 
>>> dfnew= reduce_samples (data.copy(), sname='strata_name', 
                             section_indexes = (16, 29 ),)
>>> dfnew[0]
...    hole_number               strata_name     rock_name  ...      r     rp  remark
    0         H502                  mudstone           J2z  ...    NaN    NaN     NaN
    16        H502                 siltstone           NaN  ...  35.74  59.23     NaN
    17        H502    fine-grained sandstone           NaN  ...  35.74  59.23     NaN
    18        H502                 siltstone           NaN  ...  35.74  59.23     NaN
    19        H502    fine-grained sandstone           NaN  ...  35.74  59.23     NaN
    20        H502                  mudstone           NaN  ...  35.74  59.23     NaN
    21        H502                 siltstone           NaN  ...  35.74  59.23     NaN
    22        H502    fine-grained sandstone           NaN  ...  59.61  59.23     NaN
    23        H502                 siltstone           NaN  ...  59.61  59.23     NaN
    24        H502    fine-grained sandstone           NaN  ...  59.61  59.23     NaN
    25        H502  Coarse-grained sandstone           NaN  ...  59.61  59.23     NaN
    26        H502                  mudstone           NaN  ...  82.33  59.23     NaN
    27        H502    fine-grained sandstone           NaN  ...  82.33  59.23     NaN
    28        H502  Coarse-grained sandstone           J2z  ...  82.33  59.23     NaN
    29        H502                      coal  (J2y)  2coal  ...  82.33  59.23     NaN
    0         H502                 siltstone           NaN  ...    NaN    NaN     NaN

[16 rows x 23 columns]
>>> # specify the column name and kname without section indexes 
>>> dfnew= reduce_samples (
    data.copy(), sname='strata_name', data, zname='depth', kname='k', 
    ignore_index= True )[0]
... dfnew[0].index # index is reset 
... RangeIndex(start=0, stop=16, step=1)

""".format(
    params=_param_docs,
    )
                                  
def _concat_compressed_xs_xr (
        xs_indexes:List[int], 
        xr_indexes: List[int], 
        compressed_frames:List[DataFrame], 
        xr:DataFrame  ):
    """ Concat the compressed frames from `xs` with the valid frames.
    
    Use the index of different frames to merge the frame by respecting the 
    depth positions. For instance, if the valid secion of aquifer is framed 
    between two invalid sections composed of missing 'k' values, the both
    sections are shrank and their compressed frames are also framed the 
    section of valid data. This keep the position of the 
    aquifer intact. This is usefull for prediction purpose. 
    
    :param xs_indexes: list of int 
        indices of invalid sections 
    :param xr_indexes: list of int ,
        indices of valid section of aquifer. valid data 
    :param compressed_frames: pandas dataframe 
        the compressed frames from `xs`. 
    :param xr: dataframe 
        valid data ( contain the aquifer sections )
    """
    pos = [ np.array(k).mean() for k in xs_indexes ]
    dics = dict ( zip ( pos , compressed_frames))
    
    dics [np.array(xr_indexes).mean()]= xr 
    # sorted strata in ascending occurence 
    sm = dict (
        sorted (dics.items () , key= lambda x:x[0])
        )
    c= list(sm.values ())
    return  pd.concat (c )

    
def is_valid_depth (z, /, zname =None , return_z = False): 
    """ Assert whether depth is valid in dataframe of two-dimensional 
    array passed to `z` argument. 
    
    Parameters 
    ------------
    z: ndarray, pandas series or dataframe 
        If Dataframe is given, 'zname' must be supplied to fetch or assert 
        the depth existence of the depth in `z`. 
    zname: str,int , 
        the name of depth column. 'name' needs to be supplied when `z` is 
        given whereas index is needed when `z` is an ndarray with two 
        dimensional. 
        
    return_X_z: bool, default =False
        returns z series or array  if set to ``True``. 
    
    Returns 
    ---------
    z0, is_z: array /bool, 
        An array-like 1d of `z` or 'True/False' whether z exists or not. 
        
    Example 
    --------
    >>> from watex.datasets import load_hlogs 
    >>> from watex.utils.hydroutils import is_valid_depth 
    >>> d= load_hlogs () 
    >>> X= d.frame 
    >>> is_valid_depth(X, zname='depth') # is dataframe , need to pass 'zname'
    ... True
    >>> is_valid_depth (X, zname = 'depth', return_z = True)
    ... 0        0.00
        1        2.30
        2        8.24
        3       22.46
        4       44.76
         
        176    674.02
        177    680.18
        178    681.68
        179    692.97
        180    693.37
        Name: depth_top, Length: 181, dtype: float64
    """
    is_z =True 
    z = _assert_all_types(z, np.ndarray , pd.Series, pd.DataFrame, 
                          objname ='Depth') 
    zname = _assert_all_types(zname, str, objname ="'zname"
                              ) if zname is not None else None  
    if hasattr(z, '__array__') and hasattr (z, 'name'): 
        zname = z.name 
        
    elif hasattr (z ,'columns' ): 
        # assert whether depth 
        # mape a copy to not corrupt X since the function 
        # remove the depth in columns 
        z_copy = z.copy() 
        if zname is None: 
            raise ValueError ("'zname' ( Depth column name ) can not be None"
                              " when a dataframe is given.")
        # --> deals with depth 
        # in the case depth is given while 
        # dataframe is given. 
        # if z is not None: 
        #     zname =None # set None 
        if zname is not None : 
            # erased the depth and name
            try: 
                _, z0 = is_depth_in(
                z_copy, name = zname, error = 'raise') 
            except Exception as err:
                if return_z: 
                    raise DepthError("Depth name 'zname' " + str(
                        err).replace ('E', 'e') )
                    
                else: is_z= False  
                
        zname= z0.name 
    elif hasattr (z, '__array__'): 
        if not _is_arraylike_1d (z): 
            raise ValueError ("Multidimensional 'k' array is not allowed"
                              " Expect one-dimensional array.")
        z0= pd.Series (z, name =zname) if zname is not None else z 

    return z0 if return_z else is_z  

def get_aquifer_section (
        arr_k, /, zname=None, kname = None,  z= None, 
        return_index = False, return_sections = True 
        ) : 
    _assert_all_types( arr_k, pd.DataFrame, np.ndarray)
    
    if z is not None: 
        ms = (f"Depth {type(z).__name__} size must be consistent with"
             f" {type (arr_k).__name__!r};got {len(z)} and {len(arr_k)}."
             )
        _assert_all_types(z, np.ndarray, pd.Series)
        
        if not _is_arraylike_1d(z): 
            raise DepthError ("Depth supports only one-dimensional array,"
                             f" not {type(z).__name__!r}.")
            # check depth z array 
        z= check_y (
            z, 
            input_name= "Array of depth 'z'", 
            to_frame =True, 
            )
        if not _check_consistency_size(z, arr_k, error ='ignore'): 
            raise DepthError (ms)
                
    if (z is None and zname is not None ): 
        z = is_valid_depth ( arr_k , zname = zname , return_z = True )
        zname = z.name 
        
    elif ( z is None and zname is None ): 
           raise TypeError ("Expects an array of depth 'z' or  depth column"
                            " name 'zname' in the dataframe.")    
        
    if hasattr (arr_k ,'columns' ):
        # deal with arr_k 
        if kname is None: 
            raise ValueError ("'kname' ( Permeability coefficient ) column name"
                              " cannot be None when a dataframe is given.") 
        else: 
            _assert_all_types(kname, str , int , float,  objname="'kname'") 
            
        if isinstance (kname , (int, float)): 
            kname = int (kname) 
            if kname > len(arr_k.columns): 
                raise IndexError (f"'kname' at index {kname} is out of the "
                                  f"dataframe column size={len(arr_k.columns)}")
                
            kname = arr_k.columns[kname]
            
        if kname not in arr_k.columns:
            raise ValueError (f"'kname' {kname!r} not found in dataframe.")
        
        arr_k = arr_k[kname] 
        arr_k= arr_k.values 
        
    elif hasattr (arr_k, '__array__'): 
        if not _is_arraylike_1d (arr_k): 
            raise ValueError ("Multidimensional 'k' array is not allowed"
                              " Expect one-dimensional array.")

    # for consistency, set all to 1d array 
    z = reshape (z) ; arr_k = reshape (arr_k)

    indexes,  = np.where (~np.isnan (arr_k)) 
    if hasattr (indexes, '__len__'): 
        # +1 for Python indexing
        indexes =[ indexes [0 ] , indexes [-1]] 
        
    sections = z[indexes ]
    
    return ( [* indexes ], [* sections ])   if ( 
        return_index and return_sections ) else  ( 
            [*indexes ] if return_index else  [*sections])

get_aquifer_section.__doc__="""\
Detect a single aquifer section (upper and lower) in depth.  

This is useful trip to compute the thickness of the aquifer.

Parameters 
-----------
arr_k: ndarray or dataframe 
    Data that contains mainly the aquifer values. It can also contains the 
    depth values. If the depth is included in the `arr_k`, `zname` needs to 
    be supplied for recovering and depth. 
    
{params.core.zname}
{params.core.kname}
{params.core.z}

return_index: bool, default =False , 
    Returns the positions (indexes) of the upper and lower sections of the
     aquifer found in the dataframe `arr_k`. 
return_sections: bool, default=True, 
    Returns the sections (upper and lower) of the aquifers. 

Returns 
--------
up, low :list of upper and lower section values of aquifer.
    - (upix, lowix ): Tuple of indexes of lower and upper sections  
    - (up, low): Tuple of aquifer sections (upper and lower)  
    - (upix, lowix), (up, low) : positions and sections values of aquifers 
        if `return_index` and return_sections` are ``True``.  

Example
-------
>>> from watex.datasets import load_hlogs 
>>> from watex.utils.hydroutils import get_aquifer_section 
>>> data = load_hlogs ().frame # return all data including the 'depth' values 
>>> get_aquifer_section (data , zname ='depth', kname ='k')
... [197.12, 369.71] # section starts from 197.12 -> 369.71 m 
>>> get_aquifer_section (data , zname ='depth', kname ='k', return_index=True) 
... ([16, 29], [197.12, 369.71]) # upper and lower-> position 16 and 29.


""".format(
    params=_param_docs,
    )
    
def _kp (k, /,  kr= (.01 , .07 ), string = False ) :
    """ Default permeability 'k' mapping using dict to validate the continue 
    value 'k' 
    :param k: float, 
        continue value of the permeability coefficient 
    :param kr: Tuple, 
        range of permeability coefficient to categorize 
    :param string: bool, str 
        label to prefix the the categorial value. 
    :return: float/str - new categorical value . 

    """
    d = {0: k <=0 , 1: 0 < k <= kr[0], 2: kr[0] < k <=kr[1], 3: k > kr[1] 
         }
    label = 'k' if str(string).lower()=='true' else str(string )
    for v, value in d.items () :
        if value: return v if not string else  ( 
                label + str(v) if not math.isnan (v) else np.nan ) 

def classify_k (
        o:DataFrame| Series | ArrayLike, /,  func: callable|F= None , 
        kname:str=None, inplace:bool =False, string:str =False, 
        default_func:bool=False  
        ):
    """ Categorize the permeability coefficient 'k'
    
    Map the continuous 'k' into categorial classes. 
    
    Parameters 
    ----------
    o: ndarray of pd.Series or Dataframe
        data containing the permeability coefficient k contineous values. 
        If data is passsed as a pandas dataframe, the column containing the 
        k-values `kname` needs to be specified. 
    func: callable 
        Function to specifically map the permeability coefficient column 
        in the dataframe of serie. If not given, the default function can be 
        enabled instead from param `default_func`. 
    inplace: bool, default=False 
        Modified object inplace and return None 
    string: bool, 
        If set to "True", categorized map from 'k'  should be prefixed by "k". 
        However is string value is given , the prefix is changed according 
        to this label. 
    default_ufunc: bool, 
        Default function for mapping k is setting to ``True``. Note that, this 
        could probably not fitted your own data. So  it is recommended to 
        provide your own function for mapping 'k'. However the default 'k' 
        mapping is given as follow: 
            
        - k0 {0}: k = 0 
        - k1 {1}: 0 < k <= .01 
        - k2 {2}: .01 < k <= .07 
        - k3 {3}: k> .07 
    Returns
    --------
    o: None,  ndarray, Series or Dataframe 
        return None only if dataframe is given and `inplace` is set 
        to ``True`` i.e modified object inplace. 
        
    Examples 
    --------
    >>> import numpy as np 
    >>> from watex.datasets import load_hlogs 
    >>> from watex.utils.hydroutils import classify_k 
    >>> _, y0 = load_hlogs (as_frame =True) 
    >>> # let visualize four nonzeros values in y0 
    >>> y0.k.values [ ~np.isnan (y0.k ) ][:4]
    ...  array([0.054, 0.054, 0.054, 0.054])
    >>> classify_k (y0 , kname ='k', inplace =True, use_default_func=True )
    >>> # let see again the same four value in the dataframe 
    >>> y0.k.values [ ~np.isnan (y0.k ) ][:4]
    ... array([2., 2., 2., 2.]) 
    
    """
    _assert_all_types(o, pd.Series, pd.DataFrame, np.ndarray)
    
    dfunc = lambda k : _kp (k, string = string ) # default 
    func = func or   ( dfunc if default_func else None ) 
    if func is None: 
        raise TypeError ("'ufunc' cannot be None when the default"
                         " 'k' mapping function is not triggered.")
    oo= copy.deepcopy (o )
    if hasattr (o, 'columns'):
        if kname is None: 
            raise ValueError ("kname' is not set while dataframe is given. "
                              "Please specify the name of permeability column.")
        is_in_if( o, kname )
  
        if inplace : 
            o[kname] = o[kname].map (func) 
            return 
        oo[kname] = oo[kname].map (func) 
        
    elif hasattr(o, 'name'): 
        oo= oo.map(func ) 
  
    elif hasattr(o, '__array__'): 
        oo = np.array (list(map (func, o )))
        
    return oo 

#XXXTODO compute t parameters 
def transmissibility (s, d, time, ): 
    """Transmissibility T represents the ability of aquifer's water conductivity.
    
    It is the numeric equivalent of the product of hydraulic conductivity times
    aquifer's thickness (T = KM), which means it is the seepage flow under the
    condition of unit hydraulic gradient, unit time, and unit width
    
    """
    ... 
      
def check_flow_objectivity ( y ,/,  values, classes  ) :
    """ Function checks the flow rate objectivity
    
    If objective is set to `flow` i.e the prediction focuses on the flow
    rate, there are some conditions that the target `y` needs to meet when 
    values are passed for classes categorization. 
    
    :param values: list of values to encoding the numerical target `y`. 
        for instance ``values=[0, 1, 2]`` 
    :param objective: str, relate to the flow rate prediction. Set to 
        ``None`` for any other predictions. 
    :param prefix: the prefix to add to the class labels. For instance, if 
        the `prefix` equals to ``FR``, class labels will become:: 
            
            [0, 1, 2] => [FR0, FR1, FR2]
            
    :param classes: list of classes names to replace the default `FR` that is 
        used to specify the flow rate. For instance, it can be:: 
            
            [0, 1, 2] => [sf0, sf1, sf2]
    :returns:
        (y, classes): Tuple, 
        - y: array-like 1d  of categorized  `y` 
        - classes: list of flow rate classes. 
    """
    msg= ("Objective is 'flow' whereas the target value is set to {0}."
          " Target is defaultly encoded to hold integers {1}. If"
          " the auto-categorization does not fit the real values"
          " of flow ranges, please set the range of the real flow values"
          " via param `values` or `label_values`."
          ) 
    y=check_y( 
        y, 
        input_name=" Flow array 'y'", 
        to_frame=True
        )
    if values is None:
        msg = ("Missing values for categorizing 'y'; the number of"
                " occurence in the target is henceforth not allowed."
                )
        warnings.warn("Values are not set. The new version does not" 
                      " tolerate the number of occurrence to be used."
                      " Provide the list of flow values instead.",
                      DeprecationWarning )
        raise TypeError (msg)
        
    elif values is not None: 
        if isinstance(values,  (int, float)): 
           y =  categorize_target(y , labels = int(values) )
           warnings.warn(msg.format(values, np.unique (y) ))
           values = np.unique (y)
        
        elif isinstance(values, (list, tuple, np.ndarray)):
            y = np.unique(y) 
            if len(values)!=len(y): 
                warnings.warn("Size of unique identifier class labels"
                              " and the given values might be consistent."
                              f" Idenfier sizes = {len(y)} whereas given "
                              f" values length are ={len(values)}. Will"
                              " use the unique identifier labels instead.")
                values = y 
                
            y = categorize_flow(y, values, classes=classes  )
        else : 
            raise ValueError("{type (values).__name__!r} is not allow"
                             " Expect a list of integers.")
            
    classes = classes or values 
    return y, classes 
 
@catmapflow2(cat_classes=['FR0', 'FR1', 'FR2', 'FR3'])#, 'FR4'] )
def categorize_flow(
        target: Series | ArrayLike[T] ,
        flow_values: List [float],
        **kwargs
    ) -> Tuple[ List[float], T, List[str]]: 
    """ 
    Categorize `flow` into different classes. If the optional
    `flow_classes`  argument is given, it should be erased the
    `cat_classes` argument of decororator `deco.catmapflow`.
    
    Parameters 
    ------------
    target: array-like, pandas.Series, 
        Flow array to be categorized
    
    flow_values: list of str 
        Values for flow categorization; it distributes the flow values as
        numerical values. For instance can be ranged as a tuple of bounds 
        as below :: 
    
            flow_values= [0.0, [0.0, 3.0], [3.0, 6.0], [6.0, 10.0], 10.0] (1)
            
        or it can also accept the list of integer label identifiers as::
            
            flow_values =[0. , 3., 6., 10.] (2)
        
        For instance runing the step (2) shoud convert the flow rate bounds to 
        reach the step (1). The arrangement of the flow rate obeys some criteria 
        which depend of the types of hydraulic system required according to the
        number of inhabitants living on a survey locality/villages or town.
        The common request flow rate during the campaigns for drinling 
        water supply can be  organized as follow: 
            
            flow_values =[0,  1,  3 , 10  ]
            classes = ['FR0', 'FR1', 'FR2', 'FR3']
    
        where :
            - ``FR0`` equals to values =0  -> dry boreholes 
            - ``FR1`` equals to values between  0-1(0< value<=1) for Village 
                hydraulic systems (VH)
            - ``FR2`` equals to values between  1-1 (1< value<=3) for improved  
                village hydraulic system (IVH)
            - ``FR3`` greather than 3 (>3) for urban hydraulic system (UH)
            
            Refer to [1]_ for more details. 
        
    classes: list of str , 
        literal labels of categorized flow rates. If given, should be 
        consistent with the size of `flow_values`'
    
        
    Returns 
    ---------
    (new_flow_values, target, classes)
        - ``new_flow_values``: Iterable object as type (2) 
        - ``target``: Raw flow iterable object to be categorized
        - ``classes``: If given , see ``classes`` params. 
            
    References 
    -------------
    .. [1] Kouadio, K.L., Kouame, L.N., Drissa, C., Mi, B., Kouamelan, K.S., 
        Gnoleba, S.P.D., Zhang, H., et al. (2022) Groundwater Flow Rate 
        Prediction from Geo‐Electrical Features using Support Vector Machines. 
        Water Resour. Res. :doi:`10.1029/2021wr031623`
        
    .. [2] Kra, K.J., Koffi, Y.S.K., Alla, K.A. & Kouadio, A.F. (2016) Projets 
        d’émergence post-crise et disparité territoriale en Côte d’Ivoire. 
        Les Cah. du CELHTO, 2, 608–624.
        
        
    """
    classes =  kwargs.pop('classes', None)

    new_flow_values = []
    inside_inter_flag= False
    
    if isinstance(flow_values, (tuple, np.ndarray)): 
        flow_values =list(flow_values)
    # Loop and find 
    for jj, _iter in enumerate(flow_values) : 
        if isinstance(_iter, (list, tuple, np.ndarray)): 
            inside_inter_flag = True 
            flow_values[jj]= list(_iter)
 
    if inside_inter_flag: 
        new_flow_values =flow_values 
    
    if inside_inter_flag is False: 
        flow_values= sorted(flow_values)
        # if 0. in flow_values : 
        #     new_flow_values.append(0.) 
        for ss, val in enumerate(flow_values) : 
            if ss ==0 : 
                #append always the first values. 
                 new_flow_values.append(val) 
            # if val !=0. : 
            else:
                if val ==flow_values[-1]: 
                    new_flow_values.append([flow_values[ss-1], val])
                    new_flow_values.append(val)
                else: 
                   new_flow_values.append([flow_values[ss-1], val])
 
    return new_flow_values, target, classes        

@writef(reason='write', from_='df')
def exportdf (
    df : DataFrame =None,
    refout: Optional [str] =None, 
    to: Optional [str] =None, 
    savepath:Optional [str] =None,
    modname: str  ='_wexported_', 
    reset_index: bool =True
) -> Tuple [DataFrame, Union[str, str], bool ]: 
    """ 
    Export dataframe ``df``  to `refout` files. 
    
    `refout` file can be Excell sheet file or '.json' file. To get more details 
    about the `writef` decorator , see :doc:`watex.utils.decorator.writef`. 
    
    :param refout: 
        Output filename. If not given will be created refering to the 
        exported date. 
        
    :param to: Export type; Can be `.xlsx` , `.csv`, `.json` and else.
       
    :param savepath: 
        Path to save the `refout` filename. If not given
        will be created.
    :param modname: Folder to hold the `refout` file. Change it accordingly.
        
    :returns: 
        - `df_`: new dataframe to be exported. 
        
    """
    if df is None :
        warnings.warn(
            'Once ``df`` arguments in decorator :`class:~decorator.writef`'
            ' is selected. The main type of file ready to be written MUST be '
            'a pd.DataFrame format. If not an error raises. Please refer to '
            ':doc:`~.utils.decorator.writef` for more details.')
        
        raise FileHandlingError(
            'No dataframe detected. Please provided your dataFrame.')

    df_ =df.copy(deep=True)
    if reset_index is True : 
        df_.reset_index(inplace =True)
    if savepath is None :
        savepath = savepath_(modname)
        
    return df_, to,  refout, savepath, reset_index   

def categorize_target(
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
        of category to split 'y'. For instance ``label=3`` and applied on 
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
    
    if validate_labels(arr, new_names, return_bool= True): 
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
    """ A shadow function of :func:`watex.utils.funcutils.cattarget`. 
    
    :param ar: array-like of numerical values 
    :param labels: int or list of int, 
        the number of category to split 'ar'into. 
    :param order: str, optional, 
        the order of label to ne categorized. If None or any other values, 
        the categorization of labels considers only the leangth of array. 
        For instance a reverse array and non-reverse array yield the same 
        categorization samples. When order is set to ``strict``, the 
        categorization  strictly consider the value of each element. 
        
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


def validate_labels (t, /, labels, return_bool = False): 
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
    >>> validate_labels (ybin, [0, 1])
    ... [0, 1] # all labels exist. 
    >>> validate_labels (y, [0, 1, 3])
    ... ValueError: Value '3' is missing in the target.
    >>> validate_labels (ybin, 0 )
    ... [0]
    >>> validate_labels (ybin, [0, 5], return_bool=True ) # no raise error
    ... False
        
    """
    
    if not is_iterable(labels):
        labels =[labels] 
        
    t = np.array(t)
    mask = np.isin(t, labels) 
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

def _validate_samples (*dfs , error:str ='raise'): 
    """ Validate data . 
     check shapes and the columns items in the data.
     
    :param dfs: list of dataframes or array-like 
        Dataframe must have the same size along axis 1. If error is 'ignore'
        error is muted if the length ( along axis 0) of data does not fit 
        each other. 
    :param error: str, default='raise' 
        Raise absolutely error if data has not the same shape, size and items 
        in columns. 
    :return: 
        valid_dfs: List of valida data. If 'error' is 'ignore' , It still 
        returns the list of valid data and excludes the invalid all times 
        leaving an userwarnmimg.
        
    """
    shape_init = dfs[0].shape[1]
    [ _assert_all_types(df, np.ndarray, pd.DataFrame) for df in dfs ]
    diff_shape , shapes  , cols = [], [],[]
    
    col_init = dfs[0].columns if hasattr (dfs[0] , 'columns') else [] 
    valid_dfs =[]
    for k , df in enumerate (dfs) : 
        if df.shape[1] != shape_init :
            diff_shape.append(k) 
        else: valid_dfs.append (df )
        
        shapes.append (df.shape)
        if hasattr (df, 'columns'): 
            cols.append (list(df.columns ))
            
    countshapes = list(Counter (shapes )) # iterable object 
    occshapes = countshapes [0] # the most occurence shape
    if len(diff_shape )!=0 : 
        v=f"{'s' if len(diff_shape)>1 else ''}"
        mess = ("Shapes for all data must be consistent; got " 
                f"at the position{v} {smart_format(diff_shape)}.")
        
        if error =='raise': 
            raise ValueError (mess + f" Expects {occshapes}")

        warnings.warn(mess + f"The most frequent shape is {occshapes}"
                      " Please check or reverify your data. This might lead to"
                      " breaking code or invalid results. Use at your own risk."
                      )
        shape1 = list(map (lambda k:k[1],  countshapes))
        
        if set (shape1) !=1 : 
            raise ValueError ("Shape along axis 1 must be consistent. "
                              f"Got {smart_format (countshapes)}. Check the "
                              f"data at position{v} {smart_format(diff_shape)} "
                ) 
            
    colsset = set ( list(itertools.chain (*cols ) ) ) 
 
    if len(colsset ) != len(col_init) : 
        raise DatasetError ("Expect identical columns for all data"
                            " Please check your data.") 
    
    return valid_dfs 

@deprecated ("Format is no longer used, replaced by"
             " `_AquiferGroup._format` instead.")        
def _format_groups ( dic , /, name = 'Label'): 
    """ Represent the aquifer group and true labels preponderance """
    ag=["{:7}".format("Label{} (".format("s" if len(dic)>1 else ''))]
    for k, (label, repr_val ) in enumerate ( dic.items() ): 
        prep , g  = repr_val 
        ag += ["{0:^3}: {1:>10} -> {2:>7}{3:>3}".format (
            label if k==0 else "{:>10}".format(label),
            'importance', round(prep *100, 3) , "%") ]
        
        ag +=["{:^3}[ ( 'Aquifer group':\n".format("=")]
        ag+=["{:>50}:{:>15},\n".format( k, round(v, 3)) 
             for k, v in g.items() ]
        
        ag+='{:>40}'.format(")],\n ") 
        
    ag+=["{:>7}".format(")")]
    
    return print(''.join (ag) ) 

def _name_mxs_labels(*s , sep ='', prefix =""): 
    """ Name the Mixture Strategy labels from a list of labels and 
    similarity group 
    
    Parameters 
    -----------
    s: list 
        List of of pair (label, similarity ) 
    Returns
    --------
    mxs: list, 
        combined similarity names. 
        
    Example
    --------
    >>> from watex.utils.hydroutils import _name_mxs_labels 
    >>> _name_mxs_labels ( (1, 2) , (2, 4 ), (3, 7 )) 
    ... [12, 24, 37]
    >>> _name_mxs_labels ( (1, 2) , (2, 4 ), (3, 7 ), prefix ='k') 
    ... ['k12', 'k24', 'k37']
    >>> _name_mxs_labels((1, 'groupI'), (2, 'groupII'), sep='_', prefix='k')
    ... ['k1_groupI', 'k1_groupII']
    
    """
    for o in s : 
        if not is_iterable(o):
            raise ValueError (
                "Wrong value. Expect a pair values (label, similar group)"
                 " got: {o}")
        if len(o) !=2 :
            raise ValueError ("Expect a pair values (label, similar group_)."
                              " not {o}")
    mxs =list() 
    for o in s : 
        xs = str(prefix) + str(o[0]) + str(sep) + str(o[1])
        try : 
            xs = int (xs )
        except : 
            pass 
        finally: mxs.append (xs )
    return mxs 
 
def _MXS_if_no(context,  /,  y_true , y_pred , cmsg ='', trailer = "*"): 
    """ Make MXS according to the context whether a similarity 
     between the true labels in 'y_true' and NGA labels is found or not. 
     
    :param y_true: array-like 1d 
        array_like containing the true labels 
    :param y_pred: array_like 1d 
        array of the NGA predicted labels. 
    :param context: str , {'no similarity', }
    :param csmg:str, 
        formatage message is wrong context is passed in the wrong function. 

    :returns: 
        - y_mxs: array-like 1d , MXS new labels created 
        - group_classes_: dict, the labels in NGA labels and their 
            possible renamed values. Most of the case, this happens  
            when the the groups values are given as interger classes rather 
            than string. 
        - group_labels: The similar labels found at the same time in 
            'y_true' and NGA labels. 
        - sim_groups: groups  of pair composed of the similar label and 
            and the label in the predicted NGA. 
            
    :example: 
        >>> import numpy as np 
        >>> from watex.utils.hydroutils import _MXS_if_no
        >>> y_true = np.arange (5) 
        >>> y_pred = np.arange (1, 6) 
        >>> _, d, *_= _MXS_if_no ('no', y_true =y_true , y_pred =y_pred )
        >>> d 
        ... {1: '1*', 2: '2*', 3: '3*', 4: '4*', 5: '5'}
        >>> _, d, *_= _MXS_if_no ('no', y_true =y_true , y_pred =y_pred, 
                                  trailer =None)
        >>> d
        ... {1: 5, 2: 6, 3: 7, 4: 8, 5: 9} # rename labels 
        
    """
    assert str(context).lower() in {'no', 'no similarity', 
        'similarity does not exist', 'False','similarity not found'
        }, cmsg.format (_MXS_if_yes.__name__, 'at least ONE')
    
    # similarity groups in pair (true label , similar group )     
    sim_groups =None 
    group_labels =None # NGA similar groups 
    y_mxs = y_pred.copy().astype ( object )

    # get the label from similarity groups: 
    true_labels = np.unique (y_true ) 
    #  group_labels = [ group  for _, group in s ]
    NGA_labels = np.unique ( y_pred ) 
    # Rename the NGA labels using the trailer or 
    # add constant; 
    group_classes_ = dict() 
    if any([ l in true_labels for l in NGA_labels ]): 
        pseudo_NGA_labels = _create_mxs_pseudo_labels (
            y_true=y_true , y_pred=y_pred , group_labels= None, 
            trailer =trailer)
        for klabel in NGA_labels : 
            nklabel = pseudo_NGA_labels.get(klabel) 
            klabel_ix,  = np.where (y_pred ==klabel)
            y_mxs [klabel_ix ] = nklabel
            # keep it into the modified group classes 
            group_classes_ [klabel] = nklabel 

    return y_mxs , group_classes_ , group_labels , sim_groups 

def _create_mxs_pseudo_labels(
        y_true, y_pred, group_labels = None , trailer ='*'): 
    """ create pseudo MXS labels  and save it in pseudo-dict. 
    
    if labels not in the group is found in the class labels of the 'y_true', 
    rename it using the MXS trailer '*' as a special class label. 
    otherwise skipped. 
    
    If the group label is not found in the class labels of the 'y_true', it 
    does not need to rename it. Keep it intact , however because, the dtype has 
    change to string, the class label should no longer be an integer. 
    
    :param y_true: array-like 1d , 
        array of the class label in 'y_true'  
    :param y_pred: array-like 1d, 
        array of the predicted class (Mixture array) that contains 
        the NGA labels. 
    :param group_labels: list, 
        list of the label from 'y_pred' that similarity has been found in 
        the 'y_true'. For this reason, since its similarities have a special 
        class label nomenclatures, it will be discraded from the 'y_pred' i.e 
        the predicted NGA labels. Thus only the NGA labels except the  
        `group_labels` are used for renaming.
    :param trailer: str, default='*'
        The Mixture strategy marker to differentiate the existing class label  
        in 'y_true' with the predicted labels 'y_pred' especially when  
        the the same class labels are also present the true label with the 
        same label-identifier name. This usefull  to avoid any confusion  for
        both labels  in `y_true` and `y_pred` for better demarcation and 
        distinction. Note that if the `trailer`is set to ``None`` and both 
        `y_true` and `y_pred` are numeric data, the labels in `y_pred` are 
        systematically renamed to be distinct with the ones in the 'y_true'. 
        
    :returns: 
        pseudo_dict: dict, 
            dictionnary composed of the NGA labels that are not in `group_labels`
            and whose their labels have been renamed. 
    :example: 
        >>> from watex.utils.hydroutils import _create_mxs_pseudo_labels 
        >>> import numpy as np 
        >>> y_true = np.arange (5) 
        >>> y_pred = np.arange (1, 6) 
        >>> group_labels =[2, 3] # only 2 and 3 that have similarity 
        >>> _create_mxs_pseudo_labels (y_true, y_pred, group_labels )
        ... {1: '1*', 4: '4*', 5: '5*'}
        >>> # create a pseudo MXS labels when  group is None
        >>> _create_mxs_pseudo_labels (y_true, y_pred, None )
        ... {'1': '1*', '2': '2*', '3': '3*', '4': '4*', '5': '5'}
        >>> # *comments 
            # the above results demarcated the label in y_pred that 
            # exist in y_true using the default trailer '*'
        >>> #  because the bith y_true and y_pred are numeric , let set 
        >>> # the trailer to None 
        >>> _create_mxs_pseudo_labels (y_true, y_pred, None , trailer = None)
        ... {1: 5, 2: 6, 3: 7, 4: 8, 5: 9}
        >>> # * comments: 
            # Gives the differents map changes . Thus label 1 in y_pred 
            # become label 5, label 2 become label 6 and so on. 
            # this is performed to avoid confusing the label in y_true 
            # where 1, 2, 3, 4 are also presents. 
        >>> # let create a map where y_true and y_pred are different and 
        >>> # not numeric values 
        >>> y_true_no = np.array (['k1', 'k2', 'k3']) 
        >>> y_pred_no = np.array(['c1', 'c2', 'c3'])
        >>> _create_mxs_pseudo_labels (y_true_no, y_pred_no, None )
        
    """ 
    group_labels = group_labels or  []
    if not hasattr (group_labels, '__len__'): 
        raise ValueError ("Group label can't be None and must be an iterable"
                           f" object. Got: {type(group_labels).__name__!r}"
                           )
    if not (_is_arraylike_1d(y_pred ) and _is_arraylike_1d(y_true)): 
        raise TypeError ("'y' expects to be an array-like 1d ") 
        
    _check_consistency_size(y_true, y_pred) 
    
    true_labels_orig = np.unique (y_true) 
    NGA_labels = np.unique (y_pred)
    pseudo_dict = {} 
    # compute the labels not 
    # in the group 
    labels_not_in_goups = is_in_if(NGA_labels, group_labels , 
                           return_diff= True)
    if labels_not_in_goups is None:
        return  pseudo_dict 
    
    pseudo_labels = np.array(labels_not_in_goups) 
    
    # check whether both data are given as numeric data
    # so the numeric label can be rename by topping the max value 
    # got from the true_labels to the predicted label 
    # provided that trailer is None.
    is_numeric = False 
    if (_is_numeric_dtype(true_labels_orig) 
        and _is_numeric_dtype(labels_not_in_goups, to_array=True)
        ): is_numeric = True 
    
    # manage trailer 
    trailer = None if trailer in ('', None) else str(trailer) 
    if trailer is None:
        # -> improve the warning message 
        nlabs= is_in_if(NGA_labels, true_labels_orig, 
                               return_intersect=True)
        warn_msg = (
            "Note that {0} label{1} in 'y_pred' {2} also availabe in "
            "'y_true' with the same label-identifier and are not renamed."
            )
         
        warn_msg = warn_msg.format (
            len(nlabs), "s" if len(nlabs) > 1 else '', "are" if len(
                nlabs)>1 else 'is') if nlabs is not None else ""
        
        if len(group_labels) ==0: 
            if not is_numeric: 
                msg = ("Trailer is empty while one or both y_true and the"
                        " predicted 'y_pred' arrays are not a numeric data."
                        " {} This might lead to unexpected results by confusing"
                        " the predicted labels in 'y_pred' with the true"
                        " labels in 'y_pred'. Use at your own risk."
                        )

                if nlabs: warnings.warn(msg.format(warn_msg))
                trailer =''
            if is_numeric and trailer is None: 
                pseudo_labels = _mixture_num_label_if_0_in (
                    true_labels_orig, labels_not_in_goups )
            
        elif len(group_labels)!=0 : 
            warnings.warn(
                "Be aware! the trailer is empty. You may probably confuse"
                " the true labels in 'y_true' to the predicted labels."
                " This will create unexpected results when both arrays labels"
                " are confused. {} In pratice, this behavior is not tolerable."
                " Be sure, you know what you are doing. Use at your own risk."
                          )
            warnings.warn(msg.format(warn_msg))
            trailer ='' 
            
    if trailer is not None:
        pseudo_labels = list(pseudo_labels) 
        # [0 , 2 , 3 ]
    if not is_numeric or trailer is not None: 
        # Put the true labels origin into a list of string 
        # to perform element wise comparison  
        for k , items in enumerate (labels_not_in_goups): 
            if items in list(true_labels_orig): 
                pseudo_labels[k] = str(items) + trailer
            else:  pseudo_labels[k] = items
        # Numpy format the string labels 
        pseudo_labels = np.array(pseudo_labels ) 
        
    pseudo_dict = dict(zip (labels_not_in_goups, pseudo_labels )) 

    return  pseudo_dict 

def _mixture_num_label_if_0_in (true_labels, labels_to_rename) :
    """ Isolated part of _create_mxs_pseudo_labels """
    new_labels = np.array (labels_to_rename ) 
    if 0 in labels_to_rename: 
        new_labels += max(true_labels) + 1 # skip the 0 
        # true_labels =[0 , 1, 2]
        # NGA_labels =[ 0, 1, 2 ] 
        # both 
        # NGA_labels = 2+1 + NGA_labels = [3, 4, 5]
        # 0 in true_labels only i.e NGA labels [1, 2]
        # NGA lavels = 2 + [1, 2]-> [3, 4] != true_labels 
        # 0 n NGA labels only 
    else: 
        # true_labels =[1, 2]
        # NGA_labels =[0, 1, 2 ] 
        # NGA_labels = 2 + NGA_labels = [2, 3, 4]
        new_labels += max(true_labels)
    # reconvert to integer 
    return  new_labels.astype (np.int32 ) 
                 
def _MXS_if_yes (context , /, slg , y_pred, y_true,  sep=None,  prefix= None, 
                 cmsg='' , trailer = "*" ): 
    """ Make MXS target when similarity is found between a label in 'y_true' and 
    label in the predicted NGA. 

    :param y_pred: array_like 1d 
        array of the NGA predicted labels. 
    :param context: str , {'similarity exists'}
    :param csmg:str, 
        formatage message is wrong context is passed in the wrong function. 

    :returns: 
        - y_mxs: array-like 1d , MXS new labels created 
        - group_classes_: dict, the labels in NGA labels and their 
            possible renamed values. Most of the case, this happens  
            when the the groups values are given as interger classes rather 
            than string. 
        - group_labels: The similar labels found at the same time in 
            'y_true' and NGA labels. 
        - sim_groups: groups  of pair composed of the similar label and 
            and the label in the predicted NGA. 

    """
    assert str(context).lower() in {
        'similarity exists', 'yes', 'True', 'similarity is found'}, \
        cmsg.format (_MXS_if_no.__name__, 'NO')
        
    if not is_iterable(slg): 
        raise TypeError ("similarity group must be an iterable."
                         " Got: {type(s).__name__!r}")
 
    sim_groups = _name_mxs_labels(*slg, sep = sep, prefix =prefix )
    true_labels , group_labels = zip (*slg )
    if not _is_numeric_dtype(y_pred): 
        tempy = to_dtype_str(y_pred, return_values = True )
    else : tempy = y_pred.copy()

    if not all ([ l in np.unique (tempy) for l in group_labels ]): 
        # list the invalid groups 
        # not in the NGA labels 
        msg = listing_items_format(group_labels, 
                             "Invalid similar groups",  
                             "Group must be the labels in the predicted NGA.",
                             verbose = False , inline =True ,
                             )
        raise AquiferGroupError (msg)
    
    y_mxs = np.full (y_pred.shape , fill_value= np.nan , dtype = object )

    # Get the index of each NGA labels
    NGA_label_indices = { 
        label: np.where (y_pred == label )[0] for label in np.unique (y_pred )
        }
    # create a dict of pseudolabels not in group_labels  
    pseudo_NGA_labels = _create_mxs_pseudo_labels (
        y_true, y_pred, group_labels, trailer =trailer )
    group_classes_ = dict() 
    for klabel , vindex in NGA_label_indices.items () :
        if klabel in  group_labels : # [ 4, 4, 2 ]
            # --------------------------------------------------------
            # we can simply get h from indices, however it there is the 
            # same k duplicate in groups labels, index will always be 
            # fetched from first occurence, which seems heuristic  
            elt_index =  group_labels.index (klabel )  
            nklabel = sim_groups [elt_index ] 
            # print(klabel, nklabel)
            y_mxs [ vindex ] = nklabel
            group_classes_ [klabel] = nklabel
            # # --------------------------------------------------
        elif klabel not in group_labels : 
            nklabel = pseudo_NGA_labels.get(klabel) 
            y_mxs [ vindex ] = nklabel 
            group_classes_ [klabel] = nklabel 
    
    return y_mxs , group_classes_ , group_labels , sim_groups 

@deprecated("Function is henceforth deprecated. Note use anymore in"
            " MXS strategy implementation. It has been replaced by"
            " :func:`~._mixture_num_label_if_0_in` more stable."
            " It should be removed soon in a future realease. ")
def _mixture_group_label_if ( label_k, t_labels): 
    """ Start counting remaining labels from the maximum value of 
    label found in the 't_labels' """
    # Use the max element in the true labels 
    # and append it to the remain labels whose 
    # are not found as similarity groups  
    # this is possible if the simpilary group are numery datatype 
    # However if if string , keep it in the datasets 
    # The goal of this is to not be confuse with the existing
    #  true labels with the valid k labels found in the y_true
    
    # find the group label which exists in the t_labels and 
    # create pseudo group 
    # labels_in = 
    if _is_numeric_dtype(t_labels , to_array=True) :
        max_in_t_labels = max (t_labels )  
    try : 
        label_k = int (label_k) 
    except : # where k is not a numeric 
        # if label_k in t_labels: 
        pass 
    else : 
        label_k += max_in_t_labels  
        
    return label_k 

def _kmapping (arr, /): 
    """ Check whether the true labels 'y_true' have numeric dtypes 
    otherwise, create a integer labels to  substitute 
    the true labels. For instance: 
        
        >>> ['k1', 'k2', 'k3'] - > [1, 2, 3]
    :param arr: array-like 1d 
        array of onedimensional 
    """
    ytransf =arr.copy() 
    classes = None 
    if not _is_numeric_dtype(arr , to_array =True) : 
        if not _is_arraylike_1d(arr): 
            raise ValueError ("Array must be one-dimensional,"
                              " got shape: '{np.array(arr).shape}'")
            
        unik_labels = np.unique (arr)
        new_labels = np.arange(1, len(unik_labels)+ 1 )  
        for tlab, nlab  in zip (unik_labels, new_labels ) : 
            indices, = np.where (arr ==tlab)
            ytransf[indices ] = nlab 
        classes = dict ( zip ( new_labels, unik_labels ) ) 
    # try to convert to int32 
    try : ytransf = ytransf.astype (np.int32 )
    except: pass 
    return ytransf, classes 