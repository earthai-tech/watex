# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>


"""
Hydrogeological utilities 
================================
Hydrogeological parameters of aquifer are the essential and crucial basic data 
in the designing and construction progress of geotechnical engineering and 
groundwater dewatering, which are directly related to the reliability of these 
parameters.

"""
from __future__ import annotations 
import os 
import inspect
import warnings 
import numpy as np
import pandas as pd  
from ..decorators import catmapflow2, writef  
from ..exceptions import FileHandlingError 
from  ..typing import (
    List, 
    Tuple, 
    Optional, 
    Union, T,
    Series, 
    DataFrame, 
    ArrayLike, 
    F
    ) 

#XXXTODO compute t parameters 
def transmissibility (s, d, time, ): 
    """Transmissibility T represents the ability of aquifer's water conductivity.
    
    It is the numeric equivalent of the product of hydraulic conductivity times
    aquifer's thickness (T = KM), which means it is the seepage flow under the
    condition of unit hydraulic gradient, unit time, and unit width
    
    """
      
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
            
    :classes: list of classes names to replace the default `FR` that is 
        used to specify the flow rate. For instance, it can be:: 
            
            [0, 1, 2] => [sf0, sf1, sf2]
       
    """
    msg= ("Objective is 'flow' whereas the target value is set to {0}."
          " Target is defaultly encoded to hold integers {1}. If"
          " the auto-categorization does not fit the real values"
          " of flow ranges, please set the range of the real flow values"
          " via param `values` or `label_values`."
          ) 
    if values is None:
        msg = ("Missing values for categorizing 'y'; the number of"
                " occurence in the target is henceforth not allowed."
                )
        warnings.warn("Values are not set. The new version does not" 
                      " tolerate the number of occurrence to be used."
                      "Provided the list of flow values instead.",
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
) -> Tuple [DataFrame, Union[str], bool ]: 
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



def savepath_ (nameOfPath): 
    """
    Shortcut to create a folder 
    :param nameOfPath: Path name to save file
    :type nameOfPath: str 
    
    :return: 
        New folder created. If the `nameOfPath` exists, will return ``None``
    :rtype:str 
        
    """
 
    try :
        savepath = os.path.join(os.getcwd(), nameOfPath)
        if not os.path.isdir(savepath):
            os.mkdir(nameOfPath)#  mode =0o666)
    except :
        warnings.warn("The path seems to be existed !")
        return
    return savepath 
         
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
def _assert_all_types (
        obj: object , 
        *expected_objtype: type 
 ) -> object: 
    """ Quick assertion of object type. Raise an `TypeError` if 
    wrong type is given."""
    # if np.issubdtype(a1.dtype, np.integer): 
    if not isinstance( obj, expected_objtype): 
        raise TypeError (
            f'Expected {smart_format(tuple (o.__name__ for o in expected_objtype))}'
            f' type{"s" if len(expected_objtype)>1 else ""} '
            f'but `{type(obj).__name__}` is given.')
            
    return obj 

def smart_format(iter_obj, choice ='and'): 
    """ Smart format iterable ob.
    
    :param iter_obj: iterable obj 
    :param choice: can be 'and' or 'or' for optional.
    
    :Example: 
        >>> from watex.utils.funcutils import smart_format
        >>> smart_format(['model', 'iter', 'mesh', 'data'])
        ... 'model','iter','mesh' and 'data'
    """
    str_litteral =''
    try: 
        iter(iter_obj) 
    except:  return f"{iter_obj}"
    
    iter_obj = [str(obj) for obj in iter_obj]
    if len(iter_obj) ==1: 
        str_litteral= ','.join([f"{i!r}" for i in iter_obj ])
    elif len(iter_obj)>1: 
        str_litteral = ','.join([f"{i!r}" for i in iter_obj[:-1]])
        str_litteral += f" {choice} {iter_obj[-1]!r}"
    return str_litteral

def is_iterable (y, /)->bool: 
    """ Asserts iterable object and returns 'True' or 'False' """
    return hasattr (y, '__iter__') 

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