# -*- coding: utf-8 -*-
#       Author: Kouadio K.Laurent<etanoyau@gmail.con>
#       Create:on Fri Sep 10 15:37:59 2021
#       Licence: MIT

import os 
import warnings
from abc import (
    ABCMeta, 
    abstractmethod
    )   
import pickle 
import joblib
from pprint import pprint  
import numpy as np
import pandas as pd

from ._property import (
    P ,
    )
from ._typing import (
    T, 
    List, 
    Tuple,
    Union, 
    Callable , 
    Optional,
    Array,
    DType,
    NDArray, 
    Series, 
    DataFrame, 
    )
from .utils.func_utils import (
    savepath_,
    smart_format, 
    _assert_all_types,
)
from .utils.decorator import writef 
from .exceptions import (
    HeaderError, 
    FileHandlingError, 
    ResistivityError
)
from .utils._watexlog import watexlog
 

__logger = watexlog().get_watex_logger(__name__)

# TODO: 
class WATer (ABCMeta): 
    """ Should be a SuperClass for methods classes. 
    
    Instanciate the class shoud raise an error. It should initialize arguments 
    as well for |ERP| and for |VES|. The `Water` should set the 
    attributes and check whether attributes  are suitable for  what the 
    specific class expects to. """
    
    @abstractmethod 
    def __init__(self, *args, **kwargs): 
        pass 



def _assert_data (data :DataFrame  ): 
    """ Assert  the data and return the property dataframe """
    
    data = _assert_all_types(
        data, list, tuple, np.ndarray, pd.Series, pd.DataFrame) 
    
    if isinstance(data, pd.DataFrame): 
        
        cold , ixc =list(), list()
        for i , ckey in enumerate(data.columns): 
            for kp in P().isenr : 
                if ckey.lower() .find(kp) >=0 : 
                    cold.append (kp); ixc.append(i)
                    break 
                    
        if len (cold) ==0: 
            raise ValueError ('Expected smart_format(P().isenr) '
                ' columns, but not found in the given dataframe.'
                )
                
        dup = cold.copy() 
        # filter and remove one by one duplicate columns.
        list(filter (lambda x: dup.remove(x), set(cold)))
        dup = set(dup)
        if len(dup) !=0 :
            raise HeaderError(
                f'Duplicate column{"s" if len(dup)>1 else ""}'
                f' {smart_format(dup)} found. It seems to be {smart_format(dup)}'
                f'column{"s" if len(dup)>1 else ""}. Please provide'
                '  the right column name in the dataset.'
                )
        data_ = data [cold] 
  
        col = list(data_.columns)
        for i, vc in enumerate (col): 
            for k in P().isenr : 
                if vc.lower().find(k) >=0 : 
                    col[i] = k ; break 
                
    return data_
 

def _is_erp_series (
        data : Series ,
        dipolelength : Optional [float] = None 
        ) -> DataFrame : 
    """ Validate the series.  
    
    `data` should be the resistivity values with the one of the following 
    property index names ``resistivity`` or ``rho``. Will raises error 
    if not detected. If a`dipolelength` is given, a data should include 
    each station positions values. 
    
    Parameters 
    -----------
    
    data : pandas Series object 
        Object of resistivity values 
    
    dipolelength: float
        Distance of dipole during the whole survey line. If it is
        is not given , the station location should be computed and
        filled using the default value of the dipole. The *default* 
         value is set to ``10 meters``. 
        
    Return 
    --------
    
    A dataframe of the property indexes such as
    ['station', 'easting','northing', 'resistivity'] 
    
    Raises 
    ------ 
    Error if name does not match the `resistivity` column name. 
    
    Examples 
    --------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> data = pd.Series (np.abs (np.random.rand (42)), name ='res') 
    >>> data = _is_erp_series (data)
    >>> data.columns 
    ... Index(['station', 'easting', 'northing', 'resistivity'], dtype='object')
    >>> data = pd.Series (np.abs (np.random.rand (42)), name ='NAN') 
    >>> data = _is_erp_series (data)
    ... ResistivityError: Unable to detect the resistivity column: 'NAN'.
    
    """
    
    data = _assert_all_types(data, pd.Series) 
    is_valid = False 
    for p in P().iresistivity : 
        if data.name.lower().find(p) >=0 :
            data.name = p ; is_valid = True ; break 
    
    if not is_valid : 
        raise ResistivityError(
            f"Unable to detect the resistivity column: {data.name!r}."
            )
    
    if is_valid: 
        df = _is_erp_dataframe  (pd.DataFrame (
            {
                data.name : data , 
                'NAN' : np.zeros_like(data ) 
                }
            ),
                dipolelength = dipolelength,
            )
    return df 

    
    
def _is_erp_dataframe (
        data :DataFrame ,
        dipolelength : Optional[float] = None 
        ) -> DataFrame:
    """ Ckeck whether the dataframe contains the electrical resistivity 
    profiling (ERP) index properties. 
    
    DataFrame should be reordered to fit the order of index properties. 
    Anyway it should he dataframe filled by ``0.`` where the property is
    missing. However if `station` property is not given. station` property 
    should be set by using the dipolelength default value equals to ``10.``.
    
    Parameters 
    ----------
    
    data : Dataframe object 
        Dataframe object. The columns dataframe should match the property 
        ERP property object such as: 
            ['station','easting','northing','resistivity' ]
            
    dipolelength: float
        Distance of dipole during the whole survey line. If the station 
        is not given as  `data` columns, the station location should be 
        computed and filled the station columns using the default value 
        of the dipole. The *default* value is set to ``10 meters``. 
        
    Returns
    --------
    A new data with index properties.
        
    Raises 
    ------
    
    - None of the columns does not match the property indexes.  
    - Find duplicated values in the given data header.
    
    Examples
    --------
    >>> import numpy as np 
    >>> from watex.bases import _is_erp_dataframe 
    >>> df = pd.read_csv ('data/erp/testunsafedata.csv')
    >>> df.columns 
    ... Index(['x', 'stations', 'resapprho', 'NORTH'], dtype='object')
    >>> df = _is_erp_dataframe (df) 
    >>> df.columns 
    ... Index(['station', 'easting', 'northing', 'resistivity'], dtype='object')
    
    """
    data = _assert_all_types(data, pd.DataFrame)
    datac= data.copy() 
    
    def _is_in_properties (h ):
        """ check whether the item header `h` is in the property values. 
        Return `h` and it correspondence `key` in the property values. """
        for key, values in P().idicttags.items() : 
            for v in values : 
                if h.lower().find (v)>=0 :
                    return h, key 
        return None, None 
    
    def _check_correspondence (pl, dl): 
        """ collect the duplicated name in the data columns """
        return [ l for l in pl for d  in dl if d.lower().find(l)>=0 ]
        
    cold , c = list(), list()
    for i , ckey in enumerate(list(datac.columns)): 
        h , k = _is_in_properties(ckey)
        cold.append (h) if h is not None  else h 
        c.append(k) if k is not None else k
        
    if len (cold) ==0: 
        raise HeaderError (
            'Unable to find the expected smart_format(P().isenr) '
            ' properties in the data columns `{list(data.columns)}`'
            )

    dup = cold.copy() 
    # filter and remove one by one duplicate columns.
    list(filter (lambda x: dup.remove(x), set(cold)))
    dup = set(dup) ; ress = _check_correspondence(P().isenr, dup)
    if len(dup) !=0 :
        raise HeaderError(
            f'Duplicate column{"s" if len(dup)>1 else ""}' 
            f' {smart_format(dup)} {"are" if len(dup)>1 else "is"} '
            f'found. It seems correspond to {smart_format(ress)}. '
            'Please ckeck your data column names. '
            )
            
    # fetch the property column names and 
    # replace by 0. the non existence column
    # reorder the column to match 
    # ['station','easting','northing','resistivity' ]
    data_ = data[cold] 
    data_.columns = c  
    data_= data_.reindex (columns =P().isenr, fill_value =0.) 
    dipolelength = _assert_all_types(
        dipolelength , float, int) if dipolelength is not None else None 
    
    if (np.all (data_.station) ==0. 
        and dipolelength is None 
        ): 
        dipolelength = 10.
        data_.station = np.arange (
            0 , data_.shape[0] * dipolelength  , dipolelength ) 
        
    return data_


def fetch_model(
        modelfile: str,
        modelpath: str = None,
        default: bool= True,
        modname: Optional[str] = None,
        verbose: int = 0
                ): 
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
        - `pickledmodel`: model dumped and all parameters if default is `False`
        
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
       __logger.info(f"Loading models `{os.path.basename(modelfile)}`!")
       try : 
           pickledmodel = joblib.load(modelfile)
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
               pickledmodel= pickle.load (modf)
           __logger.info(f"Model `{os.path.basename(modelfile)!r} deserialized"
                         "  using Python pickle module.`!")
           
           dmsg=f"Model {modelfile!r} deserizaled from  {modelfile!r}!"
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
                model_class_params  =( pickledmodel[modname]['best_model'], 
                                   pickledmodel[modname]['best_params_'], 
                                   pickledmodel[modname]['best_scores'],
                                   )
            if not default: 
                model_class_params=pickledmodel[modname]
                
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
            model_class_params.append((pickledmodel[mm]['best_model'], 
                                      pickledmodel[mm]['best_params_'],
                                      pickledmodel[modname]['best_scores']))
    
        if verbose > 0: 
               pprint('Should return a list of tuple pairs:`best model`and '
                      ' `model best parameters.')
               
        return model_class_params

    return pickledmodel


   
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

    









        