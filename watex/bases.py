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
from .utils.funcutils import (
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

#XXX TODO: 
class WATer (ABCMeta): 
    """ Should be a SuperClass for methods classes. 
    
    Instanciate the class shoud raise an error. It should initialize arguments 
    as well for |ERP| and for |VES|. The `Water` should set the 
    attributes and check whether attributes  are suitable for  what the 
    specific class expects to. """
    
    @abstractmethod 
    def __init__(self, *args, **kwargs): 
        pass 


def fetch_model(
        modelfile: str,
        modelpath: str = None,
        default: bool = True,
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

    









        