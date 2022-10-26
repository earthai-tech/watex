# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations 

import os 
import re 
import sys
import csv 
import copy  
import json
import h5py
import yaml
import shutil
import numbers 
import inspect  
import warnings
import itertools
import subprocess 
from six.moves import urllib 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from .._watexlog import watexlog
from ..typing import ( 
    Tuple,
    Dict,
    Optional,
    Iterable,
    Any,
    ArrayLike,
    F,
    T,
    List ,
    DataFrame, 
    Sub,
    NDArray, 
    )
from ..property import P
from ..exceptions import ( 
    EDIError,
    ParameterNumberError, 
    )
from ._dependency import import_optional_dependency
_logger = watexlog.get_watex_logger(__name__)

_msg= ''.join([
    'Note: need scipy version 0.14.0 or higher or interpolation,',
    ' might not work.']
)
_msg0 = ''.join([
    'Could not find scipy.interpolate, cannot use method interpolate'
     'check installation you can get scipy from scipy.org.']
)

try:
    import scipy
    scipy_version = [int(ss) for ss in scipy.__version__.split('.')]
    if scipy_version[0] == 0:
        if scipy_version[1] < 14:
            warnings.warn(_msg, ImportWarning)
            _logger.warning(_msg)
            
    import scipy.interpolate as spi

    interp_import = True
 # pragma: no cover
except ImportError: 
    
    warnings.warn(_msg0)
    _logger.warning(_msg0)
    
    interp_import = False
    
#-----


def to_numeric_dtypes (
        arr: NDArray | DataFrame, *, columns:List[str] = None, 
        return_feature_types:bool =False , missing_values:float = np.nan, 
    )-> DataFrame : 
    """ Convert array to dataframe and coerce arguments to appropriate dtypes. 
    
    Parameters 
    -----------
    arr: Ndarray or Dataframe, shape (M=samples, N=features)
        Array of dataframe to create 
    columns: list of str, optional 
        Usefull to create a dataframe when array is given. Be aware to fit the 
        number of array columns (shape[1])
    return_feature_types: bool, default=False, 
        return the list of numerical and categorial features
    missing_values: float: 
        Replace the missing or empty string if exist in the dataframe.
    Returns 
    --------
    df or (df, nf, cf): Dataframe of values casted to numeric types 
        also return `nf` and `cf`  if `return_feature_types` is set
        to``True``.
    
    Examples
    ---------
    >>> from watex.datasets.dload import load_bagoue
    >>> from watex.utils.funcutils import to_numeric_dtypes
    >>> X, y = load_bagoue (as_frame =True ) 
    >>> X0 =X[['shape', 'power', 'magnitude']]
    >>> X0.dtypes 
    ... shape        object
        power        object
        magnitude    object
        dtype: object
    >>> df = to_numeric_dtypes(X0)
    >>> df.dtypes 
    ... shape         object
        power        float64
        magnitude    float64
        dtype: object
        
    """
    if isinstance (arr, np.ndarray) and columns is None: 
        warnings.warn("Array is given while columns is not supplied.")
    # reconvert data to frame 
    df = pd.DataFrame (arr, columns =columns  
                       ) if isinstance (arr, np.ndarray) else arr 
    
    nf,cf =[], []
    #replace empty string by Nan if NaN exist in dataframe  
    df= df.replace(r'^\s*$', missing_values, regex=True)
    
    # check the possibililty to cast all 
    # the numerical data 
    for serie in df.columns: 
        try: 
            df= df.astype(
                {serie:np.float64})
            nf.append(serie)
        except:
            cf.append(serie)
            continue
        
    return (df, nf, cf) if return_feature_types else df 

            
def parse_attrs (attr, /, regex=None ): 
    """ Parse attributes using the regular expression.
    
    Remove all string non-alphanumeric and some operator indicators,  and 
    fetch attributes names. 
    
    Parameters 
    -----------
    
    attr: str, text litteral containing the attributes 
        names 
        
    regex: `re` object, default is 
        Regular expresion object. the default is:: 
            
            >>> import re 
            >>> re.compile (r'per|mod|mul|add|sub|[_#&*@!_,;\s-]\s*', 
                                flags=re.IGNORECASE) 
    Returns
    -------
    attr: List of attributes 
    
    Examples
    ---------
    >>> from watex.utils.funcutils import parse_attrs 
    >>> parse_attrs('lwi_sub_ohmSmulmagnitude')
    ... ['lwi', 'ohmS', 'magnitude']
    
    
    """
    regex = regex or re.compile (r'per|mod|mul|add|sub|[_#&*@!_,;\s-]\s*', 
                        flags=re.IGNORECASE) 
    attr= list(filter (None, regex.split(attr)))
    return attr 
    
def url_checker (url: str , install:bool = False, 
                      raises:str ='ignore')-> bool : 
    """
    check whether the URL is reachable or not. 
    
    function uses the requests library. If not install, set the `install`  
    parameter to ``True`` to subprocess install it. 
    
    Parameters 
    ------------
    url: str, 
        link to the url for checker whether it is reachable 
    install: bool, 
        Action to install the 'requests' module if module is not install yet.
    raises: str 
        raise errors when url is not recheable rather than returning ``0``.
        if ``raises` is ``ignore``, and module 'requests' is not installed, it 
        will use the django url validator. However, the latter only assert 
        whether url is right but not validate its reachability. 
              
    Returns
    --------
        ``True``{1} for reacheable and ``False``{0} otherwise. 
        
    Example
    ----------
    >>> from watex.utils.funcutils import url_checker 
    >>> url_checker ("http://www.example.com")
    ...  0 # not reacheable 
    >>> url_checker ("https://watex.readthedocs.io/en/latest/api/watex.html")
    ... 1 
    
    """
    isr =0 ; success = False 
    
    regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        #domain...
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )
    
    try : 
        import requests 
    except ImportError: 
        if install: 
            success  = is_installing('requests', DEVNULL=True) 
        if not success: 
            if raises=='raises': 
                raise ModuleNotFoundError(
                    "auto-installation of 'requests' failed."
                    " Install it mannually.")
                
    else : success=True  
    
    if success: 
        try:
            get = requests.get(url) #Get Url
            if get.status_code == 200: # if the request succeeds 
                isr =1 # (f"{url}: is reachable")
                
            else:
                warnings.warn(
                    f"{url}: is not reachable, status_code: {get.status_code}")
                isr =0 
        
        except requests.exceptions.RequestException as e:
            if raises=='raises': 
                raise SystemExit(f"{url}: is not reachable \nErr: {e}")
            else: isr =0 
            
    if not success : 
        # use django url validation regex
        # https://github.com/django/django/blob/stable/1.3.x/django/core/validators.py#L45
        isr = 1 if re.match(regex, url) is not None else 0 
        
    return isr 

    
def shrunkformat (text: str | Iterable[Any] , 
                  chunksize: int =7 , insert_at: str = None, 
                  sep =None, 
                 ) : 
    """ format class and add elipsis when classe are greater than maxview 
    
    :param text: str - a text to shrunk and format. Can also be an iterable
        object. 
    :param chunksize: int, the size limit to keep in the formatage text. *default* 
        is ``7``.
    :param insert_at: str, the place to insert the ellipsis. If ``None``,  
        shrunk the text and put the ellipsis, between the text beginning and 
        the text endpoint. Can be ``beginning``, or ``end``. 
    :param sep: str if the text is delimited by a kind of character, the `sep` 
        parameters could be usefull so it would become a starting point for 
        word counting. *default*  is `None` which means word is counting from 
        the space. 
        
    :example: 
        
    >>> import numpy as np 
    >>> from watex.utils.funcutils import shrunkformat
    >>> text=" I'm a long text and I will be shrunked and replace by ellipsis."
    >>> shrunkformat (text)
    ... 'Im a long ... and replace by ellipsis.'
    >>> shrunkformat (text, insert_at ='end')
    ...'Im a long ... '
    >>> arr = np.arange(30)
    >>> shrunkformat (arr, chunksize=10 )
    ... '0 1 2 3 4  ...  25 26 27 28 29'
    >>> shrunkformat (arr, insert_at ='begin')
    ... ' ...  26 27 28 29'
    
    """
    is_str = False 
    chunksize = int (_assert_all_types(chunksize, float, int))
                   
    regex = re.compile (r"(begin|start|beg)|(end|close|last)")
    insert_at = str(insert_at).lower().strip() 
    gp = regex.search (insert_at) 
    if gp is not None: 
        if gp.group (1) is not None:  
            insert_at ='begin'
        elif gp.group(2) is not None: 
            insert_at ='end'
        if insert_at is None: 
            warnings.warn(f"Expect ['begining'|'end'], got {insert_at!r}"
                          " Default value is used instead.")
    if isinstance(text , str): 
        textsplt = text.strip().split(sep) # put text on list 
        is_str =True 
        
    elif hasattr (text , '__iter__'): 
        textsplt = list(text )
        
    if len(textsplt) < chunksize : 
        return  text 
    
    if is_str : 
        rl = textsplt [:len(textsplt)//2][: chunksize//2]
        ll= textsplt [len(textsplt)//2:][-chunksize//2:]
        
        if sep is None: sep =' '
        spllst = [f'{sep}'.join ( rl), f'{sep}'.join ( ll)]
        
    else : spllst = [
        textsplt[: chunksize//2 ] ,textsplt[-chunksize//2:]
        ]
    if insert_at =='begin': 
        spllst.insert(0, ' ... ') ; spllst.pop(1)
    elif insert_at =='end': 
        spllst.pop(-1) ; spllst.extend ([' ... '])
        
    else : 
        spllst.insert (1, ' ... ')
    
    spllst = spllst if is_str else str(spllst)
    
    return re.sub(r"[\[,'\]]", '', ''.join(spllst), 
                  flags=re.IGNORECASE 
                  ) 
    

def is_installing (
        module: str , 
        upgrade: bool=True , 
        action: bool=True, 
        DEVNULL: bool=False,
        verbose: int=0,
        **subpkws
    )-> bool: 
    """ Install or uninstall a module/package using the subprocess 
    under the hood.
    
    Parameters 
    ------------
    module: str,
        the module or library name to install using Python Index Package `PIP`
    
    upgrade: bool,
        install the lastest version of the package. *default* is ``True``.   
        
    DEVNULL:bool, 
        decline the stdoutput the message in the console 
    
    action: str,bool 
        Action to perform. 'install' or 'uninstall' a package. *default* is 
        ``True`` which means 'intall'. 
        
    verbose: int, Optional
        Control the verbosity i.e output a message. High level 
        means more messages. *default* is ``0``.
         
    subpkws: dict, 
        additional subprocess keywords arguments 
    Returns 
    ---------
    success: bool 
        whether the package is sucessfully installed or not. 
        
    Example
    --------
    >>> from watex import is_installing
    >>> is_installing(
        'tqdm', action ='install', DEVNULL=True, verbose =1)
    >>> is_installing(
        'tqdm', action ='uninstall', verbose =1)
    """
    #implement pip as subprocess 
    # refer to https://pythongeeks.org/subprocess-in-python/
    if not action: 
        if verbose > 0 :
            print("---> No action `install`or `uninstall`"
                  f" of the module {module!r} performed.")
        return action  # DO NOTHING 
    
    success=False 

    action_msg ='uninstallation' if action =='uninstall' else 'installation' 

    if action in ('install', 'uninstall', True) and verbose > 0:
        print(f'---> Module {module!r} {action_msg} will take a while,'
              ' please be patient...')
        
    cmdg =f'<pip install {module}> | <python -m pip install {module}>'\
        if action in (True, 'install') else ''.join([
            f'<pip uninstall {module} -y> or <pip3 uninstall {module} -y ',
            f'or <python -m pip uninstall {module} -y>.'])
        
    upgrade ='--upgrade' if upgrade else '' 
    
    if action == 'uninstall':
        upgrade= '-y' # Don't ask for confirmation of uninstall deletions.
    elif action in ('install', True):
        action = 'install'

    cmd = ['-m', 'pip', f'{action}', f'{module}', f'{upgrade}']

    try: 
        STDOUT = subprocess.DEVNULL if DEVNULL else None 
        STDERR= subprocess.STDOUT if DEVNULL else None 
    
        subprocess.check_call(
            [sys.executable] + cmd, stdout= STDOUT, stderr=STDERR,
                              **subpkws)
        if action in (True, 'install'):
            # freeze the dependancies
            reqs = subprocess.check_output(
                [sys.executable,'-m', 'pip','freeze'])
            [r.decode().split('==')[0] for r in reqs.split()]

        success=True
        
    except: 

        if verbose > 0 : 
            print(f'---> Module {module!r} {action_msg} failed. Please use'
                f' the following command: {cmdg} to manually do it.')
    else : 
        if verbose > 0: 
            print(f"{action_msg.capitalize()} of `{module}` "
                      "and dependancies was successfully done!") 
        
    return success 

def smart_strobj_recognition(
        name: str  ,
        container: List | Tuple | Dict[Any, Any ],
        stripitems: str | List | Tuple = '_', 
        deep: bool = False,  
) -> str : 
    """ Find the likelihood word in the whole containers and 
    returns the value.
    
    :param name: str - Value of to search. I can not match the exact word in 
    the `container`
    :param container: list, tuple, dict- container of the many string words. 
    :param stripitems: str - 'str' items values to sanitize the  content 
        element of the dummy containers. if different items are provided, they 
        can be separated by ``:``, ``,`` and ``;``. The items separators 
        aforementioned can not  be used as a component in the `name`. For 
        isntance:: 
            
            name= 'dipole_'; stripitems='_' -> means remove the '_'
            under the ``dipole_``
            name= '+dipole__'; stripitems ='+;__'-> means remove the '+' and
            '__' under the value `name`. 
        
    :param deep: bool - Kind of research. Go deeper by looping each items 
         for find the initials that can fit the name. Note that, if given, 
         the first occurence should be consider as the best name... 
         
    :return: Likelihood object from `container`  or Nonetype if none object is
        detected.
        
    :Example:
        >>> from watex.utils.funcutils import smart_strobj_recognition
        >>> from watex.methods import ResistivityProfiling 
        >>> rObj = ResistivityProfiling(AB= 200, MN= 20,)
        >>> smart_strobj_recognition ('dip', robj.__dict__))
        ... None 
        >>> smart_strobj_recognition ('dipole_', robj.__dict__))
        ... dipole 
        >>> smart_strobj_recognition ('dip', robj.__dict__,deep=True )
        ... dipole 
        >>> smart_strobj_recognition (
            '+_dipole___', robj.__dict__,deep=True , stripitems ='+;_')
        ... 'dipole'
        
    """

    stripitems =_assert_all_types(stripitems , str, list, tuple) 
    container = _assert_all_types(container, list, tuple, dict)
    ix , rv = None , None 
    
    if isinstance (stripitems , str): 
        for sep in (':', ",", ";"): # when strip ='a,b,c' seperated object
            if sep in stripitems:
                stripitems = stripitems.strip().split(sep) ; break
        if isinstance(stripitems, str): 
            stripitems =[stripitems]
            
    # sanitize the name. 
    for s in stripitems :
        name = name.strip(s)     
        
    if isinstance(container, dict) : 
        #get only the key values and lower them 
        container_ = list(map (lambda x :x.lower(), container.keys())) 
    else :
        # for consistency put on list if values are in tuple. 
        container_ = list(container)
        
    # sanitize our dummny container item ... 
    #container_ = [it.strip(s) for it in container_ for s in stripitems ]
    if name.lower() in container_: 
        try:
            ix = container_.index (name)
        except ValueError: 
            raise AttributeError("{name!r} attribute is not defined")
        
    if deep and ix is None:
        # go deeper in the search... 
        for ii, n in enumerate (container_) : 
            if n.find(name.lower())>=0 : 
                ix =ii ; break 
    
    if ix is not None: 
        if isinstance(container, dict): 
            rv= list(container.keys())[ix] 
        else : rv= container[ix] 

    return  rv 

def repr_callable_obj(obj: F  , skip = None ): 
    """ Represent callable objects. 
    
    Format class, function and instances objects. 
    
    :param obj: class, func or instances
        object to format. 
    :param skip: str , 
        attribute name that is not end with '_' and whom it needs to be 
        skipped. 
        
    :Raises: TypeError - If object is not a callable or instanciated. 
    
    :Examples: 
        
    >>> from watex.utils.funcutils import repr_callable_obj
    >>> from watex.methods.electrical import  ResistivityProfiling
    >>> repr_callable_obj(ResistivityProfiling)
    ... 'ResistivityProfiling(station= None, dipole= 10.0, 
            auto_station= False, kws= None)'
    >>> robj= ResistivityProfiling (AB=200, MN=20, station ='S07')
    >>> repr_callable_obj(robj)
    ... 'ResistivityProfiling(AB= 200, MN= 20, arrangememt= schlumberger, ... ,
        dipole= 10.0, station= S07, auto= False)'
    >>> repr_callable_obj(robj.fit)
    ... 'fit(data= None, kws= None)'
    
    """
    regex = re.compile (r"[{'}]")
    
    # inspect.formatargspec(*inspect.getfullargspec(cls_or_func))
    if not hasattr (obj, '__call__') and not hasattr(obj, '__dict__'): 
        raise TypeError (
            f'Format only callabe objects: {type (obj).__name__!r}')
        
    if hasattr (obj, '__call__'): 
        cls_or_func_signature = inspect.signature(obj)
        objname = obj.__name__
        PARAMS_VALUES = {k: None if v.default is (inspect.Parameter.empty 
                         or ...) else v.default 
                    for k, v in cls_or_func_signature.parameters.items()
                    # if v.default is not inspect.Parameter.empty
                    }
    elif hasattr(obj, '__dict__'): 
        objname=obj.__class__.__name__
        PARAMS_VALUES = {k:v  for k, v in obj.__dict__.items() 
                         if not ((k.endswith('_') or k.startswith('_') 
                                  # remove the dict objects
                                  or k.endswith('_kws') or k.endswith('_props'))
                                 )
                         }
    if skip is not None : 
        # skip some inner params 
        # remove them as the main function or class params 
        if isinstance(skip, (tuple, list, np.ndarray)): 
            skip = list(map(str, skip ))
            exs = [key for key in PARAMS_VALUES.keys() if key in skip]
        else:
            skip =str(skip).strip() 
            exs = [key for key in PARAMS_VALUES.keys() if key.find(skip)>=0]
 
        for d in exs: 
            PARAMS_VALUES.pop(d, None) 
            
    # use ellipsis as internal to stdout more than seven params items 
    if len(PARAMS_VALUES) >= 7 : 
        f = {k:PARAMS_VALUES.get(k) for k in list(PARAMS_VALUES.keys())[:3]}
        e = {k:PARAMS_VALUES.get(k) for k in list(PARAMS_VALUES.keys())[-3:]}
        
        PARAMS_VALUES= str(f) + ', ... , ' + str(e )

    return str(objname) + '(' + regex.sub('', str (PARAMS_VALUES)
                                          ).replace(':', '=') +')'


def accept_types (
        *objtypes: list , 
        format: bool = False
        ) -> List[str] | str : 
    """ List the type format that can be accepted by a function. 
    
    :param objtypes: List of object types.
    :param format: bool - format the list of the name of objects.
    :return: list of object type names or str of object names. 
    
    :Example: 
        >>> import numpy as np; import pandas as pd 
        >>> from watex.utils.funcutils import accept_types
        >>> accept_types (pd.Series, pd.DataFrame, tuple, list, str)
        ... "'Series','DataFrame','tuple','list' and 'str'"
        >>> atypes= accept_types (
            pd.Series, pd.DataFrame,np.ndarray, format=True )
        ..."'Series','DataFrame' and 'ndarray'"
    """
    return smart_format(
        [f'{o.__name__}' for o in objtypes]
        ) if format else [f'{o.__name__}' for o in objtypes] 

def read_from_excelsheets(erp_file: str = None ) -> List[DataFrame]: 
    
    """ Read all Excelsheets and build a list of dataframe of all sheets.
   
    :param erp_file:
        Excell workbooks containing `erp` profile data.
        
    :return: A list composed of the name of `erp_file` at index =0 and the 
      datataframes.
      
    """
    
    allfls:Dict [str, Dict [T, List[T]] ] = pd.read_excel(
        erp_file, sheet_name=None)
    
    list_of_df =[os.path.basename(os.path.splitext(erp_file)[0])]
    for sheets , values in allfls.items(): 
        list_of_df.append(pd.DataFrame(values))

    return list_of_df 

def check_dimensionality(obj, data, z, x):
    """ Check dimensionality of data and fix it.
    
    :param obj: Object, can be a class logged or else.
    :param data: 2D grid data of ndarray (z, x) dimensions.
    :param z: array-like should be reduced along the row axis.
    :param x: arraylike should be reduced along the columns axis.
    
    """
    def reduce_shape(Xshape, x, axis_name=None): 
        """ Reduce shape to keep the same shape"""
        mess ="`{0}` shape({1}) {2} than the data shape `{0}` = ({3})."
        ox = len(x) 
        dsh = Xshape 
        if len(x) > Xshape : 
            x = x[: int (Xshape)]
            obj._logging.debug(''.join([
                f"Resize {axis_name!r}={ox!r} to {Xshape!r}.", 
                mess.format(axis_name, len(x),'more',Xshape)])) 
                                    
        elif len(x) < Xshape: 
            Xshape = len(x)
            obj._logging.debug(''.join([
                f"Resize {axis_name!r}={dsh!r} to {Xshape!r}.",
                mess.format(axis_name, len(x),'less', Xshape)]))
        return int(Xshape), x 
    
    sz0, z = reduce_shape(data.shape[0], 
                          x=z, axis_name ='Z')
    sx0, x =reduce_shape (data.shape[1],
                          x=x, axis_name ='X')
    data = data [:sz0, :sx0]
    
    return data , z, x 


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

def make_introspection(Obj: object , subObj: Sub[object])->None: 
    """ Make introspection by using the attributes of instance created to 
    populate the new classes created.
    
    :param Obj: callable 
        New object to fully inherits of `subObject` attributes.
        
    :param subObj: Callable 
        Instance created.
    """
    # make introspection and set the all  attributes to self object.
    # if Obj attribute has the same name with subObj attribute, then 
    # Obj attributes get the priority.
    for key, value in  subObj.__dict__.items(): 
        if not hasattr(Obj, key) and key  != ''.join(['__', str(key), '__']):
            setattr(Obj, key, value)
            
def cpath (savepath=None , dpath=None): 
    """ Control the existing path and create one of it does not exist.
    
    :param savepath: Pathlike obj, str 
    :param dpath: str, default pathlike obj
    
    """
    if dpath is None:
        file, _= os.path.splitext(os.path.basename(__file__))
        dpath = ''.join(['_', file,
                         '_']) #.replace('.py', '')
    if savepath is None : 
        savepath  = os.path.join(os.getcwd(), dpath)
        try:os.mkdir(savepath)
        except: pass 
    if savepath is not None:
        try :
            if not os.path.isdir(savepath):
                os.mkdir(savepath)#  mode =0o666)
        except : pass 
    return savepath   
  

def sPath (name_of_path:str):
    """ Savepath func. Create a path  with `name_of_path` if path not exists.
    
    :param name_of_path: str, Path-like object. If path does not exist,
        `name_of_path` should be created.
    """
    
    try :
        savepath = os.path.join(os.getcwd(), name_of_path)
        if not os.path.isdir(savepath):
            os.mkdir(name_of_path)#  mode =0o666)
    except :
        warnings.warn("The path seems to be existed!")
        return
    return savepath 


def format_notes(text:str , cover_str: str ='~', inline=70, **kws): 
    """ Format note 
    :param text: Text to be formated.
    
    :param cover_str: type of ``str`` to surround the text.
    
    :param inline: Nomber of character before going in liine.
    
    :param margin_space: Must be <1 and expressed in %. The empty distance 
        between the first index to the inline text 
    :Example: 
        
        >>> from watex.utils import funcutils as func 
        >>> text ='Automatic Option is set to ``True``.'\
            ' Composite estimator building is triggered.' 
        >>>  func.format_notes(text= text ,
        ...                       inline = 70, margin_space = 0.05)
    
    """
    
    headnotes =kws.pop('headernotes', 'notes')
    margin_ratio = kws.pop('margin_space', 0.2 )
    margin = int(margin_ratio * inline)
    init_=0 
    new_textList= []
    if len(text) <= (inline - margin): 
        new_textList = text 
    else : 
        for kk, char in enumerate (text): 
            if kk % (inline - margin)==0 and kk !=0: 
                new_textList.append(text[init_:kk])
                init_ =kk 
            if kk ==  len(text)-1: 
                new_textList.append(text[init_:])
  
    print('!', headnotes.upper(), ':')
    print('{}'.format(cover_str * inline)) 
    for k in new_textList:
        fmtin_str ='{'+ '0:>{}'.format(margin) +'}'
        print('{0}{1:>2}{2:<51}'.format(fmtin_str.format(cover_str), '', k))
        
    print('{0}{1:>51}'.format(' '* (margin -1), cover_str * (inline -margin+1 ))) 
    
    
def sanitize_fdataset(
    _df: DataFrame
) ->Tuple[DataFrame, int]: 
    """ Sanitize the feature dataset. 
    
    Recognize the columns provided  by the users and resset according 
    to the features labels disposals :attr:`~Features.featureLabels`."""
    
    utm_flag =0 
    
    def getandReplace(
            optionsList:List[str],
            params:List[str], df:DataFrame
            ) -> List[str]: 
        """
        Function to  get parames and replace to the main features params.
        
        :param optionsList: 
            User options to qualified the features headlines. 
        :type optionsList: list
        
        :param params: Exhaustive parameters names. 
        :type params: list 
        
        :param df: pd.DataFrame collected from `features_fn`. 
        
        :return: sanitize columns
        :rtype: list 
        """
        
        columns = [c.lower() for c in df.columns] 
        for ii, celemnt in enumerate(columns): 
            for listOption, param in zip(optionsList, params): 
                 for option in listOption:
                     if param =='lwi': 
                        if celemnt.find('eau')>=0 : 
                            columns[ii]=param 
                            break
                     if re.match(r'^{0}+'.format(option), celemnt):
                         columns[ii]=param
                         if columns[ii] =='east': 
                             utm_flag =1
                         break

        return columns, utm_flag 

    new_df_columns, utm_flag = getandReplace(
        optionsList=P.param_options,
            params=P.param_ids,
            df= _df
                                  )
    df = pd.DataFrame(data=_df.to_numpy(), columns= new_df_columns)
    return df , utm_flag
     
             
def interpol_scipy (
        x_value,
        y_value,
        x_new,
        kind="linear",
        plot=False,
        fill="extrapolate"
        ):
    
    """
    function to interpolate data 
    
    Parameters 
    ------------
    * x_value : np.ndarray 
        value on array data : original abscissA 
                
    * y_value : np.ndarray 
        value on array data : original coordinates (slope)
                
    * x_new  : np.ndarray 
        new value of absciss you want to interpolate data 
                
    * kind  : str 
        projection kind maybe : "linear", "cubic"
                
    * fill : str 
        kind of extraolation, if None , *spi will use constraint interpolation 
        can be "extrapolate" to fill_value.
        
    * plot : Boolean 
        Set to True to see a wiewer graph

    Returns 
    --------
        np.ndarray 
            y_new ,new function interplolate values .
            
    :Example: 
        
        >>> import numpy as np 
        >>>  fill="extrapolate"
        >>>  x=np.linspace(0,15,10)
        >>>  y=np.random.randn(10)
        >>>  x_=np.linspace(0,20,15)
        >>>  ss=interpol_Scipy(x_value=x, y_value=y, x_new=x_, kind="linear")
        >>>  ss
    """
    
    func_=spi.interp1d(x_value, y_value, kind=kind,fill_value=fill)
    y_new=func_(x_new)
    if plot :
        plt.plot(x_value, y_value,"o",x_new, y_new,"--")
        plt.legend(["data", "linear","cubic"],loc="best")
        plt.show()
    
    return y_new


def _remove_str_word (char, word_to_remove, deep_remove=False):
    """
    Small funnction to remove a word present on  astring character 
    whatever the number of times it will repeated.
    
    Parameters
    ----------
        * char : str
                may the the str phrases or sentences . main items.
        * word_to_remove : str
                specific word to remove.
        * deep_remove : bool, optional
                use the lower case to remove the word even the word is uppercased 
                of capitalized. The default is False.

    Returns
    -------
        str ; char , new_char without the removed word .
        
    Examples
    ---------
    >>> from watex.utils import funcutils as func
    >>> ch ='AMTAVG 7.76: "K1.fld", Dated 99-01-01,AMTAVG, 
    ...    Processed 11 Jul 17 AMTAVG'
    >>> ss=func._remove_str_word(char=ch, word_to_remove='AMTAVG', 
    ...                             deep_remove=False)
    >>> print(ss)
    
    """
    if type(char) is not str : char =str(char)
    if type(word_to_remove) is not str : word_to_remove=str(word_to_remove)
    
    if deep_remove == True :
        word_to_remove, char =word_to_remove.lower(),char.lower()

    if word_to_remove not in char :
        return char

    while word_to_remove in char : 
        if word_to_remove not in char : 
            break 
        index_wr = char.find(word_to_remove)
        remain_len=index_wr+len(word_to_remove)
        char=char[:index_wr]+char[remain_len:]

    return char

def stn_check_split_type(data_lines): 
    """
    Read data_line and check for data line the presence of 
    split_type < ',' or ' ', or any other marks.>
    Threshold is assume to be third of total data length.
    
    :params data_lines: list of data to parse . 
    :type data_lines: list 
 
    :returns: The split _type
    :rtype: str
    
    :Example: 
        >>> from watex.utils  import funcutils as func
        >>> path =  data/ K6.stn
        >>> with open (path, 'r', encoding='utf8') as f : 
        ...                     data= f.readlines()
        >>>  print(func.stn_check_split_type(data_lines=data))
        
    """

    split_type =[',', ':',' ',';' ]
    data_to_read =[]
    # change the data if data is not dtype string elements.
    if isinstance(data_lines, np.ndarray): 
        if data_lines.dtype in ['float', 'int', 'complex']: 
            data_lines=data_lines.astype('<U12')
        data_lines= data_lines.tolist()
        
    if isinstance(data_lines, list):
        for ii, item in enumerate(data_lines[:int(len(data_lines)/3)]):
             data_to_read.append(item)
             # be sure the list is str item . 
             data_to_read=[''.join([str(item) for item in data_to_read])] 

    elif isinstance(data_lines, str): data_to_read=[str(data_lines)]
    
    for jj, sep  in enumerate(split_type) :
        if data_to_read[0].find(sep) > 0 :
            if data_to_read[0].count(sep) >= 2 * len(data_lines)/3:
                if sep == ' ': return  None  # use None more conventional 
                else : return sep 

def minimum_parser_to_write_edi (edilines, parser = '='):
    """
    This fonction validates edifile for writing , string with egal.
    we assume that dictionnary in list will be for definemeasurment
    E and H fied. 
    
    :param edilines: list of item to parse 
    :type edilines: list 
    
    :param parser: the egal is use  to parser edifile .
                    can be changed, default is `=`
    :type parser: str 
  
    """
    if isinstance(edilines,list):
        if isinstance(edilines , tuple) : edilines =list(edilines)
        else :raise TypeError('<Edilines> Must be on list')
    for ii, lines in enumerate(edilines) :
        if isinstance(lines, dict):continue 
        elif lines.find('=') <0 : 
            raise EDIError (
             f'None <"="> found on this item<{edilines[ii]}> of '
            ' the edilines list. list can not be parsed. Please'
            ' put egal sign "=" between key and value '
            )
    
    return edilines 
            

def round_dipole_length(value, round_value =5.): 
    """ 
    small function to graduate dipole length 5 to 5. Goes to be reality and 
    simple computation .
    
    :param value: value of dipole length 
    :type value: float 
    
    :returns: value of dipole length rounded 5 to 5 
    :rtype: float
    """ 
    mm = value % round_value 
    if mm < 3 :return np.around(value - mm)
    elif mm >= 3 and mm < 7 :return np.around(value -mm +round_value) 
    else:return np.around(value - mm +10.)
    
def display_infos(infos, **kws):
    """ Display unique element on list of array infos
    
    :param infos: Iterable object to display. 
    :param header: Change the `header` to other names. 
    
    :Example: 
    >>> from watex.utils.funcutils import display_infos
    >>> ipts= ['river water', 'fracture zone', 'granite', 'gravel',
         'sedimentary rocks', 'massive sulphide', 'igneous rocks', 
         'gravel', 'sedimentary rocks']
    >>> display_infos('infos= ipts,header='TestAutoRocks', 
                      size =77, inline='~')
    """

    inline =kws.pop('inline', '-')
    size =kws.pop('size', 70)
    header =kws.pop('header', 'Automatic rocks')

    if isinstance(infos, str ): 
        infos =[infos]
        
    infos = list(set(infos))
    print(inline * size )
    mes= '{0}({1:02})'.format(header.capitalize(),
                                  len(infos))
    mes = '{0:^70}'.format(mes)
    print(mes)
    print(inline * size )
    am=''
    for ii in range(len(infos)): 
        if (ii+1) %2 ==0: 
            am = am + '{0:>4}.{1:<30}'.format(ii+1, infos[ii].capitalize())
            print(am)
            am=''
        else: 
            am ='{0:>4}.{1:<30}'.format(ii+1, infos[ii].capitalize())
            if ii ==len(infos)-1: 
                print(am)
    print(inline * size )

def fr_en_parser (f, delimiter =':'): 
    """ Parse the translated data file. 
    
    :param f: translation file to parse.
    
    :param delimiter: str, delimiter.
    
    :return: generator obj, composed of a list of 
        french  and english Input translation. 
    
    :Example:
        >>> file_to_parse = 'pme.parserf.md'
        >>> path_pme_data = r'C:/Users\Administrator\Desktop\__elodata
        >>> data =list(BS.fr_en_parser(
            os.path.join(path_pme_data, file_to_parse)))
    """
    
    is_file = os.path.isfile (f)
    if not is_file: 
        raise IOError(f'Input {f} is not a file. Please check your file.')
    
    with open(f, 'r', encoding ='utf8') as ft: 
        data = ft.readlines()
        for row in data :
            if row in ( '\n', ' '):
                continue 
            fr, en = row.strip().split(delimiter)
            yield([fr, en])

def convert_csvdata_from_fr_to_en(csv_fn, pf, destfile = 'pme.en.csv',
                                  savepath =None, delimiter =':'): 
    """ Translate variable data from french csv data  to english with 
    parser file. 
    
    :param csv_fn: data collected in csv format.
    
    :param pf: parser file. 
    
    :param destfile: str,  Destination file, outputfile.
    
    :param savepath: Path-Like object, save data to a path. 
                      
    :Example: 
        # to execute this script, we need to import the two modules below
        >>> import os 
        >>> import csv 
        >>> from watex.utils.funcutils import convert_csvdata_from_fr_to_en
        >>> path_pme_data = r'C:/Users\Administrator\Desktop\__elodata
        >>> datalist=convert_csvdata_from_fr_to_en(
            os.path.join( path_pme_data, _enuv2.csv') , 
            os.path.join(path_pme_data, pme.parserf.md')
                         savefile = 'pme.en.cv')
    """
    # read the parser file and separed english from french 
    parser_data = list(fr_en_parser (pf,delimiter) )
    
    with open (csv_fn, 'r', encoding ='utf8') as csv_f : 
        csv_reader = csv.reader(csv_f) 
        csv_data =[ row for row in csv_reader]
    # get the index of the last substring row 
    ix = csv_data [0].index ('Industry_type') 
    # separateblock from two 
    csv_1b = [row [:ix +1] for row in csv_data] 
    csv_2b =[row [ix+1:] for row in csv_data ]
    # make a copy of csv_1b
    csv_1bb= copy.deepcopy(csv_1b)
   
    for ii, rowline in enumerate( csv_1bb[3:]) : # skip the first two rows 
        for jj , row in enumerate(rowline): 
            for (fr_v, en_v) in  parser_data: 
                # remove the space from french parser part
                # this could reduce the mistyping error 
                fr_v= fr_v.replace(
                    ' ', '').replace('(', '').replace(
                        ')', '').replace('\\', '').lower()
                 # go  for reading the half of the sentence
                row = row.lower().replace(
                    ' ', '').replace('(', '').replace(
                        ')', '').replace('\\', '')
                if row.find(fr_v[: int(len(fr_v)/2)]) >=0: 
                    csv_1bb[3:][ii][jj] = en_v 
    
    # once translation is done, concatenate list 
    new_csv_list = [r1 + r2 for r1, r2 in zip(csv_1bb,csv_2b )]
    # now write the new scv file 
    if destfile is None: 
        destfile = f'{os.path.basename(csv_fn)}_to.en'
        
    destfile.replace('.csv', '')
    
    with open(f'{destfile}.csv', 'w', newline ='',encoding ='utf8') as csvf: 
        csv_writer = csv.writer(csvf, delimiter=',')
        csv_writer.writerows(new_csv_list)
        # for row in  new_csv_list: 
        #     csv_writer.writerow(row)
    savepath = cpath(savepath , '__pme')
    try :
        shutil.move (f'{destfile}.csv', savepath)
    except:pass 
    
    return new_csv_list
    
def parse_md_data (pf , delimiter =':'): 
    
    if not os.path.isfile (pf): 
        raise IOError( " Unable to detect the parser file. "
                      "Need a Path-like object ")
    
    with open(pf, 'r', encoding ='utf8') as f: 
        pdata = f.readlines () 
    for row in pdata : 
        if row in ('\n', ' '): 
            continue 
        fr, en = row.strip().split(delimiter)
        fr = sanitize_unicode_string(fr)
        en = en.strip()
        # if capilize, the "I" inside the 
        #text should be in lowercase 
        # it is better to upper the first 
        # character after striping the whole 
        # string
        en = list(en)
        en[0] = en[0].upper() 
        en = "".join(en)

        yield fr, en 
        
def sanitize_unicode_string (str_) : 
    """ Replace all spaces and remove all french accents characters.
    
    :Example:
    >>> from watex.utils.funcutils import sanitize_unicode_string 
    >>> sentence ='Nos clients sont extrêmement satisfaits '
        'de la qualité du service fourni. En outre Nos clients '
            'rachètent frequemment nos "services".'
    >>> sanitize_unicode_string  (sentence)
    ... 'nosclientssontextrmementsatisfaitsdelaqualitduservice'
        'fournienoutrenosclientsrachtentfrequemmentnosservices'
    """
    sp_re = re.compile (r"[.'()-\\/’]")
    e_re = re.compile(r'[éèê]')
    a_re= re.compile(r'[àâ]')

    str_= re.sub('\s+', '', str_.strip().lower())
    
    for cobj , repl  in zip ( (sp_re, e_re, a_re), 
                             ("", 'e', 'a')): 
        str_ = cobj.sub(repl, str_)
    
    return str_             
                  
def read_main (csv_fn , pf , delimiter =':',
               destfile ='pme.en.csv') : 
    
    parser_data = list(parse_md_data(pf, delimiter) )
    parser_dict =dict(parser_data)
    
    with open (csv_fn, 'r', encoding ='utf8') as csv_f : 
        csv_reader = csv.reader(csv_f) 
        csv_data =[ row for row in csv_reader]
        
    # get the index of the last substring row 
    # and separate block into two from "Industry_type"
    ix = csv_data [0].index ('Industry_type') 
    
    csv_1b = [row [:ix +1] for row in csv_data] 
    csv_2b =[row [ix+1:] for row in csv_data ]
    # make a copy of csv_1b
    csv_1bb= copy.deepcopy(csv_1b)
    copyd = copy.deepcopy(csv_1bb); is_missing =list()
    
    # skip the first two rows 
    for ii, rowline in enumerate( csv_1bb[3:]) : 
        for jj , row in enumerate(rowline):
            row = row.strip()
            row = sanitize_unicode_string(row )
            csv_1bb[3:][ii][jj] = row 
            
    #collect the missing values 
    for ii, rowline in enumerate( csv_1bb[3:]) : 
        for jj , row in enumerate(rowline): 
            if row not in parser_dict.keys():
                is_missing.append(copyd[3:][ii][jj])
    is_missing = list(set(is_missing))       
    
    # merge the prior two blocks and build the dataframe
    new_csv_list = [r1 + r2 for r1, r2 in zip(csv_1bb, csv_2b )]
    df = pd.DataFrame (
        np.array(new_csv_list [1:]),
        columns =new_csv_list [0] 
                       )
    for key, value in parser_dict.items(): 
        # perform operation in place and return None 
        df.replace (key, value, inplace =True )
    

    df.to_csv (destfile)
    return  df , is_missing 
    

def _isin (
        arr: ArrayLike | List [float] ,
        subarr: Sub [ArrayLike] |Sub[List[float]] | float, 
        return_mask:bool=False, 
) -> bool : 
    """ Check whether the subset array `subcz` is in  `cz` array. 
    
    :param arr: Array-like - Array of item elements 
    :param subarr: Array-like, float - Subset array containing a subset items.
    :param return_mask: bool, return the mask where the element is in `arr`.  
    :return: True if items in  test array `subarr` are in array `arr`. 
    
    """
    arr = np.array (arr );  subarr = np.array(subarr )

    return (True if True in np.isin (arr, subarr) else False
            ) if not return_mask else np.isin (arr, subarr) 

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
     

def drawn_boundaries(erp_data, appRes, index):
    """
    Function to drawn anomaly boundary 
    and return the anomaly with its boundaries
    
    :param erp_data: erp profile 
    :type erp_data: array_like or list 
    
    :param appRes: resistivity value of minimum pk anomaly 
    :type appRes: float 
    
    :param index: index of minimum pk anomaly 
    :type index: int 
    
    :return: anomaly boundary 
    :rtype: list of array_like 

    """
    f = 0 # flag to mention which part must be calculated 
    if index ==0 : 
        f = 1 # compute only right part 
    elif appRes ==erp_data[-1]: 
        f=2 # compute left part 
    
    def loop_sideBound(term):
        """
        loop side bar from anomaly and find the term side 
        
        :param term: is array of left or right side of anomaly.
        :type term: array 
        
        :return: side bar 
        :type: array_like 
        """
        tem_drawn =[]
        maxT=0 

        for ii, tem_rho in enumerate(term) : 

            diffRes_betw_2pts= tem_rho - appRes 
            if diffRes_betw_2pts > maxT : 
                maxT = diffRes_betw_2pts
                tem_drawn.append(tem_rho)
            elif diffRes_betw_2pts < maxT : 
                # rho_limit = tem_rho 
                break 
        return np.array(tem_drawn)
    # first broke erp profile from the anomalies 
    if f ==0 or f==2 : 
        left_term = erp_data[:index][::-1] # flip left term  for looping
        # flip again to keep the order 
        left_limit = loop_sideBound(term=left_term)[::-1] 

    if f==0 or f ==1 : 
        right_term= erp_data[index :]
        right_limit=loop_sideBound(right_term)
    # concat right and left to get the complete anomaly 
    if f==2: 
        anomalyBounds = np.append(left_limit,appRes)
                                   
    elif f ==1 : 
        anomalyBounds = np.array([appRes]+ right_limit.tolist())
    else: 
        left_limit = np.append(left_limit, appRes)
        anomalyBounds = np.concatenate((left_limit, right_limit))
    
    return appRes, index, anomalyBounds 


       
def find_position_from_sa(
        an_res_range, 
        pos=None,
        selectedPk=None): 
    """
    Function to select the main `pk` from both :func:`get_boundaries`.
    
    :paran an_res_range: anomaly resistivity range on |ERP| line. 
    :type an_res_range: array_like 
    
    :param pos: position of anomaly boundaries (inf and sup):
                anBounds = [90, 130]
                - 130 is max boundary and 90 the  min boundary 
    :type pos: list 
    
    :param selectedPk: 
        
        User can set its own position of the right anomaly. Be sure that 
        the value provided is right position . 
        Could not compute again provided that `pos`
        is not `None`.
                
    :return: anomaly station position. 
    :rtype: str 'pk{position}'
    
    :Example:
        
        >>> from watex.utils.funcutils import find_positon_from_sa
        >>> resan = np.array([168,130, 93,146,145])
        >>> pk= find_pk_from_selectedAn(
        ...    resan, pos=[90, 13], selectedPk= 'str20')
        >>> pk
    
    .. |ERP| replace:: Electrical Resistivity Profiling
    
    """
    #compute dipole length from pos
    if pos is not None : 
        if isinstance(pos, list): 
            pos =np.array(pos)
    if pos is None and selectedPk is None : 
        raise ParameterNumberError(
            'Give at least the anomaly boundaries'
            ' before computing the selected anomaly position.')
        
    if selectedPk is not None :  # mean is given
        if isinstance(selectedPk, str):
            if selectedPk.isdigit() :
                sPk= int(selectedPk)
            elif selectedPk.isalnum(): 
                oss = ''.join([s for s in selectedPk
                    if s.isdigit()])
                sPk =int(oss)
        else : 
            try : 
                sPk = int(selectedPk)
                
            except : pass 

        if pos is not None : # then compare the sPk and ps value
            try :
                if not pos.min()<= sPk<=pos.max(): 
                    warnings.warn('Wrong position given <{}>.'
                                  ' Should compute new positions.'.
                                  format(selectedPk))
                    _logger.debug('Wrong position given <{}>.'
                                  'Should compute new positions.'.
                                  format(selectedPk))
                    
            except UnboundLocalError:
                print("local variable 'sPk' referenced before assignment")
            else : 
                
                return 'pk{}'.format(sPk ), an_res_range
                
            
        else : 
            selectedPk='pk{}'.format(sPk )
            
            return selectedPk , an_res_range
    

    if isinstance(pos, list):
        pos =np.array(pos)  
    if isinstance(an_res_range, list): 
        an_res_range =np.array(an_res_range)
    dipole_length = (pos.max()-pos.min())/(len(an_res_range)-1)

    tem_loc = np.arange(
        pos.min(), pos.max()+dipole_length, dipole_length)
    
    # find min value of  collected anomalies values 
    locmin = np.where (an_res_range==an_res_range.min())[0]
    if len(locmin) >1 : locmin =locmin[0]
    pk_= int(tem_loc[int(locmin)]) # find the min pk 

    selectedPk='pk{}'.format(pk_)
    
    return selectedPk , an_res_range
 
def fmt_text(
        anFeatures=None, 
        title = None,
        **kwargs) :
    """
    Function format text from anomaly features 
    
    :param anFeatures: Anomaly features 
    :type anFeatures: list or dict
    
    :param title: head lines 
    :type title: list
    
    :Example: 
        
        >>> from watex.utils.funcutils import fmt_text
        >>> fmt_text(anFeatures =[1,130, 93,(146,145, 125)])
    
    """
    if title is None: 
        title = ['Ranking', 'rho(Ω.m)', 'position pk(m)', 'rho range(Ω.m)']
    inline =kwargs.pop('inline', '-')
    mlabel =kwargs.pop('mlabels', 100)
    line = inline * int(mlabel)
    
    #--------------------header ----------------------------------------
    print(line)
    tem_head ='|'.join(['{:^15}'.format(i) for i in title[:-1]])
    tem_head +='|{:^45}'.format(title[-1])
    print(tem_head)
    print(line)
    #-----------------------end header----------------------------------
    newF =[]
    if isinstance(anFeatures, dict):
        for keys, items in anFeatures.items(): 
            rrpos=keys.replace('_pk', '')
            rank=rrpos[0]
            pos =rrpos[1:]
            newF.append([rank, min(items), pos, items])
            
    elif isinstance(anFeatures, list): 
        newF =[anFeatures]
    
    
    for anFeatures in newF: 
        strfeatures ='|'.join(['{:^15}'.format(str(i)) \
                               for i in anFeatures[:-1]])
        try : 
            iter(anFeatures[-1])
        except : 
            strfeatures +='|{:^45}'.format(str(anFeatures[-1]))
        else : 
            strfeatures += '|{:^45}'.format(
                ''.join(['{} '.format(str(i)) for i in anFeatures[-1]]))
            
        print(strfeatures)
        print(line)
    
    
def find_feature_positions (
        anom_infos,
        anom_rank,
        pks_rhoa_index,
        dl): 
    """
    Get the pk bound from ranking of computed best points
    
    :param anom_infos:
        
        Is a dictionnary of best anomaly points computed from 
        :func:`drawn_anomaly_boundaries2` when `pk_bounds` is not given.  
        see :func:`find_position_bounds`
        
    :param anom_rank: Automatic ranking after selecting best points 
        
    :param pk_rhoa_index: 
        
        Is tuple of selected anomaly resistivity value and index in the whole
        |ERP| line. for instance: 
            
            pks_rhoa_index= (80., 17) 
            
        where "80" is the value of selected anomaly in ohm.m and "17" is the 
        index of selected points in the |ERP| array. 
        
    :param dl: 
        
        Is the distance between two measurement as `dipole_length`. Provide 
        the `dl` if the *default* value is not right. 
        
    :returns: 
        
        Refer to :doc:`.exmath.select_anomaly`
    
    .. |ERP| replace:: Electrical Resistivity Profiling
    
    """     
    rank_code = '{}_pk'.format(anom_rank)
    for key in anom_infos.keys(): 
        if rank_code in key: 
            pk = float(key.replace(rank_code, ''))

            rhoa = list(pks_rhoa_index[anom_rank-1])[0]
            codec = key
            break 
         
    ind_rhoa =np.where(anom_infos[codec] ==rhoa)[0]
    if len(ind_rhoa) ==0 : ind_rhoa =0 
    leninf = len(anom_infos[codec][: int(ind_rhoa)])
    
    pk_min = pk - leninf * dl 
    lensup =len(anom_infos[codec][ int(ind_rhoa):])
    pk_max =  pk + (lensup -1) * dl 
    
    pos_bounds = (pk_min, pk_max)
    rhoa_bounds = (anom_infos[codec][0], anom_infos[codec][-1])
    
    return pk, rhoa, pos_bounds, rhoa_bounds, anom_infos[codec]


def find_position_bounds( 
        pk,
        rhoa,
        rhoa_range,
        dl=10.
        ):
    """
    Find station position boundary indexed in |ERP| line. Usefull 
    to get the boundaries indexes `pk_boun_indexes` for |ERP| 
    normalisation  when computing `anr` or else. 
    
    .. |ERP| replace:: Electrical Resistivity Profiling
    
    :param pk: Selected anomaly station value 
    :type pk: float 
    
    :param rhoa: Selected anomaly value in ohm.m 
    :type rhoa: float 
    
    :rhoa_range: Selected anomaly values from `pk_min` to `pk_max` 
    :rhoa_range: array_like 
    
    :parm dl: see :func:`find_position_from_sa` docstring.
    
    :Example: 
        
        >>> from watex.utils.funcutils import find_position_bounds  
        >>> find_position_bounds(pk=110, rhoa=137, 
                          rhoa_range=np.array([175,132,137,139,170]))
        
    
    """

    if isinstance(pk, str): 
        pk = float(pk.replace(pk[0], '').replace('_pk', ''))
        
    index_rhoa = np.where(rhoa_range ==rhoa)[0]
    if len(index_rhoa) ==0 : index_rhoa =0 
    
    leftlen = len(rhoa_range[: int(index_rhoa)])
    rightlen = len(rhoa_range[int(index_rhoa):])
    
    pk_min = pk - leftlen * dl 
    pk_max =  pk + (rightlen  -1) * dl 
    
    return pk_min, pk_max 


def wrap_infos (
        phrase ,
        value ='',
        underline ='-',
        unit ='',
        site_number= '',
        **kws) : 
    """Display info from anomaly details."""
    
    repeat =kws.pop('repeat', 77)
    intermediate =kws.pop('inter+', '')
    begin_phrase_mark= kws.pop('begin_phrase', '--|>')
    on = kws.pop('on', False)
    if not on: return ''
    else : 
        print(underline * repeat)
        print('{0} {1:<50}'.format(begin_phrase_mark, phrase), 
              '{0:<10} {1}'.format(value, unit), 
              '{0}'.format(intermediate), "{}".format(site_number))
        print(underline * repeat )
    
def drawn_anomaly_boundaries2(
        erp_data,
        appRes, 
        index
        ):
    """
    Function to drawn anomaly boundary 
    and return the anomaly with its boundaries
    
    :param erp_data: erp profile 
    :type erp_data: array_like or list 
    
    :param appRes: resistivity value of minimum pk anomaly 
    :type appRes: float 
    
    :param index: index of minimum pk anomaly 
    :type index: int 
    
    :return: anomaly boundary 
    :rtype: list of array_like 

    """
    f = 0 # flag to mention which part must be calculated 
    if index ==0 : 
        f = 1 # compute only right part 
    elif appRes ==erp_data[-1]: 
        f=2 # compute left part 
    
    def loop_sideBound(term):
        """
        loop side bar from anomaly and find the term side 
        
        :param term: is array of left or right side of anomaly.
        :type trem: array 
        
        :return: side bar 
        :type: array_like 
        """
        tem_drawn =[]
        maxT=0 

        for ii, tem_rho in enumerate(term) : 

            diffRes_betw_2pts= tem_rho - appRes 
            if diffRes_betw_2pts > maxT : 
                maxT = diffRes_betw_2pts
                tem_drawn.append(tem_rho)
            elif diffRes_betw_2pts < maxT : 
                # rho_limit = tem_rho 
                break 
        # print(tem_drawn)
        return np.array(tem_drawn)
    # first broke erp profile from the anomalies 
    if f==2 : # compute the left part 
        # flip array and start backward counting 
        temp_erp_data = erp_data [::-1] 
        sbeg = appRes   # initialize value 
        for ii, valan in enumerate(temp_erp_data): 
            if valan >= sbeg: 
                sbeg = valan 
            elif valan < sbeg: 
                left_term = erp_data[ii:]
                break 
            
        left_term = erp_data[:index][::-1] # flip left term  for looping
        # flip again to keep the order 
        left_limit = loop_sideBound(term=left_term)[::-1] 

    if f==0 or f ==1 : 
        right_term= erp_data[index :]
        right_limit=loop_sideBound(right_term)
    # concat right and left to get the complete anomaly 
    if f==2: 
        anomalyBounds = np.append(left_limit,appRes)
                                   
    elif f ==1 : 
        anomalyBounds = np.array([[appRes]+ right_limit.tolist()])
    else: 
        left_limit = np.append(left_limit, appRes)
        anomalyBounds = np.concatenate((left_limit, right_limit))
    
    return appRes, index, anomalyBounds  

def get_boundaries(df): 
    """
    Define anomaly boundary `upper bound` and `lowerbound` from 
    :ref:`define_position_bounds` location. 
        
    :param df: Dataframe pandas contained  the columns 
                'pk', 'x', 'y', 'rho', 'dl'. 
    returns: 
        - `autoOption` triggered the automatic Option if nothing is specified 
            into excelsheet.
        - `ves_loc`: Sounding curve location at pk 
        - `posMinMax`: Anomaly boundaries composed of ``lower`` and ``upper``
            bounds.
           Specific names can be used  to define lower and upper bounds:: 
                
               `lower`: 'lower', 'inf', 'min', 'min', '1' or  'low'
               `upper`: 'upper', 'sup', 'maj', 'max', '2, or 'up'
               
        To define the sounding location, can use:: 
            `ves`:'ves', 'se', 'sond','vs', 'loc', '0' or 'dl'
            
    """
    shape_=[ 'V','W', 'U', 'H', 'M', 'C', 'K' ]
    type__= ['EC', 'NC', 'CP', 'CB2P']
        # - ``EC`` for Extensive conductive. 
        # - ``NC`` for narrow conductive. 
        # - ``CP`` for conductive PLANE 
        # - ``CB2P`` for contact between two planes. 
    shape =None 
    type_ =None 
    
    def recoverShapeOrTypefromSheet(
            listOfAddedArray,
            param): 
        """ Loop the array and get whether an anomaly shape name is provided.
        
        :param listOfAddedArray: all Added array values except 
         'pk', 'x', 'y', 'rho' are composed of list of addedArray.
        :param param: Can be main description of different `shape_` of `type__` 
        
        :returns: 
            - `shape` : 'V','W', 'U', 'H', 'M', 'C' or  'K' from sheet or 
             `type` : 'EC', 'NC', 'CP', 'CB2P'
            - listOfAddedArray : list of added array 
        """
        param_ =None 
        for jj, colarray in enumerate(listOfAddedArray[::-1]): 
            tem_=[str(ss).upper().strip() for ss in list(colarray)] 
            for ix , elem in enumerate(tem_):
                for param_elm in param: 
                    if elem ==param_elm : 
                        # retrieves the shape and replace by np.nan value 
                        listOfAddedArray[::-1][jj][ix]=np.nan  
                        return param_elm , listOfAddedArray
                    
        return param_, listOfAddedArray
    
    def mergeToOne(
            listOfColumns,
            _df
            ):
        """ Get data from other columns annd merge into one array.
        
        :param listOfColumns: Columns names 
        :param _df: dataframe to retrieve data to one
        """
        new_array = np.full((_df.shape[0],), np.nan)
        listOfColumnData = [ _df[name].to_numpy() for name in listOfColumns ]
        # loop from backward so we keep the most important to the first row 
        # close the main df that composed `pk`,`x`, `y`, and `rho`.
        # find the shape 
        shape, listOfColumnData = recoverShapeOrTypefromSheet(listOfColumnData, 
                                                        param =shape_)
        type_, listOfColumnData = recoverShapeOrTypefromSheet(listOfColumnData, 
                                                        param =type__)
  
        for colarray in listOfColumnData[::-1]: 
           for ix , val  in enumerate(colarray): 
               try: 
                   if not np.isnan(val) : 
                       new_array[ix]=val
               except :pass  
        
        return shape , type_,  new_array 
    
    
    def retrieve_ix_val(
            array
            ): 
        """ Retrieve value and index  and build `posMinMax boundaries
        
        :param array: array of main colum contains the anomaly definitions or 
                a souding curve location like :: 
                
                sloc = [NaN, 'low', NaN, NaN, NaN, 'ves', NaN,
                        NaN, 'up', NaN, NaN, NaN]
                `low`, `ves` and `up` are the lower boundary, the electric 
                sounding  and the upper boundary of the selected anomaly 
                respectively.
        For instance, if dipole_length is =`10`m, t he location (`pk`)
            of `low`, `ves` and `up` are 10, 50 and 80 m respectively.
            `posMinMax` =(10, 80)
        """
        
        lower_ix =None 
        upper_ix =None
        ves_ix = None 

        array= array.reshape((array.shape[0],) )
        for ix, val in enumerate(array):
            for low, up, vloc in zip(
                    ['lower', 'inf', 'min', 'min', '1', 'low'],
                    ['upper', 'sup', 'maj', 'max', '2', 'up'], 
                    ['ves', 'se', 'sond','vs', 'loc', '0', 'dl']
                    ): 
                try : 
                    floatNaNor123= np.float(val)
                except: 
                    if val.lower().find(low)>=0: 
                        lower_ix = ix 
                        break
                    elif val.lower().find(up) >=0: 
                        upper_ix = ix 
                        break
                    elif val.lower().find(vloc)>=0: 
                        ves_ix = ix 
                        break 
                else : 
                    if floatNaNor123 ==1: 
                        lower_ix = ix 
                        break
                    elif floatNaNor123 ==2: 
                        upper_ix = ix 
                        break 
                    elif floatNaNor123 ==0: 
                        ves_ix = ix 
                        break 
                           
        return lower_ix, ves_ix, upper_ix 
    
    # set pandas so to consider np.inf as NaN number.
    
    pd.options.mode.use_inf_as_na = True
    
    # unecesseray to specify the colum of sounding location.
    # dl =['drill', 'dl', 'loc', 'dh', 'choi']
    
    _autoOption=False  # set automatic to False one posMinMax  
    # not found as well asthe anomaly location `ves`.
    posMinMax =None 
    #get df columns from the 4-iem index 
    for sl in ['pk', 'sta', 'loc']: 
        for val in df.columns: 
            if val.lower()==sl: 
                pk_series = df[val].to_numpy()
                break 

    listOfAddedColumns= df.iloc[:, 4:].columns
    
    if len(listOfAddedColumns) ==0:
        return True,  shape, type_, None, posMinMax,  df   
 
    df_= df.iloc[:, 4:]
    # check whether all remains dataframe values are `NaN` values
    if len(list(df_.columns[df_.isna().all()])) == len(listOfAddedColumns):
         # If yes , trigger the auto option 
        return True,  shape, type_, None, posMinMax,  df.iloc[:, :4] 
  
    # get the colum name with any nan values 
    sloc_column=list(df_.columns[df_.isna().any()])
    # if column contains  one np.nan, the sloc colum is found 
    sloc_values = df_[sloc_column].to_numpy()
    
    if len(sloc_column)>1 : #
    # get the value from single array
        shape , type_,  sloc_values =  mergeToOne(sloc_column, df_)


    lower_ix, ves_ix ,upper_ix   = retrieve_ix_val(sloc_values)
    
    # if `lower` and `upper` bounds are not found then start or end limits of
    # selected anomaly  from the position(pk) of the sounding curve. 
    if lower_ix is None : 
        lower_ix =ves_ix 
    if upper_ix is None: 
        upper_ix = ves_ix 
   
    if (lower_ix  and upper_ix ) is None: 
        posMinMax =None
    if posMinMax is None and ves_ix is None: _autoOption =True 
    else : 
        posMinMax =(pk_series[lower_ix], pk_series[upper_ix] )
        
    if ves_ix is None: ves_loc=None
    else : ves_loc = pk_series[ves_ix]
    
    return _autoOption, shape, type_, ves_loc , posMinMax,  df.iloc[:, :4]  
    
 
def reshape(arr , axis = None) :
    """ Detect the array shape and reshape it accordingly, back to the given axis. 
    
    :param array: array_like with number of dimension equals to 1 or 2 
    :param axis: axis to reshape back array. If 'axis' is None and 
        the number of dimension is greater than 1, it reshapes back array 
        to array-like 
    
    :returns: New reshaped array 
    
    :Example: 
        >>> import numpy as np 
        >>> from watex.utils.funcutils import reshape 
        >>> array = np.random.randn(50 )
        >>> array.shape
        ... (50,)
        >>> ar1 = reshape(array, 1) 
        >>> ar1.shape 
        ... (1, 50)
        >>> ar2 =reshape(ar1 , 0) 
        >>> ar2.shape 
        ... (50, 1)
        >>> ar3 = reshape(ar2, axis = None)
        >>> ar3.shape # goes back to the original array  
        >>> ar3.shape 
        ... (50,)
        
    """
    arr = np.array(arr)
    if arr.ndim > 2 : 
        raise ValueError('Expect an array with max dimension equals to 2' 
                         f' but {str(arr.ndim)!r} were given.')
        
    if axis  not in (0 , 1, -1, None): 
        raise ValueError(f'Wrong axis value: {str(axis)!r}')
        
    if axis ==-1:
        axis =None 
    if arr.ndim ==1 : 
        # ie , axis is None , array is an array-like object
        s0, s1= arr.shape [0], None 
    else : 
        s0, s1 = arr.shape 
    if s1 is None: 
        return  arr.reshape ((1, s0)) if axis == 1 else (arr.reshape (
            (s0, 1)) if axis ==0 else arr )
    try : 
        arr = arr.reshape ((s0 if s1==1 else s1, )) if axis is None else (
            arr.reshape ((1, s0)) if axis==1  else arr.reshape ((s1, 1 ))
            )
    except ValueError: 
        # error raises when user mistakes to input the right axis. 
        # (ValueError: cannot reshape array of size 54 into shape (1,1)) 
        # then return to him the original array 
        pass 

    return arr   
    
    
def ismissing(refarr, arr, fill_value = np.nan, return_index =False): 
    """ Get the missing values in array-like and fill it  to match the length
    of the reference array. 
    
    The function makes sense especially for frequency interpollation in the 
    'attenuation band' when using the audio-frequency magnetotelluric methods. 
    
    :param arr: array-like- Array to extended with fill value. It should be  
        shorter than the `refarr`. Otherwise it returns the same array `arr` 
    :param refarr: array-like- the reference array. It should have a greater 
        length than the array 
    :param fill_value: float - Value to fill the `arr` to match the length of 
        the `refarr`. 
    :param return_index: bool or str - array-like, index of the elements element 
        in `arr`. Default is ``False``. Any other value should returns the 
        mask of existing element in reference array
        
    :returns: array and values missings or indexes in reference array. 
    
    :Example: 
        
    >>> import numpy as np 
    >>> from watex.utils.funcutils import ismissing
    >>> refreq = np.linspace(7e7, 1e0, 20) # 20 frequencies as reference
    >>> # remove the value between index 7 to 12 and stack again
    >>> freq = np.hstack ((refreq.copy()[:7], refreq.copy()[12:] ))  
    >>> f, m  = ismissing (refreq, freq)
    >>> f, m  
    ...array([7.00000000e+07, 6.63157895e+07, 6.26315791e+07, 5.89473686e+07,
           5.52631581e+07, 5.15789476e+07, 4.78947372e+07,            nan,
                      nan,            nan,            nan,            nan,
           2.57894743e+07, 2.21052638e+07, 1.84210534e+07, 1.47368429e+07,
           1.10526324e+07, 7.36842195e+06, 3.68421147e+06, 1.00000000e+00])
    >>> m # missing values 
    ... array([44210526.68421052, 40526316.21052632, 36842105.73684211,
           33157895.2631579 , 29473684.78947368])
    >>>  _, m_ix  = ismissing (refreq, freq, return_index =True)
    >>> m_ix 
    ... array([ 7,  8,  9, 10, 11], dtype=int64)
    >>> # assert the missing values from reference values 
    >>> refreq[m_ix ] # is equal to m 
    ... array([44210526.68421052, 40526316.21052632, 36842105.73684211,
           33157895.2631579 , 29473684.78947368]) 
        
    """
    return_index = str(return_index).lower() 
    fill_value = _assert_all_types(fill_value, float, int)
    if return_index in ('false', 'value', 'val') :
        return_index ='values' 
    elif return_index  in ('true', 'index', 'ix') :
        return_index = 'index' 
    else : 
        return_index = 'mask'
    
    ref = refarr.copy() ; mask = np.isin(ref, arr)
    miss_values = ref [~np.isin(ref, arr)] 
    miss_val_or_ix  = (ref [:, None] == miss_values).argmax(axis=0
                         ) if return_index =='index' else ref [~np.isin(ref, arr)] 
    
    miss_val_or_ix = mask if return_index =='mask' else miss_val_or_ix 
    # if return_missing_values: 
    ref [~np.isin(ref, arr)] = fill_value 
    #arr= np.hstack ((arr , np.repeat(fill_value, 0 if m <=0 else m  ))) 
    #refarr[refarr ==arr] if return_index else arr 
    return  ref , miss_val_or_ix   


def fillNaN(arr, method ='ff'): 
    """ Most efficient way to back/forward-fill NaN values in numpy array. 
    
    Parameters 
    ---------- 
    arr : ndarray 
        Array containing NaN values to be filled 
    method: str 
        Method for filling. Can be forward fill ``ff`` or backward fill `bf``. 
        or ``both`` for the two methods. Default is `ff`. 
        
    Returns
    -------
    new array filled. 
    
    Notes 
    -----
    When NaN value is framed between two valid numbers, ``ff`` and `bf` performs 
    well the filling operations. However, when the array is ended by multiple 
    NaN values, the ``ff`` is recommended. At the opposite the ``bf`` is  the 
    method suggested. The ``both``argument does the both tasks at the expense of 
    the computation cost. 
    
    Examples 
    --------- 
        
    >>> import numpy as np 
    >>> from from watex.utils.funcutils import fillNaN 
    >>> arr2d = np.random.randn(7, 3)
    >>> # change some value into NaN 
    >>> arr2d[[0, 2, 3, 3 ],[0, 2,1, 2]]= np.nan
    >>> arr2d 
    ... array([[        nan, -0.74636104,  1.12731613],
           [ 0.48178017, -0.18593812, -0.67673698],
           [ 0.17143421, -2.15184895,         nan],
           [-0.6839212 ,         nan,         nan]])
    >>> fillNaN (arr2d) 
    ... array([[        nan, -0.74636104,  1.12731613],
           [ 0.48178017, -0.18593812, -0.67673698],
           [ 0.17143421, -2.15184895, -2.15184895],
           [-0.6839212 , -0.6839212 , -0.6839212 ]])
    >>> fillNaN(arr2d, 'bf')
    ... array([[-0.74636104, -0.74636104,  1.12731613],
           [ 0.48178017, -0.18593812, -0.67673698],
           [ 0.17143421, -2.15184895,         nan],
           [-0.6839212 ,         nan,         nan]])
    >>> fillNaN (arr2d, 'both')
    ... array([[-0.74636104, -0.74636104,  1.12731613],
           [ 0.48178017, -0.18593812, -0.67673698],
           [ 0.17143421, -2.15184895, -2.15184895],
           [-0.6839212 , -0.6839212 , -0.6839212 ]])
    
    References 
    ----------
    Some function below are edited by the authors in pyQuestion.com website. 
    There are other way more efficient to perform this task by calling the module 
    `Numba` to accelerate the computation time. However, at the time this script 
    is writen (August 17th, 2022) , `Numba` works with `Numpy` version 1.21. The
    latter  is older than the one used in for writting this package (1.22.3 ). 
    
    For furher details, one can refer to the following link: 
    https://pyquestions.com/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    
    """
    def ffill (arr): 
        """ Forward fill."""
        idx = np.where (~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate (idx, axis =1 , out =idx )
        return arr[np.arange(idx.shape[0])[:, None], idx ]
    
    def bfill (arr): 
        """ Backward fill """
        idx = np.where (~mask, np.arange(mask.shape[1]) , mask.shape[1]-1)
        idx = np.minimum.accumulate(idx[:, ::-1], axis =1)[:, ::-1]
        return arr [np.arange(idx.shape [0])[:, None], idx ]
    
    method= str(method).lower().strip() 
    
    if arr.ndim ==1: 
        arr = reshape(arr, axis=1)  
        
    if method  in ('backward', 'bf',  'bwd'):
        method = 'bf' 
    elif method in ('forward', 'ff', 'fwd'): 
        method= 'ff' 
    elif method in ('both', 'ffbf', 'fbwf', 'bff', 'full'): 
        method ='both'
    if method not in ('bf', 'ff', 'both'): 
        raise ValueError ("Expect a backward <'bf'>, forward <'ff'> fill "
                          f" or both <'bff'> not {method!r}")
    mask = np.isnan (arr)  
    if method =='both': 
        arr = ffill(arr) ;
        #mask = np.isnan (arr)  
        arr = bfill(arr) 
    
    return (ffill(arr) if method =='ff' else bfill(arr)
            ) if method in ('bf', 'ff') else arr    
    
    
def get_params (obj: object 
                ) -> Dict: 
    """
    Get object parameters. 
    
    Object can be callable or instances 
    
    :param obj: object , can be callable or instance 
    
    :return: dict of parameters values 
    
    :examples: 
        >>> from sklearn.svm import SVC 
        >>> from watex.utils.funcutils import get_params 
        >>> sigmoid= SVC (
            **{
                'C': 512.0,
                'coef0': 0,
                'degree': 1,
                'gamma': 0.001953125,
                'kernel': 'sigmoid',
                'tol': 1.0 
                }
            )
        >>> pvalues = get_params( sigmoid)
        >>> {'decision_function_shape': 'ovr',
             'break_ties': False,
             'kernel': 'sigmoid',
             'degree': 1,
             'gamma': 0.001953125,
             'coef0': 0,
             'tol': 1.0,
             'C': 512.0,
             'nu': 0.0,
             'epsilon': 0.0,
             'shrinking': True,
             'probability': False,
             'cache_size': 200,
             'class_weight': None,
             'verbose': False,
             'max_iter': -1,
             'random_state': None
         }
    """
    if hasattr (obj, '__call__'): 
        cls_or_func_signature = inspect.signature(obj)
        PARAMS_VALUES = {k: None if v.default is (inspect.Parameter.empty 
                         or ...) else v.default 
                    for k, v in cls_or_func_signature.parameters.items()
                    # if v.default is not inspect.Parameter.empty
                    }
    elif hasattr(obj, '__dict__'): 
        PARAMS_VALUES = {k:v  for k, v in obj.__dict__.items() 
                         if not (k.endswith('_') or k.startswith('_'))}
    
    return PARAMS_VALUES

    
    
def fit_by_ll(ediObjs): 
    """ Fit edi by location and reorganize EDI (sort EDI) according to the site  
    longitude and latitude coordinates. 
    
    EDIs data are mostly reading in an alphabetically order, so the reoganization  

    according to the location(longitude and latitude) is usefull for distance 
    betwen site computing with a right position at each site.  
    
    :param ediObjs: list of EDI object , composed of a collection of 
        pycsamt.core.edi.Edi object 
    :type ediObjs: pycsamt.core.edi.Edi_Collection 

    
    :returns: array splitted into ediObjs and Edifiles basenames 
    :rtyple: tuple 
    
    :Example: 
        >>> import numpy as np 
        >>> from watex.methods.em import EM
        >>> from watex.utils.funcutils import fit_by_ll
        >>> edipath ='data/edi_ss' 
        >>> cediObjs = EM (edipath) 
        >>> ediObjs = np.random.permutation(cediObjs.ediObjs) # shuffle the  
        ... # the collection of ediObjs 
        >>> ediObjs, ediObjbname = fit_by_ll(ediObjs) 
        ...
    
    """
    #get the ediObjs+ names in ndarray(len(ediObjs), 2) 
    objnames = np.c_[ediObjs, np.array(
        list(map(lambda obj: os.path.basename(obj.edifile), ediObjs)))]
    lataddlon = np.array (list(map(lambda obj: obj.lat + obj.lon , ediObjs)))
    sort_ix = np.argsort(lataddlon) 
    objnames = objnames[sort_ix ] 
    #ediObjs , objbnames = np.hsplit(objnames, 2) 
    return objnames[:, 0], objnames[:, -1]
   
    
def make_ids(arr, prefix =None, how ='py', skip=False): 
    """ Generate auto Id according to the number of given sites. 
    
    :param arr: Iterable object to generate an id site . For instance it can be 
        the array-like or list of EDI object that composed a collection of 
        pycsamt.core.edi.Edi object. 
    :type ediObjs: array-like, list or tuple 

    :param prefix: string value to add as prefix of given id. Prefix can be 
        the site name.
    :type prefix: str 
    
    :param how: Mode to index the station. Default is 'Python indexing' i.e. 
        the counting starts by 0. Any other mode will start the counting by 1.
    :type cmode: str 
    
    :param skip: skip the strong formatage. the formatage acccording to the 
        number of collected file. 
    :type skip: bool 
    :return: ID number formated 
    :rtype: list 
    
    :Example: 
        >>> import numpy as np 
        >>> from watex.utils.func_utils import make_ids 
        >>> values = ['edi1', 'edi2', 'edi3'] 
        >>> make_ids (values, 'ix')
        ... ['ix0', 'ix1', 'ix2']
        >>> data = np.random.randn(20)
        >>>  make_ids (data, prefix ='line', how=None)
        ... ['line01','line02','line03', ... , line20] 
        >>> make_ids (data, prefix ='line', how=None, skip =True)
        ... ['line1','line2','line3',..., line20] 
        
    """ 
    fm='{:0' + ('1' if skip else '{}'.format(int(np.log10(len(arr))) + 1)) +'}'
    id_ =[str(prefix) + fm.format(i if how=='py'else i+ 1 ) if prefix is not 
          None else fm.format(i if how=='py'else i+ 1) 
          for i in range(len(arr))] 
    return id_    
    
def show_stats(nedic , nedir, fmtl='~', lenl=77, obj='EDI'): 
    """ Estimate the file successfully read reading over the unread files

    :param nedic: number of input or collected files 
    :param nedir: number of files read sucessfully 
    :param fmt: str to format the stats line 
    :param lenl: length of line denileation."""
    
    def get_obj_len (value):
        """ Control if obj is iterable then take its length """
        try : 
            iter(value)
        except :pass 
        else : value =len(value)
        return value 
    nedic = get_obj_len(nedic)
    nedir = get_obj_len(nedir)
    
    print(fmtl * lenl )
    mesg ='|'.join( ['|{0:<15}{1:^2} {2:<7}',
                     '{3:<15}{4:^2} {5:<7}',
                     '{6:<9}{7:^2} {8:<7}%|'])
    print(mesg.format('Data collected','=',  nedic, f'{obj} success. read',
                      '=', nedir, 'Rate','=', round ((nedir/nedic) *100, 2),
                      2))
    print(fmtl * lenl ) 
    
def concat_array_from_list (list_of_array , concat_axis = 0) :
    """ Concat array from list and set the None value in the list as NaN.
    
    :param list_of_array: List of array elements 
    :type list of array: list 
    
    :param concat_axis: axis for concatenation ``0`` or ``1``
    :type concat_axis: int 
    
    :returns: Concatenated array with shape np.ndaarry(
        len(list_of_array[0]), len(list_of_array))
    :rtype: np.ndarray 
    
    :Example: 
        
    >>> import numpy as np 
    >>> from watex.utils.funcutils import concat_array_from_list 
    >>> np.random.seed(0)
    >>> ass=np.random.randn(10)
    >>> ass = ass2=np.linspace(0,15,10)
    >>> concat_array_from_list ([ass, ass]) 
    
    """
    concat_axis =int(_assert_all_types(concat_axis, int, float))
    if concat_axis not in (0 , 1): 
        raise ValueError(f'Unable to understand axis: {str(concat_axis)!r}')
    
    list_of_array = list(map(lambda e: np.array([np.nan])
                             if e is None else np.array(e), list_of_array))
    # if the list is composed of one element of array, keep it outside
    # reshape accordingly 
    if len(list_of_array)==1:
        ar = (list_of_array[0].reshape ((1,len(list_of_array[0]))
                 ) if concat_axis==0 else list_of_array[0].reshape(
                        (len(list_of_array[0]), 1)
                 )
             ) if list_of_array[0].ndim ==1 else list_of_array[0]
                     
        return ar 

    #if concat_axis ==1: 
    list_of_array = list(map(
            lambda e:e.reshape(e.shape[0], 1) if e.ndim ==1 else e ,
            list_of_array)
        ) if concat_axis ==1 else list(map(
            lambda e:e.reshape(1, e.shape[0]) if e.ndim ==1 else e ,
            list_of_array))
                
    return np.concatenate(list_of_array, axis = concat_axis)
    
def station_id (id_, is_index= 'index', how=None, **kws): 
    """ 
    From id get the station  name as input  and return index `id`. 
    Index starts at 0.
    
    :param id_: str, of list of the name of the station or indexes . 
    
    :param is_index: bool 
        considered the given station as a index. so it remove all the letter and
        keep digit as index of each stations. 
        
    :param how: :param how: Mode to index the station. Default is 
        'Python indexing' i.e.the counting starts by 0. Any other mode will 
        start the counting by 1. Note that if `is_index` is ``True`` and the 
        param `how` is set to it default value ``py``, the station index should 
        be downgraded to 1. 
        
    :param kws: additionnal keywords arguments from :func:`~.make_ids`.
    
    :return: station index. If the list `id_` is given will return the tuple.
    
    :example:
        
    >>> from watex.utils.funcutils import station_id 
    >>> dat1 = ['S13', 's02', 's85', 'pk20', 'posix1256']
    >>> station_id (dat1)
    ... (13, 2, 85, 20, 1256)
    >>> station_id (dat1, how='py')
    ... (12, 1, 84, 19, 1255)
    >>> station_id (dat1, is_index= None, prefix ='site')
    ... ('site1', 'site2', 'site3', 'site4', 'site5')
    >>> dat2 = 1 
    >>> station_id (dat2) # return index like it is
    ... 1
    >>> station_id (dat2, how='py') # considering the index starts from 0
    ... 0
    
    """
    is_iterable =False 
    is_index = str(is_index).lower().strip() 
    isix=True if  is_index in ('true', 'index', 'yes', 'ix') else False 
    
    regex = re.compile(r'\d+', flags=re.IGNORECASE)
    try : 
        iter (id_)
    except : 
        id_= [id_]
    else : is_iterable=True 
    
    #remove all the letter 
    id_= list(map( lambda o: regex.findall(o), list(map(str, id_))))
    # merge the sequences list and for consistency remove emty list or str 
    id_=tuple(filter (None, list(itertools.chain(*id_)))) 
    
    # if considering as Python index return value -1 other wise return index 
    
    id_ = tuple (map(int, np.array(id_, dtype = np.int32)-1)
                 ) if how =='py' else tuple ( map(int, id_)) 
    
    if (np.array(id_) < 0).any(): 
        warnings.warn('Index contains negative values. Be aware that you are'
                      " using a Python indexing. Otherwise turn 'how' argumennt"
                      " to 'None'.")
    if not isix : 
        id_= tuple(make_ids(id_, how= how,  **kws))
        
    if not is_iterable : 
        try: id_ = id_[0]
        except : warnings.warn("The station id is given as a non iterable "
                          "object, but can keep the same format in return.")
        if id_==-1: id_= 0 if how=='py' else id_ + 2 

    return id_

def assert_doi(doi): 
    """
     assert the depath of investigation Depth of investigation converter 

    :param doi: depth of investigation in meters.  If value is given as string 
        following by yhe index suffix of kilometers 'km', value should be 
        converted instead. 
    :type doi: str|float 
    
    :returns doi:value in meter
    :rtype: float
           
    """
    if isinstance (doi, str):
        if doi.find('km')>=0 : 
            try: doi= float(doi.replace('km', '000')) 
            except :TypeError (" Unrecognized value. Expect value in 'km' "
                           f"or 'm' not: {doi!r}")
    try: doi = float(doi)
    except: TypeError ("Depth of investigation must be a float number "
                       "not: {str(type(doi).__name__!r)}")
    return doi
    
    
def parse_json(json_fn =None,
               data=None, 
               todo='load',
               savepath=None,
               verbose:int =0,
               **jsonkws) :
    """ Parse Java Script Object Notation file and collect data from JSON
    config file. 
    
    :param json_fn: Json filename, URL or output JSON name if `data` is 
        given and `todo` is set to ``dump``.Otherwise the JSON output filename 
        should be the `data` or the given variable name.
    :param data: Data in Python obj to serialize. 
    :param todo: Action to perform with JSON: 
        - load: Load data from the JSON file 
        - dump: serialize data from the Python object and create a JSON file
    :param savepath: If ``default``  should save the `json_fn` 
        If path does not exist, should save to the <'_savejson_'>
        default path .
    :param verbose: int, control the verbosity. Output messages
    
    .. see also:: Read more about JSON doc
            https://docs.python.org/3/library/json.html
         or https://www.w3schools.com/python/python_json.asp 
         or https://www.geeksforgeeks.org/json-load-in-python/
         ...
 
    :Example: 
        >>> PATH = 'data/model'
        >>> k_ =['model', 'iter', 'mesh', 'data']
        >>> try : 
            INVERS_KWS = {
                s +'_fn':os.path.join(PATH, file) 
                for file in os.listdir(PATH) 
                          for s in k_ if file.lower().find(s)>=0
                          }
        except :
            INVERS=dict()
        >>> TRES=[10, 66,  70, 100, 1000, 3000]# 7000]     
        >>> LNS =['river water','fracture zone', 'MWG', 'LWG', 
              'granite', 'igneous rocks', 'basement rocks']
        >>> import watex.utils.funcutils as FU
        >>> geo_kws ={'oc2d': INVERS_KWS, 
                      'TRES':TRES, 'LN':LNS}
        # serialize json data and save to  'jsontest.json' file
        >>> FU.parse_json(json_fn = 'jsontest.json', 
                          data=geo_kws, todo='dump', indent=3,
                          savepath ='data/saveJSON', sort_keys=True)
        # Load data from 'jsontest.json' file.
        >>> FU.parse_json(json_fn='data/saveJSON/jsontest.json', todo ='load')
    
    """
    todo, domsg =return_ctask(todo)
    # read urls by default json_fn can hold a url 
    try :
        if json_fn.find('http') >=0 : 
            todo, json_fn, data = fetch_json_data_from_url(json_fn, todo)
    except:
        #'NoneType' object has no attribute 'find' if data is not given
        pass 

    if todo.find('dump')>=0:
        json_fn = get_config_fname_from_varname(
            data, config_fname= json_fn, config='.json')
        
    JSON = dict(load=json.load,# use loads rather than load  
                loads=json.loads, 
                dump= json.dump, 
                dumps= json.dumps)
    try :
        if todo=='load': # read JSON files 
            with open(json_fn) as fj: 
                data =  JSON[todo](fj)  
        elif todo=='loads': # can be JSON string format 
            data = JSON[todo](json_fn) 
        elif todo =='dump': # store data in JSON file.
            with open(f'{json_fn}.json', 'w') as fw: 
                data = JSON[todo](data, fw, **jsonkws)
        elif todo=='dumps': # store data in JSON format not output file.
            data = JSON[todo](data, **jsonkws)

    except json.JSONDecodeError: 
        raise json.JSONDecodeError(f"Unable {domsg} JSON {json_fn!r} file. "
                              "Please check your file.", f'{json_fn!r}', 1)
    except: 
        msg =''.join([
        f"{'Unrecognizable file' if todo.find('load')>=0 else'Unable to serialize'}"
        ])
        
        raise TypeError(f'{msg} {json_fn!r}. Please check your'
                        f" {'file' if todo.find('load')>=0 else 'data'}.")
        
    cparser_manager(f'{json_fn}.json',savepath, todo=todo, dpath='_savejson_', 
                    verbose=verbose , config='JSON' )

    return data 
 
def fetch_json_data_from_url (url:str , todo:str ='load'): 
    """ Retrieve JSON data from url 
    :param url: Universal Resource Locator .
    :param todo:  Action to perform with JSON:
        - load: Load data from the JSON file 
        - dump: serialize data from the Python object and create a JSON file
    """
    with urllib.request.urlopen(url) as jresponse :
        source = jresponse.read()
    data = json.loads(source)
    if todo .find('load')>=0:
        todo , json_fn  ='loads', source 
        
    if todo.find('dump')>=0:  # then collect the data and dump it
        # set json default filename 
        todo, json_fn = 'dumps',  '_urlsourcejsonf.json'  
        
    return todo, json_fn, data 
    
def parse_csv(
        csv_fn:str =None,
        data=None, 
        todo='reader', 
         fieldnames=None, 
         savepath=None,
         header: bool=False, 
         verbose:int=0,
         **csvkws
   ) : 
    """ Parse comma separated file or collect data from CSV.
    
    :param csv_fn: csv filename,or output CSV name if `data` is 
        given and `todo` is set to ``write|dictwriter``.Otherwise the CSV 
        output filename should be the `c.data` or the given variable name.
    :param data: Sequence Data in Python obj to write. 
    :param todo: Action to perform with JSON: 
        - reader|DictReader: Load data from the JSON file 
        - writer|DictWriter: Write data from the Python object 
        and create a CSV file
    :param savepath: If ``default``  should save the `csv_fn` 
        If path does not exist, should save to the <'_savecsv_'>
        default path.
    :param fieldnames: is a sequence of keys that identify the order
        in which values in the dictionary passed to the `writerow()`
            method are written `csv_fn` file.
    :param savepath: If ``default``  should save the `csv_fn` 
        If path does not exist, should save to the <'_savecsv_'>
        default path .
    :param verbose: int, control the verbosity. Output messages
    :param csvkws: additional keywords csv class arguments 
    
    .. see also:: Read more about CSV module in:
        https://docs.python.org/3/library/csv.html or find some examples
        here https://www.programcreek.com/python/example/3190/csv.DictWriter 
        or find some FAQS here: 
    https://stackoverflow.com/questions/10373247/how-do-i-write-a-python-dictionary-to-a-csv-file
        ...
    :Example:
        >>> import watex.utils.funcutils as FU
        >>> PATH = 'data/model'
        >>> k_ =['model', 'iter', 'mesh', 'data']
        >>> try : 
            INVERS_KWS = {
                s +'_fn':os.path.join(PATH, file) 
                for file in os.listdir(PATH) 
                          for s in k_ if file.lower().find(s)>=0
                          }
        except :
            INVERS=dict()
        >>> TRES=[10, 66,  70, 100, 1000, 3000]# 7000]     
        >>> LNS =['river water','fracture zone', 'MWG', 'LWG', 
              'granite', 'igneous rocks', 'basement rocks']
        >>> geo_kws ={'oc2d': INVERS_KWS, 
                      'TRES':TRES, 'LN':LNS}
        >>> # write data and save to  'csvtest.csv' file 
        >>> # here the `data` is a sequence of dictionary geo_kws
        >>> FU.parse_csv(csv_fn = 'csvtest.csv',data = [geo_kws], 
                         fieldnames = geo_kws.keys(),todo= 'dictwriter',
                         savepath = 'data/saveCSV')
        # collect csv data from the 'csvtest.csv' file 
        >>> FU.parse_csv(csv_fn ='data/saveCSV/csvtest.csv',
                         todo='dictreader',fieldnames = geo_kws.keys()
                         )
    
    """
    todo, domsg =return_ctask(todo) 
    
    if todo.find('write')>=0:
        csv_fn = get_config_fname_from_varname(
            data, config_fname= csv_fn, config='.csv')
    try : 
        if todo =='reader': 
            with open (csv_fn, 'r') as csv_f : 
                csv_reader = csv.reader(csv_f) # iterator 
                data =[ row for row in csv_reader]
                
        elif todo=='writer': 
            # write without a blank line, --> new_line =''
            with open(f'{csv_fn}.csv', 'w', newline ='',
                      encoding ='utf8') as new_csvf:
                csv_writer = csv.writer(new_csvf, **csvkws)
                csv_writer.writerows(data) if len(
                    data ) > 1 else csv_writer.writerow(data)  
                # for row in data:
                #     csv_writer.writerow(row) 
        elif todo=='dictreader':
            with open (csv_fn, 'r', encoding ='utf8') as csv_f : 
                # generate an iterator obj 
                csv_reader= csv.DictReader (csv_f, fieldnames= fieldnames) 
                # return csvobj as a list of dicts
                data = list(csv_reader) 
        
        elif todo=='dictwriter':
            with open(f'{csv_fn}.csv', 'w') as new_csvf:
                csv_writer = csv.DictWriter(new_csvf, **csvkws)
                if header:
                    csv_writer.writeheader()
                # DictWriter.writerows()expect a list of dicts,
                # while DictWriter.writerow() expect a single row of dict.
                csv_writer.writerow(data) if isinstance(
                    data , dict) else csv_writer.writerows(data)  
                
    except csv.Error: 
        raise csv.Error(f"Unable {domsg} CSV {csv_fn!r} file. "
                      "Please check your file.")
    except: 

        msg =''.join([
        f"{'Unrecognizable file' if todo.find('read')>=0 else'Unable to write'}"
        ])
        
        raise TypeError(f'{msg} {csv_fn!r}. Please check your'
                        f" {'file' if todo.find('read')>=0 else 'data'}.")
    cparser_manager(f'{csv_fn}.csv',savepath, todo=todo, dpath='_savecsv_', 
                    verbose=verbose , config='CSV' )
    
    return data  
   
def return_ctask (todo:Optional[str]=None) -> Tuple [str, str]: 
    """ Get the convenient task to do if users misinput the `todo` action.
    
    :param todo: Action to perform: 
        - load: Load data from the config [YAML|CSV|JSON] file
        - dump: serialize data from the Python object and 
            create a config [YAML|CSV|JSON] file."""
            
    def p_csv(v, cond='dict', base='reader'):
        """ Read csv instead. 
        :param v: str, value to do 
        :param cond: str, condition if  found in the value `v`. 
        :param base: str, base task to do if condition `cond` is not met. 
        
        :Example: 
            
        >>> todo = 'readingbook' 
        >>> p_csv(todo) <=> 'dictreader' if todo.find('dict')>=0 else 'reader' 
        """
        return  f'{cond}{base}' if v.find(cond) >=0 else base   
    
    ltags = ('load', 'recover', True, 'fetch')
    dtags = ('serialized', 'dump', 'save', 'write','serialize')
    if todo is None: 
        raise ValueError('NoneType action can not be perform. Please '
                         'specify your action: `load` or `dump`?' )
    
    todo =str(todo).lower() 
    ltags = list(ltags) + [todo] if  todo=='loads' else ltags
    dtags= list(dtags) +[todo] if  todo=='dumps' else dtags 

    if todo in ltags: 
        todo = 'loads' if todo=='loads' else 'load'
        domsg= 'to parse'
    elif todo in dtags: 
        todo = 'dumps' if todo=='dumps' else 'dump'
        domsg  ='to serialize'
    elif todo.find('read')>=0:
        todo = p_csv(todo)
        domsg= 'to read'
    elif todo.find('write')>=0: 
        todo = p_csv(todo, base ='writer')
        domsg =' to write'
        
    else :
        raise ValueError(f'Wrong action {todo!r}. Please select'
                         f' the right action to perform: `load` or `dump`?'
                        ' for [JSON|YAML] and `read` or `write`? '
                        'for [CSV].')
    return todo, domsg  

def parse_yaml (yml_fn:str =None, data=None,
                todo='load', savepath=None,
                verbose:int =0, **ymlkws) : 
    """ Parse yml file and collect data from YAML config file. 
    
    :param yml_fn: yaml filename and can be the output YAML name if `data` is 
        given and `todo` is set to ``dump``.Otherwise the YAML output filename 
        should be the `data` or the given variable name.
    :param data: Data in Python obj to serialize. 
    :param todo: Action to perform with YAML: 
        - load: Load data from the YAML file 
        - dump: serialize data from the Python object and create a YAML file
    :param savepath: If ``default``  should save the `yml_fn` 
        to the default path otherwise should store to the convenient path.
        If path does not exist, should set to the default path.
    :param verbose: int, control the verbosity. Output messages
    
    .. see also:: Read more about YAML file https://pynative.com/python-yaml/
         or https://python.land/data-processing/python-yaml and download YAML 
         at https://pypi.org/project/PyYAML/
         ...

    """ 
    
    todo, domsg =return_ctask(todo)
    #in the case user use dumps or loads with 's'at the end 
    if todo.find('dump')>= 0: 
        todo='dump'
    if todo.find('load')>=0:
        todo='load'
    if todo=='dump':
        yml_fn = get_config_fname_from_varname(data, yml_fn)
    try :
        if todo=='load':
            with open(yml_fn) as fy: 
                data =  yaml.load(fy, Loader=yaml.SafeLoader)  
                # args =yaml.safe_load(fy)
        elif todo =='dump':
        
            with open(f'{yml_fn}.yml', 'w') as fw: 
                data = yaml.dump(data, fw, **ymlkws)
    except yaml.YAMLError: 
        raise yaml.YAMLError(f"Unable {domsg} YAML {yml_fn!r} file. "
                             'Please check your file.')
    except: 
        msg =''.join([
        f"{'Unrecognizable file' if todo=='load'else'Unable to serialize'}"
        ])
        
        raise TypeError(f'{msg} {yml_fn!r}. Please check your'
                        f" {'file' if todo=='load' else 'data'}.")
        
    cparser_manager(f'{yml_fn}.yml',savepath, todo=todo, dpath='_saveyaml_', 
                    verbose=verbose , config='YAML' )

    return data 
 
def cparser_manager (cfile,
                     savepath =None, 
                     todo:str ='load', dpath=None,
                     verbose =0, **pkws): 
    """ Save and output message according to the action. 
    :param cfile: name of the configuration file
    :param savepath: Path-like object 
    :param dpath: default path 
    :param todo: Action to perform with config file. Can ve 
        ``load`` or ``dump``
    :param config: Type of configuration file. Can be ['YAML|CSV|JSON]
    :param verbose: int, control the verbosity. Output messages
    
    """
    if savepath is not None:
        if savepath =='default': 
            savepath = None 
        yml_fn,_= move_cfile(cfile,savepath, dpath=dpath)
    if verbose > 0: 
        print_cmsg(yml_fn, todo, **pkws)
        
    
def get_config_fname_from_varname(data,
                                  config_fname=None,
                                  config='.yml') -> str: 
    """ use the variable name given to data as the config file name.
    :param data: Given data to retrieve the variable name 
    :param config_fname: Configurate variable filename. If ``None`` , use 
        the name of the given varibale data 
    :param config: Type of file for configuration. Can be ``json``, ``yml`` 
        or ``csv`` file. default is ``yml``.
    :return: str, the configuration data.
    
    """
    try:
        if '.' in config: 
            config =config.replace('.','')
    except:pass # in the case None is given
    
    if config_fname is None: # get the varname 
        # try : 
        #     from varname.helpers import Wrapper 
        # except ImportError: 
        #     import_varname=False 
        #     import_varname = FU.subprocess_module_installation('varname')
        #     if import_varname: 
        #         from varname.helpers import Wrapper 
        # else : import_varname=True 
        try : 
            for c, n in zip(['yml', 'yaml', 'json', 'csv'],
                            ['cy.data', 'cy.data', 'cj.data',
                             'c.data']):
                if config ==c:
                    config_fname= n
                    break 
            if config_fname is None:
                raise # and go to except  
        except: 
            #using fstring 
            config_fname= f'{data}'.split('=')[0]
            
    elif config_fname is not None: 
        config_fname= config_fname.replace(
            f'.{config}', '').replace(f'.{config}', '').replace('.yaml', '')
    
    return config_fname
    

def move_cfile (cfile:str , savepath:Optional[str]=None, **ckws):
    """ Move file to its savepath and output message. 
    If path does not exist, should create one to save data.
    :param cfile: name of the configuration file
    :param savepath: Path-like object 
    :param dpath: default path 
    
    :returns: 
        - configuration file 
        - out message 
    """
    savepath = cpath(savepath, **ckws)
    try :shutil.move(cfile, savepath)
    except: warnings.warn("It seems the path already exists!")
    
    cfile = os.path.join(savepath, cfile)
    
    msg = ''.join([
    f'--> Data was successfully stored to {os.path.basename(cfile)!r}', 
        f' and saved to {os.path.realpath(cfile)!r}.']
        )
        
    return cfile, msg

def print_cmsg(cfile:str, todo:str='load', config:str='YAML') -> str: 
    """ Output configuration message. 
    
    :param cfile: name of the configuration file
    :param todo: Action to perform with config file. Can be 
        ``load`` or ``dump``
    :param config: Type of configuration file. Can be [YAML|CSV|JSON]
    """
    if todo=='load': 
        msg = ''.join([
        f'--> Data was successfully stored to {os.path.basename(cfile)!r}', 
            f' and saved to {os.path.realpath(cfile)!r}.']
            )
    elif todo=='dump': 
        msg =''.join([ f"--> {config.upper()} {os.path.basename(cfile)!r}", 
                      " data was sucessfully loaded."])
    return msg 


def random_state_validator(seed):
    """Turn seed into a np.random.RandomState instance.
    
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
        
    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )


def is_iterable (y, /)->bool: 
    """ Asserts iterable object and returns 'True' or 'False' """
    return hasattr (y, '__iter__') 

    
def str2columns (text, /, regex=None , pattern = None): 
    """Split text from the non-alphanumeric markers using regular expression. 
    
    Remove all string non-alphanumeric and some operator indicators,  and 
    fetch attributes names. 
    
    Parameters 
    -----------
    
    text: str, 
        text litteral containing the columns the names to retrieve
        
    regex: `re` object,  
        Regular expresion object. the default is:: 
            
            >>> import re 
            >>> re.compile (r'[_#&*@!_,;\s-]\s*', flags=re.IGNORECASE) 
    pattern: str, default = '[_#&*@!_,;\s-]\s*'
        The base pattern to split the text into a columns
        
    Returns
    -------
    attr: List of attributes 
    
    Examples
    ---------
    >>> from watex.utils.funcutils import str2columns 
    >>> text = ('this.is the text to split. It is an: example of; splitting str - to text.')
    >>> tsplitted= str2columns (text ) 
    ... ['this',
         'is',
         'the',
         'text',
         'to',
         'split',
         'It',
         'is',
         'an:',
         'example',
         'of',
         'splitting',
         'str',
         'to',
         'text']

    """
    pattern = r'[_#&.*@!_,;\s-]\s*'
    regex = regex or re.compile (pattern, flags=re.IGNORECASE) 
    text= list(filter (None, regex.split(text)))
    return text 
    
    
def sanitize_frame_cols(d, /, func:F = None , regex=None, pattern:str = None, 
                            fill_pattern:str =None, inplace:bool =False ):
    """ Remove an indesirable characters and returns new columns 
    
    Use regular expression for columns sanitizing 
    
    Parameters 
    -----------
    
    d: list, columns, 
        columns to sanitize. It might contain a list of items to 
        to polish. if dataframe or series are given, the dataframe columns  
        and the name respectively will be polished and returns the same 
        dataframe.
        
    regex: `re` object,
        Regular expresion object. the default is:: 
            
            >>> import re 
            >>> re.compile (r'[_#&.)(*@!_,;\s-]\s*', flags=re.IGNORECASE) 
    pattern: str, default = '[_#&.)(*@!_,;\s-]\s*'
        The base pattern to sanitize the text in each column names. 
        
    fill_pattern: str, default='' 
        pattern to replace the non-alphabetic character in each item of 
        columns. 
    inplace: bool, default=False, 
        transform the dataframe of series in place. 

    Returns
    -------
    columns | pd.Series | dataframe. 
        return Serie or dataframe if one is given, otherwise it returns a 
        sanitized columns. 
        
    Examples 
    ---------
    >>> from watex.utils.funcutils import sanitize_frame_cols 
    >>> from watex.utils.coreutils import read_data 
    >>> h502= read_data ('data/boreholes/H502.xlsx') 
    >>> h502 = sanitize_frame_cols (h502, fill_pattern ='_' ) 
    >>> h502.columns[:3]
    ... Index(['depth_top', 'depth_bottom', 'strata_name'], dtype='object') 
    >>> f = lambda r : r.replace ('_', "'s ") 
    >>> h502_f= sanitize_frame_cols( h502, func =f )
    >>> h502_f.columns [:3]
    ... Index(['depth's top', 'depth's bottom', 'strata's name'], dtype='object')
               
    """
    isf , iss= False , False 
    pattern = r'[_#&.)(*@!_,;\s-]\s*'
    fill_pattern = fill_pattern or '' 
    fill_pattern = str(fill_pattern)
    
    regex = regex or re.compile (pattern, flags=re.IGNORECASE)
    
    if isinstance(d, pd.Series): 
        c = [d.name]  
        iss =True 
    elif isinstance (d, pd.DataFrame ) :
        c = list(d.columns) 
        isf = True
        
    else : 
        if not is_iterable(d) : c = [d] 
        else : c = d 
        
    if inspect.isfunction(func): 
        c = list( map (func , c ) ) 
    
    else : c =list(map ( 
        lambda r : regex.sub(fill_pattern, r.strip() ), c ))
        
    if isf : 
        if inplace : d.columns = c
        else : d =pd.DataFrame(d.values, columns =c )
        
    elif iss:
        if inplace: d.name = c[0]
        else : d= pd.Series (data =d.values, name =c[0] )
        
    else : d = c 

    return d 

def to_hdf5(d, /, fn= None, objname =None, close =True,  **hdf5_kws): 
    """
    Store a frame data in hierachical data format 5 (HDF5) 
    
    Note that is `d` is a dataframe, make sure that the dependency 'pytables'
    is already installed, otherwise and error raises. 
    
    Parameters 
    -----------
    d: ndarray, 
        data to store in HDF5 format 
    fn: str, 
        File path to HDF5 file.
    objname: str, 
        name of the data to store 
    close: bool, default =True 
        when data is given as an array, data can still be added if 
        close is set to ``False``, otherwise, users need to open again in 
        read mode 'r' before pursuing the process of adding. 
    hdf5_kws: dict of :class:`pandas.pd.HDFStore`  
        Additional keywords arguments passed to pd.HDFStore. they could be:
        *  mode : {'a', 'w', 'r', 'r+'}, default 'a'
    
             ``'r'``
                 Read-only; no data can be modified.
             ``'w'``
                 Write; a new file is created (an existing file with the same
                 name would be deleted).
             ``'a'``
                 Append; an existing file is opened for reading and writing,
                 and if the file does not exist it is created.
             ``'r+'``
                 It is similar to ``'a'``, but the file must already exist.
         * complevel : int, 0-9, default None
             Specifies a compression level for data.
             A value of 0 or None disables compression.
         * complib : {'zlib', 'lzo', 'bzip2', 'blosc'}, default 'zlib'
             Specifies the compression library to be used.
             As of v0.20.2 these additional compressors for Blosc are supported
             (default if no compressor specified: 'blosc:blosclz'):
             {'blosc:blosclz', 'blosc:lz4', 'blosc:lz4hc', 'blosc:snappy',
              'blosc:zlib', 'blosc:zstd'}.
             Specifying a compression library which is not available issues
             a ValueError.
         * fletcher32 : bool, default False
             If applying compression use the fletcher32 checksum.
    Returns
    ------- 
    store : Dict-like IO interface for storing pandas objects.
    
    >>> import os 
    >>> from watex.utils.funcutils import sanitize_frame_cols, to_hdf5 
    >>> from watex.utils import read_data 
    >>> data = read_data('data/boreholes/H502.xlsx') 
    >>> store_path = os.path.join('watex/datasets/data', 'h') # 'h' is the name of the data 
    >>> store = to_hdf5 (data, fn =store_path , objname ='h502' ) 
    >>> store 
    ... 
    >>> # fetch the data 
    >>> h502 = store ['h502'] 
    
    """
    if hasattr (d, '__array__') and hasattr (d, "columns"): 
        # assert whether pytables is installed 
        import_optional_dependency ('pytables') 
        store = pd.HDFStore(fn +'.h5' , **hdf5_kws)
        objname = objname or 'data'
        store[ str(objname) ] = d 
        return store 
    
    elif not hasattr(d, '__array__'): 
        d = np.asarray(d) 
 
        store= h5py.File(f"{fn}.hdf5", "w") 
        store.create_dataset("dataset_01", store.shape, 
                             dtype=store.dtype,
                             data=store
                             )
        if close : store.close () 
        
    return store 
    
    
    
    
    
    
    
    
    
    
    
    

    
        