# -*- coding: utf-8 -*-
#   Copyright © 2021  ~alias @Daniel03 <etanoyau@gmail.com> 
#   created date :Sun Sep 13 09:24:00 2020
#   edited date: Wed Jul  7 22:23:02 2021 
#   Licence: MIT 

from __future__ import annotations 

import os 
import re 
import sys 
import itertools
import inspect 
import subprocess 
import warnings
import csv 
import shutil 
from copy import deepcopy 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from .._watexlog import watexlog
from ..decorators import ( 
    deprecated, 
    catmapflow2,
    )

from ..typing import ( 
    Tuple,
    Dict,
    Any,
    ArrayLike,
    F,
    T,
    List ,
    DataFrame, 
    Sub,
    )
from ..property import P
from ..exceptions import ( 
    EDIError,
    ParameterNumberError, 
    )

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
        >>> from watex.tools.funcutils import smart_strobj_recognition
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
        ix = container_.index (name)
        
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
        
    >>> from watex.tools.funcutils import repr_callable_obj
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
                         if not (k.endswith('_') or k.startswith('_'))
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


def accept_types (*objtypes: list , 
                  format: bool = False
                  ) -> List[str] | str : 
    """ List the type format that can be accepted by a function. 
    
    :param objtypes: List of object types.
    
    :param format: bool - format the list of the name of objects.
    
    :return: list of object type names or str of object names. 
    
    :Example: 
        >>> import numpy as np; import pandas as pd 
        >>> from watex.tools.funcutils import accept_types
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

def is_installing (module, upgrade=True , DEVNULL=False,
                  action=True, verbose =0, **subpkws): 
    """ Install or uninstall a module using the subprocess under the hood.
    
    :param module: str, module name.
    
    :param upgrade:bool, install the lastest version.
    
    :param verbose:output a message.
    
    :param DEVNULL: decline the stdoutput the message in the console.
    
    :param action: str, install or uninstall a module.
    
    :param subpkws: additional subprocess keywords arguments.
    
    :Example: 
        >>> from pycsamt.tools.funcutils import is_installing
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
    
    MOD_IMP=False 

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
            _logger.info( f"{action_msg.capitalize()} of `{module}` "
                         "and dependancies was successfully done!") 
        MOD_IMP=True
        
    except: 
        _logger.error(f"Failed to {action} the module =`{module}`.")
        
        if verbose > 0 : 
            print(f'---> Module {module!r} {action_msg} failed. Please use'
                f' the following command: {cmdg} to manually do it.')
    else : 
        if verbose > 0: 
            print(f"{action_msg.capitalize()} of `{module}` "
                      "and dependancies was successfully done!") 
        
    return MOD_IMP 


def smart_format(iter_obj, choice ='and'): 
    """ Smart format iterable ob.
    
    :param iter_obj: iterable obj 
    :param choice: can be 'and' or 'or' for optional.
    
    :Example: 
        >>> from watex.tools.funcutils import smart_format
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
        
        >>> from watex.tools import funcutils as func 
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

        return columns

    new_df_columns= getandReplace(
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
    >>> from watex.tools import funcutils as func
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
        
        >>> from watex.tools  import funcutils as func
        >>> path =  os.path.join(os.environ["pyCSAMT"], 
                          'csamtpy','data', K6.stn)
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
        
        >>> from watex.tools.funcutils import display_infos
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
    """ Translate variable data from french csva data  to english with 
    varibale parser file. 
    
    :param csv_fn: data collected in csv format.
    
    :param pf: parser file. 
    
    :param destfile: str,  Destination file, outputfile.
    
    :param savepath: Path-Like object, save data to a path. 
                      
    :Example: 
        # to execute this script, we need to import the two modules below
        >>> import os 
        >>> import csv 
        >>> from watex.tools.funcutils import convert_csvdata_from_fr_to_en
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
    csv_1bb= deepcopy(csv_1b)
   
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
        >>> from watex.tools.funcutils import sanitize_unicode_string 
        >>> sentence ='Nos clients sont extrêmement satisfaits '
            'de la qualité du service fourni. En outre Nos clients '
                'rachètent frequemment nos "services".'
        >>> sanitize_unicode_string  (sentence)
        ... 'nosclientssontextrmementsatisfaitsdelaqualitduservice'
            'fournienoutrenosclientsrachtentfrequemmentnosservices'
    """
    str_ = str_.strip()
    str_= re.sub('\s+', '', str_).lower(
        ).replace('.', ''
        ).replace('à', 'a'
        ).replace("'", ""
        ).replace('(', ''
        ).replace(')', ''
        ).replace('-', ''
        ).replace('"', ''
        ).replace("é", "e"
        ).replace("è", "e"
        ).replace ('ê','e'
        ).replace("â", 'a'
        ).replace (',',""
        ).replace("\\",''
        ).replace("/", ''
        ).replace("ô", 'o'
        ).replace("’", ''
        )

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
    csv_1bb= deepcopy(csv_1b)
    copyd = deepcopy(csv_1bb); is_missing =list()
    
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
    

#XXX TODO : move the stats func 
def _stats (X_, y_true,*, y_pred,
            from_c ='geol', 
            drop_columns =None, 
            columns=None )  : 
    """ Present a short static"""
    import pandas as pd 
    
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
    s_values = s.values 
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
     
    
def _isin (
        arr: ArrayLike | List [float] ,
        subarr: Sub [ArrayLike] |Sub[List[float]] | float 
) -> bool : 
    """ Check whether the subset array `subcz` is in  `cz` array. 
    
    :param arr: Array-like - Array of item elements 
    :param subarr: Array-like, float - Subset array containing a subset items.
    :return: True if items in  test array `subarr` are in array `arr`. 
    
    """
    arr = np.array (arr );  subarr = np.array(subarr )

    return True if True in np.isin (arr, subarr) else False 

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
        
        >>> from watex.tools.funcutils import find_positon_from_sa
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
        
        >>> from watex.tools.funcutils import fmt_text
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
        
        >>> from watex.tools.funcutils import find_position_bounds  
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
    
@deprecated ('Function should be removed for the next release.')
def get_type (
        erp_array,
        posMinMax,
        pk, pos_array,
        dl
              ): 
    """
    Find anomaly type from app. resistivity values and positions locations 
    
    :param erp_array: App.resistivty values of all `erp` lines 
    :type erp_array: array_like 
    
    :param posMinMax: Selected anomaly positions from startpoint and endpoint 
    :type posMinMax: list or tuple or nd.array(1,2)
    
    :param pk: Position of selected anomaly in meters 
    :type pk: float or int 
    
    :param pos_array: Stations locations or measurements positions 
    :type pos_array: array_like 
    
    :param dl: 
        
        Distance between two receiver electrodes measurement. The same 
        as dipole length in meters. 
    
    :returns: 
        - ``EC`` for Extensive conductive. 
        - ``NC`` for narrow conductive. 
        - ``CP`` for conductive plane 
        - ``CB2P`` for contact between two planes. 
        
    :Example: 
        
        >>> from watex.tools.funcutils import get_type 
        >>> x = [60, 61, 62, 63, 68, 65, 80,  90, 100, 80, 100, 80]
        >>> pos= np.arange(0, len(x)*10, 10)
        >>> ano_type= get_type(erp_array= np.array(x),
        ...            posMinMax=(10,90), pk=50, pos_array=pos, dl=10)
        >>> ano_type
        ...CB2P

    """
    # Get position index 
    anom_type ='CP'
    index_pos = int(np.where(pos_array ==pk)[0])
    # if erp_array [:index_pos +1].mean() < np.median(erp_array) or\
    #     erp_array[index_pos:].mean() < np.median(erp_array) : 
    #         anom_type ='CB2P'
    if erp_array [:index_pos+1].mean() < np.median(erp_array) and \
        erp_array[index_pos:].mean() < np.median(erp_array) : 
            anom_type ='CB2P'
            
    elif erp_array [:index_pos +1].mean() >= np.median(erp_array) and \
        erp_array[index_pos:].mean() >= np.median(erp_array) : 
                
        if  dl <= (max(posMinMax)- min(posMinMax)) <= 5* dl: 
            anom_type = 'NC'

        elif (max(posMinMax)- min(posMinMax))> 5 *dl: 
            anom_type = 'EC'

    return anom_type   
        
    
@catmapflow2(cat_classes=['FR0', 'FR1', 'FR2', 'FR3'])#, 'FR4'] )
def categorize_flow(
        target_array: T ,
        flow_values: List [float],
        **kwargs
    ) -> Tuple[ List[float], T, List[str]]: 
    """ 
    Categorize `flow` into different classes. If the optional
    `flow_classes`  argument is given, it should be erased the
    `cat_classes` argument of decororator `deco.catmapflow`.
    
    :param target_array: Flow array to be categorized 
    
    :param flow_values: 
        
        The way to be categorized. Distribute the flow values 
        of numerical values considered as pseudo_classes like:: 
    
            flow_values= [0.0, [0.0, 3.0], [3.0, 6.0], [6.0, 10.0], 10.0] (1)
            
        if ``flow_values`` is given as follow:: 
            
            flow_values =[0. , 3., 6., 10.] (2)
        
        It should convert the type (2) to (1).
        
    :param flow_classes: 
        Values of categorized flow rates 
        
    :returns: 
        
        - ``new_flow_values``: Iterable object as type (2) 
        - ``target_array``: Raw flow iterable object to be categorized
        - ``flowClasses``: If given , see ``flow_classes`` param. 
            
    """
    flowClasses =  kwargs.pop('classes', None)

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
 
    return new_flow_values, target_array, flowClasses
    
    
   
def reshape(arr , axis = None) :
    """ Detect the array shape and reshape it accordingly, back to the given axis. 
    
    :param array: array_like with number of dimension equals to 1 or 2 
    :param axis: axis to reshape back array. If 'axis' is None and 
        the number of dimension is greater than 1, it reshapes back array 
        to array-like 
    
    :returns: New reshaped array 
    
    :Example: 
        >>> import numpy as np 
        >>> from watex.tools.funcutils import reshape 
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
    >>> from watex.tools.funcutils import ismissing
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
    >>> from from watex.tools.funcutils import fillNaN 
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
    elif method in ('both', 'ffbf', 'fbwf', 'bff'): 
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
        >>> from watex.tools.funcutils import get_params 
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
        >>> from watex.tools.funcutils import fit_by_ll
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
        >>> from watex.tools.func_utils import make_ids 
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
    >>> from watex.tools.funcutils import concat_array_from_list 
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
        
    >>> from watex.tools.funcutils import station_id 
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
        