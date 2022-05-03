# -*- coding: utf-8 -*-
#   Copyright © 2021  ~alias @Daniel03 <etanoyau@gmail.com> 
#   created date :Sun Sep 13 09:24:00 2020
#   edited date: Wed Jul  7 22:23:02 2021 

####################### import modules #######################

import os 
import sys 
import subprocess 
import warnings
import csv 
import shutil 
from copy import deepcopy 

import numpy as np 
import matplotlib.pyplot as plt

from ._watexlog import watexlog

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

###################### end import module ##################################

  
def check_dimensionality(obj, data, z, x):
    """ Check dimensionality of data and fix it.
    
    :param obj: Object, can be a class logged or else.
    :param data: 2D grid data of ndarray (z, x) dimensions
    :param z: array-like should be reduced along the row axis
    :param x: arraylike should be reduced along the columns axis.
    """
    def reduce_shape(Xshape, x, axis_name =None): 
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

def subprocess_module_installation (module, upgrade =True , DEVNULL=False,
                                    action=True, verbose =0, **subpkws): 
    """ Install or uninstall a module using the subprocess under the hood.
    
    :param module: str, module name 
    :param upgrade:bool, install the lastest version.
    :param verbose:output a message 
    :param DEVNULL: decline the stdoutput the message in the console 
    :param action: str, install or uninstall a module 
    :param subpkws: additional subprocess keywords arguments.
    
    :Example: 
        >>> from pycsamt.utils.func_utils import subprocess_module_installation
        >>> subprocess_module_installation(
            'tqdm', action ='install', DEVNULL=True, verbose =1)
        >>> subprocess_module_installation(
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
        
def smart_format(iter_obj): 
    """ Smart format iterable obj 
    :param iter_obj: iterable obj 
    
    :Example: 
        >>> from pycsamt.utils.func_utils import smart_format
        >>> smart_format(['model', 'iter', 'mesh', 'data'])
        ... 'model','iter','mesh' and 'data'
    """
    try: 
        iter(iter_obj) 
    except:  return f"{iter_obj}"
    
    iter_obj = [str(obj) for obj in iter_obj]
    if len(iter_obj) ==1: 
        str_litteral= ','.join([f"{i!r}" for i in iter_obj ])
    elif len(iter_obj)>1: 
        str_litteral = ','.join([f"{i!r}" for i in iter_obj[:-1]])
        str_litteral += f" and {iter_obj[-1]!r}"
    return str_litteral

def make_introspection(Obj , subObj): 
    """ Make introspection by using the attributes of instance created to 
    populate the new classes created.
    :param Obj: callable 
        New object to fully inherits of `subObject` attributes 
    :param subObj: Callable 
        Instance created.
    """
    # make introspection and set the all  attributes to self object.
    # if Obj attribute has the same name with subObj attribute, then 
    # Obj attributes get the priority.
    for key, value in  subObj.__dict__.items(): 
        if not hasattr(Obj, key) and key  != ''.join(['__', str(key), '__']):
            setattr(Obj, key, value)
            
def cpath (savepath=None , dpath= None): 
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
    if savepath is not None:
        try :
            if not os.path.isdir(savepath):
                os.mkdir(savepath)#  mode =0o666)
        except : pass 
    return savepath   

def show_quick_edi_stats(nedic , nedir, fmtl='~', lenl=77): 
    """ Format the Edi files and ckeck the number of edifiles
    successfully read. 
    :param nedic: number of input or collected edifiles 
    :param nedir: number of edifiles read sucessfully 
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
    print(mesg.format('EDI collected','=',  nedic, 'EDI success. read',
                      '=', nedir, 'Rate','=', round ((nedir/nedic) *100, 2),
                      2))
    print(fmtl * lenl )

def sPath (name_of_path:str):
    """ Savepath func. Create a path  with `name_of_path` 
    if path not exists.
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
def smart_format(iter_obj): 
    """ Smart format iterable obj 
    :param iter_obj: iterable obj 
    
    :Example: 
        >>> from watex.utils.func_utils import smart_format
        >>> smart_format(['.csv', '.json', '.xlsx', '.html', '.sql'])
        ... '.csv', '.json', '.xlsx', '.html' and '.sql'
    """
    str_litteral=''
    
    try: 
        iter(iter_obj) 
    except:  return f"{iter_obj}"
    
    iter_obj = [str(obj) for obj in iter_obj]
    if len(iter_obj) ==1: 
        str_litteral= ''.join([f"{i!r}" for i in iter_obj ])
    elif len(iter_obj)>1: 
        str_litteral = ','.join([f"{i!r}" for i in iter_obj[:-1]])
        str_litteral += f" and {iter_obj[-1]!r}"
    return str_litteral

def format_notes(text:str , cover_str:str='~', inline=70, **kws): 
    """ Format note 
    :param text: Text to be formated 
    :param cover_str: type of ``str`` to surround the text 
    :param inline: Nomber of character before going in liine 
    :param margin_space: Must be <1 and expressed in %. The empty distance 
                        between the first index to the inline text 
    :Example: 
        
        >>> from watex.utils import func_utils as func 
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
    
    

                
def interpol_scipy (x_value, y_value,x_new,
                    kind="linear",plot=False, fill="extrapolate"):
    
    """
    function to interpolate data 
    
    Parameters 
    ------------
        * x_value : np.ndarray 
                    value on array data : original absciss 
                    
        * y_value : np.ndarray 
                    value on array data : original coordinates (slope)
                    
        * x_new  : np.ndarray 
                    new value of absciss you want to interpolate data 
                    
        * kind  : str 
                projection kind : 
                    maybe : "linear", "cubic"
                    
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


def _set_depth_to_coeff(data, depth_column_index,coeff=1, depth_axis=0):
    
    """
    Parameters
    ----------
        * data : np.ndarray
            must be on array channel .
            
        * depth_column_index : int
            index of depth_column.
            
        * depth_axis : int, optional
            Precise kind of orientation of depth data(axis =0 or axis=1) 
            The *default* is 0.
            
        * coeff : float,
            the value you want to multiplie depth. 
            set depth to negative multiply by one. 
            The *default* is -1.

    Returns
    -------
        data : np.ndarray
            new data after set depth according to it value.
            
    :Example: 

        >>>  import numpy as np 
        >>>  np.random.seed(4)
        >>>  data=np.random.rand(4,3)
        >>>  data=data*(-1)
        >>>  print("data\n",data)
        >>>  data[:,1]=data[:,1]*(-1)
        >>>  data[data<0]
        >>>  print("data2\n",data)
    """
    
    if depth_axis==0:
        data[:,depth_column_index]=data[:,depth_column_index]*coeff
    if depth_axis==1:
        data[depth_column_index,:]=data[depth_column_index,:]*coeff  

    return data
            


def broke_array_to_(arrayData, keyIndex=0, broken_type="dict"):
    """
    broke data array into different value with their same key 

    Parameters
    ----------
        * arrayData :np.array
            data array .
            
        * keyIndex : int 
            index of column to create dict key 

    Returns
    -------
        dict 
           dico_brok ,dictionnary of array.
    """
    
    vcounts_temp,counts_temp=np.unique(arrayData[:,keyIndex], return_counts=True)
    vcounts_temp_max=vcounts_temp.max()

    dico_brok={}
    lis_brok=[]
    index=0
    deb=0
    for rowlines in arrayData :
        if rowlines[0] == vcounts_temp_max:
            value=arrayData[index:,::]
            if broken_type=="dict":
                dico_brok["{0}".format(rowlines[0])]=value
                break
            elif broken_type=="list":
                lis_brok.append(value)
                break
        elif rowlines[0] !=arrayData[index+1,keyIndex]:
            value=arrayData[deb:index+1,::]
            if broken_type=="dict":
                dico_brok["{0}".format(rowlines[0])]=value
            elif broken_type=="list":
                lis_brok.append(value)
            deb=index+1
        index=index+1
    if broken_type=="dict":
        return dico_brok
    elif broken_type=="list":
        return lis_brok
    



def dump_comma(input_car, max_value=2, carType='mixed'):
    """
    Parameters
    ----------
        * input_car : str,
            Input character.
        * max_value : int, optional
            The default is 2.
            
        * carType: str 
            Type of character , you want to entry
                 
    Returns
    -------
        Tuple of input character
            must be return tuple of float value, or string value
      
    .. note:: carType  may be as arguments parameters like ['value','val',"numeric",
              "num", "num","float","int"] or  for pure character like 
                ["car","character","ch","char","str", "mix", "mixed","merge","mer",
                "both","num&val","val&num&"]
                if not , can  not possible to convert to float or integer.
                the *defaut* is mixed 
                
    :Example: 
        
        >>> import numpy as np
        >>>  ss=dump_comma(input_car=",car,box", max_value=3, 
        ...      carType="str")
        >>>  print(ss)
        ... ('0', 'car', 'box')
    """
    
    
    
        # dump "," at the end of 
    flag=0
    
    if input_car[-1]==",":
        input_car=input_car[:-1]
    if input_car[0]==",":
        input_car="0,"+ input_car[1:]
        
    if carType.lower() in ['value','val',"numeric",
                           "num", "num","float","int"]:
        input_car=eval(input_car)
        
    elif carType.lower() in ["car","character","ch","char","str",
                             "mix", "mixed","merge","mer",
                             "both","num&val","val&num&"]:
        
        input_car=input_car.strip(",").split(",")
        flag=1
        

    if np.iterable(input_car)==False :
        inputlist=[input_car,0]

    elif np.iterable(input_car) is True :

        inputlist=list(input_car)

    input_car=inputlist[:max_value]

    if flag==1 :
        if len(inputlist)==1 :
            return(inputlist[0])
    
    return tuple(input_car)

 
                              
def _nonelist_checker(data, _checker=False ,
                      list_to_delete=['\n']):
    """
    Function to delete a special item on list in data.
    Any item you want to delete is acceptable as long as item is on a list.
    
    Parameters
    ----------
        * data : list
             container of list. Data must contain others list.
             the element to delete should be on list.
             
        *  _checker : bool, optional
              The default is False.
             
        * list_to_delete : TYPE, optional
                The default is ['\n'].

    Returns
    -------
        _occ : int
            number of occurence.
        num_turn : int
            number of turns to elimate the value.
        data : list
           data safeted exempt of the value we want to delete.
        
    :Example: 
        
        >>> import numpy as np 
        >>> listtest =[['DH_Hole', 'Thick01', 'Thick02', 'Thick03',
        ...   'Thick04', 'sample02', 'sample03'], 
        ...   ['sample04'], [], ['\n'], 
        ...  ['S01', '98.62776918', '204.7500461', '420.0266651'], ['prt'],
        ...  ['pup', 'pzs'],[], ['papate04', '\n'], 
        ...  ['S02', '174.4293956'], [], ['400.12', '974.8945704'],
        ...  ['pup', 'prt', 'pup', 'pzs', '', '\n'],
        ...  ['saple07'], [], '',  ['sample04'], ['\n'],
        ...  [''], [313.9043882], [''], [''], ['2'], [''], ['2'], [''], [''], ['\n'], 
        ...  [''], ['968.82'], [], [],[], [''],[ 0.36], [''], ['\n']]
        >>> ss=_nonelist_checker(data=listtest, _checker=True,
        ...                        list_to_delete=[''])
        >>> print(ss)
    """
    
    _occ,num_turn=0,0
    
    
    if _checker is False:
        _occ=0
        return _occ, num_turn, data 
    
   
    while _checker is True :
        if list_to_delete in data :
            for indix , elem_chker in enumerate(data):
                if list_to_delete == elem_chker :
                    _occ=_occ+1
                    del data[indix]
                    if data[-1]==list_to_delete:
                        _occ +=1
                        # data=data[:-1]
                        data.pop()
        elif list_to_delete not in data :
            _checker =False
        num_turn+=1

        
    return _occ,num_turn,data
        

                                       
def intell_index (datalist,assembly_dials =False):
    """
    function to search index to differency value to string element like 
    geological rocks and geologicals samples. It check that value are sorted
    in ascending order.

    Parameters
    ----------
        * datalist : list
                list of element : may contain value and rocks or sample .
        * assembly_dials : list, optional
                separate on two list : values and rocks or samples. 
                The default is False.

    Returns
    -------
        index: int
             index of breaking up.
        first_dial: list , 
           first sclice of value part 
        secund_dial: list , 
          second slice of rocks or sample part.
        assembly : list 
             list of first_dial and second_dial
    
    :Example: 
        
        >>> import numpy as np
        >>> listtest =[['DH_Hole', 'Thick01', 'Thick02', 'Thick03',
        ...           'Thick04','Rock01', 'Rock02', 'Rock03', 'Rock04'],
        ...           ['S01', '0.0', '98.62776918', '204.7500461','420.0266651','520', 'GRT', 
        ...            'ATRK', 'GRT', 'ROCK','GRANODIORITE'],
        ...           ['S02', '174.4293956', '313.9043882','974.8945704', 'GRT', 'ATRK', 'GRT']]
        >>> listtest2=[listtest[1][1:],listtest[2][1:]]
        >>> for ii in listtest2 :
        >>> op=intell_index(datalist=ii)
        >>> print("index:\n",op [0])
        >>> print('firstDials :\n',op [1])
        >>> print('secondDials:\n',op [2])
    """
    # assembly_dials=[]
    max_=0              # way to check whether values are in sort (ascending =True) order 
                        # because we go to deep (first value must be less than the next none)
    for ii, value in enumerate (datalist):
        try : 
            thick=float(value)
            if thick >= max_:
                max_=thick 
            else :
                _logger.warning(
                    "the input value {0} is less than the previous one."
                " Please enter value greater than {1}.".format(thick, max_) )
                warnings.warn(
                    "Value {1} must be greater than the previous value {0}."\
                    " Must change on your input data.".format(thick,max_))
        except :
            # pass
            indexi=ii
            break
        
    first_dial=datalist[:indexi]
    second_dial =datalist[indexi:]

    if assembly_dials:
        
        assembly_dials=[first_dial,second_dial]
        
        return indexi, assembly_dials
        
    return indexi, first_dial, second_dial

def _nonevalue_checker (list_of_value, value_to_delete=None):
    """
    Function similar to _nonelist_checker. the function deletes the specific 
    value on the list whatever the number of repetition of value_to_delete.
    The difference with none list checker is value to delete 
    may not necessary be on list.
    
    Parameters
    ----------
        * list_of_value : list
                list to check.
        * value_to_delete : TYPE, optional
            specific value to delete. The default is ''.

    Returns
    -------
        list
            list_of_value , safe list without the deleted value .
    
    :Example: 
        
        >>> import numpy as np 
        >>> test=['DH_Hole (ID)', 'DH_East', 'DH_North',
        ...          'DH_Dip', 'Elevation ', 'DH_Azimuth', 
        ...            'DH_Top', 'DH_Bottom', 'DH_PlanDepth', 'DH_Decr', 
        ...            'Mask', '', '', '', '']
        >>> test0= _nonevalue_checker (list_of_value=test)
        >>> print(test0)
    """
    if value_to_delete is None :
        value_to_delete  =''
    
    if type (list_of_value) is not list :
        list_of_value=list(list_of_value)
        
    start_point=1
    while start_point ==1 :
        if value_to_delete in list_of_value :
            [list_of_value.pop(ii) for  ii, elm in\
             enumerate (list_of_value) if elm==value_to_delete]
        elif value_to_delete not in list_of_value :
            start_point=0 # not necessary , just for secure the loop. 
            break           # be sure one case or onother , it will break
    return list_of_value 
        
def _strip_item(item_to_clean, item=None, multi_space=12):
    """
    Function to strip item around string values.  if the item to clean is None or 
    item-to clean is "''", function will return None value

    Parameters
    ----------
        * item_to_clean : list or np.ndarray of string 
                 List to strip item.
        * cleaner : str , optional
                item to clean , it may change according the use. The default is ''.
        * multi_space : int, optional
                degree of repetition may find around the item. The default is 12.

    Returns
    -------
        list or ndarray
            item_to_clean , cleaned item 
            
    :Example: 
        
     >>> import numpy as np
     >>> new_data=_strip_item (item_to_clean=np.array(
         ['      ss_data','    pati   ']))
     >>>  print(np.array(['      ss_data','    pati   ']))
     ... print(new_data)

    """
    if item==None :item = ' '
    
    cleaner =[(''+ ii*'{0}'.format(item)) for ii in range(multi_space)]
    
    if type(item_to_clean ) != list :#or type(item_to_clean ) !=np.ndarray:
        if type(item_to_clean ) !=np.ndarray:
            item_to_clean=[item_to_clean]
    if item_to_clean in cleaner or item_to_clean ==['']:
        warnings.warn ('No data found in <item_to_clean :{}> We gonna return None.')
        return None 

  
    try : 
        multi_space=int(multi_space)
    except : 
        raise TypeError('argument <multplier> must be'\
                        ' an integer not {0}'.format(type(multi_space)))
    
    for jj, ss in enumerate(item_to_clean) : 
        for space in cleaner:
            if space in ss :
                new_ss=ss.strip(space)
                item_to_clean[jj]=new_ss
    
    return item_to_clean
    
def _cross_eraser (data , to_del, deep_cleaner =False):
    """
    Function to delete some item present in another list. It may cheCk deeper 

    Parameters
    ----------
        * data : list
                Main data user want to filter.
        * to_del : list
                list of item you want to delete present on the main data.
        * deep_cleaner : bool, optional
                Way to deeply check. Sometimes the values are uncleaned and 
            capitalizeed . this way must not find their safety correspondace 
            then the algorth must clean item and to match all at
            the same time before eraisng.
            The *default* is False.

    Returns
    -------
        list
         data , list erased.

    :Example: 
        
        >>> data =['Z.mwgt','Z.pwgt','Freq',' Tx.Amp','E.mag','   E.phz',
        ...          '   B.mag','   B.phz','   Z.mag', '   Zphz  ']
        >>> data2=['   B.phz','   Z.mag',]
        ...     remain_data =cross_eraser(data=data, to_del=data2, 
        ...                              deep_cleaner=True)
        >>> print(remain_data)
    """
    
    data , to_del=_strip_item(item_to_clean=data), _strip_item(
        item_to_clean=to_del)
    if deep_cleaner : 
        data, to_del =[ii.lower() for ii in data], [jj.lower() 
                                                    for jj in to_del]
        
    for index, item in enumerate(data): 
        while item in to_del :
            del data[index]
            if index==len(data)-1 :
                break

    return data 

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
        str 
            char , new_char without the removed word .
        
    :Example: 
        
        >>> from pycsamt.utils  import func_utils as func
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
        
        >>> from pycsamt.utils  import func_utils as func
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
            raise 'None <"="> found on this item<{0}> of '
            ' the edilines list. list can not'\
            ' be parsed.Please put egal between key and value '.format(
                edilines[ii])
    
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
        
        >>> from watex.func_utils import display_infos
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
    
    :param f: translation file to parse 
    :param delimiter: str, delimiter
    
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
    
    :param csv_fn: data collected in csv format 
    :param pf: parser file 
    :param destfile: str,  Destination file, outputfile 
    :param savepath: [Path-Like object, save data to a path 
                      
    :Example: 
        # to execute this script, we need to import the two modules below
        >>> import os 
        >>> import csv 
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
    

    
    
    
    
    
    
    

    
        