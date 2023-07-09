# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import ( 
    annotations , 
    print_function 
    )
import os
import copy
import shutil 
from six.moves import urllib 

import numpy as np 
import pandas as pd 

from .._typing import ( 
    Any, 
    List, 
    NDArray, 
    DataFrame, 
    )
from .funcutils import ( 
    is_iterable, 
    ellipsis2false,
    smart_format,
    sPath
    )
from ._dependency import ( 
    import_optional_dependency 
    )


def array2hdf5 (
    filename: str, /, 
    arr: NDArray=None , 
    dataname: str='data',  
    task: str='store', 
    as_frame: bool =..., 
    columns: List[str, ...]=None, 
)-> NDArray | DataFrame: 
    """ Load or write array to hdf5
    
    Parameters 
    -----------
    arr: Arraylike ( m_samples, n_features) 
      Data to load or write 
    filename: str, 
      Hdf5 disk file name whether to write or to load 
    task: str, {"store", "load", default='store'}
       Action to perform. user can use ['write'|'store'] interchnageably. Both 
       does the same task. 
    as_frame: bool, default=False 
       Concert loaded array to data frame. `Columns` can be supplied 
       to construct the datafame. 
    columns: List, Optional 
       Columns used to construct the dataframe. When its given, it must be 
       consistent with the shape of the `arr` along axis 1 
       
    Returns 
    ---------
    None| data: ArrayLike or pd.DataFrame 
    
    Examples 
    ----------
    >>> import numpy as np 
    >>> from watex.utils.baseutils import array2hdf5
    >>> data = np.random.randn (100, 27 ) 
    >>> array2hdf5 ('test.h5', data   )
    >>> load_data = array2hdf5 ( 'test.h5', data, task ='load')
    >>> load_data.shape 
    Out[177]: (100, 27)
    """
    import_optional_dependency("h5py")
    import h5py 
    
    arr = is_iterable( arr, exclude_string =True, transform =True )
    act = copy.deepcopy(task)
    task = str(task).lower().strip() 
    
    if task in ("write", "store"): 
        task ='store'
    assert task in {"store", "load"}, ("Expects ['store'|'load'] as task."
                                         f" Got {act!r}")
    # for consistency 
    arr = np.array ( arr )
    h5fname = str(filename).replace ('.h5', '')
    if task =='store': 
        if arr is None: 
            raise TypeError ("Array cannot be None when the task"
                             " consists to write a file.")
        with h5py.File(h5fname + '.h5', 'w') as hf:
            hf.create_dataset(dataname,  data=arr)
            
    elif task=='load': 
        with h5py.File(h5fname +".h5", 'r') as hf:
            data = hf[dataname][:]
            
        if  ellipsis2false( as_frame )[0]: 
            data = pd.DataFrame ( data , columns = columns )
            
    return data if task=='load' else None 
   
def lowertify (*values, strip = True, return_origin: bool =... ): 
    """ Strip and convert value to lowercase. 
    
    :param value: str , value to convert 
    :return: value in lowercase and original value. 
    
    :Example: 
        >>> from watex.utils.baseutils import lowertify 
        >>> lowertify ( 'KIND')
        Out[19]: ('kind',)
        >>> lowertify ( "KIND", return_origin =True )
        Out[20]: (('kind', 'KIND'),)
        >>> lowertify ( "args1", 120 , 'ArG3') 
        Out[21]: ('args1', '120', 'arg3')
        >>> lowertify ( "args1", 120 , 'ArG3', return_origin =True ) 
        Out[22]: (('args1', 'args1'), ('120', 120), ('arg3', 'ArG3'))
        >>> (kind, kind0) , ( task, task0 ) = lowertify(
            "KIND", "task ", return_origin =True )
        >>> kind, kind0, task, task0 
        Out[23]: ('kind', 'KIND', 'task', 'task ')
        """
    raw_values = copy.deepcopy(values ) 
    values = [ str(val).lower().strip() if strip else str(val).lower() 
              for val in values]

    return tuple (zip ( values, raw_values)) if ellipsis2false (
        return_origin)[0]  else tuple (values)

def save_or_load(
    fname:str, /,
    arr: NDArray=None,  
    task: str='save', 
    format: str='.txt', 
    compressed: bool=...,  
    comments: str="#",
    delimiter: str=None, 
    **kws 
): 
    """Save or load Numpy array. 
    
    Parameters 
    -----------
    fname: file, str, or pathlib.Path
       File or filename to which the data is saved. 
       - >.npy , .npz: If file is a file-object, then the filename is unchanged. 
       If file is a string or Path, a .npy extension will be appended to the 
       filename if it does not already have one. 
       - >.txt: If the filename ends in .gz, the file is automatically saved in 
       compressed gzip format. loadtxt understands gzipped files transparently.
       
    arr: 1D or 2D array_like
      Data to be saved to a text, npy or npz file.
      
    task: str {"load", "save"}
      Action to perform. "Save" for storing file into the format 
      ".txt", "npy", ".npz". "load" for loading the data from storing files. 
      
    format: str {".txt", ".npy", ".npz"}
       The kind of format to save and load.  Note that when loading the 
       compressed data saved into `npz` format, it does not return 
       systematically the array rather than `np.lib.npyio.NpzFile` files. 
       Use either `files` attributes to get the list of registered files 
       or `f` attribute dot the data name to get the loaded data set. 

    compressed: bool, default=False 
       Compressed the file especially when file format is set to `.npz`. 

    comments: str or sequence of str or None, default='#'
       The characters or list of characters used to indicate the start 
       of a comment. None implies no comments. For backwards compatibility, 
       byte strings will be decoded as 'latin1'. This is useful when `fname`
       is in `txt` format. 
      
     delimiter: str,  optional
        The character used to separate the values. For backwards compatibility, 
        byte strings will be decoded as 'latin1'. The default is whitespace.
        
    kws: np.save ,np.savetext,  np.load , np.loadtxt 
       Additional keywords arguments for saving and loading data. 
       
    Return 
    ------
    None| data: ArrayLike 
    
    Examples 
    ----------
    >>> import numpy as np 
    >>> from watex.utils.baseutils import save_or_load 
    >>> data = np.random.randn (2, 7)
    >>> # save to txt 
    >>> save_or_load ( "test.txt" , data)
    >>> save_or_load ( "test",  data, format='.npy')
    >>> save_or_load ( "test",  data, format='.npz')
    >>> save_or_load ( "test_compressed",  data, format='.npz', compressed=True )
    >>> # load files 
    >>> save_or_load ( "test.txt", task ='load')
    Out[36]: 
    array([[ 0.69265852,  0.67829574,  2.09023489, -2.34162127,  0.48689125,
            -0.04790965,  1.36510779],
           [-1.38349568,  0.63050939,  0.81771051,  0.55093818, -0.43066737,
            -0.59276321, -0.80709192]])
    >>> save_or_load ( "test.npy", task ='load')
    Out[39]: array([-2.34162127,  0.55093818])
    >>> save_or_load ( "test.npz", task ='load')
    <numpy.lib.npyio.NpzFile at 0x1b0821870a0>
    >>> npzo = save_or_load ( "test.npz", task ='load')
    >>> npzo.files
    Out[44]: ['arr_0']
    >>> npzo.f.arr_0
    Out[45]: 
    array([[ 0.69265852,  0.67829574,  2.09023489, -2.34162127,  0.48689125,
            -0.04790965,  1.36510779],
           [-1.38349568,  0.63050939,  0.81771051,  0.55093818, -0.43066737,
            -0.59276321, -0.80709192]])
    >>> save_or_load ( "test_compressed.npz", task ='load')
    ...
    """
    r_formats = {"npy", "txt", "npz"}
   
    (kind, kind0), ( task, task0 ) = lowertify(
        format, task, return_origin =True )
    
    assert  kind.replace ('.', '') in r_formats, (
        f"File format expects {smart_format(r_formats, 'or')}. Got {kind0!r}")
    kind = '.' + kind.replace ('.', '')
    assert task in {'save', 'load'}, ( 
        "Wrong task {task0!r}. Valid tasks are 'save' or 'load'") 
    
    save= {'.txt': np.savetxt, '.npy':np.save,  
           ".npz": np.savez_compressed if ellipsis2false(
               compressed)[0] else np.savez 
           }
    if task =='save': 
        arr = np.array (is_iterable( arr, exclude_string= True, 
                                    transform =True ))
        save.get(kind) (fname, arr, **kws )
        
    elif task =='load': 
         ext = os.path.splitext(fname)[1].lower() 
         if ext not in (".txt", '.npy', '.npz', '.gz'): 
             raise ValueError ("Unrecognized file format {ext!r}."
                               " Expect '.txt', '.npy', '.gz' or '.npz'")
         if ext in ('.txt', '.gz'): 
            arr = np.loadtxt ( fname , comments= comments, 
                              delimiter= delimiter,   **kws ) 
         else : 
            arr = np.load(fname,**kws )
         
    return arr if task=='load' else None 
 
#XXX TODO      
def request_data (
    url:str, /, 
    task: str='get',
    data: Any=None, 
    as_json: bool=..., 
    as_text: bool = ..., 
    stream: bool=..., 
    raise_status: bool=..., 
    save2file: bool=..., 
    filename:str =None, 
    **kws
): 
    """ Fetch remotely data
 
    Request data remotely 
    https://docs.python-requests.org/en/latest/user/quickstart/#raw-response-content
    
    
    r = requests.get('https://api.github.com/user', auth=('user', 'pass'))
    r.status_code
    200
    r.headers['content-type']
    'application/json; charset=utf8'
    r.encoding
    'utf-8'
    r.text
    '{"type":"User"...'
    r.json()
    {'private_gists': 419, 'total_private_repos': 77, ...}
    
    """
    import_optional_dependency('requests' ) 
    import requests 
    
    as_text, as_json, stream, raise_status, save2file = ellipsis2false(
        as_text, as_json,  stream, raise_status , save2file)
    
    if task=='post': 
        r = requests.post(url, data =data , **kws)
    else: r = requests.get(url, stream = stream , **kws)
    
    if save2file and stream: 
        with open(filename, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)
    if raise_status: 
        r.raise_for_status() 
        
    return r.text if as_text else ( r.json () if as_json else r )

def get_remote_data(
    rfile:str, /,  
    savepath: str=None, 
    raise_exception: bool =True
): 
    """ Try to retrieve data from remote.
    
    Parameters 
    -------------
    rfile: str or PathLike-object 
       Full path to the remote file. It can be the path to the repository 
       root toward the file name. For instance, to retrieve the file 
       ``'AGSO.csv'`` which is located in ``watex/etc/`` directory then the 
       full path should be ``'watex/etc/AGSO.csv'``
        
    savepath: str, optional 
       Full path to place where to downloaded files should be located. 
       If ``None`` data is saved to the current directory.
     
    raise_exception: bool, default=True 
      raise exception if connection failed. 
      
    Returns 
    ----------
    status: bool, 
      ``False`` for failure and ``True`` otherwise i.e. successfully 
       downloaded. 
       
    """
    connect_reason ="""\
    ConnectionRefusedError: No connection could  be made because the target 
    machine actively refused it.There are some possible reasons for that:
     1. Server is not running as well. Hence it won't listen to that port. 
         If it's a service you may want to restart the service.
     2. Server is running but that port is blocked by Windows Firewall
         or other firewall. You can enable the program to go through 
         firewall in the inbound list.
    3. there is a security program on your PC, i.e a Internet Security 
        or Antivirus that blocks several ports on your PC.
    """  
    #git_repo , git_root= AGSO_PROPERTIES['GIT_REPO'], AGSO_PROPERTIES['GIT_ROOT']
    # usebar bar progression
    print(f"---> Please wait while fetching {rfile!r}...")
    try: import_optional_dependency ("tqdm")
    except:pbar = range(3) 
    else: 
        import tqdm  
        data =os.path.splitext( os.path.basename(rfile))[0]
        pbar = tqdm.tqdm (total=3, ascii=True, 
                          desc =f'get-{os.path.basename(rfile)}', 
                          ncols =97
                          )
    status=False
    root, rfile  = os.path.dirname(rfile), os.path.basename(rfile)
    for k in range(3):
        try :
            urllib.request.urlretrieve(root,  rfile )
        except: 
            try :
                with urllib.request.urlopen(root) as response:
                    with open( rfile,'wb') as out_file:
                        data = response.read() # a `bytes` object
                        out_file.write(data)
            except TimeoutError: 
                if k ==2: 
                    print("---> Established connection failed because"
                       "connected host has failed to respond.")
            except:pass 
        else : 
            status=True
            break
        try: pbar.update (k+1)
        except: pass 
    
    if status: 
        try: 
            pbar.update (3)
            pbar.close ()
        except:pass
        # print(f"\n---> Downloading {rfile!r} was successfully done.")
    else: 
        print(f"\n---> Failed to download {rfile!r}.")
    # now move the file to the right place and create path if dir not exists
    if savepath is not None: 
        if not os.path.isdir(savepath): 
            sPath (savepath)
        shutil.move(os.path.realpath(rfile), savepath )
        
    if not status:
        if raise_exception: 
            raise ConnectionRefusedError(connect_reason.replace (
                "ConnectionRefusedError:", "") )
        else: print(connect_reason )
    
    return status

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    