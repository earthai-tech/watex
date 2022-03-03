# -*- coding: utf-8 -*-
#       Author: Kouadio K.Laurent<etanoyau@gmail.con>
#       Create:on Tue Oct 12 15:37:59 2021
#       Licence: MIT

"""
..warnings:: This module is a core use to retreive data from local,  
    git and zenodo record. Modified this module  presume to enhance the way  
    the codes are written and you know what you are doing. Making a copy 
    before modifying this script is recommended.

"""
import os 
import time
import sys 
import subprocess 
import concurrent.futures
import shutil  
import zipfile
from six.moves import urllib 

from ..utils.ml_utils import (fetchSingleTGZData, 
                                  subprocess_module_installation)
from ..utils._watexlog import watexlog
from ..utils.exceptions import WATexExtractionError 

__logger = watexlog().get_watex_logger(__name__)

LOCAL_DIR = 'data/geo_fdata'
f__= os.path.join(LOCAL_DIR , 'main.bagciv.data.csv')
ZENODO_RECORD_ID_OR_DOI = '10.5281/zenodo.5571534'
GIT_ROOT = 'https://raw.githubusercontent.com/WEgeophysics/watex/master/' 
GIT_REPO= 'https://github.com/WEgeophysics/watex'
# from Zenodo: 'https://zenodo.org/record/5560937#.YWQBOnzithE'
DATA_PATH = 'data/__tar.tgz' 
TGZ_FILENAME = '/fmain.bagciv.data.tar.gz'
CSV_FILENAME = '/__tar.tgz_files__/___fmain.bagciv.data.csv'
DATA_URL = GIT_ROOT  + DATA_PATH  + TGZ_FILENAME

__all__=['fetchDataFromLocalandWeb']

def fetchDataFromLocalandWeb(f=f__): 
    """Retreive Bagoue dataset from Github repository or zenodo record. 
    
    It will take a while when fetching data for the first time outsite of 
    this repository. Since cloning the repository come with examples dataset  
    located to its appropriate directory. It's probably a rare to fectch using 
    internet unless dataset  as well as the tarfile are  deleted from its
    located directory.
    
    :param f: str of Path-Like obj 
        main dataset to fetch.
        
    :Example:
        
        >>> import watex.datasets.property as DSProps
        >>> DSProps.fetchDataFromLocalandWeb()
            ---> Please wait while decompressing 'fmain.bagciv.data.tar.gz' file... 
            ---> Decompressing 'fmain.bagciv.data.tar.gz' file failed ! 
            ---> Please wait while fetching data from 'https://github.com/WEgeophysics/watex'...
            ... Please wait for the second attempt!
            ... We try for the last attempt, please wait once again ...
            ---> Trying the alternative way using <blob/master>...
            ---> Forcing downloading by changing the root instead...
            ---> Downloading from 'https://github.com/WEgeophysics/watex' failed!
            ---> Please wait while the record <10.5281/zenodo.5571534> is downloading...
            ---> Record <10.5281/zenodo.5571534> was sucessffuly downloaded...
            ---> Find the record <10.5281/zenodo.5571534=BagoueCIV__dataset__main.zip> in 'data/geo_fdata'.
            ---> Please wait while unziping the record...
            ---> 'main.bagciv.data.csv' was successfully decompressed  and saved to 'data/geo_fdata'
            ---> Dataset='main.bagciv.data.csv' was successfully retreived.
            ---> Extraction of `main.bagciv.data.csv` was successfully done!
    """
    mess =f"Fetching {os.path.basename(f)!r} from "
    IMP_TQDM =False 
    try : 
        import tqdm
        # from tqdm.notebook  import trange
    except:# Install bar progression
        try :
            IMP_TQDM= subprocess_module_installation('tqdm')
        except: 
            with concurrent.futures.ThreadPoolExecutor() as executor: 
                modules =[ 'notebook', 'ipywidgets', 'tqdm']
                try : 
                    _RES=list(executor.map(
                        subprocess_module_installation, modules))
                except : 
                    results = [executor.submit(
                        subprocess_module_installation, args =[mod, True])
                                               for mod in modules]
                    _RES =[f.result() for f in 
                           concurrent.futures.as_completed(results)] 
                    
            if len(set(_RES)) ==1: 
                # mean all modules were executed successffuly 
                IMP_TQDM = _RES[0]
            if IMP_TQDM: 
                # from tqdm.notebook  import trange 
                import tqdm 
                
    pbar =tqdm.tqdm(unit ='B', total= 1, ascii=False,
                    desc ='WEgeophysics-WATex', ncols =77)
    for _ in range(1):
        total , start =0, time.perf_counter() 
        if not os.path.isdir(LOCAL_DIR ):
            os.makedirs(LOCAL_DIR )
        is_f_file = _fromlocal(f)
        if not is_f_file: 
            __logger.info(f" File {os.path.basename(f)!r} Does not exist "
                          "in local directory.")
            is_f_file =  _fromgithub()
            if not is_f_file :
                __logger.info(mess + 'Github failed! We try Zenodo record.')
                is_f_file = _fromzenodo()
    
        if not is_f_file : 
            __logger.info(mess + 'Zenodo failed!')
            __logger.info (f"Unable to fetch {os.path.basename(f)!r} from Web")
            end = time.perf_counter() 
            time.sleep(abs(start -end))
            pbar.update(total)
            return 
        __logger.info(f"{os.path.basename(f)!r} was successfully loaded.")
        end = time.perf_counter() 
        time.sleep(abs(start -end))
        
        if is_f_file: 
            total =1
            pbar.update(total)
            
    return f

def _fromlocal (f=f__): 
    """ check whether the local file exists and return file name."""
    is_file =os.path.isfile(f)
    if not is_file :
        try: 
            __logger.info("Fetching data from"
                          f" {TGZ_FILENAME.replace('/', '')}")
            print("\n---> Please wait while decompressing"
                  f" {TGZ_FILENAME.replace('/', '')!r} file... ")
            
            f0=fetchSingleTGZData(DATA_PATH +TGZ_FILENAME, 
                               rename_outfile='main.bagciv.data.csv')
            
        except : 
            __logger.info(f"Fetching  {TGZ_FILENAME.replace('/', '')!r} failed")
            print("---> Decompressing"
                  f" {TGZ_FILENAME.replace('/', '')!r} file failed ! ")
            return False 
        else : 
            print(f"---> Decompressed  {TGZ_FILENAME.replace('/', '')!r}"
                  " was sucessfully done!")
            if os.path.isfile (f0): return f0
    return f 

def _fromgithub( f=f__, root = GIT_ROOT):
    """ Get file from github and if file exists create your local directory  
        and save file."""
    # make a request
    if not os.path.isdir(LOCAL_DIR): 
        os.makedirs(LOCAL_DIR)
    success =False 
    #'https://raw.githubusercontent.com/WEgeophysics/watex/master/data/geo_fdata/main.bagciv.data.csv'
    rootf = os.path.join(GIT_ROOT, f)
    atp =[f"---> Please wait while fetching data from {GIT_REPO!r}...", 
          '... Please wait for the second attempt!',
          '... We try for the last attempt, please wait once again ...']
    for i in range(3): 
        try : 
            # first attemptts to 03 
            print(atp[i])
            urllib.request.urlretrieve(rootf, f)
        except TimeoutError: 
            if i ==2:
                print("---> Established connection failed because connected"
                      " host has failed to respond.")
            success =False 
        except:success =False 
        else : success=True 
        if success:
            break 
    if not success:
        # CHANGEGIT Root 
        try:
            print("---> Trying the alternative way using <blob/master>...")
            rootf0= 'https://github.com/WEgeophysics/watex/blob/master/' +f 
            urllib.request.urlretrieve(rootf0, f)
        except :success =False 
        else:success =True 
    if not success: 
        print("---> Force downloading by changing the root instead...")
        #'https://github.com/WEgeophysics/watex/blob/master/data/geo_fdata/main.bagciv.data.csv'
        #second attempts 
        try : 
            with urllib.request.urlopen(rootf) as testfile, open(f, 'w') as fs:
                    fs.write(testfile.read().decode())
        except : 
            # third attempts
            try:
                import requests 
                response = requests.get(rootf)
                with open(os.path.join(LOCAL_DIR,
                                       os.path.basename(f)), 'wb') as fs:
                    fs.write(response.content)
            except: 
                success=False
        else : 
            print(f"---> Downloading from {GIT_REPO!r} was successfully done!")
            success =True
            
    if not success: 
        print(f"---> Failed to download the dataset from {GIT_REPO!r} !")
        
        return False    
    if success :
        # assume the data is locate in current directory
        # then move to the right place in Local dir 
        if os.path.isfile('main.bagciv.data.csv'):
            move_file('main.bagciv.data.csv', LOCAL_DIR)
        
    print("---> Fetching `main.bagciv.data.csv`from {GIT_REPO!r}"
          " was successfully done!") 
    return f 


def _fromzenodo( doi = ZENODO_RECORD_ID_OR_DOI, path = LOCAL_DIR): 
    """Fetch data from zenodo records with ``doi`` and ``path``
    :param doi: Zenodo get obj 
        Record of zenodo database. see https://zenodo.org/
    :param path: Path to stored the record archived.
    
    :returns: File or record.
        Here fetch the bagoue dataset.
        
    :Example:
        >>> import watex.datasets.property as  DSProps
        >>> DSProps._fromzenodo()
    """
    if not os.path.isdir(path): 
        os.makedirs(path)
    success_import=False     
    try:
        import zenodo_get
    except: 
        # this will take a while if the connection is low. Please be patient.
        try: 
            print("---> Zenodo_get is installing. Please wait ...")
            subprocess_module_installation('zenodo_get')
            watexlog.get_watex_logger().info(
                "Intallation of `zenodo_get` was successfully done!") 
            success_import=True
            print("---> Intallation of `zenodo_get` was successfully done!")
        except : 
            # Connection problem may happens. 
            print('---> Zenodo_get installation failed!')
            __logger.info("Failed to force installation of `zenodo_get`")
            
    else: success_import=True 

    if not success_import: 
        raise ConnectionError("Unable to retrieve data from record= "
                              f"<{ZENODO_RECORD_ID_OR_DOI!r}.")
    # if zenodo_get is already installed Then used to 
    # wownloaed the record by calling the subprocess methods
    __logger.info(" `zenodo_get` package already installed!") 
        
    print(f"---> Please wait while the record <{ZENODO_RECORD_ID_OR_DOI}>"
          " is downloading...")
    try :
        subprocess.check_call([sys.executable, '-m', 'zenodo_get', doi])
    except: 
        raise ConnectionError (
            f"CalledProcessError: <{ZENODO_RECORD_ID_OR_DOI}> returned "
            "non-zero exit status 1. Please check your internet!")
        
    print(f"---> Record <{ZENODO_RECORD_ID_OR_DOI}>"
              " was sucessffuly downloaded...")
    if not os.path.isdir(path): 
        os.makedirs(path)
        
    # check whether Archive file is '.rar' or '.zip' 
    is_zipORrar =os.path.isfile ('BagoueCIV__dataset__main.rar')
    if is_zipORrar : ziprar_file = 'BagoueCIV__dataset__main.rar'
    else: ziprar_file = 'BagoueCIV__dataset__main.zip'
    
    # For consistency add curent work directory and move zip_rar file to 
    # the path =LOCAL_DIR and also move the md5sums file.
    move_file(os.path.join(os.getcwd(),ziprar_file), path)
    
    if os.path.isfile(os.path.join(os.getcwd(), 'md5sums.txt')):
        move_file(os.path.join(os.getcwd(), 'md5sums.txt'), path)
    
    print(f"---> Find the record <{ZENODO_RECORD_ID_OR_DOI}={ziprar_file}>"
           f" in {path!r}.")
    print("---> Please wait while "
          f"{'unziping' if not is_zipORrar else 'unraring'}"
          " the record...")
    #Now unzip file in the LOCAL DIR then move the file to 
    # it right place and rename it. 
    f0=unZipFileFetchedFromZenodo(path, zip_file =ziprar_file )
    
    try : 
        # if file exists then remove the archive 
        os.remove(os.path.join(LOCAL_DIR, ziprar_file))
    except :  pass 

    return f0
    
def unZipFileFetchedFromZenodo(zipdir =LOCAL_DIR, 
                               zip_file ='BagoueCIV__dataset__main.rar'):
    """ Unzip or Unrar the archived file and shift from  the local 
    directory created if not exits. """
    zipORrar_ex = zip_file.replace('BagoueCIV__dataset__main', '')
    zip_file=zip_file.replace(zipORrar_ex, '')
    # file is in zip #'/__tar.tgz_files__/___fmain.bagciv.data.csv'
    raw_location = zip_file + CSV_FILENAME 

    if zipORrar_ex=='.zip':
        try : 
            # CSV_FILENAME[1:]= '__tar.tgz_files__/___fmain.bagciv.data.csv',
            zip_location= os.path.join(zipdir, zip_file +'.zip') 
            fetchSingleZIPData(zip_file= zip_location, zipdir = zipdir , 
                               file_to_extract=CSV_FILENAME[1:],
                               savepath=zipdir, 
                               rename_outfile='main.bagciv.data.csv' )
        except : 
            raise OSError(f"Unzip <{zip_file}+'.zip'> failed."
                          'Please try again.')
 
    elif zipORrar_ex=='.rar':
        fetchSingleRARData(zip_file = zip_file, file_to_extract= raw_location, 
                   zipdir =zipdir )
        
    if os.path.isfile (zipdir + '/___fmain.bagciv.data.csv'): 
        os.rename(zipdir + '/___fmain.bagciv.data.csv',
                  zipdir + '/main.bagciv.data.csv')

    # Ascertain the file
    is_f_file = _fromlocal(zipdir + '/main.bagciv.data.csv')
    if is_f_file ==zipdir + '/main.bagciv.data.csv':
        print("---> Extraction of `main.bagciv.data.csv` "
              "was successfully done!")
        
    return is_f_file 


def fetchSingleRARData(zip_file, member_to_extract, zipdir ):
    """ RAR archived file domwloading process."""
    
    from watex.utils.exceptions import WATexExtractionError 
    rarmsg = ["--> Please wait while using `rarfile` module to "
              f"<{zip_file}> decompressing...", 
              "--> Please wait while using `unrar` module to "
              f"<{zip_file}> decompressing..."]
    
    for i, name in enumerate('rarfile', 'unrar'):
        installation_succeeded =False 
        try :
            if i==0: import rarfile
            elif i==1:  from unrar import rarfile
        except : 
            try:
                print(f"---> {name} is installing. Please wait ...")
                subprocess.check_call([sys.executable, '-m', 'pip', 
                                       'install',name])
                reqs = subprocess.check_output([sys.executable,'m', 'pip',
                                                'freeze'])
                [r.decode().split('==')[0] for r in reqs.split()]
                __logger.info(f"Intallation of {name!r} was successfully done!") 
                print(f"---> Installing of {name!r} is sucessfully done!")
            except : 
                print("--> Failed to install {name!r} module !")
                if name =='unrar': 
                    print("---> Couldn't find path to unrar library. Please refer"
                          " to https://pypi.org/project/unrar/ and download the "
                          "UnRAR library. src: http://www.rarlab.com/rar/unrarsrc-5.2.6.tar.gz "
                          "or  src(Window): (http://www.rarlab.com/rar/UnRARDLL.exe)."
                          )
                    raise  WATexExtractionError (
                       "Failed to install UnrarLibrary!") 
                continue 
            else :
                installation_succeeded=True 
    
        if installation_succeeded : 
            print(f"---> Please wait while `<{zip_file+'.rar'}="
                  "main.bagciv.data.csv>`is unraring...")
        # rarfile.RarFile.(os.path.join(zipdir, zip_file +'.rar'))
        __logger.info("Extract {os.path.basename(CSV_FILENAME)!r}"
                      " from {zip_file + '.rar'} file.")
        #--------------------------work on the rar extraction since -----------
        # rar can not extract larger file excceed fo 50
        # we are working to find the way to automatically decompressed rarfile.
        # and keep it to the local directory.
        print(rarmsg[i])
        decompress_succeed =False 
        try : 
            with rarfile.RarFile(os.path.join(zipdir,
                                              zip_file +'.rar'))as rar_ref:
                rar_ref.extract(member=member_to_extract, path = zipdir)
        except :
            print("--> Failed the read enough data: req=33345 got>=52 files.")
            import warnings
            warnings.warn("Minimal Rar version needed for decompressing. "
                "As (major*10 + minor), so 2.9 is 29.RAR3: 10, 20, 29"
                "RAR5 does not have such field in archive, it’s simply"
                  " set to 50."
                )
            continue 
        else : decompress_succeed=True 
        
        if decompress_succeed:
            break 
        
    if not decompress_succeed:    
    
        print(f"---> Please unrar the <{zip_file!r}> with an appropriate !"
              " software. Failed to read enough data more than 50. ")      
        raise  WATexExtractionError (
            "Failed the read enough data: req=33345 got>=52 files.")
     
    # rarfile.RarFile().extract(member=raw_location, path = zipdir)
    #----------------------------------------------------------------------
    if decompress_succeed:
        print(f"---> Unraring the `{zip_file}=main.bagciv.data.csv`"
          "was successfully done.")
        
def fetchSingleZIPData(zip_file, zipdir, **zip_kws ): 
    """ Find only the archived zip file and save to the current directory.
    
    Parameters 
    -----------
    zip_file: str or Path-like obj 
        Name of archived zip file
    zipdir : str or Path-like obj 
        Directory where `zip_file` is located. 
        
    Examples
    --------
    >>> from watex.datasets.property import fetchSingleZIPData
    >>> fetchSingleZIPData(zip_file= zip_file, zipdir = zipdir, 
         file_to_extract='__tar.tgz_files__/___fmain.bagciv.data.csv',
        savepath=save_zip_file, rename_outfile='main.bagciv.data.csv')
    """
    
    is_zip_file = os.path.isfile(zip_file)
    if not is_zip_file: 
        raise FileNotFoundError(f"{os.path.basename(zip_file)!r} is wrong file!"
                                " Please provide the right file.")
    #ZipFile.extractall(path=None, members=None, pwd=None)
    # path: location where zip file needs to be extracted; if not 
    #     provided, it will extract the contents in the current
    #     directory.
    # members: list of files to be extracted. It will extract all 
    #     the files in the zip if this argument is not provided.
    # pwd: If the zip file is encrypted, then pass the password in
    #     this argument default is None.
    # remove the first '/'--> 
    if not os.path.isfile(zip_file): 
        zip_file=os.path.join(zipdir, zip_file)
        
    with zipfile.ZipFile(zip_file,'r') as zip_ref:
        try : 
            # extract in the current directory 
            fetchedfile = retrieveZIPmember(zip_ref, **zip_kws ) 
        except : 
            raise  WATexExtractionError (
            f"Unable to retreive file from zip {zip_file!r}")
        print(f"---> Dataset={os.path.basename(fetchedfile)!r} "
              "was successfully retreived.")
            
    
def retrieveZIPmember(zipObj, *, 
                      file_to_extract='__tar.tgz_files__/___fmain.bagciv.data.csv',
                      savepath=None, rename_outfile='main.bagciv.data.csv' ) : 
    """ Retreive  member from zip and collapse the extracted directory by "
    "saving into a  new  directory
    
    Parameters
    -----------
    ZipObj: Obj zip 
        Reference zip object 
    file_to_extract:str or Path-Like Object 
        File to extract existing in zip archived. It should be a name list 
        of archived file. 
    savepath: str or Path-Like obj 
        Destination path after fetching the single data from zip archive.
        
    rename_outfile:str or Path-Like obj 
        Rename the `file_to_extract` if think it necessary. 
    
    Returns
    --------
        The name of path retreived. If file is renamed than shoud take it 
        new names.
    """
    if not os.path.isdir(savepath) :
        os.makedirs(savepath)
    if savepath is None: savepath =os.getcwd()
    
    if file_to_extract in zipObj.namelist(): 
        member2extract=zipObj.getinfo(file_to_extract)
        zipObj.extractall(members = [member2extract])
        
        shutil.move (os.path.join(os.getcwd(), file_to_extract), savepath)
        # destroy the previous path 
        if savepath != os.path.join(os.getcwd(),
                                    os.path.dirname(file_to_extract)): 
            # detroy the root if only if the savepath is different 
            # from the raw then extract member into the directory created.
            shutil.rmtree(os.path.join(os.getcwd(),
                                   os.path.dirname(file_to_extract)))
        
        if rename_outfile is not None: 
            os.rename(os.path.join(savepath, 
                                   os.path.basename (file_to_extract)), 
                      os.path.join(savepath, rename_outfile))
        elif rename_outfile is None: 
            rename_outfile= os.path.basename(file_to_extract)
            
    print(f"---> {rename_outfile!r} was successfully decompressed"
          f"  and saved to {savepath!r}"
          )
    
    return rename_outfile 
 
def move_file(filename, directory): 
    if os.path.isfile(filename):
        shutil.move(filename, directory)



