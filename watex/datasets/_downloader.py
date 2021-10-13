# -*- coding: utf-8 -*-
#       Author: Kouadio K.Laurent<etanoyau@gmail.con>
#       Create:on Tue Oct 12 15:37:59 2021
#       Licence: MIT

"""
..warnings:: This module is a core of the way to retreive data from local,  
    git and zenodo record. Modified this module  presume to enhance the way  
    the codes are written and you know what you are doing. Making a copy 
    before modifying this script is recommended.

"""
import os 
import sys 
import subprocess 
import shutil  
from six.moves import urllib 

from watex.utils.ml_utils import fetchSingleTGZData
from watex.utils._watexlog import watexlog

__logger = watexlog().get_watex_logger(__name__)

LOCAL_DIR = 'data/geo_fdata'
f__= os.path.join(LOCAL_DIR , 'main.bagciv.data.csv')
ZENODO_RECORD_ID_OR_DOI = '10.5281/zenodo.5560937'
GIT_ROOT = 'https://raw.githubusercontent.com/WEgeophysics/watex/master/' 
GIT_REPO= 'https://github.com/WEgeophysics/watex'
# from Zenodo: 'https://zenodo.org/record/5560937#.YWQBOnzithE'
DATA_PATH = 'data/__tar.tgz' 
TGZ_FILENAME = '/fmain.bagciv.data.tar.gz'
CSV_FILENAME = '/__tar.tgz_files__/___fmain.bagciv.data.csv'
DATA_URL = GIT_ROOT  + DATA_PATH  + TGZ_FILENAME

__all__=['fetchDataFromLocalandWeb']

def fetchDataFromLocalandWeb(f=f__): 
    """Retreive Bagoue dataset from Github repository or zenodo record."""
    
    mess =f"Fetching {os.path.basename(f)!r} from "
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
        return 
    __logger.info(f"{os.path.basename(f)!r} was successfully loaded.")
    
    return f

def _fromlocal (f=f__): 
    """ check whether the local file exists and return file name."""
    is_file =os.path.isfile(f)
    if not is_file :
        try: 
            __logger.info("Fetching data from"
                          f" {TGZ_FILENAME.replace('/', '')}")
            print("---> Please wait while decompressing"
                  f" {TGZ_FILENAME.replace('/', '')!r} file... ")
            
            f0=fetchSingleTGZData(DATA_PATH +TGZ_FILENAME, 
                               rename_outfile='main.bagciv.data.csv')
            
        except : 
            __logger.info(f"Fetching  {TGZ_FILENAME.replace('/', '')!r} failed")
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
    success =False 
    if not os.path.isdir(LOCAL_DIR ):
        os.makedirs(LOCAL_DIR )
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
                print("--> Established connection failed because connected"
                      " host has failed to respond")
            success =False 
        except:success =False 
        else : succes=True 
        if succes:
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
        print("---> Forcing downloading by changing the root instead...")
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
                print(f"---> Downloading from {GIT_REPO!r} failed!")
                success=False 
        else : 
            print(f"---> Downloading from {GIT_REPO!r} was successfully done!")
            success =True
    if not success: return False    
    if success :
        # assume the data is locate in current directory
        # then move to the right place in Local dir 
        if os.path.isfile('main.bagciv.data.csv'):
            move_file('main.bagciv.data.csv', LOCAL_DIR)
        
    print("---> Fetching `main.bagciv.data.csv`from {GIT_REPO!r}"
          " was successfully done!") 
    return f 


def _fromzenodo( doi = ZENODO_RECORD_ID_OR_DOI, path = LOCAL_DIR): 
    """Fetch data from zenodo records with ``doi`` and ``path``"""
    success_import=False 
    try:
        import zenodo_get
    except: 
        #implement pip as subprocess
        # and download the record using zenodo get 
        # this will take a while if the connection is low.
        # Please be patient.
        try: 
            print("---> Zenodo_get is installing. Please wait ...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install',
            'zenodo_get'])
            reqs = subprocess.check_output([sys.executable,'-m', 'pip',
                                            'freeze'])
            #installed_packages  =...
            [r.decode().split('==')[0] for r in reqs.split()]
            #get the list of installed dependancies 
            watexlog.get_watex_logger().info(
                "Intallation of `zenodo_get` was successfully done!") 
            success_import=True
            
        except : 
            # Connection problem can occur and failed can happens. 
            print('---> Zenodo_get installation failed!')
            __logger.info("Fail to force installation of `zenodo_get`")
            
    else: success_import=True 

    if not success_import: 
        raise ConnectionError("Unable to retrieve data from record "
                              f"<{ZENODO_RECORD_ID_OR_DOI!r}.")
    
    # if zenodo_get is already installed Then used to 
    # wownloaed the record by calling the subprocess methods
    __logger.info(" `zenodo_get` package already installed!") 
        
    print(f"---> Please wait while the record <{ZENODO_RECORD_ID_OR_DOI}>"
          " is downloading...")
    subprocess.check_call([sys.executable, '-m', 'zenodo_get', doi])
        
    print(f"---> Record <{ZENODO_RECORD_ID_OR_DOI}>"
              " was sucessuffly downloaded...")
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
    
    return f0
    
    
def unZipFileFetchedFromZenodo(zipdir =LOCAL_DIR, 
                               zip_file ='BagoueCIV__dataset__main.rar'):
    """ Unzip or Unrar the archived file and shift from  the local 
    directory created if not exits. """
    zipORrar_ex = zip_file.replace('BagoueCIV__dataset__main', '')
    zip_file=zip_file.replace(zipORrar_ex, '')
    # file is in zip #'/__tar.tgz_files__/___fmain.bagciv.data.csv'
    raw_location = zip_file + CSV_FILENAME 
    
    if zipORrar_ex=='.rar':
        try :
            import rarfile
        except : 
            print("---> rarfile is installing. Please wait ...")
            subprocess.check_call([sys.executable, '-m', 'pip', 
                                   'install','rarfile'])
            reqs = subprocess.check_output([sys.executable,'m', 'pip',
                                            'freeze'])
            [r.decode().split('==')[0] for r in reqs.split()]
            __logger.info("Intallation of `rarfile` was successfully done!") 
            print("---> Installing of `rarfile` is sucessfully done!")
            
        print(f"---> Please wait while `<{zip_file+'.rar'}="
              "main.bagciv.data.csv>`is unraring...")
        
        # rarfile.RarFile.(os.path.join(zipdir, zip_file +'.rar'))
        __logger.info("Extract {os.path.basename(CSV_FILENAME)!r}"
                      " from {zip_file + '.rar'} file.")
        #--------------------------work on the rar extraction since -----------
        # rar can not extract larger file excceed fo 50
        # we are working to find the way to automatically decompressed rarfile.
        # and keep it to the local directory
        with rarfile.RarFile(os.path.join(zipdir, zip_file +'.rar'))as rar_ref: 
            rar_ref.extract(member=raw_location, path = zipdir)
        # rarfile.RarFile().extract(member=raw_location, path = zipdir)
        #----------------------------------------------------------------------
        print(f"---> Unraring the `{zip_file}=main.bagciv.data.csv`"
              "was successfully done.")
        
    elif zipORrar_ex=='.zip':
        import zipfile
        try : 
            #ZipFile.extractall(path=None, members=None, pwd=None)
            # path: location where zip file needs to be extracted; if not 
            #     provided, it will extract the contents in the current
            #     directory.
            # members: list of files to be extracted. It will extract all 
            #     the files in the zip if this argument is not provided.
            # pwd: If the zip file is encrypted, then pass the password in
            #     this argument default is None.
            with zipfile.ZipFile(
                    os.path.join(zipdir, zip_file +'.zip'), 'r') as zip_ref:
                zip_ref.extractall(zipdir, members =raw_location)
            # and now get the file and move it to the directory 
        except : 
            raise OSError(f'Unzip <{zip_file}==main.bagciv.data.csv> failed.'
                          'Please try again.')
 
    if os.path.isfile (zipdir + '/___fmain.bagciv.data.csv'): 
        os.rename(zipdir + '/___fmain.bagciv.data.csv',
                  zipdir + '/main.bagciv.data.csv')

    # Ascertain the file
    is_f_file = _fromlocal(zipdir + '/main.bagciv.data.csv')
    if is_f_file ==zipdir + '/main.bagciv.data.csv':
        print("Extraction of `main.bagciv.data.csv` was successfully done!")
        
    return is_f_file 

def move_file(filename, directory): 
    if os.path.isfile(filename):
        shutil.move(filename, directory)




