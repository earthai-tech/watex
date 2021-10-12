# -*- coding: utf-8 -*-
#       Author: Kouadio K.Laurent<etanoyau@gmail.con>
#       Create:on Tue Oct 12 15:37:59 2021
#       Licence: MIT
import os 
import sys 
import subprocess 
import shutil  
from six.moves import urllib 

from watex.utils.ml_utils import fetch_geo_data
from watex.utils._watexlog import watexlog

__logger = watexlog().get_watex_logger(__name__)

LOCAL_DIR = 'data/geo_fdata'
f__= os.path.join(LOCAL_DIR , 'main.bagciv.data.csv')
ZENODO_RECORD_ID_OR_DOI = '10.5281/zenodo.5560937'
GIT_ROOT = 'https://raw.githubusercontent.com/WEgeophysics/watex/master/' 
#'https://github.com/WEgeophysics/watex/master/'
# 'https://github.com/ageron/handson-ml/'
# from Zenodo: 'https://zenodo.org/record/5560937#.YWQBOnzithE'
DATA_PATH = 'data/__tar.tgz'  # 'BagoueCIV__dataset__main/__tar.tgz_files__'
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
        __logger.info ("Unable to fetch {os.path.basename(f)!r} from Web")
        return 
    __logger.info(f"{os.path.basename(f)!r} was successfully loaded.")
    
    return f

def _fromlocal (f=f__): 
    """ check whether the local file exists and return file name."""
    is_file =os.path.isfile(f)
    if not is_file :
        try: 
            __logger.info(f"Fetching data from {TGZ_FILENAME.replace('/', '')}")
            fetch_geo_data(DATA_URL,DATA_PATH)
        except : 
            __logger.info(f"Fetching  {TGZ_FILENAME.replace('/', '')!r} failed")
            return False 
        else : 
            # go to 'data/__tar.tgz/__tar.tgz_files__/___fmain.bagciv.data.csv
            if os.path.isfile( DATA_PATH + CSV_FILENAME): 
                # move file to data path.
                shutil.move(DATA_PATH + CSV_FILENAME, DATA_PATH)
    return f 

def _fromgithub( f=f__, root = GIT_ROOT):
    """ Get file from github and if file exists create your local directory  
        and save file."""
    # make a request
    success =False 
    if not os.path.isdir(LOCAL_DIR ):
        os.mkdirs(LOCAL_DIR )
    try : 
        # first attemptts
        urllib.request.urlretrieve(root, f)
    except: 
        # CHANGEGIT Root 
        root= 'https://github.com/WEgeophysics/watex/blob/master/' 
        urllib.request.urlretrieve(root, f)
        success =True 
    if not success: 
        rootf = os.path.join(GIT_ROOT, f)
        #'https://github.com/WEgeophysics/watex/blob/master/data/geo_fdata/main.bagciv.data.csv'
        #second attempts 
        try : 
            with urllib.request.urlopen(rootf) as testfile, open(f, 'w') as fs:
                    fs.write(testfile.read().decode())
        except : 
            # third attempts 
            import requests 
            response = requests.get(rootf)
            with open(os.path.join(LOCAL_DIR, os.path.basename(f)), 'wb') as fs:
                fs.write(response.content)
                
    is_file = _fromlocal(f) ==f     
    if not is_file: return False 
    
    return f

def _fromzenodo( doi = ZENODO_RECORD_ID_OR_DOI, path = LOCAL_DIR): 
    """Fetch data from zenodo records with ``doi`` and ``path``"""
    success_import=False 
    try:
        import zenodo_get
    except: 
        #implement pip as subprocess
        try: 
            subprocess.check_call([sys.executable, '-m', 'pip', 'install',
            'zenodo_get'])
            reqs = subprocess.check_output([sys.executable,'-m', 'pip', 'freeze'])
            #installed_packages  =...
            [r.decode().split('==')[0] for r in reqs.split()]
            #get the list of installed dependancies 
            watexlog.get_watex_logger().info(
                "Intallation of `zenodo_get` was successfully done!") 
            
            success_import=True
        except : 
            watexlog.get_watex_logger().info(
                "Fail to force installation of `zenodo_get`")
            
            success_import=False
    else:
        watexlog.get_watex_logger().info(
                " `zenodo_get` package already exists!") 
        success_import=True     
        subprocess.check_call([sys.executable, '-m', 'zenodo_get', doi])
        
    if not success_import: 
        raise f"Unable to retrieve data from record <{ZENODO_RECORD_ID_OR_DOI!r}."
        
    return unZipFileFetchedFromZenodo()
    
    
def unZipFileFetchedFromZenodo(zipdir =LOCAL_DIR, zip_file ='BagoueCIV__dataset__main'):
    """ Unzip or Unrar the archived file and shift to the local
    directory created if not exits. """
    # file is in zip 
    import zipfile 
    archive_file ='zip'
    archive_file_found=False
    
    try : 
        if not os.path.isdir(zipdir): 
            os.mkdirs(zipdir)
        # with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        #     zip_ref.extractall(directory_to_extract_to)
        with zipfile.ZipFile(os.path.join(zipdir, zip_file), 'r') as zip_ref:
            zip_ref.extractall(zipdir)
        # and now get the file and move it to the directory 
    except : 
        try :
            import rarfile
        except : 
            subprocess.check_call([sys.executable, '-m', 'pip', 'install','rarfile'])
            reqs = subprocess.check_output([sys.executable,'m', 'pip', 'freeze'])
            [r.decode().split('==')[0] for r in reqs.split()]
            __logger.info("Intallation of `rarefile` was successfully done!") 
            archive_file='rar'
            
    else:
        archive_file_found =True 
    # raw location og 
    raw_location = zip_file + CSV_FILENAME #'/__tar.tgz_files__/___fmain.bagciv.data.csv'
    
    if archive_file=='rar': 
        rf= rarfile.RarFile(os.path.join(zipdir, zipfile, ".rar"))
        for file in rf.infolist(): 
            if file == raw_location:
                archive_file_found =True 
    if archive_file_found :        
        shutil.move(raw_location, zipdir)
    
    is_f_file = _fromlocal(f__)
    
    return is_f_file 



if __name__=='__main__': 
    d= 'https://raw.githubusercontent.com/ageron/handson-ml/master/' 
    p= 'data/housing'
    h_url = d + p +"/housing.tgz"
    
    # fetch_geo_data2(data_url=h_url, data_path =p, tgz_filename ="/housing.tgz")
    # _fromgithub()
    print(_fromzenodo())







