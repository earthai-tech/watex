# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com> 
#   Created on Sat Oct  1 15:24:33 2022
"""
Remote Loader 
==============

Fetch data online from zenodo record or repository.  
"""
from __future__ import (
    print_function , 
    annotations 
    )
import os 
import time
import sys 
import subprocess 
import concurrent.futures
import shutil  
import zipfile
import warnings 
from six.moves import urllib 

from .._typing import (
    Optional, 
    )
from ..utils.funcutils import (
    is_installing 
)
from ..exceptions import (
    ExtractionError 
)
from ..utils.mlutils import (
    fetchSingleTGZData, 
    subprocess_module_installation
    )
from .._watexlog import  watexlog
_logger = watexlog().get_watex_logger(__name__)

##### config repo data ################################################

_DATA = 'data/geodata/main.bagciv.data.csv'
_ZENODO_RECORD= '10.5281/zenodo.5571534'

_TGZ_DICT = dict (
    # path = 'data/__tar.tgz', 
    tgz_f = 'data/__tar.tgz/fmain.bagciv.data.tar.gz', 
    csv_f = '/__tar.tgz_files__/___fmain.bagciv.data.csv'
 )

_GIT_DICT = dict(
    root  = 'https://raw.githubusercontent.com/WEgeophysics/watex/master/' , 
    repo = 'https://github.com/WEgeophysics/watex' , 
    blob_root = 'https://github.com/WEgeophysics/watex/blob/master/'
 )
_GIT_DICT ['url_tgz'] = _GIT_DICT.get ('root') + _TGZ_DICT.get('tgz_f')


def loadBagoueDataset (): 
    """Load a Bagoue dataset  
    
    Example 
    --------
    >>> from watex.datasets import Loader 
    >>> loadBagoueDataset ()
    ... dataset:   0%|                                          | 0/1 [00:00<?, ?B/s]
    ... ### -> Wait while decompressing 'fmain.bagciv.data.tar.gz' file ... 
    ... --- -> Fail to decompress 'fmain.bagciv.data.tar.gz' file
    ... --- -> 'main.bagciv.data.csv' not found in the  local machine 
    ... ### -> Wait while fetching data from 'https://raw.githubusercontent.com/WEgeophysics/watex/master/'...
    ... +++ -> Load data from 'https://raw.githubusercontent.com/WEgeophysics/watex/master/' successfully done!
    ... dataset: 100%|##################################| 1/1 [00:03<00:00,  3.38s/B]
    
    """
    # LOCAL_DIR = 'data/geodata'
    # DATA_DIR= os.path.join(LOCAL_DIR, 'main.bagciv.data.csv')
    # DATA_PATH = 'data/__tar.tgz' 
    # TGZ_FILENAME = '/fmain.bagciv.data.tar.gz'
    # CSV_FILENAME = '/__tar.tgz_files__/___fmain.bagciv.data.csv'
    # DATA_URL = GIT_ROOT  + DATA_PATH  + TGZ_FILENAME 
    # blob root = 'https://github.com/WEgeophysics/watex/blob/master/'
    # GIT_ROOT = 'https://raw.githubusercontent.com/WEgeophysics/watex/master/' 
    # GIT_REPO= 'https://github.com/WEgeophysics/watex'
    # from Zenodo: 'https://zenodo.org/record/5560937#.YWQBOnzithE'
    
    Loader ( 
        zenodo_record= _ZENODO_RECORD,
        content_url=  _GIT_DICT.get('root'),
        repo_url=  _GIT_DICT.get ('repo'),
        tgz_file=_GIT_DICT.get('url_tgz'),
        blobcontent_url =  _GIT_DICT.get ('blob_root'),
        zip_or_rar_file= 'BagoueCIV__dataset__main.rar',
        csv_file =  _TGZ_DICT.get('csv_f'),
        verbose=  10 
          ).fit(_DATA)
    
class Loader: 
    """ Load data from online 
    
    Parameters 
    ----------
    *zenodo_record*: str 
        A zenod digital object identifier (doi) of filepath to zenodo record.
        
    *content_url*: str, 
        File path to the repository user content. If your use GitHub where the 
        data is located in default branch for example a master branch, it 
        can be 'https://raw.githubusercontent.com/WEgeophysics/watex/master/' 
    *repo_url*: str 
        A url for repository that host the project 
        
    *tgzfile*: str, 
        Data can be save in TGZ file format. It that is the case, can provide 
        to fetch the data if all attempt to fetched the file failed. 
    *verbose*: int, 
        Level of verbosity. Higher equals to more messages. 
        
    *root2blobcontent*: str 
        Root to blob master is a nested way to the convenient way to retrieve
        raw data in GitHUB
    *csv_file*: str 
        Path to the main csv file to retreive in the record.   
    """

    def __init__(self, 
                 zenodo_record:str = None, 
                 content_url:str = None, 
                 repo_url: str = None, 
                 tgz_file:str = None, 
                 blobcontent_url:str = None, 
                 zip_or_rar_file:str = None, 
                 csv_file: str = None, 
                 verbose: int =0 ,  
                 ): 

        self.zenodo_record = zenodo_record 
        self.content_url = content_url
        self.blobcontent_url = blobcontent_url 
        self.repo_url =repo_url
        self.tgz_file = tgz_file 
        self.zip_or_rar_file=zip_or_rar_file 
        self.csv_file = csv_file 
        
        self.verbose = verbose
        
        self.f_= None 
        
    @property 
    def update_zenodo_record (self):
        return self.zenodo_record 
    
    @update_zenodo_record.setter 
    def update_zenodo_record(self, uzr): 
        self.zenodo_record = uzr 
        
    @property 
    def f(self): 
        return self.f_ 
    @f.setter 
    def f (self, file): 
        """ assert the file exists"""
        self.f_ = file 
        
        
    def fit(self , f:str = None): 
        
        """ Retreive Bagoue dataset from Github repository or zenodo record. 
        
        It will take a while when fetching data for the first time outsite of 
        this repository. Since cloning the repository come with examples dataset  
        located to its appropriate directory. It's probably a rare to fectch using 
        internet unless dataset  as well as the tarfile are  deleted from its
        located directory.
        
        Parameters
        ------------
        f : str 
            `f` is the reference to the main file containing the data acting 
            like a path -like object.
        
        Returns 
        -------
        ``self``  :class:`~.Loader` instance
        
        Notes 
        ---------
        Retreiving  dataset line Bagoue dataset from Github repository or zenodo 
        record. It could take a while to fetch data for the first time outsite of 
        therepository. Since cloning the repository come with examples dataset  
        located to its appropriate directory, it's probably not useful to fectch 
        the data from internet unless the dataset ( with the tarfileor not ) are
        deleted from the local directory. 
        
        Example
        ---------
        >>> from watex.datasets.load import Loader 
        >>> loadObj = Loader (
                zenodo_record= '10.5281/zenodo.5571534',
                content_url=  'https://raw.githubusercontent.com/WEgeophysics/watex/master/',
                repo_url= 'https://github.com/WEgeophysics/watex',
                tgz_file='https://raw.githubusercontent.com/WEgeophysics/watex/master/data/__tar.tgz/fmain.bagciv.data.tar.gz',
                blobcontent_url =   'https://github.com/WEgeophysics/watex/blob/master/',
                zip_or_rar_file= 'BagoueCIV__dataset__main.rar',
                csv_file =  '/__tar.tgz_files__/___fmain.bagciv.data.csv',
                verbose=  10
                )
        >>> loadObj.fit('data/geodata/main.bagciv.data.csv')
        ... ### -> Wait while decompressing 'fmain.bagciv.data.tar.gz' file ... 
        ... --- -> Fail to decompress 'fmain.bagciv.data.tar.gz' file
        ... --- -> 'main.bagciv.data.csv' not found in the  local machine  
        ... ### -> Wait while fetching data from 'https://raw.githubusercontent.com/WEgeophysics/watex/master/'...
        ... +++ -> Load data from 'https://raw.githubusercontent.com/WEgeophysics/watex/master/' successfully done!
        dataset: 100%|##################################| 1/1 [00:04<00:00,  4.95s/B]
        Out[23]: <watex.datasets.load.Loader at 0x2210bedf880>
     
        """
        #--++++++-------import tqdm package 
        TQDM= False 
        try : 
            import tqdm 
        except ImportError: 
            is_success = is_installing('tqdm'
                                       )
            if not is_success: 
                warnings.warn("'Auto-install tqdm' failed. Could be installed it manually"
                              " Can get 'tqdm' here <https://pypi.org/project/tqdm/> ")
                _logger.info ("Failed to install automatically 'tqdm'. Can get the " 
                              "package via  https://pypi.org/project/tqdm/")
            else : TQDM = True 
            
        else: TQDM = True 
        
        #--++++++-------
        
        if f is not None: 
            self.f= f 
            
        mess =f" Unable to load {os.path.basename(self.f)!r} from "
        
        if not TQDM: 
            with concurrent.futures.ThreadPoolExecutor() as executor: 
                modules =[ 'notebook', 'ipywidgets', 'tqdm']
                try : 
                    is_success =list(executor.map(
                        subprocess_module_installation, modules))
                except : 
                    results = [executor.submit(
                        subprocess_module_installation, args =[mod, True])
                                               for mod in modules]
                    is_success =[f.result() for f in 
                           concurrent.futures.as_completed(results)]
                    # if n all modules were executed successffuly 
                    # force tqm 
                    TQDM = is_success [0] if len(set(is_success))==1 else False 
                
        pbar = range(1) if not TQDM else tqdm.tqdm(range(1) ,ascii=True, 
                     unit='B', desc ="dataset", ncols =77)

        for _ in pbar :
            total , start =0, time.perf_counter() 
            if not os.path.isdir(os.path.dirname (self.f) ):
                os.makedirs(os.path.dirname (self.f) )
                
            # --> seek local file 
            is_file = self._fromlocal(self.f)
            if not is_file: 
                if self.verbose > 3: 
                    print(f"--- -> {os.path.basename(self.f)!r} not found in the "
                          " local machine  ")
                    
                _logger.info(f"{os.path.basename(self.f)!r} file is missing ")
                
                is_file =  self._fromgithub()
                if not is_file :
                    _logger.info(mess + 'Github')
                    is_file = self._fromzenodo()
        
            if not is_file : 
                _logger.info(mess + 'Zenodo')
                _logger.info (f"Unable to fetch {os.path.basename(f)!r} from online")
                end = time.perf_counter() 
                time.sleep(abs(start -end))
                pbar.update(total)
                
                return 
            _logger.info(f"{os.path.basename(f)!r} was successfully loaded.")
            
            end = time.perf_counter() 
            time.sleep(abs(start -end))
            
            if is_file: 
                total =1
                pbar.update(total)
                
        return self #f
    
    
    def _fromzenodo(self,  
            zenodo_record: str = None,  # ZENODO_RECORD_ID_OR_DOI, # LOCAL_DIR, 
            f: str = None,  
            zip_or_rar_file : str = None,
            csv_file : Optional[str]= None, 
            )-> str: 
        """Fetch data from zenodo records with ``zenodo_record`` and ``f``
        
        Here is the way to fetch the main dataset from the record using the 
        module `zenodo_get`  
        
        Parameters 
        -----------
        zenodo_record: str or Zenodo get obj 
            Record of zenodo database. see https://zenodo.org/
        f : str 
             Path -like object. f is the main file containing the data 
             
        zip_or_rar: str 
            Path like object to *.zip or *.rar file.
            
        csv_file: str 
            Path to the main csv file to retreive in the record. 
            
        Returns 
        ---------
         str : File or record path
            Here is the way to fetch the main dataset 
            
        Example 
        --------
        >>> from watex.datasets.load import ( _DATA , _ZENODO_RECORD , 
                                             _TGZ_DICT, _GIT_DICT, Loader)
        >>> Loader (verbose = 10 )._fromzenodo(
            f = _DATA, zenodo_record =_ZENODO_RECORD,
            zip_or_rar_file= 'BagoueCIV__dataset__main.rar',
            csv_file =  _TGZ_DICT.get('csv_f')
            )
        """
        if f is not None: 
            self.f = f 
        if zenodo_record is not None: 
            self.zenodo_record= zenodo_record
            
        if zip_or_rar_file is not None: 
            self.zip_or_rar_file = zip_or_rar_file 
            
        if self.zenodo_record  is None:
            raise TypeError (
                "Expect a zenodo record <'XXX/zenodo.YYYYY'>, get: 'None'")
            
        if not os.path.isdir(os.path.dirname(self.f ) ): 
            os.makedirs(os.path.dirname(self.f ) )
        success_import=False     
        try:
            import zenodo_get
        except: 
            # this will take a while if the connection is low. 
            # Please be patient.
            try: 
                if self.verbose : 
                    print("--- -> wait while zenodo_get is installing ...")
                is_ = is_installing ('zenodo_get')
                
                if is_ : 
                    _logger.info("'+++ -> zenodo_get' installation complete. ") 
                    success_import=True
                
                    if self.verbose > 3 : 
                        print("+++ -> zenodo_get' installation complete. ")
            except : 
                # Connection problem may happens. 
                if self.verbose > 3 : 
                    print('--- -> Fail to install Zenodo_get')
                _logger.info("Fail to  install `zenodo_get`")
                
        else: 
            success_import=True 

        if not success_import: 
            raise ConnectionError(
                F"Unable to retrieve data from record= <{self.zenodo_record!r}>.")
            
        # if zenodo_get is already installed Then used to 
        # downloaed the record by calling the subprocess methods
        _logger.info(" 'zenodo_get' package already installed") 
            
        if self.verbose: 
            print(f"### -> wait while the record {self.zenodo_record!r}"
               " is downloading...")
        try :
            subprocess.check_call([sys.executable, '-m', 'zenodo_get',
                                   self.zenodo_record])
        except: 
            raise ConnectionError (
                f"CalledProcessError: {self.zenodo_record!r} returned "
                "non-zero exit status 1. Please check your internet!")
            
        if self.verbose: 
            print(f"+++ -> Record {self.zenodo_record!r} successfully downloaded.")
        
        if not os.path.isdir(os.path.dirname(self.f ) ): 
            os.makedirs(os.path.dirname(self.f ) )
            
        # check whether Archive file is '.rar' or '.zip' 
        _, ex = os.path.splitext (self.zip_or_rar_file) 
        is_zipORrar =os.path.isfile (self.zip_or_rar_file )
        
        if is_zipORrar :
            ziprar_file = os.path.basename (self.zip_or_rar_file ) # 
            
        # else: ziprar_file = 'BagoueCIV__dataset__main.zip'
        
        # is_zipORrar =os.path.isfile ('BagoueCIV__dataset__main.rar')
        # if is_zipORrar : ziprar_file = 'BagoueCIV__dataset__main.rar'
        # else: ziprar_file = 'BagoueCIV__dataset__main.zip'
        
        # For consistency add curent work directory and move zip_rar file to 
        # the path =LOCAL_DIR and also move the md5sums file.
        move_file(os.path.join(os.getcwd(),ziprar_file), 
                  os.path.dirname(self.f ) )
        
        if os.path.isfile(os.path.join(os.getcwd(), 'md5sums.txt')):
            move_file(os.path.join(os.getcwd(), 'md5sums.txt'), 
                      os.path.dirname(self.f ) )
            
        if self.verbose > 3 :
            print(f"### -> Record <{zenodo_record!r}={ziprar_file!r}> found in "
                   f" {os.path.dirname(self.f )!r}.")
            print(f"### -> Wait while {'unziping' if not is_zipORrar else 'unraring'}"
                  " the record...")
            
        #Now unzip file in the LOCAL DIR then move the file to 
        # it right place and rename it. 
        f0=self.unZipFileFetchedFromZenodo(
            f= os.path.dirname(self.f ) , 
            zip_file =self.zip_or_rar_file, 
            csv_file = self.csv_file , 
            )
        
        try : 
            # if file exists then remove the archive 
            os.remove(os.path.join(self.f , ziprar_file))
        except :  pass 

        return f0
    
    def _fromgithub( self, 
                    f: str=None , content_url:str=None  
                    ) -> bool | str:
        """ Fetch the data from repository if file is hosted there. It creates
        path to the local matchine and save file.
        
        Parameters 
        -----------
        *f* : str 
             Path -like object. f is the main file containing the data 
        *content_url*: str, 
            File path to the repository user content. If your use GitHub where the 
            data is located in default branch for example a master branch, it 
            can be 'https://raw.githubusercontent.com/WEgeophysics/watex/master/' 
        *repo_url*: str 
            A url for repository that host the project
            
        """
        # make a request
        
        if f is not None: 
            self.f = f 
            
        if content_url is not None: 
            self.content_url  = content_url
            
        if not os.path.isdir(os.path.dirname(self.f)): 
            os.makedirs(os.path.dirname(self.f))
            
        success =False 
        #'https://raw.githubusercontent.com/WEgeophysics/watex/master/data/geo_fdata/main.bagciv.data.csv'
        rootf = os.path.join(self.content_url,  self.f)
        atp =[f"### -> Wait while fetching data from {self.content_url!r}...", 
              '... ', '... ']
        
        for i in range(3): 
            try : 
                # first attemptts to 03
                print(atp[i], end ='')
                
                urllib.request.urlretrieve(rootf, self.f)
            except TimeoutError: 
                if i ==2:
                    if self.verbose> 3: 
                        print("--- -> Established connection failed because "
                              " connected host has failed to respond.")
                success =False 
            except:success =False 
            else : success=True 
            if success:
                break 
            
        if not success:
            # CHANGEGIT Root 
            try:
                if self.verbose > 3 : 
                    print("### -> An alternative way using <blob/master>...")
                rootf0= self.blobcontent_url + self.f 
                urllib.request.urlretrieve(rootf0, self.f )
            except :success =False 
            else:success =True 
        if not success:
            if self.verbose:
                print("---> Coerce the root instead ...")
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
                    with open(os.path.join(
                            os.path.dirname (self.f),os.path.basename(self.f)),
                            'wb') as fs:
                        fs.write(response.content)
                except: 
                    success=False
            else :
                if self.verbose: 
                    print(f"+++ -> Load data from {self.content_url!r} "
                          "successfully done!")
                success =True
                
        if not success: 
            print(f"--- -> Fail to download data from {self.content_url!r} !")
            return False   
        
        if success :
            # assume the data is locate in current directory
            # then move to the right place in Local dir 
            if os.path.isfile(os.path.basename (self.f)):
                move_file(os.path.basename (self.f), os.path.dirname (self.fn))
            
        # print("---> Fetching `main.bagciv.data.csv`from {GIT_REPO!r}"
        #       " was successfully done!")
        if success: 
            print()
            print( f"+++ -> Load data from {self.content_url!r} successfully done!") 
        
        return self.f 
    
    def _fromlocal (self, f: str = None # DATA_DIR
                    ) -> str : 
        """ Check whether the local file exists and return file name. 
        
        Turn on all the possibility i.e read the *.tgz and *.tar file if exist 
        in the local machine. 
        
        Parameters 
        -----------
        f : str 
             Path -like object. f is the main file containing the data 
             
        
        """
        if f is not None: 
            self.f = f 
        
        is_file =os.path.isfile(self.f)
        
        if not is_file and self.tgz_file is not None:
            tgz= os.path.basename(self.tgz_file)
            
            try: 
                if self.verbose > 3 : 
                    print()
                    print(f"### -> Wait while decompressing {tgz!r} file ... ")
                    
                f0=fetchSingleTGZData(
                    self.tgz_file, rename_outfile=os.path.basename(self.f)
                    )
                
                _logger.info(f"Decompressed {tgz!r} successufully done.")
                
            except : 
                _logger.info(f"Fail to decompres{tgz!r} ")
                if self.verbose: 
                    print(f"--- -> Fail to decompress {tgz!r} file")
                
                return False 
            else : 
                if self.verbose: 
                    print(f"+++ -> Decompressed  {tgz!r} sucessfully done!")
                # return new file if file alread created in the local 
                # machine.
       
                self.f = f0
                    
        return self.f if os.path.isfile (self.f) else False 
    
    def unZipFileFetchedFromZenodo(self, 
                                   f: str  = None , # LOCAL_DIR, 
                                   #'BagoueCIV__dataset__main.rar',
                                   zip_or_rar_file: str = None, 
                                   #  '/__tar.tgz_files__/___fmain.bagciv.data.csv', 
                                   csv_file:str =None, 
                                    ):
        """ Unzip or Unrar the archived file and shift from  the local 
        directory created if not exits. 
        
        Parameters 
        -----------
        f : str 
             Path -like object. f is the main file containing the data 
             
        zip_or_rar: str 
            Path like object to *.zip or *.rar file.
            
        csv_file: str 
            Path to the main csv file to retreive in the record. 
            
        Returns 
        ---------
         str : path like object to the unzipped File 


        """
        # zipORrar_ex = zip_file.replace('BagoueCIV__dataset__main', '')
        # zip_file=zip_file.replace(zipORrar_ex, '')
        if f is not None: 
            self.f = f 
        if zip_or_rar_file  is not None: 
            self.zip_or_rar_file = zip_or_rar_file 
        if csv_file is not None: 
            self.csv_file = csv_file  
            
        zipORrar_ex  = os.path.splitext(self.zip_or_rar_file )[1]
        self.zip_or_rar_file=self.zip_or_rar_file.replace(zipORrar_ex, '')
        
        # file is in zip #'/__tar.tgz_files__/___fmain.bagciv.data.csv'
        raw_location = self.zip_or_rar_file + self.csv_file 
        zipdir = os.path.dirname (self.f)
        
        if zipORrar_ex=='.zip':
            try : 
                # CSV_FILENAME[1:]= '__tar.tgz_files__/___fmain.bagciv.data.csv',
                zip_location= os.path.join(zipdir, self.zip_or_rar_file +'.zip') 
                fetchSingleZIPData(zip_file= zip_location, zipdir = zipdir , 
                                   file_to_extract=self.csv_file[1:],
                                   savepath=zipdir, 
                                   rename_outfile=os.path.basename (self.f) ,
                                   verbose= self.verbose 
                                   )
            except : 
                raise OSError(f"Unzip {self.zip_or_rar_file +'.zip'}!r> failed."
                              'Please try again.')
     
        elif zipORrar_ex=='.rar':
            fetchSingleRARData(zip_file = self.zip_or_rar_file, 
                               file_to_extract= raw_location, 
                       zipdir =zipdir )
            #'/___fmain.bagciv.data.csv'):
        if os.path.isfile (zipdir + '/' + os.path.basename(self.csv_file)): 
            os.rename(zipdir + '/' + os.path.basename(self.csv_file),
                      zipdir + '/' + os.path.basename (self.f) #'main.bagciv.data.csv'
                      )

        # Ascertain the file
        f0 = self._fromlocal(zipdir + '/' + os.path.basename (self.f))
        if f0 ==zipdir + '/' + os.path.basename (self.f):
            print(f"+++ -> Extraction of {'/' + os.path.basename (self.f)} complete!")
            
        return f0 
    

def fetchSingleRARData(
        zip_file :str ,
        member_to_extract:str,
        zipdir: str , 
        verbose: False, 
        )-> None:
    """ RAR archived file domwloading process."""
    
    rarmsg = ["--> Please wait while using `rarfile` module to "
              f"<{zip_file}> decompressing...", 
              "--> Please wait while using `unrar` module to "
              f"<{zip_file}> decompressing..."]
    
    for i, name in enumerate('rarfile', 'unrar'):
        installation_succeeded =False 
        try :
            if i==0: 
                import rarfile
            elif i==1: 
                from unrar import rarfile
        except : 
            try:
                print(f"---> {name!r} is installing. Please wait ...")
                is_installing(name)
            except : 
                print("--> Failed to install {name!r} module !")
                if name =='unrar': 
                    print("---> Couldn't find path to unrar library. Please refer"
                          " to https://pypi.org/project/unrar/ and download the "
                          "UnRAR library. src: http://www.rarlab.com/rar/unrarsrc-5.2.6.tar.gz "
                          "or  src(Window): (http://www.rarlab.com/rar/UnRARDLL.exe)."
                          )
                    raise  ExtractionError (
                       "Fail to install UnrarLibrary!") 
                continue 
            else :
                _logger.info(f"Intallation of {name!r} was successfully done!") 
                print(f"---> Installing of {name!r} is sucessfully done!")
                installation_succeeded=True 
    
        if installation_succeeded : 
            print(f"---> Please wait while `<{zip_file+'.rar'}="
                  "main.bagciv.data.csv>`is unraring...")
        # rarfile.RarFile.(os.path.join(zipdir, zip_file +'.rar'))
        _logger.info("Extract {os.path.basename(CSV_FILENAME)!r}"
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
        raise  ExtractionError (
            "Failed the read enough data: req=33345 got>=52 files.")
     
    # rarfile.RarFile().extract(member=raw_location, path = zipdir)
    #----------------------------------------------------------------------
    if decompress_succeed:
        print(f"---> Unraring the `{zip_file}=main.bagciv.data.csv`"
          "was successfully done.")
        
def fetchSingleZIPData(
        zip_file:str,
        zipdir:str, 
        **zip_kws 
        )-> None: 
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
        # try : 
            # extract in the current directory 
            fetchedfile = retrieveZIPmember(zip_ref, **zip_kws ) 
        # except : 
        #     raise  ExtractionError (
        #     f"Unable to retreive file from zip {zip_file!r}")
        # print(f"---> Dataset={os.path.basename(fetchedfile)!r} "
        #       "was successfully retreived.")
            
    
def retrieveZIPmember(
        zipObj, *, 
        file_to_extract:str ='__tar.tgz_files__/___fmain.bagciv.data.csv',
        savepath: Optional[str] =None, 
        rename_outfile: str ='main.bagciv.data.csv' 
        ) -> str: 
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
    if savepath is None: 
        savepath =os.getcwd()
    if not os.path.isdir(savepath) :
        os.makedirs(savepath)
    
    
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
 
def move_file(filename:str , directory:str )-> str: 
    if os.path.isfile(filename):
        shutil.move(filename, directory)