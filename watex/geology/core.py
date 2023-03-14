# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Created date: Thu Sep 22 10:50:13 2022

"""
Core geology 
=====================
The core module deals with  geological data, the structural infos 
and borehole data.

"""

import os
import warnings
import shutil 
from six.moves import urllib 
from pprint import pprint 

from ..utils.funcutils import ( 
    smart_format, 
    sPath,
    )
from ..utils._dependency import ( 
    import_optional_dependency )
from ..property import (
    Config 
    )
from ..exceptions import ( 
    GeoPropertyError, 
    )
from .._watexlog import watexlog 
_logger = watexlog().get_watex_logger(__name__ )

__all__=[
    "Base", 
    "set_agso_properties", 
    "mapping_stratum", 
    "fetching_data_from_repo", 
    "get_agso_properties"
    ] 

class Base: 
    """
    Base class of container of geological informations  for stratigraphy model  
    log creation of exploration area.
    
    Each station is condidered as an attribute and  framed by two closest  
    points from the station offsets. The class deals with the true resistivity
    values collected on exploration area from drilling or geolgical companies. 
    Indeed, the input true resistivity values into the occam2d inversion data 
    could yield an accuracy underground map. The challenge to build a pseudolog 
    framed between two stations allow to know the layers disposal or supperposition
    from to top to the investigation depth. The aim is to  emphasize a  large 
    conductive zone usefful in of groundwater exploration. 
    The class `Geodrill` deals at this time with Occam 2D inversion files or 
    Bo Yang model (x,y,z) files. We intend to extend later with other external 
    softwares like the Modular System EM (MODEM) and else.
    It's also possible to generate output straighforwardly for others external 
    softwares  like Golder sofwares('surfer')or Oasis Montaj:
        
        - Surfer: :https://www.goldensoftware.com/products/surfer) 
        - Oasis: http://updates.geosoft.com/downloads/files/how-to-guides/Oasis_montaj_Gridding.pdf
         <https://www.seequent.com/products-solutions/geosoft-oasis-montaj/>
       
    Note: 
        If the user has a golder software installed on its computer, it 's 
        possible to use the output files generated here  to yield a 2D map so 
        to compare both maps to see the difference between model map (
            only inversion files and detail-sequences map after including the 
            input true resistivity values and layer names) 
    Futhermore, the "pseudosequences model" could match and describe better 
    the layers disposal (thcikness and contact) in underground than the 
    raw model map which seems to be close to the reality  when `step descent` 
    parameter is not too small at all. 


    """
    
    def __init__(self, verbose: int =0 , **kwargs):
        self._logging = watexlog.get_watex_logger(self.__class__.__name__)
        self.verbose= verbose 
        
#++++ configure the geological rocks from files:AGSO & AGSO.STCODES +++++++++++
__agso_properties =dict(
    GIT_REPO = 'https://github.com/WEgeophysics/watex', 
    GIT_ROOT ='https://raw.githubusercontent.com/WEgeophysics/watex/master/',
    props_dir = 'watex/etc/',
    props_files = ['AGSO.csv', 'AGSO_STCODES.csv'], 
    props_codes = ['code', 'label', 'name', 'pattern','size',
            'density', 'thickness', 'color']
    )

def set_agso_properties (download_files = True ): 
    """ Set the rocks and their properties from inner files located in 
        < 'watex/etc/'> folder."""
        
    msg= ''.join([
        "Please don't move or delete the properties files located in", 
        f" <`{__agso_properties['props_dir']}`> directory."])
    mf =list()
    __agso= [ os.path.join(os.path.realpath(__agso_properties['props_dir']),
                           f) for f in __agso_properties['props_files']]
    for f in __agso: 
        agso_exists = os.path.isfile(f)
        if not agso_exists: 
            mf.append(f)
            continue 
    
    if len(mf)==0: download_files=False 
    if download_files: 
        for file_r in mf:
            success = fetching_data_from_repo(props_files = file_r, 
                      savepath = os.path.join(
                          os.path.realpath('.'), __agso_properties['props_dir'])
                      )
            if not success:
                msg_ = ''.join([ "Unable to retreive the geostructure ",
                      f"{os.path.basename(file_r)!r} property file from",
                      f" {__agso_properties['GIT_REPO']!r}."])
                warnings.warn(f"Geological structure file {file_r} "
                              f"is missing. {msg_}") 
                _logger.warn( msg_)
                raise GeoPropertyError(
                    f"No property file {os.path.basename(file_r)!r}"
                    f" is found. {msg}.")
    for f in __agso:
        with open(f,'r' , encoding ='utf8') as fs: 
            yield([stratum.strip('\n').split(',')
                    for stratum in fs.readlines()])
            
def mapping_stratum(download_files =True): 
    """ Map the rocks properties  from  _geocodes files and fit 
    each rock to its properties. 
    
    :param download_files: bool 
        Fetching data from repository if the geostrutures files are missing.
    :return: Rocks and structures data  in two diferent dictionnaries
    """
    # get code description _index 
    ix_= __agso_properties['props_codes'].index('name')
    def mfunc_(d): 
        """ Set individual layer in dict of properties """
        _p= {c: k.lower() if c not in ('code', 'label', 'name') else k 
                 for c,  k in zip(__agso_properties['props_codes'], d) }
        id_= d[ix_].replace('/', '_').replace(
            ' ', '_').replace('"', '').replace("'", '').lower()
        return id_, _p 
    rock_and_structural_props =list()
    for agso_data in tuple(set_agso_properties(download_files)): 
        # remove the header of the property file
        rock_and_structural_props.append(
            dict(map( lambda x: mfunc_(x), agso_data[1:])))
     
    return   tuple(rock_and_structural_props)

def fetching_data_from_repo(repo_file, savepath =None ): 
    """ Try to retrieve data from github repository.
    
    :param repo_file: str or Path-like object 
        Give the full path from the repository root to the file name.
        For instance, we want to retrieve the file 'AGSO.csv' which is located 
        in <watex/etc/> directory then the full path 
        is: --> 'watex/etc/AGSO.csv'
        
    :return:`status`: Either ``False` for failed downloading 
            or ``True`` for successfully downloading
    """
    fmsg =['... 1rst attempt...','... 2nd attempt...','... 3rd attempt...']
    status=False 
    git_repo = __agso_properties['GIT_REPO']
    git_root = __agso_properties['GIT_ROOT']
    
    # Install bar progression
    import_optional_dependency ("tqdm")
    from tqdm.notebook  import trange 
    # max attempts =3 :  
    print("---> Please wait while fetching"
          f" {repo_file!r} from {git_repo!r}...")
    for k in trange(3, ascii=True, desc ='geotools', ncols =107):
    #for i in tqdm(range(3), ascii=True, desc ='WEgeophysics', ncols =107):
        for _ in trange(1, ascii=True ,desc =fmsg [k],ncols =107):
            try :
                urllib.request.urlretrieve(git_root,  repo_file )
            except: 
                try :
                    with urllib.request.urlopen(git_root) as response:
                        with open( repo_file,'wb') as out_file:
                            data = response.read() # a `bytes` object
                            out_file.write(data)
        
                except TimeoutError: 
                    if k ==2: 
                        print("---> Established connection failed "
                           " because connected host has failed to respond.")
                except:pass 
            else : status=True

        if status: break

    if status: print(f"---> Downloading {repo_file!r} from {git_repo!r} "
                 "was successfully done!")
    else: print(f"---> Failed to download {repo_file!r} from {git_repo!r}!")
    # now move the file to the right place and create path if dir not exists
    if savepath is not None: 
        if not os.path.isdir(savepath): 
            sPath (savepath)
        shutil.move(os.path.realpath(repo_file), savepath )
    if not status:pprint(connect_reason )
    
    return status

def get_agso_properties(config_file =None, orient ='series'): 
    """ Get the geostructures files from <'watex/etc/'> and 
    set the properties according to the desire type. When `orient` is 
    ``series`` it will return a dictionnary with key equal to 
    properties name and values are the properties items.
    
    :param config_file: Path_Like or str 
        Can be any property file provided hat its obey the disposal of 
        property files found in   `__agso_properties`.
    :param orient: string value, ('dict', 'list', 'series', 'split',
        'records’, ''index') Defines which dtype to convert
        Columns(series into).For example, 'list' would return a 
        dictionary of lists with Key=Column name and Value=List 
        (Converted series). For furthers details, please refer to
        https://www.geeksforgeeks.org/python-pandas-dataframe-to_dict/
        
    :Example: 
        >>> import watex.utils.geotools as GU
        >>> data=get_agso_properties('watex/etc/AGSO_STCODES.csv')
        >>> code_descr={key:value for key , value in zip (data["CODE"],
                                                       data['__DESCRIPTION'])}
    """
    msg= ''.join(["<`{0}`> is the software property file. Please don't move "
        " or delete the properties files located in <`{1}`> directory."])
    
    pd_pos_read = Config().parsers 
 
    ext='none'
    if config_file is None: 
        config_file = os.path.join(os.path.realpath('.'), os.path.join(
                       __agso_properties['props_dir'],
                       __agso_properties ['props_files'][0]))
    if config_file is not None: 
        is_config = os.path.isfile(config_file)
        if not is_config : 
            if os.path.basename(config_file) in __agso_properties['props_files']:
                _logger.error(f"Unable to find  the geostructure property" 
                              f"{os.path.basename(config_file)!r} file."
                              )
                warnings.warn(msg.format(os.path.basename(config_file) , 
                                         __agso_properties['props_dir']))
            raise FileExistsError(f"File `{config_file}`does not exist")
            
        _, ext = os.path.splitext(config_file)
        if ext not in pd_pos_read.keys():
            _logger.error(f"Unable to read {config_file!r}. Acceptable formats"
                          f" are {smart_format(list(pd_pos_read.keys()))}.")
            raise TypeError(
                f"Format {ext!r} cannot be read. Can only read "
                 f"{smart_format(list(pd_pos_read.keys()))} files."
                )
    agso_rock_props = pd_pos_read[ext](config_file).to_dict(orient)
    if ('name' or 'NAME') in agso_rock_props.keys(): 
        agso_rock_props['__DESCRIPTION'] = agso_rock_props ['name']
        del agso_rock_props['name']
        
    return  agso_rock_props

##############connection git error ##########################
connect_reason ="""<ConnectionRefusedError><No connection could  '
            be made because the target machine actively refused it>.
            There are some possible reasons for that:
         1. The server is not running. Hence it wont listen to that port. 
             If it's a service you may want to restart the service.
         2. The server is running but that port is blocked by Windows Firewall
             or other firewall. You can enable the program to go through 
             firewall in the Inbound list.
        3. There is a security program on your PC, i.e a Internet Security 
            or Antivirus that blocks several ports on your PC.
        """  
#+++++ end  AGSO & AGSO.STCODES configuration +++++++++++++++++++++++++++++++++