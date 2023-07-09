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
from __future__ import annotations 
import os
import warnings
from importlib import resources
import numpy as np 

from ..utils.baseutils import get_remote_data
from ..utils.funcutils import ( 
    ellipsis2false, 
    smart_format, 
    is_iterable,
    key_search, 
    )
from ..utils.geotools import ( 
    find_similar_structures, 
    )
from ..property import (
    Config 
    )
from ..utils.validator import _is_arraylike_1d 
from ..exceptions import ( 
    GeoPropertyError, 
    )
from .._watexlog import watexlog 
from .._typing import List 

_logger = watexlog().get_watex_logger(__name__ )

__all__=[
    "GeoBase", 
    "set_agso_properties", 
    "mapping_stratum", 
    "get_agso_properties"
    ] 
                                  
#++++ configure the geological rocks from AGSO DataBase +++++++++++++++++++++++
EMOD = 'watex.etc' ; buffer_file = 'AGSO.csv'
with resources.path (EMOD, buffer_file) as buff : 
     props_buf  = str(buff) 
AGSO_PROPERTIES =dict(
    GIT_REPO = 'https://github.com/WEgeophysics/watex', 
    GIT_ROOT ='https://raw.githubusercontent.com/WEgeophysics/watex/master/',
    props_dir = os.path.dirname (props_buf),
    props_files = ['AGSO.csv', 'AGSO_STCODES.csv'], 
    props_codes = ['code', 'label', 'name', 'pattern','size',
            'density', 'thickness', 'color']
    )
#++++ end configuration +++++++++++++++++++++++++++++++++++++++++++++++++++++++

class GeoBase: 
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
        
        for key in list(kwargs.keys()): 
            setattr(self, key, kwargs[key])
            
    @staticmethod 
    def find_properties(
        data=None, *, 
        constraint: bool=..., 
        keep_acronyms: bool=..., 
        fill_value: str=None, 
        kind: str='geology', 
        attribute :str='code', 
        property: str='description',
        ): 
        """ Find rock/structural properties or constraint the geological info 
        to fit the rock in AGSO database.
        
        Parameters 
        -----------
        data: Arraylike one-dimensional or pd.Series, optional 
           Arraylike containing the geological or structural informations. 
           
        constraint: bool, default=False 
           fit the geological or structual info to match the AGSO geology 
           or structural database infos. 
           
        keep_acronym: bool, default=False, 
           Use the acronym of the structural and geological database 
           info to replace the full geological/structural name. 
           
        fill_value: str, optional 
           If None, the undetermined structured are not replaced. They are 
           kept in the data. However, if the fill_value is provided, the 
           missing structure from data is replaced by the fill_value. 
           
         kind: str,  default='geology'
           Is the type of geo-data to fetch in the database. It can be 
           ['geology'|'samples'|'structural']. Any other values except 
           the 'geology' will returns the structural samples. 
           
        attribute: str, default='code'
          The name of attribute to collect as the keys. It can be 
          - 'code', 'label', 'description', 'pattern' or  
          - 'pat_size', 'pat_density', 'pat_thickness', 'color'
        
        property: str, default='description' 
          The name of property to collect as the values. By default is 
          the geological or structural mames. 
          
        Returns 
        ---------
        - attributes/property: tuple (str ) 
           A key, value in pair of the attribute and property fetched 
           in the AGSO database. 
        - data constrainted. Pd.series or array
           Data with items fitted with the property names in the AGSO 
           database. 
           
        Examples
        ---------
        >>> from watex.geology.core import GeoBase 
        >>> GeoBase.find_properties () [:7]
        Out[7]: 
        (('AGLT', 'argillite'),
         ('ALUV', 'alluvium'),
         ('AMP', 'amphibolite'),
         ('ANS', 'anorthosite'),
         ('ANT', 'andesite'),
         ('APL', 'aplite'),
         ('ARKS', 'arkose'))
        >>> # make data 
        >>> geodata =['gran', 'basalt', 'migmatite', 'sand', 'tuff']
        >>> GeoBase.find_properties (geodata, constraint =True )
        Out[8]: 
        array(['granodior', 'basalt', 'migmatite', 'sandstone', 'tuff'],
              dtype='<U9')
        >>> geodata +=['Unknow structure']
        >>> GeoBase.find_properties (geodata, constraint =True,
                                        fill_value='NA' )
        Out[9]: 
        array(['granodiorite', 'basalt', 'migmatite', 'sandstone', 'tuff', 'NA'],
              dtype='<U16')
        >>> GeoBase.find_properties (geodata, constraint =True,
                                        fill_value='NA' , keep_acronyms=True )
        Out[10]: array(['grd', 'blt', 'mig', 'sdst', 'tuf', 'NA'], dtype='<U16')
        """
        constraint, keep_acronyms= ellipsis2false(constraint, keep_acronyms)
        kind = str(kind).lower().strip() 
        if 'geology'.find (kind) >=0: kind ='geology'
            
        fname = buffer_file if kind =='geology' else 'AGSO_STCODES.csv'
        path_file =  os.path.join( AGSO_PROPERTIES.get("props_dir"), fname) 
        _agso_data=get_agso_properties(path_file)
        dp =list ()
        for cod  in ( attribute, property ): 
            d = key_search (str( cod), 
                            default_keys= list(_agso_data.keys()), 
                            deep= True, 
                            parse_keys= False, 
                            raise_exception= True 
                    )
            dp.append (d[0])
        # unpack attribute and properties 
        attribute , property = dp 
        attribute = attribute.upper(); property = str(property).upper() 
        prop_data={key:value for key , value in zip (
            _agso_data[attribute], _agso_data[property])}
        if not constraint: 
            return tuple (prop_data.items() )
        
        if data is None: 
            raise TypeError (
                "Data cannot be None when constraint is set to ``True``")
        # for consistency 
        data = np.array (
            is_iterable (data , exclude_string= True, transform= True) ) 
        if not  _is_arraylike_1d(data ): 
            raise GeoPropertyError (
                "Geology or Geochemistry samples expects"
               f" one dimensional array. Got shape ={data.shape}")
                
        found=False # flags if structure is found
        for kk, item in enumerate ( data) : 
            for key,  value in prop_data.items(): 
                if str(value).lower().find (str(item).lower() )>=0: 
                    data[kk] = str(key).lower() if keep_acronyms else value
                    found = True 
                    break
            # if item not found then 
            # property data.
            if not found:
                if fill_value is not None:
                    data[kk] = fill_value
            found =False 
        return data 
    
    @staticmethod
    def getProperties(properties =['electrical_props', '__description'], 
                       sproperty ='electrical_props'): 
        """ Connect database and retrieve the 'Eprops'columns and 'LayerNames'
        
        :param properties: DataBase columns.
        :param sproperty : property to sanitize. Mainly used for the properties
            in database composed of double parenthesis. Property value 
            should be removed and converted to tuple of float values.
        :returns:
            - `_gammaVal`: the `properties` values put on list. 
                The order of the retrieved values is function of 
                the `properties` disposal.
        """
        #------------------------------------
        from .database import GeoDataBase
        #-----------------------------------
        def _fs (v): 
            """ Sanitize value and put on list 
            :param v: value 
            :Example:
                
                >>> _fs('(416.9, 100000.0)'))
                ...[416.9, 100000.0]
            """
            try : 
                v = float(v)
            except : 
                v = tuple([float (ss) for ss in 
                         v.replace('(', '').replace(')', '').split(',')])
            return v
        # connect to geodataBase 
        try : 
            _dbObj = GeoDataBase()
        except: 
            _logger.debug('Connection to database failed!')
        else:
            _gammaVal = _dbObj._retreive_databasecolumns(properties)
            if sproperty in properties: 
                indexEprops = properties.index(sproperty )
                try:
                    _gammaVal [indexEprops] = list(map(lambda x:_fs(x),
                                                   _gammaVal[indexEprops]))
                except TypeError:
                    _gammaVal= list(map(lambda x:_fs(x),
                                         _gammaVal))
        return _gammaVal
    
    @staticmethod
    def findGeostructures(
        res: float|List[float, ...], /, 
        db_properties=['electrical_props', '__description']
        ): 
        """ Find the layer from database and keep the ceiled value of 
        `_res` calculated resistivities"""
     
        structures = find_similar_structures(res)
        if len(structures) !=0 or structures is not None:
            if structures[0].find('/')>=0 : 
                ln = structures[0].split('/')[0].lower() 
            else: ln = structures[0].lower()
            return ln, res
        else: 
            valEpropsNames = GeoBase.getProperties(db_properties)
            indeprops = db_properties.index('electrical_props')
            for ii, elecp_value  in enumerate(valEpropsNames[indeprops]): 
                if elecp_value ==0.: continue 
                elif elecp_value !=0 : 
                    try : 
                        iter(elecp_value)
                    except : pass 
                    else : 
                        if  min(elecp_value)<= res<= max(elecp_value):
                            ln= valEpropsNames[indeprops][ii]
                            return ln, res

#---------------------------CORE Utilities ------------------------------------
def set_agso_properties (download_files = True ): 
    """ Set the rocks and their properties from inner files located in 
        < 'watex/etc/'> folder."""
        
    msg= ''.join([
        "Please don't move or delete the properties files located in", 
        f" <`{AGSO_PROPERTIES['props_dir']}`> directory."])
    mf =list()
    __agso= [ os.path.join(os.path.realpath(AGSO_PROPERTIES['props_dir']),
                           f) for f in AGSO_PROPERTIES['props_files']]
    for f in __agso: 
        agso_exists = os.path.isfile(f)
        if not agso_exists: 
            mf.append(f)
            continue 
    
    if len(mf)==0: download_files=False 
    if download_files: 
        for file_r in mf:
            success = get_remote_data(props_files = file_r, 
                      savepath = os.path.join(
                          os.path.realpath('.'),
                          AGSO_PROPERTIES['props_dir']), 
                      )
            if not success:
                msg_ = ''.join([ "Unable to retreive the geostructure ",
                      f"{os.path.basename(file_r)!r} property file from",
                      f" {AGSO_PROPERTIES['GIT_REPO']!r}."])
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
    ix_= AGSO_PROPERTIES['props_codes'].index('name')
    def mfunc_(d): 
        """ Set individual layer in dict of properties """
        _p= {c: k.lower() if c not in ('code', 'label', 'name') else k 
                 for c,  k in zip(AGSO_PROPERTIES['props_codes'], d) }
        id_= d[ix_].replace('/', '_').replace(
            ' ', '_').replace('"', '').replace("'", '').lower()
        return id_, _p 
    rock_and_structural_props =list()
    for agso_data in tuple(set_agso_properties(download_files)): 
        # remove the header of the property file
        rock_and_structural_props.append(
            dict(map( lambda x: mfunc_(x), agso_data[1:])))
     
    return  tuple(rock_and_structural_props)

def get_agso_properties(config_file =None, orient ='series'): 
    """ Get the geostructures files from <'watex/etc/'> and 
    set the properties according to the desire type. When `orient` is 
    ``series`` it will return a dictionnary with key equal to 
    properties name and values are the properties items.
    
    :param config_file: Path_Like or str 
        Can be any property file provided that its obey the disposal of 
        property files found in   `AGSO_PROPERTIES`.
    :param orient: string value, ('dict', 'list', 'series', 'split',
        'records’, ''index') Defines which dtype to convert
        Columns(series into).For example, 'list' would return a 
        dictionary of lists with Key=Column name and Value=List 
        (Converted series). For furthers details, please refer to
        https://www.geeksforgeeks.org/python-pandas-dataframe-to_dict/
        
    :Example: 
        >>> import watex.geology.core import get_agso_properties
        >>> data=get_agso_properties('watex/etc/AGSO_STCODES.csv')
        >>> code_descr={key:value for key , value in zip (data["CODE"],
                        data['__DESCRIPTION'])}
    """
    msg= ''.join(["<`{0}`> is the software property file. Please don't move"
        " or delete the properties files located in <`{1}`> directory."])
    
    pd_pos_read = Config().parsers 
    ext='none'
    if config_file is None: 
        config_file = os.path.join(os.path.realpath('.'), os.path.join(
                       AGSO_PROPERTIES['props_dir'],
                       AGSO_PROPERTIES ['props_files'][0]))
    if config_file is not None: 
        is_config = os.path.isfile(config_file)
        if not is_config : 
            if os.path.basename(config_file) in AGSO_PROPERTIES['props_files']:
                _logger.error(f"Unable to find  the geostructure property" 
                              f"{os.path.basename(config_file)!r} file."
                              )
                warnings.warn(msg.format(os.path.basename(config_file) , 
                                         AGSO_PROPERTIES['props_dir']))
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



    