# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Thu Sep 22 11:52:42 2022
#   Revised on Sat Oct  1 15:24:33 2022
"""
GeoDataBase
============
Special class to manage outputs-input requests from-into SQL  database 
Editing this module presume that you are aware of what you  are doing. 
The module is a core of geology sub-packages. 
However the the way the dataBase is arranged can be enhanced  and adapted 
for better convenient or other suitable purposes.

"""

import os
import sys
import numpy as np  
import pandas as pd 
import warnings
import datetime
import shutil 
import sqlite3 as sq3 
# from pg8000 import DBAPI
from ..exceptions import ( 
    GeoDatabaseError, 
    SQLError, 
    SQLManagerError
    )
from .geology import ( 
    Structures
    )
from .._watexlog import watexlog 
_logger = watexlog().get_watex_logger(__name__ )

# let set the systeme path find memory dataBase
for p in ('.', '..', '../..', 'watex/etc'): 
    # for consistency, force system to find the database path.
    sys.path.insert(0, os.path.abspath(p))  

class GeoDataBase (object): 
    """
    Core geological database class. 
    
    Currently we do not create the specific pattern for each geostructures. 
    DataBase is built is built following  structure or property code 
    definition `codef`:: 
        
        `code`,   `label`, `__description`,`pattern`, `pat_size`,`pat_density`,
        `pat_thickness`,`RGBA`, `electrical_props`, `hatch`, `colorMPL`, `FGDC` 

    Parameters 
    -----------
    **geo_structure_name** : str 
        Name of geological rocks , strata or layer.
                
    .. seealso:: 
        FGDC-Digital cartographic Standard for Geological Map Symbolisation. 
    
    """
    #  FGDC is not set yet , we use the  matplotlib pattern symbol makers 
    make_pattern_symbol =["/", "\\", "|", '-', '+', 'x', 'o', 'O', '.', 
                          '*', '\-', '\+', '\o', '\O', '\.', '\*'] 
    #use '\\' rather than '\'.
    # latter , it will be deprecated to FGDC geological map symbolization. 
    
    codef = ['code','label','__description','pattern', 'pat_size',	
             'pat_density','pat_thickness','rgb','electrical_props', 
                 'hatch', 'colorMPL', 'FGDC' ]

    # locate the geodataBase
    geoDataBase = os.path.join(
        os.path.abspath('watex/etc'),'memory.sq3')
 
    # :memory: is faster but we chose the static option :~.sq3 
    # in sql_DB contains drill holes and wells Tables 

    def __init__(
            self, 
            geo_structure_name=None
            ):
        self._logging = watexlog.get_watex_logger(self.__class__.__name__)
        self.geo_structure_name = geo_structure_name
        self.dateTime= datetime.datetime.now().utcnow()   # Get the date time now  
        self.comment =None                                
        
        self._mplcolor =None 
        self._rgb =None 
        self._electrical_props=None 
        self._pattern=None

        try : 
        # to connect geodataBse 
            self.manage_geoDataBase =DBSetting(
                db_host=os.path.dirname(self.geoDataBase), 
                                      db_name ='memory.sq3')
        except : 
            mess =''.join(['Connection to geoDataBase failed! Sorry we can ',
                           'not give a suitable reply for your request!', 
                    'It would  be process with "geological structural class." !'])
            
            warnings.warn(mess)
            self.success= 0
 
        else: 
            self.success = 1 

    def _avoid_injection (self): 
        """
        For secure, we do not firstly introduce directly the request. We will 
        check whether the object `request` exists effectively  in our
        GeoDatabase. If not, request will be redirect to structural and 
        strata classes from  module `structural` to not corrupt the memory.
        
        """
        # self.manage_geoDataBase.executeReq(" select __description  from AGS0")
        
        self.geo_structure_name=self.geo_structure_name.lower() # for consistency 
        manage = self.manage_geoDataBase.curs.execute(
            " select __description  from AGS0")
        __description = [ geoform[0] for geoform in  list(manage)]


        if self.geo_structure_name in __description : 
            self.geo_structure_exists =True 
            
        else :
            mess ='Structure <%s> does not exist  in'\
                ' our GeoDataBase yet.' % self.geo_structure_name
            
            self.geo_structure_exists =False
            warnings.warn(mess)
            self._logging.debug ('Could not find a {0} in our geoDataBase.'\
                                 ' It would be redirect to _strata and '
                                 '_structural classes for suitable processing.'.
                                 format(self.geo_structure_name))
            
            # self.manage_geoDataBase.closeDB() # close the database
            
       # if self.geo_structure_exists : 
       #     self._get_geo_structure()
        
    def _retreive_databasecolumns (self, columns): 
        """ Retreive data from database columns
        
        :param columns: Columns name is `str`. To retreive data of many columns 
                      please put the columns name on list.
        :returns: 
            list of data of each columns.
        
        :Exemple:
            >>> from watex.geology.database import GeoDataBase
            >>> dbObj = GeoDataBase()
            >>>  values = dbObj._retreive_databasecolumns(
                    ['__description', 'electrical_props'])    
        
        """
        
        if isinstance(columns, str): 
            columns =[columns]
            
        new_columns = []
        for obj in columns : 
            if obj =='name'or obj.find('name')>=0: 
                obj = '__description'
            if obj not in self.codef :
                self._logging.debug(
                    f'Object `{obj}` not found in {self.codef}!'
                    'Please provide the right column name.')
                warnings.warn(f'Object `{obj}` not found in {self.codef}!'
                    'Please provide the right column name.')
            else:
                new_columns.append(obj)
            
        if len(new_columns) ==0 : # Object not found in database 
            self._logging.error('None object found in the database !')
            return 
        
        _l=[]  
        
        for obj in new_columns : 
            manage = self.manage_geoDataBase.curs.execute(
                " select %s  from AGS0"% obj)
            valuesdb = [ geoform[0] for geoform in  list(manage)]
            if len(new_columns)==1: 
                _l= valuesdb
            else: _l.append(valuesdb)
            
        self.manage_geoDataBase.closeDB() # close the database
        return _l
            
        
    def _get_geo_structure(self, structure_name=None):
        """
        After checking wether the name of structures exists , 
        let find the geoformation properties   from geodatabase  . 
      
        :param struture_name:   name of geological rock or layer 
        :type struture_name: str
        
        """
        if structure_name is not None :
            self.geo_structure_name = structure_name.lower() 
        
        if self.geo_structure_name is None : 
            warnings.warn('No name is inputted as geological formation. Sorry ,'
                          ' your request is aborted !')
            raise SQLManagerError(
                'No name is given as geological formation. Sorry ,'\
                 ' your request is aborted !')
                
        if self.geo_structure_name is not None :
            self._avoid_injection()
            
        if self.geo_structure_exists : 
            __geonames = list(self.manage_geoDataBase.curs.execute(
                "Select * from AGS0 where __description = '{0}'".\
                  format(self.geo_structure_name.lower())))[0]  
     
            # once value is get then set attribute 
            #for each column of geoDataBase
            for ii, codec in enumerate(self.codef) :
                if self.codef [10] == codec : 
                    self.colorMPL= __geonames[ii]
                    # if '#' in __geonames[ii] : # we assume that value
                    #colorMPL is hexadecimal value eg : #0000ff
                    self.__setattr__(codec, self.colorMPL) # color MPL format
                    # to  string tuple  like '(1.0, 0.5, 0.23)'
                    # else :# value in RGB color (0-1 --> 0 to 255 bits)

                elif self.codef [8]== codec : # set electrical properties
                # (min,max) line (1e-5,5.2e0 )
                    self.electrical_props = __geonames[ii] 
                    self.__setattr__(codec , self.electrical_props) # get 
                    #color matplotlib from property attributes

                else :  self.__setattr__(codec , __geonames[ii])
                
    def _reminder_geo_recorder(self, geo_structure_name ): 
        """
        To have reminder of geological formation into the geodatabase ,
        this method allow to output information 
        if the structure does not exist, An error will occurs. 
        
        :param geo_structure_name: name of geological formation 
        :type geo_structure_name: str 
        
        """
        mess ='--->"{}" Querry  successfully executed ! '  
        
        if geo_structure_name is not None :
            self.geo_structure_name = geo_structure_name.lower() 
        
        if self.geoDataBase is not None : 
            self._avoid_injection() 
        if self.geo_structure_exists is True : 
                # keep only the tuple values 
            __geonames = list(self.manage_geoDataBase.curs.execute(
                "Select * from AGS0 where __description = '{0}'".\
                          format(self.geo_structure_name)))[0] 
            

            print(mess.format(self.geo_structure_name))
            print(__geonames)
            
        
        self.manage_geoDataBase.closeDB() # close the database 
        
                
    def _update_geo_structure (self, geo_formation_name =None, **kws): 
        """
        Update _indormation into geoDataBase . 
        
        Remember that the geodatabase is build following this table
        codef 'code','label','__description','pattern', 'pat_size',
        'pat_density','pat_thickness','rgb','electrical_props',  'hatch',
        'colorMPL', 'FGDC'. 
        
        :param geo_formation_name:  name of formation be sure the formation
                             already exists in the geoDataBase 
                             if not an error occurs 
        :type geo_formation_name: str
        
        - Update the electrical property of basement rocks = [1e99, 1e6 ]
        
        :Example:
            
            >>> from watex.geology.database import GeoDataBase 
            >>> GeoDataBase()._update_geo_structure(
                **{'__description':'basement rocks', 
                    'electrical_props':[1e99, 1e6 ]})
        """

        # find geological rocks name if is in keywords dict
        if geo_formation_name is None : 
            if '__description' in list(kws.keys()) : 
                geo_formation_name=str(kws['__description'])
            elif 'name' in list(kws.keys()) : 
                geo_formation_name=str(kws['name'])
            else : 
                raise SQLError(
                    ' Unable to find a new geological structure name.')
    
        if not isinstance(geo_formation_name, str) : 
            raise SQLError(
                'Unacceptable rock/layer name ={0}.'\
                    ' Please provide a right rock/layer name.')
        geo_formation_name=str(geo_formation_name) # for consistency 
        
        if geo_formation_name is not None :
            self.geo_structure_name = geo_formation_name.lower() 
        
        # build new_dictionnary without the keyname and key value of dictionnay 
        tem_geodict ={geokey:geovalue for geokey , geovalue in kws.items() 
                      if not (geokey =='__description' or geokey =='name') 
                      }
        
        if self.geo_structure_name is not None :
            self._avoid_injection()
        if self.geo_structure_exists is False : 
            mess ="".join([
                ' Actually geological formation name =  <{0}> can '.format(
                    self.geo_structure_name),
                 'not be updated because', 
                ' it doesnt  exist in our DataBase. To set new geological ',
                'formation with their corresponding values ', 
                ' see { _add_geo_structure } method .'])
            self._logging.warn(mess)
            raise SQLError(
                'Update name= {0}  failed ! it doesnt not exist in '
                'geoDataBase'.format(self.geo_structure_name))
                    
            
        elif self.geo_structure_exists : # even the geostructure exists ,
        #let check wether the key provided 
            for geo_key in list (tem_geodict.keys()) : # is among the geocodes keys
                if geo_key not in self.codef : #if key provided not in geodatable keys 
                    
                    mess =''.join([
                        "Sorry the key = {0} is wrong! key doesnt".format(geo_key),
                        " exist in geoDataBase.Please provide a right ", 
                        " keys among = {0}". format(tuple(self.codef[2:]))])
                    self._logging.error(mess)
                    raise SQLManagerError(mess)
        
                elif geo_key in self.codef :
                    if geo_key.find('pat') >= 0 : 
                        try : # keep value to real
                            update_geo_values = float(kws[geo_key]) 
                        except : 
                            msg =''.join([
                                'update failed ! Could not convert',
                                ' value = {0} to float.'.format(kws[geo_key]), 
                                'Please try again later.'])
                            self._logging.error(msg)
                            raise SQLError(msg)
                            
                    # let get the formatage of all values  (properties values )  
                    elif geo_key.find('colorMPL') >=0 : 
                        self.colorMPL = kws[geo_key] 
                        update_geo_values = self.colorMPL
                    # let fill automatically the "rgb" and the colorMPL                
                    elif geo_key.find('rgb')>=0 :  
                        # keep the rgb value g : R125G90B29 and compute the colorMPL
                        self.rgb = kws[geo_key]
                        update_geo_values = self.rgb
                    elif geo_key .find('hatch') >=0 : 
                        self.hatch = kws[geo_key]
                        update_geo_values = self.hatch
                    elif geo_key.find('electrical_props') >=0 : 
                        self.electrical_props =kws[geo_key]
                        update_geo_values = self.electrical_props

                    else : update_geo_values =str (kws[geo_key])
                    # now must be put on the data base
                    # fill automaticall colorMPL when rgb is provided
                    if geo_key.find('rgb') >= 0 :  
                    
                        self.manage_geoDataBase.curs.execute(
                            "update AGS0 set rgb = '{0}'  where __description ='{1}'".
                              format( update_geo_values[0] , 
                                     self.geo_structure_name))
                        self.manage_geoDataBase.curs.execute(
                            "update AGS0 set colorMPL= '{0}'  where __description ='{1}'".
                              format(update_geo_values[1] , 
                                     self.geo_structure_name))
                    
                    else :
                        __oldvalues = list(
                            self.manage_geoDataBase.curs.execute(
                                "Select * from AGS0 where __description ='{0}'".
                                   format(self.geo_structure_name)))[0]
                        self.manage_geoDataBase.curs.execute(
                            "update AGS0 set {0}= '{1}'  where __description ='{2}'".
                               format(geo_key, update_geo_values ,
                                      self.geo_structure_name))
                        __newvalues = list(
                            self.manage_geoDataBase.curs.execute(
                                "Select * from AGS0 where __description ='{0}'".\
                                   format(self.geo_structure_name)))[0]
                    # inpout new info to database from cursor         
                    self.manage_geoDataBase.commit()    
                    
                    if geo_key.find('rgb') >=0 : 
                        print('---> {0} colors was successfully set to '
                              'rgb = {1} & matplotlib rgba = {2}.'.
                              format(self.geo_structure_name,
                                     update_geo_values[0],
                                     update_geo_values[1]))
                    else : 
                        fmt_mess = '---> {0} was successfully set to'\
                            ' geoDataBase.\n ** Old value = {1} \n is '\
                                '**updated to \n New value = {2}'
                        print(fmt_mess.format(
                            self.geo_structure_name,__oldvalues, __newvalues ))
                        
            self.manage_geoDataBase.closeDB() # close the database
            
    @property 
    def hatch (self):
        return self._hatch 
    @hatch.setter 
    def hatch (self, mpl_hatch):
        mm=0  # counter  of hach symbol present on the value provided 
        mpl_hatch =str(mpl_hatch) # for consitency put on string value 
        # removed the tuple sign  "( and )" if provided 
        if '('  in mpl_hatch  : mpl_hatch =mpl_hatch.replace('(', '')
        if   ')' in mpl_hatch: mpl_hatch =mpl_hatch.replace(')', '')
        
        # chech whether the value provided is right 
        if mpl_hatch == 'none' : self._hatch =mpl_hatch 
        else :
            for mpstr  in mpl_hatch : 
                if mpstr in self.make_pattern_symbol  : 
                    mm +=1 
            if len(mpl_hatch) == mm :  #all value are symbols  and put the  
                self._hatch = '(' + mpl_hatch +')'
            else : self._hatch ='none' # abandon value and initialise to None 


    @staticmethod
    def _add_geo_structure( new_geological_rock_name=None , **kws) : 
        """
        Add new _geological information  into geodatabase .
        
        DataBase properties:
        --------------------
            - code
            - label
            - __description
            - pattern 
            - pat_size	
            - pat_density
            - pat_thickness
            - rgb'
            - electrical_props
            - hatch
            - colorMPL
            - FGDC 
            
        .. note:: `__description` could be replaced by `name`.
                    `code` , `label` and `FGDC` dont need to be fill. 
                    Values are rejected if given.
    
        :param new_geological_rock_name: new name of geological formation to add 
        :type new_geological_rock_name: str 
        
        :param informations: 
            dict , must be on keyward keys  when keywords keys 
                are provided , program will check whether all keys are
                effectively the right keys. if not will aborted the process.
        :type informations: dict
        
        :Example: 
            
            >>> from watex.geology.database import GeoDataBase 
            >>> geodatabase_obj= GeoDataBase._add_geo_structure( **{
            ...                                     'name': 'massive sulfure', 
            ...                                     'pattern': 218., 
            ...                                     'pat_size': 250., 
            ...                                     'pat_density': 0.75, 
            ...                                     'pat_thickness': 2., 
            ...                                     'rgb': 'R128B28',
            ...                                     'hatch': '+.+.o.+', 
            ...                                     'electrical_props':[1e0 , 1e-2],
            ...                                     } )
        """
        
        def __generate_structure_code (__description , __geocodeList) : 
            """
            Each input geological description will generate a code and label 

            :param __description: name of geological formation 
            :type __description: str 
  
            :returns: geological formation code 
            :rtype: str 
            
            """
            def _rev_func_code (code, CODE): 
                """
                generate code and check thin the new code does not exist in 
                the database.

                :param code:  new_generate code 
                :type code: str 
                :param CODE: codes already exists in dataBase 
                :type CODE: str 
                
                """
                # actually the lencode is > than 3 
                mm=0
                while code  in CODE :
                    if code not in CODE : 
                        break 
                    if mm > len(code): 
                        mm=0
                    code = code + code[mm]
                    mm=mm+1
            
                return code
                
            # fisrtly code is marked by three ,main letters 
            if len(__description) == 3 :code=  __description.upper() 
            elif len(__description) > 3 : code =__description[:4].upper() 
            if len(__description) < 3 :
                nadd = 0
                code =__description # loop thin you find a code =3 
                while nadd < 2 :
                    if len(code )== 3 : 
                        break 
                    code += code[nadd]
                    nadd +=1 
                code =code.upper()
            # then check whether the new code generate exist or not.
            
            for cof in __geocodeList  : 
                if code  not in __geocodeList : return code 
                if code in __geocodeList : 
                    code =_rev_func_code(code =code , CODE=__geocodeList)  

            return code             # return new code that not exist in geocodes 
            
        _logger.info (
            'Starting process  new geological information into GeoDatabase')
        
        # find geological rocks name if is in keywords dict
        if new_geological_rock_name is None : 
            if '__description' in list(kws.keys()) : 
                new_geological_rock_name=str(kws['__description'])
            elif 'name' in list(kws.keys()) : 
                new_geological_rock_name=str(kws['name'])
            else : 
                raise SQLError(
                    ' ! Unable to find a new geo_logical structure name.')
    
        if not isinstance(new_geological_rock_name, str) : 
            raise SQLError(
                'Unacceptable rocks names ={0}.'
                 ' Please provide a right rock name.')
        new_geological_rock_name=str(new_geological_rock_name) # 
        
        
        # ---------------------------call Geodatabse --------------------------
        
        geoDataBase_obj =  GeoDataBase(new_geological_rock_name)
        
        # initialise to 'none' value 
        mmgeo={geokey : 'none' for geokey in geoDataBase_obj.codef } 
        
        if geoDataBase_obj.success ==1 :  geoDataBase_obj._avoid_injection()
        elif geoDataBase_obj.success  ==0:
            mess = "Connection to SQL geoDataBase failed ! Try again later." 
            warnings.warn(mess)
            raise GeoDatabaseError(mess)
            
        #----------------------------------------------------------------------
       
        if geoDataBase_obj.geo_structure_exists : 
            mess ='! Name {0} already exists in our GeoDataBase. Could not add '\
                'again as new geostructure. Use "_update_geo_geostructure method"'\
                    ' to update infos if you need !'.format(
                        geoDataBase_obj.geo_structure_name)
            warnings.warn(mess)
            _logger.error(mess)
            raise SQLError(mess)
            
        if geoDataBase_obj.geo_structure_exists is False : 
  
            # make an copy of codef useful in the case where 
            # user provided "name" as key instead of "__description" 
            import copy 
            new_codef = copy.deepcopy(geoDataBase_obj.codef)
            #  set the first value of keys to fill 
            mmgeo ['__description'] = str(new_geological_rock_name) 
            if 'name' in list(kws.keys()) : 
                 new_codef[2]= 'name'
            
            # get the list of geo_code values in DataBase 
            geoDataBase_obj.manage_geoDataBase.curs.execute(
                'Select code  from AGS0 ')
            
            __geocode =[codegeo[0] for codegeo in  
                        list(geoDataBase_obj.manage_geoDataBase.curs)]
            
            
            for key in list(kws.keys()) : 

                if key not in new_codef  : 
                    raise SQLError(
                        'Process aborted ! wrong <{0}> key!'
                         ' could not add new informations. '
                         'Please check your key !'.format(key))
                #  set code and labels from geo_description name 
                elif key  in new_codef:  
                
                    try : # check if name is provided intead of  __description 
                    # name (generaly code and label are the same)
                        mmgeo['code'] = __generate_structure_code (
                            new_geological_rock_name, __geocode)
                        mmgeo['label']= __generate_structure_code (
                            new_geological_rock_name,  __geocode)
 
                    except : # user can provide name instead of __description 
                        mmgeo['code'] = __generate_structure_code (
                            new_geological_rock_name, __geocode)
                        mmgeo['label']= __generate_structure_code (
                            new_geological_rock_name, __geocode )
 
                    
                    if key.find('pat')>= 0 :
                        geoDataBase_obj.pattern = kws[key]
                
                        for kvalue  in ['pattern', 'pat_size',
                                        't_density','pat_thickness']: 
                            if key == kvalue : 
                                mmgeo[kvalue]= kws[key]
                                
                    # set RGB value and MPL colors eg : R128G128B --.(0.50, 0.5, 1.0)
                    if key =='rgb' :
                        geoDataBase_obj.rgb= kws[key]
                        # set at the same time rgb value and color MPL 
                        mmgeo['rgb']=  geoDataBase_obj.rgb[0] 
                        mmgeo['colorMPL']=  geoDataBase_obj.rgb[1]
                        
                    # set Matplotlib color whether the rgb is not provided .
                    # if provided will skip 
                    if key=='colorMPL': 
                        if mmgeo['colorMPL'] =='none' : 
                            geoDataBase_obj.colorMPL= kws[key]
                            mmgeo['colorMPL']= geoDataBase_obj.colorMPL
                    # optional keys 
                    
                    if key == 'electrical_props' : 
                        geoDataBase_obj.electrical_props= kws[key]
                        mmgeo['electrical_props']= geoDataBase_obj.electrical_props
                    if key == 'hatch': 
                        
                        mmgeo['hatch']= str(kws[key])
                    if key == 'FGDC': 
                        mmgeo['FGDC']= str(kws[key])
                        
            # print(geoDataBase_obj.success)
            # now build insert all info 
            mm_sql=[]
            for codk in new_codef:  # build info in order and input to GeodataBase
                if codk == 'name' : codk = '__description'
                mm_sql.append(mmgeo[codk])

        
            reqSQL = 'insert into AGS0 ({0}) values ({1})'.format(
                ','.join(['{0}'.format(key) for key in geoDataBase_obj.codef ]),
                ','.join(['?' for ii in range(len(geoDataBase_obj.codef))]))
 
            try : 
                # geoDataBase_obj.manage_geoDataBase.curs.execute(reqSQL , mm_sql )
                geoDataBase_obj.manage_geoDataBase.curs.execute(
                    reqSQL , mm_sql )
                
            except : 
                mess='Process to set {0} infos failed!  Try again '\
                    'later! '.format(new_geological_rock_name)
                warnings.warn (mess)
                _logger.error(mess)
                
                raise SQLError(mess)
                
            else :
   
                geoDataBase_obj.manage_geoDataBase.commit()     
                print('---> new data ={} was successfully set into'
                      ' GeoataBase ! '.format(new_geological_rock_name)) 
            
        geoDataBase_obj.manage_geoDataBase.closeDB() # close the database 
        
                  
    @property 
    def pattern (self):
        "return geopattern"
        return self._pattern 
        
    @pattern.setter 
    def pattern (self, pattern_value):
        "configure geopattern"
        try : 
             float(pattern_value)
        except : 
            mes ='Process aborted ! Could not convert'\
                f' {pattern_value} to float number.'
            self._logging.warning(mes)
            raise SQLError(mes)

        else : 
            self._pattern = float(pattern_value)
    
    @property 
    def colorMPL(self): 
        "return geocolorMPL"
        return self._mplcolor 
    
    @colorMPL.setter 
    def colorMPL (self, mpl_color): 
        """
        configure geocolorMPL
        to set matplotlib _color in rgb value 
        value is range (0 to 1) coding to 0 to 255 bits.
        """
 
        if isinstance(mpl_color, str): # get value from database 
   
            if mpl_color.find('(')>=0 and mpl_color.find(')')>=0 :
                # build the tuple of mpl colors 
                self._mplcolor = tuple([ float(ss) for ss in 
                                         mpl_color.replace(
                                             '(', '').replace(')',
                                                              '').split(',')])
            # we assume that value colorMPL is hexadecimal value eg : #0000ff
            elif '#' in mpl_color : 
                 # assume the colorMPL is in hexadecimal 
                self._mplcolor =  str(mpl_color).lower() 
            elif 'none' in mpl_color : # initilisation  value 
                self._mplcolor =  'none' # keep the value on the dataBase 
            else : 
                import matplotlib as mpl 
                try : # try to convert color to rgba
                
                    self._mplcolor = mpl.colors.to_rgb(str ( mpl_color))
                except : 
                    raise  SQLManagerError(
                        ' Unsupported {0} color!'.format(mpl_color))  
                else :  # keep only R, G, B and abandon alpha . 
                        #Matplotlib give tuple of 4 values 
                        # as (R, G, B, alpha)
                    self._mplcolor =self._mplcolor[:3] 
                    
             # set value to database way 
        elif  isinstance(mpl_color, (list, tuple, np.ndarray)): 
            if 3 <len(mpl_color) < 3 : 
                msg =''.join(['update failed ! value = {0} '.format(mpl_color),
                              'must be a tuple of 3 values= (Red, Green, Blue)',
                             'values. Please provided a right number', 
                             '  again later.'])
                
                self._logging.error(msg)
                raise SQLError(msg)
            # let check whether the value provided can be converted to float    
            if len(mpl_color)==3 :  
                try : 
                     self._mplcolor= tuple( [float(ss) for
                                             ss in list( mpl_color)])
                except : 
                     msg =''.join(['update failed ! Could not convert value ',
                                   '= {0} to float.'.format(mpl_color), 
                                   'Please try again later.'])
                     self._logging.error(msg)
                     raise SQLError(msg)
                else : 
                    # try to check if value is under 1.
                    # because color is encoding to 1 to 255 bits 
                    for ival in  self._mplcolor: 
                        if 1 < ival <0  : 
                            if ival > 1 : fmt ='greater than 1' 
                            elif  ival <0 :
                                fmt= 'less than 0'
                            msg = ''.join([
                                'update failed ! Value provided  =',
                                f' `{ival}` is UNacceptable value ! Input ',
                                f' value is {fmt}. It must be encoding from ',
                                '1 to 255 bits as MPL colors.'])
                            raise SQLError(msg)
                            
                self._mplcolor=str( self._mplcolor) # put on str for consistency 
                
    @property 
    def rgb(self):
        "return georgb"
        return self._rgb 
    @rgb.setter 
    def rgb(self, litteral_rgb): 
        """
        configure georgb
        
        .. note:: 
            Return the rgb value and the convert rgb palette value: 
            keep the rgb value eg `R125G90B29` and compute the colorMPL
            let fill automatically the "rgb" and the colorMPL 
        """
        from ..utils.plotutils import get_color_palette
 
        self._rgb=(litteral_rgb, str(
            get_color_palette(RGB_color_palette=litteral_rgb)))
        
    @property 
    def electrical_props(self): 
        "return electrical property"
        return self._electrical_props
    
    @electrical_props.setter 
    def electrical_props(self,range_of_rocks_resvalues):
        """
        configure electrical property
        
        .. note:: Electrical_property of rocks must a tuple of resisvity ,
                 max and min bounds  eg : [2.36e-13, 2.36e-3]
        """
        # electrical props were initialised by float 0. 
        if isinstance(range_of_rocks_resvalues , str) : 
            if '(' in range_of_rocks_resvalues  : 
                self._electrical_props = tuple([ 
                    float(ss) for ss in # build the tuple of mpl colors 
                    range_of_rocks_resvalues .replace('(',
                                                '').replace(')',
                                                            '').split(',')]) 
            elif 'none' in range_of_rocks_resvalues : 
                self._electrical_props =.0
                
        elif isinstance(range_of_rocks_resvalues,(list,tuple, np.ndarray)): 
            if len(range_of_rocks_resvalues) ==2  : 
                try : 
                    self._electrical_props =[float(res) 
                                       for res in range_of_rocks_resvalues]
                except : 
                    raise SQLError(
                        ' !Could not convert input values to float.')
                else :# range the values to min to max 
                    self._electrical_props =sorted(self._electrical_props) 
                    self._electrical_props = str(tuple(self._electrical_props))
            else : 
                # force program to format value to 0. float 
                self._electrical_props = .0   
                    
        elif not isinstance(range_of_rocks_resvalues,
                            (list,tuple, np.ndarray)) or len(
                                range_of_rocks_resvalues)!=2:
            try : # 0 at initialization
                range_of_rocks_resvalues = float(range_of_rocks_resvalues)  
            except  : 
                if len(range_of_rocks_resvalues) > 1: fmt ='are'
                else :fmt ='is'
    
                mess = ''.join([
                    'Unable to set electrical property of rocks.', 
                    ' We need only minimum and maximum resistivities',
                    ' bounds. {0} {1} given'.format(
                        len(range_of_rocks_resvalues), fmt)])
                        
                self._logging.error(mess)
                warnings.warn(mess)
                raise SQLError(mess)
            else : 
                
                self._electrical_props = .0 #mean value  initialised 
                

    @property
    def _setGeoDatabase(self): 
        """
        .. note:: 
            Property of  GeoDataBase -create the GeoDataBase
            Setting geoDataBase table
            No Need to reset the DataBase at least you dropped the table,
            avoid to do that if you are not sure of what you are doing. 
        """
        from ..utils.funcutils import concat_array_from_list
        #set other attributes
        # create connection 
        # try : 
        #     # call DBSetting DB  so to connect geodataBse 
        #     manage_geoDataBase =DBSetting(db_host=os.path.dirname(self.geoDataBase), 
        #                                  db_name ='memory.sq3')
        # except : pass 
        # if success is True  : 
        # createa tABLE IN THE DATABASE
        req  = ''.join(['create table AGS0', 
                      ' (', 
                      '{0} TEXT,', 
                       ' {1} TEXT,', 
                       ' {2} TEXT,', 
                       ' {3} REAL,', 
                       ' {4} REAL,', 
                       ' {5} REAL,', 
                       ' {6} REAL,', 
                       ' {7} TEXT,', 
                       ' {8} REAL,', 
                       ' {9} TEXT,', 
                       ' {10} TEXT,', 
                       ' {11} TEXT', 
                       ')'
                       ]) 
        
        # create Request to insert value into table 
        mes ='insert into AGS0 ('+ ','.join(['{}'.format(icode)
                                             for icode in self.codef])
        enter_req = mes+ ')'
        

        # create  Geoformation objets 
        geo_formation_obj =Structures()      # set oBject of geostructures 
        
        new_codef ,  new_codef[-1], new_codef[0]= self.codef [:8], 'color', 'codes'
        # get attribute from geoformation and #build new geo_formations _codes 
        geo_form_codes =[ getattr(geo_formation_obj, codehead) 
                         for codehead in new_codef ]
        geo_form_codes =concat_array_from_list(geo_form_codes,
                                                    concat_axis=1)
        # generate other main columns of tables to fill laters 
        geo_add_form_codes = concat_array_from_list(
            list_of_array = [ np.zeros((len(geo_formation_obj.codes),)), 
                            np.full((len(geo_formation_obj.codes),), 'none'), 
                            np.array([str (clsp) 
                                      for clsp in geo_formation_obj.mpl_colorsp]),
                            np.full((len(geo_formation_obj.codes),),'none'),
                                                                      ],
                                                         concat_axis=1 )
        # create Table resquest 
        
        req = req.format(*self.codef)              # generate a request for table creation 
        enter_req = enter_req.format(*self.codef)  # generate interrequest 

        GDB_DATA = np.concatenate((geo_form_codes,geo_add_form_codes), axis =1)
        
        # generate values Host string so to avoid injection 
        # print(req)
        values_str = 'values (' + ','.join(['?' 
                                for itg in range( GDB_DATA.shape[1])]) +')'
        insert_request = ''.join([enter_req , values_str])

        # create Table
        
        try : 
            self.manage_geoDataBase.executeReq(query=req ) 
            
        except : 
            warnings.warn('Could not create {AGS0} Table !')
            raise GeoDatabaseError(
                'Table AGS0 already exists !')

        if self.manage_geoDataBase.success ==1: 
            # enter the record 
            for ii, row_geoDataBase in enumerate(GDB_DATA ): 
                row_geoDataBase =tuple(row_geoDataBase)
                self.manage_geoDataBase.executeReq(
                    query=insert_request , param =row_geoDataBase ) 
        
        self.manage_geoDataBase.commit()
        self.manage_geoDataBase.closeDB()

           
            
class DBSetting(object) : 
    """
    build a datable postgre Sql  from dict_app.py  simple way to make a transit
    between two objects One object dict_app to populate DataBase
        
    Parameters 
    ------------
    **db_name** : str  
        name of dataBase 
    **db_host** : st 
         path to database 

    Hold other additional informations: 
        
    ====================  ==============  ==================================== 
    Attributes              Type            Explanation 
    ====================  ==============  ==================================== 
    connex                  object            DataBase connection 
    curs                    object            Database cursor
    ====================  ==============  ==================================== 

    ==========================  ===============================================
     Methods                     Explanation 
    ==========================  ===============================================
    dicT_sqlDB                  send infos as  dictionnary to dataBase 
    execute req                 execute a sql_request
    drop_TableDB                drop all Tables in sql memory DB or single Table 
    closeDB                     close after requests the connection and the cursor
    commit                      transfer the data to DataBase. if not the data 
                                will still in the cursor and  not in the dataBase 
    print_last_Query            print the last operating system 
    export_req                  export the request on datasheet like excelsheet . 
    ==========================  ===============================================
        
        
    Examples
    ----------
    >>> from watex.geology.database import DBSetting 
    >>> path= os.getcwd()
    >>> nameofDB='memory.sq3'
    >>> manDB=DBSetting(db_name=nameofDB, 
    ...                   db_host=path)
    ... print(SqlQ.sql_req[-1])
    ... manDB.executeReq(SqlQ.sql_req[2])
    ... ss=manDB.print_last_Query()
    ... print(ss)
    ... manDB.export_req(SqlQ.sql_req[-1],
                         export_type='.csv')
    ... manDB.dicT_sqlDB(dictTables=Glob.dicoT, 
                       visualize_request=False)
    """
    def __init__(self, db_name =None, db_host=None): 
        self._logging = watexlog.get_watex_logger(self.__class__.__name__)
        
        self.db_host=db_host 
        self.db_name=db_name
        
        if self.db_name is not None  : 
            self.connect_DB()
            
            
    def connect_DB(self, db_host=None , db_name=None): 
        """
        Create sqqlite Database 
        
        :param db_host:   DataBase location path
        :type db_host: str
        
        :param db_name: str , DataBase name 
        :type db_name: str 
        """
        if db_host is not None :
            self.db_host = db_host 
        if db_name is not None : 
            self.db_name = db_name 
        
        mess= ''
        if self.db_name is None  :
            mess ='Could not create a DataBase ! Need to input the DataBase name.'
            
        if self.db_host is None : 
            mess ='Could not create a DataBase : No "{0}" Database path detected.'\
                ' Need to input  the path for Database location.'.format(self.db_name)
        
        if mess !='': 
            warnings.warn(mess)
            self._logging.error(mess)

        # try to connect to de dataBase 
        if self.db_host is not None :
            try : 
                self.connexDB=sq3.connect(os.path.join(self.db_host, self.db_name))
            except :
                warnings.warn("Connection to SQL %s failed !." %self.db_name)
                
                self.success=0
            else : 
                self.curs =self.connexDB.cursor()
                self.success=1
        
    def dicT_sqlDB(self, dicTables, **kwargs): 
        """
        Method to create Table for sqlDataBase . 
        Enter Data to DataBase from dictionnary. 
        Interface objet : Database _Dictionnary  
        to see how dicTable is arranged , may consult dict_app module 
                
        Parameters
        ----------
        * dictTables: dict
            Rely on dict_app.py module. it populates the  datababse 
            from dictionnay app 
               
        Returns
        ---------
        req: str 
          Execute queries from dict_app 
            
        Examples
        -----------
        >>> from watex.geology.database import DBSetting 
        >>> mDB=DBSetting (dbname='memory.sq3, 
        ...                   db_host =os.getcwd()')
        >>> mDB.dicT_sqlDB(dicTables=Glob.dicoT,
        ...                  visualize_request=False)
        >>> ss=mB.print_last_query()
        >>> print(ss)
        """
        visual_req=kwargs.pop('visualize_request', False)
        
        field_dico ={'i':'INTEGER',"t":"TEXT",'b':'BOOL',
                     'd': 'HSTORE',"k": "SERIAL", 'n':'NULL',
                     "f": 'REAL','s':'VARCHAR','date':'date',
                     'by':'BYTHEA','l':'ARRAY',
                     }

        for table in dicTables: 
            req="CREATE TABLE %s (" % table
            pk=""
            for ii, descrip in enumerate(dicTables[table]): 
                field=descrip[0]
                tfield=descrip[1] # Type of field 
                # for keys in  field_dico.keys():
                if tfield in field_dico.keys():
                    # if tfield == keys : 
                    typefield=field_dico[tfield]
                else :
                    # sql vriable nom :'s':'VARCHAR'
                    typefield='VARCHAR(%s)'%tfield
                        
                req= req+'%s %s, ' %(field, typefield)
                

            if pk=='': 
                req=req [:-2] + ")" # delete the last ',' on req.
            else : 
                req =req +"CONSTRAINT %s_pk PRIMARY KEYS(%s))" %(pk,pk)
                
            if visual_req is True : # print the request built .
                print(req)                
            try : 
                self.executeReq(req)
            except : 
                pass # the case where the table already exists. 
            
        return req

    
    def executeReq(self, query, param=None):
        """
        Execute request  of dataBase  with detection of error. 
        
        Parameters  
        -----------
        * query: str  
            sql_query 
        * param: str 
            Default is None . 
        
        raise 
        -------
            Layout of the wrong sql queries . 
            
        return  
        -------
         True or False: int  
             TWether the request has been successuful run  or not.
              
        Examples
        -----------
        >>> from watex.geology.database import DBSetting
        >>> for keys in Glob.dicoT.keys(): 
        ...        reqst='select * from %s'%keys
        >>>  ss=DBSetting(dbname='memory.sq3, db_host =os.getcwd()'
                          ).executeReq(query=reqst)
        >>>  print(ss)
        """
        
        try :
            if param is None :
                self.curs.execute(query)# param)
            else :
                self.curs.execute(query, param)
                
        except: 
            warnings.warn(f'Request SQL {query} failed. May trouble of SQL  '
                          'server connexion. Please try again later ')
            # raise (f'Request SQL {query}executed failed',err)
            return 0
        else : 
            return 1
    
    def drop_TableDB(self, dicTables, drop_table_name=None ,
                     drop_all=False): 
        """
        Drop the name of table on dataBase or all databases.

        Parameters
        ----------
        * dicTables : dict
                application dictionnary. Normally provide from 
                dict_app.py module 
            
        * drop_table_name : str, optional
                field name of dictionnay (Table Name). 
                The default is None.
            
        * drop_all : Bool, optional
                Must select if you need to drop all table. 
                The default is False.

        Note
        ------
        Raise an exception of errors occurs. 

        """
        
        if drop_all is False and drop_table_name is None : 
            raise 'Must be input at least one name contained of keys in the dicT_app'
        elif drop_all is True : 
            for keys in dicTables.keys():
                req="DROP TABLE %s" %keys
                self.executeReq(req)
        elif drop_table_name is not None : 
            if drop_table_name in dicTables.keys(): 
                req="DROP TABLE %s" % drop_table_name
            else : 
                raise'No such name in the dictionnary application Table!'\
                    'Dict_app keys Tables Names are : ¨{0}'.format(dicTables.keys())
                    
            self.executeReq(req)
            
        self.connexDB.commit()
    
    def closeDB(self):
        """
        simple method to close Database. 
        """
        
        if self.connexDB : 
            # self.curs.close()
            self.connexDB.close()
            
    def commit(self) :
        """
        special commit method for the database when cursor  and connexion 
        are still open.
        """
        if self.connexDB : 
            self.connexDB.commit()
    
    def print_query(self, column_name=None ) : 
        """
        return the result of the previous query.
        
        Parameters 
        ------------
        * query_table_nam : str
            name of table to fetch colounm data .
        """
        if  column_name is not None :
            return self.curs.fetchone()
        else :
            return self.curs.fetchall()
    
    def export_req(self, query =None ,
                   export_type='.csv', **kwargs):
        """
        method to export data  from DataBase 

        Parameters
        ----------
        * query : str, optional
            Sql requests. You may consult sql_request files. 
            The default is None.
        * export_type : Str, optional
            file extension. if None , it will export on simple file. 
            The default is '.csv'.
        * kwargs : str
            Others parameters.

        Returns
        ----------
        None: 
            Print wrong SQL request messages. 

        Example
        ----------
        >>> from watex.geology.database import DBSetting
        >>> from sqlrequests import SqlQ
        >>> DBSetting(dbname='memory.sq3, db_host =os.getcwd()
                      ).executeReq(SqlQ.sql_req[2])
        >>> ss=manageDB.print_last_Query()
        >>> print(ss)
        >>> manageDB.export_req(SqlQ.sql_req[-1],
                            export_type='.csv',
                            )
        """
        
        exportfolder=kwargs.pop('savefolder','savefiles')
        filename =kwargs.pop('filename','req_file')
        indexfile=kwargs.pop('index',False)
        headerfile=kwargs.pop("header",True)
        
        if query is None : 
            raise Exception ("SQL requests (%s) no found ! Please Try to put "\
                             " your sql_requests"% query)
        elif query is not None : 
            
            df_sql=pd.read_sql_query(query,self.connexDB)
            
        if filename.endswith('.csv'):
            export_type ='.csv'
            filename=filename[:-4]
        elif filename.endswith(('.xlxm', '.xlsx', '.xlm')): 
            export_type='.xlsx'
            filename=filename[:-5]
        else : 
              assert export_type  is not None , 'Must input the type to export file.'\
                  ' it maybe ".csv" or ".xlsx"'

        #-----export to excel sheet  
        if export_type in ['csv','.csv', 'comma delimited',
                             'comma-separated-value','comma sperated value',
                                       'comsepval']:
            # export to excelsheet:  
            df_sql.to_csv(filename+'.csv', header=headerfile,
                  index =indexfile,sep=',', encoding='utf8')
            
            sql_write =1 
        elif export_type in ['xlsx','.xlsx', 'excell',
                             'Excell','excel','Excel','*.xlsx']: 
            df_sql.to_excel(filename+'.xlsx',sheet_name=filename[:-3],
                index =indexfile)
            sql_write =0

        #wite a new folder
        if exportfolder is not None : 
            try : 
                os.mkdir('{0}'.format(exportfolder))    
            except OSError as e :
                print(os.strerror(e.errno))
        sql_path=os.getcwd()
        savepath=sql_path+'/{0}'.format(exportfolder)
        
        if sql_write ==1 : 
            shutil.move(filename +'.csv', savepath)
            print('---> outputDB_file <{0}.csv> has been written.'.format(filename))
        elif sql_write ==0 :  
            shutil.move(filename +'.xlsx',savepath)
            print('---> outputDB_file <{0}.xlsx> has been written.'.format(filename))


            
                

                
            
            
            
