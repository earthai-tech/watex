# -*- coding: utf-8 -*-
# Created on Tue May 17 11:30:51 2022
#       Author: LKouadio <etanoyau@gmail.com>
#       Licence: MIT

"""
Module EM 
==========

The EM module is related for a few meter exploration in the case of groundwater 
exploration. Module provides some basics processing step for EMAP data fitering
and remove noises. Commonly the methods mostly used in the groundwater 
exploration is the audio-magnetoteluric because iof the shortest frequency 
and rapid executions. Furthermore, we can also listed some other advantages 
such as: 
    
    * is useful for imaging both deep geologic structure and near-surface 
        geology and can provide significant details.(ii) 
    *  includes a backpack portable system that allows for use in difficult 
        terrain. 
    * the technique requires no high-voltage electrodes, and logistics 
        are relatively easy to support in the field. Stations can be acquired 
        almost anywhere and can be placed any distance apart. This allows for
        large-scale regional reconnaissance exploration or detailed surveys of 
        local geology and has no environmental impact 

:notes: For deep implementation or explorating a large scale of EM/AMT data  
    processing, it is recommended to use the package `pycsamt`_. 
    
"""
from __future__ import annotations 

import warnings 
import os
import re
import functools 
import numpy as np 

from .._watexlog import watexlog
from ..exceptions import ( 
    EDIError, 
    TopModuleError, 
    FitError, 
    EMError, 
) 
from ..tools.funcutils import ( 
    is_installing,
    _assert_all_types, 
    make_ids, 
    show_stats, 
    fit_by_ll, 
    reshape, 
    smart_strobj_recognition, 
    repr_callable_obj, 
    
    ) 
from ..tools.exmath import ( 
    scalePosition, 
    fittensor, 
    betaj, 
    interpolate1d,
    interpolate2d, 
    rhoa2z, 
    z2rhoa, 
    mu0, 
    )
from ..bases.site import Location 
from ..tools.coreutils import ( 
    makeCoords, 
    )
from ..property import (
    IsEdi 
    )
from ..typing import ( 
    ArrayLike, 
    Optional, 
    List,
    Tuple, 
    Dict, 
    NDArray, 
    DType,
    EDIO,
    T,
    F, 
    )

HAS_MOD=False 

try : 
    import pycsamt 
except ImportError: 
    HAS_MOD=is_installing (
            'pycsamt'
            )
    if not HAS_MOD: 
        warnings.warn(" Package 'pycsamt' not found. Please install it"
                      " mannually") 
        raise TopModuleError( "Module Not found. Prior install the module"
                             "`pycsamt` instead.")
else : 
    HAS_MOD=True 
    
if HAS_MOD : 
    from pycsamt.ff.core import (
        edi, 
        z as EMz 
        ) 

_logger = watexlog.get_watex_logger(__name__)

class EM(IsEdi): 
    """
    Create EM object as a collection of EDI-file. 
    
    Collect edifiles and create an EM object. It sets  the properties from 
    audio magnetotelluric,two(2) components XY and YX will be set and calculated.
    Can read MT data instead, However the full handling transfer function like 
    Tipper and Spectra  is not completed. Use  other MT softwares for a long 
    periods data. 
    
    Arguments 
    ---------
    survey_name: str 
        location name where the date where collected . If surveyname is None  
        can chech on edifiles. 

    longitude: array-like, shape (N,) 
        longitude coordinate values  collected from EDIs 
        
    latitude: array-like, shape (N, )
        Latitude coordinate values collected from EDIs 
        
    elevation: array-like, shape (N,) 
        Elevation coordinates collected from EDIs 

    res_xy|res_yx :dict
         {stn: res_xy|res_yx} ndarray value of resivities from 2 comps xy|yx 
         where 'stn' is station name. 
         
    phs_xy|phs_yx: dict 
         dict <{stn: res_phs|phs_yx}>  ndarray value of phase from 2  comps xy|yx 
         
    z_xy|res_yx:dict
        dict < {stn: z_xy|z_yx}>  (in degree) ndarray value of impedance from 
        2 comps xy|yx 
        
    XX_err_xy|XX_err_yx:  dict, 
           dict  of error values {stn: XX_err_xy|XX_err_yx} ndarray value of 
           impedance from 2 comps xy|yx XX : res|phs|z  stn : name of site eg
           stn :S00, S01 , .., Snn
    """

    def __init__(self, survey_name:str  =None ): 
        self._logging = watexlog.get_watex_logger(self.__class__.__name__)
    
        self.survey_name =survey_name
        self.Location_= Location()
      
        self._latitude = None
        self._longitude=None
        self._elevation= None 
        self.ediObjs_ = None 
        self.data_= None 


    @property 
    def latitude(self): 
        return self._latitude 

    @latitude.setter 
    def latitude(self, latitude):
        self._assertattr ( 'latitude', latitude,
            self.Location_.lat
            )

    @property 
    def longitude(self): 
        return self._longitude 
    
    @longitude.setter 
    def longitude(self, longitude):
        self._assertattr ( 'longitude', longitude,
            self.Location_.lon 
            )

    @property 
    def elevation(self): 
        return self._elevation 
    
    @elevation.setter 
    def elevation(self, elevation):
        self._assertattr ('elevation', elevation,
            self.Location_.elev 
            )
        
    @property 
    def stnames(self):
        return self._station_names
    @stnames.setter 
    def stnames (self, edi_stations):
        try : _assert_all_types(edi_stations, list,
                                tuple, np.ndarray)
        except : self._station_names = self.id_ 
        else : self._station_names = list(
            map(lambda n: n.replace('.edi', ''), edi_stations))
        if len(set (self._station_names)) ==1 : 
            self._station_names = self.id 
            
        if len(self._station_names) != len(self.ediObjs_): 
            self._station_names = self.id  
    
    def is_valid (self, 
        obj: str | EDIO 
        )-> edi.Edi  : 
        """Assert that the given argument is an EDI -object from modules 
        EDi of pyCSAMT and MTpy packages. A TypeError will occurs otherwise.
        
        :param obj: Full path EDI file or `pycsamt`_.
        :type obj: str or str or  pycsamt.core.edi.Edi or mtpy.core.edi.Edi 
        
        :return: Identical object after asserting.
        
        """
        IsEdi.register (edi.Edi)
        if isinstance(obj, str):
            obj = edi.Edi(obj) 
        try : 
            obj = _assert_all_types (obj, IsEdi)
        except AttributeError: 
            # force checking instead
            obj = _assert_all_types (obj, edi.Edi)
            
        return  obj 
              
    def _assertattr (self, name, value,  locprop ): 
        """ Read and set attributes from location object . 
        For instance:: 
            >>> name = 'longitude',object = self.Location.lon
            >>> self._longitude = np.array ( 
            ...     list (map ( lambda o : self.Location.lon , 
            ...                longitude)))
        """
        if isinstance(value, (float, int)): 
            value= [value]
        s=np.zeros_like(value)
        for i, p in enumerate(value): 
            locprop = p ; s[i] = locprop 
        setattr(self, f'_{name}', s )
        
        
    def fit (self, 
             data: str|List[EDIO]
             ):
        """
        Assert and make EM object from a collection EDIs. 
        
        Parameters 
        ----------- 
        data : str, or list or :class:`pycsamt.core.edi.Edi` object 
            Edi or collection of Edis or EDI-objects 
            
        Returns
        -------- 
        self: EM object from a collection EDIs 
        
        Examples 
        --------
        >>> from watex.methods.em import EM 
        >>> emObjs = EM().fit (r'data/edis')
        >>> emObjs.ediObjs_ 
        ... 

        """
        def _fetch_headinfos (cobj,  attr): 
            """ Set attribute `attr` from collection object `cobj`."""
            return list(map (lambda o: getattr(o, attr), cobj))
        
        self._logging.info (
            'Read, collect EDI-objects from <%s>'% self.__class__.__name__)
        
        rfiles =[] # count number of reading files.
        

        self.data_ = data
        
        # if ediObjs is not None: 
        #     self.ediObjs_ = ediObjs 
            
        if isinstance(self.data_, str): 
            # single edi and get the path 
            if os.path.isfile (self.data_): 
                if not self._assert_edi(self.data_, False): 
                    raise EDIError(" Unrecognized SEG- EDI-file. Follow the "
                                   "[Wight, D.E., Drive, B., 1988.] to build a"
                                   " correct EDI- file.")
                #edipath = os.path.dirname (self.data_) 
                rfiles=[os.path.dirname (self.data_)]
                self.ediObjs_ = np.array ([self.is_valid(self.data_)]) 
                #self.data_=[self.data_]
                
            elif os.path.dirname (self.data_): 
                # path is given and read  
                rfiles= os.listdir(self.data_) 
                
                self.data_= sorted ([ os.path.join(self.data_, edi ) 
                    for edi in rfiles if edi.endswith ('.edi')])  
     
            else : 
                raise EDIError (f"Object {self.data_!r} is not an EDI object!")

        elif isinstance(self.data_, (tuple, list)): 
            self.data_= sorted(self.data_)
            rfiles = self.data_.copy() 

        if self.data_ is not None:
            try :
                self.ediObjs_ = list(map(
                    lambda o: self.is_valid(o), self.data_)) 
            except: 
                # in the case a single object is given at the param
                # the list-of edifiles rather than ediObjs
                self.ediObjs_ = list(map(
                    lambda o: self.is_valid(o), [self.data_])) 
                
        # for consistency 
        if self.ediObjs_ is not None:
            if not isinstance (self.ediObjs_,(list,tuple, np.ndarray)): 
                 self.ediObjs_ = [self.ediObjs_]
        
            rfiles = self.ediObjs_
                
            try:
                self.ediObjs_ = list(map(
                    lambda o: self.is_valid(o), self.ediObjs_)) 
            except EDIError: 
                raise EMError ("Expect a list of EDI objects not "
                                  f"{type(self.ediObjs_[0]).__name__!r}")
            
        # sorted ediObjs from latlong  
        self.ediObjs_ , self.edinames = fit_by_ll(self.ediObjs_)
        # reorganize  edis in lon lat order. 
        self.edifiles = list(map(lambda o: o.edifile , self.ediObjs_))

        try:
            show_stats(rfiles, self.ediObjs_)
        except: pass 
        
        #--get coordinates values and correct lon_lat ------------
        lat  = _fetch_headinfos(self.ediObjs_, 'lat')
        lon  = _fetch_headinfos(self.ediObjs_, 'lon')
        elev = _fetch_headinfos(self.ediObjs_, 'elev')
        lon,*_ = scalePosition(lon) if len(self.ediObjs_)> 1 else lon 
        lat,*_ = scalePosition(lat) if len(self.ediObjs_)> 1 else lat
        # ---> get impednaces, phase tensor and
        # resistivities values form ediobject
        self._logging.info('Setting impedances and phases tensors and'
                           'resisvitivity values from a collection ediobj.')
        zz= [edi_obj.Z.z for edi_obj in self.ediObjs_]
        zz_err= [edi_obj.Z.z_err for edi_obj in self.ediObjs_]

        rho= [edi_obj.Z.resistivity for edi_obj in self.ediObjs_]
        rho_err= [edi_obj.Z.resistivity_err for edi_obj in self.ediObjs_]

        phs= [edi_obj.Z.phase for edi_obj in self.ediObjs_]
        phs_err= [edi_obj.Z.phase_err for edi_obj in self.ediObjs_]
        
        # Create the station ids 
        self.id = make_ids(self.ediObjs_, prefix='S')
    
        self.longitude= lon 
        self.latitude= lat  
        self.elevation= elev
        
        # get frequency array from the first value of edifiles.
        self.freq_array = self.ediObjs_[0].Z.freq

        #---> set ino dictionnary the impdance and phase Tensor 
        self._z = {key:value for key , value in zip (self.id, zz)}
        self._z_err ={key:value for key , value in zip (self.id, zz_err)}

        self._res = {key:value for key , value in zip (self.id, rho)} 
        self._res_err ={key:value for key , value in zip (self.id, rho_err)}

        self._phs ={key:value for key , value in zip (self.id, phs)}
        self._phs_err ={key:value for key , value in zip (self.id, phs_err)}
        
        self.stnames = self.edinames 
        
        self.freqs_ = self.getfullfrequency (self.ediObjs_)
        self.refreq_ = self.getreferencefrequency(self.ediObjs_)
        
        return self 
    
    def rewrite (self, 
                 data:str|List[EDIO] =None,
                 *,  
                 by: str  = 'name', 
                 prefix: Optional[str]  = None, 
                 dataid: Optional[List[str]] =None, 
                 savepath: Optional[str] = None, 
                 how: str ='py', 
                 correct_ll: bool =True, 
                 make_coords: bool =False, 
                 reflong: Optional[str | float] =None, 
                 reflat: Optional[str | float]=None, 
                 step: str  ='1km',
                 edi_prefix: Optional[str] =None, 
                 export: bool =True, 
                 **kws
                 )-> object: 
        
        """ Rewrite Edis, correct station coordinates and dipole length. 
        
        Can rename the dataid,  customize sites and correct the positioning
        latitudes and longitudes. 
        
        Parameters 
        ------------
  
        data: Path-like object for  list of pycsamt.core.edi.Edi objects
            Collection of edi object from pycsamt.core.edi.Edi 
        dataid: list 
            list of ids to  rename the existing EDI-dataid from  
            :class:`Head.dataid`. If given, it should match the length of 
            the collections of `ediObjs`. A ValueError will occurs if the 
            length of ids provided is out of the range of the number of EDis
            objects 

        by: str 
            Rename according to the inner module Id. Can be ``name``, ``id``, 
            ``number``. Default is ``name``. If :attr:`~.EM.survey_name`
            is given, the whole survey name should be overwritten. Conversly, the  
            argument ``ix`` outputs the number of formating stations excluding 
            the survey name. 
            
        prefix: str
            Prefix the number of the site. It could be the abbreviation   
            of the survey area. 

        correct_ll: bool,
            Write the scaled positions( longitude and latitude). Default is
            ``True``. 
            
        make_coords: bool 
            Useful to hide the real coordinates of the sites by generating 
            a 'fake' coordinates for a specific purposes. When setting to ``True``
            be sure to provide the `reflong` and `reflat` values otherwise and 
            error will occurs. 
            
        reflong: float or string 
            Reference longitude  in degree decimal or in DD:MM:SS for the  
            site considered as the origin of the lamdmark.
            
        reflat: float or string 
            Reference latitude in degree decimal or in DD:MM:SS for the reference  
            site considered as the landmark origin.
            
        step: float or str 
            Offset or the distance of seperation between different sites in meters. 
            If the value is given as string type, except the ``km``, it should be 
            considered as a ``m`` value. Only meters and kilometers are accepables.
            Default value of seperation between the site is ``1km``. 
             
        savepath: str 
            Full path of the save directory. If not given, EDIs  should be 
            outputed in the created directory. 
    
        how: str 
            The way to index the stations. Default is the Python indexing
            i.e. the counting starts by 0. Any other value will start counting 
            the site from 1.
            
        export: bool, 
            Export new edi-files 
            
        kws: dict 
            Additionnal keyword arguments from `~Edi.write_edifile` and 
            :func:`watex.tools.coreutils.make_ll_coordinates`. 
            
        Returns 
        --------
        EM: :class:`~.EM` instance  
            An EM object. 
            
        Examples
        ---------
        >>> from pycsamt.core.edi import Edi_Collection
        >>> edipath = r'/Users/Daniel/Desktop/edi'
        >>> savepath =  r'/Users/Daniel/Desktop/ediout'
        >>> cObjs = Edi_collection (edipath)
        >>> cObjs.rewrite_edis(by='id', edi_prefix ='b1',savepath =savepath)
        
        """
        def replace_reflatlon (  olist , nval, kind ='reflat'):
            """ Replace Definemeaseurement Reflat and Reflong by the interpolated
            values.
            
            :param olist: old list composing the read EDI measurement infos.
            :type olist: list 
            :param nval: New reflat or reflong list. Mostly is the DD:MM:SS 
                value interpolated. 
            :param kind: Type of measurement to write. 
            :type kind:str 
            
            :return: List of old element replaced. 
            :rtype: list 
            """
            try : 
                for ii, comp in enumerate (olist):
                    if comp.strip().find(kind)>=0: 
                        olist[ii]= f' {kind}={nval}\n'
                        break 
            except:
                pass
            return olist 
        regex = re.compile('\d+', re.IGNORECASE)
        by = str(by).lower() 
        if by.find('survey')>=0 :
            by ='name'
        
        prefix = str(prefix) 
        
        if data is not None:
           self.data_ = data
           
        if self.ediObjs_ is None: 
           self.fit(self.data_)
           
        self.id = make_ids(self.ediObjs_, prefix='S', how= how )
           
        if how !='py': 
            self.id = make_ids(self.ediObjs_, prefix='S',
                                    cmode =None)  
        if dataid is None: 
            if prefix !='None' : 
                dataid = list(map(lambda s: s.replace('S', prefix), self.id))
                
            elif by =='name': 
                # get the first name of dataId of the EDI ediObjs  and filled
                # the rename dataId. remove the trail'_'  
                name = self.survey_name or  regex.sub(
                    '', self.ediObjs_[0].Head.dataid).replace('_', '') 
                # remove prefix )'S' and keep only the digit 
                dataid = list(map(lambda n: name + n, regex.findall(
                    ''.join(self.id)) ))
                
            elif by.find('num')>=0: 
               
               dataid = regex.findall(''.join(self.id))  
               
            elif by =='id': 
                dataid = self.id 
                
            elif by =='ix': 
                dataid = list(map(
                    lambda x: str(int(x)), regex.findall(''.join(self.id))))  
            else :
                dataid = list(map(lambda obj: obj.Head.dataid, self.ediObjs_))

        elif dataid is not None: 
            if not np.iterable(dataid): 
                raise ValueError('DataId parameter should be an iterable '
                                 f'object, not {type(dataid).__name__!r}')
            if len(dataid) != len(self.ediObjs_): 
                raise ValueError (
                    'DataId length must have the same length with the number'
                    ' of collected EDIs({0}). But {1} {2} given.'.format(
                    len(self.ediObjs_), len(dataid),
                    f"{'is' if len(dataid)<=1 else 'are'}"))
       
    
        if make_coords: 
            if (reflong or reflat) is None: 
                raise ValueError('Reflong and reflat params must not be None!')
            self.longitude, self.latitude = makeCoords(
               reflong = reflong, reflat= reflat, nsites= len(self.ediObjs_),
               step = step , **kws) 
        # clean the old main Edi section info and 
        # and get the new values
        if correct_ll or make_coords:
            londms,*_ = scalePosition(self.longitude, todms=True)
            latdms,*_ = scalePosition(self.latitude, todms=True)

        # collect new ediObjs 
        cobjs = np.zeros_like (self.ediObjs_, dtype=object ) 
        
        for k, (obj, did) in enumerate(zip(self.ediObjs_, dataid)): 
            obj.Head.edi_header = None  
            obj.Head.dataid = did 
            obj.Info.ediinfo = None 
            
            if correct_ll or make_coords:
                obj.Head.long = float(self.longitude[k])
                obj.Head.lat = float(self.latitude[k])
                obj.Head.elev = float(self.elevation[k])
                oc = obj.DefineMeasurement.define_measurement
                oc= replace_reflatlon(oc, nval= latdms[k])
                oc= replace_reflatlon(oc, nval= londms[k], kind='reflong')
                oc = replace_reflatlon(oc, nval= self.elevation[k], 
                                       kind='refelev')
                obj.DefineMeasurement.define_measurement = oc 
            # Empty the previous MTEMAP infos and 
            # fetch the attribute values newly set.
            obj.MTEMAP.mtemapsectinfo =None 
            obj.MTEMAP.sectid= did
            
            if export: 
                obj.write_edifile(
                    savepath = savepath ,new_edifilename = edi_prefix, 
                    **kws)
            cobjs[k] = obj 
        
        self.ediObjs_ = cobjs 
        
        return self 

    def getfullfrequency  (self, 
                            data: Optional[str|List[EDIO]] = None,
                            to_log10:bool  =False 
                            )-> ArrayLike[DType[float]]: 
        """ Get the frequency with clean data. 
        
        The full or plain frequency is array frequency with no missing  data during 
        the data collection. Note that when using |NSAMT|, some data are missing 
        due to the weak of missing frequency at certain band especially in the 
        attenuation band. 
        
        :param data: full path to EDI files or collection of  `pycsamt`_ 
            package Edi-objects 
        :type data: path-like object or list of pycsamt.core.edi objects 
        
        :param to_log10: export frequency to base 10 logarithm 
        :type to_log10: bool, 
        
        :returns: frequency with clean data. Out of `attenuation band` if survey 
            is completed with  |NSAMT|. 
        :rtype: array_like, shape(N, )

        :example: 
            >>> from watex.methods.em import EM
            >>> from pycsamt.core.edi import Edi_collection 
            >>> edipath = 'data/edis' 
            >>> cObjs = Edi_collection (edipath) # object from Edi_collection 
            >>> ref = EM().getfullfrequency (cObjs.ediObjs)  
            >>> ref
            ... array([7.00000e+04, 5.88000e+04, 4.95000e+04, 4.16000e+04, 3.50000e+04,
                   2.94000e+04, 2.47000e+04, 2.08000e+04, 1.75000e+04, 1.47000e+04,
                   ...
                   1.12500e+01, 9.37500e+00, 8.12500e+00, 6.87500e+00, 5.62500e+00])
            >>> len(ref)
            ... 55 
            >>> # however full frequency can just be fetched using the attribute `freqs_` 
            >>> emObj = EM().fit(edipath)       # object from EM 
            >>> emObjs.freqs_ 
            ... array([7.00000e+04, 5.88000e+04, 4.95000e+04, 4.16000e+04, 3.50000e+04,
                   2.94000e+04, 2.47000e+04, 2.08000e+04, 1.75000e+04, 1.47000e+04,
                   ...
                   1.12500e+01, 9.37500e+00, 8.12500e+00, 6.87500e+00, 5.62500e+00])
            
        """
        if data is not None: 
            self.data_ = data 
        if self.ediObjs_ is None: 
            self.fit(self.data_)
        
        lenfs = np.array([len(ediObj.Z._freq) for ediObj in self.ediObjs_ ] ) 
        ix_fm = np.argmax (lenfs) ; f= self.ediObjs_ [ix_fm].Z._freq 
        
        return np.log10(f) if to_log10 else f 
    
    def make2d (self,
                data: Optional[str|List[EDIO]] =None , 
                out:str = 'resxy',
                *, 
                kind:str = 'complex' , 
                **kws 
                )-> NDArray[DType[float]]: 
        """ Out 2D resistivity, phase (error) and tensor matrix from EDI-collection
        objects. Matrix for number of frequency x number of sites. 
        
        The function asserts whether all data from all frequencies are available. 
        The missing values should be filled by NaN. 
        
        Parameters 
        ----------- 
        data: Path-like object or list of pycsamt.core.edi objects
            Collections of EDI-objects from `pycsamt`_ or full path to EDI files.
        out: str 
            kind of data to output. Be sure to provide the component to retrieve 
            the attribute from the collection object. Except the `error` and 
            frequency attribute, the missing component to the attribute will 
            raise an error. for instance ``resxy`` for xy component. Default is 
            ``zxy``. 
        kind : bool or str 
            focus on the tensor output. Note that the tensor is a complex number 
            of ndarray (nfreq, 2,2 ). If set to``modulus`, the modulus of the complex 
            tensor should be outputted. If ``real`` or``imag``, it returns only
            the specific one. Default is ``complex``.
            
        kws: dict 
            Additional keywords arguments from :func:`~.getfullfrequency `. 
        
        Returns 
        -------- 
        mat2d : np.ndarray(nfreq, nstations) 
            the matrix of number of frequency and number of Edi-collectes which 
            correspond to the number of the stations/sites. 
        
        Examples 
        ---------
        >>> from watex.methods.em import EM 
        >>> edipath ='data/edis'
        >>> emObjs= EM().fit(edipath)
        >>> phyx = EM().make2d (emObjs.ediObjs_, 'phaseyx')
        >>> phyx 
        ... array([[ 26.42546593,  32.71066454,  30.9222746 ],
               [ 44.25990541,  40.77911136,  41.0339148 ],
               ...
               [ 37.66594686,  33.03375863,  35.75420802],
               [         nan,          nan,  44.04498791]])
        >>> phyx.shape 
        ... (55, 3)
        >>> # get the real number of the yy componet of tensor z 
        >>> zyy_r = make2d (ediObjs, 'zyx', kind ='real')
        ... array([[ 4165.6   ,  8665.64  ,  5285.47  ],
               [ 7072.81  , 11663.1   ,  6900.33  ],
               ...
               [   90.7099,   119.505 ,   122.343 ],
               [       nan,        nan,    88.0624]])
        >>> # get the resistivity error of component 'xy'
        >>> resxy_err = EM.make2d (emObjs.ediObjs_, 'resxy_err')
        >>> resxy_err 
        ... array([[0.01329037, 0.02942557, 0.0176034 ],
               [0.0335909 , 0.05238863, 0.03111475],
               ...
               [3.33359942, 4.14684926, 4.38562271],
               [       nan,        nan, 4.35605603]])
        >>> phyx.shape ,zyy_r.shape, resxy_err.shape  
        ... ((55, 3), (55, 3), (55, 3))
        
        """
        def fit2dall(objs, attr, comp): 
            """ Read all ediObjs and replace all missing data by NaN value. 
            
            This is useful to let the arrays at each station to  match the length 
            of the complete frequency rather than shrunking  up some data. The 
            missing data should be filled by NaN values. 
            
            """
            zl = [getattr( ediObj.Z, f"{attr}")[tuple (_c.get(comp))]
                  for ediObj in objs ]
            
            if name =='z': 
                if kind =='modulus': 
                    zl = [ np.abs (v) for v in zl]
                    zl = [fittensor(self.freqs_, ediObj.Z._freq, v)
                          for ediObj ,  v  in zip(objs, zl)]
                if kind in ('real' , 'complex') : 
                    zr = [fittensor(self.freqs_, ediObj.Z._freq, v.real)
                          for ediObj ,  v  in zip(objs, zl)]
                    
                if kind in ('imag', 'complex'): 
                    zi= [fittensor(self.freqs_, ediObj.Z._freq, v.imag)
                          for ediObj ,  v  in zip(objs, zl)]
                    
                if kind =='complex': 
                    zl = [ r + 1j * im for r, im in zip (zr, zi)]
                    
                    
                zl = zl if kind in ('modulus', 'complex') else (
                    zr if kind =='real' else zi )    
            else : 
                zl = [fittensor(self.freqs_, ediObj.Z._freq, v)
                      for ediObj ,  v  in zip(objs, zl)]
                
            # stacked the z values alomx axis=1. 
            return np.hstack ([ reshape (o, axis=0) for o in zl])
            
        out = str(out).lower().strip () 
        kind = str(kind).lower().strip() 
        if kind.find('imag')>=0 :
            kind ='imag'
        if kind not in ('modulus', 'imag', 'real', 'complex'): 
            raise ValueError(f"Unacceptable argument {kind!r}. Expect "
                             "'modulus','imag', 'real', or 'complex'.")
        # get the name for extraction using regex 
        regex1= re.compile(r'res|rho|phase|phs|z|tensor|freq')
        regex2 = re.compile (r'xx|xy|yx|yy')
        regex3 = re.compile (r'err')
        
        m1 = regex1.search(out) 
        m2= regex2.search (out)
        m3 = regex3.search(out)
        
        if m1 is None: 
            raise ValueError (f" {out!r} does not match  any 'resistivity',"
                              " 'phase' 'tensor' nor 'frequency'.")
        m1 = m1.group() 
        
        if m1 in ('res', 'rho'):
            m1 = 'resistivity'
        if m1 in ('phase', 'phs'):
            m1 = 'phase' 
        if m1 in ('z', 'tensor'):
            m1 ='z' 
        if m1  =='freq':
            m1 ='_freq'
            
        if m2 is None or m2 =='': 
            if m1 in ('z', 'resistivity', 'phase'): 
                raise ValueError (
                    f"{'Tensor' if m1=='z' else m1.title()!r} component "
                    f"is missing. Use e.g. '{m1}_xy' for 'xy' component")
        m2 = m2.group() if m2 is not None else m2 
        m3 = m3.group () if m3 is not None else '' 
        
        if m3 =='err':
            m3 ='_err'
        # read/assert edis and get the complete frequency 
        if data is not None: 
            self.data_= data 
        if self.ediObjs_ is None: 
            self.fit(self.data_ )

        #=> slice index for component retreiving purpose 
        _c= {
              'xx': [slice (None, len(self.freqs_)), 0 , 0] , 
              'xy': [slice (None, len(self.freqs_)), 0 , 1], 
              'yx': [slice (None, len(self.freqs_)), 1 , 0], 
              'yy': [slice (None, len(self.freqs_)), 1,  1] 
        }
        #==> returns mat2d freq 
        if m1 =='_freq': 
            f2d  = [fittensor(self.freqs_, ediObj.Z._freq, ediObj.Z._freq)
                  for ediObj in self.ediObjs_
                  ]
            return  np.hstack ([ reshape (o, axis=0) for o in f2d])
        
        # get the value for exportation (attribute name and components)
        name = m1 + m3 if (m3 =='_err' and m1 != ('_freq' or 'z')) else m1 
        #print(name, m1 , m2)
        mat2d  = fit2dall(objs= self.ediObjs_, attr= name, comp= m2)
        
        return mat2d 
    
    def getreferencefrequency (self,
                                data: Optional[str|List[EDIO]] = None,
                                to_log10: bool =False
                                ): 
        """ Get the reference frequency from collection Edis objects.
        
        The highest frequency with clean data should be selected as the  
        reference frequency
        
        Parameters 
        ---------- 
        data: list  of  pycsamt.core.edi.Edi or mtpy.core.edi.Edi objects 
            Collections of EDI-objects from `pycsamt`_ 
            
        to_log10: bool, 
            outputs the reference frequency into base 10 logarithm in Hz.
        
        Returns 
        -------
        rf : float 
            the reference frequency at the clean data in Hz 
            
        Examples 
        ---------
        >>> from watex.methods.em import EM  
        >>> edipath ='data/3edis'
        >>> ref = EM().getreferencefrequency(edipath, to_log10=True)
        >>> ref 
        ... 4.845098040014257 # in Hz 
        
        References 
        ----------
        http://www.zonge.com/legacy/PDF_DatPro/Astatic.pdf
        
        """
        if data is not None: 
            self.data_= data 
        if self.ediObjs_ is None: 
            self.fit(self.data_)
            
        self.freqs_= self.getfullfrequency (self.ediObjs_)
        # fit z and find all missing data from complete frequency f 
        # we take only the componet xy for fitting.

        zxy = [fittensor(self.freqs_, ediObj.Z._freq, ediObj.Z.z[:, 0, 1].real)
              for ediObj in self.ediObjs_
              ]
        # stacked the z values alomx axis=1. 
        arr2d = np.hstack ([ reshape (o, axis=0) for o in zxy])

            
        ix_nan = reshape (np.argwhere(np.isnan(arr2d).any(axis =1) ))
            # create bool array and mask the row of NaN 
        mask = np.full_like (self.freqs_, fill_value = True , dtype=bool)
        mask[[*ix_nan] ] = False 
        # get the reference frequency and index 
        return  self.freqs_ [mask].max() if not to_log10 else np.log10(
            self.freqs_ [mask].max())
    
    def export2newedis (self, 
                        ediObj: EDIO , 
                        new_Z: NDArray[DType[complex]], 
                        savepath:str =None, 
                        **kws)-> object :
        """ Export new EDI files from the former object with  a given new  
        impedance tensors. 
        
        The export is assumed a new output EDI resulting from multiples 
        corrections applications. 
        
        Parameters 
        -----------
        ediObj: str or  pycsamt.core.edi.Edi 
            Full path to Edi file or object from `pycsamt`_ 
        
        new_Z: ndarray (nfreq, 2, 2) 
            Ndarray of impendance tensors Z. The tensor Z is 3D array composed of 
            number of frequency `nfreq`and four components (``xx``, ``xy``, ``yx``,
            and ``yy``) in 2X2 matrices. The  tensor Z is a complex number. 
        
        Returns 
        --------
         ediObj from pycsamt.core.edi.Edi 
         
        """
        
        ediObj = self.is_valid(ediObj)
        ediObj.write_new_edifile( new_Z=new_Z, savepath = savepath , **kws)
        return ediObj 
    
   
    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        return repr_callable_obj  (self)
       
    
    def __getattr__(self, name):
        if name.endswith ('_'): 
            if name not in self.__dict__.keys(): 
                if name in ('data_', 'ediObjs_', 'freqs_', 'refreq_'): 
                    raise FitError (
                        f'Fit the {self.__class__.__name__!r} object first'
                        )
                
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        appender  = "" if rv is None else f'. Do you mean {rv!r}'
        
        raise AttributeError (
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{appender}{"" if rv is None else "?"}'
            )

class updateZ(EM): 
    """ A decorator for impedance tensor updating. 
    
    Update a Z object from each EDI object composing the collection objects 
    and output a new EDI-files is `option` is set to ``write``. 
    
    :param option: str - kind of action to perform with new Z collection.
        When `option` is set to ``write``. The new EDI -files are exported.
        Any other values should return only the updated impedance tensors.
        
    :returns: A collection of  :class:`pycsamt.core.z.Z` impedance tensor 
        objects.
    """
    
    def __init__(self, option:str = 'write'): 
        self.option = option
    
    def __call__ (self, func:F):
        
        @functools.wraps (func)
        def new_func ( *args, **kws): 
            """ Wrapper  to make new Z. The updated Z is a collection
            object from ':class:`pycsamt.core.z.Z` """
            
            (ediObjs , freq,  z_dict), kwargs = func (*args, **kws)
            
            # pop the option argument if user provides it 
            option = kwargs.pop('option', None)
            self.option  = option  or self.option 
            # create an empty array to collect each new Z object 
            Zc = np.empty((len(ediObjs), ), dtype = object )
            
            for kk in range  (len(ediObjs)):
                # create a new Z object for each Edi
                Z= self._make_zObj(kk,freq=freq, z_dict = z_dict  )
                if self.option =='write': 
                    self.export2newedis(ediObj=ediObjs[kk] , new_Z=Z, 
                                    **kwargs)
                Zc[kk] =Z
                
            return Zc 
          
        return new_func 
    
    def _make_zObj (self, 
                    kk: int ,
                    *, 
                    freq: ArrayLike[DType[float]], 
                    z_dict: Dict[str, NDArray[DType[complex]]]
                    )-> NDArray[DType[complex]]: 
        """ Make new Z object from frequency and dict tensor component Z. 
        
        :param kk: int 
            index of routine to retrive the tensor data. It may correspond of 
            the station index. 
        :param freq: array-like 
            full frequency of component 
        :param z_dict: dict, 
            dictionnary of all tensor component. 

        """
        Z = EMz.Z(
            z_array=np.zeros((len(freq ), 2, 2),dtype='complex'),
            z_err_array=np.zeros((len(freq), 2, 2)),
            freq=freq 
            )
        zxx = z_dict.get('zxx') 
        zxy = z_dict.get('zxy') 
        zyx = z_dict.get('zyx') 
        zyy = z_dict.get('zyy') 
        # dont raise any error if the component 
        # does not exist.
        if zxx is not None: 
            Z.z[:, 0,  0] = reshape (zxx[:, kk], 1) 
        if zxy is not None: 
            Z.z[:, 0,  1] = reshape (zxy[:, kk], 1)
        if zyx is  not None: 
            Z.z[:, 1,  0] = reshape (zyx[:, kk], 1) 
        if zyy is not None: 
            Z.z[:, 1,  1] = reshape (zyy[:, kk], 1)
            
        # set the z_err 
        zxx_err = z_dict.get('zxx_err') 
        zxy_err = z_dict.get('zxy_err') 
        zyx_err = z_dict.get('zyx_err') 
        zyy_err = z_dict.get('zyy_err') 
        
        if zxx_err is not None: 
            Z.z_err[:, 0,  0] = reshape (zxx_err[:, kk], 1) 
        if zxy_err is not None: 
            Z.z_err[:, 0,  1] = reshape (zxy_err[:, kk], 1)
        if zyx_err is not None: 
            Z.z_err[:, 1,  0] = reshape (zyx_err[:, kk], 1) 
        if zyy_err is not None: 
            Z.z_err[:, 1,  1] = reshape (zyy_err[:, kk], 1)
        
        Z.compute_resistivity_phase()
        
        return Z 
        
class Processing (EM) :
    """ Base processing of EM object 
    
    Fast process EMAP and AMT data. Tools are used for data sanitizing, 
    removing noises and filtering. 
    
    
    Parameters 
    ----------
    data: Path-like object or list  of  `pycsamt.core.edi.Edi` objects 
        Collections of EDI-objects from `pycsamt`_ 
    
    freqs: array-like, shape (N)
        Frequency array. It should be the complete frequency used during the 
        survey area. It can be get using the :func:`getfullfrequency ` 
        No need if ediObjs is provided. 
        
    window_size : int
        the length of the window. Must be greater than 1 and preferably
        an odd integer number. Default is ``5``
        
    component: str 
       field tensors direction. It can be ``xx``, ``xy``,``yx``, ``yy``. If 
       `arr2d`` is provided, no need to give an argument. It become useful 
       when a collection of EDI-objects is provided. If don't specify, the 
       resistivity and phase value at component `xy` should be fetched for 
       correction by default. Change the component value to get the appropriate 
       data for correction. Default is ``xy``.
       
    mode: str 
        mode of the border trimming. Should be 'valid' or 'same'.'valid' is used 
        for regular trimimg whereas the 'same' is used for appending the first
        and last value of resistivity. Any other argument except 'valid' should 
        be considered as 'same' argument. Default is ``same``.     
       
    method: str, default ``slinear``
        Interpolation technique to use. Can be ``nearest``or ``pad``. Refer to 
        the documentation of :doc:`~.interpolate2d`. 
        
    out : str 
        Value to export. Can be ``sfactor``, ``tensor`` for corrections factor 
        and impedance tensor. Any other values will export the static corrected  
        resistivity. 
        
    c : int, 
        A window-width expansion factor that must be input to the filter 
        adaptation process to control the roll-off characteristics
        of the applied Hanning window. It is recommended to select `c` between 
        ``1``  and ``4``.  Default is ``2``. 

    Examples 
    --------
    >>> import matplotlib.pyplot as plt 
    >>> from watex.methods.em import Processing 
    >>> edipath = 'data/edis'
    >>> p = Processing().fit(edipath) 
    >>> p.window_size =2 
    >>> p.component ='yx'
    >>> rc= p.tma()
    >>> # get the resistivy value of the third frequency  at all stations 
    >>> p.res2d_[3, :]  
    ... array([ 447.05423001, 1016.54352954, 1415.90992189,  536.54293994,
           1307.84456036,   65.44806698,   86.66817791,  241.76592273,
           ...
            248.29077039,  247.71452712,   17.03888414])
    >>>  # get the resistivity value corrected at the third frequency 
    >>> rc [3, :]
    ... array([ 447.05423001,  763.92416768,  929.33837349,  881.49992091,
            404.93382163,  190.58264151,  160.71917654,  163.30034875,
            394.2727092 ,  679.71542811,  953.2796567 , 1212.42883944,
            ...
            164.58282866,   96.60082159,   17.03888414])
    >>> plt.semilogy (np.arange (self.res2d_.shape[1] ), self.res2d_[3, :], '--',
                      np.arange (self.res2d_.shape[1] ), rc[3, :], 'ok--')
 
    References 
    -----------
    .. [1] http://www.zonge.com/legacy/PDF_DatPro/Astatic.pdf
        
    """
    
    def __init__(self,
                 window_size:int =5, 
                 component:str ='xy', 
                 mode: str ='same', 
                 method:str ='slinear', 
                 out:str  ='srho', 
                 c: str =2, 
                 **kws): 
        super().__init__(**kws)
        
        self._logging = watexlog.get_watex_logger(self.__class__.__name__)
        self.window_size =window_size 
        self.component = component 
        self.mode = mode 
        self.method = method 
        self.out = out 
        self.c = c
        

    def tma (self,
             data:str|List[EDIO]
    )-> NDArray[DType[float]] :
        
        """ A trimmed-moving-average filter to estimate average apparent
        resistivities at a single static-correction-reference frequency. 
        
        The TMA filter option estimates near-surface resistivities by averaging
        apparent resistivities along line at the selected static-correction 
        reference frequency. The highest frequency with clean data should be 
        selected as the reference frequency.
        
        Parameters 
        ----------
        data: path-like object or list  of  pycsamt.core.edi.Edi 
            Collections of EDI-objects from `pycsamt`_ 
    
        Returns 
        -------
        rc or cf: np.ndarray, shape  (N, M)
            EMAP apparent  resistivity static shift corrected or static 
            correction factor or impedance tensor.
            
        References 
        -----------
        .. [1] http://www.zonge.com/legacy/PDF_DatPro/Astatic.pdf
        
        """
        if data is not None: 
            self.data_ = data 
        if self.ediObjs_ is None: 
            self.fit(self.data_)
        # assert filter arguments 
        self.res2d_ , self.phs2d_ , self.freqs_, self.c, self.window_size, \
            self.component, self.out = self._make2dblobs ()
 
        #  interpolate resistivity and phases 
        self.phs2d_= interpolate2d(
            self.phs2d_, method =self.method)
        self.res2d_= interpolate2d(
            self.res2d_, method =self.method)
        # get the index of the reference frequency  and collect 
        # the resistivity and phase at that frequency 
        ix_rf = np.int(reshape (np.argwhere (self.freqs_==self.refreq_)))  
        # normalize log frequency and take the normalize ref freq 
        norm_f = (np.log10(self.freqs_) / np.linalg.norm(
            np.log10(self.freqs_)))
        # compute the slope at each normalize frequency 
        slope2d = np.arctan( (np.deg2rad(self.phs2d_) / (
            np.pi /4 )) -1 ) / (np.pi /2 )
        log_rho2d = np.log10 (self.res2d_) + norm_f[:, None] * slope2d 
        # extrapolate up 
        # replace the up frequency thin the index of rf by interpolating up 
        log_rho2d [:ix_rf, :] = np.log10 (
            self.res2d_[:ix_rf, : ]) + np.log10(
                np.sqrt(2)) * slope2d[:ix_rf, :]
        
        # make an array of weight factor wf 
        wf = np.zeros_like(log_rho2d) # average logj 
        # For each station collect a group of window-size log(rj ), 
        # #i.e. for window size =5 station index j, i = j-2 to j+2. 
        half_window = self.window_size //2 
        for ii in range(log_rho2d.shape[1]):
            
            if ii ==0 or ii ==log_rho2d.shape[1] -1: 
                w = (log_rho2d[ :, :ii + half_window +1] 
                     if  ii - half_window < 0 else 
                     log_rho2d[:, ii-half_window:] 
                     ) if self.mode =='valid' else log_rho2d[:, ii][:, None]
      
            elif ii - half_window < 0: 
                w= log_rho2d[ :, :ii + half_window +1]
                
            elif ii + half_window +1 > log_rho2d.shape[1]: 
                w= log_rho2d[:, ii-half_window:]
    
            else : 
                # Discard the lowest and highest valued log(rj ) 
                # from the group of five and average the remaining
                # three => avg_logj.
                w= log_rho2d[:, ii-half_window : half_window + ii + 1 ]
                try : 
                    ls = [ np.delete (w[ii, :] , np.where (
                        np.logical_or(w[ii, :] ==w[ii, :].max(),
                                      w[ii, :] ==w[ii, :].min())
                        )) for ii in range (len(w))]
                    
                    w = np.vstack (ls)
                except : 
                    # in the case the ls has some array with different length 
                    # do the average manually and set as an array of axis 1.  
                    ls = [np.sum(w[ii, :])/ len(w[ii, :]
                                                ) for ii in range(len(w))]
                    w = np.array(ls)[:, None] # make axis = 1
                
            wf[:, ii] = np.average (w, axis = 1)
            
        # compute the correction factor cf
        cf = np.power(10, wf, dtype =float)/ np. power(10, log_rho2d) 
        
        rc = self.res2d_ * cf 
        
        if self.out =='z': 
            rc = rhoa2z(rc, self.phs2d_, self.freq_s)

        return   cf if self.out =='sf' else rc   
      
    def _make2dblobs (self, 
                      data :Optional[str|List[EDIO]] = None 
                      ): 
        """ Asserts argument of |EMAP| filter and returns useful arguments.
        
        data: path-like object or list  of  pycsamt.core.edi.Edi 
            Collections of EDI-objects from `pycsamt`_ 
            
        :note: created to collect argument of EMAP filters. Refer to functions 
        :func:`~.tma`, :func:`~.flma` and :func:`~.ama` documentation. 
            
        """
        self.component= str(self.component).lower().strip() 
        self.out= str(self.out).lower().strip() 
        
        if data is not None: 
            self.data_ = data 
        if self.ediObjs_ is None: 
            self.fit(self.data_)
            
        try : 
            self.c = int (self.c) 
        except : 
            raise TypeError(
                f'Expect an integer value not {type(self.c).__name__!r}')
        
        if self.out.find ('factor') >= 0 or self.out =='sf': 
            self.out ='sf'
        elif self.out in ('z', 'impedance', 'tensor'): self.out ='z'
        
        if self.component not in ('xx', 'xy', 'yx', 'yy'): 
            raise ValueError(f"Unacceptable component {self.component!r}. "
                             "Expect 'xx', 'xy', 'yx' or 'yy'")
        
        self.res2d_= self.make2d(self.ediObjs_, 
                                 out=f'res{self.component}')
        self.phs2d_ = self.make2d(self.ediObjs_, 
                                  out=f'phase{self.component}')
        
            
        if len(self.res2d_) != len(self.freqs_): 
            raise ValueError ("Resistivity and frequency arrays must have a same"
                          f" length. But {len(self.res2d_)} & {len(self.freqs_)}"
                          " were given")
        if len(self.res2d_) != len(self.phs2d_): 
            raise ValueError ("Resistivity and phase must have the same length."
                              f" But {len(self.res2d_)} & {len(self.phs2d_)} "
                              "were given.")
        try : 
            self.window_size = int(self.window_size)
        except ValueError : 
            raise ValueError (
                'Could not convert {type(self.window_size).__name__!r} '
                 'to integer: {self.window_size!r}')
     
        self.res2d_ = np.array (self.res2d_)
        if self.window_size > self.res2d_.shape [1]:
            raise ValueError ("window size might not be less than"
                              f" {str(self.res2d_.shape [1])!r}")
        
        return (self.res2d_ , self.phs2d_ , self.freqs_, self.c,
                self.window_size, self.component, self.out) 
    
    def ama (self, data:str|List[EDIO]
             )-> NDArray[DType[float]] :
        """ 
        Use an adaptive-moving-average filter to estimate average apparent 
        resistivities at a single static-correction-reference frequency.. 
        
        The AMA filter estimates static-corrected apparent resistivities at a 
        single reference frequency by calculating a profile of average impedances 
        along the length of the line. Sounding curves are then shifted so that they
        intersect the averaged profile. 
        
        Parameters 
        ----------
        data: path-like object or list  of  pycsamt.core.edi.Edi 
            Collections of EDI-objects from `pycsamt`_ 
            
        Returns 
        -------
        rc or z: np.ndarray, shape  (N, M)
            EMAP apparent  resistivity static shift corrected  or static 
            correction tensor 
            
        References 
        -----------
        .. [1] http://www.zonge.com/legacy/PDF_DatPro/Astatic.pdf
        .. [2] Torres-Verdin and Bostick, 1992,  Principles of spatial surface 
            electric field filtering in magnetotellurics: electromagnetic array profiling
            (EMAP), Geophysics, v57, p603-622.https://doi.org/10.1190/1.2400625
            
        """
        if data is not None: 
            self.data_ = data 
        if self.ediObjs_ is None: 
            self.fit(self.data_)
            
        # assert filter arguments 
        self.res2d_ , self.phs2d_ , self.freqs_, self.c, self.window_size, \
            self.component, self.out = self._make2dblobs ()
        #  interpolate resistivity and phases 
        self.phs2d_= interpolate2d(self.phs2d_, method =self.method)
        self.res2d_= interpolate2d(self.res2d_, method =self.method,)
        
        # convert app. resistivity and impedance phase  to 
        # impedance values, Zj, for each station
        omega0 = 2 * np.pi * self.freqs_
        zj = np.sqrt(self.res2d_ * omega0[:, None] * mu0 ) * (np.cos (
            np.deg2rad(self.phs2d_)) + 1j * np.sin(np.deg2rad(self.phs2d_)))
        
        # compute the weight factor for convoluting 
        # L = dipole length = L : 1 is fixed dipole -length 
        w = np.array([betaj (xj = ii, L= 1 , W= self.window_size) 
                      for ii in range(self.window_size)])
        #print(w)
        zjr = np.zeros_like(self.res2d_) 
        zji = zjr.copy() 
        
        for ii in range (len(zj)): 
            w_exp = [ k * self.window_size for k in range(1, self.c +1 )]
            zcr=list(); zci = list()
            # compute Zk(xk, w) iteratively
            # with adpatavive W expanded to 1 to c 
            for wind_k  in w_exp : 
                w= np.array([betaj (xj = jj, L= 1, W= wind_k
                                    ) for jj in range(wind_k)
                             ])
                # block mode to same to keep the same dimensions
                zcr.append(np.convolve(zj[ii, :].real, w[::-1], 'same'))
                zci.append(np.convolve(zj[ii, :].imag, w[::-1], 'same'))
            # and take the average 
            zjr [ii, :] = np.average (np.vstack (zcr), axis = 0)
            zji[ii, :] = np.average (np.vstack (zci), axis = 0)
               
     
        zjc = zjr + 1j * zji 
        rc = z2rhoa(zjc, self.freqs_)  
        if self.mode =='same': 
            rc[:, 0] = self.res2d_[:, 0]
            zjc[:, 0] = zj [:, 0]
        
        return zjc if self.out =='z' else rc 

    def flma (self, 
              data:str|List[EDIO]
        )-> NDArray[DType[float]] :
        """ Use a fixed-length-moving-average filter to estimate average apparent
        resistivities at a single static-correction-reference frequency. 
        
        The FLMA filter estimates static-corrected apparent resistivities at a 
        single reference frequency by calculating a profile of average impedances 
        along the length of the line. Sounding curves are then shifted so that they
        intersect the averaged profile. 
        
        Parameters 
        ----------
        data: path-like object or list  of  pycsamt.core.edi.Edi 
            Collections of EDI-objects from `pycsamt`_ 
   
        Returns 
        -------
        rc or z : np.ndarray, shape  (N, M)
            EMAP apparent  resistivity static shift corrected  or static 
            correction impedance tensor. 
        
     
        References 
        -----------
        .. [1] http://www.zonge.com/legacy/PDF_DatPro/Astatic.pdf
        
        """
        if data is not None: 
            self.data_ = data 
        if self.ediObjs_ is None: 
            self.fit(self.data_)
    
        # assert filter arguments 
        self.res2d_ , self.phs2d_ , self.freqs_, self.c, self.window_size, \
            self.component, self.out = self._make2dblobs ()
        #  interpolate resistivity and phases 
        self.phs2d_= interpolate2d(self.phs2d_, method =self.method)
        self.res2d_= interpolate2d(self.res2d_, method =self.method)
        
        # convert app. resistivity and impedance phase  to 
        #impedance values, Zj, for each station
        omega0 = 2 * np.pi * self.freqs_
        zj = np.sqrt(self.res2d_ * omega0[:, None] * mu0 ) * (np.cos (
            np.deg2rad(self.phs2d_)) + 1j * np.sin(np.deg2rad(self.phs2d_)))
        
        # compute the weight factor for convoluting 
        # L = dipole length = L
        w = np.array([betaj (xj = ii, L= 1 , W= self.window_size) 
                      for ii in range(self.window_size)])
        
        zjr = np.zeros_like(self.res2d_) 
        zji = zjr.copy() 
        for ii in range(len(zjr)) :
            # block mode to same to keep the same array dimensions
            zjr[ii, :] = np.convolve(zj[ii, :].real, w[::-1], 'same')
            zji[ii, :] = np.convolve(zj[ii, :].imag, w[::-1], 'same')
        # recover the static apparent resistivity from reference freq 
        zjc = zjr + 1j * zji 
        rc = z2rhoa (zjc, self.freqs_) #np.abs(zjc)**2 / (omega0[:, None] * mu0 )
        
        if self.mode =='same': 
            rc[:, 0]= self.res2d_[:, 0]
            zjc[:, 0]= zj [:, 0]
        
        return zjc if self.out =='z' else rc 
    
    def skew(self,
             data :Optional[str|List[EDIO]] = None, 
             method:str ='swift'
             )-> NDArray[DType[float]]: 
        r"""
        The conventional asymmetry parameter based on the Z magnitude. 
        
        Parameters 
        ---------
        data: str of path-like or list of pycsamt.core.edi.Edi 
            EDI data or EDI object with full impedance tensor Z. 
        
        method: str 
            Kind of correction. Can be ``swift`` for the remove distorsion proposed 
            by Swift in 1967. The value close to 0. assume the 1D and 2D structures 
            and 3D otherwise. Conversly to ``bahr`` for the remove distorsion proposed  
            by Bahr in 1991. The latter threshold is set to 0.3. Above this value 
            the structures is 3D. 
        
        Returns 
        ------- 
        skw, mu : Tuple of ndarray-like , shape (N, M )
            - Array of skew at each frequency 
            - rotational invariant ``mu`` at each frequency. 
            
        See also 
        -------- 
        
        The |EM| signal is influenced by several factors such as the dimensionality
        of the propagation medium and the physical anomalies, which can distort the
        |EM| field both locally and regionally. The distortion of Z was determined 
        from the quantification of its asymmetry and the deviation from the conditions 
        that define its dimensionality. The parameters used for this purpose are all 
        rotational invariant because the Z components involved in its definition are
        independent of the orientation system used. The conventional asymmetry
        parameter based on the Z magnitude is the skew defined by Swift (1967) as
        follows:
        
        .. math:: skew_{swift}= |\frac{Z_{xx} + Z_{yy}}{ Z_{xy} - Z_{yx}}| 
            
        When the :math:`skew_{swift}`  is close to ``0.``, we assume a 1D or 2D model
        when the :math:`skew_{swift}` is greater than ``>=0.2``, we assume 3D local 
        anomaly (Bahr, 1991; Reddy et al., 1977).
        
        Furthermore, Bahr (1988) proposed the phase sensitive skew which calculates
        the skew taking into account the distortions produced in Z over 2D structures
        by shallow conductive anomalies and is defined as follows:
        
        .. math::
            
            skew_{Bahr} & = & \sqrt{ \frac{|[D_1, S_2] -[S_1, D_2]|}{|D_2|}} \quad \text{where} 
            
            S_1 & = & Z_{xx} + Z_{yy} \quad ; \quad  S_2 = Z_{xy} + Z_{yx} 
            
            D_1 & = &  Z_{xx} - Z_{yy} \quad ; \quad  D_2 = Z_{xy} - Z_{yx}
            
        Note that The phase differences between two complex numbers :math:`C_1` and 
        :math:`C_2` and the corresponding amplitude  products are now abbreviated 
        by the commutators:
            
        .. math:: 
          
            [C_1, C_2] & = & \text{Im} C_2*C_1^{*}
            
            [C_1, C_2]  & = & \text{Re} C_1 * \text{Im}C_2  - R_e(C_2)* \text{Im}C_1
                        
        Indeed, :math:`skew_{Bahr}` measures the deviation from the symmetry condition
        through the phase differences between each pair of tensor elements,considering
        that phases are less sensitive to surface distortions(i.e. galvanic distortion).
        The :math:`skew_{Bahr}` threshold is set at ``0.3`` and higher values mean 
        3D structures (Bahr, 1991).
        
        
        Examples
        --------
        >>> from watex.methods.em import Processing 
        >>> edipath = 'data/edis'
        >>> p = Processing().fit(edipath) 
        >>> sk,_ = p.skew()
        >>> sk[0:, ]
        ... array([0.45475527, 0.7876896 , 0.44986397])
        
        References 
        ----------
            
        Bahr, K., 1991. Geological noise in magnetotelluric data: a classification 
            of distortion types. Physics of the Earth and Planetary Interiors 66
            (1–2), 24–38.
        Barcelona, H., Favetto, A., Peri, V.G., Pomposiello, C., Ungarelli, C., 2013.
            The potential of audiomagnetotellurics in the study of geothermal fields: 
            A case study from the northern segment of the La Candelaria Range,
            northwestern Argentina. J. Appl. Geophys. 88, 83–93.
            https://doi.org/10.1016/j.jappgeo.2012.10.004   
            
        Swift, C., 1967. A magnetotelluric investigation of an electrical conductivity 
           anomaly in the southwestern United States. Ph.D. Thesis, MIT Press. Cambridge. 
           
           
        """
        if data is not None: 
            self.data_ = data 
        if self.ediObjs_ is None: 
            self.fit(self.data_)
            
        self.method = method 
        if self.method not in ('swift', 'bahr'): 
            raise ValueError(
                f'Expected argument ``swift`` or ``bahr`` not: {self.method!r}')
            
        Zxx= self.make2d(self.ediObjs_,'zxx')
        Zxy = self.make2d(self.ediObjs_,'zxy')
        Zyx = self.make2d(self.ediObjs_,'zyx')
        Zyy= self.make2d(self.ediObjs_,'zyy')
        
        S1 =Zxx + Zyy; S2 = Zxy + Zyx; D1 =Zxx-Zyy ;  D2= Zxy-Zyx 
        D1S2 = (S2 * np.conj(D1)).imag ; S1D2 = (D2 * np.conj(S1)).imag 
        
        if method =='swift': 
            skw = np.abs ( S1  / D2 )
        else : 
            skw = np.sqrt(np.abs( D1S2 - S1D2))/np.abs(D2)
            
        mu = np.sqrt(np.abs(D1S2) + np.abs (S1D2))/ np.abs(D2) 
            
        return skw, mu


    def restoreTensorZ(self,
                       data: str|List[EDIO],
                       *, 
                       buffer: Tuple[float]=None, 
                       method:str ='pd',
                       **kws 
                       ): 
        """ Fix the weak and missing signal at the 'dead-band`- and recover the 
        missing impedance tensor values. 
        
        The function uses the complete frequency (frequency with clean data) collected 
        thoughout the survey to recover by inter/extrapolating the missing or weak 
        frequencies thereby restoring the impedance tensors at that 'dead-band'. Note 
        that the 'dead- band' also known as 'attenuation -band' is where the AMT 
        signal is weak or generally abscent. 
    
        Parameters 
        ---------- 
        data: Path-like object or list of pycsamt.core.edi objects
            Collections of EDI-objects from `pycsamt`_ or full path to EDI files. 
            
        buffer: list [max, min] frequency in Hz
            list of maximum and minimum frequencies. It must contain only two values.
            If `None`, the max and min of the clean frequencies are selected. Moreover
            the [min, max] frequency should not compulsory to fit the frequency range in 
            the data. The given frequency can be interpolated to match the best 
            closest frequencies in the data. 
      
        method: str, optional  
            Method of interpolation. Can be ``base`` for `scipy.interpolate.interp1d`
            ``mean`` or ``bff`` for scaling methods and ``pd`` for pandas interpolation 
            methods. Note that the first method is fast and efficient when the number 
            of NaN in the array if relatively few. It is less accurate to use the 
            `base` interpolation when the data is composed of many missing values.
            Alternatively, the scaled method(the  second one) is proposed to be the 
            alternative way more efficient. Indeed, when ``mean`` argument is set, 
            function replaces the NaN values by the nonzeros in the raw array and 
            then uses the mean to fit the data. The result of fitting creates a smooth 
            curve where the index of each NaN in the raw array is replaced by its 
            corresponding values in the fit results. The same approach is used for
            ``bff`` method. Conversely, rather than averaging the nonzeros values, 
            it uses the backward and forward strategy  to fill the NaN before scaling.
            ``mean`` and ``bff`` are more efficient when the data are composed of a
            lot of missing values. When the interpolation `method` is set to `pd`, 
            function uses the pandas interpolation but ended the interpolation with 
            forward/backward NaN filling since the interpolation with pandas does
            not deal with all NaN at the begining or at the end of the array. Default 
            is ``pd``.
            
        fill_value: array-like or ``extrapolate``, optional
            If a ndarray (or float), this value will be used to fill in for requested
            points outside of the data range. If not provided, then the default is
            NaN. The array-like must broadcast properly to the dimensions of the 
            non-interpolation axes.
            If a two-element tuple, then the first element is used as a fill value
            for x_new < x[0] and the second element is used for x_new > x[-1]. 
            Anything that is not a 2-element tuple (e.g., list or ndarray,
            regardless of shape) is taken to be a single array-like argument meant 
            to be used for both bounds as below, above = fill_value, fill_value.
            Using a two-element tuple or ndarray requires bounds_error=False.
            Default is ``extrapolate``. 
            
        kws: dict 
            Additional keyword arguments from :func:`~interpolate1d`. 
        
        Returns 
        --------
            Array-like of pycsamt.core.z.Z objects 
            Array collection of new Z impedances objects with dead-band tensor 
            recovered. :class:`pycsamt.core.z.Z` are ndarray (nfreq, 2, 2). 
            2x2 matrices for components xx, xy and yx, yy. 
    
        See also  
        ---------
        One main problem in collecting |NSAMT| data is the signal level in the 
        'attenuation band'. Compared to the |CSAMT| method (Wang and Tan, 2017; 
        Zonge and Hughes, 1991),the natural signals are not under our control and 
        suffer from frequency  ranges with little or no signal.  Most notably, the 
        |NSAMT| 'dead-band' between approximately 1 kHz and 4 kHz, but also a signal 
        low in the vicinityof 1 Hz where the transition to magnetospheric energy 
        sources occurs (Goldak and Olson, 2015). In this band, natural source signals
        are generally  absent. The EM energy is dissipated and often cultural |EM| 
        noise fills the gap (Zonge, 2000). The response is extrapolated from results 
        observed top frequencies( For instance at 20, 40, 250, and 500 Hz).Experience
        indicates that the natural source signal level at 2000 Hz can be expected 
        to approach 10-6 γ/√Hz (Zheng, 2010; Zonge, 2000).
    
        References 
        ----------
        Goldak, D.K., Olson, R.W., 2015. New developments in |AMT| exploration :
            Case study from Darnley Bay. CSEG Rec. 22–27.
        Wang, K., Tan, H., 2017. Research on the forward modeling of |CSAMT| in 
            three-dimensional axial anisotropic media. J. Appl. Geophys. 146, 27–36.
            https://doi.org/10.1016/j.jappgeo.2017.08.007
        Zonge, I., 2000. |NSAMT| Imaging. 3322 East Fort Lowell Road, Tucson, AZ 85716 USA. 
        Zonge, L., Hughes, L.J., 1991. |CSAMT|. Soc. Explor. Geophys. 2, 713–809.
           
        Examples 
        --------
        >>> import numpy as np 
        >>> import matplotlib.pyplot as plt 
        >>> from watex.methods.em import Processing
        >>> path2edi = 'data/edis'
        >>> pObjs= Processing().fit(path2edi)
        >>> # One can specify the frequency buffer like the example below, However 
        >>> # it is not necessaray at least there is a a specific reason to fix the frequencies 
        >>> buffer = [1.45000e+04,1.11500e+01]
        >>> zobjs_b =  pObjs.restoreTensorZ(pObjs.ediObjs_, buffer = buffer
                                            ) # with buffer 
        
        """
        def z_transform (z , rfq, fq,  slice_= None): 
            """ Route to do the same task for real, imaginary and error """
            with np.errstate(all='ignore'):
                z = reshape(z) 
                z = fittensor(compfreq= fq, refreq =rfq, z = z  ) 
                z = interpolate1d(arr=z , method = self.method, **kws )
            return z [slice_] 
            
        if data is not None: 
            self.data_ = data 
        if self.ediObjs_ is None: 
            self.fit(self.data_)
            
        self.method = method 
        
        # get the frequencies obj 
        zObjs = np.array (list(map(lambda o: o.Z, self.ediObjs_)) ,
                          dtype =object) 
        #read all frequency length and take the max frequency 
        # known  as the complete frequencies ( That compose all values)
        freqsize = np.array (list (map (lambda o:len(o._freq), zObjs)))
        ix_max  = np.argmax(freqsize)
        # get the complete freq 
        cfreq = zObjs[ix_max]._freq  
        
        # control the buffer and get the the range of frequency 
        buffer = self.controlFrequencyBuffer(cfreq, buffer)
        ix_buf,  = np.where ( np.isin (cfreq, buffer)) 
        ## index for slice the array in the case the buffer is set 
        ix_s , ix_end = ix_buf ; ix_end += 1 ; slice_= slice (ix_s,  ix_end) 
        s_cfreq = cfreq [slice_] # slice frequency within the buffer 
        
        # make a new Z objects 
        # make a new object 
        new_zObjs =np.zeros_like (zObjs, dtype =object )
        # loop to correct the Z impedance object values 
        for kk, ediObj in enumerate (self.ediObjs_):
            new_Z = EMz.Z(z_array=np.zeros((len(s_cfreq), 2, 2),
                                           dtype='complex'),
                        z_err_array=np.zeros((len(s_cfreq), 2, 2)),
                        freq=s_cfreq)
            new_Z = self._tfuncZtransformer(
                ediObj, new_Z, tfunc= z_transform,
                cfreq= cfreq, ix_s= ix_s, ix_end= ix_end 
                )
            new_zObjs[kk] = new_Z 
            
        return new_zObjs 

    def _tfuncZtransformer (self,  
                            ediObj: EDIO , 
                            new_Z: NDArray [DType[complex]], 
                            tfunc: F, 
                            cfreq: ArrayLike, slice_: slice =None, 
                            ix_s: int = 0 , 
                            ix_end: int  = -1, 
                            )-> NDArray [DType[complex]]: 
        """ Loop and transform the previous tensor to a new tensor from a 
        transform function `tfunc`. 
        
        :param ediObj: EDI-object from EM. 
        :param new_Z: new tensoor of 2 x2 matrix , complex number 
        :param tfunc: Callable, function for transform the tensor 
        :param cfreq: new interpolate frequency use to initialize the new tensor 
        :param slice_: slice object , to preserve data in the previous tensor 
        :param ix_s: int, index of startting point to read the tensor from 
            previous tensor 
        :param ix_end: int, end point to stop reading the previous tensor. 
        
        :note: the new tensor is composed of shape (cfreq, 2 , 2 ), 2 x2 matrices
            for the four component xx, xy, yx, yy . 
        :return: NDArray of shpe (cfreq, 2 * 2 ), dtype = complex 
        
        """
        for ii in range(2):
            for jj in range(2):
                # need to look out for zeros in the impedance
                # get the indicies of non-zero components
                nz_index = np.nonzero(ediObj.Z.z[:, ii, jj])
                if len(nz_index[0]) == 0:
                    continue
                # get the non_zeros components and interpolate 
                # frequency to recover the component in dead-band frequencies 
                # Use the whole edi
                with np.errstate(all='ignore'):
                    zfreq = ediObj.Z._freq
                    z_real = reshape(ediObj.Z.z[nz_index, ii, jj].real) 
                    z_real = tfunc (z_real, rfq=cfreq, fq=zfreq, 
                                          slice_=slice_ 
                                          )
                    z_imag = reshape(ediObj.Z.z[nz_index, ii, jj].imag) 
                    z_imag = tfunc (z_imag, rfq=cfreq, fq=zfreq, 
                                          slice_=slice_ 
                                          )
                    z_err = reshape(ediObj.Z.z_err[nz_index, ii, jj]) 
                    z_err = tfunc (z_err, rfq=cfreq, fq=zfreq,
                                         slice_=slice_ 
                                         )
                # Use the new dimension of the z and slice z according 
                # the buffer range. make the new index start at 0. 
                new_nz_index = slice (
                    * np.array([ix_s, ix_end],dtype=np.int32)-ix_s)
       
                new_Z.z[new_nz_index, ii, jj] = reshape(
                    z_real, 1)   + 1j * reshape(z_imag, 1) 
                new_Z.z_err[new_nz_index, ii, jj] = reshape(z_err, 1)
        
        # compute resistivity and phase for new Z object
        new_Z.compute_resistivity_phase()
        return new_Z 
    
    @staticmethod 
    def freqInterpolation (
            y:ArrayLike[DType[T]] ,
            /, 
            buffer:Optional[Tuple[float]] = None ,  
            kind: str  ='freq' 
            )-> ArrayLike[DType[T]]: 
        """ Interpolate frequency in frequeny buffer range.  
        
        :param y: array-like, shape(N, ) - Can be a frequency array or periods
            note that the frequency is not in log10 Hz. 
        :param buffer: list of maximum and minimum frequency. It should contains 
            only two values. If `None`, the max and min frequencies are used 
        :param kind: str 
            type of given data. Can be 'period'  if the value is given as periods 
            or 'frequency' otherwise. Any other value should be considered as a
            frequency values. 
            
        :return: array_like, shape (N2, ) 
            New interpolated frequency with N2 size 
            
        :example: 
            >>> from watex.methods.em import Processing
            >>> pobj = Processing().fit('data/edis')
            >>> f = getfullfrequency (pobj.ediObjs_)
            >>> buffer = [5.86000e+04, 1.6300e+01]
            >>> f 
            ... array([7.00000e+04, 5.88000e+04, 4.95000e+04, 4.16000e+04, 3.50000e+04,
                   2.94000e+04, 2.47000e+04, 2.08000e+04, 1.75000e+04, 1.47000e+04,
                   ...
                   2.75000e+01, 2.25000e+01, 1.87500e+01, 1.62500e+01, 1.37500e+01,
                   1.12500e+01, 9.37500e+00, 8.12500e+00, 6.87500e+00, 5.62500e+00])
            >>> new_f = freqInterpolation(f, buffer = buffer)
            >>> new_f 
            ... array([5.88000000e+04, 4.93928459e+04, 4.14907012e+04, 3.48527859e+04,
                   2.92768416e+04, 2.45929681e+04, 2.06584471e+04, 1.73533927e+04,
                   ...
                   2.74153120e+01, 2.30292565e+01, 1.93449068e+01, 1.62500000e+01])
                
        """
        kind =str (kind).lower().strip() 
        if kind.find('peri')>=0 :
            kind ='periods'
        y = 1./ np.array (y) if kind =='periods' else  np.array (y)
        
        buffer = Processing.controlFrequencyBuffer(y, buffer ) 
        ix_s, ix_end  =  np.argwhere (np.isin(y, buffer)) 
    
        y = y[slice ( int(ix_s), int(ix_end) +1)]
        # put frequency in logspace and return
        # the same order like the input value
        y = np.log10 (y)
        if y[0] < y[-1]: 
            f = np.logspace(y.min() ,y.max() , len(y))
        else : 
            f = np.logspace(y.min(),y.max() , len(y))[::-1]
        
        return f 
    
    @staticmethod 
    def controlFrequencyBuffer (
            freq: ArrayLike[DType[T]], 
            buffer:Optional[Tuple[float]] = None 
            )-> ArrayLike[DType[T]] :
        """ Assert the frequency buffer and find the nearest value if the 
        value of the buffer is not in frequency ranges .
        
        :param freq: array-like of frequencies 
        :param buffer: list of maximum and minimum frequency. It should contains 
            only two values. If `None`, the max and min frequencies are selected 
        :returns: Buffer frequency range 
        
        :Example: 
        >>> import numpy as np 
        >>> from watex.methods.em import Processing
        >>> freq_ = np.linspace(7e7, 1e0, 20) # 20 frequencies as reference
        >>> buffer = Processing.controlFrequencyBuffer(freq_, buffer =[5.70e7, 2e1])
        >>> freq_ 
        ... array([7.00000000e+07, 6.63157895e+07, 6.26315791e+07, 5.89473686e+07,
               5.52631581e+07, 5.15789476e+07, 4.78947372e+07, 4.42105267e+07,
               4.05263162e+07, 3.68421057e+07, 3.31578953e+07, 2.94736848e+07,
               2.57894743e+07, 2.21052638e+07, 1.84210534e+07, 1.47368429e+07,
               1.10526324e+07, 7.36842195e+06, 3.68421147e+06, 1.00000000e+00])
        >>> buffer 
        ... array([5.52631581e+07, 1.00000000e+00])
        
        """
        
        if buffer is not None: 
            if np.iterable(buffer): 
                if 1 < len(buffer) > 2 :
                    raise ValueError('Frequency buffer expects two values [max, min].'
                                     f' But {"is" if len(buffer)==1 else "are"} given ')
                if len(set(buffer))==1: 
                    raise ValueError('Expect distinct values [max, min].'
                                     f'{str(buffer[0])!r} is given in twice.')
                    
                for i, v in enumerate(buffer) : 
                    
                    if str(v).lower() =='min': 
                        buffer[i] = freq.min() 
                    elif str(v).lower() =='max': 
                        buffer[i]= freq.max() 
                    elif isinstance(v, str): 
                        raise ValueError(f"Expect 'min' or 'max' not: {v!r}")
                    # Find the absolute difference with each value   
                    # Get the index of the smallest absolute difference
                    arr_diff = np.abs(freq - v )
                    buffer[i] = freq[arr_diff.argmin()]
                
                buffer = np.array(buffer) 
                buffer.sort() ;  buffer = buffer [::-1]
                
        if buffer is None: 
            buffer = np.array([freq.max(), freq.min()])
            
        if buffer.min() < freq.min(): 
            raise ValueError(
                f'Given value {round(buffer.min(), 4) } is out of frequency range.'
                f' Expect a frequency greater than or equal to {round(freq.min(), 4)}'
                )
            
        if buffer.max() > freq.max() : 
            raise ValueError(
                f'Given value {round(buffer.max(),4)} is out of frequency range.'
                f' Expect a frequency less than or equal {round(freq.max(),4)}')
            
        return buffer 


    def qc (self, 
            data: Optional [str|List[EDIO]]=None ,
            * ,  
            tol: float = .5 , 
            return_freq: bool =False 
            )->Tuple[float, ArrayLike]: 
        """ Check the quality control of the collected EDIs. 
        
        Analyse the data in the EDI collection and return the quality control value.
        It indicate how percentage are the data to be representative.
       
        :param data: Path-like object or list  of  pycsamt.core.edi.Edi objects 
                Collections of EDI-objects from `pycsamt`_
        :param tol: float, 
            the tolerance parameter. The value indicates the rate from which the 
            data can be consider as meaningful. Preferably it should be less than
            1 and greater than 0. At this value. Default is ``.5`` means 50 % 
            
        :param return_freq: bool 
            return the interpolated frequency if set to ``True``. Default is ``False``.
            
        :returns: Tuple (float , index )  or (float, array-like, shape (N, ))
            return the quality control value and interpolated frequency if  
            `return_freq`  is set to ``True`` otherwise return the index of useless 
            data. 
            
        :Example: 

            >>> from watex.methods.em import Processing
            >>> pobj = Processing().fit('data/edis')
            >>> f = pobj.getfullfrequency (pobj.ediObjs_)
            >>> len(f)
            ... 55 # 55 frequencies 
            >>> c, = pobj.qc (pobj.ediObjs_, tol = .6 ) # mean 60% to consider the data as
            >>> # representatives 
            >>> c  # the representative rate in the whole EDI- collection
            ... 0.95 # the whole data at all stations is safe to 95%. 
            >>> # now check the interpolated frequency 
            >>> c, freq_new,  = pobj.qc (pobj.ediObjs_, tol=.6 , return_freq =True)
            >>> len(freq_new)
            ... 53  # delete two frequencies 
            
        """
        if isinstance (tol, str): 
            tol = tol.replace('%', '')
        try : 
            tol = float (tol)
        except TypeError : 
            raise TypeError (f"Unable to convert {type(tol).__name__!r} "
                             "to float.")
        except ValueError: 
            raise ValueError(f"Expect 'float' not {type(tol).__name__!r}: "
                             f"{(tol)!r}")
        if tol ==0.: 
            raise ValueError ("Expect a value  greater than '0' and less than '1.'")
            
        if 1 < tol <=100: 
            tol /= 100. 
        if tol > 100: 
            raise ValueError ("Value should be greater than '0' and less than '1'")
            
        if data is not None: 
            self.data_ = data 
        if self.ediObjs_ is None: 
            self.fit(self.data_)
        
        f=self.freqs_.copy() 
     
        try : 
            # take a sample of collected edi 
            # and make two day array
            # all frequency at all stations 
            ar = self.make2d (self.ediObjs_, 'freq') 
        except : 
            try : 
                ar = self.make2d(self.ediObjs_, 'zxy')
            except: ar = self.make2d (self.ediObjs_, 'zyx')
        # compute the ratio of NaN in axis =0 
        
        nan_sum  =np.nansum(np.isnan(ar), axis =1) 
        rr= np.around ( nan_sum / ar.shape[1] , 2) 
        # compute the ratio ck
        # ck = 1. -    rr[np.nonzero(rr)[0]].sum() / (
        #     1 if len(np.nonzero(rr)[0])== 0 else len(np.nonzero(rr)[0])) 
        ck =  (1. * len(rr) - len(rr[np.nonzero(rr)[0]]) )  / len(rr) 
        
        index = reshape (np.argwhere (rr > tol))
        ar_new = np.delete (rr , index , axis = 0 ) 
        new_f = np.delete (f[:, None], index, axis =0 )
        # interpolate freq 
        if f[0] < f[-1]: 
            f =f[::-1] # reverse the array 
            ar_new = ar_new [::-1] # or np.flipud(np.isnan(ar)) 
        
        new_f = np.logspace(np.log10(new_f.min()) ,np.log10(new_f.max()),
                            len(new_f))[::-1]
        
        return np.around (ck, 2), new_f   if return_freq else index   


    @updateZ(option = 'write')
    def getValidData(self, 
                     data:Optional[str|List[EDIO]]=None,
                     tol:float = .5 ,  
                     **kws 
                     )-> NDArray[DType[complex]]: 
        """ Rewrite EDI with the valid data.  
        
        Function analyzes the data  to keep the good ones. The goodness of the data 
        depends on the  `threshold` rate.  For instance 50% means to consider an 
        impedance tensor 'z'  valid if the quality control shows at least that score 
        at each frequency of all stations.  
        
        Parameters 
        ----------
        data: Path-like object or list of  :class:`pycsamt.core.edi.Edi`  
            collections of EDI-objects from `pycsamt`_ 
                
        tol : float, 
            tolerance parameter. The value indicates the rate from which the data 
            can be consider as a valid. The valid data selection should be soft when
            the tolerance parameter  is  close to '1' and hard otherwise. As the 
            `tol` value decreases, the selection  becomes severe. 
            Default is ``.5`` means 50 %  
            
        kws: dict , 
            Additional keywords arguments for EDI file exporting 
            
        Returns 
        -------
        Zc:class:`pycsamt.core.z.Z` impedance tensor objects.
            
        Examples 
        --------
        >>> from watex.methods.em import Processing 
        >>> pObj = Processing ().fit('data/edis')
        >>> f= pObj.freqs_
        >>> len(f) 
        ... 55
        >>> zObjs_soft = pObj.getValidData (ediObjs, tol= 0.3, 
                                         option='None' ) # None doesn't export EDI-file
        >>> len(zObjs_soft[0]._freq) # suppress 3 tensor data 
        ... 52 
        >>> zObjs_hard  = pObj.getValidData(p.ediObjs_, tol = 0.6 )
        >>> len(zObjs_hard[0]._freq)  # suppress only two 
        ... 53
        
        """
        
        def delete_useless_tensor (z ,  index , axis = 0):
            """Remove meningless tensor data"""
            return np.delete (z, index , axis )
        def set_null(freq, objs): 
            """Set null in the case the component doesn't exist"""
            return np.zeros ((len(f), len(objs)), dtype = np.float32)
        
        if data is not None: 
            self.data_= data 
        if self.ediObjs_ is None: 
            self.fit(self.data_)
        # ediObjs = get_ediObjs(ediObjs) 
        _, no_ix = self.qc(self.ediObjs_ , tol= tol  ) 
        f = self.freqs_.copy() 
    
        ff = np.delete (f[:, None], no_ix, 0)
        # interpolate frequency 
        new_f  = Processing.freqInterpolation (reshape (ff)) 
        
        # gather the 2D z objects
        
        # -XX--
        try : 
            zxx = delete_useless_tensor(
                self.make2d (self.ediObjs_, 'zxx'), no_ix) 
            zxx = interpolate2d(zxx)
            zxx_err = delete_useless_tensor(
                self.make2d (self.ediObjs_, 'zxx_err') , no_ix ) 
            zxx_err = interpolate2d (zxx_err )
        except :
            zxx = set_null(new_f, self.ediObjs_)
            zxx_err= zxx.copy() 
            
        # -XY--    
        try :
            zxy = delete_useless_tensor(
                self.make2d (self.ediObjs_, 'zxy'), no_ix )  
            zxy= interpolate2d( zxy)
            zxy_err = delete_useless_tensor( 
                self.make2d (self.ediObjs_, 'zxy_err') , no_ix )
            zxy_err = interpolate2d(zxy_err)
        except: 
            zxy = set_null(new_f, self.ediObjs_)
            zxy_err= zxy.copy() 
    
        # -YX--
        try:
        
            zyx = delete_useless_tensor( 
                self.make2d (self.ediObjs_, 'zyx') , no_ix ) 
            zyx = interpolate2d(zyx)
            zyx_err = delete_useless_tensor(
                self.make2d (self.ediObjs_, 'zyx_err') , no_ix ) 
            zyx_err = interpolate2d( zyx_err )
        except: 
            zyx = set_null(new_f, self.ediObjs_)
            zyx_err= zyx.copy() 
            
        # -YY--
        try:
            zyy = delete_useless_tensor(
                self.make2d (self.ediObjs_, 'zyy'), no_ix ) 
            zyy = interpolate2d(zyy)
            zyy_err = delete_useless_tensor(
                self.make2d (self.ediObjs_, 'zyy_err') , no_ix ) 
            zyy_err = interpolate2d(zyy_err)
            
        except :  
            zyy = set_null(new_f, self.ediObjs_)
            zyy_err= zyy.copy() 
            
      
        z_dict = { 'zxx': zxx ,'zxy': zxy ,
                    'zyx': zyx,'zyy': zyy, 
                    'zxx_err': zxx_err ,'zxy_err': zxy_err ,
                    'zyx_err': zyx_err, 'zyy_err': zyy_err
            } 
        
        return (self.ediObjs_, new_f , z_dict ), kws



       

    
  
