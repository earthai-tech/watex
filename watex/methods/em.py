# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
:mod:`~watex.methods.em` module is related for a few meter exploration in 
the case of groundwater exploration. Module provides some basics processing 
steps for EMAP data filtering and remove noises. 
"""
from __future__ import annotations 
import os
import re
import functools 
import warnings 
import numpy as np 

from .._watexlog import watexlog
from ..edi import Edi 
from ..exceptions import ( 
    EDIError, 
    TopModuleError, 
    FrequencyError, 
    NotFittedError, 
    EMError,
    ZError, 
    
) 
from ..externals.z import Z as EMz 
from ..utils.funcutils import ( 
    _assert_all_types,
    is_iterable,
    assert_ratio,
    make_ids,
    show_stats, 
    fit_by_ll, 
    reshape, 
    smart_strobj_recognition, 
    repr_callable_obj,
    remove_outliers, 
    normalizer, 
    random_selector, 
    listing_items_format
    ) 
from ..utils.exmath import ( 
    scalePosition, 
    fittensor,
    get2dtensor, 
    betaj, 
    interpolate1d,
    interpolate2d, 
    get_full_frequency, 
    find_closest, 
    rhoa2z, 
    z2rhoa, 
    mu0,
    )
from ..utils.coreutils import ( 
    makeCoords, 
    )
from ..property import (
    IsEdi
    )
from ..site import Location 
from .._typing import ( 
    ArrayLike, 
    Optional, 
    List,
    Tuple, 
    Dict, 
    NDArray, 
    DType,
    EDIO,
    ZO, 
    T,
    F, 
    )
from ..utils._dependency import ( 
    import_optional_dependency
    )
from ..utils.validator import ( 
    _validate_tensor, 
    _assert_z_or_edi_objs
    )

_logger = watexlog.get_watex_logger(__name__)

__all__ =['EM',
          'Processing',
          'ZC',
          ]

class EM(IsEdi): 
    """
    Create EM object as a collection of EDI-file. 
    
    Collect edifiles and create an EM object. It sets  the properties from 
    audio-magnetotelluric. The two(2) components XY and YX will be set and 
    calculated.Can read MT data instead, however the full handling transfer 
    function like Tipper and Spectra  is not completed. Use  other MT 
    softwares for a long periods data. 
    
    Parameters 
    -------------
    survey_name: str 
        location name where the date where collected . If surveyname is None  
        can chech on edifiles. 

    Attributes 
    -----------
    ediObjs_: Array-like of shape (N,) 
        array of the collection of edifiles read_sucessfully  
    data_: Array-like of shape (N, ) 
        array of all edifiles feed in the `EM` modules whatever sucessuffuly 
        read or not. 
    edinames_: array-like of shape (N,) 
        array of all edi-names sucessfully read 
    edifiles_: array of shape (N, ) 
        array of all edifiles if given. 
    freqs_: array-like of shape (N, ) 
        Array of the frequency range from EDIs 
    refreq_: float, 
        Reference refrequency for data correction. Note the reference frequency
        is the highest frequency with clean data. 
        
    Properties 
    ------------
    
    longitude: array-like, shape (N,) 
        longitude coordinate values  collected from EDIs 
        
    latitude: array-like, shape (N, )
        Latitude coordinate values collected from EDIs 
        
    elevation: array-like, shape (N,) 
        Elevation coordinates collected from EDIs 
        
    """

    def __init__(self, survey_name:str  =None , verbose=0): 
        self._logging = watexlog.get_watex_logger(self.__class__.__name__)
    
        self.survey_name =survey_name
        self.Location_= Location()
        self.verbose=verbose
        self._latitude = None
        self._longitude=None
        self._elevation= None 

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
        )-> Edi  : 
        """Assert that the given argument is an EDI -object from modules 
        EDI or EDI from pycsamt and MTpy packages. 
        A TypeError will occurs otherwise.
        
        Parameters 
        ------------
        obj: str, :class:`pycsamt.core.edi.Edi` or :class:`mtpy.core.edi.Edi` 
            Full path EDI file or `pycsamt`_ or `MTpy`_ objects.
       
        Return
        -------
        obj:str, :class:`pycsamt.core.edi.Edi` or :class:`mtpy.core.edi.Edi`
            Identical object after asserting.
        
        """
        emsg=("{0!r} object detected while the package is not installed"
              " yet. To pass ['pycsamt' | 'mtpy'] EDI-object to 'EM'"
              " class for basic implementation,prior install the {0!r}"
              " first. Use 'pip' or 'conda' for installation."
             )
        IsEdi.register (Edi)
        if isinstance(obj, str):
            obj = Edi().fit(obj) 
   
        if "pycsamt" in str(obj):   
            try : 
                import_optional_dependency ("pycsamt")
            except ImportError: 
                raise TopModuleError(emsg.format("pycsamt"))
            else : 
                #XXX TODO : prior revising the pkg structure 
                # to pycsamt.core since ff subpackage does
                # no longer exist in pycsamt newest version
                from pycsamt.ff.core import edi 
                IsEdi.register (edi.Edi )
                
        elif "mtpy" in str(obj): 
            try : 
                import_optional_dependency ("mtpy")
            except ImportError: 
                raise TopModuleError(emsg.format("mtpy"))
            else : 
                from mtpy.core import edi
                IsEdi.register (edi.Edi)
                
        try : 
            obj = _assert_all_types (
                obj, IsEdi, objname="Wrong Edi-Objects or EDI-path,")
        except ( TypeError, AttributeError): 
            # force checking instead
            obj = _assert_all_types (obj, Edi, objname="EDI")
            
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
        
    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'ediObjs_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 
        
    def _read_emo (self, d, / ): 
        """ read, assert and EDI parse data. 
        Parameters
        ----------
        d: string, list 
           EDI path-like object when value is passed as string. 
           List of EDI-file or EDI objects. 
        """ 
        
        emsg =("Wrong EDI {}. More details about the SEG-EDI can be found"
               " in <https://www.mtnet.info/docs/seg_mt_emap_1987.pdf> "
               )
        types =None # object 
        rf =[] # count number of reading files.
        
        # when d is str; expect to be a path or file 
        if isinstance (d, str): 
            if os.path.isfile (d):
                types = 'ef'# edifile 
                if not self._assert_edi(d, False):
                    raise EDIError(emsg.format('file'))
                    
                rf= [ d ] 
                # now build an EDI object with valid EDI
            elif os.path.isdir (d): 
                types='ep' #edipath 
                # count number of files. 
                rf = os.listdir (d ) # collect all files in the path 
                
                d = sorted ([ os.path.join(d, edi ) 
                             for edi in rf if edi.endswith ('.edi')])  
            else: 
                raise EDIError(emsg.format('object')) 
                
        self.data_ = is_iterable(d, exclude_string =True , transform =True )
        try :
            self.data_= sorted( self.data_) # sorted edi 
        except : 
            # skip TypeError: '<' not supported between instances 
            # of 'Edi' and 'Edi'
            pass 
    
        if not types: rf = self.data_ # if objects is set 
        
        # read EDI objects 
        try: 
            self.ediObjs_ = list(map(lambda o: self.is_valid(o), self.data_))  
        except EDIError: 
            objn = type(self.ediObjs_[0]).__name__
            raise EMError (f"Expect a list of EDI objects. Got {objn!r}") 
        
        if self.verbose:
            try:show_stats(rf, self.ediObjs_)
            except: pass 
        

    def _get_tensor_and_err_values (self, attr ): 
        """ Get tensor with error and put in dictionnary 
        of station/tensor values.
        
        :param attr: attribute from Z objects.
        :return: A 3D tensor (nfreq, 2, 2) and tensor at each station.  
        :example: 
            >>> import watex 
            >>> test_edi= watex.datasets.load_edis (
                samples =2 , return_data =True ) 
            >>> et = watex.EM().fit(test_edi )
            >>> et._get_tensor_and_err_values ('z')
        """
        self.inspect 
        # ---> get impedances, phase tensor and
        # resistivities values form ediobject
        self._logging.info('Setting impedances and phases tensors and'
                           'resistivity values from a collection ediobj')
        t = [getattr (edi_obj.Z, attr ) for edi_obj in self.ediObjs_]
        # put all station ( key) /values in
        tdict = {key:value for key , value in zip (self.id, t)}
        
        return t, tdict
    
    def fit(self, 
             data: str|List[EDIO]
             )->"EM":
        """
        Assert and make EM object from a collection EDIs. 
        
        Parameters 
        ----------- 
        data : str, or list or :class:`pycsamt.core.edi.Edi` object 
            Full path to EDI files or collection of EDI-objects 
            
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
   
        self._read_emo(data ) 
        # sorted ediObjs from latlong  
        self.ediObjs_ , self.edinames = fit_by_ll(
            self.ediObjs_)
        # reorganize  edis according 
        # to lon lat order. 
        self.edifiles = list(map(
            lambda o: o.edifile , self.ediObjs_))

        #--get coordinates values 
        # and correct lon_lat ---
        lat  = _fetch_headinfos(
            self.ediObjs_, 'lat')
        lon  = _fetch_headinfos(
            self.ediObjs_, 'lon')
        elev = _fetch_headinfos(
            self.ediObjs_, 'elev')
        lon,*_ = scalePosition(
            lon) if len(self.ediObjs_)> 1 else lon 
        lat,*_ = scalePosition(
            lat) if len(self.ediObjs_)> 1 else lat
        
        # Create the station ids 
        self.id = make_ids(self.ediObjs_, prefix='S')
    
        self.longitude= lon 
        self.latitude= lat  
        self.elevation= elev
        try : 
            self.elevation= self.elevation.astype (
                np.float64)
        except :pass 
        self.stnames = self.edinames 
        # get frequency array from the first value of edifiles.
        self.freq_array = self.ediObjs_[0].Z.freq

        self.freqs_ = self.getfullfrequency ()
        self.refreq_ = self.getreferencefrequency()
        
        return self 
    
    def rewrite (
        self, 
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
        )-> "EM": 
        
        """ Rewrite Edis, correct station coordinates and dipole length. 
        
        Can rename the dataid,  customize sites and correct the positioning
        latitudes and longitudes. 
        
        Parameters 
        ------------
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
            :func:`watex.utils.coreutils.makeCoords`. 
            
        Returns 
        --------
        EM: :class:`~.EM` instance
             Returns ``self`` for easy method chaining.
            
        Examples
        ---------
        >>> from watex.methods.em import EM
        >>> edipath = r'data/edis'
        >>> savepath =  r'/Users/Daniel/Desktop/ediout'
        >>> emObjs = EM().fit(edipath)
        >>> emObjs.rewrite_edis(by='id', edi_prefix ='b1',
                                savepath =savepath)
        >>> # 
        >>> # second example to write 7 samples of edi from 
        >>> # Edi objects inner datasets 
        >>> #
        >>> import watex as wx 
        >>> edi_sample = wx.fetch_data ('edis', key ='edi', 
                                        samples =7, return_data =True ) 
        >>> emobj = wx.EM ().fit(edi_sample)
        >>> emobj.rewrite(by='station', prefix='PS')
        
        """
        def replace_reflatlon (  olist , nval, kind ='reflat'):
            """ Replace Definemeasurement Reflat and Reflong by the interpolated
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

        self.inspect 
        
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
       
        long, lat = self.longitude, self.latitude 
        
        if make_coords: 
            if (reflong or reflat) is None: 
                raise ValueError('Reflong and reflat params must not be None!')
            long, lat = makeCoords(
               reflong = reflong, reflat= reflat, nsites= len(self.ediObjs_),
               step = step , **kws) 
        # clean the old main Edi section info and 
        # and get the new values
        if correct_ll:
            londms,*_ = scalePosition(self.longitude)
            latdms,*_ = scalePosition(self.latitude) 

        # collect new ediObjs 
        cobjs = np.zeros_like (self.ediObjs_, dtype=object ) 
        
        for k, (obj, did) in enumerate(zip(self.ediObjs_, dataid)): 
            obj.Head.edi_header = None  
            obj.Head.dataid = did 
            obj.Info.ediinfo = None 
            if correct_ll or make_coords:
                obj.Head.long = float(long[k])
                obj.Head.lat = float(lat[k])
                obj.Head.elev = float(self.elevation[k])
                oc = obj.DefineMeasurement.define_measurement
                oc= replace_reflatlon(oc, nval= latdms[k])
                oc= replace_reflatlon(oc, nval= londms[k],  kind='reflong')
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

    def getfullfrequency  (
            self, 
            to_log10:bool  =False 
        )-> ArrayLike[DType[float]]: 
        """ Get the frequency with clean data. 
        
        The full or plain frequency is array frequency with no missing  data during 
        the data collection. Note that when using |NSAMT|, some data are missing 
        due to the weak of missing frequency at certain band especially in the 
        attenuation band. 
        
        Parameters 
        -----------
        to_log10: bool, default=False, 
            export frequency to base 10 logarithm 
            
        Returns 
        --------
        f: Arraylike 1d of shape(N, )
            frequency with clean data. Out of `attenuation band` if survey 
            is completed with  |NSAMT|. 
        
        See Also
        --------
        watex.utils.exmath.get_full_frequency: 
            Get the complete frequency with no missing signals. 
            
        Example 
        -------
        >>> import watex as wx 
        >>> edi_sample = wx.fetch_data ('edis', return_data=True, samples = 12 )
        >>> wx.EM().fit(edi_sample).getfullfrequency(to_log10 =True )
        array([4.76937733, 4.71707639, 4.66477553, 4.61247466, 4.56017382,
               4.50787287, 4.45557204, 4.40327104, 4.35097021, 4.29866928,
               4.24636832, 4.19406761, 4.14176668, 4.08946565, 4.03716465,
               ...
               2.67734228, 2.62504479, 2.57274385, 2.52044423, 2.46814047,
               2.41584107, 2.36353677, 2.31124512, 2.25892448, 2.20663701,
               2.15433266, 2.10202186, 2.04972182, 1.99743007])

        """
        self.inspect 
        return get_full_frequency (self.ediObjs_ , to_log10 = to_log10) 

    def make2d (
        self,
        out:str = 'resxy',
        *, 
        kind:str = 'complex' , 
        **kws 
        )-> NDArray[DType[float]]: 
        """ Out 2D resistivity, phase-error and tensor matrix from a collection
        of EDI-objects. 
        
        Matrix depends of the number of frequency times number of sites. 
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
            ``resxy``. 
        kind : bool or str 
            focuses on the tensor output. Note that the tensor is a complex number 
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
        >>> phyx = EM().make2d ('phaseyx')
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
        >>> resxy_err = EM.make2d ('resxy_err')
        >>> resxy_err 
        ... array([[0.01329037, 0.02942557, 0.0176034 ],
               [0.0335909 , 0.05238863, 0.03111475],
               ...
               [3.33359942, 4.14684926, 4.38562271],
               [       nan,        nan, 4.35605603]])
        >>> phyx.shape ,zyy_r.shape, resxy_err.shape  
        ... ((55, 3), (55, 3), (55, 3))
        
        """
        self.inspect 
        
        def fit2dall(objs, attr, comp): 
            """ Read all ediObjs and replace all missing data by NaN value. 
            
            This is useful to let the arrays at each station to  match the length 
            of the complete frequency rather than shrunking  up some data. The 
            missing data should be filled by NaN values. 
            
            """
            # use  [getattr( ediObj.Z, f"{attr}")[tuple ( _c.get(comp))... 
            # where 
            # #=> slice index for component retreiving purpose 
            # _c= {
            #       'xx': [slice (None, len(self.freqs_)), 0 , 0] , 
            #       'xy': [slice (None, len(self.freqs_)), 0 , 1], 
            #       'yx': [slice (None, len(self.freqs_)), 1 , 0], 
            #       'yy': [slice (None, len(self.freqs_)), 1,  1] 
            # }
            
            zl = [getattr( ediObj.Z, f"{attr}")[tuple ( 
                self.tslicer().get(comp))]
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
           
        # read/assert edis and get the complete frequency     
        name , m2 = _validate_tensor(out = out , kind = kind, **kws )
        
        #==> returns mat2d freq 
        # if m1 =='_freq': 
        if name =='_freq': 
            f2d  = [fittensor(self.freqs_, ediObj.Z._freq, ediObj.Z._freq)
                  for ediObj in self.ediObjs_
                  ]
            return  np.hstack ([ reshape (o, axis=0) for o in f2d])
        
        # # get the value for exportation (attribute name and components)
        mat2d  = fit2dall(objs= self.ediObjs_, attr= name, comp= m2)
        
        return mat2d 
 
    def getreferencefrequency (
        self,
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
        self.inspect 
        self.freqs_= self.getfullfrequency ()
        # fit z and find all missing data from complete frequency f 
        # we take only the component xy for fitting.

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
    
    def _exportedi (self, ediObj: EDIO)-> "EDIO" :
        """Isolated part for validate EDI for multiple EDI exports 
        (:meth:`exportedis`). 
        """
        try : 
            ediObj = self.is_valid(ediObj ) 
        except : 
            if isinstance (ediObj, str):
                ediObj = Edi().fit(ediObj )
                
            if hasattr (ediObj, '__iter__'): 
                ediObj = ediObj[0]
                if  _assert_z_or_edi_objs(ediObj )!='EDI' :
                    raise EDIError("Obj {ediObj!r} is not an EDI-object.")
        return ediObj 
       
    def exportedis(
        self, 
        ediObjs: List [EDIO], 
        new_Z: List [ZO], 
        savepath = None, 
        **kws 
        ): 
        """Export EDI files from multiples EDI or z objects
        
        Export new EDI file from the former object with  a given new  
        impedance tensors. The export is assumed a new output EDI 
        resulting from multiples corrections applications.
        
        Parameters 
        -----------
        ediObjs: list of string  :class:`watex.edi.Edi` 
            Full path to Edi file/object or object from class:`EM` objects. 
  
        new_Z: list of ndarray (nfreq, 2, 2) 
            A collection of Ndarray of impedance tensors Z. 
            The tensor Z is 3D array composed of number of frequency 
            `nfreq`and four components (``xx``, ``xy``, ``yx``,
            and ``yy``) in 2X2 matrices. The  tensor Z is a complex number. 
            
        savepath:str, Optional 
           Path to save a new EDI file. If ``None``, outputs to `_outputEDI_`
           folder.
           
        Returns 
        --------
         ediObj from watex.edi.Edi 
         
        See Also
        ---------
        exportedi: 
            Export single EDI from
            
        """
        ediObjs = is_iterable(
            ediObjs , 
            exclude_string =True , 
            transform =True 
            )
        new_Z = is_iterable(
            new_Z , 
            exclude_string =True , 
            transform =True 
            )
        
        for e, z  in zip (ediObjs, new_Z ): 
            e= self._exportedi(e) 
            e.write_new_edifile( 
                new_Z=z,
                savepath = savepath , 
                **kws
                )
            
    def tslicer ( 
        self,  
        freqs= None, 
        z= None, 
        component ='xy'
        ): 
        """Returns tensor 2d from components 
        
        Parameters 
        -----------
        freqs: arraylike 
           full frequency that composed the tensor. If ``None``, use the 
           components in 
        Z: ArrayLike 3D 
           Tensor is composed of 3D array of shape (n_freqs, 2, 2) 
        component: str, 
          components along side to retrieve . Can be ['xx'|'xy'|'yx'|'yy']
          
        .. versionadded:: v0.2.0 
        
        Returns 
        ---------
        z or slice: Arralike 2D tensor, or dict 
           Returns 2D tensor or dictionnary of components index slicers. 
           
        """
        # set component slices into a dict
        freqs = freqs if freqs is not None else  self.freqs_ 
        
        c = {
              'xx': [slice (None, len(freqs)), 0 , 0] , 
              'xy': [slice (None, len(freqs)), 0 , 1], 
              'yx': [slice (None, len(freqs)), 1 , 0], 
              'yy': [slice (None, len(freqs)), 1,  1] 
        }
        
        return  z [tuple (c.get(component))] if z is not None else c 
       
    @staticmethod 
    def get_z_from( edi_obj_list, /, ): 
        """ Get z object from Edi object. 
        Parameters 
        -----------
        z_or_edis_obj_list: list of :class:`watex.edi.Edi` or \
            :class:`watex.externals.z.Z` 
            A collection of EDI- or Impedances tensors objects. 
            
        .. versionadded:: v0.1.9 
        
        Returns
        --------
        Z: list of :class:`watex.externals.z.Z`
           List of impedance tensor Objects. 
          
        """
        obj_type  = _assert_z_or_edi_objs (edi_obj_list)
        return   edi_obj_list  if obj_type =='Z' else [
            edi_obj_list[i].Z for i in range (len( edi_obj_list)  )]   
    
    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        return repr_callable_obj  (self, skip =('edifiles', 'freq_array', 'id'))
       
    
    def __getattr__(self, name):
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        appender  = "" if rv is None else f'. Do you mean {rv!r}'
        
        raise AttributeError (
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{appender}{"" if rv is None else "?"}'
            )
        
class _zupdate(EM): 
    """ A decorator for impedance tensor updating. 
    
    Update a Z object from each EDI object composing the collection objects 
    and output a new EDI-files if `option` is set to ``write``. 
    
    Parameters 
    ------------
    
    option: str , default='write'
        kind of action to perform with new Z collection.
        When `option` is set to ``write``. The new EDI -files are exported.
        Any other values should return only the updated impedance tensors.
        
    Returns 
    --------
    Z |None : A collection of  :class:`watex.externals.z.Z` impedance tensor 
        objects. None if `option` is to ``write``. 
        
    """
    
    def __init__(self, option: str = 'write'): 
        self.option = option
    
    def __call__ (self, func:F):
        
        @functools.wraps (func)
        def new_func ( *args, **kws): 
            """ Wrapper  to make new Z. The updated Z is a collection
            object from ':class:`pycsamt.core.z.Z` """
            
            (ediObjs , freq,  z_dict), kwargs = func (*args, **kws)
    
            # pop the option argument if user provides it 
            option = kwargs.pop('option', None)
            rotate = kwargs.pop ('rotate', 0. ) 
            ediO= kwargs.pop('ediObjs', None )
            
            self.option = str(option).lower()  or self.option 
            # create an empty array to collect each new Z object 
           
            if isinstance (z_dict, dict ): 
                # # rather than using ediObjs 
                # which could be None if the idea 
                # is just to correct the tensor z
                # therefor fetch the shape [1] of a random 
                # components which correspond 
                # to the number of stations of collected EDI 
                nsites = list( z_dict.values()) [0].shape [1 ] 

                Zc = np.empty((nsites, ), dtype = object )
                for kk in range  (nsites):
                    # create a new Z object for each Edi
                    Z= self._make_zObj(kk,freq=freq, 
                                       z_dict = z_dict , 
                                       rotate= rotate, 
                                       )
                    Zc[kk] =Z
            else : 
                # write corrected EDis with no loops 
                Zc = z_dict 
            
            if self.option in ('write', 'true') :
                objtype = _assert_z_or_edi_objs(ediObjs)
                if objtype =='Z'and ediO is None:
                    raise EMError ("Missing EDI-objects. Z object only cannot"
                                   " be exported. Provide the EDI-objects as"
                                   " a keyword argument `ediObjs` to collect"
                                   " the metadata of the site <Head-Infos-"
                                   " Measurement...> for rewritting. It does "
                                   " not alter the new corrected or"
                                   " interpolated Z.")
                
                ediO = ediObjs if objtype=='EDI'else ediO
                ediO = is_iterable(
                    ediO , 
                    exclude_string =True, 
                    transform =True 
                    )
                assert len(ediO) ==len(Zc), (
                    "EDI-objects and Z collection must be consistent."
                    f" Got {len(ediO)} and {len(Zc)}.")
                
                self.exportedis(
                    ediObjs=ediO, 
                    new_Z=Zc, 
                    **kwargs
                    ) 
                
            return Zc if self.option !='write' else None 
          
        return new_func 
    
    def _make_zObj (
        self, 
        kk: int ,
        *, 
        freq: ArrayLike[DType[float]], 
        z_dict: Dict[str, NDArray[DType[complex]]], 
        rotate=0.
        )-> NDArray[DType[complex]]: 
        """ Make new Z object from frequency and dict tensor component Z. 
        
        :param kk: int 
            index of routine to retrieve the tensor data. It may correspond of 
            the station index. 
        :param freq: array-like 
            full frequency of component 
        :param z_dict: dict, 
            dictionnary of all tensor component.
        :param rotate: float, 
          Angle to rotate the impedance tensor accordingly. 

        """
        Z = EMz(
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
            Z._z[:, 0,  0] = reshape (zxx[:, kk], 1) 
        if zxy is not None: 
            Z._z[:, 0,  1] = reshape (zxy[:, kk], 1)
        if zyx is  not None: 
            Z._z[:, 1,  0] = reshape (zyx[:, kk], 1) 
        if zyy is not None: 
            Z._z[:, 1,  1] = reshape (zyy[:, kk], 1)
            
        # set the z_err 
        zxx_err = z_dict.get('zxx_err') 
        zxy_err = z_dict.get('zxy_err') 
        zyx_err = z_dict.get('zyx_err') 
        zyy_err = z_dict.get('zyy_err') 
        
        if zxx_err is not None: 
            Z._z_err[:, 0,  0] = reshape (zxx_err[:, kk], 1) 
        if zxy_err is not None: 
            Z._z_err[:, 0,  1] = reshape (zxy_err[:, kk], 1)
        if zyx_err is not None: 
            Z._z_err[:, 1,  0] = reshape (zyx_err[:, kk], 1) 
        if zyy_err is not None: 
            Z._z_err[:, 1,  1] = reshape (zyy_err[:, kk], 1)
            
        Z.rotate(rotate ) # rotate angle if given 
        Z.compute_resistivity_phase(
            z_array = Z._z , z_err_array = Z._z_err, freq = freq )

        return Z
 
        
class Processing (EM) :
    """ Base processing of EM object 
    
    Fast process EMAP and AMT data. Tools are used for data sanitizing, 
    removing noises and filtering. 
    
    Parameters 
    ----------
    data: Path-like object or list  of  :class:watex.edi.Edi` or \
        `pycsamt.core.edi.Edi` objects 
        Collections of EDI-objects 
    
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
        Interpolation technique to use. Can also be ``nearest``. Refer to 
        the documentation of :func:`~.interpolate2d`. 
        
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
    >>> plt.semilogy (np.arange (p.res2d_.shape[1] ), p.res2d_[3, :], '--',
                      np.arange (p.res2d_.shape[1] ), rc[3, :], 'ok--')
 
    References 
    -----------
    .. [1] http://www.zonge.com/legacy/PDF_DatPro/Astatic.pdf
        
    """
    
    def __init__(
        self,
        window_size:int =5, 
        component:str ='xy', 
        mode: str ='same', 
        method:str ='slinear', 
        out:str  ='srho', 
        c: int =2, 
        **kws
        ): 
        super().__init__(**kws)
        
        self._logging= watexlog.get_watex_logger(self.__class__.__name__)
        self.window_size=window_size 
        self.component=component 
        self.mode=mode 
        self.method=method 
        self.out=out 
        self.c=c
        

    def tma (
        self,
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
        self.inspect
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
        ix_rf = int(reshape (np.argwhere (self.freqs_==self.refreq_)))  

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
            rc = rhoa2z(rc, self.phs2d_, self.freqs_)

        return cf if self.out =='sf' else rc   

    def _make2dblobs (
        self, 
        ): 
        """ Asserts argument of |EMAP| filter and returns useful arguments.
 
        :note: created to collect argument of EMAP filters. Refer to functions 
        :func:`~.tma`, :func:`~.flma` and :func:`~.ama` documentation. 
            
        """

        self.component= str(self.component).lower().strip() 
        self.out= str(self.out).lower().strip() 
        
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
        
        self.res2d_= self.make2d(out=f'res{self.component}')
        self.phs2d_ = self.make2d(out=f'phase{self.component}')
        
            
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
    
    def ama (
        self, 
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
        self.inspect 

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
            try : 
                
                zjr [ii, :] = np.average (np.vstack (zcr), axis = 0)
                zji[ii, :] = np.average (np.vstack (zci), axis = 0)
            except : 
                # when  array dimensions is out of the concatenation 
                # then shrunk it to match exactly the first array 
                # thick append when window width is higher and the number of 
                # the station 
                shr_zrc = [ ar [: len(zcr[0])] for ar in zcr ]
                shr_zrci = [ ar [: len(zci[0])] for ar in zci ]
                zjr [ii, :] = np.average (np.vstack (shr_zrc), axis = 0)
                zji[ii, :] = np.average (np.vstack (shr_zrci), axis = 0)
     
        zjc = zjr + 1j * zji 
        rc = z2rhoa(zjc, self.freqs_)  
        if self.mode =='same': 
            rc[:, 0] = self.res2d_[:, 0]
            zjc[:, 0] = zj [:, 0]
        
        return zjc if self.out =='z' else rc 

    def flma (
        self, 
        )-> NDArray[DType[float]] :
        """ A fixed-length-moving-average filter to estimate average apparent
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
        self.inspect 
    
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
    
    def skew(
        self,
        method:str ='swift', 
        return_skewness:bool=False, 
        suppress_outliers:bool=False, 
        )-> NDArray[DType[float]]: 
        r"""
        The conventional asymmetry parameter based on the Z magnitude. 
        
        The EM signal is influenced by several factors such as the dimensionality
        of the propagation medium and the physical anomalies, which can distort the
        EM field both locally and regionally. The distortion of Z was determined 
        from the quantification of its asymmetry and the deviation from the conditions 
        that define its dimensionality. The parameters used for this purpose are all 
        rotational invariant because the Z components involved in its definition are
        independent of the orientation system used. The conventional asymmetry
        parameter based on the Z magnitude is the skew defined by Swift (1967) as
        follows:
        
        .. math:: 
        
            skew_{swift}= |\frac{Z_{xx} + Z_{yy}}{ Z_{xy} - Z_{yx}}| 
            
        When the :math:`skew_{swift}`  is close to ``0.``, we assume a 1D or 2D model
        when the :math:`skew_{swift}` is greater than ``>=0.2``, we assume 3D local 
        anomaly (Bahr, 1991; Reddy et al., 1977).  It is generally considered 
        that an electrical structure of :math:`skew < 0.4` can be treated as a 
        2D medium.
        
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
          
            \[C_1, C_2] & = & \text{Im} C_2*C_1^*
            
            \[C_1, C_2]  & = & \text{Re} C_1 * \text{Im}C_2  - R_e(C_2)* \text{Im}C_1
                        
        Indeed, :math:`skew_{Bahr}` measures the deviation from the symmetry condition
        through the phase differences between each pair of tensor elements,considering
        that phases are less sensitive to surface distortions(i.e. galvanic distortion).
        The :math:`skew_{Bahr}` threshold is set at ``0.3`` and higher values mean 
        3D structures (Bahr, 1991).
        
        
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
            
        return_skewness: str, 
           Typically returns the type of skewness. ``'skew'`` or ``mu`` for 
           skew and rotation- all invariant values respectively. Any other 
           value return both skew and rotational invariant. 
 
        suppress_outliers: bool, default=False, 
           Remove the outliers (if applicable in the data ) before
           normalizing. 
           
           .. versionadded:: 0.1.6 
           
        Returns 
        --------- 
        skw, mu : Tuple of ndarray-like , shape (N, M )
            - Array of skew at each frequency 
            - rotational invariant ``mu`` at each frequency that
              measures of phase differences in the impedance tensor. 
            
        See Also
        ---------
        watex.utils.plot_skew: 
            For phase sensistive skew visualization - naive plot. 
        watex.view.TPlot.plotSkew: 
            For consistent plot of phase sensitive skew visualization. Allow 
            customize plots. 
        watex.view.TPlot.plot_phase_tensors: 
            Plot skew as ellipsis visualization by turning the `tensor`
            parameter to ``skew``. 

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

        self.inspect 
            
        self.method = str(method).lower()
        if self.method not in ('swift', 'bahr'): 
            raise ValueError(
                f'Expected argument ``swift`` or ``bahr`` not: {self.method!r}')
            
        return_skewness= str(return_skewness).lower() 
        if  ( 'mu' in return_skewness or 
             'rot' in return_skewness or 
             'invariant' in return_skewness): 
            return_skewness ='mu' 
        elif 'skew' in return_skewness: 
            return_skewness ='skew'
            
        Zxx= self.make2d('zxx')
        Zxy = self.make2d('zxy')
        Zyx = self.make2d('zyx')
        Zyy= self.make2d('zyy')
        
        S1 =Zxx + Zyy; S2 = Zxy + Zyx; D1 =Zxx-Zyy ;  D2= Zxy-Zyx 
        D1S2 = (S2 * np.conj(D1)).imag ; S1D2 = (D2 * np.conj(S1)).imag 
        
        if method =='swift': 
            skw = np.abs ( S1  / D2 )  
        else : 
            skw = np.sqrt(np.abs( D1S2 - S1D2))/np.abs(D2)
            
        mu = np.sqrt(np.abs(D1S2) + np.abs (S1D2))/ np.abs(D2) 
        
        if suppress_outliers: 
            skw = remove_outliers(skw, fill_value= np.nan ) 
            mu = remove_outliers(mu, fill_value= np.nan )
            
        skw = normalizer(skw) ; mu = normalizer(mu )
            
        return skw if return_skewness=='skew' else (
            mu if return_skewness=='mu' else skw, mu)
   

    def zrestore(
        self,
        *, 
        tensor:str=None, 
        component:str=None, 
        buffer: Tuple[float]=None, 
        method:str ='pd',
        **kws 
        ): 
        """ Fix the weak and missing signal at the 'dead-band`- and recover the 
        missing impedance tensor values. 
        
        The function uses the complete frequency (frequency with clean data) 
        collected thoughout the survey to recover by inter/extrapolating the 
        missing or weak  frequencies thereby restoring the impedance tensors 
        at that 'dead-band'. Note that the 'dead- band' also known as 
        'attenuation -band' is where the AMT signal is weak or generally 
        abscent. 
 
        Parameters 
        ----------

        tensor: str, optional, ["resistivity"|"phase"|"z"|"frequency"]
           Name of the :term:`tensor`. If the name of tensor is given, 
           function  returns the tensor valuein two-dimensionals composed 
           of (n_freq , n_sites) where ``n_freq=number of frequency`` 
           and ``n_sations`` number of sites. Note that if the tensor is 
           passed as boolean values ``True``, the ``resistivity`` tensor is 
           exported by default and the ``component``should be the component 
           passed to :class:`Processing` at initialization. 
          
        buffer: list [max, min] frequency in Hz
            list of maximum and minimum frequencies. It must contain only two 
            values. If `None`, the max and min of the clean frequencies are 
            selected. Moreover the [min, max] frequency should not compulsory 
            to fit the frequency range in the data. The given frequency can 
            be interpolated to match the best closest frequencies in the data. 
      
        method: str, optional , default='pd' 
            Method of Z interpolation. Use ``base`` for `scipy` interpolation, 
            ``mean`` or ``bff`` for scaling methods and ``pd`` for pandas 
            interpolation methods. Note that the first method is fast and 
            efficient when the number of NaN in the array if relatively few. 
            It is less accurate to use the `base` interpolation when the data 
            is composed of many missing values. Alternatively, the scaled 
            method(the  second one) is proposed to be the alternative way more 
            efficient. Indeed, when ``mean`` argument is set, function replaces 
            the NaN values by the nonzeros in the raw array and then uses the 
            mean to fit the data. The result of fitting creates a smooth curve 
            where the index of each NaN in the raw array is replaced by its 
            corresponding values in the fit results. The same approach is used 
            for ``bff`` method. Conversely, rather than averaging the nonzeros 
            values, it uses the backward and forward strategy  to fill the 
            NaN before scaling. ``mean`` and ``bff`` are more efficient when 
            the data are composed of a lot of missing values. When the 
            interpolation `method` is set to ``pd``, function uses the pandas 
            interpolation but ended the interpolation with forward/backward 
            NaN filling since the interpolation with pandas does not deal with 
            all NaN at the begining or at the end of the array. 
            
        fill_value: array-like, str, optional, default='extrapolate', 
            If a ndarray (or float), this value will be used to fill in for 
            requested points outside of the data range. If not provided, 
            then the default is NaN. The array-like must broadcast properly 
            to the dimensions of the  non-interpolation axes. If two-element 
            in tuple, then the first element is used as a fill value
            for ``x_new < x[0]`` and the second element is used for 
            ``x_new > x[-1]``. Anything that is not a 2-element tuple 
            (e.g., list or ndarray,regardless of shape) is taken to be a 
            single array-like argument meant to be used for both bounds as 
            below, above = fill_value, fill_value. Using a two-element tuple 
            or ndarray requires ``bounds_error=False``.
            
        kws: dict 
            Additional keyword arguments from :func:`~interpolate1d`. 
        
        Returns 
        --------
            Array-like of :class:`watex.external.z.Z` objects 
            Array collection of new Z impedances objects with dead-band tensor 
            recovered. :class:`watex.externals.z..Z` are ndarray (nfreq, 2, 2). 
            2x2 matrices for components xx, xy and yx, yy. If tensor given, 
            it returns a collection of 2D tensor of each stations. 
    
        More 
        ------
        One main problem in collecting |NSAMT| data is the signal level in the 
        'attenuation band'. Compared to the |CSAMT| method (Wang and Tan, 2017; 
        Zonge and Hughes, 1991),the natural signals are not under our control 
        and suffer from frequency  ranges with little or no signal.  Most 
        notably, the |NSAMT| 'dead-band' between approximately 1 kHz and 4 
        kHz, but also a signal low in the vicinityof 1 Hz where the transition 
        to magnetospheric energy sources occurs (Goldak and Olson, 2015). 
        In this band, natural source signals are generally  absent. The EM 
        energy is dissipated and often cultural |EM| noise fills the gap 
        (Zonge, 2000). The response is extrapolated from results observed 
        top frequencies( For instance at 20, 40, 250, and 500 Hz).Experience
        indicates that the natural source signal level at 2000 Hz can be 
        expected to approach 10-6 γ/√Hz (Zheng, 2010; Zonge, 2000).
        
        See Also
        ----------
        scipy.interpolate.interp1d: 
            Interpolate 1D. 
            
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
        >>> zobjs_b =  pObjs.zrestore(buffer = buffer
                                            ) # with buffer 
        
        """
        self.inspect 

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
        ix_s , ix_end = ix_buf ; ix_end += 1 
        slice_= slice (ix_s,  ix_end) 
        s_cfreq = cfreq [slice_] # slice frequency within the buffer 
        
        # --> make a new Z objects 
        # make a new object 
        new_zObjs =np.zeros_like (zObjs, dtype =object )
        # loop to correct the Z impedance object values 
        for kk, ediObj in enumerate (self.ediObjs_):
            new_Z = EMz(z_array=np.zeros((len(s_cfreq), 2, 2),
                                           dtype='complex'),
                        z_err_array=np.zeros((len(s_cfreq), 2, 2)),
                        freq=s_cfreq)
            new_Z = self._tfuncZtransformer(
                ediObj, 
                new_Z, 
                tfunc= self._z_transform,
                slice_= slice_, 
                cfreq= cfreq, 
                ix_s= ix_s, 
                ix_end= ix_end, 
                method=method, 
                )
            new_zObjs[kk] = new_Z 
            
        if tensor: 
            tensor = str(tensor).lower() 
            tensor = 'res' if tensor =='true' else tensor 
            new_zObjs = get2dtensor(
                new_zObjs, tensor = tensor , 
                component = component or self.component, 
                )
            
        return new_zObjs 
    
    def _z_transform (
        self, 
        z , 
        rfq, 
        fq,  
        slice_= None, 
        method= 'pd',  
        **kws
        ): 
        """ Route to do the same task for real, imaginary and error.
        
        :param z: Impedance tensor z 
        :param rfq: reference frequency in Hz 
        :param fq: complete frequency in Hz
        :param slice_: frequency buffer indexes 
        :param method: tensor interpolation 
        :param kws: additional keywords arguments from :func:`~interpolate1d`. 
        """
        with np.errstate(all='ignore'):
            z = reshape(z) 
            z = fittensor(compfreq= fq, refreq =rfq, z = z  ) 
            z = interpolate1d(arr=z , method =method, **kws )
        return z [slice_] 
    
    def _tfuncZtransformer (
        self,  
        ediObj: EDIO , 
        new_Z: NDArray [DType[complex]], 
        tfunc: F, 
        cfreq: ArrayLike, 
        slice_: slice =None, 
        ix_s: int = 0 , 
        ix_end: int  = -1, 
        method: str ='pd', 
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
                    z_real = tfunc (z_real, 
                                    rfq=cfreq, 
                                    fq=zfreq, 
                                    slice_=slice_ , 
                                    method =method, 
                                    )
                    z_imag = reshape(ediObj.Z.z[nz_index, ii, jj].imag) 
                    
                    z_imag = tfunc (z_imag, 
                                    rfq=cfreq, 
                                    fq=zfreq, 
                                    slice_=slice_, 
                                    method =method, 
                                          )
                    z_err = reshape(ediObj.Z.z_err[nz_index, ii, jj]) 
                    z_err = tfunc (z_err, 
                                   rfq=cfreq, 
                                   fq=zfreq,
                                    slice_=slice_ , 
                                    method =method, 
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
        """ Assert buffer and find the nearest value if the 
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
        msg =("Buffer for frequency expects two values ('max', 'min'). ")
        
        if not hasattr (freq, '__array__'): 
            freq = np.array(freq , dtype = np.float64 )
            
        if buffer is None: 
            return np.array([freq.max(), freq.min()])
            
        if ( 
                isinstance (buffer, str) 
                or not hasattr(buffer , '__iter__')
                ): 
            raise ValueError(msg + f"But {type (buffer).__name__!r} is given")
            
        if len(buffer) < 2: 
            raise FrequencyError (msg + f"Got {len(buffer)}.") 
            
        if len(buffer ) > 2: 
            warnings.warn(msg + 
                          f"Got {len(buffer)}. The frequencies"
                          f" '({max(buffer), min(buffer)})' should be "
                          "used as ('max', 'min') instead.")
        
            buffer = np.array ( [ max(buffer), min(buffer)] )
        
        if not hasattr(buffer, '__array__'): 
            buffer = np.array ( buffer ) 
            #buffer.sort() # for consistency 
            
        if ( 
                buffer.min() < freq.min() 
                or buffer.max () > freq.max()
                ): 
            raise FrequencyError (
                f"Buffer frequency '{(buffer.max(), buffer.min())}'"
                " is out of the range. Expect frequencies be"
                f" within {(freq.min(),freq.max())}"
                )
        # The buffer does not need to fit the exact 
        # frequency value in the frequency data. 
        # Could Find the absolute difference with each value   
        # Get the index of the smallest absolute difference. 
        minf = freq  [ np.abs (freq - buffer.min()).argmin() ]  
        maxf = freq  [ np.abs (freq - buffer.max()).argmin() ]  
        buffer = np.array ( [ maxf , minf ] ) 
        
        # if min and max frequency are
        # identical , raise error 
        if len( set(buffer))  !=2: 
            raise FrequencyError ("Frequency buffer ('max', 'min') should"
                                  f" be distinct as possible. Got {buffer}")
        return buffer 

    def qc (
        self,  
        tol: float = .5 , 
        *, 
        return_freq: bool =False,
        return_ratio:bool=False, 
        to_log10: bool=True, 
        )->Tuple[float, ArrayLike]: 
        """ Check the quality control of the collected EDIs. 
        
        Analyse the data in the EDI collection and return the quality control value.
        It indicates how percentage are the data to be representative.
        
        Parameters
        ------------
        tol: float, 
            the tolerance parameter. The value indicates the rate from which the 
            data can be consider as meaningful. Preferably it should be less than
            1 and greater than 0. At this value. Default is ``.5`` means 50 % 
            
        return_freq: bool, Default =False 
            return the interpolated frequency if set to ``True``. 
        
        return_ratio: bool, default=False, 
           return only the ratio of the representation of the data. 
           
           .. versionadded:: 0.1.5
           
        to_log10:bool, default=False
           convert the interpolated frequency into a log10. 
           
        Returns 
        ---------
        Tuple (float , index )  or (float, array-like, shape (N, ))
            return the quality control value and interpolated frequency if  
            `return_freq`  is set to ``True`` otherwise return the index of useless 
            data. 
            
        Examples
        ---------
        >>> from watex.methods.em import Processing
        >>> pobj = Processing().fit('data/edis')
        >>> f = pobj.getfullfrequency ()
        >>> # len(f)
        >>> # ... 55 # 55 frequencies 
        >>> c,_ = pobj.qc ( tol = .4 ) # mean 60% to consider the data as
        >>> # representatives 
        >>> c  # the representative rate in the whole EDI- collection
        >>> # ... 0.95 # the whole data at all stations is safe to 95%. 
        >>> # now check the interpolated frequency 
        >>> c, freq_new  = pobj.qc ( tol=.6 , return_freq =True)
            
        """
        self.inspect 
        tol = assert_ratio(tol , bounds =(0, 1), exclude_value =0, 
                           name ='tolerance', as_percent =True )
        
        f=self.freqs_.copy() 
     
        try : 
            # take a sample of collected edi 
            # and make two day array
            # all frequency at all stations 
            ar = self.make2d ('freq') 
        except : 
            try : 
                ar = self.make2d( 'zxy')
            except: ar = self.make2d ('zyx')
        # compute the ratio of NaN in axis =0 
        
        nan_sum  =np.nansum(np.isnan(ar), axis =1) 
        rr= np.around ( nan_sum / ar.shape[1] , 2) 
        # compute the ratio ck
        # ck = 1. -    rr[np.nonzero(rr)[0]].sum() / (
        #     1 if len(np.nonzero(rr)[0])== 0 else len(np.nonzero(rr)[0])) 
        # ck =  (1. * len(rr) - len(rr[np.nonzero(rr)[0]]) )  / len(rr) 
        ck = 1 - nan_sum.sum() / (ar.shape [0] * ar.shape [1]) 
        
        index = reshape (np.argwhere (rr > tol))
        ar_new = np.delete (rr , index , axis = 0 ) 
        new_f = np.delete (f[:, None], index, axis =0 )
        # interpolate freq 
        if f[0] < f[-1]: 
            f =f[::-1] # reverse the array 
            ar_new = ar_new [::-1] # or np.flipud(np.isnan(ar)) 
        
        new_f = np.logspace(np.log10(new_f.min()) ,np.log10(new_f.max()),
                            len(new_f))[::-1]
        if not to_log10: 
            new_f = np.power(10 , new_f)
            
        return np.around (ck, 2) if return_ratio else (
            np.around (ck, 2), new_f   if return_freq else index )   


    @_zupdate(option = 'none')
    def getValidTensors(
        self, 
        tol:float = .5,  
        **kws 
        )-> NDArray[DType[complex]]: 
        """Select valid tensors from tolerance threshold and write EDI if 
        applicable.
        
        Function analyzes the data and keep the good ones. The goodness of the data 
        depends on the  `threshold` rate.  For instance 50% means to consider an 
        impedance tensor 'z'  valid if the quality control shows at least that score 
        at each frequency of all stations.  
        
        Parameters 
        ----------
        data: Path-like object or list of  :class:`pycsamt.core.edi.Edi`  
            collections of EDI-objects from `pycsamt`_ . `data` params is 
            passed to :meth:`~.Processing.fit` method. 
                
        tol : float, 
            tolerance parameter. The value indicates the rate from which the data 
            can be consider as a valid. The valid data selection should be soft when
            the tolerance parameter  is  close to '1' and hard otherwise. As the 
            `tol` value decreases, the selection  becomes severe. 
            Default is ``.5`` means 50 %  
            
        kws: dict , 
            Additional keywords arguments for EDI file exporting 
            
        Returns 
        --------
        Zc: :class:`watex.externals.z.Z` impedance tensor objects.
            
        Examples 
        --------
        >>> from watex.methods.em import Processing 
        >>> pObj = Processing ().fit('data/edis')
        >>> f= pObj.freqs_
        >>> len(f) 
        ... 55
        >>> zObjs_hard = pObj.getValidTensors (tol= 0.3 ) # None doesn't export EDI-file
        >>> len(zObjs_hard[0]._freq) # suppress 3 tensor data 
        ... 52 
        >>> zObjs_soft  = pObj.getValidTensors(p.ediObjs_, tol = 0.6 , option ='write')
        >>> len(zObjs_soft[0]._freq)  # suppress only two 
        ... 53
        
        """
        
        def delete_useless_tensor (z ,  index , axis = 0):
            """Remove meningless tensor data"""
            return np.delete (z, index , axis )
        def set_null(freq, objs): 
            """Set null in the case the component doesn't exist"""
            return np.zeros ((len(f), len(objs)), dtype = np.float32)
        
        self.inspect 
        # ediObjs = get_ediObjs(ediObjs) 
        _, no_ix = self.qc(tol=tol ) 
        f = self.freqs_.copy() 
    
        ff = np.delete (f[:, None], no_ix, 0)
        # interpolate frequency 
        new_f  = Processing.freqInterpolation (reshape (ff)) 
        
        # gather the 2D z objects
        # -XX--
        try : 
            zxx = delete_useless_tensor(
                self.make2d ( 'zxx'), no_ix) 
            zxx = interpolate2d(zxx)
            zxx_err = delete_useless_tensor(
                self.make2d ('zxx_err') , no_ix ) 
            zxx_err = interpolate2d (zxx_err )
        except :
            zxx = set_null(new_f, self.ediObjs_)
            zxx_err= zxx.copy() 
            
        # -XY--    
        try :
            zxy = delete_useless_tensor(
                self.make2d ( 'zxy'), no_ix )  
            zxy= interpolate2d( zxy)
            zxy_err = delete_useless_tensor( 
                self.make2d ( 'zxy_err') , no_ix )
            zxy_err = interpolate2d(zxy_err)
        except: 
            zxy = set_null(new_f, self.ediObjs_)
            zxy_err= zxy.copy() 
    
        # -YX--
        try:
        
            zyx = delete_useless_tensor( 
                self.make2d ('zyx') , no_ix ) 
            zyx = interpolate2d(zyx)
            zyx_err = delete_useless_tensor(
                self.make2d ( 'zyx_err') , no_ix ) 
            zyx_err = interpolate2d( zyx_err )
        except: 
            zyx = set_null(new_f, self.ediObjs_)
            zyx_err= zyx.copy() 
            
        # -YY--
        try:
            zyy = delete_useless_tensor(
                self.make2d ( 'zyy'), no_ix ) 
            zyy = interpolate2d(zyy)
            zyy_err = delete_useless_tensor(
                self.make2d ( 'zyy_err') , no_ix ) 
            zyy_err = interpolate2d(zyy_err)
            
        except :  
            zyy = set_null(new_f, self.ediObjs_)
            zyy_err= zyy.copy() 
            
      
        z_dict = {
            'zxx': zxx ,
            'zxy': zxy ,
            'zyx': zyx,
            'zyy': zyy, 
            'zxx_err': zxx_err ,
            'zxy_err': zxy_err ,
            'zyx_err': zyx_err, 
            'zyy_err': zyy_err
            } 
        
        return (self.ediObjs_, new_f , z_dict ), kws
     
    @staticmethod
    @_zupdate (option ='none')
    def interpolate_z (
        z_or_edis_obj_list,  / , 
        **kws 
        ): 
        """ Interpolate z and return new interpolated z objects 
        
        Interpolated frequencies is useful to have all frequencies into the 
        same scale. 
        
        .. versionadded:: v0.2.0 
        
        Parameters 
        ------------
        z_or_edis_obj_list: list of :class:`watex.edi.Edi` or \
            :class:`watex.externals.z.Z` 
            A collection of EDI- or Impedances tensors objects.
            
        kws: dict, 
           Additional keywords  to export EDI or rotate EDI/Z. 
           - ``"option"``: export EDI if set to ``write``. 
           - ``rotate`` : float, a rotate angle for Z if value is given. 
           
        
        Returns
        --------
        Z: list of :class:`watex.externals.z.Z`
           List interpolated impedance tensor Objects 
           or ``None`` if `option` is set to ``write``. 
 
        Examples 
        ----------
        >>> import watex as wx 
        >>> sedis = wx.fetch_data ('huayuan', samples = 12 , 
                                         return_data =True , key='raw')
        >>> p = wx.EMProcessing ().fit(sedis) 
        >>> ff = [ len(ediobj.Z._freq)  for ediobj in p.ediObjs_] 
        [53, 52, 53, 55, 54, 55, 56, 51, 51, 53, 55, 53]
        >>> Zcol = p.interpolate_z (sedis)
        >>> ffi = [ len(z.freq) for z in Zcol ]
        [56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56]
        >>> # visualize seven Z values at the first site component xy 
        >>> p.ediObjs_[0].Z.z[:, 0, 1][:7]
        array([ 4165.6 +2070.13j,  7072.81+6892.41j,  8725.84+5874.15j,
               14771.8 -2831.28j, 21243.7 -6802.36j,  6381.48+3411.65j,
                5927.85+5074.27j])
        >>> Zcol [0].z[:, 0, 1 ][:7]
        array([ 4165.6 +2070.13j,  4165.6 +2070.13j,  7072.81+6892.41j,
                8725.84+5874.15j, 14771.8 -2831.28j, 21243.7 -6802.36j,
                6381.48+3411.65j])

        """ 
        def set_null(freq, objs): 
            """Set null in the case the component doesn't exist"""
            return np.zeros ((len(f), len(objs)), dtype = np.float32)
        
        zobjs_init = EM.get_z_from (z_or_edis_obj_list)
        # gather the 2D z objects
        f = get_full_frequency(zobjs_init )

        zdict = {}
        for comp in ('xx', 'xy', 'yx', 'yy'): 
            try : 
                zx= get2dtensor(zobjs_init, component=comp, 
                                kind ='complex', 
                                )
                zx= interpolate2d(zx)
                zx_err = get2dtensor(
                    zobjs_init, 
                    tensor ='z_err', 
                    component=comp, kind='real'
                    )
                zx_err = interpolate2d (zx_err)
            except :
                # if component does not exist 
                zx = set_null(f, zobjs_init)
                zx_err= zx.copy() 
            
            zdict [f'z{comp}']=zx 
            zdict [f'z{comp}_err'] = zx_err 
            
        
        return (zobjs_init, f, zdict), kws 

    def drop_frequencies (
            self, 
            freqs= None, 
            tol =None, 
            interpolate: bool =False, 
            rotate= 0. , 
            export =False , 
            **kws 
            ): 
        """ Drop useless frequencies in the EDI data.
        
        Due to the terrain constraints, topographic and interferences noises 
        some frequencies are not meaningful to be kept in the data. The 
        function allows to explicitely remove the bad frequencies after 
        analyses and interpolated the remains. If bad frequencies are not known
        which is common in real world, the tolerance parameter `tol` can be 
        set to automatically detect with 50% smoothness in data selection. 
        
        .. versionadded:: v0.2.0 
        
        
        Parameters 
        -----------
        tol: float, 
            the tolerance parameter. The value indicates the rate from which the 
            data can be consider as meaningful. Preferably it should be less than
            1 and greater than 0. At this value. If ``None``, the list of 
            frequencies to drop must be provided. If the `tol` parameter is 
            set to ``auto``, the selection of useless frequencies is tolerate 
            to 50%. 

        freqs: list , Optional 
           The list of frequencies to remove in the: term:`EDI`objects. If 
           ``None``, the `tol` parameter must be provided, otherwise an error
           will raise. If the
           
           return the interpolated frequency if set to ``True``. 
        Interpolate: bool, default=False, 
           Interpolate the frequencies after bad frequencies removal. 
           
        rotate: float, default=0.  
            Rotate Z array by angle alpha in degrees.  All angles are 
            referenced to geographic North, positive in clockwise direction.
            (Mathematically negative!). In non-rotated state, X refs to North 
            and Y to East direction. 
            
            Note that if `rotate` is given, it is only used in `interpolation`
            i.e interpolation is set to ``True``. 

        export: bool , default =False, 
           Output new sanitized EDIs. 
        
        Returns 
        ---------
        Zcol:  or (float, array-like, shape (N, ))
            return the quality control value and interpolated frequency if  
            `return_freq`  is set to ``True`` otherwise return the index of useless 
            data. 
   
        Examples 
        ----------
        >>> import watex as wx 
        >>> sedis = wx.fetch_data ('huayuan', samples = 12 , 
                                   return_data =True , key='raw')
        >>> p = wx.EMProcessing ().fit(sedis) 
        >>> ff = [ len(ediobj.Z._freq)  for ediobj in p.ediObjs_] 
        >>> ff
        [53, 52, 53, 55, 54, 55, 56, 51, 51, 53, 55, 53]
        >>> p.ediObjs_[0].Z.z[:, 0, 1][:7]
        array([ 4165.6 +2070.13j,  7072.81+6892.41j,  8725.84+5874.15j,
               14771.8 -2831.28j, 21243.7 -6802.36j,  6381.48+3411.65j,
                5927.85+5074.27j])
        >>> Zcol = p.drop_frequencies (tol =.2 )
        >>> Zcol [0].z[:, 0, 1 ][:7]
        array([ 4165.6 +2070.13j,  7072.81+6892.41j,  8725.84+5874.15j,
               14771.8 -2831.28j, 21243.7 -6802.36j,  6381.48+3411.65j,
                5927.85+5074.27j])
        >>> [ len(z.freq) for z in Zcol ]
        [53, 52, 52, 53, 53, 53, 53, 50, 49, 53, 53, 52]
        >>> p.verbose =True 
        >>> Zcol = p.drop_frequencies (tol =.2 , interpolate= True )
        Frequencies:     1- 81920.0    2- 48.5294    3- 5.625  Hz have been dropped.
        >>> [ len(z.freq) for z in Zcol ] # all are interpolated to 53 frequencies
        [53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53]
        >>> Zcol = p.drop_frequencies (tol =.2 , interpolate= True , export =True )
        >>> # drop a specific frequencies 
        >>> # let visualize the 7 frequencies of our ediObjs 
        >>> p.freqs_ [:7]
        array([81920., 70000., 58800., 49500., 41600., 35000., 29400.])
        >>> # let try to drop 49500 and 29400 frequencies explicitly. 
        >>> Zcol = p.drop_frequencies (freqs = [49500 , 29400] )
        >>> # let check whether this frequencies still available in the data 
        >>> Zcol [5].freq[:7] 
        array([81920., 70000., 58800., 41600., 35000., 24700., 20800.])
        >>> # frequencies do not need to match exactly the value in frequeny 
        >>> # range. Here is an example 
        >>> Zcol = p.drop_frequencies (freqs = [49800 , 29700] )
        Frequencies:     1- 49500.0    2- 29400.0  Hz have been dropped.
        >>> # explicitly it drops the 49500 and 29400 Hz the closest. 

        """ 

        self.inspect 
        
        if tol is None and freqs is None: 
            raise EMError ("tolerance parameter or frequency values to delete"
                           " could not be None. Consider ``tol='auto'``to"
                           "  automatically delect the bad frequencies.")
            
        if str(tol).lower() == 'auto': 
            tol =.5 
   
        # make a copy of ediobjs  
        ediObjs = np.array( self.ediObjs_, dtype =object) 
        
        if tol is not None: 
            _ , index_freq = self.qc (tol = tol ) 
            freqs = self.freqs_ [ index_freq ]

        if freqs is not None: 
            if is_iterable(freqs, exclude_string=True): 
                # find the closest frequency that match the 
                # frequency in the complete freq 
                freqs = find_closest (self.freqs_, freqs )
            
        # randomly select frequency 
        freqs = random_selector (self.freqs_ , value = freqs) 
            
        # put back frequency from highest to lowest 
        # for consistency 
        freqs = np.sort (freqs )[::-1 ] 
       
        #if self.verbose: 
        listing_items_format(freqs ,begintext= "Frequencies" , 
                             endtext="Hz have been dropped.", 
                             inline =True , verbose =self.verbose 
                             )
        # use mask to set a new collection of Z 
        Zobj = []
        for kk , edio  in enumerate (ediObjs ): 
            mask  = np.isin ( edio.Z._freq,  freqs)
            # mask = np.ones ( len( edio.Z._freq), dtype = bool ) 
            # mask [ u_freqs] = False 
            z_new  = edio.Z._z [~mask , :, :]  
            # similar to np.delete (edio.Z._z , u_freqs, axis =0 )
            z_err_new  = edio.Z._z_err [~mask , :, :] 
            new_freq = edio.Z._freq[ ~mask ] 
            
            Z =EMz (
                z_array= z_new , z_err_array= z_err_new , freq = new_freq
                ) 
            Zobj.append(Z )
            
        if interpolate:
            Zobj = self.interpolate_z (Zobj, 
                                       rotate = rotate
                                       )
        
        if export: 
            self.exportedis(ediObjs= ediObjs, new_Z=Zobj, 
                            **kws)

        return Zobj
    
class ZC(EM): 
    """Impedance tensor multiple EDI correction class. 
    
    Applied filters in a collections of :term:`EDI` objects. 
    
    .. versionadded:: v0.2.0 
    
    Parameters 
    ------------
    
    data: Path-like object or list  of  :class:watex.edi.Edi` or \
        `pycsamt.core.edi.Edi` objects 
        Collections of EDI-objects 

    window_size : int
        the length of the window. Must be greater than 1 and preferably
        an odd integer number. Default is ``5``
        
    c : int, default=2 
        A window-width expansion factor that must be input to the filter 
        adaptation process to control the roll-off characteristics
        of the applied Hanning window. It is recommended to select `c` 
        between ``1``  and ``4`` [1]_. 

    References 
    -----------
    .. [1] Torres-Verdin and Bostick, 1992,  Principles of spatial surface 
           electric field filtering in magnetotellurics: electromagnetic array 
           profiling(EMAP), Geophysics, v57, p603-622.https://doi.org/10.1190/1.2400625
           
    Examples 
    ----------
    >>> import watex
    >>> from watex.methods import ZC 
    >>> edi_sample = watex.fetch_data ('edis', samples =17, return_data =True) 
    >>> zo = ZC ().fit(edi_sample) 
    >>> zo.ediObjs_[0].Z.resistivity[:, 0, 1][:10] # for xy components 
    array([ 427.43690401,  524.87391142,  732.85475419, 1554.3189371 ,
           3078.87621649, 1550.62680093,  482.64709443,  605.3153687 ,
            499.49191936,  468.88692879])
    >>> zss = zo.remove_static_shift(ss_fx =0.7 , ss_fy =0.85 )
    >>> zss[0].resistivity[:, 0, 1][:10] # corrected xy components 
    array([ 278.96395263,  319.11187959,  366.43170231,  672.24446295,
           1344.20120487,  691.49270688,  260.25625996,  360.02452498,
            305.97381587,  273.46251961])
    """
    def __init__(
        self, 
        window_size:int =5, 
        c: int =2, 
        **kws
        ): 
        super().__init__(**kws)
        
        self.filter = filter
        self.window_size=window_size 
        self.c=c 

    
    @_zupdate (option ='none')
    def remove_ss_emap (
            self, 
            fltr ='ama', 
            out= False, 
            **kws
            ): 
        """
        Filter Z to remove the static schift using the EMAP moving average 
        filters. 
        
        Three available filters: 
            
            - 'ama': Adaptative moving average 
            - 'tma': Trimming moving-average 
            - 'flma': Fixed-length dipole moving average 
        
        Could export new Edi if the keyword argument `export` is set 
        to ``True``
        
        Parameters 
        ------------
        fltr: str , default='ama'
           Type of filter to apply. Default is Adaptative moving-average of 
           Torres-verdin [1]_. Can be ['ama'|'tma'|'flma']
 
           The AMA filter estimates static-corrected apparent resistivities at 
           a single reference frequency by calculating a profile of average 
           impedances  along the length of the line. Sounding curves are 
           then shifted so that they intersect the averaged profile. 
           
        out: bool , default =False, 
           Output new filtered EDI. Otherwise return Z collections objects 
           of corrected Tensors. 
           
        Returns 
        ---------
        Z: list of :class:`watex.externals.z` objects or None 
           Return ``None`` by default (when export is set to ``False``) 
         
            
        References 
        -----------
        .. [1] Torres-Verdin and Bostick, 1992,  Principles of spatial surface 
               electric field filtering in magnetotellurics: electromagnetic array 
               profiling(EMAP), Geophysics, v57, p603-622.https://doi.org/10.1190/1.2400625
               
        See Also 
        ---------
        remove_static_shift: 
            Remove static shift using the spatial filter median and write 
            a new edifile. 
            
        Examples 
        ---------
        >>> import watex 
        >>> from watex.methods import ZC 
        >>> edi_sample = watex.fetch_data ('edis', samples =17 , return_data =True ) 
        >>> zo = ZC ().fit(edi_sample)
        >>> zo.ediObjs_[0].Z.z[:, 0, 1][:7]
        array([10002.46 +9747.34j , 11679.44 +8714.329j, 15896.45 +3186.737j,
               21763.01 -4539.405j, 28209.36 -8494.808j, 19538.68 -2400.844j,
                8908.448+5251.157j])
        >>> zc = zo.remove_ss_emap() 
        >>> zc[0].z[:, 0, 1] [:7]
        array([10.08886765+9.83154376j, 11.78033448+8.78960895j,
               16.03377371+3.21426607j, 21.95101282-4.57861929j,
               28.45305051-8.56819159j, 19.70746763-2.42158403j,
                8.98540488+5.29651986j])
        """
        self.inspect 
        
        p = Processing(out ='z ') 
        p.ediObjs_ = self.ediObjs_
        p.freqs_ = self.freqs_

        flr = str(fltr).lower().strip() 
        assert flr in {"ama", "tma", "flma"}, (
            f"Filter {fltr!r} is not available. Expect"
            " adaptative moving-average: 'ama',"
            " trimming moving-average: 'tma' or"
            " fixed-dipole length moving-average: 'flma'" 
            )
        # Gather the 2D into z objects
        zd = dict () 
        # correct all components if applicable 
        for comp in ('xx', 'xy', 'yx', 'yy'): 
            p.component = comp 
            try: 
                if flr=='tma': 
                    zc = p.ama () 
                elif flr=='flma': 
                    zc = p.flma () 
                else : 
                    zc = p.ama () 
    
                zc_err = self.make2d (f"z{comp}_err") 
                
            except : 
                # In the case some components 
                # are missing, set to null 
                zc = np.zeros (
                    (len(self.freqs_), len(self.ediObjs_)),
                    dtype = np.complex128)
                zc_err= zc.copy() 
            
            zd[f'z{comp}'] = zc 
            zd[f'z{comp}_err']=zc_err
            
        # manage edi -export 
        option =kws.pop('option', None )
        option = 'write' if out else None 
        # reset  option 
        kws.__setitem__('option', option ) 
        
    
        return (self.ediObjs_, self.freqs_ , zd ), kws
   
    @_zupdate (option ='none')
    def remove_static_shift (
        self, 
        ss_fx:float =None, 
        ss_fy:float= None, 
        out:bool =False , 
        rotate:float=0., 
        **kws
        ): 
        """
        Remove the static shift from correction factor from x and y. 
        
        The correction factors `ss_fx` and `ss_fy` are used for the 
        resistivity in the x and y  components for static shift removal.
        
        Factors can be determined by using the :meth:`get_ss_correction_factors`
        If ``None``, factors are found using the spatial median filter. 
        Assume the original observed tensor Z is built by a static shift 
        :math:`S` and an unperturbated "correct" :math:`Z_{0} :
        
        .. math::
  
             Z = S * Z_{0}

        therefore the correct Z will be: 
        
        .. math::
            
            Z_{0} = S^(-1) * Z

        Parameters 
        -----------
        
        ss_fx: float, Optional  
           static shift factor to be applied to x components
           (ie z[:, 0, :]).  This is assumed to be in resistivity scale. 
           If None should be automatically computed using  the 
           spatial median filter. 
           
        ss_fy: float, optional 
           static shift factor to be applied to y components 
           (ie z[:, 1, :]).  This is assumed to be in resistivity scale. If
           ``None`` , should be computed using the spatial filter median. 
        
        rotate: float, default=0.  
            Rotate Z array by angle alpha in degrees.  All angles are referenced
            to geographic North, positive in clockwise direction.
            (Mathematically negative!).
            In non-rotated state, X refs to North and Y to East direction.
            
        out: bool , default =False, 
           Output new filtered EDI. Otherwise return Z collections objects 
           of corrected Tensors. 
           
        ss: dict, 
           Additional kweyword arguments passed to 
           :meth:`get_ss_correction_factors`
           
        Returns
        --------
        
        ( static shift matrix,coorected_z) : np.ndarray ((2, 2)) or \
            watex.externals.z.Z
            If ``export`` is ``True`` export to new edis and returns 
            nothing. 
        
        Note 
        -----
        The factors are in resistivity scale, so the entries of  the matrix 
        "S" need to be given by their square-roots. Furhermore, `ss_fx` and 
        `ss_fy` must be supplied for a manual correction. If one argument of 
        the aforementionned parameters are missing, the auto factor computation 
        could be triggered and reset the given previous given factor. 
        
        Examples 
        ----------
        >>> import watex 
        >>> from watex.methods import ZC 
        >>> edi_sample = watex.fetch_data ('edis', samples =17 , return_data =True ) 
        >>> zo = ZC ().fit(edi_sample)
        >>> zo.ediObjs_[0].Z.z[:, 0, 1][:7]
        array([10002.46 +9747.34j , 11679.44 +8714.329j, 15896.45 +3186.737j,
               21763.01 -4539.405j, 28209.36 -8494.808j, 19538.68 -2400.844j,
                8908.448+5251.157j])
        >>> zc = zo.remove_static_shift () 
        >>> zc[0].z[:, 0, 1] [:7]
        array([ 8028.46578676+7823.69394148j,  9374.49231974+6994.54856416j,
               12759.27171475+2557.831671j  , 17468.06097719-3643.54946031j,
               22642.21817697-6818.35022516j, 15682.70444455-1927.03534064j,
                7150.35801004+4214.83658174j])
        """
        self.inspect 
        
        if (ss_fx is None or ss_fy is None ): 
            ss_fx , ss_fy = self.get_ss_correction_factors(
                **kws )
            
        ZObjs =[]
        new_ediObjs=[]

        for ii, ediObj in enumerate (self.ediObjs_): 
            z0 = EMz(z_array=ediObj.Z._z , 
                     z_err_array= ediObj.Z._z_err, 
                freq = ediObj.Z._freq 
                ) 
            # now rotate if possible 
            # and correct data. 
            if rotate: 
                z0.rotate (rotate ) 
            ss_cor, zcor = z0.remove_ss(
                reduce_res_factor_x=ss_fx,
                reduce_res_factor_y=ss_fy, 
                )
            # reset the ediObjs to the new Z 
            z0._z = zcor  
            ediObj.Z._z= zcor 
            
            ZObjs.append (z0 ) 
            new_ediObjs.append (ediObj )
            
        # export data to 
        # new edis 
        
        skws ={
               "option": 'write' if out  else  None, 
               } 
        #print(new_ediObjs)
       # print(new_ediObjs)
        return ( new_ediObjs, 
                self.freqs_, 
                ZObjs ), skws 
    
    
    @_zupdate (option ='none')
    def remove_distortion (
        self,
        distortion: NDArray, 
        /, 
        error:NDArray=None, 
        out =False, 
        **kws, 
        ):
        """
        Remove distortion D form an observed impedance tensor Z. 
        
        Allow to obtain the uperturbed "correct" :math:`Z_{0}` expressed as: 
            
        .. math:: 
            
           Z = D * Z_{0}


        Parameters 
        ----------
        distortion_tensor: np.ndarray(2, 2, dtype=real) 
           Real distortion tensor as a 2x2
   
        error: np.ndarray(2, 2, dtype=real), Optional 
          Propagation of errors/uncertainties included 
          
        out: bool , default =False, 
           Output new filtered EDI. Otherwise return Z collections objects 
           of corrected Tensors. 

        Returns 
        ----------
        d, new_z, new_z_err: NDArray ( 2 x2 , dtype =real )
          - input distortion tensor
          - impedance tensor with distorion removed
          -  impedance tensor error after distortion is removed 
          If ``export=True``, export to new EDI and return None. 
          
        Examples 
        ---------
        >>> import watex 
        >>> from watex.methods import ZC 
        >>> edi_sample = watex.fetch_data ('edis', samples =17 , return_data =True ) 
        >>> zo = ZC ().fit(edi_sample)
        >>> zo.ediObjs_[0].Z.z[:, 0, 1][:7]
        array([10002.46 +9747.34j , 11679.44 +8714.329j, 15896.45 +3186.737j,
               21763.01 -4539.405j, 28209.36 -8494.808j, 19538.68 -2400.844j,
                8908.448+5251.157j])
 		>>> distortion = np.array([[1.2, .5],[.35, 2.1]])
        >>> zc = zo.remove_distortion (distortion)
        >>> zc[0].z[:, 0, 1] [:7]
 		array([ 9724.52643923+9439.96503198j, 11159.25927505+8431.1101919j ,
                14785.52643923+3145.38324094j, 19864.708742  -4265.80166311j,
                25632.53518124-8304.88093817j, 17889.15373134-2484.60144989j,
                 8413.19671642+4925.46660981j])
        
        """
        self.inspect 
        
        distortion = np.array (distortion )
        if distortion.shape != (2, 2) :
            raise ZError ("Wrong shape for distortion. Expect shape="
                          "(2, 2, dtype=real) for xx, xy, yx and yy components.")
        ZObjs =[]
        new_ediObjs =[]
        for ii, ediObj in enumerate (self.ediObjs_): 
            z0 = EMz(z_array=ediObj.Z.z ,
                     z_err_array = ediObj.Z.z_err, 
                freq = ediObj.Z._freq 
                ) 
            d, new_z, new_z_err = z0.remove_distortion(
                distortion_tensor=distortion, distortion_err_tensor=error)
            z0._z = new_z
            z0._z_err = new_z_err
            ediObj.Z._z= new_z 
            #z0.compute_resistivity_phase(new_z, new_z_err ) 
            # for consistency 
            ZObjs.append (z0 ) 
            new_ediObjs.append (ediObj )
            
        # export data to 
        # new edis 
        
        skws ={
               "option": 'write' if out  else  None,  
               # "ediObjs": new_ediObjs
               } 

        return (new_ediObjs, 
                self.freqs_, 
                ZObjs ), skws 
    
    def get_ss_correction_factors(
        self, 
        /, 
        r=1000., 
        nfreq=21,
        skipfreq=5, 
        tol=.12
        ) -> Tuple[float, float]:
        """
        Compute the  static shift correction factor from a station 
        using a spatial median filter.  
        
        This will find those station within the given radius (meters).  
        Then it will find the median static shift for the x and y modes 
        and remove it, given that it is larger than the shift 
        tolerance away from 1.  
        
        Parameters 
        -------------
        r: float, default=1000. 
           radius to look for nearby stations, in meters.
 
        nfreq: int, default=21 
           number of frequencies calculate the median static shift.  
           This is assuming the first frequency is the highest frequency.  
           Cause usually highest frequencies are sampling a 1D earth.  
    
        skipfreq** : int, default=5 
           number of frequencies to skip from the highest frequency.  
           Sometimes the highest frequencies are not reliable due to noise 
           or low signal in the :term:`AMT` deadband.  This allows you to 
           skip those frequencies.
                           
    
        tol: float, default=0.12
           Tolerance on the median static shift correction.  If the data is 
           noisy the correction factor can be biased away from 1.  Therefore 
           the shift_tol is used to stop that bias.  If 
           ``1-tol < correction < 1+tol`` then the correction factor is set 
           to ``1``
  
        Returns
        -------
        (sx_x,  ss_y): (float, float)
            static shift corrections factor for x and y modes
    
        Examples 
        ---------
        >>> import watex 
        >>> from watex.methods import ZC 
        >>> edi_sample = watex.fetch_data ('edis', samples =17 ,
                                           return_data =True ) 
        >>> zo = ZC ().fit(edi_sample).get_ss_correction_factors () 
        Out[16]: (1.5522030221266643, 0.742682340427651)
        
        """
        self.inspect 
        # convert meters to decimal degrees so 
        # we don't have to deal with zone
        # changes
        meter_to_deg_factor = 8.994423457456377e-06
        dm_deg = r * meter_to_deg_factor

        edi_obj_init= self.ediObjs_[0] 
        edi_obj_init.Z.compute_resistivity_phase()

        interp_freq = self.freqs_[skipfreq:nfreq + skipfreq]

        # Find stations near by and store them in a list
        emap_obj = []
        # for kk, edi in enumerate(edi_list):
        for kk, edi in enumerate (self.ediObjs_): 
            edi_obj = edi
            delta_d = np.sqrt((edi_obj_init.lat - edi_obj.lat) ** 2 +
                          (edi_obj_init.lon - edi_obj.lon) ** 2)
            if delta_d <= dm_deg:
                edi_obj.delta_d = float(delta_d) / meter_to_deg_factor
                emap_obj.append(edi_obj)

        if len(emap_obj) == 0:
            if self.verbose: 
                print('No stations found within given '
                      'radius {0:.2f} m'.format(r))
            return 1.0, 1.0

        # extract the resistivity values 
        # from the near by stations
        res_array = np.zeros((len(emap_obj), nfreq, 2, 2))
        if self.verbose: 
            print('These stations are within the given'
                  ' {0} m radius:'.format(r))
        for kk, emap_obj_kk in enumerate(emap_obj):
            if self.verbose: 
                
                print('\t{0} --> {1:.1f} m'.format(
                    emap_obj_kk.station, emap_obj_kk.delta_d))
                
            interp_idx = np.where((interp_freq >= emap_obj_kk.Z.freq.min()) &
                              (interp_freq <= emap_obj_kk.Z.freq.max()))

            interp_freq_kk = interp_freq[interp_idx]
            Z_interp = emap_obj_kk.interpolateZ(interp_freq_kk)
            Z_interp.compute_resistivity_phase()
            res_array[
                kk,
                interp_idx,
                :,
                :] = Z_interp.resistivity[
                0:len(interp_freq_kk),
                :,
                :]

        # compute the static shift of x-components
        ss_x = edi_obj_init.Z.resistivity[
            skipfreq:nfreq + skipfreq, 0, 1] / np.median(
                res_array[:, :, 0, 1], axis=0)
        ss_x = np.median(ss_x)

        # check to see if the estimated static
        # shift is within given tolerance
        if  ( 
                1 - tol < ss_x 
                and ss_x < 1 + tol
                ):
            ss_x = 1.0

        # compute the static shift of y-components
        ss_y = edi_obj_init.Z.resistivity[
            skipfreq:nfreq + skipfreq, 1, 0] / np.median(
                res_array[:, :, 1, 0], axis=0)
        ss_y = np.median(ss_y)

        # check to see if the estimated static 
        # shift is within given tolerance
        if ( 
                1 - tol < ss_y 
                and ss_y < 1 + tol
                ):
            ss_y = 1.0

        return ss_x, ss_y
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
