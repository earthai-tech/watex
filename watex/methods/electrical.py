# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Created date: Thu Apr 14 17:45:55 2022

from __future__ import annotations 

import os 
import re 
import copy
import warnings
import numpy as np 
import pandas as pd

from .._docstring import refglossary
from .._watexlog import watexlog 
from ..decorators import refAppender 
from ..tools.funcutils import (
    repr_callable_obj,
    smart_format,
    smart_strobj_recognition , 
    is_installing,
    make_ids, 
    show_stats,
    )
from ..tools.coreutils import (
    _assert_station_positions,
    defineConductiveZone, 
    fill_coordinates, 
    erpSelector, 
    vesSelector,
    parseDCArgs ,
) 
from ..tools.exmath import (
    shape, 
    type_, 
    power, 
    magnitude, 
    sfi,
    ohmicArea, 
    invertVES,
    )
from ..typing import  ( 
    List, 
    Optional, 
    NDArray, 
    Series , 
    DataFrame,
    F

    )
from ..property import( 
    ElectricalMethods
    ) 
from ..exceptions import (
    NotFittedError, 
    VESError, 
    ERPError,
    StationError, 
    )
_logger = watexlog().get_watex_logger(__name__ )

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

__all__=['DCProfiling', 'DCSounding',
         'ResistivityProfiling', 'VerticalSounding'
         ]

class DCProfiling(ElectricalMethods)  : 
    """ A collection of DC-resistivity profiling classes. 
    
    It reads and compute electrical parameters. Each line compose a specific
    object and gather all the attributes of :class:`~.ResistivityProfiling` for
    easy use. For instance, the expeced drilling location point  and its 
    resistivity value for two survey lines ( line1 and line2) can be fetched 
    as:: 
        
        >>> <object>.line1.sves_ ; <object>.line1.sves_resistivity_ 
        >>> <object>.line2.sves_ ; <object>.line2.sves_resistivity_ 
    
    
    Arguments 
    ----------
    
    **stations**: list or str (path-like object )
        list of station name where the drilling is expected to be located. It 
        strongly linked to the name of used to specify the center position of 
        each dipole when the survey data is collected. Each survey can have its 
        own way for numbering the positions, howewer if the station is given 
        it should be one ( presumed to be the suitable point for drilling) in 
        the survey lines. Commonly it is called the `sves` which mean at this 
        point, the DC-sounding will be operated. Be sure to provide the correct
        station to compute the electrical parameters. 
        
        It is recommed to provide the positioning of the station expected to 
        hold the drillings. However if `stations` is None, the auto-way for 
        computing electrical features should be triggered. User can also 
        provide the list of stations by hand. In that case, each station should
        numbered from 1 not 0. For instance: 
            
            *  in a survey line of 20 positions. We considered the station 13 
                as the best point to locate the drilling. Therefore the name  
                of the station should be 'S13'. In other survey line (line2)
                the second point of my survey is considered the suitable one 
                to locate my drilling. Considering the two survey lines, the 
                list of stations sould be '['S13', 'S2']
                
            * `stations` can also be arrange in a single to be parsed which 
                refer to the string arguments. 
            
    **dipole**: float 
        The dipole length used during the exploration area. If `dipole` value 
        is set as keyword argument,i.e. the station name is overwritten and 
        is henceforth named according to the  value of the dipole. For instance
        for `dipole` equals to ``10m``, the first station should be ``S00``, 
        the second ``S10`` , the third ``S20`` and so on. However, it is 
        recommend to name the station using counting numbers rather than using 
        the dipole  position.
        
    **auto**: bool 
        Auto dectect the best conductive zone. If ``True``, the station 
        position should be  the  `station` of the lower resistivity value 
        in |ERP|. 
    
    **read_sheets**: bool, 
        Read the data in sheets. Here its assumes the data  of each survey 
        lines are arrange in a single excell worksheets. Note that if 
        `read_sheets` is set to ``True`` and the file is not in excell format, 
        a TypError will raise. 
        
    **kws**: dict 
         Additional |ERP| keywords arguments  
         
    Examples
    ---------
    (1) -> Get DC -resistivity profiling from the individual Resistivity object 
    
    >>> from watex.methods import ResistivityProfiling 
    >>> from watex.methods import DCProfiling  
    >>> robj1= ResistivityProfiling(auto=True) # auto detection 
    >>> robj1.utm_zone = '50N'
    >>> robj1.fit('data/erp/testsafedata.xlsx') 
    >>> robj1.sves_
    ... 'S036'
    >>> robj2= ResistivityProfiling(auto=True, utm_zone='40S') 
    >>> robj2.fit('data/erp/l11_gbalo.xlsx') 
    >>> robj2.sves_ 
    ... 'S006'
    >>> # read the both objects 
    >>> dcobjs = DCProfiling()
    >>> dcobjs.fit([robj1, robj2]) 
    >>> dcobjs.sves_ 
    ... array(['S036', 'S006'], dtype=object)
    >>> dcobjs.line1.sves_ # => robj1.sves_
    >>> dcobjs.line2.sves_ # => robj2.sves_ 
    
    (2) -> Read from a collection of excell data 
    
    >>> datapath = r'data/erp'
    >>> dcobjs.read_sheets=True 
    >>> dcobjs.fit(datapath) 
    >>> dcobjs.nlines_  # getting the number of survey lines 
    ... 9
    >>> dcobjs.sves_ # stations of the best conductive zone 
    ... array(['S017', 'S006', 'S000', 'S036', 'S036', 'S036', 'S036', 'S036',
           'S001'], dtype='<U33')
    >>> dcobjs.sves_resistivities_ # the lower conductive resistivities 
    ... array([  80,   50, 1101,  500,  500,  500,  500,  500,   93], dtype=int64)
    >>> dcobjs.powers_ 
    ... array([ 50,  60,  30,  60,  60, 180, 180, 180,  40])
    >>> dcobjs.sves_ # stations of the best conductive zone 
    ... array(['S017', 'S006', 'S000', 'S036', 'S036', 'S036', 'S036', 'S036',
           'S001'], dtype='<U33')
    
    (3) -> Read data and all sheets, assumes all data are arranged in a sheets
    
    >>> dcobjs.read_sheets=True
    >>> dcobjs.fit(datapath) 
    >>> dcobjs.nlines_ # here it assumes all the data are in single worksheets.
    ... 4 
    >>> dcobjs.line4.conductive_zone_ # conductive zone of the line 4 
    ... array([1460, 1450,  950,  500, 1300, 1630, 1400], dtype=int64)
    >>> dcobjs.sfis_
    >>> array([1.05085691, 0.07639077, 0.03592814, 0.07639077, 0.07639077,
           0.07639077, 0.07639077, 0.07639077, 1.08655919])
    >>> dcobjs.line3.sfi_ # => robj1.sfi_
    ... array([0.03592814]) # for line 3 
    
    """
    
    def __init__(self, 
                 stations: Optional[List[str]]= None,
                 dipole: float = 10.,
                 auto: bool = False, 
                 read_sheets:bool=False, 
                 **kws
                 ):
        super().__init__(**kws)
        
        self._logging=watexlog.get_watex_logger(self.__class__.__name__)
        
        self.stations=stations 
        self.dipole=dipole 
        self.auto=auto 
        self.read_sheets=read_sheets 
        
        for k in list (kws.keys()): 
            setattr (self, k, kws[k])
        
    def fit(self, 
            data : List[str] | List [DataFrame],
            **kws)-> object : 
        """ Read and fit the collections of data  
        
        Parameters 
        ----------
        **data**: List of path-like obj, or :class:`~.ResistivityProfiling`
            object. Data containing the collection of DC-resistivity values of 
            of multiple survey areas. 
                
        **kws**: str, 
            Additional keyword from :func:watex.tools.coreutils.parseStations`.
            It refers to the `station_delimiter` parameters. If the attribute 
            :attr:`~.ResistivityProfilings.stations` is given as a path-like 
            object. If the stations are disposed in the same line, it is 
            convenient to provide the delimiter to parse the stations. 
            
        Returns 
        -------
            object instanciated from :class:`~.ResistivityProfiling`.
            
        Notes
        ------
        The stations should numbered from 1 not 0 and might fit the number of 
        the survey line. Each survey line expect to hold one positionning 
        drilling. 
        
        
        """
        self._logging.info (f" {self.__class__.__name__!r} collects the "
                            "resistivity objects ")
        
        #-> Initialize collection objects 
        # - collected the unreadable data; readable data  
        self.isnotvalid_= list() ; self.data_= list() 
        
        # check whether object is readable as ERP objs
        #  -> if not assume a path or file is given 
        
        if not _readfromdcObjs (self, data):
            _readfrompath (self, data,**kws)
            
        # makeids objects 
        self.ids_ = np.array(make_ids (self.survey_names_,'line',None, True)) 
        
        # set each line as an object with attributes
        # can be retrieved like self.line1_.sves_ 
        self.lines_ = np.empty_like (self.ids_, dtype =object )
        for kk, (id_ , line) in enumerate (zip( self.ids_, self.data_)) : 
            obj = type (f"{line}", (ElectricalMethods,), line.__dict__ )
            self.__setattr__(f"{id_}", obj)
            self.lines_[kk]= obj  # set lines objects 
            
        # -> lines numbers 
        self.nlines_ = self.lines_.size 
        
        if self.verbose > 3: 
            print("Each line is an object class inherits from all attributes" 
                  " of DC-resistivity profiling object. For instance the"
                  "the right drilling point of the first line  can be fetched"
                  "  as: <self.line1.sves_> ")
            
        # can also retrieve an attributes in other ways 
        # make usefull attributess
        if self.verbose > 7: 
            print("Populate the other  attributes and data can be"
                  " fetched as array of N-number of survey lines.  ")
            
        # set expected the drilling point positions and resistivity values  
        self.sves_ = _geterpattr ('sves_', self.data_)
        self.sves_resistivities_ =  _geterpattr (
            'sves_resistivity_', self.data_ ).astype(float)
        
        # set the expected drilling points coordinates  at each line 
        for name in  ('lat', 'lon', 'east','north'): 
            setattr (self, f"sves_{name}s_", _geterpattr (
                f"sves_{name}_", self.data_).astype(float))

        # set the predictor parameters attributes 
        for name in  ('power', 'magnitude', 'type','sfi'): 
            setattr (self, f"{name}s_", _geterpattr (f"{name}_", self.data_) 
                     if name =='type' else  _geterpattr (f"{name}_", self.data_
                                                         ).astype(float) 
                     )

        return self 
    
    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        return repr_callable_obj  (self, 'line')
       
    
    def __getattr__(self, name):
        if name.endswith ('_'): 
            if name not in self.__dict__.keys(): 
                if name in (
                        'data_', 'resistivities_', 'sves_lons_', 'sves_lats_',
                        'sves_easts_', 'sves_norths_', 'sves_resistivities_',
                        'powers_', 'magnitudes_','shapes_','types_','sfis_'
                        'lines_', 'nlines_', 'ids_', 'survey_names_', 
                        'isnotvalid_'): 
                    raise NotFittedError (
                        f'Fit the {self.__class__.__name__!r} object first'
                        )
                
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        appender  = "" if rv is None else f'. Do you mean {rv!r}'
        
        raise AttributeError (
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{appender}{"" if rv is None else "?"}'
            )        

@refAppender(refglossary.__doc__)
class DCSounding(ElectricalMethods) : 
    """ Direct-Current Electrical Sounding 
    
    A collection of |VES| class and computed predictors paramaters accordingly. 
    
    The VES is carried out to speculate about the existence of a fracture zone
    and the layer thicknesses. Commonly, it comes as supplement methods to |ERP| 
    after selecting the best conductive zone when survey is made on 
    one-dimensional. Data from each DC-sounding site can be retrieved using::
        
        >>> <object>.site<number>.<:attr:`~.VerticalSounding.<attr>_`
        
    For instance to fetch the DC-sounding data position and the resistivity 
    in depth of the fractured zone for the first site, we use:: 
        
        >>> <object>.site1.fractured_zone_
        >>> <object>.site1.fractured_zone_resistivity_
    
    Arguments 
    -----------
    
    **fromS**: float , list of float
        The collection of the depth in meters from which one expects to find a 
        fracture zone outside of pollutions. Indeed, the `fromS` parameter is 
        used to speculate about the expected groundwater in the fractured rocks 
        under the average level of water inrush in a specific area. For 
        instance in `Bagoue region`_ , the average depth of water inrush 
        is around ``45m``.So the `fromS` can be specified via the water inrush 
        average value. 
        
    **rho0**: float 
        Value of the starting resistivity model. If ``None``, `rho0` should be
        the half minumm value of the apparent resistivity  collected. Units is
        in Ω.m not log10(Ω.m)
        
    **h0**: float 
        Thickness  in meter of the first layers in meters.If ``None``, it 
        should be the minimum thickess as possible ``1.m`` . 
    
    **strategy**: str 
        Type of inversion scheme. The defaut is Hybrid Monte Carlo (HMC) known
        as ``HMCMC``. Another scheme is Bayesian neural network approach (``BNN``). 
        
    **vesorder**: int 
        The index to retrieve the resistivity data of a specific sounding point.
        Sometimes the sounding data are composed of the different sounding 
        values collected in the same survey area into different |ERP| line.
        For instance:
            
            +------+------+----+----+----+----+----+
            | AB/2 | MN/2 |SE1 | SE2| SE3| ...|SEn |
            +------+------+----+----+----+----+----+
            
        Where `SE` are the electrical sounding data values  and `n` is the 
        number of the sounding points selected. `SE1`, `SE2` and `SE3` are 
        three  points selected for |VES| i.e. 3 sounding points carried out 
        either in the same |ERP| or somewhere else. These sounding data are 
        the resistivity data with a  specific numbers. Commonly the number 
        are randomly chosen. It does not refer to the expected best fracture
        zone selected after the prior-interpretation. After transformation 
        via the function :func:`~watex.tools.coreutils.vesSelector`, the header  
        of the data should hold the `resistivity`. For instance, refering to 
        the table above, the data should be:
            
            +----+----+-------------+-------------+-------------+-----+
            | AB | MN |resistivity  | resistivity | resistivity | ... |
            +----+----+-------------+-------------+-------------+-----+
        
        Therefore, the `vesorder` is used to select the specific resistivity
        values i.e. select the corresponding sounding number  of the |VES| 
        expecting to locate the drilling operations or for computation. For 
        esample, `vesorder`=1 should figure out: 
            
            +------+------+----+--------+----+----+------------+
            | AB/2 | MN/2 |SE2 |  -->   | AB | MN |resistivity |
            +------+------+----+--------+----+----+------------+
        
        If `vesorder` is ``None`` and the number of sounding curves are more 
        than one, by default the first sounding curve is selected ie 
        `rhoaIndex` equals to ``0``
        
    **typeofop**: str 
        Type of operation to apply  to the resistivity 
        values `rhoa` of the duplicated spacing points `AB`. The *default* 
        operation is ``mean``. Sometimes at the potential electrodes ( `MN` ),the 
        measurement of `AB` are collected twice after modifying the distance
        of `MN` a bit. At this point, two or many resistivity values are 
        targetted to the same distance `AB`  (`AB` still remains unchangeable 
        while while `MN` is changed). So the operation consists whether to the 
        average ( ``mean`` ) resistiviy values or to take the ``median`` values
        or to ``leaveOneOut`` (i.e. keep one value of resistivity among the 
        different values collected at the same point `AB` ) at the same spacing 
        `AB`. Note that for the ``LeaveOneOut``, the selected 
        resistivity value is randomly chosen.
        
    **objective**: str 
        Type operation to output. By default, the function outputs the value
        of pseudo-area in :math:`$ohm.m^2$`. However, for plotting purpose by
        setting the argument to ``view``, its gives an alternatively outputs of
        X and Y, recomputed and projected as weel as the X and Y values of the
        expected fractured zone. Where X is the AB dipole spacing when imaging 
        to the depth and Y is the apparent resistivity computed.
        
    **kws**: dict 
        Additionnal keywords arguments from |VES| data operations. 
        See :func:`watex.tools.exmath.vesDataOperator` for futher details.
        
    Examples 
    --------
    (1) -> read a single DC Electrical Sounding file 
    
    >>> from watex.methods.electrical import DCSounding
    >>> dsobj = DCSounding ()  
    >>> dsobj.fromS = 30. # start detecting the fracture zone from 30m depth.
    >>> dsobj.fit('data/ves/ves_gbalo.xlsx')
    >>> dsobj.ohmic_areas_
    ...  array([523.25458506])
    >>> dsobj.site1.fractured_zone_ # show the positions of the fracture zone 
    ... array([ 28.,  32.,  36.,  40.,  45.,  50.,  55.,  60.,  70.,  80.,  90.,
           100.])
    >>> dsobj.line1.fractured_zone_resistivity_
    ... array([ 68.74273843,  71.57116555,  74.39959268,  77.2280198 ,
                80.76355371,  84.29908761,  87.83462152,  91.37015543,
                98.44122324, 105.51229105, 112.58335886, 119.65442667])
    
    (2) -> read multiple sounding files 
    
    >>> dsobj.fit('data/ves')
    >>> dsobj.ohmic_areas_  
    ... array([ 523.25458506,  523.25458506, 1207.41759558]) 
    >>> dsobj.nareas_ 
    ... array([2., 2., 3.]) 
    >>> dsobj.survey_names_
    ... ['ves_gbalo', 'ves_gbalo', 'ves_gbalo_unique']
    >>> dsobj.nsites_ 
    ... 3 
    >>> dsobj.site1.ohmic_area_
    ... 523.2545850558677  # => dsobj.ohmic_areas_ -> line 1:'ves_gbalo'
    
    
    """
           
    def __init__(self,
                 fromS:float=45.,
                 rho0:float=None, 
                 h0 :float=1., 
                 read_sheets:bool=False, 
                 strategy:str='HMCMC',
                 vesorder:int=None, 
                 typeofop:str='mean',
                 objective: Optional[str] = 'coverall',
                 **kws) -> None : 
        super().__init__(**kws) 
        
        self._logging = watexlog.get_watex_logger(self.__class__.__name__)
        self.fromS=fromS 
        self.vesorder=vesorder 
        self.typeofop=typeofop
        self.objective=objective 
        self.rho0=rho0, 
        self.h0=h0
        self.strategy = strategy, 
        self.read_sheets= read_sheets
        
        for key in list( kws.keys()): 
            setattr(self, key, kws[key])
            
    def fit(self, data : List[str] | List [DataFrame], **kws): 
        """ Fit the DC- electrical sounding 
        
        Fit the sounding |VES| curves and computed the ohmic-area and set  
        all the features for demarcating fractured zone from the selected 
        anomaly. 
        
        Parameters 
        -----------
        data:  list of path-like object, or DataFrames
            The string argument is a path-like object. It must be a valid file
            wich encompasses the collected data on the field. It shoud be
            composed of spacing values `AB` and  the apparent resistivity 
            values `rhoa`. By convention `AB` is half-space data i.e `AB/2`. 
            So, if `data` is given, params `AB` and `rhoa` should be kept to
            ``None``. If `AB` and `rhoa` is expected to be inputted, user must
            set the `data`  to ``None`` values for API purpose. If not an error
            will raise. Or the recommended way is to use the `vesSelector` tool
            in :func:`watex.tools.vesSelector` to buid the |VES| data before 
            feeding it to the algorithm. See the example below.
            
        kws: dict 
            additional keywords arguments, specific to the readable files. 
            Refer to :method:`watex.property.Config.parsers` . Use the key()
            to get all the readables format. 
            
        Returns 
        -------
         object: A collection of |VES| objects 
         
        """
        self._logging.info (f" {self.__class__.__name__!r} collects the "
                            "resistivity objects ")
        
        #-> Initialize collection objects 
        # - collected the unreadable data ; readable data  
        self.isnotvalid_= list() ; self.data_= list() 
        
        # check whether object is readable as ERP objs
        #  -> if not assume a path or file is given 
        if not _readfromdcObjs (self, data, VerticalSounding, VESError):
            _readfrompath (self, data, VerticalSounding,  **kws)
            
        self.ids_ = np.array(make_ids (self.survey_names_, 'site', None, True)) 
        
        # set each line as an object with attributes
        # can be retrieved like self.site1_.fractured_zone_ 
        self.sites_ = np.empty_like (self.ids_, dtype =object )
        for kk, (id_ , site) in enumerate (zip( self.ids_, self.data_)) : 
            obj = type (f"{site}", (ElectricalMethods,), site.__dict__ )
            self.__setattr__(f"{id_}", obj)
            self.sites_[kk]= obj  # set site objects 
            
        # -> lines numbers 
        self.nsites_ = self.sites_.size 
        
        if self.verbose > 3: 
            print("Each line is an object class inherits from all attributes" 
                  " of DC-electrical sounding object. For instance the number"
                  " of ohmic areas computed of the first line can be fetched"
                  "  as: <self.sit1.ohmic_area_> ")
            
        # can also retrieve an attributes in other ways 
        # make usefull attributess
        if self.verbose > 7: 
            print("Populate the other  attributes and data can be"
                  " fetched as array of N-number of survey lines.  ")
           
        # set expected the drilling point positions and resistivity values  
        self.ohmic_areas_ = _geterpattr ('ohmic_area_', self.data_).astype(float)
        self.nareas_ =  _geterpattr (
            'nareas_', self.data_ ).astype(float)
        
        # All other attributes can be retrieved. For instance line1
        # self.site1.XY_, self.site1.XYarea_  or 
        # self.site1.AB_ ,  self.site.XY_

        if self.verbose > 7: 
            print("Parameters numbers are well computed ")
            
        return self 

    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        return repr_callable_obj  (self, 'line')
       
    
    def __getattr__(self, name):
        if name.endswith ('_'): 
            if name not in self.__dict__.keys(): 
                if name in ('data_','n_areas_', 'ohmic_areas_', 'isnotvalid_'
                            'nlines_', 'survey_names_'): 
                    raise NotFittedError (
                        f'Fit the {self.__class__.__name__!r} object first'
                        )
                
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        appender  = "" if rv is None else f'. Do you mean {rv!r}'
        
        raise AttributeError (
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{appender}{"" if rv is None else "?"}'
            )        
        
@refAppender(refglossary.__doc__)
class ResistivityProfiling(ElectricalMethods): 
    """ Class deals with the Electrical Resistivity Profiling (ERP).
    
    The electrical resistivity profiling is one of the cheap geophysical 
    subsurface imaging method. It is most preferred to find groundwater during
    the campaigns of drinking water supply, especially in developing countries.
    Commonly, it is used in combinaision with the  the vertical electrical
    sounding |VES| to speculated about the layer thickesses and the existence of
    the fracture zone. 
    
    Arguments 
    ----------
    
    **station**: str 
        Station name where the drilling is  expected to be located. The 
        station should numbered from 1 not 0. So if ``S00` is given, the 
        station name should be set to ``S01``. Moreover, if `dipole` value 
        is set as keyword argument,i.e. the station is  named according
        to the  value of the dipole. For instance for `dipole` equals to 
        ``10m``, the first station should be ``S00``, the second ``S10`` , 
        the third ``S20`` and so on. However, it is recommend to name the
        station using counting numbers rather than using the dipole 
        position.
            
    **dipole**: float 
        The dipole length used during the exploration area. 
        
    **auto**: bool 
        Auto dectect the best conductive zone. If ``True``, the station 
        position should be  the  `station` of the lower resistivity value 
        in |ERP|. 
    
    **kws**: dict 
         Additional |ERP| keywords arguments  
         
    Examples
    --------
    >>> from watex.methods.electrical import ResistivityProfiling 
    >>> rObj = ResistivityProfiling(AB= 200, MN= 20,station ='S7') 
    >>> rObj.fit('data/erp/testunsafedata.csv')
    >>> rObj.sfi_
    ... array([0.03592814])
    >>> rObj.power_, robj.position_zone_
    ... 90, array([ 0, 30, 60, 90])
    >>> rObj.magnitude_, robj
    >>> rObj.magnitude_, rObj.conductive_zone_
    ... 268, array([1101, 1147, 1345, 1369], dtype=int64)
    >>> robj.dipole
    ... 30
    
    """
    
    def __init__ (self, 
                  station: str | None = None,
                  dipole: float = 10.,
                  auto: bool = False, 
                  **kws): 
        super().__init__(**kws) 
        
        self._logging = watexlog.get_watex_logger(self.__class__.__name__)
        self.dipole=dipole
        self.station=station
        self.auto=auto 
        self.table_= None 
        
        for key in list( kws.keys()): 
            setattr(self, key, kws[key])
            

            
    def fit(self, data : str | NDArray | Series | DataFrame ,
             columns: str | List [str] = None, 
             **kws
            ) -> object: 
        """ Fitting the :class:`~.ResistivityProfiling` 
        and populate the class attributes.
        
        Parameters 
        ---------- 
        **data**: Path-like obj, Array, Series, Dataframe. 
            Data containing the the collected resistivity values in 
            survey area. 
                
        **columns**: list, 
            Only necessary if the `data` is given as an array. No need to 
            to explicitly defin when `data` is a dataframe or a Pathlike
            object.
            
        **kws**: dict, 
            Additional keyword arguments; e.g. to force the station to 
            match at least the best minimal resistivity value in the 
            whole data collected in the survey area. 
                
        Returns 
        -------
            object instanciated for chaining methods. 
            
        Notes
        ------
        The station should numbered from 1 not 0. So if ``S00`  is given, the 
        station name should be set to ``S01``. Moreover, if `dipole` value is
        set as keyword argument, i.e. the station is  named according to the 
        value of the dipole. For instance for `dipole` equals to ``10m``, 
        the first station should be ``S00``, the second ``S10``, the third 
        ``S20`` and so on. However, it is recommend to name the station using 
        counting numbers rather than using the dipole position.
        
        """
        
        self._logging.info(f'`Fit` method from {self.__class__.__name__!r}'
                           ' is triggered ')
        if isinstance(data, str): 
            if not os.path.isfile (data): 
                raise TypeError ( f'{data!r} object should be a file,'
                                 f' got {type(data).__name__!r}'
                                 )
        
        data = erpSelector(data, columns) 
        self.data_ = copy.deepcopy(data) 
        
        self.data_, self.utm_zone = fill_coordinates(
            self.data_, utm_zone= self.utm_zone, datum = self.datum ,
            epsg= self.epsg,verbose = self.verbose) 
        self.resistivity_ = self.data_.resistivity 
        # convert app.rho to the concrete value 
        # if log10 rho are provided.
        if self.fromlog10: 
            self.resistivity_ = np.power(
                10, self.resistivity_)
            if self.verbose > 7 : 
                print("Resistivity profiling data should be overwritten to "
                      " take the concrete values rather than log10(ohm.meters)"
                      )
            self.data_['resistivity'] = self.resistivity_
        
        self._logging.info(f'Retrieving the {self.__class__.__name__!r} '
                           ' components and recompute the coordinate values...')
        
        self.position_ = self.data_.station 
        self.lat_ = self.data_.latitude  
        self.lon_= self.data_.longitude 
        self.east_ = self.data_.easting 
        self.north_ = self.data_.northing 
        
        if self.verbose > 7: 
            print(f'Compute {self.__class__.__name__!r} parameter numbers.' )
            
        self._logging.info(f'Assert the station {self.station!r} if given' 
                           'or auto-detected otherwise.')
        
        # assert station and use the automatic station detection  
        ##########################################################
        if self.auto and self.station is not None: 
            warnings.warn (
                f"Station {self.station!r} is given while 'auto' is 'True'."
                  " Only the auto-detection is used instead...", UserWarning)
                
            self.station = None 
        
        if self.station is None: 
            if not self.auto: 
                warnings.warn("Station number is missing! By default the " 
                              "automatic-detection should be triggered.")
                self.auto = True 

        if self.station is not None: 
            if self.verbose > 7 : 
                print("Assert the given station and recomputed the array position."
                      )
                self._logging.warn(
                    f'Station value {self.station!r} in the given data '
                    'should be overwritten...')
                
        # recompute the position and dipolelength 
        self.position_, self.dipole = _assert_station_positions(
            df = self.data_, **kws)
        self.data_['station'] = self.position_ 
        
        ############################################################
        # Define the selected anomaly (conductive_zone )
        # ix: s the index of drilling point in the selected 
        # conductive zone while 
        # pos: is the index of drilling point in the whole 
        # survey position  
        self.conductive_zone_, self.position_zone_, ix, pos,  =\
            defineConductiveZone(
                        self.resistivity_,
                        s= self.station, 
                        auto = self.auto,
                        #keep Python numbering index (from 0 ->...),
                        keepindex = True, 
                        
                        # No need to implement the dipole computation 
                        # for detecting the sation position if the 
                        # station is given
                        # dipole =self.dipole if self.station is None else None,
                        
                        p = self.position_
            )

        if self.verbose >7 : 
            print('Compute the property values at the station location ' 
                  ' expecting for drilling location <`sves`> at'
                  f' position {str(pos+1)!r}')
            
        # Note that `sves` is the station location expecting to 
        # hold the drilling operations at this point. It is considered  
        # as the select point of the conductive zone. 
        self.sves_ = f'S{pos:03}' 
        
        self._logging.info ('Loading main params value from the expecting' 
                            f' drilling location: {self.sves_!r}')
    
        self.sves_lat_ = self.lat_[pos] 
        self.sves_lon_= self.lon_[pos] 
        self.sves_east_ = self.east_[pos]
        self.sves_north_= self.north_[pos] 
        self.sves_resistivity_= self.resistivity_[pos]
        
        # Compute the predictor parameters 
        self.power_ = power(self.position_zone_)
        self.shape_ = shape(self.conductive_zone_ ,
                             s= ix , 
                             p= self.position_)
        self.magnitude_ = magnitude(self.conductive_zone_)
        self.type_ = type_ (self.resistivity_)
        self.sfi_ = sfi(cz = self.conductive_zone_, 
                         p = self.position_zone_, 
                         s = ix, 
                         dipolelength= self.dipole
                         )
        
        if self.verbose > 7 :
            pn = ('type', 'shape', 'magnitude', 'power' , 'sfi')
            print(f"Parameter numbers {smart_format(pn)}"
                  " were successfully computed.") 

        return self 

    def summary(self,
                keeponlyparams: bool = False, 
                return_table: bool =False, 
                ) -> object | DataFrame : 
        """ Summarize the most import parameters for prediction purpose.
        
        If `keeponlyparams` is set to ``True``. Method should output only 
        the main important params for prediction purpose...
        
        """
        
        try:
             getattr(self, 'type_'); getattr(self, 'sfi_')
        except NotFittedError:
            raise NotFittedError(
                "Can't call the method 'summary' without fitting the"
                f" {self.__class__.__name__!r} object first.")
        
        usefulparams = (
            'station','dipole',  'sves_lon_', 'sves_lat_','sves_east_', 
            'sves_north_', 'sves_resistivity_', 'power_', 'magnitude_',
            'shape_','type_','sfi_')
        
        table_= pd.DataFrame (
            {f"{k[:-1] if k.endswith('_') else k }": getattr(self, k , np.nan )
             for k in usefulparams}, index=range(1)
            )
        table_.station = self.sves_
        table_.set_index ('station', inplace =True)
        table_.rename (columns= {'sves_lat':'latitude', 'sves_lon':'longitude',
                        'sves_east':'easting', 'sves_north':'northing'},
                       inplace =True)
        if keeponlyparams: 
            table_.reset_index(inplace =True )
            table_.drop(
                ['station', 'dipole', 'sves_resistivity', 
                 'latitude', 'longitude', 'easting', 'northing'],
                axis='columns',  inplace =True )
         
    
        self.table_ = table_ 
            
        return table_ if return_table else self  
        
            
    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        return repr_callable_obj  (self)
       
    
    def __getattr__(self, name):
        if name.endswith ('_'): 
            if name not in self.__dict__.keys(): 
                if name in ('data_', 'resistivity_', 'lat_', 'lon_', 
                            'easting_', 'northing_', 'sves_lon_', 'sves_lat_',
                            'sves_east_', 'sves_north_', 'sves_resistivity_',
                            'power_', 'magnitude_','shape_','type_','sfi_'): 
                    raise NotFittedError (
                        f'Fit the {self.__class__.__name__!r} object first'
                        )
                
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        appender  = "" if rv is None else f'. Do you mean {rv!r}'
        
        raise AttributeError (
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{appender}{"" if rv is None else "?"}'
            )

    
@refAppender(refglossary.__doc__)    
class VerticalSounding (ElectricalMethods): 
    """ 
    Vertical Electrical Sounding (VES) class; inherits of ElectricalMethods 
    base class. 
    
    The VES is carried out to speculate about the existence of a fracture zone
    and the layer thicknesses. Commonly, it comes as supplement methods to |ERP| 
    after selecting the best conductive zone when survey is made on 
    one-dimensional. 
    
    Arguments 
    -----------
    
    **fromS**: float 
        The depth in meters from which one expects to find a fracture zone 
        outside of pollutions. Indeed, the `fromS` parameter is used to  
        speculate about the expected groundwater in the fractured rocks 
        under the average level of water inrush in a specific area. For 
        instance in `Bagoue region`_ , the average depth of water inrush 
        is around ``45m``.So the `fromS` can be specified via the water inrush 
        average value. 
        
    **rho0**: float 
        Value of the starting resistivity model. If ``None``, `rho0` should be
        the half minumm value of the apparent resistivity  collected. Units is
        in Ω.m not log10(Ω.m)
        
    **h0**: float 
        Thickness  in meter of the first layers in meters.If ``None``, it 
        should be the minimum thickess as possible ``1.m`` . 
    
    **strategy**: str 
        Type of inversion scheme. The defaut is Hybrid Monte Carlo (HMC) known
        as ``HMCMC``. Another scheme is Bayesian neural network approach (``BNN``). 
        
    **vesorder**: int 
        The index to retrieve the resistivity data of a specific sounding point.
        Sometimes the sounding data are composed of the different sounding 
        values collected in the same survey area into different |ERP| line.
        For instance:
            
            +------+------+----+----+----+----+----+
            | AB/2 | MN/2 |SE1 | SE2| SE3| ...|SEn |
            +------+------+----+----+----+----+----+
            
        Where `SE` are the electrical sounding data values  and `n` is the 
        number of the sounding points selected. `SE1`, `SE2` and `SE3` are 
        three  points selected for |VES| i.e. 3 sounding points carried out 
        either in the same |ERP| or somewhere else. These sounding data are 
        the resistivity data with a  specific numbers. Commonly the number 
        are randomly chosen. It does not refer to the expected best fracture
        zone selected after the prior-interpretation. After transformation 
        via the function :func:`~watex.tools.coreutils.vesSelector`, the header  
        of the data should hold the `resistivity`. For instance, refering to 
        the table above, the data should be:
            
            +----+----+-------------+-------------+-------------+-----+
            | AB | MN |resistivity  | resistivity | resistivity | ... |
            +----+----+-------------+-------------+-------------+-----+
        
        Therefore, the `vesorder` is used to select the specific resistivity
        values i.e. select the corresponding sounding number  of the |VES| 
        expecting to locate the drilling operations or for computation. For 
        esample, `vesorder`=1 should figure out: 
            
            +------+------+----+--------+----+----+------------+
            | AB/2 | MN/2 |SE2 |  -->   | AB | MN |resistivity |
            +------+------+----+--------+----+----+------------+
        
        If `vesorder` is ``None`` and the number of sounding curves are more 
        than one, by default the first sounding curve is selected ie 
        `rhoaIndex` equals to ``0``
        
    **typeofop**: str 
        Type of operation to apply  to the resistivity 
        values `rhoa` of the duplicated spacing points `AB`. The *default* 
        operation is ``mean``. Sometimes at the potential electrodes ( `MN` ),the 
        measurement of `AB` are collected twice after modifying the distance
        of `MN` a bit. At this point, two or many resistivity values are 
        targetted to the same distance `AB`  (`AB` still remains unchangeable 
        while while `MN` is changed). So the operation consists whether to the 
        average ( ``mean`` ) resistiviy values or to take the ``median`` values
        or to ``leaveOneOut`` (i.e. keep one value of resistivity among the 
        different values collected at the same point `AB` ) at the same spacing 
        `AB`. Note that for the ``LeaveOneOut``, the selected 
        resistivity value is randomly chosen.
        
    **objective**: str 
        Type operation to output. By default, the function outputs the value
        of pseudo-area in :math:`$ohm.m^2$`. However, for plotting purpose by
        setting the argument to ``view``, its gives an alternatively outputs of
        X and Y, recomputed and projected as weel as the X and Y values of the
        expected fractured zone. Where X is the AB dipole spacing when imaging 
        to the depth and Y is the apparent resistivity computed.
        
    **kws**: dict 
        Additionnal keywords arguments from |VES| data operations. 
        See :func:`watex.tools.exmath.vesDataOperator` for futher details.
        
    See also 
    ---------
    `Kouadio et al 2022 <https://doi.org/10.1029/2021WR031623>`_
    
    References
    ----------
    *Koefoed, O. (1970)*. A fast method for determining the layer distribution 
        from the raised kernel function in geoelectrical sounding. Geophysical
        Prospecting, 18(4), 564–570. https://doi.org/10.1111/j.1365-2478.1970.tb02129.x .
        
    *Koefoed, O. (1976)*. Progress in the Direct Interpretation of Resistivity 
        Soundings: an Algorithm. Geophysical Prospecting, 24(2), 233–240.
        https://doi.org/10.1111/j.1365-2478.1976.tb00921.x .
        
        
    Examples
    --------
    >>> from watex.methods import VerticalSounding 
    >>> from watex.tools import vesSelector 
    >>> vobj = VerticalSounding(fromS= 45, vesorder= 3)
    >>> vobj.fit('data/ves/ves_gbalo.xlsx')
    >>> vobj.ohmic_area_ # in ohm.m^2
    ... 349.6432550517697
    >>> vobj.nareas_ # number of areas computed 
    ... 2
    >>> vobj.area1_, vobj.area2_ # value of each area in ohm.m^2 
    ... (254.28891096053943, 95.35434409123027) 
    >>> vobj.roots_ # different boundaries in pairs 
    ... [array([45.        , 57.55255255]), array([ 96.91691692, 100.        ])]
    >>> data = vesSelector ('data/ves/ves_gbalo.csv', index_rhoa=3)
    >>> vObj = VerticalSounding().fit(data)
    >>> vObj.fractured_zone_ # AB/2 position from 45 to 100 m depth.
    ... array([ 45.,  50.,  55.,  60.,  70.,  80.,  90., 100.])
    >>> vObj.fractured_zone_resistivity_
    ...array([57.67588974, 61.21142365, 64.74695755, 68.28249146, 75.35355927,
           82.42462708, 89.4956949 , 96.56676271])
    >>> vObj.nareas_ 
    ... 2
    >>> vObj.ohmic_area_
    ... 349.6432550517697
    
    """
    
    def __init__(self,
                 fromS: float = 45.,
                 rho0: float = None, 
                 h0 : float = 1., 
                 strategy: str = 'HMCMC',
                 vesorder: int = None, 
                 typeofop: str = 'mean',
                 objective: Optional[str] = 'coverall',
                 **kws) -> None : 
        super().__init__(**kws) 
        
        self._logging = watexlog.get_watex_logger(self.__class__.__name__)
        self.fromS=fromS 
        self.vesorder=vesorder 
        self.typeofop=typeofop
        self.objective=objective 
        self.rho0=rho0, 
        self.h0=h0
        self.strategy = strategy
        self.table_= None 
        
        for key in list( kws.keys()): 
            setattr(self, key, kws[key])
            

    def fit(self, data: str | DataFrame, **kwd ): 
        """ Fit the sounding |VES| curves and computed the ohmic-area and set  
        all the features for demarcating fractured zone from the selected 
        anomaly. 
        
        Parameters 
        -----------
        data:  Path-like object, DataFrame
            The string argument is a path-like object. It must be a valid file
            wich encompasses the collected data on the field. It shoud be
            composed of spacing values `AB` and  the apparent resistivity 
            values `rhoa`. By convention `AB` is half-space data i.e `AB/2`. 
            So, if `data` is given, params `AB` and `rhoa` should be kept to
            ``None``. If `AB` and `rhoa` is expected to be inputted, user must
            set the `data`  to ``None`` values for API purpose. If not an error
            will raise. Or the recommended way is to use the `vesSelector` tool
            in :func:`watex.tools.vesSelector` to buid the |VES| data before 
            feeding it to the algorithm. See the example below.
            
        AB: array-like 
            The spacing of the current electrodes when exploring in deeper. Units  
            are in meters. Note that the `AB` is by convention equals to `AB/2`. 
            It's taken as half-space of the investigation depth. 
        
        MN: array-like 
            Potential electrodes distances at each investigation depth. Note 
            by convention the values are half-space and equals to `MN/2`.
        
        rhoa: array-like 
            Apparent resistivity values collected in imaging in depth. Units 
            are in Ω.m not log10(Ω.m)
        
        kwds: dict 
            additional keywords arguments, specific to the readable files. 
            Refer to :method:`watex.property.Config.parsers` . Use the key()
            to get all the readables format. 
            
        Returns 
        -------
         object: a DC -resistivity |VES| object.  

        """
        
        def prettyprinter (n, r,v): 
            """ Display some details when verbose is higher... 
            
            :param n: int : number of areas 
            :param r: array-like. Pair values of integral bounds (-inf, +inf)
            :param v: array-float - values of pseudo-areas computed.  """
            print('=' * 73 )
            print('| {0:^15} | {1:>15} | {2:>15} | {3:>15} |'.format(
                'N-area', 'lb:-AB/2 (m)','ub:-AB/2(m)', 'ohmS (Ω.m^2)' ))
            print('=' * 73 )
            for ii in range (n): 
                print('| {0:^15} | {1:>15} | {2:>15} | {3:>15} |'.format(
                    ii+1, round(r[ii][0]), round(r[ii][1]), round(v[ii], 3)))
                print('-'*73)
        
        self._logging.info (f'`Fit` method from {self.__class__.__name__!r}'
                           ' is triggered')
        
        if self.verbose >= 7 : 
            print(f'Range {str(self.vesorder)!r} of resistivity data of the  '
                  'should be selected as the main sounding data. ')
        self.data_ = vesSelector(
            data = data, index_rhoa= self.vesorder, **kwd )
        self.max_depth_ = self.data_.AB.max()
        
        if self.fromlog10: 
            self.resistivity_ = np.power(
                10, self.resistivity_)
            if self.verbose > 7 : 
                print("Sounding resistivity data should be converted to "
                      "the concrete resistivity values (ohm.meters)"
                      )
            self.data_['resistivity'] = self.resistivity_
            
        if self.fromS >= self.max_depth_ : 
            raise VESError(
                " Process of the depth monitoring is aborted! The searching"
                f" point of param 'fromS'<{self.fromS}m> ' is expected to "
                 f" be less than the maximum depth <{self.max_depth_}m>.")
        
        if self.verbose >= 3 : 
            print("Pseudo-area should be computed from AB/2 ={str(self.fromS)}"
                  f" to {self.max_depth_} meters. "
                  )
        r = ohmicArea( data = self.data_ , sum = False, ohmSkey = self.fromS,  
                    objective = self.objective , typeofop = self.typeofop, 
                    )
        self._logging.info(f'Populating {self.__class__.__name__!r} property'
                           ' attributes.')
        oc, gc = r 
        
        ohmS, self.err_, self.roots_ = list(oc) 
        self.nareas_ = len(ohmS) 
        
        self._logging.info(f'Setting the {self.nareas_} pseudo-areas calculated.')
        for ii in range(self.nareas_): 
            self.__setattr__(f"area{ii+1}_", ohmS[ii])
            
        self.roots_ = np.split(self.roots_, len(self.roots_)//2 ) if len(
            self.roots_) > 2 else [np.array(self.roots_)]
        
        if self.verbose >= 7 : 
            prettyprinter(n= self.nareas_, r= self.roots_, v= ohmS)

        self.ohmic_area_= sum(ohmS) # sum the different spaces 
        
        self.XY_ , _, self.XYarea_ = list(gc) 
        self.AB_ = self.XY_[:, 0] 
        self.resistivity_ = self.XY_[:, 1] 
        self.fractured_zone_= self.XYarea_[:, 0] 
        self.fractured_zone_resistivity_ = self.XYarea_[:, 1] 
        
        if self.verbose > 7 :
            print("The Parameter numbers were successfully computed.") 
        return self 
    
    def summary(self,
                keeponlyparams: bool = False, 
                return_table: bool =False, 
                ) -> DataFrame | object : 
        """ Summarize the most import features for prediction purpose.
        
        :param keeponlyparams: bool, 
            If `keeponlyparams` is set to ``True``. Method should output only 
            the main important params for prediction purpose. 
        :param return_table: bool, 
            if ``True``, returns only the summarized table. 
            
        """
        
        try:
             getattr(self, 'ohmic_area_'); getattr(self, 'fractured_zone_')
        except NotFittedError:
            raise NotFittedError(
                "Can't call the method 'summary' without fitting the"
                f" {self.__class__.__name__!r} object first.")
        
        usefulparams = (
            'area', 'AB','MN', 'arrangememt','utm_zone', 'objective', 'rho0',
             'h0', 'fromS', 'max_depth_', 'ohmic_area_', 'nareas_')
        
        table_= pd.DataFrame (
            {f"{k}": getattr(self, k , np.nan )
             for k in usefulparams}, index=range(1)
            )
        table_.area = self.area
        table_.set_index ('area', inplace =True)
        table_.rename (columns= {
            'max_depth_':'max_depth',
            'ohmic_area_':'ohmic_area',
            'nareas_':'nareas'
                            },
                           inplace =True)
        if keeponlyparams: 
            table_.reset_index(inplace =True )
            table_.drop( 
                [ el for el in list(table_.columns) if el !='ohmic_area'],
                axis='columns',  inplace =True
                )

        self.table_ = table_ 
            
        return table_ if return_table else self  
        
    def invert( self, data: str | DataFrame , strategy=None, **kwd): 
        """ Invert1D the |VES| data collected in the exporation area.
        
        :param data: Dataframe pandas - contains the depth measurement AB from 
            current electrodes, the potentials electrodes MN and the collected 
            apparents resistivities. 
        
        :param rho0: float - Value of the starting resistivity model. If ``None``, 
            `rho0` should be the half minumm value of the apparent resistivity  
            collected. Units is in Ω.m not log10(Ω.m)
        :param h0: float -  Thickness  in meter of the first layers in meters.
             If ``None``, it should be the minimum thickess as possible ``1.``m. 
        
        :param strategy: str - Type of inversion scheme. The defaut is Hybrid Monte 
            Carlo (HMC) known as ``HMCMC``. Another scheme is Bayesian neural network
            approach (``BNN``). 
        
        :param kwd: dict - Additionnal keywords arguments from |VES| data  
            operations. See :doc:`watex.utils.exmath.vesDataOperator` for futher
            details.
        
        .. |VES| replace: Vertical Electrical Sounding 
        
        """
        self.data_ = getattr(self, 'data_', None)
        if self.data_ is None: 
            raise NotFittedError(f'Fit the {self.__class__.__name__!r} object first')
  
        # invert data 
        #XXX TODO 
        if strategy is not None: 
            self.strategy = strategy 
            
        invertVES(data= self.data_, h0 = self.h0 , rho0 = self.rho0,
                  typeof = self.strategy , **kwd)
        
        return self 
    
    def __repr__(self):
        """ Pretty format for programmer following the API... """
        return repr_callable_obj(self)
       
    def __getattr__(self, name):
        if name.endswith ('_'): 
            if name not in self.__dict__.keys(): 
                if name in ('data_', 'resistivity_', 'ohmic_area_', 'err_', 
                            'roots_', 'XY_', 'XYarea_', 'AB_','resistivity_',
                            'fractured_zone_', 'fractured_zone__resistivity_'): 
                    raise NotFittedError (
                        f'Fit the {self.__class__.__name__!r} object first'
                        )
                
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        appender  = "" if rv is None else f'. Do you mean {rv!r}'
        
        raise AttributeError (
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{appender}{"" if rv is None else "?"}'
            )

    
def _readfromdcObjs(self, data: List[object ] ,
                     dcmethod:object=ResistivityProfiling ,  
                     exception: F = ERPError ): 
    """ Read object metadata object. 
    
    A set of :class:`.ResistivityProfiling` objects.
    
    :param data: list-a collection of  DC-resistivity method 
        objects
    
    :returns: bool- whether an object is readable as a DC-resistivity 
        profiling or sounding object or not.``False`` otherwise.  
    """

    self._logging.info (f"Read a collection '{dcmethod.__name__!r}' objects")
    
    # assert whether the method is implemented 
    if dcmethod.__name__  not in ( 'ResistivityProfiling', 
                                  'VerticalSounding'):
        raise NotImplementedError(
        f"Method {dcmethod.__name__!r} is not implemented")
        
    if not isinstance( data, (list, tuple, np.ndarray)): 
        data =[data]
    # assert whether each element composing the data is ERP object  
    s = set ([ isinstance (o, dcmethod ) for o in data  ]) 
    if len(s)!=1 or (len(s) ==1  and not tuple(s)[0]): 
        return False 
    
    # show the progress bar        
    pbar = data if not TQDM else tqdm.tqdm(data ,ascii=True, unit='B',
                 desc ="dc-erp" if dcmethod.__name__ =='ResistivityProfiling'\
                     else'dc-ves',
                 ncols =77)
    
    for kk , o in enumerate(pbar) :
        try: 
            if isinstance (o, dcmethod ): 
                self.data_.append(o) 
        except : self.isnotvalid_.append(o) 
        
    #     pbar.update(kk) if TQDM else ''
    # (pbar.close (), print('-completed-') ) if TQDM else ''
    
    if len(self.data_)==0 : 
        warnings.warn("No DC-resistvity profiling data detected. Make a collection" 
                      f" of profiling object using {dcmethod.__name__!r} class."
                      )
        raise exception("None DC-Resistivity profiling data found!"
                       )
        
    #show stats 
    if self.verbose > 0:
        print()
        show_stats (data , self.data_,
                    obj = 'DC-ERP' if dcmethod.__name__=='ResistivityProfiling' \
                        else 'DC-VES' ,
                    lenl=79)
        
    # make a ids 
    if self.verbose > 3 : 
        print("Set the ids for each line e.g. line1 for the first line.")
    
    self.survey_names_ = np.array(make_ids(self.data_, 'line', None, True))
    
    return True 

    
def _readfrompath (self, data: List[str] ,
                   dcmethod: object= ResistivityProfiling, 
                   **kws ): 
    """ Read data from a file or a path-like object. 
    
    It collects the list of |ERP| or |VES| files and create a DC -resistivity
    object from a DC -resistivity method. 
    
    :param data: str or path-like object, 
    
    :param kws: Additional keyword from 
        :func:`watex.tools.coreutils.parseStations`. It refers to the 
        `station_delimiter` parameters. 
        
    """
    self._logging.info (" {self.__class__.__name__!r} collects the "
                        "resistivity objects ")
    
    # assert whether the method is implemented 
    if dcmethod.__name__  not in ( 'ResistivityProfiling', 
                                  'VerticalSounding'):
        raise NotImplementedError(
        f"Method {dcmethod.__name__!r} is not implemented")
        
        
    ddict = dict() 
    regex = re.compile (r'[$& #@%^!]', flags=re.IGNORECASE)
    
    self.survey_names_ = None  # initialize 
    if isinstance(data, str ): 
        if os.path.isfile (data): 
             data =[data ]
        elif os.path.dirname(data): 
            data = [os.path.join( data, d ) for d in os.listdir(data)] 
        else : raise FileNotFoundError("File not found")
       
    if self.read_sheets: 
        _, ex = os.path.splitext( data[0])
        if ex != '.xlsx': 
            raise TypeError (" Reading multisheets expects an excel file."
                             " extension not: {ex!r}")
        for d in data : 
            try: 
                ddict.update ( **pd.read_excel (d , sheet_name =None))
            except : pass 
                
            #collect stations names
        if len(ddict)==0 : 
            raise ERPError ("Can'find the DC-resistitivity profiling data "
                            )
        self.survey_names_ = list(map(
            lambda o: regex.sub('_', o).lower(), ddict.keys()))

        if self.verbose > 3: 
            print(f"Number of the collected data from stations are"
                  f" : {len(self.survey_names_)}")
            
        data = list(ddict.values ())
        
    # make a survey id from collection object 
    if self.survey_names_ is None: 
        self.survey_names_ = list(map(lambda o :regex.sub(
            '_',  os.path.basename(o)), data ))
        
    # remove the extension and keep files names 
    self.survey_names_ = list(
        map(lambda o: o.split('.')[0], self.survey_names_)) 
    

    # populate and assert stations and fromS   
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # if list of station is not given for each file 
    # note that here station is station where one expect to 
    # locate a drilling drilling i.e. sves
    _parse_dc_args(self, dcmethod,  **kws)
  
    # show the progress bar 
    pbar = data if not TQDM else tqdm.tqdm(data ,ascii=True, unit='B',
                 desc ="dc-erp" if dcmethod.__name__ =='ResistivityProfiling'\
                     else'dc-ves',
                 ncols =77)
  
    # -> read the data and make dc Objs 
    for kk,  o  in enumerate (pbar)  :
        try :
            if dcmethod.__name__=='ResistivityProfiling':
                dcObj = dcmethod( 
                    station = self.stations[kk] , 
                    dipole= self.dipole,
                    auto=True if self.stations[kk] is None else self.auto, 
                    utm_zone = self.utm_zone, 
                    )
                self.data_.append (dcObj.fit(o).summary(
                    keeponlyparams=True))
                self.stations[kk] = dcObj.sves_ 
                
            elif dcmethod.__name__ =='VerticalSounding': 
                dcObj = dcmethod(
                    fromS=self.fromS[kk], 
                    vesorder=self.vesorder,
                    typeofop=self.typeofop,
                    objective=self.objective,
                    rho0=self.rho0, 
                    h0=self.h0,
                    strategy=self.strategy
                    )
                self.data_.append (dcObj.fit(o).summary(
                    keeponlyparams=True))
   
        except : 
            self.isnotvalid_.append(o)
            
    #     pbar.update(kk) if TQDM else ''
    # (pbar.close (), print('-completed-') ) if TQDM else ''
    
    if self.verbose > 0:
        #show stats 
        print()
        show_stats (data , self.data_,
                    obj = 'DC-ERP' if dcmethod.__name__=='ResistivityProfiling' \
                        else 'DC-VES' ,
                    lenl=79)
        
    if self.verbose > 3: 
            print(" Number of file unsucceful read is:"
                  f" {len(self.isnotvalid_)}")
    
    
def _parse_dc_args(self, dcmethod: object , **kws): 
    """ parse dc arguments to  fit the number of survey lines and populate
    sanitize the attributes accordingly.
    
    :param kws: Additional keyword from 
        :func:`watex.tools.coreutils.parseDCArgs`. It refers to the 
        `station_delimiter` parameters. 
    """  
    flag=0
    if dcmethod.__name__=='ResistivityProfiling': 
        sf , arg = self.stations , 'stations'
        flag=0
    elif dcmethod.__name__=='VerticalSounding': 
        sf, arg =self.fromS , 'fromS'
        flag=1
    
        
    if sf is None: 
        sf= np.repeat ([45.], len(self.survey_names_)) if flag else np.repeat(
            [None], len(self.survey_names_)) 
        
    elif sf is not None: 
        if os.path.isfile (str(sf)): 
            sf=parseDCArgs(sf, arg=arg, **kws)
        elif isinstance (sf, str): 
            sf= [sf]
        if isinstance(sf, (int , float)) and flag: 
            sf= np.repeat ([sf], len(self.survey_names_))
        
        msg =''.join([ 
                f"### Number of {arg!r} does not fit the number of"
                f" {'sites' if arg =='fromS' else 'stations'}. "
                "Expect {0} but {1} {2} given."
            ])
        
    if len(sf)!= len(self.survey_names_): 
        self._logging.error (msg)
        warnings.warn(msg.format(len(self.survey_names_),len(sf), 
            f"{'is' if len(sf)<2 else 'are'}") )
            
        if self.verbose > 3: 
            print("-->!Number of DC-resistivity data read sucessfully"
                  f"= {len(self.survey_names_)}. Number of the given"
                  "  stations considered as a drilling points"
                  f"={len(sf)}. Station must fit each survey lines."
                  )
                    
        raise StationError (msg.format(
            len(self.survey_names_), len(sf) , f"{'is' if len(sf) <=1 else 'are'}",
            ))
        
    if not flag: 
        self.stations = sf 
    elif flag:
        self.fromS = sf 
        
        
def _geterpattr (attr , dl ): 
    """ Get attribute from the each DC-resistivity object and 
    collect into numpy array. 
    
    If `stack` is ``True``, it will collect stacked data allong axis 1. 
    :param attr: attribute name 
    :param dl: list of erp object 
    """
    # np.warnings.filterwarnings(
    #     'ignore', category=np.VisibleDeprecationWarning)  
    return np.array(list(map(lambda o : getattr(o, attr), dl )), 
                    dtype =object)     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        