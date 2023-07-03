# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Created on Thu Sep 22 12:21:11 2022

"""
Stratigraphic 
==============
Construct layers model from given layers properties such as density, porosity 
permeability, transmissivity, resistivity , patches and so on ... 
Build Statigraphic model from Inversion models blocks. This should be used to 
predict the log under each station as well as the thicknesses from the collected 
true boreholes and well resistivity data in the survey area. 

"""
import os
import warnings
import copy
from importlib import resources
import numpy as np

from .._typing import ( 
    NDArray, List 
    )
from .._watexlog import watexlog 
from ..decorators import (
    gplot2d
    )
from ..exceptions import ( 
    NotFittedError, 
    ModelError, 
    DepthError,
    )
from .core import (
    GeoBase 
    )
# from .database import ( 
#     GeoDataBase,
#     )
from ..utils.exmath import gradient_descent 
from ..utils.funcutils import ( 
    serialize_data, 
    load_serialized_data,
    smart_format, 
    # concat_array_from_list,
    parse_json, 
    parse_yaml,
    is_iterable, 
    convert_value_in, 
    ellipsis2false
    )
from ..utils.geotools import (
    _sanitize_db_items,
    _assert_model_type,
    assert_len_lns_tres, 
    assert_station, 
    fit_rocks, 
    fit_stratum_property, 
    get_s_thicknesses, 
    print_running_line_prop, 
    plot_stratalog,
    get_closest_gap,
    lns_and_tres_split,
    # find_similar_structures, 
    display_ginfos
    
    )
from ..utils._dependency import ( 
    import_optional_dependency
    )

TQDM= False 
try : 
    import_optional_dependency("tqdm")
except ImportError: 
    pass 
else: 
    import tqdm 
    TQDM = True 
_logger = watexlog().get_watex_logger(__name__ )

__all__=["GeoStrataModel"]


class GeoStrataModel(GeoBase):
    """
    Create a stratigraphic model from 2D inversion models blocks. 

    The Inversion model block is two dimensional array of shape 
    (n_vertical_nodes, n_horizontal_nodes ). Can use external packages to
     build blocks and provide the 2Dblock into `crm` parameter. 

    The challenge of this class  is firstly to delineate with much accuracy 
    the existing layer boundary (top and bottom) and secondly,to predict the 
    stratigraphy log before the drilling operations at each station. Moreover, 
    it's a better way to select the right drilling locationand also to estimate 
    the thickness of existing layer such as water table layer as well as to 
    figure out the water reservoir rock in the case of groundwater 
    exploration [1]_. 

    Note that if the model blocks is build from external softwares. User can
    provided model usefull details into the keyword arguments of the `fit` 
    methods. For instance the `model_station_locations`, `model_depth`, 
    `model_x_nodes` etc. See more details in `fit` method docstring. 

    Parameters 
    ----------
    area:str, optional 
      The name of the area where the survey is done. 

    beta:  int,  default=5              
            Value to  divide into the CRM blocks to improve 
            the computation times.                       
    n_epochs:  int,  default=100
       Number of iterations for new resistivity model construction (NM). 
            
    ptol: float, default=.1   
       the error value to tolerate during NM construction between the `tres` 
       values given and the calculated resistivity in `crm`. Higher is the 
       error, less is the accuracy. 
     
    eta: float, default=1e4 
       It is the learning rate for gradient descent computing to reach its 
       convergence. If `kind` is  set to `polynomial` the default value should 
       be  set`1e-8`.  
          
    kind: str , default='linear'        
       Kind of model function to compute the best fit model to replace the
       value in `crm` . Can be 'linear' or 'polynomial'. if `polynomial` is 
       set, specify the `degree`.
        
    degree: int, default=1
         Polynomial function degree to implement gradient descent algorithm. 
         If `kind` is set to `Polynomial` the default `degree` is ``3.`` 
         
    z0: float, default=0. 
       The elevation at the first stations. Note here, we consider that no
       topography is included and the default is the level of the sea. If 
       the `model_depth`if provided in the `fit_params`, no need to specify. 
       
    max_depth: float, default=700 
      The maximum depth for building the block. by default, the unit is in 
      meters. If `model_depth` is given, no need to provide. 
      
    step: float, default=50 
       The step between stations. Here the unit is in meters. If the 
       `model_station_locations` if given through the `fit_params`, no need to 
       specified. 
       
    doi: default='1km' 
       the depth of investigation of the skin-depth. By default without any 
       unit suffix, it should be in meters. 
       
    tolog10: bool, default=False 
       Convert the true values of layer resistivities to log10. 

    verbose: bool, default=False  
      Output messages and warn users. 

    Attributes
    -----------
    nm_: Ndarray of ( n_depth , n_stations )     
        The New resistivity model (NM) matrix with the same dimension 
        with `crm` model blocks.
    nmSites_: ArrayLike 
       The recomputed station locations of the new resistivity model (NM)
    crmSites_: ArrayLike, 
       The recomputed stations locations of the model-calculated resistivity 
       (CRM)
       
    Examples
    ----------
    >>> import numpy as np 
    >>> from watex.geology.stratigraphic import GeoStrataModel
    >>> 
    >>> # (1):  Using the CRM without any inversion specified files. 
    >>> 
    >>> # test while files 
    >>> np.random.seed (42)
    >>> # generate a model calculated resistivity, layers and true 
    >>> # resistivities values
    >>> crm = np.abs( np.random.randn (215 , 70 ) *1000 )
    >>> tres = np.linspace (crm.min() +1  , crm.max() +1 , 12 ) # 
    >>> layers = ['granites', 'gneiss', 'sedim.']
    >>> gs= GeoStrataModel( to_log10 =True )
    >>> gs.fit(crm, tres = tres, layers =layers).buildNM(display_infos =True )
    >>> print( gs.nm_.shape)
    build-NM: 18B [00:01, 12.60B/s]
    ----------------------------------------------------------------------
                          layers [auto=automatic](13)                     
    ----------------------------------------------------------------------
       1. hard rock (auto)                2. clay (auto)                  
       3. conglomerate (auto)             4. sedimentary rock (auto)      
       5. gravel (auto)                   6. igneous rock (auto)          
       7. fresh water (auto)              8.Sedim.                        
       9.Gneiss                          10. *struture not found (auto)   
      11. saprolite (auto)               12. duricrust (auto)             
      13.Granites                      
    ----------------------------------------------------------------------
    (215, 70) 
    >>> gs.strataModel () # plot strata model, by default kind ='NM'
    >>> gs.plotStrata ('s7')  # plot strata log at station S7 
    Out[7]: watex.geology.stratigraphic.PseudoStratigraphic
    >>> 
    >>> # (2): Works with occam2d inversion files if 'pycsamt' or 'mtpy' 
    >>> # is installed. It will call the module Geodrill from pycsamt to 
    >>> make occam2d 2D resistivity block for our demo.
    >>> # It presumes pycsamt is installed. 
    >>> 
    >>> from pycsamt.geodrill.geocore import Geodrill 
    >>> path=r'data/inversfiles/inver_res/K4' # path to inversion files 
    >>> inversion_files = {'model_fn':'Occam2DModel', 
    ...                   'mesh_fn': 'Occam2DMesh',
    ...                    "iter_fn":'ITER27.iter',
    ...                   'data_fn':'OccamDataFile.dat'
    ...                    }
    >>> tres =[10, 66, 70, 180, 1000, 2000, 3000, 7000, 15000 ] 
    >>> layers =['river water', 'fracture zone', 'granite']
    >>> inversion_files = {key:os.path.join(path, vv) for key,
                    vv in inversion_files.items()}
    >>> gdrill= Geodrill (**inversion_files, 
                         input_resistivities=input_resistivity_values
                         )
    >>> # we can collect the 'model_res' and occam2d inversion usefull 
    >>> # 'attributes' from  `gdrill object` and passed to the 
    >>> # 'GeoStrataModel' then fit_params keyword arguments method as 
    >>> geosObj = GeoStrataModel(ptol =0.1).fit(
                         crm = gdrill.model_res ,
                         tres=gdrill.input_resistivities, 
                         layers=gdrill.input_layer_names, 
                         model_x_nodes=gdrill.model_x_nodes, 
                         model_stations= gdrill.station_names,
                         model_depth= gdrill.geo_depth, 
                         model_station_locations= gdrill.station_locations,
                         data_fn = gdrill.data_fn , 
                         mesh_fn=gdrill.mesh_fn, 
                         iter_fn= gdrill.iter_fn
                         model_fn= gdrill.model_fn)
    >>> geosObj.buildNM () 
    >>> zmodel = geosObj._zmodel
    >>> geosobj.nm_ # New constructed resistivity 2D model block

    Notes
    ------
    Modules work properly with occam2d inversion files if 'pycsamt' or 'mtpy' 
    is installed and  inherits the `GeoBase` class which works with geological
    structures and properties. Furhermore, Occam2d inversion files are also 
    acceptables for building model blocks. However the MODEM resistivity 
    files development is still ongoing.
    
    References 
    -----------
    .. [1] Kouadio, L. K., Liu, R., Malory, A. O., Liu, W., Liu, C., A novel 
           approach for water reservoir mapping using controlled source 
           audio - frequency magnetotelluric in Xingning area , Hunan 
           Province, China. Geophys. Prospect., 
           https://doi.org/10.1111/1365-2478.1338
    """
    def __init__(
        self, 
        area:str=None, 
        beta: int=5,  
        n_epochs: int=100, 
        ptol: float=0.1 , 
        eta: float=1e-4, 
        kind: str='linear', 
        degree: int=1,
        z0: float=0., 
        max_depth: float=700., 
        step: float=50., 
        doi: str='1km',
        tolog10: bool=False,
        verbose: bool=False,
        **kwargs
        ):
        super().__init__( verbose= verbose, **kwargs)

        self.area=area
        self.z0=z0
        self.step=step
        self.max_depth=max_depth 
        self.beta=beta 
        self.ptol=ptol 
        self.n_epochs=n_epochs
        self.eta=eta
        self.kind=kind
        self.degree=degree
        self.tolog10=tolog10
        self.doi=convert_value_in(doi)

        self._tres=None 
        self.s0 =None 
        self._zmodel =None
        self.nm_= None 
        self._z =None
        self.nmSites_=None
        self.crmSites_=None 
        
    def fit(self, crm: NDArray, tres: List[float]=None,
            layers: List[str]=None, **fit_params): 
        """ 
        Check, populate attributes and rebuild the model-calculated 
        resistivity (CRM). 
        
        Parameters 
        ------------
        crm : ndarray of shape(n_vertical_nodes, n_horizontal_nodes ),  
           Array-like of inversion two dimensional model blocks. Note that 
           the `n_vertical_nodes` is the node from the surface to the depth 
           while `n_horizontal_nodes` must include the station location 
           (sites/stations) 
           
        tres: List, Arraylike of float, 
           Layer true values of resistivity collected in the survey area.
           refer to [1]_ for more details. Resistivity is preferable to  
           be distinct. 

        layers: list or ArrayLike of str 
           The name of layers collected in the survey area. Their corresponding
           resistivity values are the argument of `tres`. Mostly, it refers to 
           the geological informations of collected in the area. 
                
        fit_params: dict, 
           If a keyword argument refering to inversion model details, it 
           should be prefixed by the keyword `model_` such as the the
           following attributes: 
               
           - `model_depth`: Arraylike shape (n_depth, ) of the model. It is  
             the depth the surface to the bottom of each layer that 
             composed the pseudo-boreholes. It refer to the n_vertical nodes. 
           - `model_resistivities`: ArrayLike of shape ( y_nodes , x_nodes ). 
             It is array of model block composed of resistivity values of  
             vertical nodes and horizontal nodes. It is the model-calculated 
             resistivity. No need if the value is passed to `crm` parameter. 
           - `model_x_nodes`: ArrayLike of shape (x_nodes, ) is  the  
             horizontal nodes of the model block. 
           - `model_station_locations`: Arraylike of offset of valid stations 
             locations. It might truly refers to the number of investigated 
             stations. 
           - `model_stations`: List or Arrylike of the stations names of 
             the investigated area. It might be consistent with the 
             `model_station_locations`. 
           - `model_rms`: Root-Mean-Squared values of the models after 
             inversion. By default , the target is set to ``1.0``. 
           - `model_roughness`: Roughness of the models. This is an optional 
             parameters and using when OCCAM2D software is used to invert the 
             data. 
             
           Model files can also be includes. In that case, the attributes must 
           suffixes with `_fn`. Some useful model files can be `model`, `mesh`
           `data` and `iter`. For instance to set the data file to a keyword 
           arguments it sould be ``data_fn=xxxx`` whether the new data attribute 
           should be retrived as `data_fn`. Note that: 
               
           - `model_fn` is the model files after inversions 
           - `data_fn` is the data files before inversions 
           - `iter_fn` is the iteration results files if applicable with 
              the kind of inversion software used 
           - `mesh_fn` is the mesh files construct for forward modeling. 

        Return 
        --------
        ``self``: :class:`watex.geology.stratigraphic.GeoStrataModel`. 
            return `self` for methods chaining. 
        
        References 
        -----------
        .. [1] Kouadio, L. K., Liu, R., Malory, A. O., Liu, W., Liu, C., A novel 
               approach for water reservoir mapping using controlled source 
               audio - frequency magnetotelluric in Xingning area , Hunan 
               Province, China. Geophys. Prospect., 
               https://doi.org/10.1111/1365-2478.1338
        """
        self.crm = crm 
        self.layers=layers 
        self.tres = tres 
   
        if self.tolog10: 
            self._tres = np.log10 (self._tres)
            
        if self.layers is None or self.tres is None: 
            msgn= "Layers are missing. " if self.layers is None else (
                "The layers true resistivities (TRES) are missing. " if 
                self.tres is None else '')
            raise ModelError (f"{msgn}Layers and their corresponding"
                              " resistivities(TRES) are expected for building"
                              " the new discrete resistivity model(NM)."
                              )
        self.layers = is_iterable(
            self.layers , exclude_string= True, transform =True )
        self.tres = is_iterable(self.tres, exclude_string= True, transform =True 
                                )
        
        self.set_inversion_model_attr(**fit_params)
    
        self.s0= np.zeros_like(self.crm )
            
        if self.crm is not None: 
            self._makeBlock()

        return self
    
    def buildNM (self, return_NM: bool= ..., display_infos: bool=... ):
        """ Build New discrete resistivity from the model-calculated 
        resistivity CRM 
        Trigger the NM build and return the NM building option
        
        """
        return_NM, display_infos = ellipsis2false(return_NM, display_infos)
        
        self.inspect 
        
        return self._createNM( return_NM= return_NM, 
                              display_infos= display_infos ) 
    
    def set_inversion_model_attr ( self, **model_params): 
        """ Set inversion model parameters as attributes. Note all related 
        to the raw inversion model strats with attribte `model_xxx ` while 
        all related to files must must suffixed with `_fn`. For instances 
        the folling data such ['model', 'iter', 'data' , 'mesh']  can be 
        added as keywords arguments where `_fn` is suffixed to each names.
        
        Parameters 
        -----------
        model_parameters: dict 
          Keyword arguments from inversion model parameters. The useful 
          attributes passed to accurathe model creation are: 
        
          - model_depth: Arraylike shape (n_depth, ) of the model, mostly 
            refer to the vertical nodes. 
          - model_resistivities: ArrayLike of shape ( y_nodes , x_nodes ). It 
            is array of model block composed of resistivity values of vertical 
            nodels and horizontal nodes. It is the model-calculated resistivity. 
            It is usefull to pass the same argument as `crm` parameters. 
          - model_x_nodes: ArrayLike of shape (x_nodes, ) is  the horizontal 
            nodes of the model block. 
          - model_station_locations: Arraylike of offset of valid stations 
            locations. It might truly refers to the number of investigated 
            stations. 
          - model_stations: List or Arraylike of the stations names of 
            the investigated area. It might be consistent with the 
            `model_station_locations`. 
          - model_rms: Root-Mean-Squared values of the models after inversion. 
            By default , the target is set to ``1.0``. 
          - model_roughness: Roughness of the models. This is an optional 
            parameters and using when OCCAM2D software is used to invert the 
            data.
            
          Model files can also be includes. In that case, the attributes must 
          suffixes with `_fn`. Some useful model files can be `model`, `mesh`
          `data` and `iter`. For instance to set the data file to a keyword 
          arguments it sould be ``data_fn=xxxx`` whether the new data attribute 
          should be retrived as `data_fn`. Note that: 
            
          - `model_fn` is the model files after inversions 
          - `data_fn` is the data files before inversions 
          - `iter_fn` is the iteration results files if applicable with 
            the kind of inversion software used 
          - `mesh_fn` is the mesh files construct for forward modeling. 

        """
        model_depth = model_params.pop("model_depth", None  )
        model_resistivities = model_params.pop(
            'model_resistivities', None)
        model_x_nodes = model_params.pop ('model_x_nodes', None)
        model_station_locations = model_params.pop (
            'model_station_locations', None )
        model_stations = model_params.pop("model_stations", None)
        model_rms = model_params.pop("model_rms", 1.)
        model_roughness = model_params.pop("model_roughness", 42. )
        
        if model_resistivities is not None: 
            self.crm = model_resistivities 
            
        if model_depth is not None: 
            model_depth = np.array ( model_depth, dtype = float)
            
            if len(model_depth )!= len(self.crm): 
                raise DepthError ("Model depth and model-calculated resistivity"
                                  " (CRM) must be a consistent size. Got"
                                  f" {len(model_depth)} and {len(self.crm)}.")
            self.model_depth= model_depth 
        
        if not hasattr (self, 'model_depth'): 
            # create a pseudo depth
            self.model_depth = np.linspace (self.z0, self.max_depth, len(self.crm))
            
        if model_resistivities is not None: 
            self.model_resistivities = model_resistivities 
            
        if model_x_nodes is not None: 
            self.model_x_nodes = model_x_nodes 
    
        if model_station_locations is not None: 
            self.model_station_locations= model_station_locations
            
        if not hasattr ( self, 'model_station_locations' ): 
            self.model_station_locations = np.arange (
                0 , self.crm.shape [1])* self.step 
            
        if not hasattr (self, 'model_x_nodes'): 
            self.model_x_nodes = self.model_station_locations 
            
        if not hasattr ( self, "model_resistivities"): 
            self.model_resistivities = self.crm 
            
        if model_stations is not None: 
            self.model_stations= model_stations 
            
        if not hasattr ( self, 'model_stations'): 
            self.model_stations = [f'S{i:02}' for i in range (
                len(self.model_station_locations)) ]
            
        self.model_rms = model_rms 
        self.model_roughness = model_roughness 
   
        # Append other attributes such as data, model, iter, mesh files
        # suffixed with `_fn` like `data_fn=xxx`.
        for key in list(model_params.keys ()) : 
            setattr ( self, key , model_params[key])

    def _createNM(self, crm =None, beta =5 , ptol= 0.1, **kws): 
        """ Create NM through the differents steps of NM creatings. 
        
        - step 1 : soft minimal computing 
        - step2 : model function computing 
        - step 3: add automatic layers
        - step 4: use ANN to find likehood layers
     
        :param crm: calculated resistivity model blocks 
        :param beta: number of block to build.
        :param ptol: Error tolerance parameters 
  
        """
        def s_auto_rocks (listOfauto_rocks): 
            """ Automatick rocks collected during the step 3
            :param listOfauto_rocks: List of automatic rocks from 
             differents subblocks. 
             
            :returns:rocks sanitized and resistivities. 
            """

            listOfauto_rocks= np.concatenate((listOfauto_rocks), axis =1)
            rho_= listOfauto_rocks[1, :]
            rho_=np.array([float(ss) for ss in rho_])
            r_= list(set(listOfauto_rocks[0, :]))
            hres= np.zeros((len(r_), 1))
            h_= []
            for ii, rock  in enumerate(r_): 
                for jj, ro in enumerate(listOfauto_rocks[0, :]): 
                    if rock == ro: 
                        h_.append(rho_[jj])
                m_= np.array(h_)
                hres[ii]= m_.mean()
                h_=[]
            return r_, hres 
        
        subblocks =kws.pop('subblocks', None)
        disp= kws.pop('display_infos', True)
        # n_epochs = kws.pop('n_epochs', None)
        hinfos =kws.pop('headerinfos',
                        ' Layers [auto=automatic]')
        return_NM= kws.pop( 'return_NM', False  )
        
        if subblocks is not None: 
            self.subblocks = subblocks
        
        self.s0 , errors=[], []
        #step1 : SOFMINERROR 
        if TQDM : 
            pbar =tqdm.tqdm(total= 3,ascii=True,unit='B',
                             desc ='build-NM', ncols =77)
       
        for ii in range(len(self.subblocks)):
            s1, error = self._softMinError(subblocks= self.subblocks[ii])
            self.s0.append(s1)
            errors.append(error)
            if TQDM : pbar.update(1)
        #step2 : MODELFUNCTION USING DESCENT GRADIENT 
        for ii in range(len(self.s0)):
            if 0 in self.s0[ii][:, :]: 
                s2, error = self._hardMinError(subblocks =self.subblocks[ii], 
                                            s0= self.s0[ii])
                self.s0[ii]=s2
                errors[ii]= error 
            if TQDM : pbar.update(2)
        arp_=[]
        #Step 3: USING DATABASE 
        for ii in range(len(self.s0)):
            if 0 in self.s0[ii][:, :]: 
                s3, autorock_properties= self._createAutoLayers(
                    subblocks=self.subblocks[ii], s0=self.s0[ii]  )
                arp_.append(autorock_properties)
                self.s0[ii]=s3       
        # Assembly the blocks 
        self.nm_ = np.concatenate((self.s0))
        self._z=self.nm_[:, 0]
        self.nm_ = self.nm_[:, 1:]
        
        if TQDM : 
            pbar.update(3)
            # print(' process completed')
            pbar.close()
            
        # make site blocks 
        self.nmSites_= makeBlockSites(x_nodes=self.model_x_nodes, 
                        station_location= self.model_station_locations, 
                             block_model=self.nm_ )
        self.crmSites_ = makeBlockSites(x_nodes=self.model_x_nodes, 
                            station_location= self.model_station_locations, 
                             block_model=self.model_resistivities)
        #Update TRES and LN 
        gammaL, gammarho = s_auto_rocks(arp_) 
        
        if self.layers is not None: 
            print_layers = self.layers  + [ ' {0} (auto)'.format(l) 
                                                 for l in gammaL ]
            self.layers = self.layers + gammaL
        # keep the auto_layer found     
        self.auto_layers =gammaL
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', category=RuntimeWarning)
            # reconvert to power10 since database value is 
            # not in log10
            if self.tolog10: 
                self.tres = list(np.power(10,self._tres))  + list (
                    np.power(10, np.array([float(rv) for rv in gammarho])))
            else: self.tres = list(self.tres) + [
                float(rv) for rv in gammarho]
            
        # display infos 
        if disp:
            display_ginfos(infos=print_layers, header= hinfos)
        #STEP 4: Train ANN: (still in development) to predict your 
        #l ayer: No need to plot the NM  (Need huge amount of data)
        # copy main attributes for pseudostratigraphic plot purpose 
        for name , attrval in zip(['TRES', 'LNS'], 
                              [self.tres , self.layers]):
            setattr(self, name, copy.deepcopy(attrval))
        # memorize data 
        _ps_memory_management(self)
        
        # turn off log10 to False
        # to not convert any more data 
        self.tolog10 =False 
        
        return self.nm_ if return_NM else self 
    
    def _softMinError(self, subblocks=None, **kws ): 
        """
        Replace the calculated resistivity by the true resistivity 
        using the soft minimal error (ξ)
        
        :param crm: Is the calculated resistivity model from Occam2D 
        inversion results 
        
        """
        buffer =self.ptol +1  #bufferr error  
        _z = subblocks[:, 0]
        subblocks = subblocks[:, 1:]
        # Hold the columns of depth values 
        s0 = np.zeros_like(subblocks.T)
        error =[]
        for ii in range(subblocks.shape[1]): # hnodes N
            for jj in range(subblocks.shape[0]): # znodes V
                for k in range(len(self.tres)) :
                   sfme_k = (subblocks.T[ii, jj]-self.tres[k])**2\
                       /subblocks.T[ii, jj] **2
                   error.append(sfme_k )
                   if sfme_k  <= self.ptol : 
                       if sfme_k  < buffer : # keep the best minimum 
                           buffer = sfme_k  
                           s0[ii, jj] = self.tres[k]
                           
                buffer = self.ptol +1      # initilize buffer 

        s0= np.concatenate((_z.reshape(_z.shape[0], 1), s0.T), axis =1)
        return s0, error 

    def _hardMinError(
        self, 
        tres=None, 
        subblocks=None, 
        s0=None, 
        ptol = None,
        kind='linear', 
        **kwargs 
        ): 
        """The second step introduces the model function F=W∙Z  where W
        contains the weights of parameters number and Z is V×2 matrix 
        that contains a "bias" column. If the parameter number P equal to two, 
        the model function:
            
        .. math:: 
            
            f(z)= \sum_{p=1}^{P} [w_{p-1} z ^{p-1}]
            
        becomes alinear function with: 
            
        .. math::
            
            f_1^{(1)} (z) = wz+r_0 \qual \text{with} w_1=w \aqual \text{and} w_0=r_0
            
        Indeed, the gradient descent algorithm  is used to find the 
        best parameters :math:`w` and :math:`r_0`  that  minimizes the  
        MSE loss function  :math:`J`.
        
        :param subblocks: `crm` block  
        :param s0: blocks from the first step :meth:`~._sofminError`
        :param kind: Type of model function to apply. Can also be 
                a `polynomial` by specifying the `degree` 
                into argument `degree`.
        :Example: 
            
            >>> from watex.geology.stratigraphic import GeoStrataModel
            >>> geosObj = GeoStrataModel().fit(**inversion_files, 
                              input_resistivities=input_resistivity_values) 
            >>> ss0, error = geosObj._hardMinError(subblocks=geosObj.subblocks[0],
                                     s0=geosObj.s0[0])
        """
        
        if tres is not None:
            self.tres = tres 
        if ptol is not None: 
            self.ptol = ptol 
        
        eta = kwargs.pop('eta', None)
        if eta is not None: 
            self.eta = eta 
        n_epochs =kwargs.pop('n_epochs', None)
        if n_epochs is not None: 
            self.n_epochs = n_epochs 
        kind = kwargs.pop('kind', None)
        if kind is not None:
            self.kind = kind 
        degree = kwargs.pop('degree', None) 
        if degree is not None: 
            self.degree = degree 
        
        buffer =self.ptol +1  #bufferr error 
        _z= s0[:, 0]
        s0 = s0[:, 1:].T

        subblocks=subblocks[:, 1:].T
        error =[]
        for ii in range(s0.shape[0]): # hnodes N
            F, *_= gradient_descent(z=_z,s=subblocks[ii,:],
                                         alpha= self.eta,
                                         n_epochs= self.n_epochs, 
                                         kind= self.kind)
            for jj in range(s0.shape[1]): # znodes V
                 if s0[ii, jj] ==0. : 
                    rp =F[jj]
                    for k in range(len(self.tres)) :
                        with np.errstate(all='ignore'): 
                            sfme_k = (rp -self.tres[k])**2\
                                 /rp**2
                            _ermin = abs(rp-subblocks[ii, jj])
                        error.append(sfme_k)
                        if sfme_k <= self.ptol and _ermin<= self.ptol: 
                             if sfme_k  < buffer : # keep the best minimum 
                                buffer = sfme_k  
                                s0[ii, jj]= self.tres[k]
                               
                    buffer = self.ptol +1      # initialize buffer 
                    
        s0= np.concatenate((_z.reshape(_z.shape[0], 1), s0.T), axis =1)    
        return s0, error 
        
    @classmethod 
    def geoArgumentsParser(cls, config_file =None): 
        """ Read and parse the `GeoStrataModel` arguments files from 
        the config [JSON|YAML] file.
        :param config_file: configuration file. Can be [JSON|YAML]
        
        :Example: 
            >>> GeoStrataModel.geoArgumentsParser(
                'data/saveJSON/cj.data.json')
            >>> GeoStrataModel.geoArgumentsParser(
                'data/saveYAML/cy.data.yml')
        """
        if config_file.endswith('json'): 
            args = parse_json(config_file)
        elif config_file.endswith('yaml') or config_file.endswith('yml'):
            args = parse_yaml(config_file)
        else: 
            raise ValueError('Can only parse JSON and YAML data.')
        
        return cls(**args)
            
        
    def _makeBlock (self): 
        """ Construct the differnt block  based on `beta` param. Separate blocks 
        from number of vertical nodes generated by the first `beta` value applied 
        to the `crm`."""

        self.zmodel_ = np.concatenate((self.model_depth.reshape(
            self.model_depth.shape[0], 1),  self.model_resistivities), axis =1) 
                                    
        vv = self.zmodel_[-1, 0] / self.beta 
        for ii, nodev in enumerate(self.zmodel_[:, 0]): 
            if nodev >= vv: 
                npts = ii       # collect number of points got.
                break 
        self._subblocks =[]
        
        bp, jj =npts, 0
        if len(self.zmodel_[:, 0]) <= npts: 
            self._subblocks.append(self.zmodel_)
        else: 
            for ii , row in enumerate(self.zmodel_) : 
                if ii == bp: 
                    _tp = self.zmodel_[jj:ii, :]
                    self._subblocks.append(_tp )
                    bp +=npts
                    jj=ii
                    
                if len(self.zmodel_[jj:, 0])<= npts: 
                    self._subblocks.append(self.zmodel_[jj:, :])
                    break 
                
        return self._subblocks 
 
    @property 
    def subblocks(self): 
        """ Model subblocks divised by `beta`"""
        return self._subblocks 
    
    @subblocks.setter 
    def subblocks(self, subblks):
        """ keep subblocks as :class:`~GeoStrataModel` property"""
        
        self._subblocks = subblks 

    @gplot2d(reason='model',cmap='jet_r', plot_style ='pcolormesh',
             show_grid=False )
    def strataModel(self, kind ='nm', **kwargs): 
        """ 
        Visualize the   `strataModel` after `nm` creating using decorator from 
        :class:'~.geoplot2d'. 
        
        :param kind: can be : 
            - `nm` mean new model plots after inputs the `tres`
            - `crm` means calculated resistivity from occam model blocks 
            *default* is `nm`.
        :param plot_misft:  Set to ``True`` if you want to visualise the error 
            between the `nm` and `crm`. 
        :param scale: Can be ``m`` or ``km`` for plot scale 
        :param in_percent`: Set to `True` to see your plot map scaled in %.
        
        :Example: 
            
            >>> from watex.geology.stratigraphic import GeostrataModel
            >>> geosObj = GeostrataModel().fit(**inversion_files,
                                  input_resistivities=input_resistivity_values, 
                                  layers=input_layer_names)
            >>> geosObj.strataModel(kind='nm', misfit_G =False)
        """
        m_='watex.geology.GeostrataModel.strataModel'
        def compute_misfit(rawb, newb, percent=True): 
            """ Compute misfit with calculated block and new model block """
            m_misfit = .01* np.sqrt (
                (rawb - newb)**2 /rawb**2 ) 
            if percent is True: 
                m_misfit= m_misfit *100.
            return m_misfit 
        
        if isinstance(kind, bool): 
            kind ='nm' if kind else 'crm'

        depth_scale = kwargs.pop('scale', 'm')
        misfit_G =kwargs.pop('misfit_G', False)
        misfit_percentage = kwargs.pop('in_percent', True)
        
        kind = _assert_model_type(kind)
        if self.nm_ is None: 
            self._createNM()  
                
        if kind =='nm':
            data = self.nmSites_ 
        if kind =='crm': 
            data = self.crmSites_

      # compute model_misfit
        if misfit_G: 
            if kind =='crm':
                if self.verbose:
                    warnings.warn(
                        "By default, the plot should be the stratigraphic"
                        " misfit<misfit_G>.")
    
            self._logging.info('Visualize the stratigraphic misfit.')
            data = compute_misfit(rawb=self.crmSites_ , newb= self.nmSites_, 
                                  percent = misfit_percentage)
            
            if self.verbose:
                print('{0:-^77}'.format('StrataMisfit info'))
                print('** {0:<37} {1} {2} {3}'.format(
                    'Misfit max ','=',data.max()*100., '%' ))                      
                print('** {0:<37} {1} {2} {3}'.format(
                    'Misfit min','=',data.min()*100., '%' ))                          
                print('-'*77)
            
        if self.verbose:
            warnings.warn(
                f'Data stored from {m_!r} should be moved on binary drive and'
                ' method arguments should be keywordly only.', FutureWarning)
            
        return (data, self.model_stations, self.model_station_locations,
            self.model_depth, self.doi, depth_scale, self.model_rms, 
            self.model_roughness, misfit_G )
    
    def _createAutoLayers(self, tres =None, layers =None, subblocks=None, 
                         s0=None, ptol = None,  **kws):
        """ 
        The third step of replacement using the geological database. 
        
        The third step consists to find the rock  γ_L in the Γ with the 
         ceiled mean value γ_ρ  in E_props column is close to the calculated 
        resistivity r_11. Once the rock γ_L  is found,the calculated 
        resistivity r_11 is replaced by γ_ρ. Therefore, the rock γ_L is
         considered as an automatic layer. At the same time,the TRES and LN
         is updated by adding   GeoStratigraphy_ρ  and  γ_L respectively to 
         the existing given data. 
         
        """
        # tres = kws.pop('tres', None)
        disp = kws.pop('display_infos', False)
        hinfos = kws.pop('header', 'Automatic layers')
  
        _z= s0[:, 0]
        s0 = s0[:, 1:].T
        _temptres , _templn =[], []
        subblocks=subblocks[:, 1:].T

        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', category=RuntimeWarning)
            for ii in range(s0.shape[0]): # hnodes N
                for jj in range(s0.shape[1]): # znodes V
                    if s0[ii, jj] ==0. : 
                        lnames, lcres =self.findGeostructures(
                            np.power(10, subblocks[ii, jj]))
                        _temptres.append(np.log10(lcres))
                        _templn.append(lnames)
    
        auto_rocks_names_res, automatics_resistivities =\
            _normalizeAutoresvalues(_templn,_temptres )
            
        for ii in range(s0.shape[0]): # hnodes N
           for jj in range(s0.shape[1]): # znodes V
               if s0[ii, jj] ==0. :
                   for k in range(automatics_resistivities.shape[1]): 
                       subblocks[ii, jj] == automatics_resistivities[0,:][k]
                       s0[ii, jj]= automatics_resistivities[1,:][k]
                       break 
                      
        s0= np.concatenate((_z.reshape(_z.shape[0], 1), s0.T), axis =1) 
        # display infos 
        if disp: display_ginfos(infos=layers,header= hinfos)
        
        return  s0, auto_rocks_names_res
    
    @staticmethod
    def _strataPropertiesOfSite(obj, station =None, display_s=True): 
        """ Build all properties of strata under each station.  
        
        Parameters
        ----------
        station: str or int
            Use normal count to identify the number of site to plot or use 
            the name of station preceed of letter `S`. For instance site 
            1 matches the station `S00` litterally
        display_s:bool
            Display the log layer infos as well as layers thicknesses
        
        Examples
        --------
        >>> from watex.geology.stratigraphic import GeoStrataModel 
        >>> import watex.utils.geotools as GU 
        >>> geosObj = GeoStrataModel().fit( input_resistivities=TRES, 
        ...              layers=LNS,**INVERS_KWS)
        >>> geosObj._strataPropertiesOfSite(geosObj,station= 'S05')
        """
        
        def stamping_ignored_rocks(fittedrocks, lns): 
            """ Stamping the pseudo rocks and ignored them during plot."""
            ir= set(fittedrocks).difference(set(lns))
            for k in range( len(fittedrocks)): 
                if fittedrocks[k] in ir : 
                    fittedrocks[k] ='$(i)$'
            return fittedrocks
        
        # assert the station, get it appropriate index and take the tres 
        # at that index 
        if station is None: 
            stns = ["S{:02}".format(i) for i in range(obj.nmSites_.shape[1])]
            obj._logging.error('None station is found. Please select one station'
                                f' between {smart_format(stns)}')
            if obj.verbose:
                warnings.warn("NoneType can not be read as station name."
                                " Please provide your station name. list of " 
                                 f" sites are {smart_format(stns)}")
            raise ValueError("NoneType can not be read as station name."
                             " Please provide your station name.")
        
        if  obj.nmSites_ is None: 
                obj._createNM()
        
        try : 
            id0= int(station.lower().replace('s', ''))
        except : 
            id_ =assert_station(id= station, nm = obj.nmSites_)
            station_ = 'S{0:02}'.format(id_)
        else : 
            id_ =assert_station(id= id0 + 1, nm = obj.nmSites_)
            station_ = 'S{0:02}'.format(id0)
        obj.logS = obj.nmSites_[:, id_] 
     
        # assert the input  given layers and tres 
        is_the_same_length, msg  = assert_len_lns_tres(
            obj.LNS , obj.TRES)
        
        if is_the_same_length :
            # then finf 
            pslns = obj.LNS
            pstres= obj.TRES 
            ps_lnstres   = [(a, b) for a , b 
                            in zip(obj.LNS, obj.TRES)] 
        if not is_the_same_length:
            # find the pseudoTRES and LNS for unknowrocks or layers
            msg +=  "Unknow layers should be ignored."   
            _logger.debug(msg)
            warnings.warn(msg) if obj.verbose else None
            
            pslns, pstres, ps_lnstres =  fit_tres(
                                            obj.LNS, obj.TRES, 
                                            obj.auto_layers)
        # now build the fitting rocks 
        fitted_rocks =fit_rocks(logS_array= obj.nmSites_[:,id_],
                                   lns_=pslns , tres_=pstres)
        # set the raws fitted rocks 
        import copy
        setattr(obj, 'fitted_rocks_r', copy.deepcopy(fitted_rocks) )
        # change the pseudo-rocks  located in fitted rocks par ignored $i$
        # and get  the stamped rocks 
        setattr(obj, 'fitted_rocks',stamping_ignored_rocks(
            fitted_rocks, obj.LNS ) )
        
        # fit stratum property 
        sg, _, zg, _= fit_stratum_property (obj.fitted_rocks,
                                obj._z, obj.logS)
        obj.log_thicknesses, obj.log_layers,\
            obj.coverall = get_s_thicknesses( 
            zg, sg,display_s= display_s, station = station_ )
            
        # set the dfault layers properties hatch and colors from database
        obj.hatch , obj.color =  fit_default_layer_properties(
            obj.log_layers) 
        
        return obj
    
    @staticmethod
    def plotStrata(station, zoom=None, annotate_kws=None, **kws):
        """ Build the Stratalog. 
        :param station: station to visualize the plot.
        :param zoom: float  represented as visualization ratio
            ex: 0.25 --> 25% view from top =0.to 0.25* investigation depth 
            or a list composed of list [top, bottom].
        
        :Example: 
            >>> input_resistivity_values =[10, 66,  700, 1000, 1500,  2000, 
                                    3000, 7000, 15000 ] 
            >>> input_layer_names =['river water', 'fracture zone', 'granite']
            # Run it to create your model block alfter that you can only use 
            #  `plotStrata` only
            # >>> obj= quick_read_geomodel(lns = input_layer_names, 
            #                             tres = input_resistivity_values)
            >>> plotStrata(station ='S00')
        
        """
        
        if annotate_kws is None: annotate_kws = {'fontsize':12}
        if not isinstance(annotate_kws, dict):
            annotate_kws=dict()
        obj = _ps_memory_management(option ='get' )  
        obj = GeoStrataModel._strataPropertiesOfSite (obj,station=station,
                                                       **kws )
        # plot the logs with attributes 
        plot_stratalog (obj.log_thicknesses, obj.log_layers, station,
                                    hatch =obj.hatch ,zoom=zoom,
                                    color =obj.color, **annotate_kws)
        print_running_line_prop(obj)
        
        return obj 

    @property 
    def tres(self): 
        """ Input true resistivity"""
        return self._tres 
    @tres.setter 
    def tres(self, ttres):
        """ Convert Tres to log 10 resistivity if tolog10 is set to ``True``."""
        ttres = is_iterable (ttres, transform =True )
        try : 
            ttres = np.array( ttres, dtype = float) 
        except: 
            raise TypeError (
                f"TRES expect numeric values. Got {np.array(ttres).dtype.name !r} ")
        self._tres = list (ttres ) 
        
    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'crm'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 
    
    
    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        _t = ("area", "beta", "n_epochs","ptol",  "eta","kind", "degree",
              "z0" , "max_depth","step","doi", "tolog10", "verbose")

        outm ='<{!r}:' + ', '.join( [
            "{}={}".format( k,   False if getattr(self, k)==... else (
                    "[...]" if len( is_iterable(getattr (
                        self, k),exclude_string=True, transform=True))>3
                 else str (getattr(self, k)))) 
            for k in _t]) + '>' 
            
        return  outm.format(self.__class__.__name__)
    
def _ps_memory_management(obj=None, option='set'): 
    """ Manage the running times for stratigraphic model construction.
    
    The script allows to avoid running several times the GeoStrataModel
     model construction to retrieve the stratigraphic log at each station.  
    It memorizes the model data for the first run and used it when calling it
    to  visualize the strata log at each station. Be aware to edit this script.
    """
    MMOD = 'watex.etc'; memory='__memory.pkl'
    with resources.path (MMOD, memory) as mfile : 
         memory_file = str(mfile) # for consistency
         
    mkeys= ('set', 'get', 'recover', 'fetch', set)
    if option not in mkeys: 
        raise ValueError('Wrong `option` argument. Acceptable '
                         f'values are  {smart_format(mkeys[:-1])}.')
        
    if option in ('set', set): 
        if obj is None: 
            raise TypeError('NoneType object cannot be set. Provide the'
                            " model object to 'obj' parameter.") 
        psobj_token = __build_ps__token(obj)
        data = (psobj_token, list(obj.__dict__.items()))
        serialize_data ( data, memory, savepath= os.path.dirname (memory_file))
        return 
    elif option in ('get', 'recover', 'fetch'): 
        memory_exists =  os.path.isfile(memory_file)
        if not memory_exists: 
            _logger.error('No memory found. Run the GeoStrataModel class'
                          ' beforehand to create your first model.')
            warnings.warn("No memory found. You need to build your"
                          " GeoStrataModel model by running the class first.")
            raise  MemoryError("Memory not found. Run the `buildNM` method"
                               " to create your model first.")
        psobj_token, data_ = load_serialized_data(memory_file )
        data = dict(data_)
        # create PseudoStratigraphicObj from metaclass and inherits from 
        # dictattributes of GeoStrataModel class
        psobj = type ('PseudoStratigraphic', (), { 
            k:v for k, v in data.items()})
        psobj.__token = psobj_token
        
        return psobj
                                
def makeBlockSites(station_location, x_nodes, block_model): 
    """ Build block that contains only the station locations values
    
    :param station_location: array of stations locations. Must be  
                self contains on the horizontal nodes (x_nodes)
    :param x_nodes: Number of nodes in horizontal 
    :param block_model: Resistivity blocks model 
    
    :return: 
        - `stationblocks`: Block that contains only the
        station location values.
        
    :Example:
        
        >>> from watex.geology.stratigraphic import makeBlockSite
        >>> mainblocks= get_location_value(
            station_location=geosObj.makeBlockSite,
             x_nodes=geosObj.model_x_nodes, block_model=geosObj.model_res )
    """
    
    index_array =np.zeros ((len(station_location), ), dtype =np.int32)
    for ii, distance in enumerate(station_location): 
        for jj , nodes in enumerate(x_nodes): 
            if nodes == distance : 
                index_array [ii]= jj
                break 
            elif nodes> distance: 
                min_= np.abs(distance-x_nodes[jj-1])
                max_= np.abs(distance - x_nodes[jj+1])
                if min_<max_: 
                    index_array [ii]= jj-1
                else: index_array [ii]=jj
                break 
    _tema=[]
    for ii in range(len(index_array )):
        a_= block_model[:, int(index_array [ii])]
        _tema.append(a_.reshape((a_.shape[0], 1)))
        
    stationblock = np.concatenate((_tema), axis=1)
    
    return stationblock 
    
def fit_default_layer_properties(layers, dbproperties_= ['hatch', 'colorMPL']): 
    """ Get the default layers properties  implemented in database. 
     
    For instance get the hatches and colors from given layers implemented in 
    the database by given the database `dbproperties_`.
    
    :param layers: str or list of layers to retrieve its properties
        If specific property is missing , ``'none'`` will be return 
    :param db_properties_: str, list or database properties 
    :return: property items sanitized
    
    :Example: 
        
    >>> import watex.geology.stratigraphic as GS
    >>> GS.fit_default_layer_properties(
    ...    ['tuff', 'granite', 'evaporite', 'saprock']))
    ... (['none', 'none', 'none', 'none'],
    ...     [(1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0),
    ...     (1.0, 0.807843137254902, 1.0)])
    """
    # for consistency check again and keep the DB properties untouchable.
    dbproperties_= ['colorMPL' if g.lower().find('mpl') >=0 else 
                    'FGDC' if g.lower()=='fgdc'else g.lower() 
                    for g in dbproperties_]
    if isinstance(layers , str): layers =[layers]
    assert_gl = ['yes' if isinstance(ll, str) else 'no' for ll in layers]
    if not len(set(assert_gl)) ==1: 
        raise TypeError("Wrong given layers. Names should be a string!")
    if 'name' or'__description' not in  dbproperties_: 
        dbproperties_.insert(0, 'name')
    
    __gammaProps = GeoBase.getProperties(dbproperties_)
    
    r_props =[['none' for i in layers] for j in range(len(__gammaProps)-1)]
    for k  , l in enumerate(layers): 
        if l  in __gammaProps[0] :
            ix= __gammaProps[0].index(l)
            for kk, gg in enumerate(r_props) : 
                gg[k]= __gammaProps[1:][kk][ix]
                
    r_props = [_sanitize_db_items(r_props[k], force=True )
               for k in range(len (r_props))]
    return tuple(r_props)    
 
def __build_ps__token(obj):
    """ Build a special token for each GeoStrataModel model. Memory uses 
    each model token to fetch the strata log at each station without 
    any model recomputation. Please don't edit anything here at least you
    knwow what your are doing. Force editing is your own risk."""
    import random 
    random.seed(42)
    __c =''.join([ i for i in  [''.join([str(c) for c in obj.crmSites_.shape]), 
     ''.join([str(n) for n in obj.nmSites_.shape]),
    ''.join([l for l in obj.layers]) + str(len(obj.layers)), 
    str(len(obj.tres))] + [''.join(
        [str(i) for i in [obj.eta, obj.beta, obj.doi,obj.n_epochs,
                          obj.ptol, str(obj._z.max())]])]])
    __c = ''.join(random.sample(__c, len(__c))).replace(' ', '')  
    n= ''.join([str(getattr(obj, f'{l}'+'_fn', str(obj.area).lower() ))
                         for l in ['model', 'iter', 'mesh', 'data']])
    n = ''.join([s.lower() 
                 for s in random.sample(n, len(n))]
                ).replace('/', '').replace('\\', '')
    
    return ''.join([n, __c]).replace('.', '')
        
    
def fit_tres(lns, tres, autorocks, force=False, **kws): 
    """ Read and get the resistivity values from tres that match the 
     the given layers.
     
    Find the layers and  their corresponding resistivity values from the 
    database especially when values in the TRES and LN are not the same
    length. It's not possible to match each value to its
    correspinding layer name. Therefore the best approach is to read the
    TRES and find the layer name in the database based on the closest value.

    :param lns: list of input layers 
    :param tres: list of input true resistivity values 
    :param autorocks: list of the autorocks found when building the new model.
    :param force: bool, force fitting resistivity value with the rocks in 
            the database whenever the size of rocks match perfectly 
            the number of the rocks. Don't do that if your are sure that the 
            TRES provided fit the  layers in LNS.
    :param kws: is database column property. Default is
        `['electrical_props', '__description']`
        
    :returns: new pseudolist contains the values of rocks retrived from 
        database as well as it closest value in TRES.
    """
    def flip_back_to_tuple(value , substitute_value, index=1): 
        """convert to tuple to list before assign values and 
          reconvert to tuple  after assignment for consistency. 
          `flip_back_to_tuple` in line in this code is the the same like :
                newTRES[ii] = list(newTRES[ii])
                newTRES[ii][1] = val
                newTRES[ii] = tuple (newTRES[ii]) 
          """ 
        value = list(value)
        if index is not None: 
            value[index] = substitute_value
        else : value = substitute_value
        return tuple (value) 

     
    ix = len(autorocks)
    lns0, tres0, rlns, rtres= lns_and_tres_split(ix,  lns, tres)
    if len(lns0) > len(tres0): 
        msg= ''.join(['Number of given layers `{0}` should not be greater ',
                      ' than the number of given resistivity values` {1}`.'])
        msg= msg.format(len(lns0), len(tres0))
        
        n_rock2drop = len(tres0)-len(lns0) 
        msg += f" Layer{'s' if abs(n_rock2drop)>1 else ''} "\
            f"{smart_format(lns0[n_rock2drop:])} should be ignored."
    
        lns0 = lns0[: n_rock2drop]
        warnings.warn(msg)
        _logger.debug(msg)
       
    if sorted([n.lower() for n in lns0]
              ) == sorted([n.lower() for n in lns]): 
        if not force: 
            return lns0, tres0, [(a, b) for a , b in zip(lns0, tres0)]
        
    r0 =copy.deepcopy(tres0)
    # for consistency, lowercase the layer name
    # get the properties [name and electrical properties]  
    # from geoDataBase try to build new list with none values 
    # loop for all layer and find their index then 
    # their elctrical values 
    #           if name exist in database then:
    #           loop DB layers names 
    #           if layer is found then get it index 
    lns0 =[ln.lower().replace('_', ' ') for ln in lns0 ]
    _gammaRES, _gammaLN = GeoBase.getProperties(**kws)

    newTRES =[None for i in tres0]
    temp=list()
    for ii, name in enumerate(lns0) : 
        if name in _gammaLN: 
            ix = _gammaLN.index (name) 
            temp.append((name,_gammaRES[ix])) 
            
    # keep the lns0 rocks that exists in the database 
    # and replace the database value by the one given 
    #in tres0 and remove the tres value with 
    # unknow layer by its corresponding value.
    if len(temp)!=0: 
        for name, value in temp: 
            ix, val = get_closest_gap (value= value, iter_obj=tres0)
            newTRES[ix]= (name, val) 
            tres0.pop(ix) 
    # try to set the values of res of layer found in 
    # the database is not set = 0 by their corresponding
    # auto -layers. if value is in TRES. We consider 
    #that the rocks does not exist and set to None
    for ii, nvalue in enumerate(newTRES):
        try: 
            iter(nvalue[1])
        except:
            if nvalue is not None and nvalue[1]==0. :
                newTRES[ii]= None 
            continue 
        else: 
            # if iterable get the index and value of layers
            # remove this values in the tres 
            ix, val = get_closest_gap (value=nvalue[1], iter_obj=tres0)
            newTRES[ii] = flip_back_to_tuple (newTRES[ii], val, 1) 
            tres0.pop(ix) 
            
    for ii, nvalue in enumerate(tres0):
        ix,_val=  get_closest_gap (value=nvalue,status ='isoff', 
                                   iter_obj=_gammaRES, 
                          condition_status =True, skip_value =0 )
        # get the index of this values in tres
        index = r0.index (_val) 
        newTRES[index] = (_gammaLN[ix], nvalue)
        
    # create for each tres its pseudorock name 
    # and pseudorock value
    # print(newTRES)
    # print(rlns, rtres )
    pseudo_lns, pseudo_tres=[], [] 
    
    for value  in newTRES: 
        if hasattr (value , '__iter__'): 
            pseudo_lns.append (value[0] )
            pseudo_tres.append (value[1] )
        else :
            pseudo_lns .append (None)
            pseudo_tres.append (np.nan )
            
    pseudo_lns +=rlns
    pseudo_tres += rtres 
    # pseudo_lns = [a [0] for a in newTRES] + rlns 
    # pseudo_tres = [b[1] for b in newTRES] + rtres 
    newTRES += [(a, b) for a , b in zip(rlns, rtres)]
    
    return pseudo_lns , pseudo_tres , newTRES 

def quick_read_geomodel(lns=None, tres=None):
    """Quick read and build the geostratigraphy model (NM) 
    
    :param lns: list of input layers 
    :param tres: list of input true resistivity values 
    
    :Example: 
        >>> import watex.geology.stratigraphic as GM 
        >>> obj= GM.quick_read_geomodel()
        >>> GC.fit_tres(obj.layers, obj.tres, obj.auto_layer_names)
    """
    PATH = 'data/occam2D'
    k_ =['model', 'iter', 'mesh', 'data']

    try : 
        INVERS_KWS = {
            s +'_fn':os.path.join(PATH, file) 
            for file in os.listdir(PATH) 
                      for s in k_ if file.lower().find(s)>=0
                      }
    except :
        INVERS_KWS=dict()
     
    TRES=[10, 66,  70, 100, 1000, 3000]# 7000] #[10,  70, 100, 1000,  3000]
    #[10, 66, 70, 100, 1000, 2000, 3000, 7000, 15000 ]      
    LNS =['river water','fracture zone', 'MWG', 'LWG', 
          'granite', 'igneous rocks', 'basement rocks']
    
    lns = lns or LNS 
    tres= tres or TRES 

    if len(INVERS_KWS) ==0: 
        _logger.error("NoneType can not be read! Need the basics Occam2D"
                         f" inversion {smart_format(k_)} files.")

        raise ValueError("NoneType can not be read! Need the basics Occam2D"
                         f" inversion {smart_format(k_)} files.")
        
    geosObj = GeoStrataModel( input_resistivities=tres, 
                      layers=lns,**INVERS_KWS)
    geosObj._createNM()
    
    return geosObj 

def _normalizeAutoresvalues(listOfstructures,listOfvalues):                            
    """ Find the different structures that exist and
    harmonize value. and return an array of originated values and 
    the harmonize values and the number of automatics layer found as 
    well as their harmonized resistivity values. 
    """
    autolayers = list(set(listOfstructures))
    hvalues= np.zeros((len(autolayers,)))
    
    temp=[]
    for ii , autol in enumerate(autolayers): 
        for jj, _alay in enumerate(listOfstructures):
            if _alay ==autol: 
                temp.append(listOfvalues[jj])
        hvalues[ii]= np.array(list(set(temp))).mean()
        temp=[]
    
    # build values array containes the res and the harmonize values 
    h= np.zeros((len(listOfvalues),))
    for ii, (name, values) in enumerate(zip (listOfstructures,
                              listOfvalues)):
        for jj, hnames in enumerate(autolayers) : 
            if name == hnames: 
                h[ii]= hvalues[jj]
    
    finalres= np.vstack((np.array(listOfvalues),h) )
    finalln = np.vstack((np.array(autolayers), hvalues))
    return  finalln, finalres 