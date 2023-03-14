# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Created on Thu Sep 22 12:21:11 2022

"""
Stratigraphic 
==============
Construct layers model from given layers properties such as density , porosity 
permeability, transmissivity, resistivity , patches and so on ... 
Build Statigraphic model from Inversion models blocks. This should be used to 
predict the log under each station as well as the thicknesses from the collected 
true boreholes and well resistivity data in the survey area. 

"""
import os
import warnings
import copy
import numpy as np

from .._typing import NDArray 
from .._watexlog import watexlog 
from ..decorators import (
    gplot2d
    )
from ..exceptions import NotFittedError
from .core import (
    Base 
    )
from .database import ( 
    GeoDataBase,
    )
from ..utils.funcutils import ( 
    serialize_data, 
    load_serialized_data,
    smart_format, 
    concat_array_from_list,
    parse_json, 
    parse_yaml,
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
    pseudostratigraphic_log,
    get_closest_gap,
    lns_and_tres_split, 
    
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

#XXXTODO: add MODEM construction block  in progress 
class GeoStrataModel(Base):

    def __init__(
        self, 
        beta=5, 
        ptol=0.1 , 
        n_epochs=100, 
        tres=None,
        eta=1e-4, 
        kind='linear', 
        degree=1, 
        build=False, 
        **kwargs
        ):
        super().__init__( **kwargs)

        self._beta =beta 
        self._ptol =ptol 
        self._n_epochs = n_epochs
        self._tres = tres
        self._eta = eta
        self._kind =kind
        self._degree = degree
        self._b = build
        
        self.s0 =None 
        self._zmodel =None
        self.nm= None 
        self.z =None
        self.nmSites=None
        self.crmSites=None 

        for key in list(kwargs.keys()): 
            setattr(self, key, kwargs[key])
            
    def fit(self, crm: NDArray =None, beta =5 , ptol= 0.1, **kws): 
        """ 
        Fit, populate attributes and construct the new stratigraphic 
        model (NM)
        
        Parameters 
        ------------
        crm : ndarray of shape(n_vertical_nodes, n_horizontal_nodes ),  
            Array-like of inversion two dimensional model blocks. Note that 
            the `n_vertical_nodes` is the node from the surface to the depth 
            while `n_horizontal_nodes` must include the station location 
            (sites/stations) 
            
        beta:  int,                
                Value to  divide into the CRM blocks to improve 
                the computation times. default is`5`                               
        n_epochs:  int,  
                Number of iterations. default is `100`
        ptols: float,   
                Existing tolerance error between the `tres` values given and 
                the calculated resistivity in `crm` 
        Return 
        --------
        ``self``: :class:`watex.geology.stratigraphic.GeoStrataModel`. 
            return `self` for methods chaining. 
        
        """
        for key in list(kws.keys()): 
            setattr(self, key, kws[key])
            
        if crm is not None: 
            self.crm=crm 
        # expect the bock is build from 
        # external modeling softwares 
        if hasattr (self, "input_resistivities"):  
            if self.input_resistivities: 
                self.tres = self.input_resistivities
        if hasattr (self, 'model_res'): 
            if self.model_res is not None : 
                self.crm = self.model_res 
                self.s0= np.zeros_like(self.model_res)
            
        if self.crm is not None: 
            self._makeBlock()
            
        self.build 
        
        return self
    
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
    
    @property 
    def n_epochs(self): 
        """ Iteration numbers"""
        return self._n_epochs 
    @n_epochs.setter 
    def n_epochs(self, n_iterations): 
        """ n_epochs must be in integers value and greater than 0"""
        try : 
            self._n_epochs = int(n_iterations)
        except: 
             TypeError('Iteration number must be `integer`') 
        else: 
            if self._n_epochs  <=0: 
                self._logging.debug(
                 " Unaceptable iteration value! Must be a positive value not "
                f"{'a negative value.' if self._n_epochs <0 else 'equal to 0.'}")
                warnings.warn(f" {self._n_epochs} is unaceptable value."
                          " Could be resset to the default value=100.")
                self._n_epochs = 100 
                
    @property 
    def beta (self): 
        """ Block constructor param"""
        return self._beta 
    @beta.setter 
    def beta(self, beta0 ):
        """ Block constructor must be integer value."""
        try : 
            self._beta = int(beta0)
        except Exception: 
            raise TypeError
        else: 
            if self._beta <=0 :
                self._logging.debug(
                    f'{self._beta} is unaceptable. Could resset to 5.')
                warnings.warn(
                    f'`{self._beta}` is unaceptable. Could resset to 5.')
                self._beta= 5
    @property 
    def ptol(self) :
        """ Tolerance parameter """
        return self._ptol 
    @ptol.setter 
    def ptol(self, ptol0): 
        """ Tolerance parameter must be different to zero and includes 
        between 0 and 1"""
        try : 
            self._ptol =float(ptol0)
        except Exception :
            raise TypeError ('Tolerance parameter `ptol` should be '
                             f'a float number not {type (ptol0)!r}.')
        else : 
            if 0 >= self._ptol >1: 
                self._logging.debug(f"Tolerance value `{self._ptol}` is "
                  "{'greater' if self._ptol >1 else 'is unacceptable value'}`."
                    "Could resset to 10%")
                warnings.warn(
                    f'Tolerance value `{self._ptol}` is unacceptable value.'
                    'Could resset to 10%')
                self._ptol = 0.1
    @property 
    def tres(self): 
        """ Input true resistivity"""
        return self._tres 
    @tres.setter 
    def tres(self, ttres):
        """ Convert Tres to log 10 resistivity """
        try : 
            self._tres =[np.log10(t) for t in ttres]
        except : 
            raise ValueError('Unable to convert TRES values') 
        
    @property 
    def build (self): 
        """ Trigger the NM build and return the NM building option """
        
        ntres ='True resistivity values'
        nln ='collected layer names (True)'
        mes =''.join([
            '{0} {1} not defined. Unable to triggered the NM construction. '
            'Please, provide the list/array of {2} of survey area.'])
         
        if self._b:
            if (self.tres and self.input_layers ) is None: 
                warnings.warn(mes.format(
                    'TRES and LN', 'are', ntres +'and'+nln))
                self._b=False 
            elif self.tres is None and self.input_layers is not None: 
                warnings.warn(mes.format('TRES', 'is', ntres))
                self._b=False 
            elif self.input_layers is None and self.tres is not None: 
                warnings.warn(mes.format('LN', 'is', nln))
                self._b=False 
                
            if not self._b:
                self._logging.debug ( "Build is set to TRUE, however,"
                    '{0}'.mes.format(
                        f'{"TRES" if self.tres is None else "LN"}',
                        'is', f'{ntres if self.tres is None else nln}')
                )
                
        if self._b: 
            self._createNM()
        
   
 
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
        def s__auto_rocks (listOfauto_rocks): 
            """ Automatick rocks collected during the step 3
            :param listOfauto_rocks: List of automatic rocks from differents
             subblocks. 
             
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
        
        iln =kws.pop('input_layers', None)
        tres =kws.pop('tres', None)
        subblocks =kws.pop('subblocks', None)
        disp= kws.pop('display_infos', True)
        n_epochs = kws.pop('n_epochs', None)
        hinfos =kws.pop('headerinfos',
                        ' Layers [auto=automatic]')
        if subblocks is not None: 
            self.subblocks = subblocks
        
        if iln is not None: 
            self.input_layers = iln 
        if tres is not None: 
            self.tres = tres 
        
        if crm is not None:
            self.crm = crm 
        if beta is not None: 
            self.beta = beta 
        if ptol is not None:
            self.ptol = ptol 
        if n_epochs is not None: 
            self.n_epochs = n_epochs 
            
        self.s0 , errors=[], []
        #step1 : SOFMINERROR 
        if TQDM : 
            pbar =tqdm.tqdm(total= 3,
                             ascii=True,unit='B',
                             desc ='geostrata', 
                             ncols =77)
            
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
                s3, autorock_properties= self._createAutoLayer(
                    subblocks=self.subblocks[ii], s0=self.s0[ii]  )
                arp_.append(autorock_properties)
                self.s0[ii]=s3       
        # Assembly the blocks 
        self.nm = np.concatenate((self.s0))
        self.z=self.nm[:, 0]
        self.nm = self.nm[:, 1:]
        
        if TQDM : 
            pbar.update(3)
            print(' process completed')
            pbar.close()
            
        # make site blocks 
        self.nmSites= makeBlockSites(x_nodes=self.model_x_nodes, 
                        station_location= self.station_location, 
                             block_model=self.nm )
        self.crmSites = makeBlockSites(x_nodes=self.model_x_nodes, 
                            station_location= self.station_location, 
                             block_model=self.model_res)

        #Update TRES and LN 
        gammaL, gammarho = s__auto_rocks(arp_) 
        
        if self.input_layers is not None: 
            print_layers = self.input_layers  + [ ' {0} (auto)'.format(l) 
                                                 for l in gammaL ]
            self.input_layers = self.input_layers + gammaL
        # keep the auto_layer found     
        self.auto_layers =gammaL
        self.tres = list(np.power(10,self._tres))  + list (np.power(10, 
                  np.array([float(rv) for rv in gammarho])))
        # display infos 
        if disp:
            display_infos(infos=print_layers,
                          header= hinfos)
        #STEP 4: Train ANN: (still in development) to predict your 
        #layer: No need to plot the NM 
        
        # copy main attributes for pseudostratigraphic plot purpose 
        import copy 
        for name , attrval in zip(['TRES', 'LNS'], 
                              [self.tres , self.input_layers]):
            setattr(self, name, copy.deepcopy(attrval))
        # memorize data 
        _ps_memory_management(self)
        
        return self.nm

        
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


    def _hardMinError(self, tres=None, subblocks=None, s0=None, ptol = None,
                         kind='linear', **kwargs ): 
        """The second step introduces the model function F=W∙Z  where W
        contains the weights of parameters number and Z is V×2 matrix 
        that contains a “bias” column. If the parameter number P equal to two, 
        the model function f(z)=∑_(p=1)^P▒〖w_(p-1) z^(p-1) 〗   becomes a
        linear function with 〖f_1〗^((1) ) (z)=  wz+r_0  with w_1=w and w_0=r_0
        he gradient descent algorithm  is used to find the best parameters w
        and r_0  that  minimizes the  MSE loss function  J .
        
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
            self._eta = eta 
        n_epochs =kwargs.pop('n_epochs', None)
        if n_epochs is not None: 
            self.n_epochs = n_epochs 
        kind = kwargs.pop('kind', None)
        if kind is not None:
            self._kind = kind 
        degree = kwargs.pop('degree', None) 
        if degree is not None: 
            self._degree = degree 
        
        buffer =self.ptol +1  #bufferr error 
        _z= s0[:, 0]
        s0 = s0[:, 1:].T

        subblocks=subblocks[:, 1:].T
        error =[]
        for ii in range(s0.shape[0]): # hnodes N
            F, *_= self.gradient_descent(z=_z,s=subblocks[ii,:],
                                         alpha= self._eta,
                                         n_epochs= self.n_epochs, 
                                         kind= self._kind)
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
        
    @staticmethod    
    def gradient_descent(z, s, alpha, n_epochs, **kws): 
        """ Gradient descent algorithm to  fit the best model parameter. 
        
        :param z: vertical nodes containing the values of depth V
        :param s: vertical vector containin the resistivity values 
        :param alpha: step descent parameter or learning rate. 
                    *Default* is ``0.01`
        :param n_epochs: number of iterations. *Default* is ``100``
                        Can be changed to other values
        :returns:
            - `F`: New model values with the best `W` parameters found.
            - `W`: vector containing the parameters fits 
            - `cost_history`: Containing the error at each Itiretaions. 
            
        :Example:
            
            >>> z= np.array([0, 6, 13, 20, 29 ,39, 49, 59, 69, 89, 109, 129, 
                             149, 179])
            >>> res= np.array( [1.59268,1.59268,2.64917,3.30592,3.76168,
                                4.09031,4.33606, 4.53951,4.71819,4.90838,
                  5.01096,5.0536,5.0655,5.06767])
            >>> fz, weights, cost_history = gradient_descent(z=z, s=res,
                                                 n_epochs=10,
                                                 alpha=1e-8,
                                                 degree=2)
            >>> import matplotlib.pyplot as plt 
            >>> plt.scatter (z, res)
            >>> plt.plot(z, fz)
        """
        kind_=kws.pop('kind', 'linear')
        kind_degree = kws.pop('degree', 1)
        
        if kind_degree >1 : kind_='poly'
        
        if kind_.lower() =='linear': 
            kind_degree = 1 
        elif kind_.lower().find('poly')>=0 : 
            if kind_degree <=1 :
                _logger.debug(
                    'The model function is set to `Polynomial`. '
                    'The degree must be greater than 1. Degree wil reset to 2.')
                warnings.warn('Polynomial degree must be greater than 1.'
                              'Value is ressetting to `2`.')
                kind_degree = 2
            try : 
                kind_degree= int(kind_degree)
            except Exception :
                raise ValueError(f'Could not `{kind_degree}` convert to integer.')
                
        
        def kindOfModel(degree, x, y) :
            """ Generate kind of model. If degree is``1`` The linear subset 
             function will use. If `degree` is greater than 2,  Matrix will 
             generate using the polynomail function.
             
            :param x: X values must be the vertical nodes values 
            :param y: S values must be the resistivity of subblocks at node x 
            
             """
            c= []
            deg = degree 
            w = np.zeros((degree+1, 1)) # initialize weights 
            
            def init_weights (x, y): 
                """ Init weights by calculating the scope of the function along 
                 the vertical nodes axis for each columns. """
                for j in range(x.shape[1]-1): 
                    a= (y.max()-y.min())/(x[:, j].max()-x[:, j].min())
                    w[j]=a
                w[-1] = y.mean()
                return w   # return weights 
        
            for i in range(degree):
                c.append(x ** deg)
                deg= deg -1 
        
            if len(c)> 1: 
                x= concat_array_from_list(c, concat_axis=1)
                x= np.concatenate((x, np.ones((x.shape[0], 1))), axis =1)
        
            else: x= np.vstack((x, np.ones(x.shape))).T # initialize z to V*2
        
            w= init_weights(x=x, y=y)
            return x, w  # Return the matrix x and the weights vector w 
        
        
        def model(Z, W): 
            """ Model function F= Z.W where `Z` id composed of vertical nodes 
            values and `bias` columns and `W` is weights numbers."""
            return Z.dot(W)
        
        # generate function with degree 
        Z, W = kindOfModel(degree=kind_degree,  x=z, y=s)
        
        # Compute the gradient descent 
        cost_history = np.zeros(n_epochs)
        s=s.reshape((s.shape[0], 1))
        
        for ii in range(n_epochs): 
            with np.errstate(all='ignore'): # rather than divide='warn'
                #https://numpy.org/devdocs/reference/generated/numpy.errstate.html
                W= W - (Z.T.dot(Z.dot(W)-s)/ Z.shape[0]) * alpha 
                cost_history[ii]= (1/ 2* Z.shape[0]) * np.sum((Z.dot(W) -s)**2)
            
        F= model(Z=Z, W=W)     # generate the new model with the best weights 
                 
        return F,W, cost_history
     
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

        self.zmodel = np.concatenate((self.geo_depth.reshape(
            self.geo_depth.shape[0], 1),  self.model_res), axis =1) 
                                    
        vv = self.zmodel[-1, 0] / self.beta 
        for ii, nodev in enumerate(self.zmodel[:, 0]): 
            if nodev >= vv: 
                npts = ii       # collect number of points got.
                break 
        self._subblocks =[]
        
        bp, jj =npts, 0
        if len(self.zmodel[:, 0]) <= npts: 
            self._subblocks.append(self.zmodel)
        else: 
            for ii , row in enumerate(self.zmodel) : 
                if ii == bp: 
                    _tp = self.zmodel[jj:ii, :]
                    self._subblocks.append(_tp )
                    bp +=npts
                    jj=ii
                    
                if len(self.zmodel[jj:, 0])<= npts: 
                    self._subblocks.append(self.zmodel[jj:, :])
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
        
    def _createAutoLayer(self, subblocks=None, s0=None,
                          ptol = None,**kws):
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
        db_properties = kws.pop('properties',['electrical_props', 
                                              '__description'] )
        tres = kws.pop('tres', None)
        disp = kws.pop('display_infos', False)
        hinfos = kws.pop('header', 'Automatic layers')
        
        if tres is not None :
            self.tres = tres 
        if ptol is not None: 
            self.ptol = ptol 
        
        def _findGeostructures(_res): 
            """ Find the layer from database and keep the ceiled value of 
            `_res` calculated resistivities"""
            
            structures = self.get_structure(_res)
            if len(structures) !=0 or structures is not None:
                if structures[0].find('/')>=0 : 
                    ln = structures[0].split('/')[0].lower() 
                else: ln = structures[0].lower()
                return ln, _res
            else: 
                valEpropsNames = self._getProperties(db_properties)
                indeprops = db_properties.index('electrical_props')
                for ii, elecp_value  in enumerate(valEpropsNames[indeprops]): 
                    if elecp_value ==0.: continue 
                    elif elecp_value !=0 : 
                        try : 
                            iter(elecp_value)
                        except : pass 
                        else : 
                            if  min(elecp_value)<= _res<= max(elecp_value):
                                ln= valEpropsNames[indeprops][ii]
                                return ln, _res
                    
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
            
        _z= s0[:, 0]
        s0 = s0[:, 1:].T
        _temptres , _templn =[], []
        subblocks=subblocks[:, 1:].T

        for ii in range(s0.shape[0]): # hnodes N
            for jj in range(s0.shape[1]): # znodes V
                if s0[ii, jj] ==0. : 
                    lnames, lcres =_findGeostructures(
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
        if disp:
            display_infos(infos=self.input_layers,
                          header= hinfos)
        
        return  s0, auto_rocks_names_res
        
    @staticmethod
    def _getProperties(properties =['electrical_props', '__description'], 
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
                                  input_layers=input_layer_names)
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
        if self.nm is None: 
            self._createNM()  
                
        if kind =='nm':
            data = self.nmSites 
        if kind =='crm': 
            data = self.crmSites

      # compute model_misfit
        if misfit_G is True : 
            if kind =='crm': 
                warnings.warn(
                    "By default, the plot should be the stratigraphic"
                    " misfit<misfit_G>.")
    
            self._logging.info('Visualize the stratigraphic misfit.')
            data = compute_misfit(rawb=self.crmSites , 
                                  newb= self.nmSites, 
                                  percent = misfit_percentage)
            
            print('{0:-^77}'.format('StrataMisfit info'))
            print('** {0:<37} {1} {2} {3}'.format(
                'Misfit max ','=',data.max()*100., '%' ))                      
            print('** {0:<37} {1} {2} {3}'.format(
                'Misfit min','=',data.min()*100., '%' ))                          
            print('-'*77)
            
        warnings.warn(
            f'Data stored from {m_!r} should be moved on binary drive and'
            ' method arguments should be keywordly only.', FutureWarning)
        
        return (data, self.station_names, self.station_location,
            self.geo_depth, self.doi, depth_scale, self.model_rms, 
            self.model_roughness, misfit_G ) 
    

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
        ...              input_layers=LNS,**INVERS_KWS)
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
            stns = ["S{:02}".format(i) for i in range(obj.nmSites.shape[1])]
            obj._logging.error('None station is found. Please select one station'
                                f' between {smart_format(stns)}')
            warnings.warn("NoneType can not be read as station name."
                            " Please provide your station name. list of sites" 
                             f" are {smart_format(stns)}")
            raise ValueError("NoneType can not be read as station name."
                             " Please provide your station name.")
        
        if  obj.nmSites is None: 
                obj._createNM()
        
        try : 
            id0= int(station.lower().replace('s', ''))
        except : 
            id_ =assert_station(id= station, nm = obj.nmSites)
            station_ = 'S{0:02}'.format(id_)
        else : 
            id_ =assert_station(id= id0 + 1, nm = obj.nmSites)
            station_ = 'S{0:02}'.format(id0)
        obj.logS = obj.nmSites[:, id_] 
     
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
            warnings.warn(msg)
            
            pslns, pstres, ps_lnstres =  fit_tres(
                                            obj.LNS, obj.TRES, 
                                            obj.auto_layers)
        # now build the fitting rocks 
        fitted_rocks =fit_rocks(logS_array= obj.nmSites[:,id_],
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
                                obj.z, obj.logS)
        obj.log_thicknesses, obj.log_layers,\
            obj.coverall = get_s_thicknesses( 
            zg, sg,display_s= display_s, station = station_ )
            
        # set the dfault layers properties hatch and colors from database
        obj.hatch , obj.color =  fit_default_layer_properties(
            obj.log_layers) 
        
        return obj
    
    @staticmethod
    def plotPseudostratigraphic(station, zoom=None, annotate_kws=None, **kws):
        """ Build the pseudostratigraphic log. 
        :param station: station to visualize the plot.
        :param zoom: float  represented as visualization ratio
            ex: 0.25 --> 25% view from top =0.to 0.25* investigation depth 
            or a list composed of list [top, bottom].
        
        :Example: 
            >>> input_resistivity_values =[10, 66,  700, 1000, 1500,  2000, 
                                    3000, 7000, 15000 ] 
            >>> input_layer_names =['river water', 'fracture zone', 'granite']
            # Run it to create your model block alfter that you can only use 
            #  `plotPseudostratigraphic` only
            # >>> obj= quick_read_geomodel(lns = input_layer_names, 
            #                             tres = input_resistivity_values)
            >>> plotPseudostratigraphic(station ='S00')
        
        """
        
        if annotate_kws is None: annotate_kws = {'fontsize':12}
        if not isinstance(annotate_kws, dict):
            annotate_kws=dict()
        obj = _ps_memory_management(option ='get' )  
        obj = GeoStrataModel._strataPropertiesOfSite (obj,station=station,
                                                       **kws )
        # plot the logs with attributes 
        pseudostratigraphic_log (obj.log_thicknesses, obj.log_layers,station,
                                    hatch =obj.hatch ,zoom=zoom,
                                    color =obj.color, **annotate_kws)
        print_running_line_prop(obj)
        
        return obj 
    
def _ps_memory_management(obj=None, option='set'): 
    """ Manage the running times for stratigraphic model construction.
    
    The script allows to avoid running several times the GeoStrataModel model
    construction to retrieve the pseudostratigraphic (PS) log at each station.  
    It memorizes the model data for the first run and used it when calling it
    to  visualize the PS log at each station. Be aware to edit this script.
    """
    memory, memorypath='__memory.pkl', 'watex/etc'
    mkeys= ('set', 'get', 'recover', 'fetch', set)
    if option not in mkeys: 
        raise ValueError('Wrong `option` argument. Acceptable '
                         f'values are  {smart_format(mkeys[:-1])}.')
        
    if option in ('set', set): 
        if obj is None: 
            raise TypeError('NoneType object can not be set.') 
        psobj_token = __build_ps__token(obj)
        data = (psobj_token, list(obj.__dict__.items()))
        serialize_data ( data, memory, savepath= memorypath )

        return 
    
    elif option in ('get', 'recover', 'fetch'): 
        memory_exists =  os.path.isfile(os.path.join(memorypath, memory))
        if not memory_exists: 
            _logger.error('No memory found. Run the GeoStrataModel class '
                          'beforehand to create your first model.')
            warnings.warn("No memory found. You need to build your "
                          " GeoStrataModel model by running the class first.")
            raise  MemoryError("Memory not found. Use the GeoStrataModel class to "
                               "create your model first.")
        psobj_token, data_ = load_serialized_data(
            os.path.join(memorypath, memory))
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
  

def display_infos(infos, **kws):
    """ Display unique element on list of array infos.
    
    :param infos: Iterable object to display. 
    :param header: Change the `header` to other names. 
    :Example: 
        >>> from watex.geology.stratigraphic import display_infos
        >>> ipts= ['river water', 'fracture zone', 'granite', 'gravel',
             'sedimentary rocks', 'massive sulphide', 'igneous rocks', 
             'gravel', 'sedimentary rocks']
        >>> display_infos('infos= ipts,header='TestAutoRocks', 
                          size =77, inline='~')
    """

    inline =kws.pop('inline', '-')
    size =kws.pop('size', 70)
    header =kws.pop('header', 'Automatic rocks')

    if isinstance(infos, str ): 
        infos =[infos]
        
    infos = list(set(infos))
    print(inline * size )
    mes= '{0}({1:02})'.format(header.capitalize(),
                                  len(infos))
    mes = '{0:^70}'.format(mes)
    print(mes)
    print(inline * size )
    am=''
    for ii in range(len(infos)): 
        if (ii+1) %2 ==0: 
            am = am + '{0:>4}.{1:<30}'.format(ii+1, infos[ii].capitalize())
            print(am)
            am=''
        else: 
            am ='{0:>4}.{1:<30}'.format(ii+1, infos[ii].capitalize())
            if ii ==len(infos)-1: 
                print(am)
    print(inline * size )
    
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
    
    __gammaProps = GeoStrataModel._getProperties(dbproperties_)
    
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
    """ Build a special token for each GeoStrataModel model. Please don't 
    edit anything here. Force editing is your own risk."""
    import random 
    random.seed(42)
    __c =''.join([ i for i in  [''.join([str(c) for c in obj.crmSites.shape]), 
     ''.join([str(n) for n in obj.nmSites.shape]),
    ''.join([l for l in obj.input_layers]) + str(len(obj.input_layers)), 
    str(len(obj.tres))] + [''.join(
        [str(i) for i in [obj._eta, obj.beta, obj.doi,obj.n_epochs,
                          obj.ptol, str(obj.z.max())]])]])
    __c = ''.join(random.sample(__c, len(__c))).replace(' ', '')                                               
    n= ''.join([str(getattr(obj, f'{l}'+'_fn'))
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
    _gammaRES, _gammaLN = GeoStrataModel._getProperties(**kws)

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
    pseudo_lns = [a [0] for a in newTRES] + rlns 
    pseudo_tres = [b[1] for b in newTRES] + rtres 
    newTRES += [(a, b) for a , b in zip(rlns, rtres)]
    
    return pseudo_lns , pseudo_tres , newTRES 



def quick_read_geomodel(lns=None, tres=None):
    """Quick read and build the geostratigraphy model (NM) 
    
    :param lns: list of input layers 
    :param tres: list of input true resistivity values 
    
    :Example: 
        >>> import watex.geology.stratigraphic as GM 
        >>> obj= GM.quick_read_geomodel()
        >>> GC.fit_tres(obj.input_layers, obj.tres, obj.auto_layer_names)
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
                      input_layers=lns,**INVERS_KWS)
    geosObj._createNM()
    
    return geosObj 

GeoStrataModel.__doc__="""\
Create a stratigraphic model from inversion models blocks. 

The Inversion model block is two dimensional array of shape 
(n_vertical_nodes, n_horizontal_nodes ). Can use external packages 
to build blocks and provide the 2Dblock into `crm` parameter. 

The challenge of this class  is firstly to delineate with much 
accuracy the existing layer boundary (top and bottom) and secondly,
to predict the stratigraphy log before the drilling operations at each 
station. Moreover, it’s a better way to select the right drilling location
and also to estimate the thickness of existing layer such as water table 
layer as well as to figure out the water reservoir rock in the case of 
groundwater exploration. 

Note that if the model blocks is build from externam softwares. You may as  
in the keywordsr argumments of `GeoStrataModel` the following attributes: 
    
    - model_res : 2D resitivity model of (n_vertical_nodes, n_horizontal_nodes)
        If `crm` is given , no need to provided it. 
    - geo_depth: Is the depth the surface to the bottom of each layer that 
        composed the pseudo-boreholes. Note the N-vertical nodes values 
    - input_resistivities: list of input resistivities. If the `tres` is passed 
        not need to given. 
        
Parameters 
----------
crm : ndarray of shape(n_vertical_nodes, n_horizontal_nodes ),  
    Array-like of inversion two dimensional model blocks. Note that 
    the `n_vertical_nodes` is the node from the surface to the depth 
    while `n_horizontal_nodes` must include the station location 
    (sites/stations) 
    
beta:  int,                
        Value to  divide into the CRM blocks to improve 
        the computation times. default is`5`                               
n_epochs:  int,  
        Number of iterations. default is `100`
tres:  array_like, 
        Truth values of resistivities. Refer to 
        :class:`~.geodrill.Geodrill` for more details
ptols: float,   
        Existing tolerance error between the `tres` values given and 
        the calculated resistivity in `crm` 
input_layers: list or array_like  
        True input_layers names : geological 
        informations of collected in the area.
            
kind: str         
    Kind of model function to compute the best fit model to replace the
    value in `crm` . Can be 'linear' or 'polynomial'. if `polynomial` is 
    set, specify the `degree. Default is 'linear'. 
    
alpha: float , 
    Learning rate for gradient descent computing.  *Default* is ``1e+4`` 
    for linear. If `kind` is  set to `polynomial` the default value should 
    be `1e-8`. 
degree: int,
     Polynomial function degree to implement gradient descent algorithm. 
     If `kind` is set to `Polynomial` the default `degree` is ``3.`` and 
     details sequences 
**nm**:  ndarray     
    The NM matrix with the same dimension with `crm` model blocks. 


Examples
----------
>>> from watex.geology.stratigraphic import GeoStrataModel 
>>> # Works with occam2d inversion files if 'pycsamt' or 'mtpy' is installed
>>> # will call the module Geodrill from pycsamt to make occam2d 2D resistivity
>>> # block for our demo. It presumes pycsamt is installed. 
>>> from pycsamt.geodrill.geocore import Geodrill 
>>> path=r'data/inversfiles/inver_res/K4' # path to inversion files 
>>> inversion_files = {'model_fn':'Occam2DModel', 
...                   'mesh_fn': 'Occam2DMesh',
...                    "iter_fn":'ITER27.iter',
...                   'data_fn':'OccamDataFile.dat'
...                    }
>>> input_resistivity_values =[10, 66, 70, 180, 1000, 2000, 
...                           3000, 50, 7] 
>>> input_resistivity_values =[10, 66, 70, 180, 1000, 2000, 
...                               3000, 7000, 15000 ] 
>>> input_layer_names =['river water', 'fracture zone', 'granite']
>>> inversion_files = {key:os.path.join(path, vv) for key,
                vv in inversion_files.items()}
>>> gdrill= Geodrill (**inversion_files, 
                     input_resistivities=input_resistivity_values
                     )
>>> # we can collect the 'model_res' and 'geo_depth_attributes' from 
>>> # `gdrill object` and passed to 'GeoStrataModel' fit method as 
>>> geosObj = GeoStrataModel(ptol =0.1).fit(crm = model_res , 
                     input_resistivities=gdrill.input_resistivity_values
                     geo_depth= gdrill.geo_depth )
>>> zmodel = geosObj._zmodel
>>> geosObj.nm # resistivity 2D model block is constructed 

Notes
------
Modules work properly with occam2d inversion files if 'pycsamt' or 'mtpy' is 
installed and  inherits the `Base package` which works with occam2d  model.
Occam2d inversion files are also acceptables for building model blocks. 
However the MODEM resistivity files development is still ongoing 

"""
# def assert_len_layers_with_resistivities(
#         real_layer_names:str or list, real_layer_resistivities: float or list ): 
#     """
#     Assert the length of of the real resistivites with their
#     corresponding layers. If the length of resistivities is larger than 
#     the layer's names list of array, the best the remained resistivities
#     should be topped up to match the same length. Otherwise if the length 
#     of layers is larger than the resistivities array or list, layer'length
#     should be reduced to fit the length of the given resistivities.
    
#     Parameters
#     ----------
#         * real_layer_names: array_like, list 
#                     list of input layer names as real 
#                     layers names encountered in area 
                    
#         * real_layer_resistivities :array_like , list 
#                     list of resistivities get on survey area
                
#     Returns 
#     --------
#         list 
#             real_layer_names,  new list of input layers 
#     """
#     # for consistency put on string if user provide a digit
#     real_layer_names =[str(ly) for ly in real_layer_names]      
    
#     if len(real_layer_resistivities) ==len(real_layer_names): 
#         return real_layer_names
    
#     elif len(real_layer_resistivities) > len(real_layer_names): 
#          # get the last value of resistivities  to match the structures
#          # names and its resistivities 
#         sec_res = real_layer_resistivities[len(real_layer_names):]        
#        # fetch the name of structure 
#         geos =Geodrill.get_structure(resistivities_range=sec_res) 
#         if len(geos)>1 : tm = 's'
#         else :tm =''
#         print(f"---> Temporar{'ies' if tm=='s' else 'y'} "
#               f"{len(geos)} geological struture{tm}."
#               f" {'were' if tm =='s' else 'was'} added."
#               " Uncertained layers should be ignored.")
        
#         real_layer_names.extend(geos)       
#         return real_layer_names 
#     elif len(real_layer_names) > len(real_layer_resistivities): 
#         real_layer_names = real_layer_names[:len(real_layer_resistivities)]        
#         return real_layer_names
    