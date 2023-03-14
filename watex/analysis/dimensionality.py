# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Created on Wed Sep 22 15:04:52 2021
"""
Reducers 
============
Reduce dimension for data visualisation.

Reduce number of dimension down to two (or to three) for instance, make  
it possible to plot high-dimension training set on the graph and often
gain some important insights by visually detecting patterns, such as 
clusters.

"""
from __future__ import annotations 
import os
import warnings
import numpy as np
import pandas as pd 
from sklearn.decomposition import (
    PCA, 
    IncrementalPCA, 
    KernelPCA
    )
from .._typing import (
    Any,
    Dict, 
    Optional, 
    ArrayLike, 
    NDArray, 
    DataFrame,
    Sub
    )

from .._watexlog import watexlog

_logger = watexlog().get_watex_logger(__name__)

__all__ = [
    'nPCA', 
    'kPCA', 
    'LLE', 
    'iPCA', 
    'get_component_with_most_variance',
    'plot_projection', 
    'find_features_importances', 
]
  
def nPCA(
    X: NDArray | DataFrame,
    n_components: float | int =None, 
    *, 
    view: bool =False, 
    return_X:bool=True, 
    plot_kws: Dict[str, Any] =None,
    n_axes: int =None, 
    **pca_kws
    )-> NDArray| 'nPCA': 
    """Normal Principal Components analysis (PCA)
    
    PCA is by far the most popular dimensional reduction algorithm. First it 
    identifies the hyperplane that lies closest to the data and project it 
    to the data onto it.
    
    Parameters 
    ------------
    X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        Training set; Denotes data that is observed at training and 
        prediction time, used as independent variables in learning. 
        When a matrix, each sample may be represented by a feature vector, 
        or a vector of precomputed (dis)similarity with each training 
        sample. :code:`X` may also not be a matrix, and may require a 
        feature extractor or a pairwise metric to turn it into one  before 
        learning a model.
    
    n_components: int, optional 
        Number of dimension to preserve. If`n_components` is ranged between 
        float 0. to 1., it indicated the number of variance ratio to preserve. 
        If ``None`` as default value the number of variance to preserve is 
        ``95%``.
            
    return_X: bool, default =True , 
        return the train set transformed with most representative varaince 
        ratio. 
        
    view: bool,default=False,  
        Plot the explained varaince as a function  of number of dimension. 
        
    n_axes: int, optional, 
        Number of importance components to retrieve the variance ratio. 
        If ``None`` the features importance is computed using the cumulative 
        variance representative of 95% .
    
    pca_kws: dict, 
        Additional matplotlib.pyplot keywords arguments passed to 
        :class:`sklearn.decomposition.PCA`
        
    Returns
    --------
    X or `nPCA` object, 
        The transformed training set or the PCA container attributes for 
        plotting purpose. 

    Examples
    ---------
    >>> from watex.analysis.dimensionality import nPCA
    >>> from watex.datasets import fetch_data
    >>> X, _= fetch_data('Bagoue analysed dataset')
    >>> pca = nPCA(X, 0.95, n_axes =3, return_X=False)
    >>> pca.components_
    >>> pca.feature_importances_
    """
    obj = type ('nPCA', (), dict())
    def findFeaturesImportances(
            fnames: ArrayLike,
            components: float |int ,
            n_axes: int =2
            )-> ArrayLike: 
        """ Retreive the features importance with variance ratio.
        
        :param fnames: array_like of feature's names
        :param components: pca components on different axes 
        """
        pc =list()
        if components.shape[0] < n_axes : 
            
            warnings.warn(f'Retrieved axes {n_axes!r} no more than'
                          f' {components.shape[0]!r}. Reset to'
                          f'{components.shape[0]!r}', UserWarning)
            n_axes = int(components.shape[0])
        
        for i in range(n_axes): 
            # reverse from higher values to lower 
            index = np.argsort(abs(components[i, :]))
            comp_sorted = components[i, :][index][::-1]
            numf = fnames [index][::-1]
            pc.append((f'pc{i+1}', numf, comp_sorted))
            
        return pc 
    
    if n_components is None: 
        # choose the right number of dimension that add up to 
        # sufficiently large proportion of the variance 0.95%
        pca=PCA(**pca_kws)
        pca.fit(X)
        cumsum =np.cumsum( pca.explained_variance_ratio_ )
        # d= np.argmax(cumsum >=0.95) +1 # for index 
        
        # we can set the n_components =d then run pca again or set the 
        # value of n_components betwen 0. to 1. indicating the ratio of 
        # the variance we wish to preserve.
        
    X0= X.copy()
    pca = PCA(n_components=n_components, **pca_kws)
    X= pca.fit_transform(X) ; obj.X=X # X_reduced = pca.fit_transform(X)
  
    if n_components is not None: 
        cumsum = np.cumsum(pca.explained_variance_ratio_ )
    
    if view: 
        import matplotlib.pyplot as plt
        
        if plot_kws is None: 
            plot_kws ={'label':'Explained variance as a function of the'
                       ' number of dimension' }
        plt.plot(cumsum)
        # plt.plot(np.full((cumsum.shape), 0.95),
        #          # np.zeros_like(cumsum),
        #          ls =':', c='r')
        plt.xlabel('N-Dimensions')
        plt.ylabel('Explained variance')
        plt.title('Explained variance as a function of the'
                    ' number of dimension')
        plt.show()
        
    # make introspection and set the all pca attributes to self.
    for key, value in  pca.__dict__.items(): 
        setattr(obj, key, value)
    
    if n_axes is None : 
        obj.n_axes = pca.n_components_
    else : 
        setattr(obj, 'n_axes', n_axes)
        
    # get the features importance and features names if 
    if isinstance (X0, pd.DataFrame): 
        obj.feature_importances_= findFeaturesImportances(
                                        np.array(list(X0.columns)), 
                                        pca.components_, 
                                        obj.n_axes)

    return X if return_X else  obj  
    
def iPCA(
    X: NDArray | DataFrame,
    n_components: float | int =None,
    *, 
    view: bool =False, 
    n_batches: int =None,
    return_X:bool=True, 
    store_in_binary_file: bool =False,
    filename: Optional[str]=None,
    **ipca_kws
 )-> NDArray| 'iPCA': 
    """ Incremental PCA 
    
    `iPCA` allows to split the trainsing set into mini-batches and feed 
    algorithm one mini-batch at a time. 
     
    Once problem with the preceeding implementation of PCA is that 
    requires the whole training set to fit in memory in order of the SVD
    algorithm to run. This is usefull for large training sets, and also 
    applying PCA online(i.e, on the fly as a new instance arrive)
     
    Parameters 
    -------------
    X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        Training set; Denotes data that is observed at training and 
        prediction time, used as independent variables in learning. 
        When a matrix, each sample may be represented by a feature vector, 
        or a vector of precomputed (dis)similarity with each training 
        sample. :code:`X` may also not be a matrix, and may require a 
        feature extractor or a pairwise metric to turn it into one  before 
        learning a model.
    
    n_components: int, optional 
        Number of dimension to preserve. If`n_components` is ranged between 
        float 0. to 1., it indicated the number of variance ratio to preserve. 
        If ``None`` as default value the number of variance to preserve is 
        ``95%``.
        
    n_batches: int, optional
        Number of batches to split the training set.
    
    store_in_binary_file: bool, default=False 
        Alternatively, we used numpy` memmap` class to manipulate a large 
        array stored in a binary file on disk as if it were entirely in 
        memory. The class load only the data it need in memory when it need
        its.
    
    filename: str,optional 
        Default binary filename to store in a binary file in  a disk.
        
    return_X: bool, default =True , 
        return the train set transformed with most representative varaince 
        ratio. 
        
    view: bool,default=False,  
        Plot the explained varaince as a function  of number of dimension. 
        
    ipca_kws: dict, 
        Additional keyword arguments passed to 
        :class:`sklearn.decomposition.IncrementalPCA`
    
    Returns 
    ----------
    X (NDArray) or `iPCA` object, 
        The transformed training set or the iPCA container attributes for 
        plotting purposes. 
    
    Examples
    ---------
    >>> from watex.analysis.dimensionality import iPCA
    >>> from watex.datasets import fetch_data 
    >>> X, _=fetch_data('Bagoue analysed data')
    >>> Xtransf = iPCA(X,n_components=None,n_batches=100, view=True)
    
    """
    obj = type ('iPCA', (), dict())
    X0=X.copy()
    if n_components is None: 
        n_components= get_component_with_most_variance(X) 
        if n_batches is None: 
            raise TypeError('NoneType can not be a number of batches.')
        if n_components > (len(X)//n_batches +1): 
            warnings.warn(f'n_components=`{n_components}` must be less '
                             'or equal to the batch number of samples='
                             f'`{len(X0)//n_batches +1}`. n_components is'
                             f' set to {len(X0)//n_batches}')
            
            n_components = len(X)//n_batches
            _logger.debug(
                f"n_components is reset to ={len(X0)//n_batches!r}")
            
    inc_pcaObj = IncrementalPCA(n_components =n_components, 
                                **ipca_kws)
    for X_batch in np.array_split(X0, n_batches):
        inc_pcaObj.partial_fit(X_batch)
    
    X= inc_pcaObj.transform(X0)
    
    if store_in_binary_file: 
        if not (filename or os.path.isfile (filename)): 
            warnings.warn('Need a binary filename stored in disk of '
                          'in memory.')
            _logger.error(
                'Need a binary filename stored in disk of in memory.')
            raise FileNotFoundError('None binary filename found.')

        X_mm = np.memmap(filename,
                         dtype= 'float32',
                         mode='readonly', 
                         shape=X0.shape)
        batch_size = X0.shape[0]//n_batches
        inc_pcaObj = IncrementalPCA(
            n_components =n_components,
            batch_size= batch_size,
            **ipca_kws)
        
        X= inc_pcaObj.fit(X_mm)
        
    obj.X=X # set X attributes 
    make_introspection(obj, inc_pcaObj)
    setattr(obj, 'n_axes', getattr(obj, 'n_components_'))
    # get the features importance and features names
    if isinstance(X0, pd.DataFrame):
        pca_components_= getattr(obj, 'components_')
        obj.feature_importances_= find_features_importances(
                                        np.array(list(X0.columns)), 
                                        pca_components_, 
                                        obj.n_axes)
    if view : 
        plot_projection(obj, obj.n_components )
        
    return X if return_X else obj  

def kPCA(
    X: NDArray | DataFrame,
    n_components: float |int =None,
    *, 
    return_X:bool=True, 
    kernel: str ='rbf',
    reconstruct_pre_image: bool =False,
    **kpca_kws
)-> NDArray | 'kPCA': 
    """Kernel PCA 
    
    `kPCA` performs complex nonlinear projections for dimentionality
    reduction.
    
    Commonly the kernel tricks is a mathematically technique that implicitly
    maps instances into a very high-dimensionality space(called the feature
    space), enabling non linear classification or regression with SVMs. 
    Recall that a linear decision boundary in the high dimensional 
    feature space corresponds to a complex non-linear decison boundary
    in the original space.
    
    Parameters 
    -------------
    X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        Training set; Denotes data that is observed at training and 
        prediction time, used as independent variables in learning. 
        When a matrix, each sample may be represented by a feature vector, 
        or a vector of precomputed (dis)similarity with each training 
        sample. :code:`X` may also not be a matrix, and may require a 
        feature extractor or a pairwise metric to turn it into one  before 
        learning a model.
    
    n_components: int, optional 
        Number of dimension to preserve. If`n_components` is ranged between 
        float 0. to 1., it indicated the number of variance ratio to preserve. 
        If ``None`` as default value the number of variance to preserve is 
        ``95%``.
        
    return_X: bool, default =True , 
        return the train set transformed with most representative varaince 
        ratio. 
        
    kernel: {'linear', 'poly', \
            'rbf', 'sigmoid', 'cosine', 'precomputed'}, default='rbf'
        Kernel used for PCA.
        
    kpca_kws: dict, 
        Additional keyword arguments passed to 
        :class:`sklearn.decomposition.KernelPCA`
    
    Returns 
    ----------
    X (NDArray) or `kPCA` object, 
        The transformed training set or the kPCA container attributes for 
        plotting purposes. 
        
    Examples
    ----------
    >>> from watex.analysis.dimensionality import kPCA
    >>> from watex.datasets import fetch_data 
    >>> X, _=fetch_data('Bagoue analysis data')
    >>> Xtransf=kPCA(X,n_components=None,kernel='rbf', 
                                gamma=0.04, view=True)
    """
    obj = type ('kPCA', (), {})
    if n_components is None: 
       n_components= get_component_with_most_variance(X) 
    Xr= X.copy() 
    kpcaObj = KernelPCA(n_components=n_components, kernel=kernel, 
                        fit_inverse_transform =reconstruct_pre_image,
                        **kpca_kws)

    obj.X= kpcaObj.fit_transform(X)
    
    if reconstruct_pre_image:
        obj.X_preimage= kpcaObj.inverse_transform(obj.X)
        # then compute the reconstruction premimage error
        from sklearn.metrics import mean_squared_error
        obj.X_preimage_error = mean_squared_error(Xr, obj.X_preimage)
        
    obj.X=X 
    # populate attributes inherits from kpca object
    make_introspection(obj, kpcaObj)
    # set axes and features importances
    set_axes_and_feature_importances(obj, Xr)
    

    return obj.X if return_X else obj 
    
def LLE(
    X: NDArray | DataFrame,
    n_components: float |int =None,
    *,
    return_X:bool=True, 
    n_neighbors: int=5, 
    **lle_kws
)->NDArray | 'LLE': 
    """ Locally Linear Embedding(LLE) 
    
    `LLE` is nonlinear dimensinality reduction based on closest neighbors 
    (c.n).
    
    LLE is another powerfull non linear dimensionality reduction(NLDR)
    technique. It is Manifold Learning technique that does not rely
    on projections like `PCA`. In a nutshell, works by first measurement
    how each training instance library lineraly relates to its closest 
    neighbors(c.n.), and then looking for a low-dimensional representation 
    of the training set where these local relationships are best preserved
    (more details shortly).Using LLE yields good resuls especially when 
    makes it particularly good at unrolling twisted manifolds, especially
    when there is too much noise.
    
    Parameters
    ----------
    X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        Training set; Denotes data that is observed at training and 
        prediction time, used as independent variables in learning. 
        When a matrix, each sample may be represented by a feature vector, 
        or a vector of precomputed (dis)similarity with each training 
        sample. :code:`X` may also not be a matrix, and may require a 
        feature extractor or a pairwise metric to turn it into one  before 
        learning a model.
    
    n_components: int, optional 
        Number of dimension to preserve. If`n_components` is ranged between 
        float 0. to 1., it indicated the number of variance ratio to preserve. 
        If ``None`` as default value the number of variance to preserve is 
        ``95%``.

    n_neighbors : int, default=5
        Number of neighbors to consider for each point.
            
    return_X: bool, default =True , 
        return the train set transformed with most representative varaince 
        ratio. 
    lle_kws: dict, 
        Additional keyword arguments passed to 
        :class:`sklearn.decomposition.LocallyLinearEmbedding`. 
        
    Returns 
    ----------
    X (NDArray) or `LLE` object, 
        The transformed training set or the LLE container attributes for 
        plotting purposes. 
         
    References
    -----------
    Gokhan H. Bakir, Jason Wetson and Bernhard Scholkoft, 2004;
    "Learning to Find Pre-images";Tubingen, Germany:Max Planck Institute
    for Biological Cybernetics.
    
    S. Roweis, L.Saul, 2000, Nonlinear Dimensionality Reduction by
    Loccally Linear Embedding.
    
    Notes
    ------
    Scikit-Learn used the algorithm based on Kernel Ridge Regression
         
    Example
    -------
    >>> from watex.analysis.dimensionality import LLE
    >>> from watex.datasets import fetch_data 
    >>> X, _=fetch_data('Bagoue analysed data')
    >>> lle_kws ={
    ...    'n_components': 4, 
    ...    "n_neighbors": 5}
    >>> Xtransf=LLE(X,**lle_kws)
    
    """
    obj=type ('LLE', (), dict())
    from sklearn.manifold import LocallyLinearEmbedding
    
    if n_components is None: 
       n_components= get_component_with_most_variance(X) 
    lleObj =LocallyLinearEmbedding(n_components=n_components, 
                                    n_neighbors=n_neighbors,**lle_kws)
    X= lleObj.fit_transform(X);  obj.X=X 
    
     # populate attributes inherits from kpca object
    make_introspection(obj, lleObj)
    # set axes and features importances
    return X if return_X else obj            
 
def make_introspection(
        Obj: object ,
        subObj: Sub[object]
        )-> None: 
    """ Make introspection by using the attributes of instance created to 
    populate the new classes created.
    
    :param Obj: callable 
        New object to fully inherits of `subObject` attributes.
        
    :param subObj: Callable 
        Instance created.
    """
    # make introspection and set the all pca attributes to self.
    for key, value in  subObj.__dict__.items(): 
        setattr(Obj, key, value)
        
def find_features_importances(
        fnames: ArrayLike,
        components: float | int,
        n_axes: int =2
        )-> ArrayLike: 
    """ Retreive the features importance with variance ratio.
    :param fnames: array_like of feature's names
    :param components: pca components on different axes 
    """
    pc =list()
    if components.shape[0] < n_axes : 
        
        warnings.warn(f"Retrieved axes {n_axes!r} no more than"
                      f" {components.shape[0]!r}. Reset to"
                      f"{components.shape[0]!r}", UserWarning)
        n_axes = int(components.shape[0])
    
    for i in range(n_axes): 
        # reverse from higher values to lower 
        index = np.argsort(abs(components[i, :]))
        comp_sorted = components[i, :][index][::-1]
        numf = fnames [index][::-1]
        pc.append((f'pc{i+1}', numf, comp_sorted))
        
    return pc 

def plot_projection(
        self,
        n_components: float| int =None,
        **plot_kws
        )-> object | None: 
    """Quick plot the N-Dimension VS explained variance Ratio.
    :param n_components: pca components on different axes 
    """
    if n_components is None: 
        warnings.warn('NoneType <n_components> could not plot projection.')
        return 
    
    try: 
        cumsum = np.cumsum(
            getattr(self,'explained_variance_ratio_' ))
    except AttributeError:
        from pprint import pprint 
        obj_name = None
        if hasattr(self, 'kernel'): 
            obj_name ='KernelPCA'
        elif hasattr(self, 'n_neighbors') and hasattr(self, 'nbrs_'): 
            obj_name ='LoccallyLinearEmbedding'
            
        if obj_name is not None:
            warnings.warn(
                f"{obj_name!r} has no attribute 'explained_variance_ratio_'"
                  ". Could not plot projection according to a variance ratio.",
                  UserWarning)
            _logger.debug(f"{self.__class__.__name__!r} inherits from "
                          f"{obj_name!r} attributes and has no attribute"
                          "'components_")
        setattr(self, 'explained_variance_ratio_', None)
            
        pprint("KernelPCA has no attribute  called 'explained_variance_ratio_'"
               ". Could not plot <N-dimension vs explained variance ratio>"
               )
        return self

    import matplotlib.pyplot as plt

    plt.plot(cumsum, **plot_kws)
    # plt.plot(np.full((cumsum.shape), 0.95),
    #          # np.zeros_like(cumsum),
    #          ls =':', c='r')
    plt.xlabel('N-Dimensions')
    plt.ylabel('Explained Variance')
    plt.title('Explained variance as a function of the'
                ' number of dimension')
    plt.show()

def get_component_with_most_variance(
        X: NDArray | DataFrame,
        **pca_kws
        )->ArrayLike:
    """ Get the number of component with 95% ratio. 
    
    :param X: Training set.
    :param pca_kws: additional pca  keywords arguments.
    """
    # choose the right number of dimension that add up to 
    # sufficiently large proportion of the variance 0.95%
    warnings.warn('Number of components is None. By default n_components'
                  ' is reset to the most variance 95%.')
    _logger.info('`n_components` is not given. By default the number of '
                  'component is reset to 95% variance in the data.')
    pca=PCA(**pca_kws)
    pca.fit(X)
    cumsum =np.cumsum( pca.explained_variance_ratio_ )
    d= np.argmax(cumsum >=0.95) +1 # for index 
    
    print(f"--> Number of components reset to {d!r} as the most "
          'representative variance (95%) in the dataset.')
    
    return d 
       
def set_axes_and_feature_importances(
        Obj: object,
        X: NDArray| DataFrame
        )-> NDArray | object: 
    """ Set n_axes<n_components_> and features attributes if `X` is 
    pd.DataFrame."""
    message ='Object %r has not attribute %r'%(Obj.__class__.__name__,
                                                   'n_components_')
    try: 
        #Try to find n_components_attributes. If not found 
        # shoud reset to 'n_components'
        setattr(Obj, 'n_axes', getattr(Obj, 'n_components_'))
    except AttributeError: #as attribute_error: 
        #raise AttributeError(message) from attribute_error
        warnings.warn(message +". Should be 'n_components' instead.'")
        _logger.debug('Attribute `n_components_` not found.'
                      ' Should be `n_components` instead.')
        setattr(Obj, 'n_axes', getattr(Obj, 'n_components'))
    # get the features importance and features names
    if isinstance(X, pd.DataFrame):
        
        try: 
            
            pca_components_= getattr(Obj, 'components_')
        except AttributeError: 
            obj_name=''
            if hasattr(Obj, 'kernel'): 
                obj_name ='KernelPCA'
                
            elif hasattr(Obj, 'n_neighbors') and hasattr(Obj, 'nbrs_'): 
                obj_name ='LoccallyLinearEmbedding'
                
            if obj_name !='':
                warnings.warn(f"{obj_name!r} has no attribute 'components_'"
                              )
                _logger.debug(f"{Obj.__class__.__name__!r} inherits from "
                              f"{obj_name!r} attributes and has not attribute"
                              "'components_")
                
            setattr(Obj, 'feature_importances_', None)
            
            return Obj
        
        Obj.feature_importances_= find_features_importances(
                                        np.array(list(X.columns)), 
                                        pca_components_, 
                                        Obj.n_axes)
        



    
    
    
    
    
    
    
    
    
    
    





