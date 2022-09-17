# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Sep 16 11:31:38 2021
# This module is a set of processing tips 
# released under a MIT- licence.
#Created on Wed Sep 22 15:04:52 2021
#@author: @Daniel03 <etanoyau@gmail.com>

from __future__ import annotations 

import warnings
import numpy as np
import pandas as pd 

from ..sklearn import (
    OrdinalEncoder ,
    SimpleImputer ,
    StandardScaler, 
    MinMaxScaler,
    PCA, 
    IncrementalPCA, 
    KernelPCA, 

    )
from ..typing import (
    List,
    Any,
    Dict, 
    Optional, 
    F, 
    T,
    Array, 
    NDArray, 
    DataFrame,
    Series, 
    Sub
    )
from ..decorators import docstring
from .._watexlog import watexlog
from ..models.validation import GridSearch
from ..bases.transformers import (
    DataFrameSelector,
    CombinedAttributesAdder
   )

_logger = watexlog().get_watex_logger(__name__)

__all__ = [
    'Reducers',
    'pcaVarianceRatio', 
    'get_best_kPCA_params', 
    'get_component_with_most_variance',
    'plot_projection', 
    'find_features_importances', 
    'prepareDataForPCA', 
    'find_features_importances', 
    'pcaVarianceRatio' 
]
          

class Reducers: 
    """ Reduce dimension for data visualisation.
    
    Reduce number of dimension down to two (or to three) for instance, make  
    it possible to plot high-dimension training set on the graph and often
    gain some important insights by visually detecting patterns, such as 
    clusters.
    
    Parameters
    ----------
     X: ndarry, or pd.Datafame, 
             Dataset compose of n_features items for dimension reducing

     n_components: Number of dimension to preserve. If`n_components` 
            is ranged between float 0. to 1., it indicated the number of 
            variance ratio to preserve. If ``None`` as default value 
            the number of variance to preserve is ``95%``.
    """
    
    def PCA(self,
            X: NDArray | DataFrame,
            n_components: float | int =None, 
            plot_projection: bool =False, 
            plot_kws: Dict[str, Any] =None,
            n_axes: int =None, 
            **pca_kws
            )-> object: 
        """Principal Components analysis (PCA) is by far the most popular
        dimensional reduction algorithm. First it identifies the hyperplane 
        that lies closest to the data and project it to the data onto it.
        
        :param X: Dataset compose of n_features items for dimension reducing
        
        :param n_components: Number of dimension to preserve. If`n_components` 
            is ranged between float 0. to 1., it indicated the number of 
            variance ratio to preserve. If ``None`` as default value 
            the number of variance to preserve is ``95%``.
                
        :param plot_projection: Plot the explained varaince as a function 
            of number of dimension. Default is``False``.
        
        :param n_axes: Number of importance components to retrieve the 
            variance ratio. If ``None`` the features importance is computed 
            usig the cumulative variance representative of 95% .
        
        :param plot_kws: Additional matplotlib.pyplot keywords arguments. 
        
        :Example: 
            >>> from watex.analysis.dimensionality import Reducers
            >>> from watex.datasets import fetch_data
            >>> X, _= fetch_data('Bagoue analysis dataset')
            >>> Reducers().PCA(X, 0.95, n_axes =3)
            >>> pca.components_
            >>> pca.features_importances_
        """
        def findFeaturesImportances(
                fnames: Array,
                components: float |int ,
                n_axes: int =2
                )-> Array: 
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
        pca = PCA(n_components=n_components, **pca_kws)
        self.X_= pca.fit_transform(X) # X_reduced = pca.fit_transform(X)
  
        if n_components is not None: 
            cumsum = np.cumsum(pca.explained_variance_ratio_ )
        
        if plot_projection: 
            import matplotlib.pyplot as plt
            
            if plot_kws is None: 
                plot_kws ={'label':'Explained variance as a function of the'
                           ' number of dimension' }
            plt.plot(cumsum, **plot_kws)
            # plt.plot(np.full((cumsum.shape), 0.95),
            #          # np.zeros_like(cumsum),
            #          ls =':', c='r')
            plt.xlabel('N-Dimensions')
            plt.ylabel('Explained Variance')
            plt.title('Explained variance as a function of the'
                        ' number of dimension')
            plt.show()
            
        # make introspection and set the all pca attributes to self.
        for key, value in  pca.__dict__.items(): 
            setattr(self, key, value)
        
        if n_axes is None : 
            self.n_axes = pca.n_components_
        else : 
            setattr(self, 'n_axes', n_axes)
            
        # get the features importance and features names
        self.feature_importances_= findFeaturesImportances(
                                        np.array(list(X.columns)), 
                                        pca.components_, 
                                        self.n_axes)
        
        return self  
    
    def incrementalPCA(self,
                       X: NDArray | DataFrame,
                       n_components: float | int =None,
                       *, 
                       n_batches: int =None,
                       store_in_binary_file: bool =False,
                       filename: Optional[str]=None,
                       **inc_pca_kws
                       )-> object: 
        """ Incremental PCA allows to split the trainsing set into mini-batches
         and feed algorithm one mini-batch at a time. 
         
         Once problem with the preceeding implementation of PCA is that 
         requires the whole training set to fit in memory in order of the SVD
         algorithm to run. This is usefull for large training sets, and also 
         applying PCA online(i.e, on the fly as a new instance arrive)
         
        :param X: ndarray, or pd.Datafame, 
            Dataset compose of n_features items for dimension reducing
        
        :param n_components: Number of dimension to preserve. If`n_components` 
                is ranged between float 0. to 1., it indicated the number of 
                variance ratio to preserve. If ``None`` as default value 
                the number of variance to preserve is ``95%``.
        :param n_batches: int
            Number of batches to split your training sets.
        
        :param store_in_binary_file: bool, 
            Alternatively, we used numpy` memmap` class to manipulate a large 
            array stored in a binary file on disk as if it were entirely in 
            memory. The class load only the data it need in memory when it need
            its.
        
        :param filename: str, 
            Default binary filename to store in a binary file in  a disk.
        
        :Example: 
            >>> from watex.analysis.dimensionality import Reducers
            >>> from watex.datasets import fetch_data 
            >>> X, _=fetch_data('Bagoue analyses data')
            >>> recObj =Reducers().incrementalPCA(X,n_components=None,
            ...                                      n_batches=100)
            >>> plot_projection(recObj,recObj.n_components )
        """
        if n_components is None: 
            n_components= get_component_with_most_variance(X) 
            if n_batches is None: 
                raise ValueError('NoneType can not be a number of batches.')
            if n_components > (len(X)//n_batches +1): 
                warnings.warn(f'n_components=`{n_components}` must be less '
                                 'or equal to the batch number of samples='
                                 f'`{len(X)//n_batches +1}`. n_components is'
                                 f' set to {len(X)//n_batches}')
                
                n_components = len(X)//n_batches
                _logger.debug(
                    f"n_components is reset to ={len(X)//n_batches!r}")
                
        inc_pcaObj = IncrementalPCA(n_components =n_components, 
                                    **inc_pca_kws)
        for X_batch in np.array_split(X, n_batches):
            inc_pcaObj.partial_fit(X_batch)
        
        self.X_= inc_pcaObj.transform(X)
        
        if store_in_binary_file: 
            if filename is None:
                warnings.warn('Need a binary filename stored in disk of '
                              'in memory. Should provide a binary file '
                              'instead.')
                _logger.error('Need a binary filename stored in disk of '
                              'in memory. Should provide a binary file '
                              'instead.')
                raise FileNotFoundError('None binary filename found.')

            X_mm = np.memmap(filename,
                             dtype= 'float32',
                             mode='readonly', 
                             shape=X.shape)
            batch_size = X.shape[0]//n_batches
            inc_pcaObj = IncrementalPCA(
                n_components =n_components,
                batch_size= batch_size,
                **inc_pca_kws)
            
            self.X_= inc_pcaObj.fit(X_mm)
            
        make_introspection(self, inc_pcaObj)
        
        setattr(self, 'n_axes', getattr(self, 'n_components_'))
        # get the features importance and features names
        if isinstance(X, pd.DataFrame):
            pca_components_= getattr(self, 'components_')
            self.feature_importances_= find_features_importances(
                                            np.array(list(X.columns)), 
                                            pca_components_, 
                                            self.n_axes)

        return self 

    def kPCA(self,
             X: NDArray | DataFrame,
             n_components: float |int =None,
             *, 
             kernel: str ='rbf',
             reconstruct_pre_image: bool =False,
             **kpca_kws
             )-> object: 
        """KernelPCA performs complex nonlinear projections for dimentionality
        reduction.
        
        Commonly the kernel tricks is a mathematically technique that implicitly
        maps instances into a very high-dimensionality space(called the feature
        space), enabling non linear classification or regression with SVMs. 
        Recall that a linear decision boundary in the high dimensional 
        feature space corresponds to a complex non-linear decison boundary
        in the original space.
        
        :param X: ndarry, or pd.Datafame, 
            Dataset compose of n_features items for dimension reducing
        
        :param n_components: Number of dimension to preserve. If`n_components` 
            is ranged between float 0. to 1., it indicated the number of 
            variance ratio to preserve. If ``None`` as default value 
            the number of variance to preserve is ``95%``.
        
        :Example:
            >>> from watex.analysis.dimensionality import Reducers
            >>> from watex.datasets import fetch_data 
            >>> X, _=fetch_data('Bagoue analyses data')
            >>> RecObj =Reducers().kPCA(X,n_components=None,
            ...                         kernel='rbf', gamma=0.04)
            >>> plot_projection(recObj,recObj.n_components )
        """
        if n_components is None: 
           n_components= get_component_with_most_variance(X) 
           
        kpcaObj = KernelPCA(n_components=n_components, kernel=kernel, 
                            fit_inverse_transform =reconstruct_pre_image,
                            **kpca_kws)

        self.X_= kpcaObj.fit_transform(X)
        
        if reconstruct_pre_image:
            self.X_preimage= kpcaObj.inverse_transform(self.X_)
            # then compute the reconstruction premimage error
            from sklearn.metrics import mean_squared_error
            self.X_preimage_error = mean_squared_error(X, self.X_preimage)
            
        # populate attributes inherits from kpca object
        make_introspection(self, kpcaObj)
        # set axes and features importances
        set_axes_and_feature_importances(self, X)

        return self
    
    def LLE(self,
            X: NDArray | DataFrame,
            n_components: float |int =None,
            *,
            n_neighbors: int, 
            **lle_kws
            )->object: 
        """ Locally Linear Embedding(LLE) is nonlinear dimensinality reduction 
        based on closest neighbors (c.n).
        
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
         X: ndarry, or pd.Datafame, 
             Dataset compose of n_features items for dimension reducing

         n_components: Number of dimension to preserve. If`n_components` 
            is ranged between float 0. to 1., it indicated the number of 
            variance ratio to preserve. If ``None`` as default value 
            the number of variance to preserve is ``95%``.
        
        References
        -----------
        Gokhan H. Bakir, Jason Wetson and Bernhard Scholkoft, 2004;
        "Learning to Find Pre-images";Tubingen, Germany:Max Planck Institute
        for Biological Cybernetics.
        
        S. Roweis, L.Saul, 2000, Nonlinear Dimensionality Reduction by
        Loccally Linear Embedding.
        
        
        Notes
        ------
        Scikit-Learn used the algorithm based on Kernel
             Ridge Regression
             
        Example
        -------
        >>> from watex.analysis.dimensionality import Reducers
        >>> from watex.datasets import fetch_data 
        >>> X, _=fetch_data('Bagoue analyses data')
        >>> lle_kws ={
        ...    'n_components': 4, 
        ...    "n_neighbors": self.closest_neighbors}
        >>> recObj= Reducers().LLE(self.X,
        ...          **lle_kws)
        >>> pprint(recObj.__dict__)
        
        """
        
        from sklearn.manifold import LocallyLinearEmbedding
        
        if n_components is None: 
           n_components= get_component_with_most_variance(X) 
        lleObj =LocallyLinearEmbedding(n_components=n_components, 
                                        n_neighbors=n_neighbors,**lle_kws)
        self.X_= lleObj.fit_transform(X)
         # populate attributes inherits from kpca object
        make_introspection(self, lleObj)
        # set axes and features importances
        return self 
    
# @docstring(GridSearch, start='Parameters', end='Examples')
def get_best_kPCA_params(
        X:NDArray | DataFrame,
        n_components: float | int =2,
        *,
        y: Array | Series=None,
        param_grid: Dict[str, Any] =None, 
        clf: F =None,
        cv: int =7,
        **grid_kws
        )-> Dict[str, Any]: 
    """ Select the Kernel and hyperparameters using GridSearchCV that lead 
    to the best performance.
    
    As kPCA( unsupervised learning algorithm), there is obvious performance
    measure to help selecting the best kernel and hyperparameters values. 
    However dimensionality reduction is often a preparation step for a 
    supervised task(e.g. classification). So we can use grid search to select
    the kernel and hyperparameters that lead the best performance on that 
    task. By default implementation we create two steps pipeline. First reducing 
    dimensionality to two dimension using kPCA, then applying the 
    `LogisticRegression` for classification. AFter use Grid searchCV to find 
    the best ``kernel`` and ``gamma`` value for kPCA in oder to get the best 
    clasification accuracy at the end of the pipeline.
    
    Parameters
    ----------
    X: ndarry, pd.DataFrame
        Training set data.
        
    n_components: Number of dimension to preserve. If`n_components` 
            is ranged between float 0. to 1., it indicated the number of 
            variance ratio to preserve. 
            
    y: array_like 
        label validation for supervised learning 
        
    param_grid: list 
        list of parameters Grids. For instance::
            
            param_grid=[{
                "kpca__gamma":np.linspace(0.03, 0.05, 10),
                "kpca__kernel":["rbf", "sigmoid"]
                }]
            
    clf: callable, 
        Can be base estimator or a composite estimor with pipeline. For 
        instance::
            
            clf =Pipeline([
            ('kpca', KernelPCA(n_components=n_components))
            ('log_reg', LogisticRegression())
            ])
            
    CV: int 
        number of K-Fold to cross validate the training set.
        
    grid_kws:dict
        Additional keywords arguments. Refer to 
        :class:`~watex.modeling.validation.GridSearch`
    
    Example
    -------
    >>> from watex.analysis.dimensionality import Reducers
    >>> from watex.datasets import fetch_data 
    >>> X, y=fetch_data('Bagoue analyses data')
    >>> rObj = Reducers()
    >>> kpca_best_params =get_best_kPCA_params(
                    X,y=y,scoring = 'accuracy',
                    n_components= 2, clf=clf, 
                    param_grid=param_grid)
    >>> kpca_best_params
    ... {'kpca__gamma': 0.03, 'kpca__kernel': 'rbf'}
    
    """

    if n_components is None: 
        n_components= get_component_with_most_variance(X)
    if clf is None: 
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline 
        
        clf =Pipeline([
            ('kpca', KernelPCA(n_components=n_components)),
            ('log_reg', LogisticRegression())
            ])
    gridObj =GridSearch(base_estimator= clf,
                        grid_params= param_grid, cv=cv,
                        **grid_kws
                        ) 
    gridObj.fit(X, y)
    
    return gridObj.best_params_
    
    
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
        fnames: Array,
        components: float | int,
        n_axes: int =2
        )-> Array: 
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
                          f"{obj_name!r} attributes and has not attribute"
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
        )->Array:
    """ Get the number of component with 95% ratio
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
        
def prepareDataForPCA(
        X: NDArray[T] | DataFrame, 
        imputer_strategy: str ='median',
        missing_values : float =np.nan,
        num_indexes: List[int] =None, 
        cat_indexes: List[int] =None, 
        selector__: bool =True, 
        scaler__: str ='StandardScaler',
        encode_categorical_features__: bool = True, 
        pd_column_list:List[str] =None
        )-> NDArray[T]: 
    
    """ Preparing the data and handle.

    Data preparation consist to imput missing values, scales the numerical 
    features and encoded the categorial features. 
    
    Parameters 
    ----------
    imputer_strategy: str 
        strategy propose to replace the missing  values. Can be ``mean`` or
        another numbers. Default is ``median``
    missing_values: float
        Value to replace the missing value in `X` ndarray or dataframe. 
        Default is ``np.nan`
    num_indexes: 
        list of indexes to select the numerical data if numerical data  
        exist in  `X` ndarray. 
    cat_indexes: 
        list of indexes to select the categorical data if categorical 
        datta exists in  `X` ndarray. 
    selector__: bool, str (True|'auto') 
        Trigger the numerical or categorical features in Dataframe selection
        and apply all the for each category the convenient data transformation.
        Default is ``True``
    scaler__: str
        type of feature scaling applied on numerical features. 
        Can be ``MinMaxScaler``. Default is ``StandardScaler``
        
    encode_categorical_features__: bool
        Encode categorical data or text attributes. 
        Default is :class:`sklearn.preprocessing.OrdinalEncoder`. 
        
     Notes
     -----
     `num_indexes` and `cat_indexes` are mainly used when type of data `x` is 
     np.ndarray(m, nf) where `m` is number of instances or examples and 
     `nf` if number of attributes or features. `selector__` is used  for 
     dataframe preprocessing.
    """
    
    if isinstance(X, pd.DataFrame): 
        type__= 'pd.df'
        
    elif isinstance(X, np.ndarray): 
        
        if num_indexes =='auto' and cat_indexes is not None: 
            num_indexes = list(set([i for i in range(X.shape[1])]
                                   ).difference( set(cat_indexes)))
               
        if cat_indexes =='auto' and num_indexes is not None: 
             cat_indexes = list(set([i for i in range(X.shape[1])]
                                    ).difference( set(num_indexes)))
               
        
        if num_indexes is not None: 
            X_num = np.float(X[:, num_indexes])
            
        if cat_indexes is not None: 
            X_cat =X[:, cat_indexes]
            
        type__='np.nd'
        
    # --> processing numerical features 
    num_col_ =list()
    
    if type__=='pd.df':
        
        if selector__ or selector__ =='auto':
            # --> process the categorical features
            if pd_column_list is not None :
                 X_columns = pd_column_list
            else :X_columns = X.columns 
            # try to convert pandas to numeric
            
            for serie in X_columns : 
                try:
                    X= X.astype(
                            {serie:np.float64})
                except:
                    continue
                # else: # pd.to_numeric(X[serie], errors ='ignore')
            try: 
                num_col_ = X.select_dtypes('number').columns.tolist()
                X_num = X[num_col_]
 
            except: 
                datafameObj =  DataFrameSelector(select_type='num')
                X_num = datafameObj.fit_transform(X)
                num_col_ =datafameObj.attribute_names 
 
            cat_col_ = list(set(X_columns).difference(set(num_col_)))
        
            X_cat = X[cat_col_]
            
        if encode_categorical_features__: 
             encodeObj= OrdinalEncoder()
             X_cat= encodeObj.fit_transform(X_cat)
   
    # replace nan by np.nan
    if imputer_strategy  is not None and not imputer_strategy:
        imputer_obj = SimpleImputer(missing_values=missing_values,
                                    strategy=imputer_strategy)
        X_num =imputer_obj.fit_transform(X_num )
        
        
    # scale the dataset 
    if scaler__=='StandardScaler' or scaler__.lower().find('stand')>=0: 
        scalerObj =StandardScaler()
        X_num= scalerObj .fit_transform(X_num)
        
    if scaler__=='MinMaxScaler' or scaler__.lower().find('min')>=0 or\
        scaler__.lower().find('max')>=0:
        scalerObj =MinMaxScaler()
        X_num= scalerObj .fit_transform(X_num)

    
    X= np.c_[X_num, X_cat]

    if type__=='pd.df': 
        X= pd.DataFrame (X, columns =num_col_ + cat_col_)
        
    # for consistency replace all np.nan by median values 
     # replace nan by np.nan
    impObj = SimpleImputer(missing_values=missing_values,
                           strategy=imputer_strategy)
    X =impObj.fit_transform(X )
      
    return X 

@docstring(prepareDataForPCA, start ='Parameters', end='Notes')
def pcaVarianceRatio(self,
                    X: NDArray | DataFrame, 
                    n_components: float|int=0.95,
                    add_attributes: bool =False,
                    attributes_ix:List[int]= None, 
                    plot_var_ratio: bool =False,
                    **ppca_kws 
                    )-> NDArray: 

    if add_attributes and attributes_ix is None: 
        warnings.warn( 'Attributes indexes|names are None. Set attributes '
                      'indexes or names to experience the attributes combinaisons'
                      )
    col=list()
    if isinstance(X, pd.DataFrame): 
        col = list(X.columns)
        
    cObj = CombinedAttributesAdder(add_attributes=add_attributes, 
                               attributes_ix=attributes_ix)
    X=cObj.fit_transform(X)  # return numpy array 


    # get the attributes and create new pdFrame
    if len(col)!=0:
        col +=cObj.attribute_names_
        X= pd.DataFrame(X, columns=col )
    if len(col) ==0 :col =None 
    # --> prepare the data 
    X= prepareDataForPCA(X=X, pd_column_list =col, **ppca_kws)

    pca =PCA(n_components=n_components)
    pca.fit_transform(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)

    if plot_var_ratio: 
        import matplotlib.pyplot as plt 
        # create figure obj 
        fig = plt.figure(figsize = self.fig_size)
        ax = fig.add_subplot(1,1,1)
        ax.plot(cumsum,
                color= self.lc, 
                linewidth = self.lw,
                linestyle = self.ls , 
                label ='Explained variance curve')
        
        if self.xlabel is None: 
            self.xlabel ='Dimensions'
        if self.ylabel is None: 
            self.ylabel ='Explained Variance'
    
        ax.set_xlabel( self.xlabel,
                      fontsize= .5 * self.font_size * self.fs )
        ax.set_ylabel (self.ylabel,
                       fontsize= .5 * self.font_size * self.fs)
        ax.tick_params(axis='both', 
                       labelsize=.5 * self.font_size * self.fs)
        
        if self.show_grid is True : 
            if self.gwhich =='minor': 
                  ax.minorticks_on() 
            ax.grid(self.show_grid,
                    axis=self.gaxis,
                    which = self.gwhich, 
                    color = self.gc,
                    linestyle=self.gls,
                    linewidth=self.glw, 
                    alpha = self.galpha
                    )
              
            if len(self.leg_kws) ==0 or 'loc' not in self.leg_kws.keys():
                 self.leg_kws['loc']='upper left'
            
            ax.legend(**self.leg_kws)
            
    
            plt.show()
            
            if self.savefig is not None :
                plt.savefig(self.savefig,
                            dpi=self.fig_dpi,
                            orientation =self.fig_orientation)

        
    return X 

pcaVarianceRatio.__doc__="""\
Plot variance ratio by experiencing  the attributes combinaisons. 

Create a new attributes using features index or litteral string operator.
and prepared data for `PCA` variance plot.

Parameters 
-----------
X: pd.DataFrame of ndarray 
    dataset composed of initial features.
    
n_components: float oR int
    Number of dimension to preserve. If`n_components` 
    is ranged between float 0. to 1., it indicated the number of 
    variance ratio to preserve. If ``None`` as default value 
    the number of variance to preserve is ``95%``.

add_attributes: bool,
    Decide to add new features values by combining 
    numerical features operation. By default ease to 
    divided two numerical features.
                
attributes_ix : str or list of int,
    Divide two features from string litteral operation betwen or 
    list of features indexes. 
    
Returns 
-------
    X: n_darry, or pd.dataframe
    new array of dataframe with new attributes. 
    
Examples
--------
>>> from from watex.view.mlplot import MLPlots
>>> from watex.datasets.data_preparing import bagoue_train_set
>>> plot_kws = {'lc':(.9,0.,.8),
        'lw' :3.,           # line width 
        'font_size':7.,
        'show_grid' :True,        # visualize grid 
       'galpha' :0.2,              # grid alpha 
       'glw':.5,                   # grid line width 
       'gwhich' :'major',          # minor ticks
        # 'fs' :3.,                 # coeff to manage font_size 
        }
>>> mlObj =MLPlots(**plot_kws)
>>> pcaVarianceRatio(mlObj,
...                     bagoue_train_set,
...                     plot_var_ratio=True,
...                     add_attributes=True)
"""
# import inspect 
# func_signature = inspect.signature(pcaVarianceRatio)
# PARAMS_VALUES = {k: v.default
#             for k, v in func_signature.parameters.items()
#             if v.default is not inspect.Parameter.empty
#             }
# @docstring(prepareDataForPCA, start ='Parameters', end='Notes')
# def pcaVarianceRatio():
#     ...
    

    
    
    
    
    
    
    
    
    
    
    





