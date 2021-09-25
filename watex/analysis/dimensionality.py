# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Sep 16 11:31:38 2021
# This module is a set of processing tips 
# released under a MIT- licence.
#Created on Wed Sep 22 15:04:52 2021
#@author: @Daniel03

import warnings
import numpy as np
import pandas as pd 
from typing import TypeVar 

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA 

from ..utils.decorator import docstring
from ..utils._watexlog import watexlog
from ..processing.transformers import (DataFrameSelector,
                                  CombinedAttributesAdder)

T= TypeVar('T', list, tuple)
_logger = watexlog().get_watex_logger(__name__)

class Reducers: 
    """ Reduce dimension for data visualisation.
    
    Reduce number of dimension down to two (or to three) make  it possible 
    to plot high-dimension trainsing set on the graph and often gain some 
    important insights by visually detecting patterns, such as clusters. 
    """
    
    def PCA(self,X, n_components=None, plot_projection=False, 
            plot_kws=None, n_axes =None,  **pca_kws ): 
        """Principal Components analysis (PCA) is by far themost popular
        dimensional reduction algorithm. First it identifies the hyperplane 
        that lies closest to the data and project it to the data onto it.
        
        :param X: Dataset compose of n_features items for dimension reducing
        
        :param n_components: Number of dimension to preserve. If`n_components` 
                is ranged between float 0. to 1., it indicated the number of 
                variance ratio to preserve. If ``None`` as default value 
                the number of variance to preserve is ``95%``.
        :param plot_projection: Plot the explained varaince as a function 
        of number of dimension. Deafualt is``False``.
        
        :param n_axes: Number of importance components to retrieve the 
            variance ratio. If ``None`` the features importance is computed 
            usig the cumulative variance representative of 95% .
        
        :param plot_kws: Additional matplotlib.pyplot keywords arguments. 
        
        :Example: 
            
            >>> from watex.analysis.dimensionality import Reducers
            >>> from .datasets.data_preparing import X_train_2
            >>> DimensionReduction().PCA(X_train_2, 0.95, n_axes =3)
            >>> pca.components_
            >>> pca.features_importances_
        """
        def findFeaturesImportances(fnames, components, n_axes=2): 
            """ Retrive the features importance with variance ratio.
            
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
        
        from sklearn.decomposition import PCA 
        
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
            plt.xlabel('Dimensions')
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
    
def prepareDataForPCA(X, 
                    imputer_strategy:str ='median',
                    missing_values :float=np.nan,
                    num_indexes:T =None, 
                    cat_indexes:T=None, 
                    selector__:bool=True, 
                    scaler__:str='StandardScaler',
                    encode_categorical_features__:bool=True, 
                    pd_column_list:T =None): 
    
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
        Default is :class:`sklearn.preprocessing.OrdinalEncoder` . 
        
     Notes
     -----
     `num_indexes` and cat_indexes` is mainly used when type of data `x` is 
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
                        X, 
                        n_components:float=0.95,
                        add_attributes:bool=False,
                        attributes_ix:T= None, 
                        plot_var_ratio:bool =False,
                        **ppca_kws ): 
    
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
                label ='pca explained variance vs Ndimension')
        
        if self.xlabel is None: 
            self.xlabel ='Ndimensions'
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
        dataset composed of initial features 
    n_components: float of int
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

    >>> from from watex.viewer.mlplot import MLPlots
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
    
if __name__=="__main__": 
#     if __package__ is None : 
#         __package__='watex'
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.linear_model import SGDClassifier
    # from .datasets import X_, y_,  X_prepared, y_prepared, default_pipeline
    from watex.datasets.data_preparing import X_train_2

    pca= Reducers().PCA(X_train_2, 0.95, plot_projection=True, n_axes =3)
    print('columnsX=', X_train_2.columns)
    print('components=', pca.components_)
    print('feature_importances_:', pca.feature_importances_)

    
    
    
    
    
    
    
    
    
    
    





