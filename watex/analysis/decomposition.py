# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Copyright (c) 2021-2022
"""
Decomposition 
================

Steps behing the principal component analysis (PCA) and matrices decomposition 

Created on Thu Oct 13 08:41:13 2022
@author: Daniel
"""

from warnings import warn 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap 
from .._docstring import _core_docs 
from ..exlib.sklearn import (train_test_split, StandardScaler, PCA )
from ..utils.plotutils import (D_COLORS, D_MARKERS)
# ---

def extract_pca (X): 
    # standize the features 
    sc = StandardScaler() 
    X= sc.fit_transform(X)
    # constructing the covariance matrix 
    cov_mat = np.cov(X.T)
    eigen_vals, eigen_vecs = np.linalg.eig (cov_mat)
    return eigen_vals, eigen_vecs

extract_pca.__doc__="""\
 A naive approach to extract PCA from training set X 
 
Parameters 
----------
{params.core.X}

Returns 
--------
Tuple (eigen_vals, eigen_vecs): 
    Eigen values and eigen vectors 
    
Notes 
-------
All consequent principal component (pc) will have the larget variance 
given the constraint that these component are uncorrelated (orthogonal)  
to other pc - even if the inputs features are corralated , the 
resulting of pc will be mutually orthogonal (uncorelated). 
Note that the PCA directions are highly sensistive to data scaling and we 
need to standardize the features prior to PCA if the features were measured 
on different scales and we assign equal importances of all features   
    
the numpy function was designed to operate on both symetric and non-symetric 
squares matrices. However you may find it return complex eigenvalues in 
certains casesA related function, `numpy.linalg.eigh` has been implemented 
to decompose Hermetian matrices which is numerically more stable to work with 
symetric matrices such as the covariance matrix. `numpy.linalg.eigh` always 
returns real eigh eigenvalues 
""".format(params = _core_docs["params"]
)
    
def total_variance_ratio (X, view =False): 
    eigen_vals, eigen_vcs = extract_pca(X)
    tot =sum(eigen_vals)
    # sorting the eigen values by decreasing 
    # order to rank the eigen_vectors
    var_exp = list(map( lambda x: x/ tot , sorted (eigen_vals, reverse =True)))
    #var_exp = [(i/tot) for i in sorted (eigen_vals, reverse =True)]
    cum_var_exp = np.cumsum (var_exp)
    if view: 
        plt.bar (range(1, len(eigen_vals)+1), var_exp , alpha =.5, 
                 align='center', label ='Individual explained variance')
        plt.step (range(1, len(eigen_vals)+1), cum_var_exp, where ='mid', 
                  label="Cumulative explained variance")
        plt.ylabel ("Explained variance ratio")
        plt.xlable ('Principal component analysis')
        plt.legend (loc ='best')
        plt.tight_layout()
        plt.show () 
    
    return cum_var_exp 
    
total_variance_ratio.__doc__="""\
Ratio of an eigenvalues :math:`\lambda_j`, as simply the fraction of 
and eigen value, :math:`\lambda_j` and the total sum of the eigen values 
as: 
 
.. math:: 
    explained variance ration = \fract{\lambda_j}{\sum{j=1}^{d} \lambda_j}
    
Using numpy cumsum function,we can then calculate the cumulative sum of 
explained variance which can be plot if `plot` is set to ``True`` via 
matplotlib set function.    
    
Parameters 
--------------
X: Nd-array, shape(M, N)
    Array of training set with  M examples and N-features

view: bool, default {'False'}
    give an overview of the total explained variance. 

Returns 
---------
cum_var_exp : array-like 
    Cumulative sum of variance total explained. 
"""

def feature_transformation (
        X, y=None, n_components =2, positive_class=1, view =False):
    # select k vectors which correspond to the k largest 
    # eigenvalues , where k is the dimesionality of the new  
    # subspace (k<=d) 
    eigen_vals, eigen_vecs = extract_pca(X)
    # -> sorting the eigen values by decreasing order 
    eigen_pairs = [ (np.abs(eigen_vals[i]) , eigen_vecs[:, i]) 
                   for i in range(len(eigen_vals))]
    eigen_pairs.sort(key =lambda k :k[0], reverse =True)
    # collect two eigen vectors that correspond to the two largest 
    # eigenvalues to capture about 60% of the variance of this datasets
    if n_components !=2 : 
        #XXX TODO: transform component > 2 
        warn("N-component !=2 is not implemented yet.", UserWarning)
    w= np.hstack((eigen_pairs[0][1][:, np.newaxis], 
                 eigen_pairs [1][1][:np.newaxis])
                 ) 
    # In pratice the number of principal component has to be 
    # determined by a tradeoff between computational efficiency 
    # and the performance of the classifier.
    
    #-> transform X onto a PCA subspace( the pc one on two)
    X_transf = X[0].dot(w)
    
    if view: 
        if y is None: 
            raise TypeError("Missing the target `y`")
            
        colors = D_COLORS [: len(np.unique (y))] 
        markers = D_MARKERS [:len(np.unique (y))] #['s', 'x', 'o'] for 03
        
        if positive_class not in np.unique (y): 
            raise ValueError( f"'{positive_class}' does not match any label "
                             "of the class. The positive class must be an  "
                             "integer label within the class values"
                             )
        for l, c, m in zip(np.unique (y),colors, markers ):
            plt.scatter(X_transf[y==positive_class, 0],
                        X_transf[y==positive_class, 1], c= c,
                        label =l, marker=m)
        plt.xlabel ('PC1')
        plt.ylabel ('PC2')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show() 
        
    return X_transf 

feature_transformation.__doc__="""\
Transform  X into a new principal components after successfully 
decomposed to the covariances matrices.    
    
Parameters 
-----------
{params.core.X} 
{params.core.y}

positive_class: int, 
    class label as an integer indenfier within the class representation. 
    
view: bool, default {{'False'}}
    give an overview of the total explained variance. 

Returns 
---------
X_transf : nd-array 
    X PCA training set transformed.
""".format(params = _core_docs["params"]
)

def _decision_region (X, y, clf, resolution =.02 ): 
    """ visuzalize the decision region """
    # setup marker generator and colors map 
    markers = tuple (D_MARKERS [:len(np.unique (y))])
    colors = tuple (D_COLORS [:len(np.unique (y))])
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface 
    x1_min , x1_max = X[:, 0].min() -1, X[:, 0].max() +1 
    x2_min , x2_max = X[:, 1].min() -1, X[:, 1].max() +1 
    
    xx1 , xx2 = np.meshgrid(np.arange (x1_min, x1_max, resolution), 
                            np.arange (x2_min, x2_max, resolution)
                            )
    z= clf.predict(np.array ([xx1.ravel(), xx2.ravel()]).T)
    z= z.reshape (xx1.shape)
    
    plt.contourf (xx1, xx2, z, alpha =.4, cmap =cmap )
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot the examples by classes 
    for idx, cl in enumerate (np.unique (y)): 
        plt.scatter (x= X[y ==cl, 0] , y = X[y==cl, 1], 
                     alpha =.6 , 
                     color = cmap(idx), 
                     edgecolors='k', 
                     marker = markers[idx], 
                     label=cl 
                     ) 
        

def decision_region (
        X, y, clf, Xt =None, yt=None, random_state = 42, test_size = .3 , 
        scaling =True, split =False,  n_components =2 , view ='X',
        resolution =.02, return_expl_variance_ratio =False, 
        **kws 
        ): 
    view = str(view).lower().strip()
    if  view in ('xt', 'test'): view ='test'
    elif view in ('x', 'train'): view = 'train'
    else:  view =None 
    if split : 
        X, Xt, y, yt = train_test_split(X, y, random_state =random_state, 
                                        test_size =test_size, **kws)
       
    pca = PCA (n_components = n_components)
    # scale data 
    sc = StandardScaler() 
    X =  sc.fit_transform(X) if scaling else X 
    if Xt is not None: 
        Xt =  sc.transform(Xt) if scaling else Xt 
        
    # dimension reduction 
    X_pca = pca.fit_transform(X)
    Xt_pca = pca.transform (Xt) if ( Xt is not None)  else None 
    # fitting the classifier clf model on the reduced datasets 
    clf.fit(X_pca, y )
    # now plot the decision regions 
    if view is not None: 
        if view =='train': 
            _decision_region(X_pca, y, clf = clf,resolution = resolution) 
        if view =='test':
            if Xt_pca is None: 
                raise TypeError("Cannot plot missing test sets (Xt, yt)")
            _decision_region(Xt_pca, yt, clf=clf, resolution =resolution )
        plt.label("PC1")
        plt.ylabel ("PC2")
        plt.legend (loc= 'lower left')
        plt.show ()
    if return_expl_variance_ratio : 
        pca =PCA(n_components =None )
        X_pca = pca.fit_transform(X)
        return pca.explained_variance_ratio_ 
    
    return X_pca 

decision_region.__doc__="""\
View decision regions for the training data reduced to two 
principal component axes. 

Parameters 
-----------
{params.core.X}
{params.core.y}
{params.core.Xt}
{params.core.yt}
{params.core.clf}

random_state: int, default {{42}}
    state of shuffling the data
test_size: float < 1 , default {{.3}}
    the size to keep remainder data into the test set . 
split: bool, False 
    Split (X,y) data into a training and test sets(Xt, yt). Here, it value is 
    triggered to ``True``, we assume (X, y) previously given are all the whole 
    dataset with target `y`. 
n_components: int, float 2 , default {{2}}
    the number of principal component to retrieve. If value is given as a 
    ratio for instance '.95' i.e. the ratio of keeping variance is 95% and the 
    `n_components can be get using the attributes scikit-learn getter as
    `<estimator>.n_components_`
view: str , ['X', 'Xt', None]
    the kind of vizualization. 'X', 'Xt' mean the training and test set decision
    region visualization respectively. If set to ``None``(default), the view 
    are muted. 
    
resolution: float, default{{.02}}
    level of the extension of numpy meshgrip to tighting layout the plot. 
return_expl_variance_ratio: bool, default is {{False}}
    returns the PCA variance ratio explaines of all principal components. 
    
kws: dict 
    Additional keywords arguments passed to  the scikit-learn function 
    :func:`sklearn.model_selection.train_test_split`
    
Returns 
---------
nd-array | arraylike (return_expl_variance_ratio=True) 
    X PCA training set transformed or PCA explained variance ratio.  
    
Examples
---------
>>> from watex.datasets import fetch_data 
>>> from sklearn.linear_model import LogisticRegression 
>>> from watex.analysis.decomposition import decision_region 
>>> lr_clf = LogisticRegression(multi_class ='ovr', random_state =1, solver ='lbgfs') 
>>> X, y = fetch_data ('bagoue training')
>>> _= decision_region(X, y, clf=lr_clf, split = True, view ='Xt') # test set view
""".format(params = _core_docs["params"]
)


















































































      