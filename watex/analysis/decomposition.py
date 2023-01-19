# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Copyright (c) 2021-2022
"""
Decomposition 
================

Steps behing the principal component analysis (PCA) and matrices decomposition 
"""

from warnings import warn 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA 
from .._docstring import _core_docs 
from ..utils.funcutils import _assert_all_types 
# ---
__all__=[
    "extract_pca", 
    "decision_region", 
    "feature_transformation", 
    "total_variance_ratio" , 
    "linear_discriminant_analysis"
    ]

def extract_pca (X): 
    # standize the features 
    sc = StandardScaler() 
    X= sc.fit_transform(X)
    # constructing the covariance matrix 
    cov_mat = np.cov(X.T)
    eigen_vals, eigen_vecs = np.linalg.eig (cov_mat)
    return eigen_vals, eigen_vecs, X

extract_pca.__doc__="""\
A naive approach to extract PCA from training set X 
 
Parameters 
----------
{params.X}

Returns 
--------
Tuple (eigen_vals, eigen_vecs, Xsc): 
    Eigen values , eigen vectors and Xsc scaled (standardized)
    
Examples
---------
>>> from watex.exlib.sklearn import SimpleImputer 
>>> from watex.utils import selectfeatures 
>>> from watex.datasets import fetch_data 
>>> from watex.analysis import extract_pca 
>>> data= fetch_data("bagoue original").get('data=dfy1') # encoded flow categories 
>>> y = data.flow ; X= data.drop(columns='flow') 
>>> # select the numerical features 
>>> X =selectfeatures(X, include ='number')
>>> # imputed the missing data 
>>> X = SimpleImputer().fit_transform(X)
>>> eigval, eigvecs, _ = extract_pca(X)
>>> eigval 
... array([2.09220756, 1.43940464, 0.20251943, 1.08913226, 0.97512157,
       0.85749283, 0.64907948, 0.71364687])

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
    eigen_vals, eigen_vcs, _ = extract_pca(X)
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
        plt.xlabel ('Principal component analysis')
        plt.legend (loc ='best')
        plt.tight_layout()
        plt.show () 
    
    return cum_var_exp 
    
total_variance_ratio.__doc__ ="""\
Compute the total variance ratio. 

Is the ratio of an eigenvalues :math:`\\lambda_j`, as simply the fraction of 
and eigen value, :math:`\\lambda_j` and the total sum of the eigen values as: 
 
.. math:: 
    
    \\text{explained_variance_ratio}= \\frac{\\lambda_j}{\\sum{j=1}^{d} \\lambda_j}
    
Using numpy cumsum function, we can then calculate the cumulative sum of 
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
    
Examples 
----------
>>> from watex.analysis import total_variance_ratio 
>>> # Use the X value in the example of `extract_pca` function   
>>> cum_var = total_variance_ratio(X, view=True)
>>> cum_var
... array([0.26091916, 0.44042728, 0.57625294, 0.69786032, 0.80479823,
       0.89379712, 0.97474381, 1.        ])
"""

def feature_transformation (
        X, y=None, n_components =2, positive_class=1, view =False):
    
    # select k vectors which correspond to the k largest 
    # eigenvalues , where k is the dimesionality of the new  
    # subspace (k<=d) 
    eigen_vals, eigen_vecs, X = extract_pca(X)
    # -> sorting the eigen values by decreasing order 
    eigen_pairs = [ (np.abs(eigen_vals[i]) , eigen_vecs[:, i]) 
                   for i in range(len(eigen_vals))]
    eigen_pairs.sort(key =lambda k :k[0], reverse =True)
    # collect two eigen vectors that correspond to the two largest 
    # eigenvalues to capture about 60% of the variance of this datasets
    if n_components !=2 : 
        #XXX TODO: transform component > 2 
        warn("N-component > 2 is not implemented yet.", UserWarning)
    w= np.hstack((eigen_pairs[0][1][:, np.newaxis], 
                 eigen_pairs [1][1][:, np.newaxis])
                 ) 
    # In pratice the number of principal component has to be 
    # determined by a tradeoff between computational efficiency 
    # and the performance of the classifier.
    
    #-> transform X onto a PCA subspace( the pc one on two)
    X_transf = X.dot(w)
    
    if view: 
        from ..utils.plotutils import make_mpl_properties
        if y is None: 
            raise TypeError("Missing the target `y`")
        # markers = tuple (D_MARKERS [:len(np.unique (y))])
        # colors = tuple (D_COLORS [:len(np.unique (y))])    
        colors = make_mpl_properties(len(np.unique (y)))
        markers = make_mpl_properties (
            len(np.unique (y)), 'markers') 
        
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
Transform  X into  new principal components after decomposing 
the covariances matrices.    
    
Parameters 
-----------
{params.X} 
{params.y}

n_components: int, default=2 
    Number of components with most total variance ratio. 
positive_class: int, 
    class label as an integer indenfier within the class representation. 
    
view: bool, default {{'False'}}
    give an overview of the total explained variance. 

Returns 
---------
X_transf : nd-array 
    X PCA training set transformed.
    
Examples 
---------
>>> from watex.analysis import feature_transformation 
>>> # Use the X, y value in the example of `extract_pca` function  
>>> Xtransf = feature_transformation(X, y=y,  positive_class = 2 , view =True)
>>> Xtransf[0] 
... array([-1.0168034 ,  2.56417088])


""".format(params = _core_docs["params"]
)

def _decision_region (X, y, clf, resolution =.02 , ax =None ): 
    """ visuzalize the decision region """
    from ..utils.plotutils import make_mpl_properties
    # setup marker generator and colors map 
    colors = tuple (make_mpl_properties(len(np.unique (y))))
    markers = tuple (make_mpl_properties (
        len(np.unique (y)), 'markers') )
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface 
    x1_min , x1_max = X[:, 0].min() -1, X[:, 0].max() +1 
    x2_min , x2_max = X[:, 1].min() -1, X[:, 1].max() +1 
    
    xx1 , xx2 = np.meshgrid(np.arange (x1_min, x1_max, resolution), 
                            np.arange (x2_min, x2_max, resolution)
                            )
    z= clf.predict(np.array ([xx1.ravel(), xx2.ravel()]).T)
    z= z.reshape (xx1.shape)
    if ax is None: 
        fig, ax= plt.subplots()
    ax.contourf (xx1, xx2, z, alpha =.4, cmap =cmap )
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    # plot the examples by classes 
    for idx, cl in enumerate (np.unique (y)): 
        ax.scatter (x= X[y ==cl, 0] , y = X[y==cl, 1], 
                     alpha =.6 , 
                     color = cmap(idx), 
                     edgecolors='black', 
                     marker = markers[idx], 
                     label=cl 
                     ) 
    return ax 
        
def decision_region (
        X, y, clf, Xt =None, yt=None, random_state = 42, test_size = .3 , 
        scaling =True, split =False,  n_components =2 , view ='X',
        resolution =.02, return_expl_variance_ratio =False, return_axe =False, 
        axe =None, 
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
            ax = _decision_region(
                X_pca, y, clf = clf,resolution = resolution, ax= axe) 
        if view =='test':
            if Xt_pca is None: 
                raise TypeError("Cannot plot missing test sets (Xt, yt)")
            ax = _decision_region(Xt_pca, yt, clf=clf, resolution =resolution,
                                  ax =axe )
        ax.set_xlabel("PC1")
        ax.set_ylabel ("PC2")
        ax.legend (loc= 'lower left')
        plt.show ()
    if return_expl_variance_ratio : 
        pca =PCA(n_components =None )
        X_pca = pca.fit_transform(X)
        return pca.explained_variance_ratio_ 
    
    return ax if return_axe else X_pca 

decision_region.__doc__="""\
View decision regions for the training data reduced to two 
principal component axes. 

Parameters 
-----------
{params.X}
{params.y}
{params.Xt}
{params.yt}
{params.clf}

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
    
return_axes: bool, default=False, 
    Return matplotlib object axe 
ax: Matplotlib.Axes object, optional 
    If not supplied, it is created.
    
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
>>> from watex.exlib.sklearn import SimpleImputer, LogisticRegression  
>>> from watex.analysis.decomposition import decision_region 
>>> data= fetch_data("bagoue original").get('data=dfy1') # encoded flow categories 
>>> y = data.flow ; X= data.drop(columns='flow') 
>>> # select the numerical features 
>>> X =selectfeatures(X, include ='number')
>>> # imputed the missing data 
>>> X = SimpleImputer().fit_transform(X)
>>> lr_clf = LogisticRegression(multi_class ='ovr', random_state =1, solver ='lbfgs') 
>>> Xpca= decision_region(X, y, clf=lr_clf, split = True, view ='Xt') # test set view
>>> Xpca[0] 
... array([-1.02925449,  1.42195127])

""".format(params = _core_docs["params"]
)

def linear_discriminant_analysis (
        X, y, n_components = 2 , view=False, verbose = 0  , return_X=True, 
 ): 
    n_components = int (_assert_all_types (n_components, int, float ))
    # standardize the features 
    eigen_vals, eigen_vcs, X = extract_pca(X)
    # compute the mean vectors which will use to 
    # construct the within classes scatter matrix 
    np.set_printoptions(precision=4) 
    mean_vecs =list() 
    for label in range (1, len(np.unique (y))): 
        mean_vecs.append (np.mean(X[y==label], axis =0 ))
        if verbose : 
            print('MV %s: %s\n' %(label , mean_vecs[label -1]))
            
    # compute the within class Sw using the mean vector; 
    # calculated by summing up the individual scatter matrices 
    
    d = X.shape [1]  # number of features
    SW =np.zeros ((d, d ))
    #==========================================================================
    # for label , mv in zip (range (1 , len(np.unique (y))), mean_vecs): 
    #     class_scatter = np.zeros ((d, d) )
    #     for row in X [y==label ]: 
    #         row, mv= row.reshape (d, 1), mv.reshape (d, 1)
    #         class_scatter += (row - mv ).dot(row - mv).T 
    #         SW += class_scatter 
    # if verbose : 
    #     print("within-class scatter matrix: %sx %s" % (
    #         SW.shape [0], SW.shape[1]))
    
    # the assumption that we are making when we are computing the  
    # scatter matrices  is that the labels in the training datasets 
    # are uniformly distributed. Howeverm, if we print the number 
    # of class labels, we see that this assumtions is violated. 
    # for instance : 
    # >>> print('class label distributions: %s' % np.bincount (y)[1:])
    #==========================================================================
    # the better way is to scale the individual scatter matrices, before 
    # sum them up as scatter matrix SW.  when dividing the scatter matrices 
    # by the number of classes examples , we can see that computig the  
    # scatter matrix is the same as covariance matrix 
    
    for label , mv in zip (range (1 , len(np.unique (y))), mean_vecs): 
        class_scatter =np.cov(X[y==label].T )
        SW += class_scatter 

    if verbose : 
        print("Scaled within-class scatter matrix: %sx %s" % (
            SW.shape [0], SW.shape[1]))
    # compute between classes scatter matrix SB 
    mean_overall = np.mean (X, axis = 0 )  # include the example from c classes 
    SB = np.zeros ((d, d)) 
    for i, mean_vec in enumerate( mean_vecs ): 
        n = X[y == i+1 , :].shape [0] 
        mean_vec = mean_vec.reshape (d, 1) # make column vector 
        mean_overall = mean_overall.reshape (d, 1)
        SB += n * (mean_vec - mean_overall ).dot ((mean_vec - mean_overall).T)
        
    if verbose : 
        if verbose : 
            print("Between within-class scatter matrix: %sx%s" % (
                SB.shape [0], SB.shape[1]))
    
    # select discriminant for the new feature subspace 
    eigen_vals, eigen_vecs = np.linalg.eig (np.linalg.inv (SW).dot(SB))
    # sort the eigen value in descending order 
    eigen_pairs = [ (np.abs (eigen_vals[i]), eigen_vecs [:, i])
                     for i in range(len(eigen_vals ))
                     ]
    eigen_pairs = sorted (eigen_pairs, key =lambda k: k[0], reverse =True )
    if verbose : 
        print("Eigen values in descending order:\n")
        for eig_val in eigen_pairs : 
            print(eig_val[0])
            
    if view : 
        tot = sum(eigen_vals.real )
        discr = [(i/tot ) for i in sorted (eigen_vals.real , reverse =True ) 
                 ]
        cum_discr = np.cumsum(discr) 
        plt.bar (range (1 , 1+ X.shape [1]), discr , alpha =.5 , 
                 align= 'center', label ='Individual "discriminability"'
                 )
        plt.step (range (1 , 1+ X.shape [1]), cum_discr , where ='mid', 
                  label ='Cumulative "discriminality"')
        plt.ylabel ('"Discriminability" ratio') 
        plt.xlabel ("Linear discriminants")
        plt.ylim ([ -0.1 , 1.1 ])
        plt.legend (loc ='best')
        plt.tight_layout() 
        plt.show () 
    
    # stack the two most discrimative eigen columns 
    # to create the transformation matrix W  for two components 
    # W = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, 
    #                eigen_pairs[1][1][:, np.newaxis].real )
    #               )
    W0 = np.hstack(tuple (eigen_pairs[k][1][:, np.newaxis].real 
                          for k in range (len(eigen_pairs)))
                   ) 
    if n_components > W0.shape [1]: 
        warn (f"n_component '{n_components}' is larger than the most  "
               f"discriminant vector '{W0.shape [1]}'")
        n_components = W0.shape [1] 
        
    W = W0 [:, :n_components]
  
    if verbose: 
        print("07 first values  of W:\n", W[:5, :])
    # we can now transform the training dataset 
    # from  matrix  W 
    return  X.dot(W) if return_X else W 


linear_discriminant_analysis.__doc__ ="""\
Linear Discriminant Analysis `LDA`. 

LDA is used as a technique for feature extraction to increase the 
computational efficiency and reduce the degree of overfitting due to the 
curse of dimensionnality in non-regularized models. The general concept  
behind LDA is very similar to the principal component analysis (PCA), but  
whereas PCA attempts to find the orthogonal component axes of minimum 
variance in a dataset, the goal in LDA is to find the features subspace 
that optimize class separability. The main steps requiered to perform 
LDA are summarized below: 
    
* Standardize the d-dimensional datasets (d is the number of features)
* For each class , compute the d-dimensional mean vectors. Thus for 
  each mean feature value, :math:`\\mu_m` with respect to the examples 
  of class :math:`i`: 
        
  .. math:: 
      
      m_i = \\frac{1}{n_i} \\sum{x\\in D_i} x_m 
        
* Construct the between-class scatter matrix, :math:`S_B` and the 
  within class scatter matrix, :math:`S_W`. Individual scatter matrices 
  are scalled :math:`S_i` before we sum them up as scatter 
  matrix :math:`S_W` as:
      
  .. math:: 
         
      \\sum{i} = \\frac{1}{n_i}S_i 
        
      \\sum{i} = \\frac{1}{n_i} \\sum{x\\in D_i} (x-m_i)(x-m_i)^T
            
  The within-class is also called the covariance matrix, thus we can compute 
  the between class scatter_matrix :math:`S_B`. 
  
  .. math:: 
      
      S_B= \\sum{i}^{n_i}(m_i-m) (m_i-m)^T 
        
  where :math:`m` is the overall mean that is computed , including examples 
  from all classes. 

* Compute the eigenvectors and corresponding eigenvalues of the matrix 
  :math:`S_W^{-1}S_B`. 
* Sort the eigenvalues by decreasing order to rank the corresponding 
  eigenvectors 
* Choose the :math:`k` eigenvectors that correspond to the :math:`k` 
  largest eigenvalues to construct :math:`dxk`-dimensional 
  transformation matrix, :math:`W`; the eigenvectors are the columns 
  of this matrix. 
* project the examples onto the new_features subspaces using the 
  transformation matrix :math:`W`.  
    
Parameters 
-----------
X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
    Training set; Denotes data that is observed at training and 
    prediction time, used as independent variables in learning. 
    When a matrix, each sample may be represented by a feature vector, 
    or a vector of precomputed (dis)similarity with each training 
    sample. :code:`X` may also not be a matrix, and may require a 
    feature extractor or a pairwise metric to turn it into one  before 
    learning a model.
    
y: array-like, shape (M, ) ``M=m-samples``, 
    train target; Denotes data that may be observed at training time 
    as the dependent variable in learning, but which is unavailable 
    at prediction time, and is usually the target of prediction. 
  
n_components: int, default =2 
    Number of components considered as the most discriminative eigen vector.
    
return_X: bool, default =True 
    return the transformed training set from `n_components`. 

view: bool ,default =False, 
    Visualize the LDA plot. If set to ``True``, the plot is triggered. 
    
Returns 
--------
X or W: ndarray (n_samples, 2 ) 
    The transformed train set (X) or matrix (W) from the most discriminative 
    eigenvector columns
    
Examples
----------
>>> from watex.datasets import fetch_data 
>>> from watex.exlib.sklearn import SimpleImputer, LogisticRegression  
>>> from watex.analysis.decomposition import linear_discriminant_analysis 
>>> data= fetch_data("bagoue original").get('data=dfy1') # encoded flow
>>> y = data.flow ; X= data.drop(columns='flow') 
>>> # select the numerical features 
>>> X =selectfeatures(X, include ='number')
>>> # imputed the missing data 
>>> X = SimpleImputer().fit_transform(X)
>>> Xtr= linear_discriminant_analysis (X, y , view =True)
"""



      