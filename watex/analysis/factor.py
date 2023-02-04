# -*- coding: utf-8 -*-
"""
Model selection with Probabilistic PCA and Factor Analysis (FA)
====================================================================

Probabilistic PCA and Factor Analysis are probabilistic models. The consequence 
is that the likelihood of new data can be used for model selection and 
covariance estimation. Here we compare PCA and FA with cross-validation on 
low rank data corrupted with homoscedastic noise 
(noise variance is the same for each feature) or heteroscedastic noise 
(noise variance is the different for each feature). In a second step we compare 
the model likelihood to the likelihoods obtained from shrinkage covariance 
estimators.

One can observe that with homoscedastic noise both FA and PCA succeed in 
recovering the size of the low rank subspace. The likelihood with PCA is 
higher than FA in this case. However PCA fails and overestimates the rank 
when heteroscedastic noise is present. Under appropriate circumstances the 
low rank models are more likely than shrinkage models.

The automatic estimation from Automatic Choice of Dimensionality for PCA. 
NIPS 2000: 598-604 by Thomas P. Minka is also compared.

# Authors: Alexandre Gramfort & Denis A. Engemann
# License: BSD 3 clause
edited by LKouadio on Tue Oct 11 16:54:26 2022
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.decomposition import PCA, FactorAnalysis 
from sklearn.covariance import  ShrunkCovariance, LedoitWolf 
from sklearn.model_selection import GridSearchCV, cross_val_score
from .._docstring import _core_docs

__all__=[ 
    "LW_score", 
    "shrunk_cov_score", 
    "compute_scores", 
    "pcavsfa", 
    "make_scedastic_data", 
    ]

def compute_scores(X, n_features , n_components = 5):
    n_components = np.arange(0, n_features, n_components)
    pca = PCA(svd_solver='full')
    fa = FactorAnalysis()
    pca_scores, fa_scores = [], []
    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))
        fa_scores.append(np.mean(cross_val_score(fa, X)))

    return pca_scores, fa_scores

compute_scores.__doc__ ="""\
Compute PCA score and Factor Analysis scores from training X. 
  
Parameters 
-----------
{params.X}

n_features: int, 
    number of features that composes X 
n_components: int, default {{5}}
    number of component to retrieve. 
Returns 
---------
Tuple (pca_scores, fa_scores): 
    Scores from PCA and FA  from transformed X 
""".format(params =_core_docs["params"])

def shrunk_cov_score(X):
    shrinkages = np.logspace(-2, 0, 30)  # Fit the models
    cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages})
    return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))

shrunk_cov_score.__doc__="""\
shrunk the covariance scores.
 
Parameters 
-----------
{params.X} 

Returns
-----------
score: score of covariance estimator (best ) with shrinkage

""".format(
    params =_core_docs["params"]
)
def LW_score(X, store_precision=True, assume_centered=False,  **kws):
    r"""Models score from Ledoit-Wolf.
    
    Parameters
    ----------
    store_precision : bool, default=True
        Specify if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data will be centered before computation.

    block_size : int, default=1000
        Size of blocks into which the covariance matrix will be split
        during its Ledoit-Wolf estimation. This is purely a memory
        optimization and does not affect results.
        
    Notes
    -----
    The regularised covariance is:
        
    .. math::
        
        (1 - text{shrinkage}) * \text{cov} + \text{shrinkage} * \mu * \text{np.identity(n_features)}
    
    where :math:`\mu = \text{trace(cov)} / n_{features}`
    and shrinkage is given by the Ledoit and Wolf formula
    
    See also
    ----------
        LedoitWolf
        
    References
    ----------
    "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices",
    Ledoit and Wolf, Journal of Multivariate Analysis, Volume 88, Issue 2,
    February 2004, pages 365-411.
    
    """
    return np.mean(cross_val_score(LedoitWolf(**kws), X))

def pcavsfa (
        X,
        #n_samples, n_features, 
        rank =10 , sigma =1. , n_components =5, 
        random_state = 42 , verbose =0 , view =False, 
  ):
    # options for n_components
    n_samples, n_features = len(X),  X.shape[1]
    n_components = np.arange(0, n_features, n_components) 
    
    rng = np.random.RandomState(random_state)
    U, _, _ = linalg.svd(rng.randn(n_features, n_features))
    
    #X = np.dot(rng.randn(n_samples, rank), U[:, :rank].T)
    X = np.dot(rng.randn(n_samples, n_features), U[:, :rank].T)
    # Adding homoscedastic noise
    X_homo = X + sigma * rng.randn(n_samples, n_features)

    # Adding heteroscedastic noise
    sigmas = sigma * rng.rand(n_features) + sigma / 2.
    X_hetero = X + rng.randn(n_samples, n_features) * sigmas


    for X, title in [(X_homo, 'Homoscedastic Noise'),
                     (X_hetero, 'Heteroscedastic Noise')]:
        pca_scores, fa_scores = compute_scores(X, n_features)
        n_components_pca = n_components[np.argmax(pca_scores)]
        n_components_fa = n_components[np.argmax(fa_scores)]
    
        pca = PCA(svd_solver='full', n_components='mle')
        pca.fit(X)
        n_components_pca_mle = pca.n_components_
        
        if verbose:
            print("best n_components by PCA CV = %d" % n_components_pca)
            print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
            print("best n_components by PCA MLE = %d" % n_components_pca_mle)
    
        
        plt.figure()
        plt.plot(n_components, pca_scores, 'b', label='PCA scores')
        plt.plot(n_components, fa_scores, 'r', label='FA scores')
        plt.axvline(rank, color='g', label='TRUTH: %d' % rank, linestyle='-')
        plt.axvline(n_components_pca, color='b',
                    label='PCA CV: %d' % n_components_pca, linestyle='--')
        plt.axvline(n_components_fa, color='r',
                    label='FactorAnalysis CV: %d' % n_components_fa,
                    linestyle='--')
        plt.axvline(n_components_pca_mle, color='k',
                    label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')
    
        # compare with other covariance estimators
        plt.axhline(shrunk_cov_score(X), color='violet',
                    label='Shrunk Covariance MLE', linestyle='-.')
        plt.axhline(LW_score(X), color='orange',
                    label='LedoitWolf MLE' % n_components_pca_mle, linestyle='-.')
    
        plt.xlabel('nb of components')
        plt.ylabel('CV scores')
        plt.legend(loc='lower right')
        plt.title(title)
    
    plt.show()
    return pca_scores, fa_scores
    
pcavsfa.__doc__="""\
Compute PCA score and Factor Analysis scores from training X and compare  
probabilistic PCA and Factor Analysis  models.
  
Parameters 
-----------
{params.X}

n_features: int, 
    number of features that composes X 
n_components: int, default {{5}}
    number of component to retrieve. 
rank: int, default{{10}}
    Bounding for ranking 
sigma: float, default {{1.}}
    data pertubator ratio for adding heteroscedastic noise
random_state: int , default {{42}}
    Determines random number generation for dataset shuffling. Pass an int
    for reproducible output across multiple function calls.
    
{params.verbose}

Returns 
---------
Tuple (pca_scores, fa_scores): 
    Scores from PCA and FA  from transformed X 
""".format(
    params =_core_docs["params"]
)    
    
    
def make_scedastic_data (
        n_samples= 1000, n_features=50, rank =  10, sigma=1., 
        random_state =42
   ): 
    """ Generate a sampling data for probabilistic PCA and Factor Analysis for  
    model comparison. 
    
    By default: 
        nsamples    = 1000 
        n_features  = 50  
        rank        =10 
        
    Returns 
    ----------
    * X: sampling data 
    * X_homo: sampling data with homoscedastic noise
    * X_hetero: sampling with heteroscedastic noise
    * n_components: number of components  50 features. 
    
    """
    # Create the data
    rng = np.random.RandomState(random_state )
    U, _, _ = linalg.svd(rng.randn(n_features, n_features))
    X = np.dot(rng.randn(n_samples, rank), U[:, :rank].T)
    
    # Adding homoscedastic noise
    X_homo = X + sigma * rng.randn(n_samples, n_features)
    
    # Adding heteroscedastic noise
    sigmas = sigma * rng.rand(n_features) + sigma / 2.
    X_hetero = X + rng.randn(n_samples, n_features) * sigmas

    # Fit the models
    n_components = np.arange(0, n_features, 5)  # options for n_components
    
    return X, X_homo, X_hetero , n_components
    
    
    
    
    
    