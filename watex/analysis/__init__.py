"""
Analysis sub-package is used for basic feature extraction, transformation and 
matrices covariance computations (:mod:`~watex.analysis.decomposition`). 
It also includes some dimensional reduction (:mod:`~watex.analysis.dimensionality`) 
and factor analysis from :mod:`~watex.analysis.factor`. 
"""
from .dimensionality import (   
    get_component_with_most_variance,
    plot_projection, 
    find_features_importances, 
    nPCA, 
    kPCA, 
    LLE, 
    iPCA, 
    )
from .decomposition import ( 
    extract_pca, 
    decision_region, 
    feature_transformation, 
    total_variance_ratio , 
    linear_discriminant_analysis
    )
from .factor import ( 
    LW_score, 
    shrunk_cov_score, 
    compute_scores, 
    pcavsfa, 
    make_scedastic_data, 
    )

__all__= [ 
    "nPCA", 
    "kPCA", 
    "LLE", 
    "iPCA",  
    "get_component_with_most_variance",
    "plot_projection", 
    "find_features_importances",  
    "extract_pca", 
    "decision_region", 
    "feature_transformation", 
    "total_variance_ratio" ,
    "linear_discriminant_analysis", 
    "LW_score", 
    "shrunk_cov_score", 
    "compute_scores", 
    "pcavsfa", 
    "make_scedastic_data", 
    ]

