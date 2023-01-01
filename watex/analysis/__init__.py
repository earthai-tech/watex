
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
    lw_score, 
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
    "lw_score", 
    "shrunk_cov_score", 
    "compute_scores", 
    "pcavsfa", 
    "make_scedastic_data", 
    ]

