"""
Utils sub-package offers several tools for data handling, parameters computation 
models estimation and evalution, and graphs visualization. The extension of the 
mathematical concepts, and the core of program are performed via the modules 
:mod:`~watex.utils.exmath` and :mod:`~watex.utils.coreutils` respectively. Whereas
the machine learning utilities and additional functionalities are performed 
with :mod:`~watex.utils.mlutils` and :mod:`~watex.utils.funcutils` respectively. 
The plot utilities from :mod:`~watex.utils.plotutils` gives several plotting 
tools for visualization.
"""

from .coreutils import ( 
    plotAnomaly, 
    vesSelector, 
    erpSelector, 
    defineConductiveZone,
    makeCoords,
    read_data,
    erpSmartDetector
    )
from .exmath import ( 
    type_,
    shape, 
    power, 
    magnitude, 
    sfi, 
    ohmicArea, 
    vesDataOperator, 
    scalePosition,
    rhoa2z, 
    z2rhoa, 
    interpolate1d, 
    interpolate2d,
    scaley, 
    fittensor, 
    get2dtensor,
    get_full_frequency, 
    get_strike, 
    get_profile_angle, 
    moving_average, 
    linkage_matrix, 
    plotOhmicArea, 
    plot_confidence_in,
    plot_sfi
    )
from .funcutils import ( 
    reshape, 
    to_numeric_dtypes, 
    smart_label_classifier, 
    remove_outliers,
    normalizer, 
    )
from .hydroutils import ( 
    select_base_stratum , 
    get_aquifer_section , 
    get_aquifer_sections, 
    get_unique_section, 
    get_compressed_vector, 
    get_xs_xr_splits, 
    reduce_samples , 
    get_sections_from_depth, 
    check_flow_objectivity, 
    make_MXS_labels, 
    predict_NGA_labels, 
    find_aquifer_groups, 
    find_similar_labels, 
    classify_k, 
    label_importance
    )

from .plotutils import ( 
    plot_mlxtend_heatmap , 
    plot_mlxtend_matrix, 
    plot_cost_vs_epochs, 
    plot_elbow, 
    plot_clusters, 
    plot_pca_components, 
    plot_naive_dendrogram, 
    plot_learning_curves, 
    plot_confusion_matrices, 
    plot_yb_confusion_matrix, 
    plot_sbs_feature_selection, 
    plot_regularization_path, 
    plot_rf_feature_importances, 
    plot_logging, 
    plot_silhouette, 
    plot_profiling, 
    plot_skew, 
    plot_strike,
    )
# to fix circular 
# import
try : 
    from .mlutils import ( 
        selectfeatures, 
        getGlobalScore, 
        split_train_test, 
        correlatedfeatures, 
        findCatandNumFeatures,
        evalModel, 
        cattarget, 
        labels_validator, 
        projection_validator, 
        rename_labels_in , 
        naive_imputer, 
        naive_scaler, 
        select_feature_importances, 
        make_naive_pipe, 
        bi_selector, 
        )
except ImportError :pass 

__all__=[
        'plotAnomaly', 
        'vesSelector', 
        'erpSelector', 
        'defineConductiveZone',
        'erpSmartDetector', 
        'makeCoords', 
        'type_',
        'shape', 
        'power', 
        'magnitude', 
        'sfi', 
        'ohmicArea', 
        'vesDataOperator', 
        'scalePosition',
        'rhoa2z', 
        'z2rhoa', 
        'interpolate1d', 
        'interpolate2d',
        'scaley', 
        'fittensor', 
        'get2dtensor',
        'get_full_frequency', 
        'selectfeatures', 
        'getGlobalScore',  
        'split_train_test', 
        'correlatedfeatures', 
        'findCatandNumFeatures',
        'evalModel',
        'get_strike', 
        'get_profile_angle', 
        'moving_average', 
        'linkage_matrix',
        'plotOhmicArea', 
        'reshape', 
        'to_numeric_dtypes' , 
        'smart_label_classifier', 
        'cattarget', 
        'labels_validator', 
        'projection_validator', 
        'rename_labels_in', 
        'read_data', 
        'plot_mlxtend_heatmap' , 
        'plot_mlxtend_matrix', 
        'plot_cost_vs_epochs', 
        'plot_elbow', 
        'plot_clusters',
        'plot_confidence_in', 
        'plot_pca_components' , 
        'plot_naive_dendrogram', 
        'plot_learning_curves', 
        'plot_confusion_matrices', 
        'plot_yb_confusion_matrix',
        'plot_sbs_feature_selection', 
        'plot_regularization_path', 
        'plot_rf_feature_importances', 
        'plot_logging',
        'plot_silhouette', 
        'select_base_stratum' , 
        'get_aquifer_sections' , 
        'get_aquifer_section', 
        'get_unique_section', 
        'get_compressed_vector', 
        'get_xs_xr_splits', 
        'reduce_samples' , 
        'get_sections_from_depth', 
        'check_flow_objectivity', 
        'naive_imputer', 
        'naive_scaler', 
        'select_feature_importances',
        'make_naive_pipe',
        'bi_selector',
        'make_MXS_labels', 
        'predict_NGA_labels', 
        'find_aquifer_groups', 
        'find_similar_labels', 
        'classify_k',
        'label_importance', 
        'plot_profiling', 
        'plot_sfi',
        'plot_skew',
        'remove_outliers', 
        'normalizer',
        'plot_strike',
        ]



