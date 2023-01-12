.. _api_ref:

===============
API Reference
===============

This is the class and function reference of watex. Please refer to
the :ref:`full user guide <user_guide>` for further details, as the class and
function raw specifications may not be enough to give full guidelines on their
uses.

.. _analysis_ref:

:mod:`watex.analysis`: Analyses
=======================================================

The module is a set of feature extraction and selection, matrices decomposition and features analyses.

.. automodule:: watex.analysis
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`analysis <analysis>` section for further details.

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: function.rst

	analysis.LLE
	analysis.pcavsfa
	analysis.compute_scores
	analysis.decision_region
	analysis.extract_pca
	analysis.feature_transformation
	analysis.find_features_importances
	analysis.get_component_with_most_variance
	analysis.iPCA
	analysis.kPCA
	analysis.linear_discriminant_analysis
	analysis.lw_score
	analysis.make_scedastic_data
	analysis.nPCA
	analysis.plot_projection
	analysis.shrunk_cov_score
	analysis.total_variance_ratio
	
	
.. _base_ref:

:mod:`watex.base`: Base classes and functions
=======================================================

.. automodule:: watex.base
    :no-members:
    :no-inherited-members:
	
**User guide:** See the :ref:`bases <base>` section for further details.

Classes
~~~~~~~~~~~~

.. currentmodule:: watex

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   base.Data
   base.Missing
   base.AdalineStochasticGradientDescent
   base.AdalineGradientDescent
   base.GreedyPerceptron
   base.MajorityVoteClassifier
   base.SequentialBackwardSelection

Functions
~~~~~~~~~~~~

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: function.rst

   base.existfeatures
   base.get_params
   base.selectfeatures

.. _cases_ref:

:mod:`watex.cases`: Case Histories
==================================================

.. automodule:: watex.cases
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`case histories <cases>` section for further details.

Classes
~~~~~~~~~~~~

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: class.rst

   cases.modeling.BaseModel
   cases.prepare.BaseSteps
   cases.features.GeoFeatures
   cases.features.FeatureInspection
   cases.processing.Preprocessing
   cases.processing.Processing

Functions
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: function.rst

   cases.prepare.base_transform
   cases.prepare.default_pipeline
   cases.prepare.default_preparation
   
.. _datasets_ref:
     
:mod:`watex.datasets`: Datasets
=================================

.. automodule:: watex.datasets 
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`datasets <datasets>` section for further details.

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: function.rst

	datasets.fetch_data
	datasets.load_bagoue
	datasets.load_boundiali
	datasets.load_edis
	datasets.load_gbalo
	datasets.load_hlogs
	datasets.load_iris
	datasets.load_semien
	datasets.load_tankesse
	datasets.make_erp 
	datasets.make_ves 
	
.. _edi_ref:

:mod:`watex.edi`: Electrical Data Interchange 
==============================================

.. automodule:: watex.edi
   :no-members:
   :no-inherited-members:

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: class.rst

   edi.Edi 

.. _exceptions_ref:

:mod:`watex.exceptions`: Exceptions 
=====================================

.. automodule:: watex.exceptions
   :no-members:
   :no-inherited-members:

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: class.rst

	exceptions.AquiferGroupError
	exceptions.ArgumentError
	exceptions.ConfigError
	exceptions.CoordinateError
	exceptions.DatasetError
	exceptions.DepthError
	exceptions.EDIError
	exceptions.EMError
	exceptions.ERPError
	exceptions.EstimatorError
	exceptions.ExtractionError
	exceptions.FeatureError
	exceptions.FileHandlingError
	exceptions.FrequencyError
	exceptions.GISError
	exceptions.GeoArgumentError
	exceptions.GeoDatabaseError
	exceptions.GeoPropertyError
	exceptions.HeaderError
	exceptions.HintError
	exceptions.LearningError
	exceptions.NotFittedError
	exceptions.ParameterNumberError
	exceptions.PlotError
	exceptions.ProcessingError
	exceptions.ResistivityError
	exceptions.SQLError
	exceptions.SQLManagerError
	exceptions.ScikitLearnImportError
	exceptions.SiteError
	exceptions.StationError
	exceptions.StrataError
	exceptions.TipError
	exceptions.TopModuleError
	exceptions.VESError
	exceptions.kError

.. _externals_ref:

:mod:`watex.externals`: External Tensor Utilities 
===========================================================

.. automodule:: watex.externals
   :no-members:
   :no-inherited-members:

classes 
~~~~~~~~~

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: class.rst

   externals.z.Z
   externals.z.ResPhase
   externals.z.Tipper 
   externals.z.ZError 

Functions
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: function.rst

	externals.z.correct4sensor_orientation
	externals.zutils.invertmatrix_incl_errors
	externals.zutils.make_log_increasing_array
	externals.zutils.multiplymatrices_incl_errors
	externals.zutils.old_z_error2r_phi_error
	externals.zutils.propagate_error_polar2rect
	externals.zutils.propagate_error_rect2polar
	externals.zutils.rotatematrix_incl_errors
	externals.zutils.rotatevector_incl_errors

.. _geology_ref:

:mod:`watex.geology`: Geology 
===============================================

.. automodule:: watex.geology
   :no-members:
   :no-inherited-members:

Classes 
~~~~~~~~~~~~

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: class.rst

   geology.core.Base
   geology.database.DBSetting
   geology.database.GeoDataBase
   geology.drilling.Borehole
   geology.drilling.Drill
   geology.geology.Geology
   geology.geology.Structural
   geology.geology.Structures
   geology.stratigraphic.GeoStrataModel 

Functions
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: function.rst

	geology.geology.setstructures 
	geology.core.get_agso_properties
	geology.core.mapping_stratum
	geology.core.fetching_data_from_repo
	geology.core.set_agso_properties

	
.. _methods_ref:

:mod:`watex.methods`:  Methods
=========================================

.. automodule:: watex.methods
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`methods <methods>` section for further details.

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: class.rst

   methods.AqGroup
   methods.AqSection
   methods.DCProfiling
   methods.DCSounding
   methods.EM
   methods.ERP
   methods.ERPCollection
   methods.Hydrogeology
   methods.MXS
   methods.Logging
   methods.Processing 
   methods.ResistivityProfiling
   methods.VerticalSounding

.. autosummary::
   :toctree: generated/
   :template: function.rst

.. _metrics_ref:

:mod:`watex.metrics`: Metrics 
=====================================================

.. automodule:: watex.metrics
   :no-members:
   :no-inherited-members:

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.ROC_curve
   metrics.confusion_matrix
   metrics.precision_recall_tradeoff
   
   
.. _models_ref:

:mod:`watex.models`:  Models
=========================================

.. automodule:: watex.models
   :no-members:
   :no-inherited-members:


Classes 
~~~~~~~~~~~~

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: class.rst

   models.BaseEvaluation
   models.GridSearch
   models.GridSearchMultiple 
   models.pModels 

Functions 
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: function.rst

    models.displayCVTables
	models.displayFineTunedResults
	models.displayModelMaxDetails
	models.getGlobalScores
	models.getSplitBestScores
	models.get_best_kPCA_params
	models.get_scorers


.. _property_ref:

:mod:`watex.property`: Property 
=====================================================

.. automodule:: watex.property
   :no-members:
   :no-inherited-members:


.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: class.rst

   property.BagoueNotes
   property.BasePlot
   property.Config
   property.ElectricalMethods
   property.Copyright
   property.IsEdi
   property.Copyright
   property.P
   property.Person
   property.References
   property.Software
   property.Water
   
.. _sites_ref:

:mod:`watex.site`: Site and Location  
=====================================================

.. automodule:: watex.site
   :no-members:
   :no-inherited-members:

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: class.rst

   site.Location

.. _transformers_ref:

:mod:`watex.transformers`: Transformers  
=====================================================

.. automodule:: watex.transformers
   :no-members:
   :no-inherited-members:


.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: class.rst

   transformers.CategorizeFeatures 
   transformers.CombinedAttributesAdder
   transformers.DataFrameSelector 
   transformers.FeatureUnion
   transformers.StratifiedUsingBaseCategory
   transformers.StratifiedWithCategoryAdder
   
.. _utils_ref:

:mod:`watex.utils`: Utilities  
=====================================================

.. automodule:: watex.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: function.rst

    utils.bi_selector
	utils.cattarget
	utils.check_flow_objectivity
	utils.classify_k
	utils.correlatedfeatures
	utils.defineConductiveZone
	utils.erpSelector
	utils.evalModel
	utils.findCatandNumFeatures
	utils.find_aquifer_groups
	utils.find_similar_labels
	utils.fittensor
	utils.getGlobalScore
	utils.get_aquifer_section
	utils.get_aquifer_sections
	utils.get_compressed_vector
	utils.get_profile_angle
	utils.get_sections_from_depth
	utils.get_strike
	utils.get_unique_section
	utils.get_xs_xr_splits
	utils.interpolate1d
	utils.interpolate2d
	utils.label_importance
	utils.labels_validator
	utils.linkage_matrix
	utils.magnitude
	utils.makeCoords
	utils.make_MXS_labels
	utils.make_naive_pipe
	utils.moving_average
	utils.naive_imputer
	utils.naive_scaler
	utils.ohmicArea
	utils.plotAnomaly
	utils.plotOhmicArea
	utils.plot_clusters
	utils.plot_confusion_matrices
	utils.plot_cost_vs_epochs
	utils.plot_elbow
	utils.plot_learning_curves
	utils.plot_logging
	utils.plot_mlxtend_heatmap
	utils.plot_mlxtend_matrix
	utils.plot_naive_dendrogram
	utils.plot_pca_components
	utils.plot_regularization_path
	utils.plot_rf_feature_importances
	utils.plot_sbs_feature_selection
	utils.plot_silhouette
	utils.plot_yb_confusion_matrix
	utils.power
	utils.predict_NGA_labels
	utils.projection_validator
	utils.read_data
	utils.reduce_samples
	utils.rename_labels_in
	utils.reshape
	utils.rhoa2z
	utils.scalePosition
	utils.scaley
	utils.select_base_stratum
	utils.select_feature_importances
	utils.selectfeatures
	utils.sfi
	utils.shape
	utils.split_train_test
	utils.to_numeric_dtypes
	utils.smart_label_classifier
	utils.type_
	utils.vesDataOperator
	utils.vesSelector
	utils.z2rhoa
   
:mod:`~watex.utils.mlutils`: Additional learning utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

.. automodule:: watex.utils.mlutils
   :no-members:
   :no-inherited-members:

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: function.rst

	utils.mlutils.correlatedfeatures
	utils.mlutils.exporttarget
	utils.mlutils.existfeatures 
	utils.mlutils.getGlobalScore
	utils.mlutils.predict
	utils.mlutils.load_data 
	utils.mlutils.test_set_check_id
	utils.mlutils.split_train_test_by_id
	utils.mlutils.discretizeCategoriesforStratification
	utils.mlutils.stratifiedUsingDiscretedCategories
	utils.mlutils.fetch_model
	utils.mlutils.dumpOrSerializeData
	utils.mlutils.loadDumpedOrSerializedData
	utils.mlutils.default_data_splitting
	utils.mlutils.fetchModel 
	utils.mlutils.cattarget 
	
:mod:`~watex.utils.plotutils`: Additional plot-utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: watex.utils.plotutils
   :no-members:
   :no-inherited-members:

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: function.rst

	utils.plotutils.plot_confusion_matrix
	utils.plotutils.savefigure
	utils.plotutils.make_mpl_properties
	utils.plotutils.resetting_colorbar_bound
	utils.plotutils.get_color_palette
	utils.plotutils.plot_bar 
	
:mod:`~watex.utils.exmath`: Additional math-utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: watex.utils.exmath
   :no-members:
   :no-inherited-members:

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: function.rst

	utils.exmath.d_hanning_window
	utils.exmath.betaj
	utils.exmath.savitzky_golay1d 
	utils.exmath.dummy_basement_curve
	utils.exmath.find_limit_for_integration
	utils.exmath.find_bound_for_integration
	utils.exmath.fitfunc
	utils.exmath.invertVES
	utils.exmath.detect_station_position
	utils.exmath.convert_distance_to_m
	utils.exmath.get_station_number 
	utils.exmath.define_conductive_zone
	utils.exmath.compute_anr
	utils.exmath.compute_lower_anomaly
	utils.exmath.select_anomaly
	utils.exmath.define_anomaly 
	utils.exmath.get_profile_angle
	utils.exmath.savgol_coeffs
	utils.exmath.savgol_filter
	
:mod:`~watex.utils.coreutils`: Additional core-utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: watex.utils.coreutils
   :no-members:
   :no-inherited-members:

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: function.rst

	utils.coreutils.fill_coordinates
	utils.coreutils.is_erp_series
	utils.coreutils.is_erp_dataframe
	utils.coreutils.dummy_basement_curve
	utils.coreutils.parseDCArgs
	
:mod:`~watex.utils.geotools`: Geology utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: watex.utils.geotools
   :no-members:
   :no-inherited-members:

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: function.rst

	utils.geotools.assert_station
	utils.geotools.base_log
	utils.geotools.fit_rocks
	utils.geotools.fit_stratum_property
	utils.geotools.get_s_thicknesses
	utils.geotools.grouped_items
	utils.geotools.lns_and_tres_split
	utils.geotools.map_bottom
	utils.geotools.map_top
	utils.geotools.set_default_hatch_color_values
	utils.geotools.smart_zoom
	utils.geotools.zoom_processing

.. _view_ref:


:mod:`watex.view`: Plotting 
===============================

.. automodule:: watex.view
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`visualization <view>` section for further details.

Classes
~~~~~~~~~~~~

.. currentmodule:: watex


.. autosummary::
   :toctree: generated/
   :template: class.rst

   view.ExPlot
   view.QuickPlot
   view.TPlot 
   view.EvalPlot
   
Functions
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: function.rst

    view.biPlot
	view.plotDendrogram
	view.plotDendroheat
	view.plotLearningInspection
	view.plotLearningInspections
	view.plotModel
	view.plotProjection
	view.plotSilhouette
	view.plot_matshow
	view.plot_model_scores
	view.plot_reg_scoring
	view.pobj
	view.viewtemplate

Click :doc:`here <api/watex>` to visit all the modules of the package.