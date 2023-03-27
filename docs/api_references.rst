
.. _api_ref:

===============
API Reference
===============

This is the class and function reference of :code:`watex`. Please refer to
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
	analysis.LW_score
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
~~~~~~~~~~~

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
~~~~~~~~~~

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: function.rst

   base.existfeatures
   base.get_params
   base.selectfeatures

.. _cases_ref:

:mod:`watex.cases`: Case Histories
===================================

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
	datasets.load_huayuan
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
	exceptions.ConfigError
	exceptions.CoordinateError
	exceptions.DatasetError
	exceptions.DCError
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
	exceptions.GeoDatabaseError
	exceptions.GeoPropertyError
	exceptions.HeaderError
	exceptions.LearningError
	exceptions.NotFittedError
	exceptions.ParameterNumberError
	exceptions.PlotError
	exceptions.ProcessingError
	exceptions.ResistivityError
	exceptions.SQLError
	exceptions.SiteError
	exceptions.StationError
	exceptions.StrataError
	exceptions.VESError
	exceptions.ZError
	exceptions.kError

.. _externals_ref:

:mod:`watex.externals`: Tensors 
===================================

.. automodule:: watex.externals
   :no-members:
   :no-inherited-members:

Classes 
~~~~~~~~~

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: class.rst

   externals.z.Z
   externals.z.ResPhase
   externals.z.Tipper 

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
================================

.. automodule:: watex.geology
   :no-members:
   :no-inherited-members:

Classes 
~~~~~~~~

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

Classes 
~~~~~~~~~~~~

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
   methods.ZC


Functions
~~~~~~~~~~~~~~~

:mod:`~watex.methods.electrical`: DC-Resistivity 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: watex.methods.electrical 
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`DC-resistivity <dc_resistivity>` section for further details.

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: function.rst

	methods.electrical.DCProfiling.summary 
	methods.electrical.DCSounding.summary 
	methods.electrical.ResistivityProfiling.summary 
	methods.electrical.ResistivityProfiling.plotAnomaly 
	methods.electrical.VerticalSounding.summary 
	methods.electrical.VerticalSounding.plotOhmicArea
    methods.electrical.VerticalSounding.invert 	

:mod:`~watex.methods.em`: EM - EMAP: short-periods Processing 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
.. automodule:: watex.methods.em 
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`EM tensors processing <em>` section for further details.

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: function.rst

	methods.em.EM.rewrite 
	methods.em.EM.getfullfrequency 
	methods.em.EM.make2d 
	methods.em.EM.getreferencefrequency 
	methods.em.EM.exportedis
	methods.em.Processing.tma 
	methods.em.Processing.flma 
	methods.em.Processing.ama 
	methods.em.Processing.skew 
	methods.em.Processing.zrestore 
	methods.em.Processing.freqInterpolation 
	methods.em.Processing.interpolate_z
	methods.em.Processing.drop_frequencies
	methods.em.Processing.controlFrequencyBuffer 
	methods.em.Processing.getValidTensors
	methods.em.Processing.qc 
	methods.em.ZC.get_ss_correction_factors
	methods.em.ZC.remove_distortion
	methods.em.ZC.remove_ss_emap
	methods.em.ZC.remove_static_shift


:mod:`~watex.methods.hydro`: Hydrogeology 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
.. automodule:: watex.methods.hydro 
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`hydrogeology <hydrogeology>` section for further details.

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: function.rst

	methods.hydro.HData.squeeze_data 
	methods.hydro.HData.get_base_stratum 
	methods.hydro.AqSection.findSection 
	methods.hydro.MXS.predictNGA 
	methods.hydro.MXS.makeyMXS 
	methods.hydro.MXS.labelSimilarity 
	methods.hydro.Logging.plot 
	methods.hydro.AqGroup.findGroups 
	
Refer to :mod:`~watex.utils.hydroutils` to get many other utilities 
for hydro-parameters calculation. 

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

   metrics.confusion_matrix
   metrics.get_eval_scores
   metrics.get_metrics
   metrics.precision_recall_tradeoff
   metrics.ROC_curve
   
   
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
	models.naive_evaluation


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
   property.P
   property.Person
   property.References
   property.Software
   property.Water
   
.. _sites_ref:

:mod:`watex.site`: Location and Profile  
=============================================

.. automodule:: watex.site
   :no-members:
   :no-inherited-members:

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: class.rst

   site.Location
   site.Profile

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
   transformers.FrameUnion
   transformers.StratifiedUsingBaseCategory
   transformers.StratifiedWithCategoryAdder
   
.. _utils_ref:

:mod:`watex.utils`: Utilities  
=====================================================

.. automodule:: watex.utils
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`utilities <utilities>` section for further details.

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
	utils.erpSmartDetector
	utils.evalModel
	utils.findCatandNumFeatures
	utils.find_aquifer_groups
	utils.find_similar_labels
	utils.fittensor
	utils.getGlobalScore
	utils.get2dtensor
	utils.get_aquifer_section
	utils.get_aquifer_sections
	utils.get_compressed_vector
	utils.get_bearing
	utils.get_distance
	utils.get_full_frequency
	utils.get_profile_angle
	utils.get_sections_from_depth
	utils.get_strike
	utils.get_target
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
	utils.normalizer
	utils.ohmicArea
	utils.plotAnomaly
	utils.plotOhmicArea
	utils.plot_clusters
	utils.plot_confidence_in
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
	utils.plot_skew
	utils.plot_strike
	utils.plot_yb_confusion_matrix
	utils.power
	utils.predict_NGA_labels
	utils.projection_validator
	utils.qc
	utils.read_data
	utils.reduce_samples
	utils.remove_outliers
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
	utils.plotutils.plot_profiling
	
:mod:`~watex.utils.exmath`: Additional math-utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: watex.utils.exmath
   :no-members:
   :no-inherited-members:

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: function.rst

	utils.exmath.betaj
	utils.exmath.compute_lower_anomaly
	utils.exmath.d_hanning_window
	utils.exmath.define_anomaly
	utils.exmath.define_conductive_zone
	utils.exmath.detect_station_position
	utils.exmath.dummy_basement_curve
	utils.exmath.find_limit_for_integration
	utils.exmath.find_bound_for_integration
	utils.exmath.fitfunc
	utils.exmath.get_anomaly_ratio
	utils.exmath.get_profile_angle
	utils.exmath.get_station_number 
	utils.exmath.get_z_from
	utils.exmath.invertVES
	utils.exmath.plot_sfi
	utils.exmath.savgol_coeffs
	utils.exmath.savgol_filter
	utils.exmath.savitzky_golay1d 
	utils.exmath.select_anomaly

	
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
	utils.coreutils.parseDCArgs


:mod:`~watex.utils.funcutils`: Other utilities 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The section exposes some additional tools expected to be 
useful for manipulating data or to be incorporated to
a third-party Python package. The list is not exhaustive. Get more 
utilities by consulting the whole :mod:`~watex.utils.funcutils` module. 

.. automodule:: watex.utils.funcutils
   :no-members:
   :no-inherited-members:

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: function.rst

	utils.funcutils.is_depth_in
	utils.funcutils.map_specific_columns
	utils.funcutils.find_by_regex
	utils.funcutils.is_in_if
	utils.funcutils.to_hdf5
	utils.funcutils.sanitize_frame_cols
	utils.funcutils.str2columns
	utils.funcutils.random_state_validator
	utils.funcutils.move_cfile
	utils.funcutils.pretty_printer
	utils.funcutils.get_config_fname_from_varname
	utils.funcutils.get_confidence_ratio
	utils.funcutils.cparser_manager
	utils.funcutils.parse_yaml
	utils.funcutils.return_ctask
	utils.funcutils.parse_csv
	utils.funcutils.fetch_json_data_from_url
	utils.funcutils.parse_json
	utils.funcutils.assert_doi
	utils.funcutils.station_id
	utils.funcutils.concat_array_from_list
	utils.funcutils.show_stats
	utils.funcutils.make_ids
	utils.funcutils.fit_by_ll
	utils.funcutils.fillNaN
	utils.funcutils.find_close_position
	utils.funcutils.make_arr_consistent
	utils.funcutils.ismissing
	utils.funcutils.reshape
	utils.funcutils.save_job
	utils.funcutils.load_serialized_data
	utils.funcutils.serialize_data
	utils.funcutils.sanitize_unicode_string
	utils.funcutils.cpath
	utils.funcutils.smart_format
	utils.funcutils.read_from_excelsheets
	utils.funcutils.accept_types
	utils.funcutils.smart_strobj_recognition
	utils.funcutils.is_installing
	utils.funcutils.shrunkformat
	utils.funcutils.url_checker
	utils.funcutils.parse_attrs
	utils.funcutils.listing_items_format
	utils.funcutils.zip_extractor
	
	
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
~~~~~~~~~

.. currentmodule:: watex


.. autosummary::
   :toctree: generated/
   :template: class.rst

   view.ExPlot
   view.EvalPlot
   view.QuickPlot
   view.TPlot 

Functions
~~~~~~~~~~~~

:mod:`~watex.view.plot`: T-E-Q- Plots  
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: watex.view.plot 
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`view  <view>` section for further details.

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: function.rst

    view.ExPlot.plotbv
	view.ExPlot.plotcutcomparison
	view.ExPlot.plothist
	view.ExPlot.plothistvstarget
	view.ExPlot.plotjoint
	view.ExPlot.plotmissing
	view.ExPlot.plotpairgrid
	view.ExPlot.plotpairwisecomparison
	view.ExPlot.plotparallelcoords
	view.ExPlot.plotradviz
    view.ExPlot.plotscatter
	view.QuickPlot.barcatdist
	view.QuickPlot.corrmatrix
	view.QuickPlot.discussingfeatures
	view.QuickPlot.histcatdist
	view.QuickPlot.joint2features
	view.QuickPlot.multicatdist
	view.QuickPlot.naiveviz
	view.QuickPlot.numfeatures
	view.QuickPlot.scatteringfeatures
	view.TPlot.plotSkew
	view.TPlot.plot_corrections
	view.TPlot.plot_ctensor2d 
	view.TPlot.plot_multi_recovery 
	view.TPlot.plot_recovery 
	view.TPlot.plot_phase_tensors 
    view.TPlot.plot_rhoa
	view.TPlot.plot_tensor2d
	

:mod:`~watex.view.mlplot`: Learning Plots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
.. automodule:: watex.view.mlplot 
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`view  <view>` section for further details.

.. currentmodule:: watex

.. autosummary::
   :toctree: generated/
   :template: function.rst
.. autosummary::
   :toctree: generated/
   :template: function.rst

    view.EvalPlot.plotPCA
	view.EvalPlot.plotPR
	view.EvalPlot.plotROC
	view.EvalPlot.plotConfusionMatrix
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
	view.plot2d
	view.pobj
	view.viewtemplate

To browse the available modules of the software, click on :doc:`long description <modules/wx_apidoc/watex>` . 