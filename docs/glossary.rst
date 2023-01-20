.. _glossary: 
    
    
==================
Glossary 
==================

This glossary gives some explicit conventions applied in watex and its API, 
while providing a reference for users. The concepts are structured into 
the related terms  :ref:`glossary_abbreviations`, :ref:`glossary_parameters`, :ref:`glossary_plot_properties`, 
:ref:`glossary_miscellaneous`, and :ref:`glossary_reference_links`. 


.. _glossary_abbreviations:

Abbreviations 
===============

.. glossary::  
		
	ohmS
		Pseudo-area of the fractured zone 
	sfi
		Pseudo-fracturing index 
	VES
		Vertical Electrical Sounding 
	ERP
		Electrical Resistivity Profiling 
	MT 
		Magnetetolluric 
	AMT 
		Audio-Magnetotellurics 
	CSAMT 
		Controlled Source Audio-Magnetotellurics 
		
	NSAMT 
		Natural Source Audio-Magnetotellurics 
	EM
		Electromagnetic 
	EMAP 
		Electromagnetic array profiling 

.. _glossary_general_concepts: 

General Concepts
=================

This is a few concepts to understand conventional words using machine learning. See additional concepts for 
more consistent in Scikit-learn `glossary <https://scikit-learn.org/stable/glossary.html>`_.
 
.. glossary::

    1d
    1d array
        One-dimensional array. A NumPy array whose ``.shape`` has length 1.
        A vector.

    2d
    2d array
        Two-dimensional array. A NumPy array whose ``.shape`` has length 2.
        Often represents a matrix.

    API
        Refers to both the *specific* interfaces for estimators implemented in
        watex and the *generalized* conventions across types of
        estimators as described in this glossary and :ref:`overviewed in the
        contributor documentation <api_overview>`.

        The specific interfaces that constitute watex's public API are
        largely documented in :ref:`api_ref`. However, we less formally consider
        anything as public API if none of the identifiers required to access it
        begins with ``_``.  We generally try to maintain :term:`backwards
        compatibility` for all objects in the public API.

        Private API, including functions, modules and methods beginning ``_``
        are not assured to be stable.

    array-like
        The most common data format for *input* to watex estimators and
        functions, array-like is any type object for which
        :func:`numpy.asarray` will produce an array of appropriate shape
        (usually 1 or 2-dimensional) of appropriate dtype (usually numeric).

        This includes:

        * a numpy array
        * a list of numbers
        * a list of length-k lists of numbers for some fixed length k
        * a :class:`pandas.DataFrame` with all columns numeric
        * a numeric :class:`pandas.Series`

        It excludes:

        * a :term:`sparse matrix`
        * an iterator
        * a generator

        Note that *output* from scikit-learn estimators and functions (e.g.
        predictions) should generally be arrays or sparse matrices, or lists
        thereof (as in multi-output :class:`tree.DecisionTreeClassifier`'s
        ``predict_proba``). An estimator where ``predict()`` returns a list or
        a `pandas.Series` is not valid.

    attribute
    attributes
        We mostly use attribute to refer to how model information is stored on
        an estimator during fitting.  Any public attribute stored on an
        estimator instance is required to begin with an alphabetic character
        and end in a single underscore if it is set in :term:`fit` or
        :term:`partial_fit`.  These are what is documented under an estimator's
        *Attributes* documentation.  The information stored in attributes is
        usually either: sufficient statistics used for prediction or
        transformation; :term:`transductive` outputs such as :term:`labels_` or
        :term:`embedding_`; or diagnostic data, such as
        :term:`feature_importances_`.
        Common attributes are listed :ref:`below <glossary_attributes>`.

        A public attribute may have the same name as a constructor
        :term:`parameter`, with a ``_`` appended.  This is used to store a
        validated or estimated version of the user's input. For example,
        :class:`decomposition.PCA` is constructed with an ``n_components``
        parameter. From this, together with other parameters and the data,
        PCA estimates the attribute ``n_components_``.

        Further private attributes used in prediction/transformation/etc. may
        also be set when fitting.  These begin with a single underscore and are
        not assured to be stable for public access.

        A public attribute on an estimator instance that does not end in an
        underscore should be the stored, unmodified value of an ``__init__``
        :term:`parameter` of the same name.  Because of this equivalence, these
        are documented under an estimator's *Parameters* documentation.

    backwards compatibility
        We generally try to maintain backward compatibility (i.e. interfaces
        and behaviors may be extended but not changed or removed) from release
        to release but this comes with some exceptions:

        Public API only
            The behavior of objects accessed through private identifiers
            (those beginning ``_``) may be changed arbitrarily between
            versions.
        As documented
            We will generally assume that the users have adhered to the
            documented parameter types and ranges. If the documentation asks
            for a list and the user gives a tuple, we do not assure consistent
            behavior from version to version.
        Deprecation
            Behaviors may change following a :term:`deprecation` period
            (usually two releases long).  Warnings are issued using Python's
            :mod:`warnings` module.
        Keyword arguments
            We may sometimes assume that all optional parameters (other than X
            and y to :term:`fit` and similar methods) are passed as keyword
            arguments only and may be positionally reordered.
        Bug fixes and enhancements
            Bug fixes and -- less often -- enhancements may change the behavior
            of estimators, including the predictions of an estimator trained on
            the same data and :term:`random_state`.  When this happens, we
            attempt to note it clearly in the changelog.
        Serialization
            We make no assurances that pickling an estimator in one version
            will allow it to be unpickled to an equivalent model in the
            subsequent version. 

        Despite this informal contract with our users, the software is provided
        as is, as stated in the license.  When a release inadvertently
        introduces changes that are not backward compatible, these are known
        as software regressions.

    callable
        A function, class or an object which implements the ``__call__``
        method; anything that returns True when the argument of `callable()
        <https://docs.python.org/3/library/functions.html#callable>`_.

    categorical feature
        A categorical or nominal :term:`feature` is one that has a
        finite set of discrete values across the population of data.
        These are commonly represented as columns of integers or
        strings. Strings will be rejected by most scikit-learn
        estimators, and integers will be treated as ordinal or
        count-valued. For the use with most estimators, categorical
        variables should be one-hot encoded. Notable exceptions include
        tree-based models such as random forests and gradient boosting
        models that often work better and faster with integer-coded
        categorical variables.

    deprecation
        We use deprecation to slowly violate our :term:`backwards
        compatibility` assurances, usually to:

        * change the default value of a parameter; or
        * remove a parameter, attribute, method, class, etc.

        We will ordinarily issue a warning when a deprecated element is used,
        although there may be limitations to this.  For instance, we will raise
        a warning when someone sets a parameter that has been deprecated, but
        may not when they access that parameter's attribute on the estimator
        instance.

        See the :ref:`Contributors' Guide <contributing_deprecation>`.

    dimensionality
        May be used to refer to the number of :term:`features` (i.e.
        :term:`n_features`), or columns in a 2d feature matrix.
        Dimensions are, however, also used to refer to the length of a NumPy
        array's shape, distinguishing a 1d array from a 2d matrix.

    docstring
        The embedded documentation for a module, class, function, etc., usually
        in code as a string at the beginning of the object's definition, and
        accessible as the object's ``__doc__`` attribute.

        We try to adhere to `PEP257
        <https://www.python.org/dev/peps/pep-0257/>`_, and follow `NumpyDoc
        conventions <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

    double underscore
    double underscore notation
        When specifying parameter names for nested estimators, ``__`` may be
        used to separate between parent and child in some contexts. The most
        common use is when setting parameters through a meta-estimator with
        :term:`set_params` and hence in specifying a search grid in
        :ref:`parameter search <grid_search>`. See :term:`parameter`.
        It is also used in :meth:`pipeline.Pipeline.fit` for passing
        :term:`sample properties` to the ``fit`` methods of estimators in
        the pipeline.

    dtype
    data type
        NumPy arrays assume a homogeneous data type throughout, available in
        the ``.dtype`` attribute of an array (or sparse matrix). We generally
        assume simple data types for scikit-learn data: float or integer.
        We may support object or string data types for arrays before encoding
        or vectorizing.  Our estimators do not work with struct arrays, for
        instance.

        Our documentation can sometimes give information about the dtype
        precision, e.g. `np.int32`, `np.int64`, etc. When the precision is
        provided, it refers to the NumPy dtype. If an arbitrary precision is
        used, the documentation will refer to dtype `integer` or `floating`.
        Note that in this case, the precision can be platform dependent.
        The `numeric` dtype refers to accepting both `integer` and `floating`.


    early stopping
        This consists in stopping an iterative optimization method before the
        convergence of the training loss, to avoid over-fitting. This is
        generally done by monitoring the generalization score on a validation
        set. When available, it is activated through the parameter
        ``early_stopping`` or by setting a positive :term:`n_iter_no_change`.

    estimator instance
        We sometimes use this terminology to distinguish an :term:`estimator`
        class from a constructed instance. For example, in the following,
        ``cls`` is an estimator class, while ``est1`` and ``est2`` are
        instances::

            cls = RandomForestClassifier
            est1 = cls()
            est2 = RandomForestClassifier()

    examples
        We try to give examples of basic usage for most functions and
        classes in the API:

        * as doctests in their docstrings (i.e. within the ``watex/`` package
          code itself).
        * as examples in the :ref:`example gallery <general_examples>`
          rendered (using `sphinx-gallery
          <https://sphinx-gallery.readthedocs.io/>`_) from scripts in the
          ``examples/`` directory, exemplifying key features or parameters
          of the estimator/function.  These should also be referenced from the
          User Guide.
        * sometimes in the :ref:`User Guide <user_guide>` (built from ``doc/``)
          alongside a technical description of the estimator.

    experimental
        An experimental tool is already usable but its public API, such as
        default parameter values or fitted attributes, is still subject to
        change in future versions without the usual :term:`deprecation`
        warning policy.

    evaluation metric
    evaluation metrics
        Evaluation metrics give a measure of how well a model performs.  We may
        use this term specifically to refer to the functions in :mod:`metrics`
        (disregarding :mod:`metrics.pairwise`), as distinct from the
        :term:`score` method and the :term:`scoring` API used in cross
        validation. See :ref:`model_evaluation`.

        These functions usually accept a ground truth (or the raw data
        where the metric evaluates clustering without a ground truth) and a
        prediction, be it the output of :term:`predict` (``y_pred``),
        of :term:`predict_proba` (``y_proba``), or of an arbitrary score
        function including :term:`decision_function` (``y_score``).
        Functions are usually named to end with ``_score`` if a greater
        score indicates a better model, and ``_loss`` if a lesser score
        indicates a better model.  This diversity of interface motivates
        the scoring API.

        Note that some estimators can calculate metrics that are not included
        in :mod:`metrics` and are estimator-specific, notably model
        likelihoods.


    feature
    features
    feature vector
        In the abstract, a feature is a function (in its mathematical sense)
        mapping a sampled object to a numeric or categorical quantity.
        "Feature" is also commonly used to refer to these quantities, being the
        individual elements of a vector representing a sample. In a data
        matrix, features are represented as columns: each column contains the
        result of applying a feature function to a set of samples.

        Elsewhere features are known as attributes, predictors, regressors, or
        independent variables.

        Nearly all estimators in scikit-learn assume that features are numeric,
        finite and not missing, even when they have semantically distinct
        domains and distributions (categorical, ordinal, count-valued,
        real-valued, interval). See also :term:`categorical feature` and
        :term:`missing values`.

        ``n_features`` indicates the number of features in a dataset.

    fitting
        Calling :term:`fit` (or :term:`fit_transform`, :term:`fit_predict`,
        etc.) on an estimator.

    fitted
        The state of an estimator after :term:`fitting`.

        There is no conventional procedure for checking if an estimator
        is fitted.  However, an estimator that is not fitted:

        * should raise :class:`exceptions.NotFittedError` when a prediction
          method (:term:`predict`, :term:`transform`, etc.) is called.
          (:func:`utils.validation.check_is_fitted` is used internally
          for this purpose.)
        * should not have any :term:`attributes` beginning with an alphabetic
          character and ending with an underscore. (Note that a descriptor for
          the attribute may still be present on the class, but hasattr should
          return False)

    function
        We provide ad hoc function interfaces for many algorithms, while
        :term:`estimator` classes provide a more consistent interface.

        In particular, watex may provide a function interface that fits
        a model to some data and returns the learnt model parameters, as in
        :func:`linear_model.enet_path`.  For transductive models, this also
        returns the embedding or cluster labels, as in
        :func:`manifold.spectral_embedding` or :func:`cluster.dbscan`.  Many
        preprocessing transformers also provide a function interface, akin to
        calling :term:`fit_transform`, as in
        :func:`preprocessing.maxabs_scale`.  Users should be careful to avoid
        :term:`data leakage` when making use of these
        ``fit_transform``-equivalent functions.

        We do not have a strict policy about when to or when not to provide
        function forms of estimators, but maintainers should consider
        consistency with existing interfaces, and whether providing a function
        would lead users astray from best practices (as regards data leakage,
        etc.)

    gallery
        See :term:`examples`.

    hyperparameter
    hyper-parameter
        See :term:`parameter`.

    impute
    imputation
        Most machine learning algorithms require that their inputs have no
        :term:`missing values`, and will not work if this requirement is
        violated. Algorithms that attempt to fill in (or impute) missing values
        are referred to as imputation algorithms.

    indexable
        An :term:`array-like`, :term:`sparse matrix`, pandas DataFrame or
        sequence (usually a list).

    induction
    inductive
        Inductive (contrasted with :term:`transductive`) machine learning
        builds a model of some data that can then be applied to new instances.
        Most estimators in watex are inductive, having :term:`predict`
        and/or :term:`transform` methods.

    joblib
        A Python library (https://joblib.readthedocs.io) used in watex to
        facilite simple parallelism and caching.  Joblib is oriented towards
        efficiently working with numpy arrays, such as through use of
        :term:`memory mapping`. See :ref:`parallelism` for more
        information.

    label indicator matrix
    multilabel indicator matrix
    multilabel indicator matrices
        The format used to represent multilabel data, where each row of a 2d
        array or sparse matrix corresponds to a sample, each column
        corresponds to a class, and each element is 1 if the sample is labeled
        with the class and 0 if not.

    leakage
    data leakage
        A problem in cross validation where generalization performance can be
        over-estimated since knowledge of the test data was inadvertently
        included in training a model.  This is a risk, for instance, when
        applying a :term:`transformer` to the entirety of a dataset rather
        than each training portion in a cross validation split.

        We aim to provide interfaces (such as :mod:`pipeline` and
        :mod:`model_selection`) that shield the user from data leakage.

    memmapping
    memory map
    memory mapping
        A memory efficiency strategy that keeps data on disk rather than
        copying it into main memory.  Memory maps can be created for arrays
        that can be read, written, or both, using :obj:`numpy.memmap`. When
        using :term:`joblib` to parallelize operations in watex, it
        may automatically memmap large arrays to reduce memory duplication
        overhead in multiprocessing.

    missing values
        Most watex estimators do not work with missing values. When they
        do (e.g. in :class:`impute.SimpleImputer`), NaN is the preferred
        representation of missing values in float arrays.  If the array has
        integer dtype, NaN cannot be represented. For this reason, we support
        specifying another ``missing_values`` value when :term:`imputation` or
        learning can be performed in integer space.
        :term:`Unlabeled data <unlabeled data>` is a special case of missing
        values in the :term:`target`.

    ``n_features``
        The number of :term:`features`.

    ``n_outputs``
        The number of :term:`outputs` in the :term:`target`.

    ``n_samples``
        The number of :term:`samples`.

    ``n_targets``
        Synonym for :term:`n_outputs`.

    narrative docs
    narrative documentation
        An alias for :ref:`User Guide <user_guide>`, i.e. documentation written
        in ``doc/modules/``. Unlike the :ref:`API reference <api_ref>` provided
        through docstrings, the User Guide aims to:

        * group tools provided by watex together thematically or in
          terms of usage;
        * motivate why someone would use each particular tool, often through
          comparison;
        * provide both intuitive and technical descriptions of tools;
        * provide or link to :term:`examples` of using key features of a
          tool.

    np
        A shorthand for Numpy due to the conventional import statement::

            import numpy as np

    online learning
        Where a model is iteratively updated by receiving each batch of ground
        truth :term:`targets` soon after making predictions on corresponding
        batch of data.  Intrinsically, the model must be usable for prediction
        after each batch. See :term:`partial_fit`.

    out-of-core
        An efficiency strategy where not all the data is stored in main memory
        at once, usually by performing learning on batches of data. See
        :term:`partial_fit`.

    outputs
        Individual scalar/categorical variables per sample in the
        :term:`target`.  For example, in multilabel classification each
        possible label corresponds to a binary output. Also called *responses*,
        *tasks* or *targets*.
        See :term:`multiclass multioutput` and :term:`continuous multioutput`.

    pair
        A tuple of length two.

    parameter
    parameters
    param
    params
        We mostly use *parameter* to refer to the aspects of an estimator that
        can be specified in its construction. For example, ``max_depth`` and
        ``random_state`` are parameters of :class:`RandomForestClassifier`.
        Parameters to an estimator's constructor are stored unmodified as
        attributes on the estimator instance, and conventionally start with an
        alphabetic character and end with an alphanumeric character.  Each
        estimator's constructor parameters are described in the estimator's
        docstring.

        We do not use parameters in the statistical sense, where parameters are
        values that specify a model and can be estimated from data. What we
        call parameters might be what statisticians call hyperparameters to the
        model: aspects for configuring model structure that are often not
        directly learnt from data.  However, our parameters are also used to
        prescribe modeling operations that do not affect the learnt model, such
        as :term:`n_jobs` for controlling parallelism.

        When talking about the parameters of a :term:`meta-estimator`, we may
        also be including the parameters of the estimators wrapped by the
        meta-estimator.  Ordinarily, these nested parameters are denoted by
        using a :term:`double underscore` (``__``) to separate between the
        estimator-as-parameter and its parameter.  Thus ``clf =
        BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=3))``
        has a deep parameter ``estimator__max_depth`` with value ``3``,
        which is accessible with ``clf.estimator.max_depth`` or
        ``clf.get_params()['estimator__max_depth']``.

        The list of parameters and their current values can be retrieved from
        an :term:`estimator instance` using its :term:`get_params` method.

        Between construction and fitting, parameters may be modified using
        :term:`set_params`.  To enable this, parameters are not ordinarily
        validated or altered when the estimator is constructed, or when each
        parameter is set. Parameter validation is performed when :term:`fit` is
        called.

        Common parameters are listed :ref:`below <glossary_parameters>`.

    pairwise metric
    pairwise metrics

        In its broad sense, a pairwise metric defines a function for measuring
        similarity or dissimilarity between two samples (with each ordinarily
        represented as a :term:`feature vector`).  We particularly provide
        implementations of distance metrics (as well as improper metrics like
        Cosine Distance) through :func:`metrics.pairwise_distances`, and of
        kernel functions (a constrained class of similarity functions) in
        :func:`metrics.pairwise_kernels`.  These can compute pairwise distance
        matrices that are symmetric and hence store data redundantly.

        See also :term:`precomputed` and :term:`metric`.

        Note that for most distance metrics, we rely on implementations from
        :mod:`scipy.spatial.distance`, but may reimplement for efficiency in
        our context. The :class:`metrics.DistanceMetric` interface is used to implement
        distance metrics for integration with efficient neighbors search.

    pd
        A shorthand for `Pandas <https://pandas.pydata.org>`_ due to the
        conventional import statement::

            import pandas as pd

    precomputed
        Where algorithms rely on :term:`pairwise metrics`, and can be computed
        from pairwise metrics alone, we often allow the user to specify that
        the :term:`X` provided is already in the pairwise (dis)similarity
        space, rather than in a feature space.  That is, when passed to
        :term:`fit`, it is a square, symmetric matrix, with each vector
        indicating (dis)similarity to every sample, and when passed to
        prediction/transformation methods, each row corresponds to a testing
        sample and each column to a training sample.

        Use of precomputed X is usually indicated by setting a ``metric``,
        ``affinity`` or ``kernel`` parameter to the string 'precomputed'. If
        this is the case, then the estimator should set the `pairwise`
        estimator tag as True.

    rectangular
        Data that can be represented as a matrix with :term:`samples` on the
        first axis and a fixed, finite set of :term:`features` on the second
        is called rectangular.

        This term excludes samples with non-vectorial structures, such as text,
        an image of arbitrary size, a time series of arbitrary length, a set of
        vectors, etc. The purpose of a :term:`vectorizer` is to produce
        rectangular forms of such data.

    sample
    samples
        We usually use this term as a noun to indicate a single feature vector.
        Elsewhere a sample is called an instance, data point, or observation.
        ``n_samples`` indicates the number of samples in a dataset, being the
        number of rows in a data array :term:`X`.

    sample property
    sample properties
        A sample property is data for each sample (e.g. an array of length
        n_samples) passed to an estimator method or a similar function,
        alongside but distinct from the :term:`features` (``X``) and
        :term:`target` (``y``). The most prominent example is
        :term:`sample_weight`; see others at :ref:`glossary_sample_props`.

        As of version 0.19 we do not have a consistent approach to handling
        sample properties and their routing in :term:`meta-estimators`, though
        a ``fit_params`` parameter is often used.

    semi-supervised
    semi-supervised learning
    semisupervised
        Learning where the expected prediction (label or ground truth) is only
        available for some samples provided as training data when
        :term:`fitting` the model.  We conventionally apply the label ``-1``
        to :term:`unlabeled` samples in semi-supervised classification.

    sparse matrix
    sparse graph
        A representation of two-dimensional numeric data that is more memory
        efficient the corresponding dense numpy array where almost all elements
        are zero. We use the :mod:`scipy.sparse` framework, which provides
        several underlying sparse data representations, or *formats*.
        Some formats are more efficient than others for particular tasks, and
        when a particular format provides especial benefit, we try to document
        this fact in watex parameter descriptions.

        Some sparse matrix formats (notably CSR, CSC, COO and LIL) distinguish
        between *implicit* and *explicit* zeros. Explicit zeros are stored
        (i.e. they consume memory in a ``data`` array) in the data structure,
        while implicit zeros correspond to every element not otherwise defined
        in explicit storage.

        Two semantics for sparse matrices are used in watex:

        matrix semantics
            The sparse matrix is interpreted as an array with implicit and
            explicit zeros being interpreted as the number 0.  This is the
            interpretation most often adopted, e.g. when sparse matrices
            are used for feature matrices or :term:`multilabel indicator
            matrices`.
        graph semantics
            As with :mod:`scipy.sparse.csgraph`, explicit zeros are
            interpreted as the number 0, but implicit zeros indicate a masked
            or absent value, such as the absence of an edge between two
            vertices of a graph, where an explicit value indicates an edge's
            weight. This interpretation is adopted to represent connectivity
            in clustering, in representations of nearest neighborhoods
            (e.g. :func:`neighbors.kneighbors_graph`), and for precomputed
            distance representation where only distances in the neighborhood
            of each point are required.

        When working with sparse matrices, we assume that it is sparse for a
        good reason, and avoid writing code that densifies a user-provided
        sparse matrix, instead maintaining sparsity or raising an error if not
        possible (i.e. if an estimator does not / cannot support sparse
        matrices).

    supervised
    supervised learning
        Learning where the expected prediction (label or ground truth) is
        available for each sample when :term:`fitting` the model, provided as
        :term:`y`.  This is the approach taken in a :term:`classifier` or
        :term:`regressor` among other estimators.

    target
    targets
        The *dependent variable* in :term:`supervised` (and
        :term:`semisupervised`) learning, passed as :term:`y` to an estimator's
        :term:`fit` method.  Also known as *dependent variable*, *outcome
        variable*, *response variable*, *ground truth* or *label*. watex
        works with targets that have minimal structure: a class from a finite
        set, a finite real-valued number, multiple classes, or multiple
        numbers. See :ref:`glossary_target_types`.

    transduction
    transductive
        A transductive (contrasted with :term:`inductive`) machine learning
        method is designed to model a specific dataset, but not to apply that
        model to unseen data.  Examples include :class:`manifold.TSNE`,
        :class:`cluster.AgglomerativeClustering` and
        :class:`neighbors.LocalOutlierFactor`.

    unlabeled
    unlabeled data
        Samples with an unknown ground truth when fitting; equivalently,
        :term:`missing values` in the :term:`target`.  See also
        :term:`semisupervised` and :term:`unsupervised` learning.

    unsupervised
    unsupervised learning
        Learning where the expected prediction (label or ground truth) is not
        available for each sample when :term:`fitting` the model, as in
        :term:`clusterers` and :term:`outlier detectors`.  Unsupervised
        estimators ignore any :term:`y` passed to :term:`fit`.


 
.. _glossary_parameters: 

Core parameters 
=================

.. glossary:: 

	as_frame
		Transform the data in a pandas DataFrame including columns with
		appropriate types (numeric). The target is
		a panda DataFrame or Series depending on the number of target columns.
		If `as_frame` is False, then returning a :class:`~watex.utils.box.Boxspace`
		dictionary-like object, with the following attributes:
			
		* data : {ndarray, dataframe} 
			The data matrix. If `as_frame=True`, `data` will be a pandas
			DataFrame.
		* resistivity: {array-like} of shape (shape[0],)
			The resistivity of the sounding point. 
		* MN: {array-like} of shape (shape[0],)
			The step value of potential electrodes increasing in meters  
		*  AB: {array-like} of shape (shape[0],)
			The step value of current electrodes increasing in meters  
		* feature_names: list
			The names of the dataset columns.
		* DESCR: str
			The full description of the dataset.
		* filename: str
			The path to the location of the data.
	  
	data
		str, filepath_or_buffer or :class:`pandas.core.DataFrame`
		Path -like object or Dataframe. If data is given as path-like object,
		data is read, asserted and validated. Any valid string path is acceptable. 
		The string could be a URL. Valid URL schemes include http, ftp, s3, gs, and
		file. For file URLs, a host is expected. A local file could be a
		file://localhost/path/to/table.csv. If you want to pass in a path object, 
		pandas accepts any :code:`os.PathLike`. By file-like object, we refer to 
		objects with a `read()` method, such as a file handle e.g. via builtin 
		`open` function or `StringIO`.

	index_rhoa
		int, 
		index of the resistivy columns to retrieve. Note that this is useful in the 
		cases many sounding values are collected in the same survey area. 
		`index_rhoa=0` fetches the first sounding values in the collection of all values. 
		
	tag
		str, 
		Name of the dataset to fectched. Tag can be a data set processing stages. 
		See `datasets <datasets>` for consistent details. 
	 
	X 
		Ndarray of shape ( M x N), :math:`M=m-samples` & :math:`N=n-features`
		training set; Denotes data that is observed at training and prediction time, 
		used as independent variables in learning. The notation is uppercase to denote 
		that it is ordinarily a matrix. When a matrix, each sample may be 
		represented by a feature vector, or a vector of precomputed (dis)similarity 
		with each training sample. :code:`X` may also not be a matrix, and 
		may require a feature extractor or a pairwise metric to turn it into one 
		before learning a model.

	y
		array-like of shape (M, ) `:math:`M=m-samples` 
		train target; Denotes data that may be observed at training time as the 
		dependent variable in learning, but which is unavailable at prediction time, 
		and is usually the target of prediction. 

	Xt
		Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
		Shorthand for "test set"; data that is observed at testing and prediction time, 
		used as independent variables in learning.The notation is uppercase to denote 
		that it is ordinarily a matrix.

	yt
		array-like, shape (M, ) ``M=m-samples``,
		test target; Denotes data that may be observed at training time as the 
		dependent variable in learning, but which is unavailable at prediction time, 
		and is usually the target of prediction. 

	tname
		str, 
		A target name or label. In supervised learning the target name is considered  
		as the reference name of `y` or label variable.   

	z
		array-like 1d, pandas.Series 
		Array of depth or a pandas series that contains the depth values. Two  
		dimensional array or more is not allowed. However when `z` is given as 
		a dataframe and `zname` is not supplied, an error raises since `zname` is 
		used to fetch and overwritten `z` from the dataframe. 

	zname
		str, int
		Name of depth columns. `zname` allows to retrieve the depth column in 
		a dataframe. If integer is passed, it assumes the index of the dataframe 
		fits the depth column. Integer value must not be out the dataframe size 
		along axis 1. Commonly `zname`needs to be supplied when a dataframe is 
		passed to a function argument. 

	kname
		str, int
		Name of permeability coefficient columns. `kname` allows to retrieve the 
		permeability coefficient 'k' in  a specific dataframe. If integer is passed, 
		it assumes the index of the dataframe  fits the 'k' columns. Note that 
		integer value must not be out the dataframe size along axis 1. Commonly
		`kname` needs to be supplied when a dataframe is passed as a positional 
		or keyword argument. 

	k
		array-like 1d, pandas.Series 
		Array of permeability coefficient 'k' or a pandas series that contains the 
		'k' values. Two  dimensional array or more is not allowed. However,
		when `k` passes as a dataframe and `kname` is not supplied, an error 
		raises since `kname` is used to retrieve `k` values from the dataframe 
		and overwritten it.

	target
		Array-like or :class:`pd.Series`
		Is the dependent variable in supervised (and semisupervised) learning, 
		passed as `y` to an estimator's fit method. Also known as dependent 
		variable, outcome variable, response variable, ground truth or label. 
		`watex`_ works with targets that have minimal structure: a class 
		from a finite set, a finite real-valued number, multiple classes, or 
		multiple numbers. Refer to `watex`_ `target types`_ . Note that 
		throughout this library, a `target` is considered as a `pd.Series` where 
		the name is `tname` and the variable `y` i.e `target = tname + y`.
		
		.. _target types: https://scikit-learn.org/stable/glossary.html#glossary-target-types
		

	model
		callable, always as a function,    
		A model estimator. An object which manages the estimation and decoding 
		of a model. The model is estimated as a deterministic function of:
			
		* parameters provided in object construction or with set_params;
		* the global numpy.random random state if the estimator’s random_state 
			parameter is set to None; and
		* any data or sample properties passed to the most recent call to fit, 
			fit_transform or fit_predict, or data similarly passed in a sequence 
			of calls to partial_fit.
			
		The estimated model is stored in public and private attributes on the 
		estimator instance, facilitating decoding through prediction and 
		transformation methods.
		Estimators must provide a fit method, and should provide `set_params` and 
		`get_params`, although these are usually provided by inheritance from 
		`base.BaseEstimator`.
		The core functionality of some estimators may also be available as a ``function``.

	clf
		callable, always as a function, classifier estimator
		A supervised (or semi-supervised) predictor with a finite set of discrete 
		possible output values. A classifier supports modeling some of binary, 
		multiclass, multilabel, or multiclass multioutput targets. Within scikit-learn, 
		all classifiers support multi-class classification, defaulting to using a 
		one-vs-rest strategy over the binary classification problem.
		Classifiers must store a classes_ attribute after fitting, and usually 
		inherit from base.ClassifierMixin, which sets their _estimator_type attribute.
		A classifier can be distinguished from other estimators with is_classifier.
		It must implement:
		* fit
		* predict
		* score
		It may also be appropriate to implement decision_function, predict_proba 
		and predict_log_proba.    

	reg
		callable, always as a function
		A regression estimator; Estimators must provide a fit method, and should 
		provide `set_params` and 
		`get_params`, although these are usually provided by inheritance from 
		`base.BaseEstimator`. The estimated model is stored in public and private 
		attributes on the estimator instance, facilitating decoding through prediction 
		and transformation methods.
		The core functionality of some estimators may also be available as a
		``function``.

	cv
		float,    
		A cross validation splitting strategy. It used in cross-validation based 
		routines. cv is also available in estimators such as multioutput. 
		ClassifierChain or calibration.CalibratedClassifierCV which use the 
		predictions of one estimator as training data for another, to not overfit 
		the training supervision.
		Possible inputs for cv are usually:
		
		* An integer, specifying the number of folds in K-fold cross validation. 
			K-fold will be stratified over classes if the estimator is a classifier
			(determined by base.is_classifier) and the targets may represent a 
			binary or multiclass (but not multioutput) classification problem 
			(determined by utils.multiclass.type_of_target).
		* A cross-validation splitter instance. Refer to the User Guide for 
			splitters available within `watex`_
		* An iterable yielding train/test splits.
		
		With some exceptions (especially where not using cross validation at all 
							  is an option), the default is ``4-fold``.
		.. _Scikit-learn: https://scikit-learn.org/stable/glossary.html#glossary

	scoring
		str, 
		Specifies the score function to be maximized (usually by :ref:`cross
		validation <cross_validation>`), or -- in some cases -- multiple score
		functions to be reported.

	random_state 
		int, RandomState instance or None, default=None
		Controls the shuffling applied to the data before applying the split.
		Pass an int for reproducible output across multiple function calls..    

	test_size 
		float or int, default=None
		If float, should be between 0.0 and 1.0 and represent the proportion
		of the dataset to include in the test split. If int, represents the
		absolute number of test samples. If None, the value is set to the
		complement of the train size. If ``train_size`` is also None, it will
		be set to 0.25.    

	n_jobs 
		int, 
		is used to specify how many concurrent processes or threads should be 
		used for routines that are parallelized with joblib. It specifies the maximum 
		number of concurrently running workers. If 1 is given, no joblib parallelism 
		is used at all, which is useful for debugging. If set to -1, all CPUs are 
		used. For instance:
		
		* `n_jobs` below -1, (n_cpus + 1 + n_jobs) are used. 
		
		* `n_jobs`=-2, all CPUs but one are used. 
		* `n_jobs` is None by default, which means unset; it will generally be 
			interpreted as n_jobs=1 unless the current joblib.Parallel backend 
			context specifies otherwise.

		Note that even if n_jobs=1, low-level parallelism (via Numpy and OpenMP) 
		might be used in some configuration.  

	verbose
		int, `default` is ``0``    
		Control the level of verbosity. Higher value lead to more messages. 

	self: 
		`Baseclass` instance 
		returns ``self`` for easy method chaining.


.. _glossary_plot_properties:

Plot properties 
================

.. glossary:: 

	savefig 
		str, Path-like object, 
		savefigure's name, *default* is ``None``
	fig_dpi
		float, 
		dots-per-inch resolution of the figure. *default* is 300   

	fig_num
		int, 
		size of figure in inches (width, height). *default* is [5, 5]

	fig_size
		Tuple (int, int) or inch 
	    size of figure in inches (width, height).*default* is [5, 5]

	fig_orientation
		str, 
		figure orientation. *default* is ``landscape``

	fig_tile
		str, 
		figure title. *default* is ``None``     

	fs
		float, 
		size of font of axis tick labels, axis labels are fs+2. *default* is 6

	ls
		str, 
		line style, it can be [ '-' | '.' | ':' ] . *default* is '-'

	lc
		str, Optional, 
		line color of the plot, *default* is ``k``

	lw
		float, Optional, 
		line weight of the plot, *default* is ``1.5``

	alpha
		float between 0 < alpha < 1, 
		transparency number, *default* is ``0.5``,   

	font_weight
		str, Optional
		weight of the font , *default* is ``bold``.

	font_style
		str, Optional
		style of the font. *default* is ``italic``

	font_size
		float, Optional
		size of font in inches (width, height). *default* is ``3``.    

	ms
		float, Optional 
		size of marker in points. *default* is ``5``

	marker
		str, Optional
		marker of stations *default* is ``o``.

	marker_style
		str, Optional
		facecolor of the marker. *default* is ``yellow``    

	marker_edgecolor
		str, Optional
		facecolor of the marker. *default* is ``yellow``

	marker_edgewidth
		float, Optional
		width of the marker. *default* is ``3``.    

	xminorticks
		float, Optional
		minortick according to x-axis size and *default* is ``1``.

	yminorticks
		float, Optional
		yminorticks according to x-axis size and *default* is ``1``.

	bins
		histograms element separation between two bar. *default* is ``10``. 

	xlim
		tuple (int, int), Optional
		limit of x-axis in plot. 

	ylim
		tuple (int, int), Optional
		limit of x-axis in plot. 

	xlabel
		str, Optional, 
		label name of x-axis in plot.

	ylabel
		str, Optional, 
		label name of y-axis in plot.

	rotate_xlabel
		float, Optional
		angle to rotate `xlabel` in plot.  

	rotate_ylabel
		float, Optional
		angle to rotate `ylabel` in plot.  

	leg_kws
		dict, Optional 
		keyword arguments of legend. *default* is empty ``dict``

	plt_kws
		dict, Optional
		keyword arguments of plot. *default* is empty ``dict``

	glc
		str, Optional
		line color of the grid plot, *default* is ``k``

	glw
		float, Optional
	    line weight of the grid plot, *default* is ``2``

	galpha
		float, Optional, 
		transparency number of grid, *default* is ``0.5``  

	gaxis
		str ('x', 'y', 'both')
		type of axis to hold the grid, *default* is ``both``

	gwhich
		str, Optional
		kind of grid in the plot. *default* is ``major``

	tp_axis
		bool, 
		axis to apply the ticks params. default is ``both``

	tp_labelsize
		str, Optional
		labelsize of ticks params. *default* is ``italic``

	tp_bottom
		bool, 
		position at bottom of ticks params. *default* is ``True``.

	tp_labelbottom
		bool, 
		put label on the bottom of the ticks. *default* is ``False``    

	tp_labeltop
		bool, 
		put label on the top of the ticks. *default* is ``True``    

	cb_orientation
		str , ('vertical', 'horizontal')    
		orientation of the colorbar, *default* is ``vertical``

	cb_aspect
		float, Optional 
		aspect of the colorbar. *default* is ``20``.

	cb_shrink
		float, Optional
		shrink size of the colorbar. *default* is ``1.0``

	cb_pad
		float, 
		pad of the colorbar of plot. *default* is ``.05``

	cb_anchor
		tuple (float, float)
		anchor of the colorbar. *default* is ``(0.0, 0.5)``

	cb_panchor
		tuple (float, float)
		proportionality anchor of the colorbar. *default* is ``(1.0, 0.5)``

	cb_label
		str, Optional 
		label of the colorbar.   

	cb_spacing
		str, Optional
		spacing of the colorbar. *default* is ``uniform``

	cb_drawedges
		bool, 
		draw edges inside of the colorbar. *default* is ``False`` 
		
	ax 
		:class:`matplotlib.axes.Axes`
		The matplotlib axes containing the plot.

	facetgrid
		:class:`FacetGrid`
		An object managing one or more subplots that correspond to conditional data
		subsets with convenient methods for batch-setting of axes attributes.

	jointgrid
		:class:`JointGrid`
		An object managing multiple subplots that correspond to joint and marginal axes
		for plotting a bivariate relationship or distribution.

	pairgrid
		class:`PairGrid`
		An object managing multiple subplots that correspond to joint and marginal axes
		for pairwise combinations of multiple variables in a dataset.
    
 
.. _glossary_miscellaneous: 
 
Miscellaneous 
==============

.. glossary:: 

	scatterplot 
		Plot data using points.

	lineplot 
		Plot data using lines.

	displot 
		Figure-level interface to distribution plot functions.

	histplot 
		Plot a histogram of binned counts with optional normalization or smoothing.

	kdeplot
		Plot univariate or bivariate distributions using kernel density estimation.

	ecdfplot 
		Plot empirical cumulative distribution functions.

	rugplot  
		Plot a tick at each observation value along the x and/or y axes.

	stripplot 
		Plot a categorical scatter with jitter.

	swarmplot
		Plot a categorical scatter with non-overlapping points.

	violinplot 
		Draw an enhanced boxplot using kernel density estimation.

	pointplot 
		Plot point estimates and CIs using markers and lines.

	boxplot 
		Draw an enhanced boxplot.

	jointplot
		Draw a bivariate plot with univariate marginal distributions.

	jointplot
		Draw multiple bivariate plots with univariate marginal distributions.

	JointGrid
		Set up a figure with joint and marginal views on bivariate data.

	PairGrid 
		Set up a figure with joint and marginal views on multiple variables.


.. _glossary_reference_links: 

Resource-links 
=================

.. glossary:: 

	Bagoue region
		https://en.wikipedia.org/wiki/Bagou%C3%A9
	Dieng et al
		http://documents.irevues.inist.fr/bitstream/handle/2042/36362/2IE_2004_12_21.pdf?sequence=1
	Kouadio et al
		https://doi.org/10.1029/2021WR031623
	FlowRatePredictionUsingSVMs
		https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021WR031623
	GeekforGeeks
		https://www.geeksforgeeks.org/style-plots-using-matplotlib/#:~:text=Matplotlib%20is%20the%20most%20popular,without%20using%20any%20other%20GUIs
	IUPAC nommenclature
		https://en.wikipedia.org/wiki/IUPAC_nomenclature_of_inorganic_chemistry
	Matplotlib scatter
		https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.scatter.html
	Matplotlib plot
		https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.plot.html
		https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.plot.html
	Matplotlib figure
		https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.figure.html
	Matplotlib figsuptitle
		https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.suptitle.html
	Properties of water
		https://en.wikipedia.org/wiki/Properties_of_water#Electrical_conductivity 
	Pandas DataFrame
		https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
	Pandas Series
		https://pandas.pydata.org/docs/reference/api/pandas.Series.html
	Scipy Optimize
		curve-fit: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
	Water
		https://en.wikipedia.org/wiki/Water
	Water triple point
		https://en.wikipedia.org/wiki/Properties_of_water#/media/File:Phase_diagram_of_water.svg
	WATex
		https://github.com/WEgeophysics/watex/
	pycsamt
		https://github.com/WEgeophysics/pycsamt




    
























































