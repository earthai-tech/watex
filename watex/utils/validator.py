# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2007-2022 The scikit-learn developers.
# All rights reserved.

# Note that this module is not the sckit-learn original file, 
# some functions have been removed to keep only  
# the usefull for of watex package. Furthermore some others 
# function have been edited and added. 

# Utilities for input validation

from functools import wraps
import inspect 
import types 
import warnings
import numbers
import operator
import joblib
import numpy as np
# mypy error: Module 'numpy.core.numeric' has no attribute 'ComplexWarning'
from numpy.core.numeric import ComplexWarning  # type: ignore
from contextlib import suppress
import scipy.sparse as sp
from inspect import signature, Parameter

from ._array_api import get_namespace, _asarray_with_order

FLOAT_DTYPES = (np.float64, np.float32, np.float16)

def _is_numeric_dtype (o, / , to_array =False ): 
    """ Determine whether the argument has a numeric datatype, when
    converted to a NumPy array.

    Booleans, unsigned integers, signed integers, floats and complex
    numbers are the kinds of numeric datatype. 
    
    :param o: object, arraylike 
        Object presumed to be an array 
    :param to_array: bool, default=False 
        If `o` is passed as non-array like list or tuple or other iterable 
        object. Setting `to_array` to ``True`` will convert array to ``True``. 
    :return: bool, 
        ``True`` if `o` has a numeric dtype and ``False`` otherwise. 
    """ 
    _NUMERIC_KINDS = set('buifc')
    if not hasattr (o, '__iter__'): 
        raise TypeError ("'o' is expected to be an iterable object."
                         f" got: {type(o).__name__!r}")
    if to_array : 
        o = np.array (o )
    if not hasattr(o, '__array__'): 
        raise ValueError (f"Expect type array, got: {type (o).__name__!r}")
    # use NUMERICKIND rather than # pd.api.types.is_numeric_dtype(arr) 
    # for series and dataframes
    return ( o.values.dtype.kind   
            if ( hasattr(o, 'columns') or hasattr (o, 'name'))
            else o.dtype.kind ) in _NUMERIC_KINDS 
        
def _check_consistency_size (ar1, ar2 , /  , error ='raise') :
    """ Check consistency of two arrays and raises error if both sizes 
    are differents. 
    Returns 'False' if sizes are not consistent and error is set to 'ignore'.
    """
    if error =='raise': 
        msg =("Array sizes must be consistent: '{}' and '{}' were given.")
        assert len(ar1)==len(ar2), msg.format(len(ar1), len(ar2))
        
    return len(ar1)==len(ar2) 

def _is_buildin (o, /, mode ='soft'): 
    """ Returns 'True' wether the module is a Python buidling function. 
    
    If  `mode` is ``strict`` only assert the specific predifined-functions 
    like 'str', 'len' etc, otherwise check in the whole predifined functions
    including the object with type equals to 'module'
    
    :param o: object
        Any object for verification 
    :param mode: str , default='soft' 
        mode for asserting object. Can also be 'strict' for the specific 
        predifined build-in functions. 
    :param module: 
    """
    assert mode in {'strict', 'soft'}, f"Unsupports mode {mode!r}, "\
        "expects 'strict'or 'soft'"
    
    return  (isinstance(o, types.BuiltinFunctionType) and inspect.isbuiltin (o)
             ) if mode=='strict' else type (o).__module__== 'builtins' 


def get_estimator_name (estimator , /): 
    """ Get the estimator name whether it is instanciated or not  
    
    :param estimator: callable or instanciated object,
        callable or instance object that has a fit method. 
    
    :return: str, 
        name of the estimator. 
    """
    name =' '
    if hasattr (estimator, '__qualname__') and hasattr(
            estimator, '__name__'): 
        name = estimator.__name__ 
    elif hasattr(estimator, '__class__') and not hasattr (
            estimator, '__name__'): 
        name = estimator.__class__.__name__ 
    return name 

def _is_cross_validated (estimator ): 
    """ Check whether the estimator has already passed the cross validation
     procedure. 
     
    We assume it has the attributes 'best_params_' and 'best_estimator_' 
    already populated.
    
    :param estimator: callable or instanciated object, that has a fit method. 
    :return: bool, 
        estimator has already passed the cross-validation procedure. 
    
    """
    return hasattr(estimator, 'best_estimator_') and hasattr (
        estimator , 'best_params_')

def _is_valid_erp(d , / ): 
    """ Returns 'True' if the given data is Electrical Resistivity Profiling"""
    return not len(d) ==0 and  ('resistivity' and 'station') in d.columns 

def _is_valid_ves (d, /)  : 
    """Returns 'True' if data is Vertical Electrical Sounding """
    return not len(d) ==0 and  ('resistivity' and 'AB') in d.columns 

def _check_array_in(obj, /, arr_name):
    """Returns the array from the array name attribute. Note that the singleton 
    array is not admitted. 
    
    This helper function tries to return array from object attribute  where 
    object attribute is the array name if exists. Otherwise raises an error. 
    
    Parameters
    ----------
    obj : object 
       Object that is expect to contain the array attribute.
    Returns
    -------
    X : array
       Array fetched from its name in `obj`. 
    """
    
    type_ = type(obj)
    try : 
        type_name = f"{obj.__module__}.{obj.__qualname__}"
        o_= f" in {obj.__name__!r}"
    except AttributeError:
        type_name = type_.__qualname__
        o_=''
        
    message = (f"Unable to find the name {arr_name!r}"
               f"{o_} from {type_name!r}") 
    
    if not hasattr (obj , arr_name ): 
        raise TypeError (message )
    
    X = getattr ( obj , f"{arr_name}") 

    if not hasattr(X, "__len__") and not hasattr(X, "shape"):
        if not hasattr(X, "__array__"):
            raise TypeError(message)
        # Only convert X to a numpy array if there is no cheaper, heuristic
        # option.
        X = np.asarray(X)

    if hasattr(X, "shape"):
        if not hasattr(X.shape, "__len__") or len(X.shape) <= 1:
            warnings.warn ( 
                "A singleton array %r cannot be considered a valid collection."% X)
            message += f" with shape {X.shape}"
            raise TypeError(message)
        
    return X 

        
def _deprecate_positional_args(func=None, *, version="1.3"):
    """Decorator for methods that issues warnings for positional arguments.
    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.
    Parameters
    ----------
    func : callable, default=None
        Function to check arguments on.
    version : callable, default="1.3"
        The version when positional arguments will result in error.
    """

    def _inner_deprecate_positional_args(f):
        sig = signature(f)
        kwonly_args = []
        all_args = []

        for name, param in sig.parameters.items():
            if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
                all_args.append(name)
            elif param.kind == Parameter.KEYWORD_ONLY:
                kwonly_args.append(name)

        @wraps(f)
        def inner_f(*args, **kwargs):
            extra_args = len(args) - len(all_args)
            if extra_args <= 0:
                return f(*args, **kwargs)

            # extra_args > 0
            args_msg = [
                "{}={}".format(name, arg)
                for name, arg in zip(kwonly_args[:extra_args], args[-extra_args:])
            ]
            args_msg = ", ".join(args_msg)
            warnings.warn(
                f"Pass {args_msg} as keyword args. From version "
                f"{version} passing these as positional arguments "
                "will result in an error",
                FutureWarning,
            )
            kwargs.update(zip(sig.parameters, args))
            return f(**kwargs)

        return inner_f

    if func is not None:
        return _inner_deprecate_positional_args(func)

    return _inner_deprecate_positional_args

def to_dtype_str (arr, /, return_values = False ): 
    """ Convert numeric or object dtype to string dtype. 
    
    This will avoid a particular TypeError when an array is filled by np.nan 
    and at the same time contains string values. 
    Converting the array to dtype str rather than keeping to 'object'
    will pass this error. 
    
    :param arr: array-like
        array with all numpy datatype or pandas dtypes
    :param return_values: bool, default=False 
        returns array values in string dtype. This might be usefull when a 
        series with dtype equals to object or numeric is passed. 
    :returns: array-like 
        array-like with dtype str 
        Note that if the dataframe or serie is passed, the object datatype 
        will change only if `return_values` is set to ``True``, otherwise 
        returns the same object. 
    
    """
    if not hasattr (arr, '__array__'): 
        raise TypeError (f"Expects an array, got: {type(arr).__name__!r}")
    if return_values : 
        if (hasattr(arr, 'name') or hasattr (arr,'columns')):
            arr = arr.values 
    return arr.astype (str ) 

def _is_arraylike_1d (x) :
    """ Returns whether the input is arraylike one dimensional and not a scalar"""
    if not hasattr (x, '__array__'): 
        raise TypeError ("Expects a one-dimensional array, "
                         f"got: {type(x).__name__!r}")
    _is_arraylike_not_scalar(x)
    return _is_arraylike_not_scalar(x) and  (  len(x.shape )< 2 or ( 
        len(x.shape ) ==2 and x.shape [1]==1 )) 

def _is_arraylike(x):
    """Returns whether the input is array-like."""
    return hasattr(x, "__len__") or hasattr(x, "shape") or hasattr(x, "__array__")


def _is_arraylike_not_scalar(array):
    """Return True if array is array-like and not a scalar"""
    return _is_arraylike(array) and not np.isscalar(array)

def _num_features(X):
    """Return the number of features in an array-like X.
    This helper function tries hard to avoid to materialize an array version
    of X unless necessary. For instance, if X is a list of lists,
    this function will return the length of the first element, assuming
    that subsequent elements are all lists of the same length without
    checking.
    Parameters
    ----------
    X : array-like
        array-like to get the number of features.
    Returns
    -------
    features : int
        Number of features
    """
    type_ = type(X)
    if type_.__module__ == "builtins":
        type_name = type_.__qualname__
    else:
        type_name = f"{type_.__module__}.{type_.__qualname__}"
    message = f"Unable to find the number of features from X of type {type_name}"
    if not hasattr(X, "__len__") and not hasattr(X, "shape"):
        if not hasattr(X, "__array__"):
            raise TypeError(message)
        # Only convert X to a numpy array if there is no cheaper, heuristic
        # option.
        X = np.asarray(X)

    if hasattr(X, "shape"):
        if not hasattr(X.shape, "__len__") or len(X.shape) <= 1:
            message += f" with shape {X.shape}"
            raise TypeError(message)
        return X.shape[1]

    first_sample = X[0]

    # Do not consider an array-like of strings or dicts to be a 2D array
    if isinstance(first_sample, (str, bytes, dict)):
        message += f" where the samples are of type {type(first_sample).__qualname__}"
        raise TypeError(message)

    try:
        # If X is a list of lists, for instance, we assume that all nested
        # lists have the same length without checking or converting to
        # a numpy array to keep this function call as cheap as possible.
        return len(first_sample)
    except Exception as err:
        raise TypeError(message) from err


def _num_samples(x):
    """Return number of samples in array-like x."""
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid collection." % x
            )
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error


def check_memory(memory):
    """Check that ``memory`` is joblib.Memory-like.
    joblib.Memory-like means that ``memory`` can be converted into a
    joblib.Memory instance (typically a str denoting the ``location``)
    or has the same interface (has a ``cache`` method).
    Parameters
    ----------
    memory : None, str or object with the joblib.Memory interface
        - If string, the location where to create the `joblib.Memory` interface.
        - If None, no caching is done and the Memory object is completely transparent.
    Returns
    -------
    memory : object with the joblib.Memory interface
        A correct joblib.Memory object.
    Raises
    ------
    ValueError
        If ``memory`` is not joblib.Memory-like.
    """
    if memory is None or isinstance(memory, str):
        memory = joblib.Memory(location=memory, verbose=0)
    elif not hasattr(memory, "cache"):
        raise ValueError(
            "'memory' should be None, a string or have the same"
            " interface as joblib.Memory."
            " Got memory='{}' instead.".format(memory)
        )
    return memory


def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.
    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(l) for l in lengths]
        )


def _make_indexable(iterable):
    """Ensure iterable supports indexing or convert to an indexable variant.
    Convert sparse matrices to csr and other non-indexable iterable to arrays.
    Let `None` and indexable objects (e.g. pandas dataframes) pass unchanged.
    Parameters
    ----------
    iterable : {list, dataframe, ndarray, sparse matrix} or None
        Object to be converted to an indexable iterable.
    """
    if sp.issparse(iterable):
        return iterable.tocsr()
    elif hasattr(iterable, "__getitem__") or hasattr(iterable, "iloc"):
        return iterable
    elif iterable is None:
        return iterable
    return np.array(iterable)


def indexable(*iterables):
    """Make arrays indexable for cross-validation.
    Checks consistent length, passes through None, and ensures that everything
    can be indexed by converting sparse matrices to csr and converting
    non-interable objects to arrays.
    Parameters
    ----------
    *iterables : {lists, dataframes, ndarrays, sparse matrices}
        List of objects to ensure sliceability.
    Returns
    -------
    result : list of {ndarray, sparse matrix, dataframe} or None
        Returns a list containing indexable arrays (i.e. NumPy array,
        sparse matrix, or dataframe) or `None`.
    """

    result = [_make_indexable(X) for X in iterables]
    check_consistent_length(*result)
    return result


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )


def has_fit_parameter(estimator, parameter):
    """Check whether the estimator's fit method supports the given parameter.
    Parameters
    ----------
    estimator : object
        An estimator to inspect.
    parameter : str
        The searched parameter.
    Returns
    -------
    is_parameter : bool
        Whether the parameter was found to be a named parameter of the
        estimator's fit method.
    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from sklearn.utils.validation import has_fit_parameter
    >>> has_fit_parameter(SVC(), "sample_weight")
    True
    """
    return parameter in signature(estimator.fit).parameters


def check_symmetric(array, *, tol=1e-10, raise_warning=True, raise_exception=False):
    """Make sure that array is 2D, square and symmetric.
    If the array is not symmetric, then a symmetrized version is returned.
    Optionally, a warning or exception is raised if the matrix is not
    symmetric.
    Parameters
    ----------
    array : {ndarray, sparse matrix}
        Input object to check / convert. Must be two-dimensional and square,
        otherwise a ValueError will be raised.
    tol : float, default=1e-10
        Absolute tolerance for equivalence of arrays. Default = 1E-10.
    raise_warning : bool, default=True
        If True then raise a warning if conversion is required.
    raise_exception : bool, default=False
        If True then raise an exception if array is not symmetric.
    Returns
    -------
    array_sym : {ndarray, sparse matrix}
        Symmetrized version of the input array, i.e. the average of array
        and array.transpose(). If sparse, then duplicate entries are first
        summed and zeros are eliminated.
    """
    if (array.ndim != 2) or (array.shape[0] != array.shape[1]):
        raise ValueError(
            "array must be 2-dimensional and square. shape = {0}".format(array.shape)
        )

    if sp.issparse(array):
        diff = array - array.T
        # only csr, csc, and coo have `data` attribute
        if diff.format not in ["csr", "csc", "coo"]:
            diff = diff.tocsr()
        symmetric = np.all(abs(diff.data) < tol)
    else:
        symmetric = np.allclose(array, array.T, atol=tol)

    if not symmetric:
        if raise_exception:
            raise ValueError("Array must be symmetric")
        if raise_warning:
            warnings.warn(
                "Array is not symmetric, and will be converted "
                "to symmetric by average with its transpose.",
                stacklevel=2,
            )
        if sp.issparse(array):
            conversion = "to" + array.format
            array = getattr(0.5 * (array + array.T), conversion)()
        else:
            array = 0.5 * (array + array.T)

    return array


def check_scalar(
    x,
    name,
    target_type,
    *,
    min_val=None,
    max_val=None,
    include_boundaries="both",
):
    """Validate scalar parameters type and value.
    Parameters
    ----------
    x : object
        The scalar parameter to validate.
    name : str
        The name of the parameter to be printed in error messages.
    target_type : type or tuple
        Acceptable data types for the parameter.
    min_val : float or int, default=None
        The minimum valid value the parameter can take. If None (default) it
        is implied that the parameter does not have a lower bound.
    max_val : float or int, default=None
        The maximum valid value the parameter can take. If None (default) it
        is implied that the parameter does not have an upper bound.
    include_boundaries : {"left", "right", "both", "neither"}, default="both"
        Whether the interval defined by `min_val` and `max_val` should include
        the boundaries. Possible choices are:
        - `"left"`: only `min_val` is included in the valid interval.
          It is equivalent to the interval `[ min_val, max_val )`.
        - `"right"`: only `max_val` is included in the valid interval.
          It is equivalent to the interval `( min_val, max_val ]`.
        - `"both"`: `min_val` and `max_val` are included in the valid interval.
          It is equivalent to the interval `[ min_val, max_val ]`.
        - `"neither"`: neither `min_val` nor `max_val` are included in the
          valid interval. It is equivalent to the interval `( min_val, max_val )`.
    Returns
    -------
    x : numbers.Number
        The validated number.
    Raises
    ------
    TypeError
        If the parameter's type does not match the desired type.
    ValueError
        If the parameter's value violates the given bounds.
        If `min_val`, `max_val` and `include_boundaries` are inconsistent.
    """

    def type_name(t):
        """Convert type into humman readable string."""
        module = t.__module__
        qualname = t.__qualname__
        if module == "builtins":
            return qualname
        elif t == numbers.Real:
            return "float"
        elif t == numbers.Integral:
            return "int"
        return f"{module}.{qualname}"

    if not isinstance(x, target_type):
        if isinstance(target_type, tuple):
            types_str = ", ".join(type_name(t) for t in target_type)
            target_type_str = f"{{{types_str}}}"
        else:
            target_type_str = type_name(target_type)

        raise TypeError(
            f"{name} must be an instance of {target_type_str}, not"
            f" {type(x).__qualname__}."
        )

    expected_include_boundaries = ("left", "right", "both", "neither")
    if include_boundaries not in expected_include_boundaries:
        raise ValueError(
            f"Unknown value for `include_boundaries`: {repr(include_boundaries)}. "
            f"Possible values are: {expected_include_boundaries}."
        )

    if max_val is None and include_boundaries == "right":
        raise ValueError(
            "`include_boundaries`='right' without specifying explicitly `max_val` "
            "is inconsistent."
        )

    if min_val is None and include_boundaries == "left":
        raise ValueError(
            "`include_boundaries`='left' without specifying explicitly `min_val` "
            "is inconsistent."
        )

    comparison_operator = (
        operator.lt if include_boundaries in ("left", "both") else operator.le
    )
    if min_val is not None and comparison_operator(x, min_val):
        raise ValueError(
            f"{name} == {x}, must be"
            f" {'>=' if include_boundaries in ('left', 'both') else '>'} {min_val}."
        )

    comparison_operator = (
        operator.gt if include_boundaries in ("right", "both") else operator.ge
    )
    if max_val is not None and comparison_operator(x, max_val):
        raise ValueError(
            f"{name} == {x}, must be"
            f" {'<=' if include_boundaries in ('right', 'both') else '<'} {max_val}."
        )

    return x


def _check_fit_params(X, fit_params, indices=None):
    """Check and validate the parameters passed during `fit`.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data array.
    fit_params : dict
        Dictionary containing the parameters passed at fit.
    indices : array-like of shape (n_samples,), default=None
        Indices to be selected if the parameter has the same size as `X`.
    Returns
    -------
    fit_params_validated : dict
        Validated parameters. We ensure that the values support indexing.
    """
    from . import _safe_indexing

    fit_params_validated = {}
    for param_key, param_value in fit_params.items():
        if not _is_arraylike(param_value) or _num_samples(param_value) != _num_samples(
            X
        ):
            # Non-indexable pass-through (for now for backward-compatibility).
            # https://github.com/scikit-learn/scikit-learn/issues/15805
            fit_params_validated[param_key] = param_value
        else:
            # Any other fit_params should support indexing
            # (e.g. for cross-validation).
            fit_params_validated[param_key] = _make_indexable(param_value)
            fit_params_validated[param_key] = _safe_indexing(
                fit_params_validated[param_key], indices
            )

    return fit_params_validated


def _get_feature_names(X):
    """Get feature names from X.
    Support for other array containers should place its implementation here.
    Parameters
    ----------
    X : {ndarray, dataframe} of shape (n_samples, n_features)
        Array container to extract feature names.
        - pandas dataframe : The columns will be considered to be feature
          names. If the dataframe contains non-string feature names, `None` is
          returned.
        - All other array containers will return `None`.
    Returns
    -------
    names: ndarray or None
        Feature names of `X`. Unrecognized array containers will return `None`.
    """
    feature_names = None

    # extract feature names for support array containers
    if hasattr(X, "columns"):
        feature_names = np.asarray(X.columns, dtype=object)

    if feature_names is None or len(feature_names) == 0:
        return

    types = sorted(t.__qualname__ for t in set(type(v) for v in feature_names))

    # mixed type of string and non-string is not supported
    if len(types) > 1 and "str" in types:
        raise TypeError(
            "Feature names only support names that are all strings. "
            f"Got feature names with dtypes: {types}."
        )

    # Only feature names of all strings are supported
    if len(types) == 1 and types[0] == "str":
        return feature_names


def _check_feature_names_in(estimator, input_features=None, *, generate_names=True):
    """Check `input_features` and generate names if needed.
    Commonly used in :term:`get_feature_names_out`.
    Parameters
    ----------
    input_features : array-like of str or None, default=None
        Input features.
        - If `input_features` is `None`, then `feature_names_in_` is
          used as feature names in. If `feature_names_in_` is not defined,
          then the following input feature names are generated:
          `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
        - If `input_features` is an array-like, then `input_features` must
          match `feature_names_in_` if `feature_names_in_` is defined.
    generate_names : bool, default=True
        Whether to generate names when `input_features` is `None` and
        `estimator.feature_names_in_` is not defined. This is useful for transformers
        that validates `input_features` but do not require them in
        :term:`get_feature_names_out` e.g. `PCA`.
    Returns
    -------
    feature_names_in : ndarray of str or `None`
        Feature names in.
    """

    feature_names_in_ = getattr(estimator, "feature_names_in_", None)
    n_features_in_ = getattr(estimator, "n_features_in_", None)

    if input_features is not None:
        input_features = np.asarray(input_features, dtype=object)
        if feature_names_in_ is not None and not np.array_equal(
            feature_names_in_, input_features
        ):
            raise ValueError("input_features is not equal to feature_names_in_")

        if n_features_in_ is not None and len(input_features) != n_features_in_:
            raise ValueError(
                "input_features should have length equal to number of "
                f"features ({n_features_in_}), got {len(input_features)}"
            )
        return input_features

    if feature_names_in_ is not None:
        return feature_names_in_

    if not generate_names:
        return

    # Generates feature names if `n_features_in_` is defined
    if n_features_in_ is None:
        raise ValueError("Unable to generate feature names without n_features_in_")

    return np.asarray([f"x{i}" for i in range(n_features_in_)], dtype=object)

def _pandas_dtype_needs_early_conversion(pd_dtype):
    """Return True if pandas extension pd_dtype need to be converted early."""
    # Check these early for pandas versions without extension dtypes
    from pandas.api.types import (
        is_bool_dtype,
        is_sparse,
        is_float_dtype,
        is_integer_dtype,
    )

    if is_bool_dtype(pd_dtype):
        # bool and extension booleans need early converstion because __array__
        # converts mixed dtype dataframes into object dtypes
        return True

    if is_sparse(pd_dtype):
        # Sparse arrays will be converted later in `check_array`
        return False

    try:
        from pandas.api.types import is_extension_array_dtype
    except ImportError:
        return False

    if is_sparse(pd_dtype) or not is_extension_array_dtype(pd_dtype):
        # Sparse arrays will be converted later in `check_array`
        # Only handle extension arrays for integer and floats
        return False
    elif is_float_dtype(pd_dtype):
        # Float ndarrays can normally support nans. They need to be converted
        # first to map pd.NA to np.nan
        return True
    elif is_integer_dtype(pd_dtype):
        # XXX: Warn when converting from a high integer to a float
        return True

    return False

def _ensure_no_complex_data(array):
    if (
        hasattr(array, "dtype")
        and array.dtype is not None
        and hasattr(array.dtype, "kind")
        and array.dtype.kind == "c"
    ):
        raise ValueError("Complex data not supported\n{}\n".format(array)) 
 
    
def _check_estimator_name(estimator):
    if estimator is not None:
        if isinstance(estimator, str):
            return estimator
        else:
            return estimator.__class__.__name__
    return None

def _set_back_to_frame (X, *,  current_step=None, columns = None, input_name ='X',
                       # type_X=None 
                       ): 
    """ If pandas dataframe passed previously is converted to Numpy array 
    after operations, the use of the columns recreates a new dataframe. 
    
    
    Parameters 
    ----------
    X: Array-like 
        Array to convert to frame. 
    current_step: str, default=None
        The stage of data.
        - {'in', 'enter'} is used to collect the name or columns of the
            the pandas series and dataframe 
        - {'out', 'back', 'convert'} is used to convert the array back to 
            series or dataframe if 
    """

    import pandas as pd 
    type_col_name = type (columns).__name_
    
    if not hasattr (X, '__array__'): 
        raise TypeError (f"Only supports array, got: {type (X).__name__!r}")
        
    if hasattr (X, 'name') :
        # keep the name of series 
        columns = X.name 
    elif hasattr (X, 'columns'): 
        # keep the columns 
        columns = X.columns 
        
    if str(current_step).lower().strip()  in {'in', 'enter'}: 
        # Get the attribute return X and columns
        return X, columns #, type (X).__name__ 
    
    if str(current_step).lower().strip() in {'out', 'back'}: 
        # if not string is given as name 
        # check whether the columns contains only one 
        # value and use it as name to skip 
        # TypeError: Series.name must be a hashable type 
        if _is_arraylike_1d(X) : 
            if not isinstance (columns, str ) and hasattr (columns, '__len__') : 
                if len(columns ) > 1: 
                    raise ValueError (
                        f"{input_name} is 1d-array, only pandas.Series "
                        "conversion can be performed while name must be a"
                         f" hashable type: got {type_col_name!r}")
                columns = columns [0]
                
                X= pd.Series (X, name =columns )
            
        else: 
            # columns is str , reconvert to a list 
            # and check whether the columns match 
            # the shape [1]
            if isinstance (columns, str ): 
                columns = [columns ]
            if not hasattr (columns, '__len__'):
                raise TypeError (" Columns for {input_name!r} must passed "
                                  f"as a list or tuple. Got {type_col_name!r}")
            if X.shape [1] != len(columns):
                raise ValueError (
                    f"Shape of passed values for {input_name} is"
                    f" {X.shape}. Columns indices imply {X.shape[1]},"
                    f"got {len(columns)}"
                                  ) 
                
            X= pd.DataFrame (X, columns = columns )
        
    return X , (X.name if hasattr (X, 'name') else X.columns ) if ( 
        hasattr(X, "name") or hasattr (X, "columns") ) else None 



def check_array(
    array,
    *,
    accept_large_sparse=True,
    dtype="numeric",
    accept_sparse=False, 
    order=None,
    copy=False,
    force_all_finite=True,
    ensure_2d=True,
    allow_nd=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    estimator=None,
    input_name="",
):

    """Input validation on an array, list, or similar.
    By default, the input is checked to be a non-empty 2D array containing
    only finite values. If the dtype of the array is object, attempt
    converting to float, raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.
   
    dtype : 'numeric', type, list of type or None, default='numeric'
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.
    order : {'F', 'C'} or None, default=None
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.
    copy : bool, default=False
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
    force_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The
        possibilities are:
        - True: Force all values of array to be finite.
        - False: accepts np.inf, np.nan, pd.NA in array.
        - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
          cannot be infinite.
          ``force_all_finite`` accepts the string ``'allow-nan'``.
           Accepts `pd.NA` and converts it into `np.nan`
    ensure_2d : bool, default=True
        Whether to raise a value error if array is not 2D.
    ensure_min_samples : int, default=1
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.
    ensure_min_features : int, default=1
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.
    estimator : str or estimator instance, default=None
        If passed, include the name of the estimator in warning messages.
    input_name : str, default=""
        The data name used to construct the error message. In particular
        if `input_name` is "X" and the data has NaN values and
        allow_nan is False, the error message will link to the imputer
        documentation.
        
    Returns
    -------
    array_converted : object
        The converted and validated array.
    """
    if isinstance(array, np.matrix):
        raise TypeError(
            "np.matrix is not supported. Please convert to a numpy array with "
            "np.asarray. For more information see: "
            "https://numpy.org/doc/stable/reference/generated/numpy.matrix.html"
        )
    xp, is_array_api = get_namespace(array)

    # reconvert to array if not a 
    # pandas series or dataframe .
    if  not ( hasattr (array , 'columns') or hasattr (array, 'name') ): 
        array = np.array (array )
         
    # store reference to original array to check if copy is needed when
    # function returns
    array_orig = array

    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, str) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, "kind"):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    # check if the object contains several dtypes (typically a pandas
    # DataFrame), and store them. If not, store None.
    dtypes_orig = None
    pandas_requires_conversion = False
    
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    if hasattr(array, "dtypes") and hasattr(array.dtypes, "__array__"):
        # throw warning if columns are sparse. If all columns are sparse, then
        # array.sparse exists and sparsity will be preserved (later).
        with suppress(ImportError):
            from pandas.api.types import is_sparse

            if not hasattr(array, "sparse") and array.dtypes.apply(is_sparse).any():
                warnings.warn(
                    "pandas.DataFrame with sparse columns found."
                    "It will be converted to a dense numpy array."
                )

        dtypes_orig = list(array.dtypes)
        pandas_requires_conversion = any(
            _pandas_dtype_needs_early_conversion(i) for i in dtypes_orig
        )
        if all(isinstance(dtype_iter, np.dtype) for dtype_iter in dtypes_orig):
            dtype_orig = np.result_type(*dtypes_orig)
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = xp.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    if pandas_requires_conversion:
        # pandas dataframe requires conversion earlier to handle extension dtypes with
        # nans
        # Use the original dtype for conversion if dtype is None
        new_dtype = dtype_orig if dtype is None else dtype
        array = array.astype(new_dtype)
        # Since we converted here, we do not need to convert again later
        dtype = None

    if force_all_finite not in (True, False, "allow-nan"):
        raise ValueError(
            'force_all_finite should be a bool or "allow-nan". Got {!r} instead'.format(
                force_all_finite
            )
        )
    estimator_name = _check_estimator_name(estimator)
    context = " by %s" % estimator_name if estimator is not None else ""
    
    if sp.issparse(array):
        _ensure_no_complex_data(array)
        array = _ensure_sparse_format(
           array,
           accept_sparse=accept_sparse,
           dtype=dtype,
           copy=copy,
           force_all_finite=force_all_finite,
           accept_large_sparse=accept_large_sparse,
           estimator_name=estimator_name,
           input_name=input_name,
       )
       
    else:
        # If np.array(..) gives ComplexWarning, then we convert the warning
        # to an error. This is needed because specifying a non complex
        # dtype to the function converts complex to real dtype,
        # thereby passing the test made in the lines following the scope
        # of warnings context manager.
        with warnings.catch_warnings():
            try:
                warnings.simplefilter("error", ComplexWarning)
                if dtype is not None and np.dtype(dtype).kind in "iu":
                    # Conversion float -> int should not contain NaN or
                    # inf (numpy#14412). We cannot use casting='safe' because
                    # then conversion float -> int would be disallowed.
                    array = _asarray_with_order(array, order=order, xp=xp)
                    if array.dtype.kind == "f":
                        _assert_all_finite(
                            array,
                            allow_nan=False,
                            msg_dtype=dtype,
                            estimator_name=estimator_name,
                            input_name=input_name,
                        )
                    array = xp.astype(array, dtype, copy=False)
                else:
                    array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
            except ComplexWarning as complex_warning:
                raise ValueError(
                    "Complex data not supported\n{}\n".format(array)
                ) from complex_warning
    
        # It is possible that the np.array(..) gave no warning. This happens
        # when no dtype conversion happened, for example dtype = None. The
        # result is that np.array(..) produces an array of complex dtype
        # and we need to catch and raise exception for such cases.
        _ensure_no_complex_data(array)
    
        
        if len(array) ==0: 
           raise ValueError (
               "Found array with 0 length while a minimum of 1 is required." )
        if ensure_2d:
            # If input is scalar raise error
            if  array.ndim == 0:
                raise ValueError(
                    "Expected 2D array, got scalar array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array)
                )
            # If input is 1D raise error
            if array.ndim == 1:
                raise ValueError(
                    "Expected 2D array, got 1D array instead. "
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample."
                )
    
        if  ( dtype_numeric 
             and ( array.values.dtype.kind if hasattr(array, 'columns') 
                  else array.dtype.kind) 
             in "USV"
             ):
            raise ValueError(
                "dtype='numeric' is not compatible with arrays of bytes/strings."
                "Convert your data to numeric values explicitly instead."
            )
        if not allow_nd and array.ndim >= 3:
            raise ValueError(
                "Found array with dim %d. %s expected <= 2."
                % (array.ndim, estimator_name)
            )
        if force_all_finite:
            _assert_all_finite(
                array,
                input_name=input_name,
                estimator_name=estimator_name,
                allow_nan=force_all_finite == "allow-nan",
            )
        
    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError(
                "Found array with %d sample(s) (shape=%s) while a"
                " minimum of %d is required."
                % (n_samples, array.shape, ensure_min_samples)
            )

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError(
                "Found array with %d feature(s) (shape=%s) while"
                " a minimum of %d is required."
                % (n_features, array.shape, ensure_min_features)
            )
                
    if copy:
        if xp.__name__ in {"numpy", "numpy.array_api"}:
            # only make a copy if `array` and `array_orig` may share memory`
            if np.may_share_memory(array, array_orig):
                array = _asarray_with_order(
                    array, dtype=dtype, order=order, copy=True, xp=xp
                )
        else:
            # always make a copy for non-numpy arrays
            array = _asarray_with_order(
                array, dtype=dtype, order=order, copy=True, xp=xp
            )
            
    return array 


def _check_large_sparse(X, accept_large_sparse=False):
    """Raise a ValueError if X has 64bit indices and accept_large_sparse=False"""
    if not accept_large_sparse:
        supported_indices = ["int32"]
        if X.getformat() == "coo":
            index_keys = ["col", "row"]
        elif X.getformat() in ["csr", "csc", "bsr"]:
            index_keys = ["indices", "indptr"]
        else:
            return
        for key in index_keys:
            indices_datatype = getattr(X, key).dtype
            if indices_datatype not in supported_indices:
                raise ValueError(
                    "Only sparse matrices with 32-bit integer"
                    " indices are accepted. Got %s indices." % indices_datatype
                )
def _ensure_sparse_format(
    spmatrix,
    accept_sparse,
    dtype,
    copy,
    force_all_finite,
    accept_large_sparse,
    estimator_name=None,
    input_name="",
):
    """Convert a sparse matrix to a given format.
    Checks the sparse format of spmatrix and converts if necessary.
    Parameters
    ----------
    spmatrix : sparse matrix
        Input to validate and convert.
    accept_sparse : str, bool or list/tuple of str
        String[s] representing allowed sparse matrix formats ('csc',
        'csr', 'coo', 'dok', 'bsr', 'lil', 'dia'). If the input is sparse but
        not in the allowed format, it will be converted to the first listed
        format. True allows the input to be any format. False means
        that a sparse matrix input will raise an error.
    dtype : str, type or None
        Data type of result. If None, the dtype of the input is preserved.
    copy : bool
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
    force_all_finite : bool or 'allow-nan'
        Whether to raise an error on np.inf, np.nan, pd.NA in X. The
        possibilities are:
        - True: Force all values of X to be finite.
        - False: accepts np.inf, np.nan, pd.NA in X.
        - 'allow-nan': accepts only np.nan and pd.NA values in X. Values cannot
          be infinite.
        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.
        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `np.nan`
    estimator_name : str, default=None
        The estimator name, used to construct the error message.
    input_name : str, default=""
        The data name used to construct the error message. In particular
        if `input_name` is "X" and the data has NaN values and
        allow_nan is False, the error message will link to the imputer
        documentation.
    Returns
    -------
    spmatrix_converted : sparse matrix.
        Matrix that is ensured to have an allowed type.
    """
    if dtype is None:
        dtype = spmatrix.dtype

    changed_format = False

    if isinstance(accept_sparse, str):
        accept_sparse = [accept_sparse]

    # Indices dtype validation
    _check_large_sparse(spmatrix, accept_large_sparse)

    if accept_sparse is False:
        raise TypeError(
            "A sparse matrix was passed, but dense "
            "data is required. Use X.toarray() to "
            "convert to a dense numpy array."
        )
    elif isinstance(accept_sparse, (list, tuple)):
        if len(accept_sparse) == 0:
            raise ValueError(
                "When providing 'accept_sparse' "
                "as a tuple or list, it must contain at "
                "least one string value."
            )
        # ensure correct sparse format
        if spmatrix.format not in accept_sparse:
            # create new with correct sparse
            spmatrix = spmatrix.asformat(accept_sparse[0])
            changed_format = True
    elif accept_sparse is not True:
        # any other type
        raise ValueError(
            "Parameter 'accept_sparse' should be a string, "
            "boolean or list of strings. You provided "
            "'accept_sparse={}'.".format(accept_sparse)
        )

    if dtype != spmatrix.dtype:
        # convert dtype
        spmatrix = spmatrix.astype(dtype)
    elif copy and not changed_format:
        # force copy
        spmatrix = spmatrix.copy()

    if force_all_finite:
        if not hasattr(spmatrix, "data"):
            warnings.warn(
                "Can't check %s sparse matrix for nan or inf." % spmatrix.format,
                stacklevel=2,
            )
        else:
            _assert_all_finite(
                spmatrix.data,
                allow_nan=force_all_finite == "allow-nan",
                estimator_name=estimator_name,
                input_name=input_name,
            )
        
    return spmatrix

def _object_dtype_isnan(X):
    return X != X

def _assert_all_finite(
    X, allow_nan=False, msg_dtype=None, estimator_name=None, input_name=""
):
    """Like assert_all_finite, but only for ndarray."""

    xp, _ = get_namespace(X)

    # if _get_config()["assume_finite"]:
    #     return

    X = xp.asarray(X)

    # for object dtype data, we only check for NaNs (GH-13254)
    if X.dtype == np.dtype("object") and not allow_nan:
        if _object_dtype_isnan(X).any():
            raise ValueError("Input contains NaN")

    # We need only consider float arrays, hence can early return for all else.
    if X.dtype.kind not in "fc":
        return

    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space `np.isinf/isnan` or custom
    # Cython implementation to prevent false positives and provide a detailed
    # error message.
    with np.errstate(over="ignore"):
        first_pass_isfinite = xp.isfinite(xp.sum(X))
    if first_pass_isfinite:
        return
    # Cython implementation doesn't support FP16 or complex numbers
    # use_cython = (
    #     xp is np and X.data.contiguous and X.dtype.type in {np.float32, np.float64}
    # )
    # if use_cython:
    #     out = cy_isfinite(X.reshape(-1), allow_nan=allow_nan)
    #     has_nan_error = False if allow_nan else out == FiniteStatus.has_nan
    #     has_inf = out == FiniteStatus.has_infinite
    # else:
    has_inf = np.isinf(X).any()
    has_nan_error = False if allow_nan else xp.isnan(X).any()
    if has_inf or has_nan_error:
        if has_nan_error:
            type_err = "NaN"
        else:
            msg_dtype = msg_dtype if msg_dtype is not None else X.dtype
            type_err = f"infinity or a value too large for {msg_dtype!r}"
        padded_input_name = input_name + " " if input_name else ""
        msg_err = f"Input {padded_input_name}contains {type_err}."
        # if estimator_name and input_name == "X" and has_nan_error:
        #     # Improve the error message on how to handle missing values in
        #     # scikit-learn.
        #     msg_err += (
        #         f"\n{estimator_name} does not accept missing values"
        #         " encoded as NaN natively. For supervised learning, you might want"
        #         " to consider sklearn.ensemble.HistGradientBoostingClassifier and"
        #         " Regressor which accept missing values encoded as NaNs natively."
        #         " Alternatively, it is possible to preprocess the data, for"
        #         " instance by using an imputer transformer in a pipeline or drop"
        #         " samples with missing values. See"
        #         " https://scikit-learn.org/stable/modules/impute.html"
        #         " You can find a list of all estimators that handle NaN values"
        #         " at the following page:"
        #         " https://scikit-learn.org/stable/modules/impute.html"
        #         "#estimators-that-handle-nan-values"
        #     )
        raise ValueError(msg_err)
        
def assert_all_finite(
    X,
    *,
    allow_nan=False,
    estimator_name=None,
    input_name="",
):
    """Throw a ValueError if X contains NaN or infinity.
    Parameters
    ----------
    X : {ndarray, sparse matrix}
        The input data.
    allow_nan : bool, default=False
        If True, do not throw error when `X` contains NaN.
    estimator_name : str, default=None
        The estimator name, used to construct the error message.
    input_name : str, default=""
        The data name used to construct the error message. In particular
        if `input_name` is "X" and the data has NaN values and
        allow_nan is False, the error message will link to the imputer
        documentation.
    """
    _assert_all_finite(
        X.data if sp.issparse(X) else X,
        allow_nan=allow_nan,
        estimator_name=estimator_name,
        input_name=input_name,
    )

def _generate_get_feature_names_out(estimator, n_features_out, input_features=None):
    """Generate feature names out for estimator using the estimator name as the prefix.
    The input_feature names are validated but not used. This function is useful
    for estimators that generate their own names based on `n_features_out`, i.e. PCA.
    Parameters
    ----------
    estimator : estimator instance
        Estimator producing output feature names.
    n_feature_out : int
        Number of feature names out.
    input_features : array-like of str or None, default=None
        Only used to validate feature names with `estimator.feature_names_in_`.
    Returns
    -------
    feature_names_in : ndarray of str or `None`
        Feature names in.
    """
    _check_feature_names_in(estimator, input_features, generate_names=False)
    estimator_name = estimator.__class__.__name__.lower()
    return np.asarray(
        [f"{estimator_name}{i}" for i in range(n_features_out)], dtype=object
    )

class PositiveSpectrumWarning(UserWarning):
    """Warning raised when the eigenvalues of a PSD matrix have issues
    This warning is typically raised by ``_check_psd_eigenvalues`` when the
    eigenvalues of a positive semidefinite (PSD) matrix such as a gram matrix
    (kernel) present significant negative eigenvalues, or bad conditioning i.e.
    very small non-zero eigenvalues compared to the largest eigenvalue.
    .. versionadded:: 0.22
    """
class DataConversionWarning(UserWarning):
    """Warning used to notify implicit data conversions happening in the code.
    This warning occurs when some input data needs to be converted or
    interpreted in a way that may not match the user's expectations.
    For example, this warning may occur when the user
        - passes an integer array to a function which expects float input and
          will convert the input
        - requests a non-copying operation, but a copy is required to meet the
          implementation's data-type expectations;
        - passes an input whose shape can be interpreted ambiguously.
    .. versionchanged:: 0.18
       Moved from sklearn.utils.validation.
    """