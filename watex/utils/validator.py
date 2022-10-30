# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2007-2022 The scikit-learn developers.
# All rights reserved.

# Note that this module is not the sckit-learn original file, 
# some functions have been removed to keep only for 
# the usefull for of watex package. Furthermore some others 
# function have been edited and added. 

# Utilities for input validation

from functools import wraps
import warnings
import numbers
import operator
import numpy as np
import scipy.sparse as sp
from inspect import signature, Parameter

import joblib


FLOAT_DTYPES = (np.float64, np.float32, np.float16)

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

def _check_array_in(obj, /, arr_name):
    """Return the array from the array name attribute. Note that the singleton 
    arry is not admitted. 
    
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