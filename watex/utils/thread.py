"""Compatibility fixes for older version of python, numpy and scipy
If you add content to this file, please give the version of the package
at which the fix is no longer needed and adapted from :mod:`sklearn.utils.fixes`
"""
# Authors: Sckit-learn developers 
# License: BSD 3 clause

# from functools import update_wrapper
# import functools
import os 
import watex
import numpy as np
import scipy
import scipy.stats
import threadpoolctl
# from .._config import config_context, get_config
import threading

from ..externals._pkgs.version import parse as parse_version

np_version = parse_version(np.__version__) 
sp_version = parse_version(scipy.__version__)

def _object_dtype_isnan(X):
    return X != X

# compatibility fix for threadpoolctl >= 3.0.0
# since version 3 it's possible to setup a global threadpool controller to avoid
# looping through all loaded shared libraries each time.
# the global controller is created during the first call to threadpoolctl.
def _get_threadpool_controller():
    if not hasattr(threadpoolctl, "ThreadpoolController"):
        return None

    if not hasattr(watex, "_watex_threadpool_controller"):
        watex._sklearn_threadpool_controller = threadpoolctl.ThreadpoolController()

    return watex._sklearn_threadpool_controller


def threadpool_limits(limits=None, user_api=None):
    controller = _get_threadpool_controller()
    if controller is not None:
        return controller.limit(limits=limits, user_api=user_api)
    else:
        return threadpoolctl.threadpool_limits(limits=limits, user_api=user_api)


threadpool_limits.__doc__ = threadpoolctl.threadpool_limits.__doc__


def threadpool_info():
    controller = _get_threadpool_controller()
    if controller is not None:
        return controller.info()
    else:
        return threadpoolctl.threadpool_info()


threadpool_info.__doc__ = threadpoolctl.threadpool_info.__doc__


# TODO: Remove when SciPy 1.9 is the minimum supported version
def _mode(a, axis=0):
    if sp_version >= parse_version("1.9.0"):
        return scipy.stats.mode(a, axis=axis, keepdims=True)
    return scipy.stats.mode(a, axis=axis)



_global_config = {
    "assume_finite": bool(os.environ.get("WATEX_ASSUME_FINITE", False)),
    "working_memory": int(os.environ.get("WATEX_WORKING_MEMORY", 1024)),
    "print_changed_only": True,
    "display": "diagram",
    "pairwise_dist_chunk_size": int(
        os.environ.get("WATEX_PAIRWISE_DIST_CHUNK_SIZE", 256)
    ),
    "enable_cython_pairwise_dist": True,
    "array_api_dispatch": False,
}
_threadlocal = threading.local()

def _get_threadlocal_config():
    """Get a threadlocal **mutable** configuration. If the configuration
    does not exist, copy the default global configuration."""
    if not hasattr(_threadlocal, "global_config"):
        _threadlocal.global_config = _global_config.copy()
    return _threadlocal.global_config


def get_config():
    """Retrieve current values for configuration set by :func:`set_config`.
    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.
    See Also
    --------
    config_context : Context manager for global scikit-learn configuration.
    set_config : Set global scikit-learn configuration.
    """
    # Return a copy of the threadlocal configuration so that users will
    # not be able to modify the configuration with the returned dict.
    return _get_threadlocal_config().copy()