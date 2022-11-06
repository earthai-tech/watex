# -*- coding: utf-8 -*-
# BSD 3-Clause License
# Copyright (c) 2007-2022 The scikit-learn developers.
# All rights reserved.
"""Tools to support array_api."""
import numpy

class _ArrayAPIWrapper:
    """sklearn specific Array API compatibility wrapper
    This wrapper makes it possible for scikit-learn maintainers to
    deal with discrepancies between different implementations of the
    Python array API standard and its evolution over time.
    The Python array API standard specification:
    https://data-apis.org/array-api/latest/
    Documentation of the NumPy implementation:
    https://numpy.org/neps/nep-0047-array-api-standard.html
    """

    def __init__(self, array_namespace):
        self._namespace = array_namespace

    def __getattr__(self, name):
        return getattr(self._namespace, name)

    def take(self, X, indices, *, axis):
        # When array_api supports `take` we can use this directly
        # https://github.com/data-apis/array-api/issues/177
        if self._namespace.__name__ == "numpy.array_api":
            X_np = numpy.take(X, indices, axis=axis)
            return self._namespace.asarray(X_np)

        # We only support axis in (0, 1) and ndim in (1, 2) because that is all we need
        # in scikit-learn
        if axis not in {0, 1}:
            raise ValueError(f"Only axis in (0, 1) is supported. Got {axis}")

        if X.ndim not in {1, 2}:
            raise ValueError(f"Only X.ndim in (1, 2) is supported. Got {X.ndim}")

        if axis == 0:
            if X.ndim == 1:
                selected = [X[i] for i in indices]
            else:  # X.ndim == 2
                selected = [X[i, :] for i in indices]
        else:  # axis == 1
            selected = [X[:, i] for i in indices]
        return self._namespace.stack(selected, axis=axis)


class _NumPyApiWrapper:
    """Array API compat wrapper for any numpy version
    NumPy < 1.22 does not expose the numpy.array_api namespace. This
    wrapper makes it possible to write code that uses the standard
    Array API while working with any version of NumPy supported by
    scikit-learn.
    """

    def __getattr__(self, name):
        return getattr(numpy, name)

    def astype(self, x, dtype, *, copy=True, casting="unsafe"):
        # astype is not defined in the top level NumPy namespace
        return x.astype(dtype, copy=copy, casting=casting)

    def asarray(self, x, *, dtype=None, device=None, copy=None):
        # Support copy in NumPy namespace
        if copy is True:
            return numpy.array(x, copy=True, dtype=dtype)
        else:
            return numpy.asarray(x, dtype=dtype)

    def unique_inverse(self, x):
        return numpy.unique(x, return_inverse=True)

    def unique_counts(self, x):
        return numpy.unique(x, return_counts=True)

    def unique_values(self, x):
        return numpy.unique(x)

    def concat(self, arrays, *, axis=None):
        return numpy.concatenate(arrays, axis=axis)


def _convert_to_numpy(array, xp):
    """Convert X into a NumPy ndarray.
    Only works on cupy.array_api and numpy.array_api and is used for testing.
    """
    supported_array_api = ["numpy.array_api", "cupy.array_api"]
    if xp.__name__ not in supported_array_api:
        support_array_api_str = ", ".join(supported_array_api)
        raise ValueError(f"Supported namespaces are: {support_array_api_str}")

    if xp.__name__ == "cupy.array_api":
        return array._array.get()
    else:
        return numpy.asarray(array)


