# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:06:32 2022

Utility methods to print system info for debugging
adapted from :func:`pandas.show_versions`
"""
# License: BSD 3 clause
import platform
import sys
from .. import __version__
from .thread import threadpool_info
# try:
#     from ._openmp_helpers import _openmp_parallelism_enabled
# except: pass 

def _get_sys_info():
    """System information
    Returns
    -------
    sys_info : dict
        system and Python version information
    """
    python = sys.version.replace("\n", " ")

    blob = [
        ("python", python),
        ("executable", sys.executable),
        ("machine", platform.platform()),
    ]

    return dict(blob)

def _get_deps_info():
    """Overview of the installed version of main dependencies
    This function does not import the modules to collect the version numbers
    but instead relies on standard Python package metadata.
    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries
    """
    deps = [
        "pip",
        "setuptools",
        "numpy",
        "scipy",
        "scikit-learn",
        "Cython",
        "pandas",
        "matplotlib",
        "joblib",
        "seaborn",
        "openpyxl", 
    ]

    deps_info = {
        "watex": __version__,
    }

    from importlib.metadata import version, PackageNotFoundError

    for modname in deps:
        try:
            deps_info[modname] = version(modname)
        except PackageNotFoundError:
            deps_info[modname] = None
    return deps_info


def show_versions():
    """Print useful debugging information"
    
    .. versionadded:: 0.1.3
    """

    sys_info = _get_sys_info()
    deps_info = _get_deps_info()

    print("\nSystem:")
    for k, stat in sys_info.items():
        print("{k:>10}: {stat}".format(k=k, stat=stat))

    print("\nPython dependencies:")
    for k, stat in deps_info.items():
        print("{k:>13}: {stat}".format(k=k, stat=stat))

    # print(
    #     "\n{k}: {stat}".format(
    #         k="Built with OpenMP", stat=_openmp_parallelism_enabled()
    #     )
    # )
    threadpoolctl_results = threadpool_info()
    
    if threadpoolctl_results:
       print()
       print("threadpoolctl info:")

       for i, result in enumerate(threadpoolctl_results):
           for key, val in result.items():
               print(f"{key:>15}: {val}")
           if i != len(threadpoolctl_results) - 1:
               print()