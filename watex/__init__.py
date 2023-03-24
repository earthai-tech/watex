# -*- coding: utf-8 -*-
# Licence:BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

from __future__ import annotations 
import os 
import sys 
import logging 
import random
import warnings 
 
# set the package name for consistency checker 
sys.path.insert(0, os.path.dirname(__file__))  
for p in ('.','..' ,'./watex'): 
    sys.path.insert(0,  os.path.abspath(p)) 
    
# assert package 
if  __package__ is None: 
    sys.path.append( os.path.dirname(__file__))
    __package__ ='watex'

# configure the logger file
# from ._watexlog import watexlog
try: 
    conffile = os.path.join(
        os.path.dirname(__file__),  "watex/wlog.yml")
    if not os.path.isfile (conffile ): 
        raise 
except: 
    conffile = os.path.join(
        os.path.dirname(__file__), "wlog.yml")

# generated version by setuptools_scm 
try:
    from . import _version
    __version__ = _version.version.split('.dev')[0]
except ImportError:
    __version__ = '0.2.0' 

# # set loging Level
logging.getLogger(__name__)#.setLevel(logging.WARNING)
# disable the matplotlib font manager logger.
logging.getLogger('matplotlib.font_manager').disabled = True
# or ust suppress the DEBUG messages but not the others from that logger.
# logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# setting up
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

# Workaround issue discovered in intel-openmp 2019.5:
# https://github.com/ContinuumIO/anaconda-issues/issues/11294
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/
try:
    # This variable is injected in the __builtins__ by the build process. 
    __WATEX_SETUP__  # type: ignore
except NameError:
    __WATEX_SETUP__ = False

if __WATEX_SETUP__:
    sys.stderr.write("Partial import of watex during the build process.\n")
else:
    from . import _distributor_init  # noqa: F401
    from . import _build  # noqa: F401
    from .utils._show_versions import show_versions
    
#https://github.com/pandas-dev/pandas
# Let users know if they're missing any of our hard dependencies
_main_dependencies = ("numpy", "scipy", "sklearn", "matplotlib", 
                      "pandas","seaborn", "openpyxl")
_missing_dependencies = []

for _dependency in _main_dependencies:
    try:
        __import__(_dependency)
    except ImportError as _e:  # pragma: no cover
        _missing_dependencies.append(
            f"{'scikit-learn' if _dependency=='sklearn' else _dependency }: {_e}")

if _missing_dependencies:  # pragma: no cover
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(_missing_dependencies)
    )
del _main_dependencies, _dependency, _missing_dependencies

# Try to suppress pandas future warnings
# and reduce verbosity.
# Setup WATex public API  
with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore', category=UserWarning)
    import watex.exlib as sklearn 
    from .exlib.gbm import XGBClassifier

from .analysis import ( 
    nPCA, 
    kPCA, 
    LLE, 
    iPCA, 
    )
from .base import ( 
    Data, 
    Missing, 
    AdalineGradientDescent, 
    AdalineStochasticGradientDescent, 
    SequentialBackwardSelection, 
    ) 
from .cases import ( 
    BaseSteps, 
    Preprocessing , 
    BaseModel,
    FeatureInspection, 
    ) 
from .datasets import ( 
    fetch_data, 
    make_erp, 
    make_ves 
    ) 
from .geology import ( 
    Structural, 
    Structures 
    )
from .methods import (
    ResistivityProfiling ,
    VerticalSounding, 
    DCProfiling, 
    DCSounding, 
    EM, 
    Processing as EMProcessing, 
    MXS, 
    )
from . models import ( 
    GridSearch, 
    GridSearchMultiple,
    get_scorers, 
    pModels
    )

from .view import ( 
    EvalPlot, 
    plotLearningInspections, 
    plotSilhouette,
    plotDendrogram, 
    plotProjection, 
    QuickPlot , 
    ExPlot,
    TPlot, 
    )

from .utils import ( 
    plotAnomaly, 
    vesSelector, 
    erpSelector, 
    read_data,
    cleaner, 
    erpSmartDetector,
    type_,
    shape, 
    power, 
    magnitude, 
    sfi, 
    ohmicArea, 
    fittensor,
    get2dtensor,
    plotOhmicArea, 
    plot_sfi,
    reshape, 
    to_numeric_dtypes, 
    smart_label_classifier,
    select_base_stratum , 
    reduce_samples , 
    make_MXS_labels, 
    predict_NGA_labels, 
    classify_k,  
    plot_elbow, 
    plot_clusters, 
    plot_pca_components, 
    plot_naive_dendrogram, 
    plot_learning_curves, 
    plot_confusion_matrices, 
    plot_sbs_feature_selection, 
    plot_regularization_path, 
    plot_rf_feature_importances, 
    plot_logging, 
    plot_silhouette, 
    plot_profiling,
    plot_confidence_in,
    qc,
    )
try : 
    from .utils import ( 
        selectfeatures, 
        naive_imputer, 
        naive_scaler,  
        make_naive_pipe, 
        bi_selector, 
        )
except ImportError :
    pass 

def setup_module(module):
    """Fixture for the tests to assure globally controllable seeding of RNGs"""

    import numpy as np

    # Check if a random seed exists in the environment, if not create one.
    _random_seed = os.environ.get("WATEX_SEED", None)
    if _random_seed is None:
        _random_seed = np.random.uniform() * np.iinfo(np.int32).max
    _random_seed = int(_random_seed)
    print("I: Seeding RNGs with %r" % _random_seed)
    np.random.seed(_random_seed)
    random.seed(_random_seed)
   
__doc__= """\
A machine learning research in water exploration 
==================================================

:code:`watex` stands for *WAT-er EX-ploration*. Packages and/or modules are 
written to solve engineering problems in the field of groundwater 
exploration (GWE). Currently, dealing with: 
    
* `geophysical (from DC-Electrical to Electromagnetic)`; 
* `hydrogeology (from drilling to parameters calculation)`;
* `hydrogeophysic (predicting permeability coefficient (k), flow rate)`; 
* `EM (processing NSAMT noised data and recover missing tensors)`; 
* `geology (for stratigraphic model generation)`;
* `more...`

`WATex`_ contributes to minimize the risk of unsucessfull drillings, 
unustainable boreholes and could hugely reduce the cost of the hydrogeology 
parameter collections.

.. _WATex: https://github.com/WEgeophysics/watex/

"""
#  __all__ is used to display a few public API. 
# the public API is determined
# based on the documentation.
    
__all__ = [ 
    "sklearn", 
    "XGBClassifier", 
    "nPCA", 
    "kPCA", 
    "LLE", 
    "iPCA",  
    "Data", 
    "Missing", 
    "AdalineGradientDescent", 
    "AdalineStochasticGradientDescent", 
    "SequentialBackwardSelection", 
    "BaseSteps", 
    "Preprocessing" , 
    "BaseModel",
    "FeatureInspection", 
    "fetch_data",
    "make_erp", 
    "make_ves" , 
    "Structural", 
    "Structures", 
    "ResistivityProfiling" ,
    "VerticalSounding", 
    "DCProfiling", 
    "DCSounding", 
    "EM", 
    "EMProcessing" , 
    "MXS", 
    "get2dtensor", 
    "GridSearch", 
    "GridSearchMultiple",
    "get_scorers", 
    "pModels", 
    "EvalPlot", 
    "plotLearningInspections", 
    "plotSilhouette",
    "plotDendrogram", 
    "plotProjection", 
    "QuickPlot" , 
    "ExPlot",
    "TPlot", 
    "plotAnomaly", 
    "vesSelector", 
    "erpSelector", 
    "read_data",
    "erpSmartDetector", 
    "type_",
    "shape", 
    "power", 
    "magnitude", 
    "sfi", 
    "qc", 
    "plot_confidence_in", 
    "ohmicArea", 
    "fittensor",
    "plotOhmicArea", 
    "plot_sfi",
    "reshape", 
    "to_numeric_dtypes", 
    "smart_label_classifier",
    "select_base_stratum" , 
    "reduce_samples" , 
    "make_MXS_labels", 
    "predict_NGA_labels", 
    "classify_k",  
    "plot_elbow", 
    "plot_clusters", 
    "plot_pca_components", 
    "plot_naive_dendrogram", 
    "plot_learning_curves", 
    "plot_confusion_matrices",  
    "plot_sbs_feature_selection", 
    "plot_regularization_path", 
    "plot_rf_feature_importances", 
    "plot_logging", 
    "plot_silhouette", 
    "plot_profiling", 
    "selectfeatures", 
    "naive_imputer", 
    "naive_scaler",  
    "make_naive_pipe", 
    "bi_selector", 
    "show_versions",
    "cleaner", 
    ]

