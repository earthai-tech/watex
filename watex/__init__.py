# -*- coding: utf-8 -*-
# Licence:BSD 3-Clause
# author: @Daniel<etanoyau@gmail.com>
"""
💧 A machine learning research package for hydrogeophysic 
===========================================================

:code:`watex` stands for *WAT-er EX-ploration*. Its packages and modules are 
written to solve real-engineering problems in the field of groundwater 
exploration (GWE). Currently, it deals with the differents methods below: 
    
    * `geophysical (from DC-Electrical to Electromagnetic)` 
    * `hydrogeology (from drilling to parameters calculation)`
    * `geology (for stratigraphic model generation)`
    * `predicting permeability coefficient (k), flow rate and else` 
    
All methods mainly focus on GWE field. One of the main advantage using `WATex`_ 
is the application of machine learning methods in the hydrogeophysic parameter 
predictions. It contributes to minimize the risk of unsucessfull drillings and 
the hugely reduce the cost of the hydrogeology parameter collections.

.. _WATex: https://github.com/WEgeophysics/watex/
.. _SDGn6: https://www.un.org/sustainabledevelopment/development-agenda/

"""
import os 
import sys 
import logging 
import random
 
__version__='0.1.2' ; __author__= 'LKouadio'

# set the package name 
# for consistency ckecker 
sys.path.insert(0, os.path.dirname(__file__))  
for p in ('.','..' ,'./watex'): 
    sys.path.insert(0,  os.path.abspath(p)) 
    
# assert packages 
if  __package__ is None: 
    sys.path.append( os.path.dirname(__file__))
    __package__ ='watex'

# configure the logger 
# from ._watexlog import watexlog
try: 
    conffile = os.path.join(
        os.path.dirname(__file__),  "watex/wlog.yml")

    if not os.path.isfile (conffile ): 
        raise 
except: 
    conffile = os.path.join(
        os.path.dirname(__file__), "wlog.yml")

# # set loging Level
logging.getLogger(__name__)#.setLevel(logging.WARNING)
# disable the matplotlib font manager logger.
logging.getLogger('matplotlib.font_manager').disabled = True
# or ust suppress the DEBUG messages but not the others from that logger.
# logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


# setting up 
# On OSX, we can get a runtime error due to multiple OpenMP libraries loaded
# simultaneously. This can happen for instance when calling BLAS inside a
# prange. Setting the following environment variable allows multiple OpenMP
# libraries to be loaded. It should not degrade performances since we manually
# take care of potential over-subcription performance issues, in sections of
# the code where nested OpenMP loops can happen, by dynamically reconfiguring
# the inner OpenMP runtime to temporarily disable it while under the scope of
# the outer OpenMP parallel section.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

# Workaround issue discovered in intel-openmp 2019.5:
# https://github.com/ContinuumIO/anaconda-issues/issues/11294
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of watex when
    # the binaries are not built
    # mypy error: Cannot determine type of '__WATEX_SETUP__'
    __WATEX_SETUP__  # type: ignore
except NameError:
    __WATEX_SETUP__ = False

if __WATEX_SETUP__:
    sys.stderr.write("Partial import of watex during the build process.\n")
    # We are not importing the rest of watex during the build
    # process, as it may not be compiled yet
else:
    # `_distributor_init` allows distributors to run custom init code.
    # For instance, for the Windows wheel, this is used to pre-load the
    # vcomp shared library runtime for OpenMP embedded in the watex/.libs
    # sub-folder.
    # It is necessary to do this prior to importing show_versions as the
    # later is linked to the OpenMP runtime to make it possible to introspect
    # it and importing it first would fail if the OpenMP dll cannot be found.
    from . import _distributor_init  # noqa: F401
    from . import _build  # noqa: F401
    from .utils._show_versions import show_versions

    # import required subpackages  
        
    __all__ = [
        "analysis", 
        "datasets", 
        "etc", 
        "exlib", 
        "tools", 
        "externals",
        "geology", 
        "show_versions", 
        "models", 
        "methods", 
        "view",
    ]


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