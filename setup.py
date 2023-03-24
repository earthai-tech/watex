#!/usr/bin/env python

from setuptools import setup #. find_packages

try:
    import builtins
except ImportError:
    # Python 2 compat: just to be able to declare that Python >=3.8 is needed.
    import __builtin__ as builtins

# This is a bit (!) hackish: we are setting a global variable so that the main
# watex __init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet.
builtins.__WATEX_SETUP__ = True

# We can actually import watex version from 
# in editable mode :$ python -m pip install -e .
try: 
    import watex  # noqa
    VERSION = watex.__version__
except: VERSION ='0.2.0'
# set global variables 
DISTNAME = "watex"
DESCRIPTION= "Machine learning research in water exploration"
with open('README.md', 'r', encoding ='utf8') as fm:
    LONG_DESCRIPTION =fm.read()
MAINTAINER = "Laurent Kouadio"
MAINTAINER_EMAIL = 'etanoyau@gmail.com'
URL = "https://github.com/WEgeophysics/watex"
DOWNLOAD_URL = "https://pypi.org/project/watex/#files"
LICENSE = "BSD-3-Clause"
PROJECT_URLS = {
    "API Documentation"  : "https://watex.readthedocs.io/en/latest/api_references.html",
    "Home page" : "https://watex.readthedocs.io",
    "Bugs tracker": "https://github.com/WEgeophysics/watex/issues",
    "Installation guide" : "https://watex.readthedocs.io/en/latest/installation.html", 
    "User guide" : "https://watex.readthedocs.io/en/latest/user_guide.html",
}
KEYWORDS= "exploration, groundwater, machine learning, water, hydro-geophysic"
# the commented metadata should be upload as
# packages rather than data. See future release about 
# setuptools: see https://packaging.python.org/en/latest/specifications/declaring-project-metadata/
PACKAGE_DATA={ 
    'watex': [
            'utils/_openmp_helpers.pxd', 
            'utils/espg.npy',
            'etc/*', 
            'wlog.yml', 
            'wlogfiles/*.txt',
                ], 
        "":["*.pxd",
            'data/*', 
            'examples/*.py', 
            'examples/*.txt', 
            ]
 }
# setting up 
#initialize
setup_kwargs = dict()
# commands
setup_kwargs['entry_points'] = {
    'watex.commands': [
        'wx=watex.cli:cli',
        ],
    'console_scripts':[
        'version= watex.cli:version', 
                    ]
      }

setup_kwargs['packages'] = [ 
    'watex',
    'watex.datasets',
    'watex.utils',
    'watex.etc',
    'watex.analysis',
    'watex.methods',
    'watex.models',
    'watex.externals',
    'watex.geology',
    'watex.exlib',
    'watex.cases', 
    'watex.view',
    'watex.datasets.data', 
    'watex.datasets.descr', 
    'watex.datasets.data.edis', 
    'watex._build', 
    'watex.externals._pkgs', 
     ]

setup_kwargs['install_requires'] = [    
    "seaborn >=0.12.0", 
    "pandas >=1.4.0",
    "cython >=0.29.33",
    "numpy >=1.23.0",
    "scipy >=1.9.0",
    "pyyaml >=5.0.0", 
    "openpyxl >=3.0.3",
    "xgboost >=1.7.3", 
    "pyproj >=3.3.0",
    "pycsamt >=1.1.2",
    "h5py >=3.2.0",
    "tables >=3.6.1",
    "tqdm >=4.64.1",
    "scikit-learn >=1.2.1", 
    "joblib >=1.2.0",
    "threadpoolctl >=3.1.0",
    "matplotlib ==3.5.3",
    "pytest"
 ]
                               
setup_kwargs['python_requires'] ='>=3.9'

setup(
 	name=DISTNAME,
 	version=VERSION,
 	author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
 	description=DESCRIPTION,
 	long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    download_url=DOWNLOAD_URL, 
    project_urls=PROJECT_URLS,
 	include_package_data=True,
 	license=LICENSE,
 	classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
        'Topic :: Scientific/Engineering',
        'Programming Language :: C ',
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        ],
    keywords=KEYWORDS,
    zip_safe=True, 
    package_data=PACKAGE_DATA,
 	**setup_kwargs
)

























