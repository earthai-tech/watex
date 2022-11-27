#!/usr/bin/env python

import watex
import os 

try:
    from setuptools import setup  
except ImportError:
    setuptools = False
    from distutils.core import setup
else:
    setuptools = True
    
with open(os.path.join(os.path.abspath('.'), 'README.md'), 'r') as fm:
    LONG_DESCRIPTION =fm.read()

setup_kwargs = {}
setup_kwargs['entry_points'] = {
    'watex.commands': [
        'say-hello=mypkg.watex_cli:cli',
        ],
    'console_scripts':[
             # 'occambuildinputs=watex.cli.occambuildinputs:main'
                     ]
     }
                     
# But many people will not have setuptools installed, so we need to handle
# the default Python installation, which only has Distutils:
if setuptools is False:
    # Different script specification style for ordinary Distutils:
    setup_kwargs['scripts'] = [
        s.split(' = ')[1].replace('.', '/').split(':')[0] + '.py' for s in 
        setup_kwargs['entry_points']['console_scripts']]
    del setup_kwargs['entry_points']

# "You must explicitly list all packages in packages: the Distutils will not
# recursively scan your source tree looking for any directory with an
# __init__.py file"
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
    'watex.cases'
     ]
# force install watex. Once watex is installed , pyyaml and pyproj 
# should already installed too. 
setup_kwargs['install_requires'] = [
    'numpy>=1.8.1',
    'scipy>=0.14.0',
    'matplotlib',
    'mtpy >=1.1.0',
    'threadpoolctl >= 2.0.0', 
    'pyyaml',
    'pyproj',
    'configparser', 
    'tqdm', 
    'pycsamt' 
    'autoapi' 
    'xgboost'
    'click' 
    'missingno'
    'pandas_profiling' 
    'pyjanitor' 
    'openpyxl'
 ]
                                     
setup_kwargs['python_requires'] ='>=3.8'

setup(
 	name="watex",
 	version=watex.__version__,
 	author="Kouadio K. Laurent",
    author_email='etanoyau@gmail.com',
    maintainer="Kouadio K. Laurent",
    maintainer_email='etanoyau@gmail.com',
 	description="A machine learning research package for Hydrogeophysic",
 	long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/WEgeophysics/watex",
    project_urls={
        "API Documentation"  : "https://watex.readthedocs.io/en/latest/api/watex.html",
        "Home page" : "https://github.com/WEgeophysics/watex/wiki",
        "Bugs tracker": "https://github.com/WEgeophysics/watex/issues",
        "Installation guide" : "https://github.com/WEgeophysics/watex/wiki/watex-installation-guide-for-Windows--and-Linux", 
        #"User guide" : "https://github.com/WEgeophysics/watex/blob/develop/docs/watex%20User%20Guide.pdf",
        },
 	include_package_data=True,
 	license="BSD 3-Clause LICENCE v3",
 	classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        # "Topic :: Software Development :: Build Tools",
        "Topic :: Software Development",
        'Topic :: Scientific/Engineering :: Geophysics',
        'Topic :: Scientific/Engineering :: Geosciences',
        'Programming Language :: C ',
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        ],
    keywords="hydrogeophysic, groundwater, machine learning, water , geophysic",
    zip_safe=True, 
    #package_dir={"": "watex"},  # Optional
 	# data_files=[('', ['watex/tools/epsg.npy',]),], #this will install datafiles in wearied palce such as ~/.local/
    package_data={'watex': [
                            'utils/_openmp_helpers.pxd', 
                            'utils/espg.npy',
                            'etc/*', 
                            'datasets/descr/*', 
                            'datasets/data/*', 
                            'wlog.yml', 
                            'wlogfiles/*.txt',
                            '_build/*'
                            
                            ], 
                    "":["*.pxd",
                        'data/*', 
                        'examples/*', 
                        'ipynb/*', 
                        '.checkpoints/*', 
                        ]
                  },
    
 	**setup_kwargs
)
























