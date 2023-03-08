#!/usr/bin/env python

import watex
import os 

try:
    from setuptools import setup # find_packages
except ImportError:
    setuptools = False
    #from distutils.core import setup
else:
    setuptools = True
    
with open(os.path.join(os.path.abspath('.'), 'README.md'), 'r', 
          encoding ='utf8') as fm:
    LONG_DESCRIPTION =fm.read()

setup_kwargs = {}

#commands
setup_kwargs['entry_points'] = {
    'watex.commands': [
        'welcome-hello=watex.watex_cli:cli',
        ],
    'console_scripts':[
              # 'occambuildinputs=watex.cli.occambuildinputs:main'
              'wx = watex.cli:cli', 
              'version = watex.cli:show_wx_version'
                      ]
      }
# setup_kwargs['entry_points'] = {}                
# But many people will not have setuptools installed, so we need to handle
# the default Python installation, which only has Distutils:
# if setuptools is False:
#     # Different script specification style for ordinary Distutils:
#     setup_kwargs['scripts'] = [
#         s.split(' = ')[1].replace('.', '/').split(':')[0] + '.py' for s in 
#         setup_kwargs['entry_points']['console_scripts']]
#     del setup_kwargs['entry_points']

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
    'watex.cases', 
    'watex.view'
     ]
# force install watex. Once watex is installed , pyyaml and pyproj 
# should already installed too. 
setup_kwargs['install_requires'] = [
    "click>=8.0.4",
    "scikit-learn>=1.1.2",
    "xgboost>=1.5.0",
    "seaborn>=0.12.0",
    "pyyaml>=5.0.0",
    "pycsamt>=1.0.0",
    "pyproj>=3.3.0",
    "joblib>=1.2.0",
    "openpyxl>=3.0.3",
    "h5py>=3.2.0",
    "tables>=3.6.0",
    "numpy>=1.23.0",
    "scipy>=1.9.0",
    "pandas>=1.4.0",
    "matplotlib==3.2.0",
    "missingno>=0.4.2",
    "pandas_profiling>=0.1.7",
    "pyjanitor>=0.1.7",
    "yellowbrick>=1.5.0",
    "mlxtend>=0.21",
    "tqdm>=4.64.1",
 ]
                                     
setup_kwargs['python_requires'] ='>=3.9'

setup(
 	name="watex",
 	version=watex.__version__,
 	author="Laurent Kouadio",
    author_email='etanoyau@gmail.com',
    maintainer="Laurent Kouadio",
    maintainer_email='etanoyau@gmail.com',
 	description="Machine learning research in water exploration",
 	long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/WEgeophysics/watex",
    project_urls={
        "API Documentation"  : "https://watex.readthedocs.io/en/latest/api_references.html",
        "Home page" : "https://watex.readthedocs.io",
        "Bugs tracker": "https://github.com/WEgeophysics/watex/issues",
        "Installation guide" : "https://watex.readthedocs.io/en/latest/installation.html", 
        "User guide" : "https://watex.readthedocs.io/en/latest/user_guide.html",
        },
    #packages=find_packages(),
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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        ],
    keywords="exploration, groundwater, machine learning, water , hydro-geophysic",
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

# if __name__ == "__main__":
#     setup_package()
























