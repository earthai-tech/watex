.. _installation:

============================
Installing  & Quickstart 
============================

Why Python ? 
============

:code:`watex` is designed using the `Python <https://www.python.org/>`_ programming language because of its easy accessibility, 
clear syntax, and interactive shell environment. Secondly, Python is freely distributed and the powerful 
numerical libraries (NumPy_ , SciPy_ , `Pandas <https://pandas.pydata.org>`_ , 
`SQLite <https://sqlite.org/index.html>`_ ), to handle a large data sets and scikit-learn_ for prediction purposes.  
 
.. _scikit-learn: http://scikit-learn.org/stable/
.. _NumPy: https://numpy.org
.. _SciPy: https://www.scipy.org


Installing
===========

:code:`watex` has been tested on the virtual machine Pop_OS Linux env and runs successfully. Please follow the 
different steps below to properly install the software. The system requires preferably  :code:`Python >=3.9` at the time of writing. 
Python can be downloaded from `anaconda <https://www.anaconda.com/distribution/>`_ . Miniforge3_ also provides a conda-based distribution
of Python and the most popular scientific libraries. 

.. prompt:: bash $
	
	python 3.9 

It is possible to install :code:`watex` from source, using anaconda prompt or GUI . 

.. _Miniforge3: https://github.com/conda-forge/miniforge#miniforge3 

From PyPI 
-------------

:code:`watex` can be obtained from `PyPI <https://pypi.org/>`__ platform distribution as: 

.. prompt:: bash $ 
   
   pip install watex

Use ``pip install -U watex`` for window users instead. Furthermore, to get the latest development of the code, 
it is recommended to install it :ref:`from source <from_source>`. 


From conda-forge 
-----------------

Installing :code:`watex` from the `conda-forge <https://conda-forge.org/>`__ channel can be achieved by 
adding ``conda-forge`` to your channels with:

.. prompt:: bash 

    conda config --add channels conda-forge
    conda config --set channel_priority strict

Once the ``conda-forge`` channel has been enabled, :code:`watex` can be installed with:

.. prompt:: bash 

    conda install watex 

It is possible to list all of the versions of :code:`watex` available on your platform with:

.. prompt:: bash 

   conda search watex --channel conda-forge

From mamba 
------------

The installation with `mamba <https://mamba.magna.com/downloads/software/>`__ derived from conda as: 

.. prompt:: bash 

    mamba install watex 

It is possible to list all of the versions of :code:`watex` available on your platform with mamba as:

.. prompt:: bash 

    mamba search watex --channel conda-forge

Alternatively, ``mamba`` repoquery may provide more information, for instance:

* Search all versions available on your platform:
  
  .. prompt:: bash 
  
       mamba repoquery search watex --channel conda-forge

* List packages depending on `watex`:
  
  .. prompt:: bash 
  
      mamba repoquery whoneeds watex --channel conda-forge

* List dependencies of `watex`:

  .. prompt:: bash
   
      mamba repoquery depends watex --channel conda-forge


.. _from_source: 

From source 
-------------

To install from the source, clone the project with ``git`` and download the latest version from the project 
webpage: https://github.com/WEgeophysics/watex : 

.. prompt:: bash $ 

   git clone https://github.com/WEgeophysics/watex.git  # add --depth 1 
  
Moreover, if you plan on submitting a pull-request, you should clone from your fork instead.


Using Prompt
-------------

* Option 1: Recommended
 
If you installed Python with conda, we recommend to create a dedicated `conda environment`_ with all the hard :ref:`dependencies <dependencies>` 
of :code:`watex`. For instance, you can globally set up a `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_ <`venv`> 
and install dependencies( see example below). Note the <`venv`>  can be any environment name. For instance <`py39`> for Python 3.9 as:

.. prompt:: bash $

	conda create -n venv python=3.9
	conda activate venv
	pip install scikit-learn xgboost seaborn pyyaml pyproj joblib openpyxl
	
Some dependencies come with others and we dont need to install the full :ref:`hard-dependencies <dependencies>` to take 
advantage of the basic implementation. However for consistency, you can install the full hard-dependencies like 

.. prompt:: bash $ 

	pip install scikit-learn numpy scipy pandas matplotlib tables h5py xgboost seaborn openpyxl pyyaml h5py joblib
	
Check the list of optional :ref:`dependencies <dependencies>` to take advantage of additional functionalities. 

.. note:: If you use ``conda install <package name>``, some dependencies are not available in conda-forge you may use :code:`pip` instead.
 
 
* Option 2: creating virtualenv_ under the root of project (Optional) 

If you want to create your virtual environment under the root folder named `watex`, the steps below can 
guide you to check whether the installation is well done. The advantage of creating the virtualenv_ under the project 
root is that you do not need to set up the jupyter notebook environment variable.  

.. prompt:: bash $ 

	python -m  venv venv`  #(on Window ) 
	python -m venv ./venv` #(on Linux)
			
You can check your new environment and list the tree packages using: 

.. prompt:: bash $ 

	ls venv/   
	tree venv/ 
	
then you can activate the environment using: 

.. prompt:: bash $ 

	venv\Scripts\activate 	# (on Window ) 
	source ./venv/bin/activate 	# (on Linux ) 
	
You may update and upgrade :code:`pip`, :code:`setuptools` and :code:`wheel` as : 

.. prompt:: bash $ 

	python -m pip install --upgrade pip
	pip install setuptools --upgrade 
	pip install wheel --upgrade
	
Finally, you can install the software full dependencies `dependencies`_ using :code:`conda` or :code:`pip`. The command should be: 

.. prompt:: bash $  

	conda install scikit-learn=1.1.2  xgboost seaborn pyyaml pyproj joblib openpyxl h5py tables numpy scipy pandas matplotlib missingno pandas_profiling pyjanitor yellowbrick mlxtend
	
For a rapid execution of the script, you can also install `scikit-learn-intelex <https://intel.github.io/scikit-learn-intelex/>`_. 

.. prompt:: bash $

	conda install scikit-learn-intelex 


.. _virtualenv: https://docs.python.org/3/tutorial/venv.html
.. _conda environment: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
	
	
Using GUI 
----------
This installation is also optional. After installing `Anaconda <https://anaconda.org/>`_, you can download the watex zip codes 
`here <https://github.com/WEgeophysics/watex/archive/refs/heads/master.zip>`_ . Then, unzip the project, open `spyder`, `pycharm` or 
any other IDEs  and set the root to your environment name. Follow the steps below for clarity. 

* open the Anaconda Navigator app
* In the left sidebar, select `Environments`, then at the bottom of the window select `Create`
* Give your new environment a suitable name and select Python 3.9 as the package, then press the green `Create` button to confirm. 
* Select the environment you have created from the list of available environments and in the package window to the right,
* Select `Not installed` from the drop-down and enter `gdal` and ` libgdal `, then click the `Apply button` in the lower right corner and a window will display confirming dependencies to install,
* Repeat the process for all dependencies. 

.. _dependencies: 

Dependencies 
=================

The following packages are the dependencies of the :code:`watex` divided into the `hard-dependencies` and the `optional dependencies`. 
The hard-dependencies are all needed for the software to run properly. 

.. table::
   :widths: auto
   :class: longtable
   
   ========================= ========================= ===========================
   **Hard dependencies**     **Minimum version**        **Come with** 
   ------------------------- ------------------------- ---------------------------
   scikit-learn              >=1.1.2                      -
   xgboost                   >=1.5.0                      -  
   seaborn                   >=0.12.0                     -
   pyyaml                    >=5.0.0                      -
   pyproj                    >=3.3.0                      -
   joblib                    >=1.2.0                      -
   openpyxl                  >=3.0.3                      - 
   h5py                      >=3.2.0                     pandas 
   tables                    >=3.6.0                     pandas     
   numpy                     >=1.23.0                    scikit-learn
   scipy                     >=1.9.0                     scikit-learn
   pandas                    >=1.4.0                     seaborn
   matplotlib                >=3.3.0                     seaborn                                                 
   ========================= ========================= ===========================


In principles the dependencies first six dependencies are the required. For instance , scikit-learn_ dependency comes with ``numpy`` and ``scipy``, 
and don't need to install again.  The following table shows the optional dependencies 

.. table::
   :widths: auto
   :class: longtable
   
   ========================= ======================
   **Optional dependencies**   **Minimum version**    
   ------------------------- ----------------------
   missingno                  >=0.4.2         
   pandas_profiling           >=0.1.7          
   pyjanitor                  >=0.1.7          
   yellowbrick                >=1.5.0        
   mlxtend                    >=0.21          
   tqdm                       >=4.64.1         
   ========================= ======================

:code:`conda` or :code:`pip` can both use to install the dependencies as: 

.. prompt:: bash $ 
   
   conda install <package-name> 
   
If the dependencies does not exist in conda-forge (e.g. ``pyproj``), use :code:`pip` instead as: 

.. prompt:: bash $ 
   
   pip install <package-name> 


Getting started 
================

For quickstart with :code:`watex`, the following import strategy is suggested:: 

	>>> import watex as wx 
	
There are two ways to import modules, classes, or functions from :code:`~watex`, the shorthand, and the complete import strategies. For instance, 
to get the list of seven geological structures and structural pieces of information, we can use: 

*  shorthand import strategy: ``wx``  
 
.. code-block:: python 

	>>> # for geological structures
	>>> #
	>>> import watex as wx 
	>>> geo_structures= wx.Structures().fit()
	>>> geo_structures.names_ [:7] 
	('argillite',
	 'alluvium',
	 'amphibolite',
	 'anorthosite',
	 'andesite',
	 'aplite',
	 'arkose')
	>>> #
	>>> # for structural infos  
	>>> # 
	>>> structurals= wx.Structural().fit() 
	>>> structurals.names_ [:7]
	('boudin_axis',
	 'fold_axial_plane',
	 'banding_gneissosity',
	 's_fabric',
	 'fault_plane',
	 'fracture___joint_set',
	 'undifferentiated_plane')
	>>> structurals.boudin_axis.code_ 
	'lsb'
	>>> structurals.boudin_axis.name_
	'Boudin Axis'

	
* complete-import strategy: ``from watex.~``	

.. code-block:: 

	>>> from watex.geology import Structures
	>>> geo_structure = Structures().fit()
	>>> geo_structure.names_[:7] 
	('argillite',
	 'alluvium',
	 'amphibolite',
	 'anorthosite',
	 'andesite',
	 'aplite',
	 'arkose')	
	>>> from watex.geology import Structural 
	>>> structurals=Structural().fit() 
	>>> structurals.names_ [:7]
	('boudin_axis',
	 'fold_axial_plane',
	 'banding_gneissosity',
	 's_fabric',
	 'fault_plane',
	 'fracture___joint_set',
	 'undifferentiated_plane')
	>>> structurals.boudin_axis.code_ 
	'lsb'
	>>> structurals.boudin_axis.name_
	'Boudin Axis'
	

In the example above, both codes yield the same results, however the `shorthand` is limited to the public API which is determined
based on the documentation. The class, functions, and modules presumed to be the most used for solving an immediate specific task, 
are displayed as public API. To more-in depth implementation, used the `complete-import strategy` instead. 
	
For more about the core and the data structure, visit the  :ref:`structure <structure>` page. However, for any issue or contributing to the 
software development, please check the :doc:`development guide <development>`.

