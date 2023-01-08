`WATex` Installation
===================================

:code:`watex` has been  tested on virtual machine Pop_OS Linux env  and run sucessfully. 
Please follow the different steps below to propertly install the software. For GUI, you can use anaconda navigator.
Open `spyder`, `pycharm` or any other IDEs unizip the project and set the root to *watex* accordingly. That's all. 	
The package root is **watex/** and all main modules are located in source directory **./watex/**  

System requirement 
^^^^^^^^^^^^^^^^^^^^^^^^

The system requiers preferabbly

.. code-block:: bash 
	
	$ python 3.9 

and can be found from https://www.anaconda.com/distribution/ . However :code:`Python >=3.9` can also be used. 

Installation  
^^^^^^^^^^^^^^

It is possible to intall :code:`watex` from source, using anaconda prompt or GUI . 


`1. From source` 
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install from source, clone the project with git: 

.. code-block:: bash 

   git clone https://github.com/WEgeophysics/watex.git 
  
Or download the latest version from the project webpage: https://github.com/WEgeophysics/watex 


`2. Using Prompt`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* You have two options. Either, need to create your root folder named `watex` and inside this root folder  create your virtual 
environmenent (venv) or gloabally setup a new env by typing: 

.. code-block:: bash

	$ conda create -n venv python=3.9
	
or 

.. code-block:: bash

	$ python -m  venv venv`  #(on Window ) 
	$ python -m venv ./venv` # (on Linux)
			
			
* Check your new environment and list the tree packages using: 

.. code-block:: bash

	$ ls venv/   
	$ tree venv/ 
	
* Then activate the environment using : 

.. code-block:: bash

	$ conda activate venv 

or 

.. code-block:: bash

	$ venv\Scripts\activate 		# (on Window ) 
	$ source ./venv/bin/activate 	# (on Linux ) 
	
* Update and upgrade `pip`, `setuptools` and `wheel` as : 

.. code-block:: bash

	$ python -m pip install --upgrade pip
	$ pip install setuptools --upgrade 
	$ pip install wheel --upgrade
	
	
Install the software dependancies using `conda` or `pip`. Note that some dependencies are not available in conda-forge. Use `pip` instead. The command should be: 

.. code-block:: bash 

	$ conda install scikit-learn=1.1.2 numpy scipy pandas matplotlib xgboost tqdm seaborn pyjanitor  missingno h5py joblib yellowbrick
	$ conda install scikit-learn-intelex 
	
	
`3. Using GUI` 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	* After installing Anaconda, open the Anaconda Navigator app
	* In the left sidebar, select `Environments`, then at the bottom of the window select `Create`
	* Give your new environment a suitable name and select Python 3.9 as the package, then press the green Create button to confirm. 
	* select the environment you have created from the list of available environments and in the packages window to the right,
	* select _Not installed_ from the drop-down and enter
	`gdal` and ` libgdal `, then click the `Apply button` in the lower right corner and a window will display confirming dependencies to install,
	* Repeat the process for all dependencies. 
	


Dependencies 
^^^^^^^^^^^^^^^^^^^^^^^
The following packages are the dependencies of the :code:`watex`. However, all are not compulsory for the software to 
run properly( base implementation) except the package following by `*`. 

	* cython
	* matplotlib>=3.3.0 *
	* numpy *
	* scipy *
	* qtpy
	* netcdf4 
	* Numexpr >= 2.6.2
	* blosc >= 1.4.1
	* pytest
	* flake8
	* flask
	* pyyaml *
	* pyproj>=1.9.6
	* pandas *
	* python-coveralls 
	* sklearn=1.1.2 *
	* joblib *
	* seaborn *
	* tqdm
	* autoapi 
	* xgboost *
	* click 
	* missingno
	* pandas_profiling 
	* pyjanitor 
	* openpyxl *
	* threadpoolctl >= 2.0.0
	* h5py >=3.2.0 *
