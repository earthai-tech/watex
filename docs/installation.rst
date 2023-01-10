.. _installation:

==================
Installing 
==================

:code:`watex` has been  tested on virtual machine Pop_OS Linux env  and run sucessfully. 
Please follow the different steps below to propertly install the software. The system requiers preferably

.. code-block:: bash 
	
	$ python 3.9 

and can be found from `anaconda <https://www.anaconda.com/distribution/>`_ . However :code:`Python >=3.9` can also be used. 

Getting started 
=================

It is possible to intall :code:`watex` from source, using anaconda prompt or GUI . 

From source 
-------------

To install from source, clone the project with git: 

.. code-block:: bash 

   $ git clone https://github.com/WEgeophysics/watex.git 
  
Or download the latest version from the project webpage: https://github.com/WEgeophysics/watex 


Using Prompt
-------------

* Option 1: Fast and easy 
 
You can globally setup a virtual environment <`venv`> and straighforwardly install dependencies. 
Note the <`venv`> can be any enviromment name. For instance <`py39`> for Python 3.9 as:

.. code-block:: bash

	$ conda create -n venv python=3.9
	$ conda activate venv
	$ pip install scikit-learn numpy scipy pandas matplotlib tables h5py xgboost seaborn openpyxl pyyaml h5py joblib
	
Have a look of optional :ref:`dependencies <dependencies>` below to take advantage of additional functionalities. 

* Option 2: Optional 

If you want create your virtual environement under the root folder named `watex`, the steps below can guide you to see 
whether the installation is well done. The advantage of creating the env under the project root is that no need to configure 
jupyter notebook enviromement variable.  

.. code-block:: bash

	$ python -m  venv venv`  #(on Window ) 
	$ python -m venv ./venv` #(on Linux)
			
You can check your new environment and list the tree packages using: 

.. code-block:: bash

	$ ls venv/   
	$ tree venv/ 
	
then you can activate the environment using: 

.. code-block:: bash

	$ venv\Scripts\activate 	# (on Window ) 
	$ source ./venv/bin/activate 	# (on Linux ) 
	
You may update and upgrade :code:`pip`, :code:`setuptools` and :code:`wheel` as : 

.. code-block:: bash

	$ python -m pip install --upgrade pip
	$ pip install setuptools --upgrade 
	$ pip install wheel --upgrade
	
Finally, you can install the software dependencies using :code:`conda` or :code:`pip`. Note that some dependencies are 
not available in conda-forge. Use :code:`pip` instead. The command should be: 

.. code-block:: bash 

	$ conda install scikit-learn=1.1.2 numpy scipy pandas matplotlib xgboost tqdm seaborn pyjanitor missingno h5py joblib yellowbrick 
	$ conda install scikit-learn-intelex # for rapid execution of the script
	
	
Using GUI 
-------------------
This installation is once again optional. After installing `Anaconda <https://anaconda.org/>`_, you can open `spyder`, `pycharm` or 
any other IDEs unizip the project and set the root to your environement name. Follow the steps below for clarity. 

* open the Anaconda Navigator app
* In the left sidebar, select `Environments`, then at the bottom of the window select `Create`
* Give your new environment a suitable name and select Python 3.9 as the package, then press the green `Create` button to confirm. 
* select the environment you have created from the list of available environments and in the package window to the right,
* select `Not installed` from the drop-down and enter `gdal` and ` libgdal `, then click the `Apply button` in the lower right corner and a window will display confirming dependencies to install,
* Repeat the process for all dependencies. 


.. _dependencies: 

Dependencies 
=================

The following packages are the dependencies of the :code:`watex`. However, all are not compulsory for the software to 
run properly(base implementation) except the package following by << * >>. 

* scikit-learn >=1.1.2 *
* numpy *
* scipy *
* pandas *
* matplotlib>=3.3.0 *
* joblib *
* seaborn *
* xgboost *
* pyyaml *
* pyproj *
* openpyxl *
* h5py >=3.2.0 *
* tables *

* tqdm 
* missingno
* pandas_profiling 
* pyjanitor 
* yellowbrick
* mlxtend 

Use :code:`$ pip install <package-name>` or :code:`$ conda install < package-name>` for installation.