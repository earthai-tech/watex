.. _installation:

======================================
Installing  & Getting Started
======================================

Python's Edge in watex Development
====================================

The choice of `Python <https://www.python.org/>`_ for developing :code:`watex` stems 
from Python's widespread recognition for simplicity, readability, and versatility. These qualities 
make Python especially suitable for both beginners and experienced programmers. Furthermore, Python is 
freely available, ensuring that :code:`watex` can be used and distributed without any licensing 
constraints.

Python's rich ecosystem of libraries plays a pivotal role in :code:`watex`'s functionality. Libraries 
such as `NumPy <https://numpy.org>`_, `SciPy <https://www.scipy.org>`_, `Pandas <https://pandas.pydata.org>`_, 
and `SQLite <https://sqlite.org/index.html>`_ are integral for data manipulation, statistical analysis, 
and database management. These tools collectively enable :code:`watex` to efficiently process large 
datasets and perform complex predictive analytics.

Moreover, :code:`watex` benefits from the integration with :ref:`scikit-learn <scikit-learn>`, a library 
that provides a wide range of machine learning algorithms for data mining, data analysis, and modeling. 
The combination of Python's accessible syntax and its powerful libraries ensures that :code:`watex` is a 
robust tool for geophysical exploration and analysis.

.. _NumPy: https://numpy.org
.. _SciPy: https://www.scipy.org
.. _scikit-learn: http://scikit-learn.org/stable/


Installation Guide for watex
============================

:code:`watex` is fully compatible with Pop_OS Linux environments and supports 
Python 3.9 or later versions. This flexibility ensures that :code:`watex` can be utilized across 
various systems and Python environments. To get started with :code:`watex`, Python needs to be installed 
on your system. For a comprehensive Python setup that includes a wide array of scientific libraries 
essential for data analysis, consider using `Anaconda <https://www.anaconda.com/products/distribution>`_ 
or :ref:`Miniforge3 <Miniforge3>`.

:ref:`Miniforge3` is particularly suitable for users looking for a lightweight conda 
environment that still provides access to the packages available through conda-forge, making it 
an excellent choice for :code:`watex` users.

** Setting Up Python 3.9 **

Ensure that Python 3.9 or a later version is installed on your system. This version of Python 
brings several improvements and features that enhance the functionality of :code:`watex`. You can 
verify your Python installation by running the following command in your terminal or command prompt:

.. prompt:: bash $

   python --version

If Python is not installed, or if you need to upgrade to Python 3.9 or later, visit the official `Python website <https://www.python.org/>`__ 
or use Anaconda/Miniforge3 to install the required version.

** Installing :code:`watex` **

Once Python 3.9 or later is set up, :code:`watex` can be installed directly from the source. 
This method ensures that you have the latest version of :code:`watex`, including all recent updates 
and features. To install from source, use the Anaconda Prompt or your system's terminal for command-line 
operations, or navigate through the Anaconda GUI.

For detailed instructions on installing :code:`watex` from the source, including cloning the repository 
and setting up a development environment, refer to the :ref:`From Source <from_source>` section.

.. _Miniforge3: https://github.com/conda-forge/miniforge#miniforge3


From PyPI
----------

Installing :code:`watex` from the Python Package Index (`PyPI <https://pypi.org/>`_) is straightforward 
with pip. This method ensures you are installing the latest stable version:

.. prompt:: bash $

   pip install watex

For Windows users, it's recommended to use the `-U` flag to upgrade :code:`watex` to the latest 
version if it's already installed:

.. prompt:: bash $

   pip install -U watex

To include optional dependencies that enhance :code:`watex` functionality, particularly useful for 
development and testing purposes, append `[dev]` to the package name:

.. prompt:: bash $

   pip install watex[dev]

This installs :code:`watex` along with additional packages specified as optional dependencies, enabling 
a comprehensive environment for both usage and development.

For those interested in the cutting-edge features or contributing to :code:`watex`, installing from the 
source is recommended. This approach allows you to access the most recent changes that might not yet be 
available in the PyPI release. Refer to :ref:`installing from source <from_source>` for detailed 
instructions.


From conda-forge
----------------

Installing :code:`watex` through `conda-forge <https://conda-forge.org/>`_ is a seamless process. 
Begin by incorporating `conda-forge` into your list of channels, ensuring that packages from this 
channel are prioritized. This setup ensures you get the latest compatible versions and dependencies 
managed by the `conda-forge` community:

.. prompt:: bash $

    conda config --add channels conda-forge
    conda config --set channel_priority strict

With `conda-forge` configured, proceed to install :code:`watex`:

.. prompt:: bash $

    conda install watex

To explore all the versions of :code:`watex` that are available for your specific platform via 
`conda-forge`, you can use the following command:

.. prompt:: bash $

   conda search watex --channel conda-forge

This command provides a comprehensive list, allowing you to choose a specific version if needed, 
though typically installing the latest version is recommended for most users.


From Mamba
----------

Installing :code:`watex` with `mamba <https://mamba.magna.com/downloads/software/>`_, a fast alternative 
to conda that leverages the same package repositories, simplifies and accelerates the setup process:

.. prompt:: bash $

    mamba install watex

Mamba also offers the capability to explore all available versions of :code:`watex` for your system, 
providing a fast way to check for updates or specific versions:

.. prompt:: bash $

    mamba search watex --channel conda-forge

Beyond basic installation, Mamba facilitates advanced package queries to understand :code:`watex` 
dependencies and reverse dependencies within your environment:

- To search for all available versions of :code:`watex` on your platform:

  .. prompt:: bash $

       mamba repoquery search watex --channel conda-forge

- To identify which packages depend on :code:`watex`:

  .. prompt:: bash $

      mamba repoquery whoneeds watex --channel conda-forge

- To list the dependencies of :code:`watex`:

  .. prompt:: bash $

      mamba repoquery depends watex --channel conda-forge

These Mamba commands provide comprehensive insights into the package ecosystem around :code:`watex`, 
aiding in managing your development environment more effectively.


From Source
-----------

Installing :code:`watex` directly from the source allows you to access the latest features and updates 
that may not yet be available in the official releases. To get started, clone the project repository 
from GitHub using ``git``. Navigate to the `WEgeophysics/watex` project page on 
GitHub to find the repository URL: `WEgeophysics/watex on GitHub <https://github.com/WEgeophysics/watex>`_.

Clone the repository with the following command:

.. prompt:: bash $

   git clone https://github.com/WEgeophysics/watex.git
   # Use the --depth 1 option to clone only the most recent commit history

If you're considering contributing to the :code:`watex` project, it's recommended to fork the 
repository on GitHub first. Cloning your forked version allows you to make changes in a separate 
branch and submit pull requests for review:

.. prompt:: bash $

   git clone https://github.com/WEgeophysics/watex.git
   # Replace 'WEgeophysics' with your GitHub username

This approach ensures you're working with a personal copy of the project, facilitating easier 
contribution and collaboration with the :code:`watex` development community.


Using the Command Line
----------------------

There are two recommended options for setting up your environment to work with :code:`watex`:

Option 1: Using Conda (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Creating a dedicated `conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ is 
advisable if you have installed Python via conda. This environment can house all the necessary 
:ref:`hard dependencies <dependencies>` for :code:`watex`. A `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_ can 
be set up as follows, where `<venv>` is your chosen name for the environment (e.g., `<py39>` for Python 3.9):

.. prompt:: bash $

    conda create -n venv python=3.9
    conda activate venv
    pip install scikit-learn seaborn pyyaml pyproj joblib openpyxl

It is not always necessary to install all :ref:`hard dependencies <dependencies>` for basic 
functionality, as some dependencies include others. However, for a complete setup, you can install 
the entire suite:

.. prompt:: bash $

    pip install numpy scipy scikit-learn pandas matplotlib tables h5py seaborn pyyaml h5py joblib

Refer to the list of optional :ref:`dependencies <dependencies>` for additional features.

.. note:: Use :code:`pip` for installing any packages, since not all packages are available via conda-forge.

Option 2: Using Virtualenv (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Creating a virtual environment within the project's root directory (for instance, within a project named `watex`) 
has numerous advantages. This approach isolates the project's dependencies from global Python 
installations, facilitating reproducibility and minimizing conflicts. Additionally, it simplifies 
the setup for Jupyter notebooks by avoiding the need for extra configuration steps.

**Step 1: Creating the Virtual Environment**

Depending on your operating system, use one of the following commands to create a virtual environment 
named `venv`. This name is conventional, but you can choose any name that suits your project:

.. prompt:: bash $

    # On Windows
    python -m venv venv
    # On Linux or macOS
    python -m venv ./venv

**Step 2: Verifying the Environment**

After creation, ensure your environment is set up correctly by listing its contents. This step is 
more about familiarization than verification:

.. prompt:: bash $

    ls venv/   # On Linux or macOS
    tree venv/ # If 'tree' is installed, for a more structured overview

**Step 3: Activating the Environment**

Activating your environment is crucial before installing any packages to ensure they are placed in the 
correct isolated space:

.. prompt:: bash $

    # On Windows
    venv\Scripts\activate
    # On Linux or macOS
    source ./venv/bin/activate

**Step 4: Updating Core Tools**

Before installing project-specific dependencies, update `pip`, `setuptools`, and `wheel` to their 
latest versions. These tools are essential for managing and installing Python packages:

.. prompt:: bash $

    python -m pip install --upgrade pip setuptools wheel

**Step 5: Installing Project Dependencies**

Install :code:`watex` and its dependencies within the activated virtual environment. While `conda` can be 
used for some packages, `pip` ensures compatibility within virtual environments. Here's how to install 
the required packages, including optional dependencies for extended functionalities:

.. prompt:: bash $

    pip install watex[dev]

Additionally, for improved performance, particularly in machine learning tasks, installing `scikit-learn-intelex` 
can accelerate certain computations:

.. prompt:: bash $

    pip install scikit-learn-intelex

**Additional Tips:**

- To deactivate the virtual environment and return to your global Python environment, simply run `deactivate` in your terminal.
- Always activate your project's virtual environment before running scripts or starting a Jupyter notebook to ensure 
  the correct packages and versions are used.

This guide aims to provide a detailed walkthrough for setting up a virtual environment tailored for 
:code:`watex` development, focusing on best practices and common conventions in Python project 
management.

.. _virtualenv: https://docs.python.org/3/tutorial/venv.html
.. _conda environment: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


Using GUI
---------

This installation method is optional. After installing `Anaconda <https://anaconda.org/>`_, download 
the `watex` zip file `here <https://github.com/WEgeophysics/watex/archive/refs/heads/master.zip>`_. Unzip 
the project, and use an IDE like `Spyder`, `PyCharm`, or any other of your choice, then set the root to 
your environment name. Follow the steps below for clarity:

* Open the Anaconda Navigator application.
* In the left sidebar, select `Environments`, then at the bottom of the window, select `Create`.
* Name your new environment appropriately and select Python 3.9 as the package, then click the green `Create` button to confirm.
* Select the environment you have created from the list of available environments, and in the package window to the right,
* Choose `Not installed` from the drop-down menu, type `gdal` and `libgdal` in the search bar, then click the `Apply` button in the lower right corner. A window will pop up confirming the dependencies to install.
* Repeat the process for all necessary dependencies.

Dependencies
============

The :code:`watex` package has several dependencies categorized into `hard-dependencies` and `optional 
dependencies`. The hard-dependencies are essential for the software to function correctly.

.. table::
   :widths: auto
   :class: longtable

   ========================= ========================= ===========================
   **Hard dependencies**     **Minimum version**        **Comes with**
   ------------------------- ------------------------- ---------------------------
   scikit-learn              >=1.1.2                    -
   seaborn                   >=0.12.0                   -
   pyyaml                    >=5.0.0                    -
   pyproj                    >=3.3.0                    -
   joblib                    >=1.2.0                    -
   h5py                      >=3.2.0                    pandas 
   tables                    >=3.6.0                    pandas     
   numpy                     >=1.23.0                   scikit-learn
   scipy                     >=1.9.0                    scikit-learn
   pandas                    >=1.4.0                    seaborn
   matplotlib                >=3.3.0                    seaborn                                                 
   ========================= ========================= ===========================

In principle, the first five dependencies are required. For example, the `scikit-learn` dependency 
includes `numpy` and `scipy`, so there is no need to install these separately. The table below lists 
the optional dependencies:

.. table::
   :widths: auto
   :class: longtable

   ========================= ======================
   **Optional dependencies**  **Minimum version**    
   ------------------------- ----------------------
   missingno                  >=0.4.2         
   pandas_profiling           >=2.10.0          
   pyjanitor                  >=0.22.0          
   yellowbrick                >=1.3            
   mlxtend                    >=0.18.0          
   tqdm                       >=4.59.0  
   xgboost                    >=1.5.0       
   ========================= ======================

Both :code:`conda` and :code:`pip` can be used to install these dependencies:

.. prompt:: bash $

   conda install <package-name>

If a dependency is not available in conda-forge (e.g., `pyproj`), use :code:`pip` instead:

.. prompt:: bash $

   pip install <package-name>

Getting Started
===============

To begin working with :code:`watex`, it's recommended to use the following import strategy:

.. code-block:: python

    >>> import watex as wx

:code:`watex` provides two approaches for importing modules, classes, or functions: the shorthand and 
the complete import strategies. Depending on your needs, you may choose one for convenience or specificity.

Shorthand Import Strategy
-------------------------

The shorthand strategy uses the `wx` prefix to access :code:`watex` functionalities. This method is 
straightforward and recommended for accessing common features or datasets:

.. code-block:: python

    # Example for accessing geological structures
    import watex as wx
    edi_data = wx.fetch_data('edis', return_data=True)
    # edi_data now contains an object with SEG EDI files

Complete Import Strategy
------------------------

The complete import strategy involves specifying the full path from the :code:`watex` package. This 
approach is more verbose but provides direct access to specific modules, classes, or functions:

.. code-block:: python

    from watex.datasets import load_edis
    edi_data = load_edis(return_data=True)
    # Returns the same EDI data as the shorthand method

While both methods yield the same result, the shorthand is generally limited to the public API, which 
encompasses the functions, classes, and modules most likely to be used for quick tasks or common workflows. The complete import strategy is preferable for more in-depth implementations or when accessing less common features.

For comprehensive details on :code:`watex`'s core functionality and data structures, refer to the 
:ref:`structure <structure>` documentation. If you encounter any issues or wish to contribute to the 
development of :code:`watex`, please consult the :doc:`development guide <development>`.


