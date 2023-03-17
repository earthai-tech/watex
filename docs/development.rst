
.. _developement: 

=============================
Development Guide
=============================

*Are you a geoscientist? and/or a developer ?*

*Welcome to WATex developement guide. We expect with your ideas and contributions to build the most famous library 
for solving geosciences issues related to ground/water exploration.*  

.. admonition:: Encouraged 

	For those who are new in Python, click on :ref:`external resources <external_resources>` for getting started with the 
	first step in Python development. 


Foreword 
===========
Before starting, remember that :code:`watex` is not only dedicated to machine learning purposes. It is a crossroad 
between machine learning and geosciences, especially hydrogeology and/or geophysics. For clarity, it can be considered as an applied machine learning library to geosciences 
especially in hydro-geophysics. However, it is not limited to that way, some useful algorithms for prediction purposes or not 
can also be developed, provided they could help users ( academic or geosciences community or else ) to achieve excellent results 
in groundwater exploration (GWE field). 

There are several online tutorials available which are geared toward specific subject areas:

- `Introduction to Machine Learning in Geophysics <https://www.epts.org/courses/standard-courses/geophysics/introduction-to-machine-learning-(ml)-for-geophysics/>`_
- `Machine Learning and AI in Geophysics <https://seg.org/Events/Applications-of-Machine-Learning-and-AI-in-Geophysics>`_
- `Machine Learning and seismic interpretation <https://wiki.seg.org/wiki/Machine_learning_and_seismic_interpretation>`_
- `Machine Learning for NeuroImaging in Python <https://nilearn.github.io/>`_
- `Machine Learning for Astronomical Data Analysis <https://github.com/astroML/sklearn_tutorial>`_


Core Philosophy 
==================

The core philosophy of :code:`watex` is inspired from `Scikit-learn <https://scikit-learn.org/stable/index.html>`_ 
and a bit more `GMT <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019GC008515>`_.  It is an open project 
and everyone is welcome to contribute. 

The project is not selective when it's come to adding new algorithms provided that the contributors have minimum knowledge 
about programming and/or geosciences. The best way to contribute and help the project is to start working on both challenges.

The project is making its road expecting to reach maximum audience in geosciences community within five years. Because of this 
aim in mind, we keep relative and not strict rules. We expect the contributors to bring their 
ideas for future possible direction ( keep in mind, always fitting the `SDG n6 <https://unric.org/en/sdg-6/>`_). Many contributors 
with diverse skills will pull the project to move forward. 

However, there are different ways to contribute: 

* Improving the documentation: If you find a typo in the documentation or have made improvements, do not hesitate to send an email to the mailing 
  list or preferably submit a GitHub pull request. Please we expect readers to be a bit lenient when it's come to the level of English language, 
  since the designer is not a native English speaker and we try to re-check the whole documentation every time as possible in a spared time
  to fix all existing grammar and typos. 
  
* Report issues you're facing, and give a "thumbs up" on issues that others reported and that are relevant to you.  It also helps 
  us if you spread the word: reference the project from your blog and articles, link to it from the website, or simply star to say "I use it":

.. raw:: html

   <a class="github-button" href="https://github.com/WEgeophysics/watex"
   data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star
   WEgeophysics/watex on GitHub">Star</a>
   <script async defer src="https://buttons.github.io/buttons.js"></script>

* Adding new algorithms. This is addressed to Python and/or Cython developers and is the one main topic we will cover here.

Development 
==============
	
Fork the repository 
------------------------

The first step of contribution to :code:`watex` is to fork the `main repository <https://github.com/watex/watex/>`__ on GitHub,
then submit a ``pull request`` (PR) as:

1. `Create an account <https://github.com/join>`_ on
   GitHub if you do not already have one.

2. Fork the `project repository
   <https://github.com/WEgeophysics/watex>`__: click on the 'Fork'
   button near the top of the page. This creates a copy of the code under your
   account on the GitHub user account. For more details on how to fork a
   repository see `this guide <https://help.github.com/articles/fork-a-repo/>`_.

3. Clone your fork of the :code:`watex` repo from your GitHub account to your
   local disk:

   .. prompt:: bash $

      git clone git@github.com:YourLogin/watex.git  # add --depth 1 if your connection is slow
      cd watex

3. Install the development :ref:`dependencies <dependencies>`:

   .. prompt:: bash $

        pip install scikit-learn numpy scipy pandas matplotlib tables h5py xgboost seaborn openpyxl pyyaml h5py joblib

.. _upstream:

4. Add the ``upstream`` remote. This saves a reference to the main
   watex repository, which you can use to keep your repository
   synchronized with the latest changes:

   .. prompt:: bash $

        git remote add upstream git@github.com:WEgeophysics/watex.git

5. Check that the `upstream` and `origin` remote aliases are configured correctly
   by running `git remote -v` which should display::

        origin	git@github.com:YourLogin/watex.git (fetch)
        origin	git@github.com:YourLogin/watex.git (push)
        upstream	git@github.com:WEgeophysics/watex.git (fetch)
        upstream	git@github.com:WEgeophysics/watex.git (push)

You should now have a working installation of watex, and your git
repository properly configured. 


Add Algorithms 
---------------

There are two steps to follow when adding algorithms to :code:`watex`.  

* Development following the scikit-learn API 
* Development following the `GMT <https://www.generic-mapping-tools.org/>`_  

Development following the scikit-learn API  (DSKL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
DSKL  adopts the *fit()* method for computing and populating attributes of the instantiated models 
even the plotting modules. For  supervising or unsupervised learning, it often implements the *transform()* 
or *predict()* methods to transform data or inferred properties which consist to:

* Choose the class of model by importing the appropriate module, class estimator, or assessor. The assessor is the class of 
  the module designed for solving a specific task. 
* Choose model hyperparameters by instantiating this class with desired values. 
* arrange data into a feature matrix and target vector following the discussion from before. 
* fit the model to your data by calling the fit() method of the instantiated model even the plotting modules. 
* apply the method to new data. For supervising learning, often labels are predicted for unknown data using 
  the prediction methods whereas for unsupervised learning, the data are often transformed or 
  inferred properties using the *transform ()* or *predict()* methods. 
 
This is very common when it comes to developers familiar with `Scikit-learn <https://scikit-learn.org/stable/index.html>`__.  

Note all classes following the DSKL must follow the Python class convention rules `PEP8 <https://peps.python.org/pep-0008/>`__. 
and adopts `fit` method for populating attributes and doing the first operation like modular calculus, validating the data structure, control the parameter etc. 

However, all the parameters directly requested by the class (class parameters ) should be the same name as instance attributes. Moreover, 
each inner attribute (attributes that are not physically known by the users ) should hold an underscore *_* at the final of the name. Here 
is an example of a demo class: 

.. code-block:: python 

	>>> class DemoClass: 
		   """class description and documentation  """"
		   def __init__(self, param1=value1 , param2=value2, **kws): 
			  self.param1=param1
			  self.param2=param2 
		   def fit(self, data,  **fit_params): 
			  """ Documentation of fit parameters """
			  X= fit_params.pop(X, None) 
			  y= fit_params.pop(y, None)
			  ...
			
			  self.param3_= ... 
			  ... 
			
			  return self 


The *fit* method must always return the object *self*. When algorithms are not designed for prediction purposes, :math:`X` 
and :math:`y` must be a *fit_params* keywords argument plus other keyword parameters.
	
Conversely to `Scikit-learn <https://scikit-learn.org/stable/index.html>`_ , all algorithms are not 
dedicated to prediction purposes since the library is not only for pure machine  learning library rather for its 
application to solve geosciences engineering problems. However,new ML algorithms can also be developed and tested with a real-case study 
for efficaciousness. The *fit_params* can be any other parameters. For that reason, *fit* method could be adopted everywhere in any function.  

If there is a geosciences problem (not related to pure hydro-geophysics) that the developer wants to solve, the module can be created under the sub-package :mod:`~watex.geology`. 

When a new algorithm for *prediction* is designed, It must adopt the **predict** methods and/or **transform** or **fit_transform**. In that case 
the new class must not contain keywords arguments like:

.. code-block:: python 

	>>> from watex.exlib.sklearn import BaseEstimator, TransformerMixin
	>>> class DemoClass(BaseEstimator, TransformerMixin): 
		   """Class description and documentation  """"
		   def __init__(self, param1=value1 , param2=value2): 
		      self.param1=param1
		      self.param2=param2 
		   def fit(self, X, y= None, **fit_params): 
		      """ Documentation of fit DemoClass method """
		      self.param3_= ... 
		      ...
		      return self 
		      
		   def predict (self, X): 
		      """Documentation of DEmoClass predict method """
		      ...
		      return Xp


In the example above, :math:`X_p` is the predicted value from :math:`X`. We can also notice that there are no-keywords arguments at the class 
initialization and inherits from :class:`~watex.exlib.BaseEstimator` and :class:`~watex.exlib.TransformerMixin`. Indeed, this useful for cross-validation 
and fine-tune hyperparameters. 

.. note:: 

	If you are not a scikit-learn user, you can design the algorithms at your own. However, user must indicate in the second line of the documentation which 
	technique or machine adopted for fine-tuning hyperparameters, such as the other machine-learning libraries:   
	`Keras <https://github.com/keras-team/keras>`__, `Tensors flow <https://github.com/tensorflow/tensorflow>`__ etc. Furthermore, the reason for 
	postponing the validation following the `Scikit-learn <https://scikit-learn.org/stable/index.html>`_  API is that the same validation  would have to be performed in ``set_params``, 
	which is used in algorithms like :class:`watex.exlib.GridSearchCV` | :mod:`watex.models.GridSearch` | :mod:`watex.models.GridSearchMultiple`. 

	
Development following  GMT (DGMT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The GMT development does not imply strict rules. However, to mark the difference with the DSKL, all GMT classes must end with underscore '_'. It does 
need the *fit* method. Once the class is called, all attributes must be initialized and the first operation is done. Here is an 
example of a DGMT development syntax: 

.. code-block:: python 

    >>> class DemoClass_: 
           """Class description and documentation """"
           def __init__(self, data, param1=None , param2=None, **kws): 
              self.data=data 
              self.param1=param1
              self.param2=param2
              ...
              for key in list(kws.keys()): 
                  setattr (self, key, kws[key])
              _fit_democlass (self.data ) 
              
           def _fit_democlass(self): 
              """ Documentation of _fit_democlass method """
              ...
              self.param3_= ... 
              ... 
              

Note the underscore "_" at the end of the class. Moreover, in DGMT, the *fit* method must start with an underscore at the 
beginning and lowercase the class. Notice the location of *_fit_democlass* after the attribute initialization and the *for* loop for 
populating extra attributes. 
	
Both DGMT and DSKL use "_"  for instance and class attributes (e.g., ``param3_``) that are not passed as parameters. 
Furthermore, the extra-sensible methods inside the class object must all adopts the *_* at the beginning. 

The reason why we added the DGMT syntax is that `GMT software <https://www.generic-mapping-tools.org/download/>`_ is most known in the geosciences 
community and many developers have started developing following this syntax, so we don't want 
to break this habit and keep it as a renowned syntax in geosciences. And also it helps geoscientist developers to keep their fashion 
practice.
 

Report Bugs  
===============

Bug reports are an important part of making watex more stable. Having a complete bug report
will allow others to reproduce the bug and provide insight into fixing it. See
`this stackoverflow article <https://stackoverflow.com/help/mcve>`_ and
`this blogpost <https://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports>`_
for tips on writing a good bug report.

Trying the bug-producing code out on the *main* branch is often a worthwhile exercise
to confirm the bug still exists. It is also worth searching existing bug reports and pull requests
to see if the issue has already been reported and/or fixed.

Bug reports must:

#. Include a short, self-contained Python snippet reproducing the problem.
   You can format the code nicely by using: 
   
   * `GitHub Flavored Markdown <https://github.github.com/github-flavored-markdown/>`_::
   

      ```python
      >>> from watex.base import Data
      >>> d= Data(...)
      ...
      ```
   * or `reStructured <https://www.writethedocs.org/guide/writing/reStructuredText/>`_ text::
	
	.. code-block::

	       >>> from watex.base import Data
	       >>> d = Data(...)
               ...

#. Include the full version string of watex and its dependencies. You can use the built-in function:

.. code-block:: python

      >>> import watex as wx 
      >>> wx.show_versions() 

#. Give a synopsis of the bug and what you expect instead.


.. _external_resources:

New to Scientific Python
=========================

For those that are new to the scientific Python ecosystem, we highly recommend the following lectures and books: 

* `Python Scientific Lecture Notes <https://scipy-lectures.org>`_. This will help you find your footing a
  bit and will improve your watex experience.  A basic understanding of NumPy arrays is recommended to make the most 
  of watex.
* `Python for Data Analysis <https://www.academia.edu/40873844/Python_for_Data_Analysis_Data_Wrangling_with_Pandas_NumPy_and_IPython_SECOND_EDITION>`_ for Data manipulations. 
* `First step Guide in Data sciences <https://jakevdp.github.io/PythonDataScienceHandbook/>`_. This will guide you as the first step 
  development with Python 
* `Machine learning and Its application of Wlodarczak <https://doi.org/10.1201/9780429448782>`_. This can help go through how importance 
  is Machine Learning nowadays. 

We also highly recommend the following book for the french speaking countries developers: 

* `Apprendre a programmer avec Python <https://www.pierre-giraud.com/python-apprendre-programmer-cours/>`_ de Gérard Swinnen. « Un tres bon 
  livre qui vous permettra de faire vos premiers pas et un progres considerable dans l'apprentissage Python ».
