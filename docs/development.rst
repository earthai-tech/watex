.. _development:

===================
Development Guide 
===================

*Are you a geoscientist or a software developer?*

*Welcome to the WATex Development Guide. Your ideas and contributions are vital 
in building the leading library for addressing geoscientific challenges in 
groundwater and water exploration.*

.. admonition:: Getting Started with Python

    If you're new to Python, we highly recommend visiting 
    :ref:`external resources <external_resources>` to familiarize yourself with 
    Python development basics.

Foreword
========
Before diving in, it's essential to understand that :code:`watex` is not solely 
focused on machine learning. It serves as a bridge between machine learning 
techniques and geoscientific disciplines, particularly in hydrogeology and 
geophysics. Think of it as an applied machine learning toolkit designed for the 
geosciences, primarily hydrogeophysics. Nonetheless, :code:`watex` is versatile. 
It supports the development of algorithms that not only predict but also aid 
academics, professionals, and the broader geoscience community in achieving 
outstanding results in groundwater exploration (GWE).

Several online tutorials cater to specific domains within geosciences and machine learning:

- `Introduction to Machine Learning in Geophysics <https://www.epts.org/courses/standard-courses/geophysics/introduction-to-machine-learning-(ml)-for-geophysics/>`_
- `Applications of Machine Learning and AI in Geophysics <https://seg.org/Events/Applications-of-Machine-Learning-and-AI-in-Geophysics>`_
- `Machine Learning in Seismic Interpretation <https://wiki.seg.org/wiki/Machine_learning_and_seismic_interpretation>`_
- `Machine Learning for NeuroImaging with Python <https://nilearn.github.io/>`_
- `Machine Learning Techniques for Astronomical Data Analysis <https://github.com/astroML/sklearn_tutorial>`_


Core Philosophy
==================

The fundamental philosophy of :code:`watex` draws inspiration from `Scikit-learn <https://scikit-learn.org/stable/index.html>`_
and, to some extent, `GMT <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019GC008515>`_. It champions open collaboration,
inviting contributions from everyone.

Contributors are welcomed regardless of their level of expertise in programming or geosciences, though a foundational knowledge
in either field is beneficial. Engaging with both aspects can significantly enhance the project's development and impact.

Aiming to significantly expand its reach within the geoscience community over the next five years, :code:`watex` maintains flexible
and inclusive guidelines. Contributors are encouraged to propose ideas that align with the Sustainable Development Goal 6 (SDG 6)
on clean water and sanitation (`SDG n6 <https://unric.org/en/sdg-6/>`_). The diversity of skills and perspectives among our contributors
is our strength, driving the project forward.

Ways to Contribute:

- **Improving Documentation**: Typos and enhancements in the documentation are welcome. You can submit changes via email to our mailing list
  or, preferably, through a GitHub pull request. Understanding that the original designers are not native English speakers, we kindly ask
  for patience regarding the language quality and assure regular revisions to improve clarity and correctness.

- **Reporting Issues**: Your feedback is invaluable. Report any issues encountered, and support others' reported issues with a "thumbs up"
  if they affect you too. Promoting :code:`watex` through blogs, articles, or simply starring the GitHub repository shows your support and helps
  spread the word.

.. raw:: html

   <a class="github-button" href="https://github.com/WEgeophysics/watex"
   data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star
   WEgeophysics/watex on GitHub">Star</a>
   <script async defer src="https://buttons.github.io/buttons.js"></script>

- **Adding New Algorithms**: This opportunity is particularly aimed at developers proficient in Python or Cython. This guide will
  delve into the specifics of contributing new algorithms to the :code:`watex` library.


Development
==============

Fork the Repository
------------------------

Contributing to :code:`watex` begins with forking the `main repository <https://github.com/watex/watex/>`__ on GitHub,
followed by submitting a ``pull request`` (PR):

1. `Create an account <https://github.com/join>`_ on GitHub if you haven't already.

2. Fork the `project repository <https://github.com/WEgeophysics/watex>`__: Click on the 'Fork'
   button near the top of the page to create a copy of the code under your GitHub account. For detailed instructions on forking a
   repository, see `this guide <https://help.github.com/articles/fork-a-repo/>`_.

3. Clone your fork of the :code:`watex` repository to your local machine:

   .. prompt:: bash $

      git clone git@github.com:YourLogin/watex.git  # Use --depth 1 if you have a slow connection
      cd watex

4. Install the necessary development :ref:`dependencies <dependencies>`:

   .. prompt:: bash $

        pip install scikit-learn numpy scipy pandas matplotlib tables h5py  seaborn  pyyaml h5py joblib

.. _upstream:

5. Add the ``upstream`` remote to keep your fork synchronized with the main repository. This step ensures you can easily fetch the latest changes:

   .. prompt:: bash $

        git remote add upstream git@github.com:WEgeophysics/watex.git

6. Verify the `upstream` and `origin` remotes are correctly set up by executing `git remote -v`, which should display:

        origin	git@github.com:YourLogin/watex.git (fetch)
        origin	git@github.com:YourLogin/watex.git (push)
        upstream	git@github.com:WEgeophysics/watex.git (fetch)
        upstream	git@github.com:WEgeophysics/watex.git (push)

With these steps, your :code:`watex` installation and Git repository are now correctly set up and ready for development.

Add Algorithms
---------------

When integrating new algorithms into :code:`watex`, two primary development
paths are available:

* Development in accordance with the scikit-learn API
* Development following the principles of `GMT <https://www.generic-mapping-tools.org/>`_

Development Following the Scikit-learn API (DSKL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The DSKL approach emphasizes the use of the *fit()* method for computing and
populating attributes of instantiated models, including plotting modules. This
methodology is applicable for both supervised and unsupervised learning, often
employing *transform()* or *predict()* methods to either transform data or infer
properties. The typical workflow involves:

* Selecting the model class by importing the appropriate estimator or assessor
  from a module. An assessor is designed for a specific task.
* Setting model hyperparameters by instantiating the chosen class with desired
  values.
* Organizing data into a features matrix and target vector as previously discussed.
* Fitting the model to your data by calling the model's fit() method, applicable
  even to plotting modules.
* Applying the trained model to new data; for supervised learning, this usually
  means predicting labels for unknown data, whereas for unsupervised learning,
  it may involve transforming data or inferring properties using the *transform()*
  or *predict()* methods.

This approach is familiar to developers acquainted with `Scikit-learn <https://scikit-learn.org/stable/index.html>`__.

It's important to note that all classes adhering to DSKL must follow Python's
class convention rules outlined in `PEP8 <https://peps.python.org/pep-0008/>`__.
This includes adopting the `fit` method for initial operations such as modular
calculus, validating data structures, and controlling parameters.

Furthermore, class parameters should bear the same name as instance attributes,
and any internal attributes (not explicitly exposed to users) should conclude
with an underscore *_*. Here is an illustrative example:

.. code-block:: python 

    >>> class DemoClass: 
           """ Class documentation. """
           def __init__(self, param1=value1, param2=value2, **kws): 
              self.param1 = param1
              self.param2 = param2 
           def fit(self, data, **fit_params): 
              """ Fit method documentation. """
              X = fit_params.pop('X', None) 
              y = fit_params.pop('y', None)
              ...
              self.param3_ = ... 
              ...
              return self 

The *fit* method is a cornerstone of model development, always returning the
instance *self*. For algorithms not aimed at prediction, :math:`X` and :math:`y`
are included as *fit_params* keywords, along with other parameters, diverging
from `Scikit-learn <https://scikit-learn.org/stable/index.html>`_ where models
primarily focus on prediction. This flexibility supports the library's broader
application in addressing geoscience engineering challenges. It enables the
development and testing of new machine learning (ML) algorithms through real-case
studies to assess their effectiveness. The *fit_params* may encompass various
parameters, allowing the *fit* method to integrate seamlessly across functions.

For geoscience issues extending beyond hydro-geophysics, developers can create
modules within the :mod:`~watex.geology` sub-package, fostering interdisciplinary
solutions. New prediction-focused algorithms should implement **predict**,
**transform**, or **fit_transform** methods, excluding keyword arguments in
their initialization to streamline model training and evaluation:

.. code-block:: python 

    >>> from watex.exlib.sklearn import BaseEstimator, TransformerMixin
    >>> class DemoClass(BaseEstimator, TransformerMixin): 
           """Class documentation here."""
           def __init__(self, param1=value1, param2=value2): 
              self.param1 = param1
              self.param2 = param2 
           def fit(self, X, y=None, **fit_params): 
              """Fit method documentation."""
              self.param3_ = ... 
              ...
              return self 
              
           def predict(self, X): 
              """Predict method documentation."""
              ...
              return Xp

In the example above, :math:`X_p` represents the predicted outcome based on :math:`X`.
This design, devoid of keyword arguments at initialization and inheriting from
:class:`~watex.exlib.BaseEstimator` and :class:`~watex.exlib.TransformerMixin`,
facilitates cross-validation and hyperparameter tuning.

.. note:: 

    For those unfamiliar with scikit-learn, algorithm design remains flexible. Yet,
    documentation should specify the adopted technique or library for hyperparameter
    optimization early on, such as `Keras <https://github.com/keras-team/keras>`__ or
    `TensorFlow <https://github.com/tensorflow/tensorflow>`__. Postponing validation
    aligns with `Scikit-learn <https://scikit-learn.org/stable/index.html>`_'s API to
    avoid redundancy in validation efforts, particularly useful in context with
    :class:`watex.exlib.GridSearchCV`, :mod:`watex.models.GridSearch`, and
    :mod:`watex.models.GridSearchMultiple`, where systematic parameter tuning and
    validation are critical.

Development Following GMT (DGMT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
DGMT development is characterized by flexibility rather than strict conventions.
Distinctively, all GMT classes should conclude with an underscore '_', setting
them apart from the DSKL approach. These classes do not require a *fit* method.
Upon instantiation, all attributes are initialized, and the initial operation
is executed. Here is an illustrative example of DGMT syntax:

.. code-block:: python 

    >>> class DemoClass_: 
           """Class documentation."""
           def __init__(self, data, param1=None, param2=None, **kws): 
              self.data = data
              self.param1 = param1
              self.param2 = param2
              ...
              for key in kws: 
                  setattr(self, key, kws[key])
              self._fit_democlass()

           def _fit_democlass(self): 
              """_fit_democlass method documentation."""
              ...
              self.param3_ = ...
              ...

The underscore "_" suffix in the class name is a hallmark of DGMT. Moreover,
the *fit* method, when present, begins with an underscore and is named in
lowercase, reflecting the class's operational context. The method *_fit_democlass*
is called following attribute initialization and a loop that dynamically assigns
additional attributes via keyword arguments.

Both DGMT and DSKL denote instance and class attributes not passed as parameters
(e.g., ``param3_``) with an underscore. Similarly, methods considered internally
significant to the class's operation also start with an underscore.

The integration of DGMT syntax within :code:`watex` pays homage to the widespread
use and familiarity of `GMT software <https://www.generic-mapping-tools.org/download/>`_
within the geosciences community. This choice respects the established coding
practices of many developers in this field, maintaining a bridge between traditional
geoscience software development and modern coding standards, thereby fostering
a comfortable and recognizable framework for geoscientist developers.

.. _report_bugs:
 
Report Bugs
===============

Reporting bugs is crucial for enhancing the stability of watex. A well-documented
bug report enables others to replicate the issue and contributes to identifying
a solution. For guidance on composing a comprehensive bug report, consult
`this Stack Overflow article <https://stackoverflow.com/help/mcve>`_ and
`this blog post by Matthew Rocklin <https://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports>`_.

Verifying the bug on the *main* branch is a recommended step to ensure the issue
persists. Additionally, review existing bug reports and pull requests to check
if the bug has been previously identified or addressed.

Effective bug reports should:

1. Present a concise, self-contained Python code snippet that demonstrates the issue.
   Employ either:
   
   - `GitHub Flavored Markdown <https://github.github.com/github-flavored-markdown/>`_ for a well-formatted display::
   
      ```python
      >>> from watex.base import Data
      >>> d = Data(...)
      ...
      ```
   
   - Or `reStructuredText <https://www.writethedocs.org/guide/writing/reStructuredText/>`_ for structured documentation::
   
      .. code-block::
   
              >>> from watex.base import Data
              >>> d = Data(...)
              ...

2. Detail the version information of watex and its dependencies, achievable through the library's 
   version display function:

   .. code-block:: python

         >>> import watex as wx
         >>> wx.show_versions()

3. Provide a brief description of the bug and the expected behavior, aiding in a quicker and more 
   accurate resolution.


.. _external_resources:

Embarking on Scientific Python
===============================

If you're venturing into the scientific Python landscape for the
first time, we've curated a list of essential resources to kickstart
your journey. These materials are meticulously selected to enrich
your understanding and utilization of watex.

Key Resources for Beginners:
----------------------------

- **Python Scientific Lecture Notes**:
  (`Scipy Lectures <https://scipy-lectures.org>`_): A cornerstone
  resource offering a foundational grasp of the Python scientific
  stack. A rudimentary knowledge of NumPy arrays is particularly
  advantageous for effective watex usage.

- **Python for Data Analysis**:
  (`View Book <https://www.academia.edu/40873844/Python_for_Data_Analysis_Data_Wrangling_with_Pandas_NumPy_and_IPython_SECOND_EDITION>`_):
  Specializes in data manipulation techniques using Pandas, NumPy, and
  IPython, equipping you with the skills for data-centric projects.

- **Python Data Science Handbook**:
  (`Beginner's Guide <https://jakevdp.github.io/PythonDataScienceHandbook/>`_):
  An all-encompassing handbook ideal for newcomers to data science,
  covering essential tools and methods for analysis and machine learning.

- **Machine Learning and Its Applications**:
  (`Insights by Wlodarczak <https://doi.org/10.1201/9780429448782>`_):
  Delve into the significance and practical applications of Machine
  Learning across various sectors, illustrating its pivotal role today.

For French-speaking Developers:
-------------------------------

- **Apprendre à programmer avec Python**:
  (`Guide by Gérard Swinnen <https://www.pierre-giraud.com/python-apprendre-programmer-cours/>`_):
  An exemplary guide for French speakers, offering a deep dive into
  Python programming and facilitating substantial progress in your
  Python learning journey.

These resources are designed to guide you through the expansive
realm of scientific Python, from introductory programming concepts to
advanced applications in machine learning. Engaging with these texts
will enhance your Python capabilities, thereby augmenting your
contributions to watex and the broader scientific discourse.


