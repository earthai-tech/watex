# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Created date: Wed Oct  5 22:31:49 2022

import re 

__all__=[
    'DocstringComponents',
    '_baseplot_params', 
    '_seealso_blurbs',
    '_core_returns', 
    '_core_params',
    'refglossary',
    '_core_docs', 
    'xgboostdoc', 
    'sklearndoc', 
    ]

sklearndoc = type ('sklearndoc', () , dict (
    __doc__ ="""\ 
    Machine Learning in Python
    
    Scikit-learn (Sklearn) is the most useful and robust library for machine 
    learning in Python. It provides a selection of efficient tools for machine 
    learning and statistical modeling including classification, regression, 
    clustering and dimensionality reduction via a consistence interface in Python. 
    This library, which is largely written in Python, is built upon NumPy, SciPy 
    and Matplotlib.
    
    It was originally called scikits.learn and was initially developed by David 
    Cournapeau as a Google summer of code project in 2007. Later, in 2010, 
    Fabian Pedregosa, Gael Varoquaux, Alexandre Gramfort, and Vincent Michel,
    from FIRCA (French Institute for Research in Computer Science and Automation), 
    took this project at another level and made the first public release 
    (v0.1 beta) on 1st Feb. 2010. At this time, itts version history is listed above 

        * May 2019: scikit-learn 0.21.0
        * March 2019: scikit-learn 0.20.3
        * December 2018: scikit-learn 0.20.2        
        * November 2018: scikit-learn 0.20.1        
        * September 2018: scikit-learn 0.20.0        
        * July 2018: scikit-learn 0.19.2        
        * July 2017: scikit-learn 0.19.0        
        * September 2016. scikit-learn 0.18.0        
        * November 2015. scikit-learn 0.17.0        
        * March 2015. scikit-learn 0.16.0        
        * July 2014. scikit-learn 0.15.0        
        * August 2013. scikit-learn 0.14
    
    Installation
    -------------
    If you already installed NumPy and Scipy, following are the two easiest 
    ways to install scikit-learn. Following command can be used to install 
    scikit-learn via::
        
        * Using pip
        
        :code:`pip install -U scikit-learn`
        
        * Using conda
        
        :conda:`install scikit-learn`
        
    On the other hand, if NumPy and Scipy is not yet installed on your Python 
    workstation then, you can install them by using either pip or conda.
    
    Another option to use scikit-learn is to use Python distributions like 
    Canopy and Anaconda because they both ship the latest version of scikit-learn.
    
    References 
    ----------
    .. https://scikit-learn.org/stable/index.html
    
    """
    ) 
)
    
xgboostdoc = type ('xgboostdoc', (), dict (
    __doc__= """\
    Extreme Gradient Boosting
    
    XGBoost XgBoost stands for Extreme Gradient Boosting, is an open-source 
    software library that implements optimized distributed gradient boosting 
    machine learning algorithms under the Gradient Boosting framework.
    
    XgBoost, which was proposed by the researchers at the University of 
    Washington. It is a library written in C++ which optimizes the training for 
    Gradient  Boosting. Before understanding the XGBoost, we first need to 
    understand the trees especially the decision tree. 
    
    Indeed , a Decision tree(DT) is a flowchart-like tree structure, where 
    each internal node denotes a test on an attribute, each branch represents 
    an outcome of the test, and each leaf node (terminal node) holds a class 
    label. A tree can be 'learned' by splitting the source set into subsets 
    based on an attribute value test. This process is repeated on each derived 
    subset in a recursive manner called recursive partitioning. The recursion 
    is completed when the subset at a node all has the same value of the target 
    variable, or when splitting no longer adds value to the predictions.
    
    References 
    -----------
    ..[1] https://www.geeksforgeeks.org/xgboost/
    ..[2] https://www.nvidia.com/en-us/glossary/data-science/xgboost/
    
    """
    )
)
    
refglossary =type ('refglossary', (), dict (
    __doc__="""\
    .. _Bagoue region: https://en.wikipedia.org/wiki/Bagou%C3%A9

    .. _Dieng et al: http://documents.irevues.inist.fr/bitstream/handle/2042/36362/2IE_2004_12_21.pdf?sequence=1
    .. _Kouadio et al: https://doi.org/10.1029/2021WR031623
    .. _FlowRatePredictionUsingSVMs: https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021WR031623

    .. _GeekforGeeks: https://www.geeksforgeeks.org/style-plots-using-matplotlib/#:~:text=Matplotlib%20is%20the%20most%20popular,without%20using%20any%20other%20GUIs

    .. _IUPAC nommenclature: https://en.wikipedia.org/wiki/IUPAC_nomenclature_of_inorganic_chemistry

    .. _Matplotlib scatter: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.scatter.html
    .. _Matplotlib plot: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.plot.html
    .. _Matplotlib pyplot: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.plot.html
    .. _Matplotlib figure: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.figure.html
    .. _Matplotlib figsuptitle: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.suptitle.html

    .. _Properties of water: https://en.wikipedia.org/wiki/Properties_of_water#Electrical_conductivity 
    .. _pandas DataFrame: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    .. _pandas Series: https://pandas.pydata.org/docs/reference/api/pandas.Series.html

    .. _scipy.optimize.curve_fit: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

    .. _Water concept: https://en.wikipedia.org/wiki/Water
    .. _Water triple point: https://en.wikipedia.org/wiki/Properties_of_water#/media/File:Phase_diagram_of_water.svg
    .. _WATex: https://github.com/WEgeophysics/watex/

    .. _pycsamt: https://github.com/WEgeophysics/pycsamt
    
    .. |ohmS| replace:: Pseudo-area of fractured zone 
    .. |sfi| replace:: Pseudo-fracturing index 
    .. |VES| replace:: Vertical Electrical Sounding 
    .. |ERP| replace:: Electrical Resistivity Profiling 
    .. |MT| replace:: Magnetetolluric 
    .. |AMT| replace:: Audio-Magnetotellurics 
    .. |CSAMT| replace:: Controlled Source |AMT| 
    .. |NSAMT| replace:: Natural Source |AMT| 
    .. |EM| replace:: electromagnetic
    .. |EMAP| replace:: |EM| array profiling

    """
    ) 
)

class DocstringComponents:
    """ Document the docstring of class, methods or functions. """
    
    regexp = re.compile(r"\n((\n|.)+)\n\s*", re.MULTILINE)

    def __init__(self, comp_dict, strip_whitespace=True):
        """Read entries from a dict, optionally stripping outer whitespace."""
        if strip_whitespace:
            entries = {}
            for key, val in comp_dict.items():
                m = re.match(self.regexp, val)
                if m is None:
                    entries[key] = val
                else:
                    entries[key] = m.group(1)
        else:
            entries = comp_dict.copy()

        self.entries = entries

    def __getattr__(self, attr):
        """Provide dot access to entries for clean raw docstrings."""
        if attr in self.entries:
            return self.entries[attr]
        else:
            try:
                return self.__getattribute__(attr)
            except AttributeError as err:
                # If Python is run with -OO, it will strip docstrings and our lookup
                # from self.entries will fail. We check for __debug__, which is actually
                # set to False by -O (it is True for normal execution).
                # But we only want to see an error when building the docs;
                # not something users should see, so this slight inconsistency is fine.
                if __debug__:
                    raise err
                else:
                    pass

    @classmethod
    def from_nested_components(cls, **kwargs):
        """Add multiple sub-sets of components."""
        return cls(kwargs, strip_whitespace=False)
    
_baseplot_params = dict( 
    savefig= """
savefig: str, Path-like object, 
    savefigure's name, *default* is ``None``
    """,
    fig_dpi="""
fig_dpi: float, 
    dots-per-inch resolution of the figure. *default* is 300   
    """, 
    fig_num="""
fig_num: int, 
    size of figure in inches (width, height). *default* is [5, 5]
    """, 
    fig_size= """
fig_size: Tuple (int, int) or inch 
   size of figure in inches (width, height).*default* is [5, 5]
    """, 
    fig_orientation="""
fig_orientation: str, 
    figure orientation. *default* is ``landscape``
    """, 
    fig_title="""
fig_tile: str, 
    figure title. *default* is ``None``     
    """, 
    fs="""
fs: float, 
     size of font of axis tick labels, axis labels are fs+2. *default* is 6
    """,
    ls="""
ls: str, 
    line style, it can be [ '-' | '.' | ':' ] . *default* is '-'
    """, 
    lc="""
lc: str, Optional, 
    line color of the plot, *default* is ``k``
    """, 
    lw="""
lw: float, Optional, 
    line weight of the plot, *default* is ``1.5``
    """, 
    alpha="""
alpha: float between 0 < alpha < 1, 
    transparency number, *default* is ``0.5``,   
    """, 
    font_weight="""
font_weight: str, Optional
    weight of the font , *default* is ``bold``.
    """, 
    font_style="""
font_style: str, Optional
    style of the font. *default* is ``italic``
    """, 
    font_size="""
font_size: float, Optional
    size of font in inches (width, height). *default* is ``3``.    
    """, 
    ms="""
ms: float, Optional 
    size of marker in points. *default* is ``5``
    """, 
    marker="""
marker: str, Optional
    marker of stations *default* is :math:`\blacktriangledown`.
    """, 
    marker_facecolor="""
marker_style: str, Optional
    facecolor of the marker. *default* is ``yellow``    
    """, 
    marker_edgecolor="""
marker_edgecolor: str, Optional
    facecolor of the marker. *default* is ``yellow``
    """, 
    marker_edgewidth="""
marker_edgewidth: float, Optional
    width of the marker. *default* is ``3``.    
    """, 
    xminorticks="""
xminorticks: float, Optional
     minortick according to x-axis size and *default* is ``1``.
    """, 
    yminorticks="""
yminorticks: float, Optional
    yminorticks according to x-axis size and *default* is ``1``.
    """, 
    bins="""
bins: histograms element separation between two bar. *default* is ``10``. 
    """, 
    xlim="""
xlim: tuple (int, int), Optional
    limit of x-axis in plot. 
    """, 
    ylim="""
ylim: tuple (int, int), Optional
    limit of x-axis in plot. 
    """,
    xlabel="""
xlabel: str, Optional, 
    label name of x-axis in plot.
    """, 
    ylabel="""
ylabel: str, Optional, 
    label name of y-axis in plot.
    """, 
    rotate_xlabel="""
rotate_xlabel: float, Optional
    angle to rotate `xlabel` in plot.  
    """, 
    rotate_ylabel="""
rotate_ylabel: float, Optional
    angle to rotate `ylabel` in plot.  
    """, 
    leg_kws="""
leg_kws: dict, Optional 
    keyword arguments of legend. *default* is empty ``dict``
    """, 
    plt_kws="""
plt_kws: dict, Optional
    keyword arguments of plot. *default* is empty ``dict``
    """, 
    glc="""
glc: str, Optional
    line color of the grid plot, *default* is ``k``
    """, 
    glw="""
glw: float, Optional
   line weight of the grid plot, *default* is ``2``
    """, 
    galpha="""
galpha:float, Optional, 
    transparency number of grid, *default* is ``0.5``  
    """, 
    gaxis="""
gaxis: str ('x', 'y', 'both')
    type of axis to hold the grid, *default* is ``both``
    """, 
    gwhich="""
gwhich: str, Optional
    kind of grid in the plot. *default* is ``major``
    """, 
    tp_axis="""
tp_axis: bool, 
    axis to apply the ticks params. default is ``both``
    """, 
    tp_labelsize="""
tp_labelsize: str, Optional
    labelsize of ticks params. *default* is ``italic``
    """, 
    tp_bottom="""
tp_bottom: bool, 
    position at bottom of ticks params. *default* is ``True``.
    """, 
    tp_labelbottom="""
tp_labelbottom: bool, 
    put label on the bottom of the ticks. *default* is ``False``    
    """, 
    tp_labeltop="""
tp_labeltop: bool, 
    put label on the top of the ticks. *default* is ``True``    
    """, 
    cb_orientation="""
cb_orientation: str , ('vertical', 'horizontal')    
    orientation of the colorbar, *default* is ``vertical``
    """, 
    cb_aspect="""
cb_aspect: float, Optional 
    aspect of the colorbar. *default* is ``20``.
    """, 
    cb_shrink="""
cb_shrink: float, Optional
    shrink size of the colorbar. *default* is ``1.0``
    """, 
    cb_pad="""
cb_pad: float, 
    pad of the colorbar of plot. *default* is ``.05``
    """,
    cb_anchor="""
cb_anchor: tuple (float, float)
    anchor of the colorbar. *default* is ``(0.0, 0.5)``
    """, 
    cb_panchor="""
cb_panchor: tuple (float, float)
    proportionality anchor of the colorbar. *default* is ``(1.0, 0.5)``
    """, 
    cb_label="""
cb_label: str, Optional 
    label of the colorbar.   
    """, 
    cb_spacing="""
cb_spacing: str, Optional
    spacing of the colorbar. *default* is ``uniform``
    """, 
    cb_drawedges="""
cb_drawedges: bool, 
    draw edges inside of the colorbar. *default* is ``False`` 
    """     
)


_core_params= dict ( 
    data ="""
data: str, filepath_or_buffer or :class:`pandas.core.DataFrame`
    Path -like object or Dataframe. If data is given as path-like object,
    data is read, asserted and validated. Any valid string path is acceptable. 
    The string could be a URL. Valid URL schemes include http, ftp, s3, gs, and
    file. For file URLs, a host is expected. A local file could be a
    file://localhost/path/to/table.csv. If you want to pass in a path object, 
    pandas accepts any :code:`os.PathLike`. By file-like object, we refer to 
    objects with a `read()` method, such as a file handle e.g. via builtin 
    `open` function or `StringIO`.
    """, 
    X= """
X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
    training set; Denotes data that is observed at training and prediction time, 
    used as independent variables in learning. The notation is uppercase to denote 
    that it is ordinarily a matrix. When a matrix, each sample may be 
    represented by a feature vector, or a vector of precomputed (dis)similarity 
    with each training sample. :code:`X` may also not be a matrix, and 
    may require a feature extractor or a pairwise metric to turn it into one 
    before learning a model.
    """, 
    y ="""
y: array-like, shape (M, ) ``M=m-samples``, 
    train target; Denotes data that may be observed at training time as the 
    dependent variable in learning, but which is unavailable at prediction time, 
    and is usually the target of prediction. 
    """, 
    Xt= """
Xt: Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
    Shorthand for "test set"; data that is observed at testing and prediction time, 
    used as independent variables in learning.The notation is uppercase to denote 
    that it is ordinarily a matrix.
    """, 
    yt="""
yt: array-like, shape (M, ) ``M=m-samples``,
    test target; Denotes data that may be observed at training time as the 
    dependent variable in learning, but which is unavailable at prediction time, 
    and is usually the target of prediction. 
    """, 
    tname="""
tname: str, 
    A target name or label. In supervised learning the target name is considered  
    as the reference name of `y` or label variable.   
    """, 
    target="""
target: Array-like or :class:`pd.Series`
    Is the dependent variable in supervised (and semisupervised) learning, 
    passed as `y` to an estimator's fit method. Also known as dependent 
    variable, outcome variable, response variable, ground truth or label. 
    `Scikit-learn`_ works with targets that have minimal structure: a class 
    from a finite set, a finite real-valued number, multiple classes, or 
    multiple numbers. Refer to `Scikit-learn`_ `target types`_ . Note that 
    throughout this library, a `target` is considered as a `pd.Series` where 
    the name is `tname` and the variable `y` i.e `target = tname + y`.
    .. _target types: https://scikit-learn.org/stable/glossary.html#glossary-target-types
    
    """,
    model="""
model: callable, always as a function,    
    A model estimator. An object which manages the estimation and decoding 
    of a model. The model is estimated as a deterministic function of::
        * parameters provided in object construction or with set_params;
        * the global numpy.random random state if the estimator’s random_state 
            parameter is set to None; and
        * any data or sample properties passed to the most recent call to fit, 
            fit_transform or fit_predict, or data similarly passed in a sequence 
            of calls to partial_fit.
    The estimated model is stored in public and private attributes on the 
    estimator instance, facilitating decoding through prediction and 
    transformation methods.
    Estimators must provide a fit method, and should provide `set_params` and 
    `get_params`, although these are usually provided by inheritance from 
    `base.BaseEstimator`.
    The core functionality of some estimators may also be available as a ``function``.
    """,
    clf="""
clf :callable, always as a function, classifier estimator
    A supervised (or semi-supervised) predictor with a finite set of discrete 
    possible output values. A classifier supports modeling some of binary, 
    multiclass, multilabel, or multiclass multioutput targets. Within scikit-learn, 
    all classifiers support multi-class classification, defaulting to using a 
    one-vs-rest strategy over the binary classification problem.
    Classifiers must store a classes_ attribute after fitting, and usually 
    inherit from base.ClassifierMixin, which sets their _estimator_type attribute.
    A classifier can be distinguished from other estimators with is_classifier.
    It must implement::
        * fit
        * predict
        * score
    It may also be appropriate to implement decision_function, predict_proba 
    and predict_log_proba.    
    """,
    reg="""
reg: callable, always as a function
    A regression estimator; Estimators must provide a fit method, and should 
    provide `set_params` and 
    `get_params`, although these are usually provided by inheritance from 
    `base.BaseEstimator`. The estimated model is stored in public and private 
    attributes on the estimator instance, facilitating decoding through prediction 
    and transformation methods.
    The core functionality of some estimators may also be available as a
    ``function``.
    """,
    cv="""
cv: float,    
    A cross validation splitting strategy. It used in cross-validation based 
    routines. cv is also available in estimators such as multioutput. 
    ClassifierChain or calibration.CalibratedClassifierCV which use the 
    predictions of one estimator as training data for another, to not overfit 
    the training supervision.
    Possible inputs for cv are usually::
        * An integer, specifying the number of folds in K-fold cross validation. 
            K-fold will be stratified over classes if the estimator is a classifier
            (determined by base.is_classifier) and the targets may represent a 
            binary or multiclass (but not multioutput) classification problem 
            (determined by utils.multiclass.type_of_target).
        * A cross-validation splitter instance. Refer to the User Guide for 
            splitters available within `Scikit-learn`_
        * An iterable yielding train/test splits.
    With some exceptions (especially where not using cross validation at all 
                          is an option), the default is ``4-fold``.
    .. _Scikit-learn: https://scikit-learn.org/stable/glossary.html#glossary
    """,
    scoring="""
scoring: str, 
    Specifies the score function to be maximized (usually by :ref:`cross
    validation <cross_validation>`), or -- in some cases -- multiple score
    functions to be reported. The score function can be a string accepted
    by :func:`sklearn.metrics.get_scorer` or a callable :term:`scorer`, not to 
    be confused with an :term:`evaluation metric`, as the latter have a more
    diverse API.  ``scoring`` may also be set to None, in which case the
    estimator's :term:`score` method is used.  See `slearn.scoring_parameter`
    in the `Scikit-learn`_ User Guide.
    """, 
    n_jobs="""
n_jobs: int, 
    is used to specify how many concurrent processes or threads should be 
    used for routines that are parallelized with joblib. It specifies the maximum 
    number of concurrently running workers. If 1 is given, no joblib parallelism 
    is used at all, which is useful for debugging. If set to -1, all CPUs are 
    used. For instance::
        * `n_jobs` below -1, (n_cpus + 1 + n_jobs) are used. 
        
        * `n_jobs`=-2, all CPUs but one are used. 
        * `n_jobs` is None by default, which means unset; it will generally be 
            interpreted as n_jobs=1, unless the current joblib.Parallel backend 
            context specifies otherwise.

    Note that even if n_jobs=1, low-level parallelism (via Numpy and OpenMP) 
    might be used in some configuration.  
    """,
    verbose="""
verbose: int, `default` is ``0``    
    Control the level of verbosity. Higher value lead to more messages. 
    """  
) 

_core_returns = dict ( 
    self="""
self: `Baseclass` instance 
    returns ``self`` for easy method chaining.
    """, 
    ax="""
:class:`matplotlib.axes.Axes`
    The matplotlib axes containing the plot.
    """,
    facetgrid="""
:class:`FacetGrid`
    An object managing one or more subplots that correspond to conditional data
    subsets with convenient methods for batch-setting of axes attributes.
    """,
    jointgrid="""
:class:`JointGrid`
    An object managing multiple subplots that correspond to joint and marginal axes
    for plotting a bivariate relationship or distribution.
    """,
    pairgrid="""
:class:`PairGrid`
    An object managing multiple subplots that correspond to joint and marginal axes
    for pairwise combinations of multiple variables in a dataset.
    """, 
 )

_seealso_blurbs = dict(
    # Relational plots
    scatterplot="""
scatterplot : Plot data using points.
    """,
    lineplot="""
lineplot : Plot data using lines.
    """,

    # Distribution plots
    displot="""
displot : Figure-level interface to distribution plot functions.
    """,
    histplot="""
histplot : Plot a histogram of binned counts with optional normalization or smoothing.
    """,
    kdeplot="""
kdeplot : Plot univariate or bivariate distributions using kernel density estimation.
    """,
    ecdfplot="""
ecdfplot : Plot empirical cumulative distribution functions.
    """,
    rugplot="""
rugplot : Plot a tick at each observation value along the x and/or y axes.
    """,

    # Categorical plots
    stripplot="""
stripplot : Plot a categorical scatter with jitter.
    """,
    swarmplot="""
swarmplot : Plot a categorical scatter with non-overlapping points.
    """,
    violinplot="""
violinplot : Draw an enhanced boxplot using kernel density estimation.
    """,
    pointplot="""
pointplot : Plot point estimates and CIs using markers and lines.
    """,

    # Multiples
    jointplot="""
jointplot : Draw a bivariate plot with univariate marginal distributions.
    """,
    pairplot="""
jointplot : Draw multiple bivariate plots with univariate marginal distributions.
    """,
    jointgrid="""
JointGrid : Set up a figure with joint and marginal views on bivariate data.
    """,
    pairgrid="""
PairGrid : Set up a figure with joint and marginal views on multiple variables.
    """,
)
                 
_core_docs = dict(
    params=DocstringComponents(_core_params),
    returns=DocstringComponents(_core_returns),
    seealso=DocstringComponents(_seealso_blurbs),
)
  

    























































