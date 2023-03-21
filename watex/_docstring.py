# -*- coding: utf-8 -*-
#   Licence: BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Created date: Wed Oct  5 22:31:49 2022

import re 

__all__=[
    'DocstringComponents',
    '_baseplot_params', 
    '_seealso_blurbs',
    '_core_returns', 
    'wx_rst_epilog',
    '_core_params',
    'refglossary',
    '_core_docs',
    'ves_doc', 
    'erp_doc',
    ]

ves_doc =type ("ves_doc", (), dict( 
    __doc__="""\
A DC-vertical Electrical resistivity data collected from {survey_name} during
the National Drinking Water Supply Program (PNAEP) occurs in 2014 in 
`Cote d'Ivoire <https://en.wikipedia.org/wiki/Ivory_Coast>`__. An illustration 
of the data arrangement is the following: 

=====   =======     =======     =======     =========
AB/2    MN/2        SE1         SE2         SE...    
=====   =======     =======     =======     =========
1       0.4         107         93          75       
2       0.4         97          91          49       
...     ...         ...         ...         ...      
100     10          79          96          98       
110     10          84          104         104      
=====   =======     =======     =======     ========= 
 
Parameters 
-----------
as_frame : bool, default=False
    If True, the data is a pandas DataFrame including columns with
    appropriate types (numeric). The target is
    a panda DataFrame or Series depending on the number of target columns.
    If `as_frame` is False, then returning a :class:`~watex.utils.Boxspace`
    dictionary-like object, with the following attributes:
    
    - data : {{ndarray, dataframe}} of shape {shape}
        The data matrix. If `as_frame=True`, `data` will be a pandas
        DataFrame.
    - resistivity: {{array-like}} of shape ({shape[0]},)
        The resistivity of the sounding point. 
    - MN: {{array-like}} of shape ({shape[0]},)
        The step value of potential electrodes increasing in meters  
    - AB: {{array-like}} of shape ({shape[0]},)
        The step value of current electrodes increasing in meters  
    - feature_names: list
        The names of the dataset columns.
        .. versionadded:: 0.23
    - DESCR: str
        The full description of the dataset.
    - filename: str
        The path to the location of the data.
        .. versionadded:: 0.20
    .. versionadded:: 0.1.2
    
index_rhoa: int, default=0 
    index of the resistivy columns to retrieve. Note that this is useful in the 
    cases many sounding values are collected in the same survey area. 
    `index_rhoa=0` fetches the first sounding values in the collection of all values. 
    
tag, data_names: None, 
    Always None for API consistency
    
kws: dict, 
    Keywords arguments pass to :func:`~watex.utils.coreutils._is_readable` 
    function for parsing data. 
    
Returns 
--------
data : :class:`~watex.utils.Boxspace`
    Dictionary-like object, with the following attributes.
    - data : {{ndarray, dataframe}} 
        The data matrix. If ``as_frame=True``, `data` will be a pandas DataFrame.

Notes
------
The array configuration is Schlumberger and the max depth investigation is 
{max_depth} meters for :math:`AB/2` (current electrodes). The sounding steps
:math:`AB` starts from {c_start} to {c_stop} meters whereas :math:`MN/2` 
(potential electrodes) starts from {p_start} to {p_stop} meters. 
The total number of sounding performers in {sounding_number} with the prefix '`SE`'.
AB, AB is in meters and SE are in ohm. meters as apparent resistivity values. 
Use the param ``index_rho`` to get the ranking of the sounding resistivity value. 
For instance ``index_rhoa=0`` fetch the first array of resistivity values (SE1).

.. _Cote d'Ivoire: https://en.wikipedia.org/wiki/Ivory_Coast

"""
    )
)
erp_doc = type ('erp_doc', (), dict ( 
    __doc__="""\
A DC-Electrical resistivity profiling data collected from {survey_name} during
the National Drinking Water Supply Program (PNAEP) occurs in 2014 in 
`Cote d'Ivoire <https://en.wikipedia.org/wiki/Ivory_Coast>`__  and an example 
of the data arrangement is the following: 

=====   =========   =========   =======     
pk      east        north       rho         
=====   =========   =========   =======    
0       382741      896203      79        	
10      382743      896193      62
20      382747      896184      51
...     ...         ...         ...         
980     382705	    894887	    55
990     382704	    895879	    58
=====   =========   =========   =======    
 
Parameters 
-----------
as_frame : bool, default=False
    If True, the data is a pandas DataFrame including columns with
    appropriate types (numeric). The target is
    a pandas DataFrame or Series depending on the number of target columns.
    If `as_frame` is False, then returning a :class:`~watex.utils.Boxspace`
    dictionary-like object, with the following attributes:
    - data : {{ndarray, dataframe}} of shape {shape}
        The data matrix. If `as_frame=True`, `data` will be a pandas
        DataFrame.
    - resistivity: {{array-like}} of shape ({shape[0]},)
        The resistivity of the sounding point. 
    - station: {{array-like}}of shape ({shape[0]},)
        The motion distance of each station that increasing in meters.
        can be considered as the station point for data collection.
    - northing: {{array-like}} of shape ({shape[0]},)
        The northing coordinates in UTM in meters at each station where 
        the data is collected. 
    - easting: {{array-like}} of shape ({shape[0]},)
        The easting coordinates in UTM are in meters at each station where the 
        data is collected. 
    - latitude: {{array-like}} of shape ({shape[0]},)
        The latitude coordinates in degree decimals or 'DD:MM.SS' at each 
        station where the data is collected.
    - longitude: {{array-like}} of shape ({shape[0]},)
        The longitude coordinates in degree decimals or 'DD:MM.SS' at each 
        the station where the data is collected.
    - DESCR: str
        The full description of the dataset.
    - filename: str
        The path to the location of the data.
tag, data_names: None, 
    Always None for API consistency 
kws: dict, 
    Keywords arguments pass to :func:`~watex.utils.coreutils._is_readable` 
    function for parsing data. 
    
Returns 
--------
data : :class:`~watex.utils.Boxspace`
    Dictionary-like object, with the following attributes.
    data : {{ndarray, dataframe}} 
        The data matrix. If `as_frame=True`, `data` will be a pandas
        DataFrame.

Notes
------
The array configuration is Schlumberger and the max depth investigation is 
{max_depth} meters for :math:`AB/2` (current electrodes). The  profiling step
:math:`AB` is fixed to {AB_distance}  meters whereas :math:`MN/2`  also fixed to
(potential electrodes) to {MN_distance}meters. The total number of station data 
collected is {profiling_number}.
`station`, `easting`, and `northing` are in meters and `rho` columns are 
in ohm. meters as apparent resistivity values.  
Furthermore, if the UTM coordinate (easting and northing) data is given as well 
as the UTM_zone, the latitude and longitude data are auto-computed and 
vice versa. The user does need to provide both coordinates data types
( UTM or DD:MM.SS)

"""
    )
)
      
refglossary =type ('refglossary', (), dict (
    __doc__="""\
.. _Bagoue region: https://en.wikipedia.org/wiki/Bagou%C3%A9

. _Cote d'Ivoire: https://en.wikipedia.org/wiki/Ivory_Coast

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

.. |ohmS| replace:: Pseudo-area of the fractured zone 
.. |sfi| replace:: Pseudo-fracturing index 
.. |VES| replace:: Vertical Electrical Sounding 
.. |ERP| replace:: Electrical Resistivity Profiling 
.. |MT| replace:: Magnetotelluric 
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
    marker of stations *default* is ``o``.
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
X:  Ndarray of shape ( M x N), :math:`M=m-samples` & :math:`N=n-features`
    training set; Denotes data that is observed at training and prediction time, 
    used as independent variables in learning. The notation is uppercase to denote 
    that it is ordinarily a matrix. When a matrix, each sample may be 
    represented by a feature vector, or a vector of precomputed (dis)similarity 
    with each training sample. :code:`X` may also not be a matrix, and 
    may require a feature extractor or a pairwise metric to turn it into one 
    before learning a model.
    """, 
    y ="""
y: array-like of shape (M, ) `:math:`M=m-samples` 
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
    z="""
z: array-like 1d, pandas.Series 
    Array of depth or a pandas series that contains the depth values. Two  
    dimensional array or more is not allowed. However when `z` is given as 
    a dataframe and `zname` is not supplied, an error raises since `zname` is 
    used to fetch and overwritten `z` from the dataframe. 
    """, 
    zname="""
zname: str, int
    Name of depth columns. `zname` allows to retrieve the depth column in 
    a dataframe. If integer is passed, it assumes the index of the dataframe 
    fits the depth column. Integer value must not be out the dataframe size 
    along axis 1. Commonly `zname`needs to be supplied when a dataframe is 
    passed to a function argument. 
    """, 
    kname="""
kname: str, int
    Name of permeability coefficient columns. `kname` allows to retrieve the 
    permeability coefficient 'k' in  a specific dataframe. If integer is passed, 
    it assumes the index of the dataframe  fits the 'k' columns. Note that 
    integer value must not be out the dataframe size along axis 1. Commonly
   `kname` needs to be supplied when a dataframe is passed as a positional 
    or keyword argument. 
    """, 
    k=""" 
k: array-like 1d, pandas.Series 
    Array of permeability coefficient 'k' or a pandas series that contains the 
    'k' values. Two  dimensional array or more is not allowed. However,
    when `k` passes as a dataframe and `kname` is not supplied, an error 
    raises since `kname` is used to retrieve `k` values from the dataframe 
    and overwritten it.
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
    of a model. The model is estimated as a deterministic function of:
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
    random_state="""
random_state : int, RandomState instance or None, default=None
    Controls the shuffling applied to the data before applying the split.
    Pass an int for reproducible output across multiple function calls..    
    """,
    test_size="""
test_size : float or int, default=None
    If float, should be between 0.0 and 1.0 and represent the proportion
    of the dataset to include in the test split. If int, represents the
    absolute number of test samples. If None, the value is set to the
    complement of the train size. If ``train_size`` is also None, it will
    be set to 0.25.    
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
            interpreted as n_jobs=1 unless the current joblib.Parallel backend 
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
    boxplot="""
boxplot : Draw an enhanced boxplot.
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
 
"""
.. currentmodule:: watex
"""

# Define replacements (used in whatsnew bullets)

wx_rst_epilog ="""

.. role:: raw-html(raw)
   :format: html
   
.. |ohmS| replace:: Pseudo-area of the fractured zone 
.. |sfi| replace:: Pseudo-fracturing index 
.. |VES| replace:: Vertical Electrical Sounding 
.. |ERP| replace:: Electrical Resistivity Profiling 
.. |MT| replace:: Magnetotelluric 
.. |AMT| replace:: Audio-Magnetotellurics 
.. |CSAMT| replace:: Controlled Source |AMT| 
.. |NSAMT| replace:: Natural Source |AMT| 
.. |EM| replace:: electromagnetic
.. |EMAP| replace:: |EM| array profiling
.. |Fix| replace:: :bdg-danger:`Fix`
.. |Enhancement| replace:: :bdg-info:`Enhancement`
.. |Feature| replace:: :bdg-success:`Feature`
.. |Major feature| replace:: :bdg-success:`Major feature`
.. |Major change| replace:: :bdg-primary:`Major change`
.. |API change| replace:: :bdg-dark:`API change`
.. |Deprecated| replace:: :bdg-warning:`Deprecated`

.. |Open Source? Yes!| image:: https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github
   :target: https://github.com/WEgeophysics/watex
   
.. |License BSD| image:: https://img.shields.io/github/license/WEgeophysics/watex?color=b&label=License&logo=github&logoColor=blue
   :alt: GitHub
   :target: https://github.com/WEgeophysics/watex/blob/master/LICENSE
   
.. |simpleicons git| image:: https://img.shields.io/badge/--F05032?logo=git&logoColor=ffffff
   :target: http://git-scm.com 
   
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7744732.svg
   :target: https://doi.org/10.5281/zenodo.7744732
   
"""  

    























































