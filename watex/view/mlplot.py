# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Learning Plots
===============

Is a set of plot templates  for visualising the ML models.  It gives a 
quick alternative for users to save their time for writting their own plot 
scripts. However to have full control of the plot, it is recommended to write 
your own plot scripts. 
Note that this module can not handle all the plots that can offer the
software. 

"""
from __future__ import annotations 
import re
import warnings
import inspect 
from abc import ABCMeta 

import numpy as np 
import pandas as pd
from scipy.cluster.hierarchy import (
    dendrogram, 
    # set_link_color_palette 
    )
import matplotlib as mpl 
import  matplotlib.pyplot  as plt
from matplotlib import cm 

from .._watexlog import watexlog
from ..analysis.dimensionality import Reducers
from ..exlib.sklearn import ( 
    learning_curve , 
    silhouette_samples, 
    )

from ..exceptions import PlotError, ArgumentError 
from ..decorators import docSanitizer , docAppender
from ..property import BasePlot 
from ..metrics import (
    precision_recall_tradeoff, 
    ROC_curve, 
    confusion_matrix_
    )
from ..utils.exmath import linkage_matrix 
from ..utils.hydroutils import categorize_flow 
from ..utils.plotutils import ( 
    D_COLORS, 
    D_MARKERS, 
    D_STYLES,
    savefigure
   ) 
from ..typing import ( 
    Generic,
    Optional, 
    Tuple, 
    V, 
    List,
    ArrayLike, 
    NDArray,
    DType, 
    DataFrame, 
    )


_logger=watexlog.get_watex_logger(__name__)

def plotBindDendro2Heatmap (
    df: DataFrame |NDArray, 
    columns: List[str] =None, 
    labels:Optional[List[str]] =None,
    metric:str ='euclidean',  
    method:str ='complete', 
    kind:str = 'design', 
    cmap:str ='hot_r', 
    fig_size:Tuple[int] =(8, 8), 
    facecolor:str ='white', 
    **kwd
):
    """
    Attached dendrogram to a heat map 
    
    Hiearchical dendrogram are often used in combination with a heat map wich 
    qllows us to represent the individual value in data array or matrix 
    containing our training examples with a color code. 
    
    Parameters 
    ------------
    df: dataframe or NDArray of (n_samples, n_features) 
        dataframe of Ndarray. If array is given , must specify the column names
        to much the array shape 1 
    columns: list 
        list of labels to name each columns of arrays of (n_samples, n_features) 
        If dataframe is given, don't need to specify the columns. 
        
    kind: str, ['squareform'|'condense'|'design'], default is {'design'}
        kind of approach to summing up the linkage matrix. 
        Indeed, a condensed distance matrix is a flat array containing the 
        upper triangular of the distance matrix. This is the form that ``pdist`` 
        returns. Alternatively, a collection of :math:`m` observation vectors 
        in :math:`n` dimensions may be passed as  an :math:`m` by :math:`n` 
        array. All elements of the condensed distance matrix must be finite, 
        i.e., no NaNs or infs.
        Alternatively, we could used the ``squareform`` distance matrix to yield
        different distance values than expected. 
        the ``design`` approach uses the complete inpout example matrix  also 
        called 'design matrix' to lead correct linkage matrix similar to 
        `squareform` and `condense``. 
        
    metric : str or callable, default is {'euclidean'}
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`.
        If ``X`` is the distance array itself, use "precomputed" as the metric.
        Precomputed distance matrices must have 0 along the diagonal.
        
    method : str, optional, default is {'complete'}
        The linkage algorithm to use. See the ``Linkage Methods`` section below
        for full descriptions in :func:`watex.utils.exmath.linkage_matrix`
        
    labels : ndarray, optional
        By default, ``labels`` is None so the index of the original observation
        is used to label the leaf nodes.  Otherwise, this is an :math:`n`-sized
        sequence, with ``n == Z.shape[0] + 1``. The ``labels[i]`` value is the
        text to put under the :math:`i` th leaf node only if it corresponds to
        an original observation and not a non-singleton cluster.
        
    cmap: str , default is {'hot_r'}
        matplotlib color map 
     
    fig_size: str , Tuple , default is {(8, 8)}
        the size of the figure 
    facecolor: str , default is {"white"}
        Matplotlib facecolor 
  
    kwd: dict 
        additional keywords arguments passes to 
        :func:`scipy.cluster.hierarchy.dendrogram` 

    """
    if columns is not None: 
        if isinstance (columns , str):
            columns = [columns]
        if len(columns)!= df.shape [1]: 
            raise TypeError("X and columns must be consistent,"
                            f" got {len(columns)} instead of {df.shape [1]}"
                            )
        df = pd.DataFrame(data = df, columns = columns )
        
    # create a new figure object  and define x axis position 
    # and y poaition , width, heigh of the dendrogram via the  
    # add_axes attributes. Furthermore, we rotate the dengrogram
    # to 90 degree counter-clockwise. 
    fig = plt.figure (figsize = fig_size , facecolor = facecolor )
    axd = fig.add_axes ([.09, .1, .2, .6 ])
    
    row_cluster = linkage_matrix(df = df, metric= metric, 
                                 method =method , kind = kind ,  
                                 )
    orient ='left' # use orientation 'right for matplotlib version < v1.5.1
    mpl_version = mpl.__version__.split('.')
    if mpl_version [0] =='1' : 
        if mpl_version [1] =='5' : 
            if float(mpl_version[2]) < 1. :
                orient = 'right'
                
    r = dendrogram(row_cluster , orientation= orient,  
                   **kwd )
    # 2. reorder the data in our initial dataframe according 
    # to the clustering label that can be accessed by a dendrogram 
    # which is essentially a Python dictionnary via a key leaves 
    df_rowclust = df.iloc [r['leaves'][::-1]]
    
    # 3. construct the heatmap from the reordered dataframe and position 
    # in the next ro the dendrogram 
    axm = fig.add_axes ([.23, .1, .6, .6]) 
    cax = axm.matshow (df_rowclust , 
                       interpolation = 'nearest' , 
                       cmap=cmap, 
                       )
    #4.  modify the asteric  of the dendogram  by removing the axis 
    # ticks and hiding the axis spines. Also we add a color bar and 
    # assign the feature and data record names to names x and y axis  
    # tick lables, respectively 
    
    axd.set_xticks ([]) # set ticks invisible 
    axd.set_yticks ([])
    for i in axd.spines.values () : 
        i.set_visible (False) 
    fig.colorbar(cax )
    axm.set_xticklabels ([''] + list (df_rowclust.columns))
    axm.set_yticklabels ([''] + list(df_rowclust .index ))
    
    plt.show () 
    
    
def plotDendrogram (df, columns =None, labels =None,metric ='euclidean',  
                   method ='complete', kind = 'design',
                   return_r =False , 
                   **kwd ): 
    """ Visualize the linkage matrix in the results of dendogram 
    
    
    Parameters 
    -----------
    df: dataframe or NDArray of (n_samples, n_features) 
        dataframe of Ndarray. If array is given , must specify the column names
        to much the array shape 1 
    columns: list 
        list of labels to name each columns of arrays of (n_samples, n_features) 
        If dataframe is given, don't need to specify the columns. 
        
    kind: str, ['squareform'|'condense'|'design'], default is {'design'}
        kind of approach to summing up the linkage matrix. 
        Indeed, a condensed distance matrix is a flat array containing the 
        upper triangular of the distance matrix. This is the form that ``pdist`` 
        returns. Alternatively, a collection of :math:`m` observation vectors 
        in :math:`n` dimensions may be passed as  an :math:`m` by :math:`n` 
        array. All elements of the condensed distance matrix must be finite, 
        i.e., no NaNs or infs.
        Alternatively, we could used the ``squareform`` distance matrix to yield
        different distance values than expected. 
        the ``design`` approach uses the complete inpout example matrix  also 
        called 'design matrix' to lead correct linkage matrix similar to 
        `squareform` and `condense``. 
        
    metric : str or callable, default is {'euclidean'}
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`.
        If ``X`` is the distance array itself, use "precomputed" as the metric.
        Precomputed distance matrices must have 0 along the diagonal.
        
    method : str, optional, default is {'complete'}
        The linkage algorithm to use. See the ``Linkage Methods`` section below
        for full descriptions in :func:`watex.utils.exmath.linkage_matrix`
        
    labels : ndarray, optional
        By default, ``labels`` is None so the index of the original observation
        is used to label the leaf nodes.  Otherwise, this is an :math:`n`-sized
        sequence, with ``n == Z.shape[0] + 1``. The ``labels[i]`` value is the
        text to put under the :math:`i` th leaf node only if it corresponds to
        an original observation and not a non-singleton cluster.
        
    return_r: bool, default='False', 
        return r-dictionnary if set to 'True' otherwise return nothing 
    
    kwd: dict 
        additional keywords arguments passes to 
        :func:`scipy.cluster.hierarchy.dendrogram` 
        
    Returns
    -------
    r : dict
        A dictionary of data structures computed to render the
        dendrogram. Its has the following keys:

        ``'color_list'``
          A list of color names. The k'th element represents the color of the
          k'th link.

        ``'icoord'`` and ``'dcoord'``
          Each of them is a list of lists. Let ``icoord = [I1, I2, ..., Ip]``
          where ``Ik = [xk1, xk2, xk3, xk4]`` and ``dcoord = [D1, D2, ..., Dp]``
          where ``Dk = [yk1, yk2, yk3, yk4]``, then the k'th link painted is
          ``(xk1, yk1)`` - ``(xk2, yk2)`` - ``(xk3, yk3)`` - ``(xk4, yk4)``.

        ``'ivl'``
          A list of labels corresponding to the leaf nodes.

        ``'leaves'``
          For each i, ``H[i] == j``, cluster node ``j`` appears in position
          ``i`` in the left-to-right traversal of the leaves, where
          :math:`j < 2n-1` and :math:`i < n`. If ``j`` is less than ``n``, the
          ``i``-th leaf node corresponds to an original observation.
          Otherwise, it corresponds to a non-singleton cluster.

        ``'leaves_color_list'``
          A list of color names. The k'th element represents the color of the
          k'th leaf.
          
    Examples 
    ----------
    >>> from watex.datasets import load_iris 
    >>> from watex.view import plotDendrogram
    >>> data = load_iris () 
    >>> X =data.data[:, :2] 
    >>> plotDendrogram (X, columns =['X1', 'X2' ] ) 

    """
    
    row_cluster = linkage_matrix(df = df, columns = columns, metric= metric, 
                                 method =method , kind = kind ,
                                 )
    #make dendogram black (1/2)
    # set_link_color_palette(['black']) 
    r= dendrogram(row_cluster, labels= labels  , 
                           # make dendogram colors (2/2)
                           # color_threshold= np.inf,  
                           **kwd)
    plt.tight_layout()
    plt.ylabel ('Euclidian distance')
    plt.show ()
    
    return r if return_r else None 

    
def plotSilhouette (X, y_pred, metric ='eucledian', **kwds ):
    r"""Plot quantifying the quality  of clustering silhouette 
    
    Parameters 
    ---------
    X : array-like of shape (n_samples_a, n_samples_a) if metric == \
            "precomputed" or (n_samples_a, n_features) otherwise
        An array of pairwise distances between samples, or a feature array.

    y_pred : array-like of shape (n_samples,)
        Label values for each sample.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`.
        If ``X`` is the distance array itself, use "precomputed" as the metric.
        Precomputed distance matrices must have 0 along the diagonal.

    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a ``scipy.spatial.distance`` metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
        
    Examples
    ---------
    >>> import numpy as np 
    >>> from sklearn.cluster import KMeans 
    >>> from watex.datasets import load_iris 
    >>> from watex.view.mlplot import plotSilhouette
    >>> d= load_iris ()
    >>> X= d.data [:, 0][:, np.newaxis] # take the first axis 
    >>> km= KMeans (n_clusters =3 , init='k-means ++', n_init =10 , 
                    max_iter = 300 , 
                    tol=1e-4, 
                    random_state =0 
                    )
    >>> y_km = km.fit_predict(X) 
    >>> plotSilhouette (X, y_km)
  
    See also 
    ---------
    Silhouette is used as graphical tools,  to plot a measure how tighly is  
    grouped the examples of the clusters are.  To calculate the silhouette 
    coefficient, three steps is allows: 
        - calculate the **cluster cohesion**, :math:`a(i)`, as the average 
            distance between examples, :math:`x^{(i)}`, and all the others 
            points
        - calculate the **cluster separation**, :math:`b^{(i)}` from the next 
            average distance between the example , :math:`x^{(i)}` amd all 
            the example of nearest cluster 
        - calculate the silhouette, :math:`s^{(i)}`, as the difference between 
            the cluster cohesion and separation divided by the greater of the 
            two, as shown here: 
                
            .. math:: 
                
                s^{(i)}=\frac{b^{(i)} - a^{(i)}}{max {{b^{(i)},a^{(i)} }}}
    
        Note that the sihouette coefficient is bound between -1 and 1 
    
    """
    cluster_labels = np.unique (y_pred) 
    n_clusters = cluster_labels.shape [0] 
    silhouette_vals = silhouette_samples(X, y = y_pred, metric = metric ,
                                         **kwds)
    y_ax_lower , y_ax_upper = 0, 0 
    yticks =[]
    
    for i, c  in enumerate (cluster_labels ) : 
        c_silhouette_vals = silhouette_vals[y_pred ==c ] 
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color =cm.jet (float(i)/n_clusters )
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, 
                 height =1.0 , 
                 edgecolor ='none', 
                 color =color, 
                 )
        yticks.append((y_ax_lower + y_ax_upper)/2.)
        y_ax_lower += len(c_silhouette_vals)
    silhouette_avg = np.mean(silhouette_vals) 
    plt.axvline (silhouette_avg, 
                 color='red', 
                 linestyle ='--'
                 )
    plt.yticks(yticks, cluster_labels +1 ) 
    plt.ylabel ("Cluster") 
    plt.xlabel ("Silhouette coefficient")
    plt.tight_layout()

    plt.show() 
    
def plotLearningCurves(
    model,  X,  y, axes=None, ylim=None, cv=5, n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5), display_legend = True, title=None,
):
    """Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    model : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
        
    display_legend: bool, default ='True' 
        display the legend
        
    Returns
    ----------
    axes: Matplotlib axes 
    
    """ 
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        model,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].hlines(
        np.mean([train_scores[-1], test_scores[-1]]), 
        train_sizes[0],
        train_sizes[-1], 
        color="gray", 
        linestyle ="--", 
        label="Convergence score"
        )
    axes[0].plot(
        train_sizes, 
        train_scores_mean, 
        "o-", 
        color="r", 
        label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, 
        "o-", 
        color="g", 
        label="Cross-validation score"
    )

    if display_legend:
        axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, 
                 test_scores_mean_sorted, "o-"
                 )
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return axes


def biPlot(
        self, 
        score: NDArray[DType [float]],
        coeff: ArrayLike[DType[float]],
        y: ArrayLike,
        y_classes: List | ArrayLike [str] =None,
        markers: str | List [str]  =None, 
        colors: str | List [str ] =None, 
        savefig : Optional[str] = None, 
        dpi: Optional[float] = None,
)-> None :
    """
    The biplot is the best way to visualize all-in-one following a PCA analysis.
    There is an implementation in R but there is no standard implementation
    in python. 
    
    Originally written by 
        
        Author: Serafeim Loukas, serafeim.loukas@epfl.ch
    
    and referenced to :href:`<https://towardsdatascience.com/...-python-7c274582c37e>`_
    Func is edited and add some new parameters to customize plots: 
        
    :param score: the projected data.
    
    :param coeff: the eigenvectors (PCs).
    
    :param y: the class labels.
    
    :param y_classes: class categories.
    
    :param markers: markers to plot classes.
    
    :param colors: colors to customize plots 
    
   """
   
    xs = score[:,0] # projection on PC1
    ys = score[:,1] # projection on PC2
    n = coeff.shape[0] # number of variables
    plt.figure(figsize=self.fig_size, #(10,8),
               dpi=self.fig_dpi #100
               )
    if y_classes is None: 
        y_classes = np.unique(y)
    if colors is None:
        colors = D_COLORS
        colors = [colors[c] for c in range(len(y_classes))]
    if markers is None:
        markers=D_MARKERS 
        markers = [markers[m] for m in range(len(y_classes))]
    for s,l in enumerate(y_classes):
        plt.scatter(xs[y==l],ys[y==l], 
                    c = colors[s], 
                    marker=markers[s]) # color based on group
    for i in range(n):
        #plot as arrows the variable scores 
        # (each variable has a score for PC1 and one for PC2)
        plt.arrow(0, 0, coeff[i,0], coeff[i,1], 
                  color = self.lc, #'k', 
                  alpha = self.alpha, #0.9,
                  linestyle = self.ls, # '-',
                  linewidth = self.lw, #1.5,
                  overhang=0.2)
        plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, 
                 "Var"+str(i+1),
                 color = 'k', 
                 ha = 'center',
                 va = 'center',
                 fontsize= self.ms * self.fs *.5 
                 )

    plt.xlabel("PC{}".format(1),
               size=self.ms* self.fs)
    plt.ylabel("PC{}".format(2),
               size=self.ms* self.fs)
    limx= int(xs.max()) + 1
    limy= int(ys.max()) + 1
    plt.xlim([-limx,limx])
    plt.ylim([-limy,limy])
    plt.grid()
    plt.tick_params(axis='both',
                    which='both', 
                    labelsize=self.ms* self.fs)
    
    if savefig is not None: 
        savefigure (plt, savefig, dpi = dpi )
    


class BaseMetricPlot(BasePlot): 
    """ Plot Supervised  Machine learning metrics
    
    Inherited from :class:`BasePlot`. Dimensional reduction plots and is 
    composed of decomposition tips, confusion matrix plots and else. 
 
    Hold others optional informations: 
        
    ==================  =======================================================
    Key Words           Description        
    ==================  =======================================================
    fig_dpi             dots-per-inch resolution of the figure
                        *default* is 300
    fig_num             number of the figure instance
                        *default* is 'Mesh'
    fig_size            size of figure in inches (width, height)
                        *default* is [5, 5]
    savefig             savefigure's name, *default* is ``None``
    fig_orientation     figure orientation. *default* is ``landscape``
    fig_title           figure title. *default* is ``None``
    fs                  size of font of axis tick labels, axis labels are
                        fs+2. *default* is 6 
    ls                  [ '-' | '.' | ':' ] line style of mesh lines
                        *default* is '-'
    lc                  line color of the plot, *default* is ``k``
    lw                  line weight of the plot, *default* is ``1.5``
    alpha               transparency number, *default* is ``0.5``  
    font_weight         weight of the font , *default* is ``bold``.        
    marker              marker of stations 
                        *default* is r"$\blacktriangledown$".
    ms                  size of marker in points. *default* is 5
    marker_style        style  of marker in points. *default* is ``o``.
    markerfacecolor     facecolor of the marker. *default* is ``yellow``
    markeredgecolor     edgecolor of the marker. *default* is ``cyan``.
    markeredgewidth     width of the marker. *default* is ``3``.
    x_minorticks        minortick according to x-axis size and *default* is 1.
    y_minorticks        minortick according to y-axis size and *default* is 1.
    font_size           size of font in inches (width, height)
                        *default* is 3.
    font_style          style of font. *default* is ``italic``
    bins                histograms element separation between two bar. 
                         *default* is ``10``. 
    xlim                limit of x-axis in plot. *default* is None 
    ylim                limit of y-axis in plot. *default* is None 
    xlabel              label name of x-axis in plot. *default* is None 
    ylabel              label name  of y-axis in plot. *default* is None 
    rotate_xlabel       angle to rotate `xlabel` in plot. *default* is None 
    rotate_ylabel       angle to rotate `ylabel` in plot. *default* is None 
    leg_kws             keyword arguments of legend. *default* is empty dict.
    plt_kws             keyword arguments of plot. *default* is empty dict
    rs                  [ '-' | '.' | ':' ] line style of `Recall` metric
                        *default* is '--'
    ps                  [ '-' | '.' | ':' ] line style of `Precision `metric
                        *default* is '-'
    rc                  line color of `Recall` metric *default* is ``(.6,.6,.6)``
    pc                  line color of `Precision` metric *default* is ``k``
    s                   size of items in scattering plots. default is ``fs*40.``
    gls                 [ '-' | '.' | ':' ] line style of grid  
                        *default* is '--'.
    glc                 line color of the grid plot, *default* is ``k``
    glw                 line weight of the grid plot, *default* is ``2``
    galpha              transparency number of grid, *default* is ``0.5``  
    gaxis               axis to plot grid.*default* is ``'both'``
    gwhich              type of grid to plot. *default* is ``major``
    tp_axis             axis  to apply ticks params. default is ``both``
    tp_labelsize        labelsize of ticks params. *default* is ``italic``
    tp_bottom           position at bottom of ticks params. *default*
                        is ``True``.
    tp_top              position at the top  of ticks params. *default*
                        is ``True``.
    tp_labelbottom      see label on the bottom of the ticks. *default* 
                        is ``False``
    tp_labeltop         see the label on the top of ticks. *default* is ``True``
    cb_orientation      orientation of the colorbar. *default* is ``vertical``
    cb_aspect           aspect of the colorbar. *default* is 20.
    cb_shrink           shrink size of the colorbar. *default* is ``1.0``
    cb_pad              pad of the colorbar of plot. *default* is ``.05``
    cb_anchor           anchor of the colorbar. *default* is ``(0.0, 0.5)``
    cb_panchor          proportionality anchor of the colorbar. *default* is 
                        `` (1.0, 0.5)``.
    cb_label            label of the colorbar. *default* is ``None``.      
    cb_spacing          spacing of the colorbar. *default* is ``uniform``
    cb_drawedges        draw edges inside of the colorbar. *default* is ``False``
    cb_format           format of the colorbar values. *default* is ``None``.
    yp_ls               [ '-' | '.' | ':' ] line style of `Predicted` label.  
                        *default* is '-'
    yp_lc               line color of the `Prediction` plot. *default* is ``k``
    yp_lw               line weight of the `Predicted` plot. *default* is ``3``
    yp_marker_style     style  of marker in  of `Prediction` points. 
                            *default* is ``o``.
    yp_markerfacecolor  facecolor of the `Predicted` label marker. 
                        *default* is ``k``
    yp_markeredgecolor  edgecolor of the `Predicted` label marker. 
                        *default* is ``r``.
    yp_markeredgewidth  width of the `Predicted`label marker. *default* is ``2``.  
    ==================  =======================================================
    """
    def __init__(self,  **kws ): 
        
        self._logging= watexlog().get_watex_logger(self.__class__.__name__)
        
        # precision(p) and recall(r) style(s) and color (c)
        self.rs =kws.pop('rs', '--')
        self.ps =kws.pop('ps', '-')
        self.rc =kws.pop('rc', (.6, .6, .6))
        self.pc =kws.pop('pc', 'k')
        
        # predicted properties 
        self.yp_lc =kws.pop('yp_lc', 'k') 
        self.yp_marker_style= kws.pop('yp_marker_style', 'o')
        self.yp_marker_edgecolor = kws.pop('yp_markeredgecolor', 'r')
        self.yp_lw = kws.pop('yp_lw', 3.)
        self.yp_ls=kws.pop('yp_ls', '-')
        self.yp_marker_facecolor =kws.pop('yp_markerfacecolor', 'k')
        self.yp_marker_edgewidth= kws.pop('yp_markeredgewidth', 2.)
        
        super().__init__(**kws) 
    
    def PCA_(self,
             X:ArrayLike,
             y:ArrayLike,
             n_components:int =None, 
             n_axes: int=2,
             y_type :str =None, 
             y_values:V=None,
             y_classes:V=None, 
             replace_y =False, 
             biplot:bool =False, 
             pc1_label:str ='Axis 1',
             pc2_label:str='Axis 2',
             y_label:str='Flow', 
             plot_dict:Generic[V] = None,
             **pca_kws):
            
        """ Plot PCA component analysis using :class:`~.sklearn.decomposition`. 
        
        PCA indentifies the axis that accounts for the largest amount of 
        variance in the trainset `X`. It also finds a second axis orthogonal 
        to the first one, that accounts for the largest amount of remaining 
        variance.
        
        :param X: Dataset compose of n_features items for dimension reducing
        
        :param y: label for prediction. The target or the predicting label  
            in supervised learning.
        
        :param n_components: Number of dimension to preserve. If`n_components` 
                is ranged between float 0. to 1., it indicates the number of 
                variance ratio to preserve. If ``None`` as default value 
                the number of variance to preserve is ``95%``.
                
        :param n_axes: Number of importance components to retrieve the 
            variance ratio. Default is ``2``. The first two importance 
            components with most variance ratio. 
            
        :param y_type: type of features `y`. Could be ``cat`` for categorial 
                features or ``num`` for numerical features. If `y` is numerical 
                features and need to be converted into a categorial features 
                set `y_type` to ``num`` to force the conversion of `y` into 
                a categorial features.
        
        :param replace_y: customize the encoded values by providing a new list
                of categorized values.
                
        :param y_values: Once `replace_y` is set to True, then `y_values` must 
                be given to convert the numerical values into a categorial 
                values contained in the list of `y_values`. Notes: values in 
                `y_values` must be self containing in `y`(numerical data.) 
                
        :param y_classes: Can replace the numerical  values encoded thoughout 
                `y_values` to text labels which match each encoded values 
                in `y_values`. For instance::
                    
                    y_values =[0, 1, 3]
                    y_classes = ['FR0', 'FR1', 'FR2', 'FR3']
                
                where :
                    - ``FR0`` equal to values =0 
                    - ``FR1`` equal values between  0-1(0< value<=1)
                    - ``FR2`` equal values between  1-1 (1< value<=3)
                    - ``FR3`` greather than 3 (>3)
                    
                Please refer to :doc:`watex.utils.decorator.catmapflow` and 
                :doc:`watex.analysis.features.categorize_flow` for futher 
                details.
                
        :param plot_dict: size and colors properties of target.
        
        :param pca_kws: additional :class:`~sklearn.decomposition.PCA`
                keywords arguments.
        
        :param biplot: biplot pca features importance (pc1 and pc2) 
                and visualize different variables according 
                to Serafeim Loukas, serafeim.loukas@epfl.ch 
        
        Usage:
            
            by default, :meth:`~watex.viewer.mlplot.MLPlot.PCA_` plot the first 
            two principal components named `pc1_label` for axis 1 and
            `pc2_label` for axis 2. if you want to plot the first component 
            `pc1` vs the third components`pc2` set the `pc2_label` to 
            `Axis 3`. Algoith should automatically detect the digit ``3``
            of Axis`3` and will consider as as `pc3 `. The same process is 
            available for other axis. 
            
        :Example: 
            
            >>> from watex.datasets.data_preparing import X_train_2
            >>> from watex.datasets import y_prepared   
            >>> pcaObj= MLPlots().PCA_(X= X_train_2, y=y_prepared, replace_y=True, 
            ...                        y_classes =['FR0', 'FR1', 'FR2', 'FR3'],
            ...                        biplot =False)
        """
        if plot_dict is None: 
            plot_dict ={'y_colors':D_COLORS, 
                        's':100.}
            
        def mere_replace(_y, y_val, y_clas): 
            """ Replace the numerical values (generaly encoded values) to 
            desired categorial features names."""
            ynew = np.zeros_like(_y, dtype ='>U12')
            for i, val in enumerate(_y):
                for name, r_val in zip(y_clas, y_val): 
                    if float(val) ==float(r_val): 
                        ynew[i] = name
                        break 
            return ynew 
        
        if (y_values and y_classes) is not None :
            replace_y = True 

        if replace_y : 
            
            if y_classes is None :
                warnings.warn('NoneType `y` can not be categorized'
                              'Provide the value of `y_classes`')
                raise TypeError('NoneType <`y`> can not be categorized. '
                                'Need at least `y_classes` to categorize `y`.')
                
            if y_values is None:
                
                y_values, *_ = np.unique(y, return_counts =True)

                warnings.warn('None  `y_values` are given. `y_values` should'
                              ' take the number of occurence in the target`y` '
                              f'as default value. y_values`={y_values} '
                              f' `and fit to {y_classes}',UserWarning)
                              
                self._logging.warning('`y_values` are not given. New values '
                                        'take the number of occurence in the'
                                        f' target`y`. New `y_values`={y_values}'
                                        f'and fit to {y_classes}.')
  
                y_values = sorted(y_values)

            if len(y_values) != len(y_classes): 
                warnings.warn(
                    f'`y_classes` and {len(y_values)!r} must have the length but '
                    f'{len(y_classes)!r} {"is" if len(y_classes)<2 else"are"}'
                     ' given.', UserWarning)
                self._logging.error('Argument `y_classes` must have the same '
                                    f'length with number of occurence {y_values!r}'
                                    f'= {len(y_values)!r}.')

                raise TypeError('Expected {0} classes but {1} {2} given'.format(
                                len(y_values), len(y_classes), 
                                f'{"is" if len(y_classes)<2 else"are"}'))
                
            y = mere_replace(_y=y, y_val=y_values, y_clas=y_classes)                    
            
        if y_type =='num': 
            if y_values is None: 
                warnings.warn('None values are found to categorize numerical'
                              'values. Please provided the `y_values` argument')
                raise TypeError('No values are found to categorize the target .'
                                '`y`')
            if y_classes is None: 
                warnings.warn('No categorial classes detected. Default '
                              'categorial classes should used.', UserWarning)
                self._logging.info('No categorial classes detected. Default '
                              'categorial classes should used.')
                
            y =  categorize_flow(y, y_values,
                                 classes=y_classes)
            
        # go for PCA analysis 
        if n_axes is None:
            n_axes = 2
            warnings.warn('None number of axes is specified. The PCA first '
                         'two components are retrieved by default;'
                         ' n_axes is = {n_axes!r}.')
            
        pca= Reducers().PCA(X,
                            n_components, 
                            n_axes =n_axes,
                            **pca_kws)
        feature_importances_ = pca.feature_importances_
        X_reduced = pca.X_ # the components
        # Get axis for plots from pca_labels
        try: 
            lbls =[int(re.findall("\d+", str_axes)[0]) 
                   for str_axes in [pc1_label, pc2_label]]
        except : 
            # remove if dot '.'exists by replacing by
            lbls =[s.replace('.','s') for s in [pc1_label, pc2_label]]
            lbls=[int ( ''.join(filter(str.isdigit, js) ) ) for js in lbls]
        else:
            pca1_ix, pca2_ix = [i-1 for i in lbls]
            if pca1_ix <0 or pca2_ix<0: 
                pca1_ix =0
                pca2_ix = pca1_ix+1
            if (pca2_ix or pca1_ix) >= n_axes : 
                warnings.warn('Axis labels must be less than the number of axes.'
                            f' Need `{n_axes!r}` but `{max(lbls)!r}` are given.'
                            'The default first two components are used.')
                pca1_ix =0
                pca2_ix = pca1_ix+1
                pc1_label , pc2_label = 'Axis 1', 'Axis 2'
                
                self._logging.debug(
                   'Axis labels must be less than the number of axes.'
                    f' Need `{n_axes!r}` but `{max(lbls)!r}` are given.'
                    'The default first two components are used.'
                    )
                
            X_= np.c_[X_reduced[:, pca1_ix],
                      X_reduced[:, pca2_ix]]
            
        # prepared defaults colors and defaults markers 
        y_palettes = plot_dict ['y_colors']
        if y_classes  is not None:
            if len(y_palettes) > len(y_classes): 
                # reduce the last colors 
                y_palettes =y_palettes[:len(y_classes)]
            if len(y_palettes) < len(y_classes): 
                # add black colors  by default
                y_palettes += ['k' for k in range(
                    len(y_classes) - len(y_palettes))]
            

        # --Plot Biplot
        if biplot: 
            
            mpl.rcParams.update(mpl.rcParamsDefault) # reset ggplot style
            # Call the biplot function for only the first 2 PCs
            cmp_= np.concatenate((pca.components_[pca1_ix, :], 
                                  pca.components_[pca2_ix, :]))
            try: 
                biPlot(self, X_, np.transpose(cmp_), y,
                        y_classes=y_classes, 
                        colors=y_palettes )
            except : 
                # plot defaults configurations  
                biPlot(self, X_reduced[:,:2],
                        np.transpose(pca.components_[0:2, :]),
                        y, 
                        y_classes=y_classes, 
                        colors=y_palettes )
                plt.show()
            else : 
                plt.show()
            
            return  
        # created a dataframe concatenate reduced dataframe + y_target
        try: 
                
            df_pca =pd.concat([
                    pd.DataFrame(X_,columns =[pc1_label, pc2_label]),
                    pd.Series(y, name=y_label)],
                axis =1)
        except TypeError: 
            # force plot using the defauts first two componnets if 
            # something goes wrong
             df_pca =pd.concat([
                    pd.DataFrame(X_reduced[:,:2],
                                 columns =[pc1_label, pc2_label]),
                    pd.Series(y, name=y_label)],
                axis =1)
             pca1_ix , pca2_ix =0,1
  
        # Extract the name of the first components  and second components
        # ranged like [('pc1',['shape', 'power',...], [-0.85927608, -0.35507183,...] ),
                    # ('pc2', ['sfi', 'power', ...],#[ 0.50104756,  0.4565256 ,... ), ...]
        # print('pc1axes =', pca1_ix, 'pc1_label=', pc1_label)
        # print('pc2axes =', pca2_ix, 'pc2_label=', pc2_label)
        pca_axis_1 = feature_importances_[pca1_ix][1][0] 
        pca_axis_2 = feature_importances_[pca2_ix][1][0]
        # Extract the name of the  values of the first components 
        # and second components in percentage.
        pca_axis_1_ratio = np.around(
            abs(feature_importances_[pca1_ix][2][0]),2) *1e2
        pca_axis_2_ratio = np.around(
            abs(feature_importances_[pca2_ix][2][0]),2) *1e2
     
        # create figure obj 
        fig = plt.figure(figsize = self.fig_size)
        ax = fig.add_subplot(1,1,1)
        
        for target , color in zip(y_classes, y_palettes): 
            ix = df_pca[y_label] ==target
            ax.scatter(df_pca.loc[ix, pc1_label], 
                       df_pca.loc[ix, pc2_label], 
                       c= color, 
                       s= plot_dict['s'])
            
        # get the max values, set the center plot  and set the
        # of the circle bounds.
        max_lim = np.ceil(abs(max([X_reduced[:, pca1_ix].max(),
                                   X_reduced[:, pca2_ix].max()])))
        
        cercle = plt.Circle((0,0),
                            max_lim,
                            color='blue',
                            fill=False)
        ax.add_artist(cercle)
        ax.set_ylim([-max_lim, max_lim])
        ax.set_xlim([-max_lim, max_lim])
        
        linev =plt.Line2D ((0, 0), (-max_lim, max_lim),
                           color = self.lc, 
                           linewidth = self.lw,
                           linestyle = self.ls,
                           marker = self.marker_style,
                           markeredgecolor = self.marker_edgecolor,
                           markeredgewidth = self.marker_edgewidth,
                           markerfacecolor = self.marker_facecolor ,
                           markersize = self.ms * self.fs
                           )
        
        lineh =plt.Line2D ((-max_lim, max_lim), (0, 0),
                           color = self.lc, 
                           linewidth = self.lw,
                           linestyle = self.ls ,
                           marker = self.marker_style,
                           markeredgecolor = self.marker_edgecolor,
                           markeredgewidth = self.marker_edgewidth,
                           markerfacecolor = self.marker_facecolor,
                           markersize = self.ms * self.fs
                           )
        
        #Create string label from pca_axis_1
        x_axis_str = pc1_label +':'+ str(pca_axis_1) +' ({}%)'.format(
            pca_axis_1_ratio )
        y_axis_str = pc2_label +':' + str(pca_axis_2) +' ({}%)'.format(
            pca_axis_2_ratio )
        
        ax.set_xlabel( x_axis_str,
                      color='k', 
                      fontsize = self.font_size * self.fs
                      )
        ax.set_ylabel(y_axis_str,
                      color='k',
                      fontsize = self.font_size * self.fs
                      )
        ax.set_title('PCA', 
                     fontsize = (self.font_size +1) * self.fs)
        ax.add_artist(linev)
        ax.add_artist(lineh)
        ax.legend(y_classes)
        ax.grid(color=self.lc,
                linestyle=self.ls,
                linewidth=self.lw/10
                )
        
        plt.show()
        
        if self.savefig is not None :
            plt.savefig(self.savefig,
                        dpi=self.fig_dpi,
                        orientation =self.fig_orientation
                        )  
        
    @docSanitizer()
    @docAppender(precision_recall_tradeoff, from_='Parameters', to= 'Notes')        
    def PrecisionRecall(self,
                        clf,
                        X,
                        y,
                        cv =3,
                        kind='vsThreshod',
                        classe_ =None,
                        method="decision_function",
                        cross_val_pred_kws =None,
                        **prt_kws): 
        """ Precision/recall Tradeoff computes a score based on the decision 
        function. 
        
        Parameters
        -----------
        kind: str 
            kind of plot. Plot precision-recall vs thresholds (``vsThreshod``)
            or precision vs recall(``vsThreshod``). Default is ``vsThreshod``
     
        Examples
        ---------
            >>> from sklearn.linear_model import SGDClassifier
            >>> from watex.datasets import X_prepared
            >>> from watex.datasets import y_prepared
            >>> sgd_clf = SGDClassifier(random_state= 42)
            >>> mlObj= MLPlots(lw =3., pc = 'k', rc='b', ps='-', rs='--')
            >>> mlObj.PrecisionRecall(clf = sgd_clf,  X= X_prepared, 
            ...                y = y_prepared, classe_=1, cv=3,
            ...                 kind='vsRecall')
            
        See also
        ---------
        For parameter definitions, please refer to
        :meth:`watex.utils.metrics.Metrics.PrecisionRecallTradeoff`
        for further details.
               
        """
        # call precision 
        prtObj = precision_recall_tradeoff(
                                clf,
                                X, 
                                y, 
                                cv =cv, 
                                classe_=classe_, 
                                method =method, 
                                cross_val_pred_kws=cross_val_pred_kws,
                                **prt_kws)
        
        # create figure obj 
        fig = plt.figure(figsize = self.fig_size)
        ax = fig.add_subplot(1,1,1)
        if kind.lower().find('thres')>=0: 
            # plot precision -recall vs Threshold 
            kind = '.vsThreshold' 
        elif kind.lower().find('vsrec')>=0: 
            # plot precision vs recall 
            kind = '.vsRecall'
        if kind=='.vsThreshold': 
            
            ax.plot(prtObj.thresholds,
                    prtObj.precisions[:-1], 
                    color = self.pc, 
                    linewidth = self.lw,
                    linestyle = self.ps, 
                    label = 'Precision',
                    **self.plt_kws )
            ax.plot(prtObj.thresholds,
                   prtObj.recalls[:-1], 
                   color = self.rc, 
                   linewidth = self.lw,
                   linestyle = self.rs , 
                   label = 'Recall',
                   **self.plt_kws)
            
            if self.xlabel is None: self.xlabel ='Threshold'
            if self.ylabel is None: self.ylabel ='Score'
        
        elif kind =='.vsRecall': 
            
            ax.plot(prtObj.recalls[:-1],
                    prtObj.precisions[:-1], 
                    color = self.lc, 
                    linewidth = self.lw,
                    linestyle = self.ls , 
                    label = 'Precision vs Recall',
                    **self.plt_kws )
        
            if self.xlabel is None: self.xlabel ='Recall'
            if self.ylabel is None: self.ylabel ='Precision'
            self.xlim =[0,1]
            
        ax.set_xlabel( self.xlabel,
                      fontsize= .5 * self.font_size * self.fs )
        ax.set_ylabel (self.ylabel,
                       fontsize= .5 * self.font_size * self.fs)
        ax.tick_params(axis='both', 
                       labelsize=.5 * self.font_size * self.fs)
        
        if self.show_grid is True : 
           if self.gwhich =='minor': 
                 ax.minorticks_on() 
           ax.grid(self.show_grid,
                   axis=self.gaxis,
                   which = self.gwhich, 
                   color = self.gc,
                   linestyle=self.gls,
                   linewidth=self.glw, 
                   alpha = self.galpha
                   )
           
        if len(self.leg_kws) ==0 or 'loc' not in self.leg_kws.keys():
             self.leg_kws['loc']='upper left'
        
        ax.legend(**self.leg_kws)
        
        if self.ylim is None: self.ylim = [0, 1]
        ax.set_ylim (self.ylim)
        if kind =='.vsRecall':
            ax.set_xlim (self.xlim)

        plt.show()
        
        if self.savefig is not None :
            plt.savefig(self.savefig,
                        dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
            
    def ROC_curve_(self, 
                    clf,
                    X,
                    y,
                    cv =3,
                    classe_ =None,
                    method="decision_function",
                    cross_val_pred_kws =None,
                    **roc_kws):
        """
        Plot receiving operating characteric (ROC) classifiers. 
        
        Parameters 
        -----------
        clf: list of objects, callable,
            Classifier or estimators. To use multiple classifiers set a list
            of classifer with their specific mmethods. For instance 
            instance::
                
                [('SDG', SGDClassifier,"decision_function" ), 
                 ('FOREST', RandomForestClassifier,"predict_proba")]
                
        X: ndarray (nsamples, nfeatures), 
            Training data (trainset) composed of n-features.
            
        y: array_like 
            Labelf for prediction. `y` is binary label by defaut. 
            If '`y` is composed of multilabel, specify  the `classe_` 
            argumentto binarize the label(`True` ot `False`). ``True``  
            for `classe_`and ``False`` otherwise.
            
        cv: int 
            K-fold cross validation. Default is ``3``
            
        classe_: float, int 
            Specific class to evaluate the tradeoff of precision 
            and recall. If `y` is already a binary classifer, `classe_` 
            does need to specify. 
            
        method: str
            Method to get scores from each instance in the trainset. 
            Could be ``decison_funcion`` or ``predict_proba`` so 
            Scikit-Learn classifuier generally have one of the method. 
            Default is ``decision_function``.
            
        roc_kws: dict 
            roc_curve additional keywords arguments.
            

        Examples 
        --------
        >>> from sklearn.linear_model import SGDClassifier
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from watex.datasets import X_prepared
        >>> from watex.datasets import y_prepared
        >>> from watex.view.mlplot import MlPlots
        >>> sgd_clf = SGDClassifier(random_state= 42)
        >>> forest_clf =RandomForestClassifier(random_state=42)
        >>> mlObj= MLPlots(lw =3., lc=(.9, 0, .8), font_size=7
        >>> clfs =[('sgd', sgd_clf, "decision_function" ), 
         ...      ('forest', forest_clf, "predict_proba")]
        >>> mlObj.ROC_curve_(clf = clfs,  X= X_prepared, 
        ...                      y = y_prepared, classe_=1, cv=3)
 
        """
        
        # if method not given as tuple
        if not isinstance(clf, (list, tuple)):
            try : 
                clf =[(clf.__name__, clf, method)]
            except AttributeError: 
                # type `clf` is ABCMeta 
                 clf =[(clf.__class__.__name__, clf, method)]
                 
        # loop and set the tuple of  (clfname , clfvalue, clfmethod)
        # anc convert to list to support item assignments
        clf = [list(pnclf) for pnclf in clf]
        for i, (clfn, _clf, _) in enumerate(clf) :
        
            if  clfn is None  or clfn =='': 
                try: 
                    clfn = _clf.__name__
                except AttributeError: 
                    # when type `clf` is ABCMeta 
                    clfn= _clf.__class__.__name__
                clf[i][0] = clfn 
                
        # reconvert to tuple values 
        clf =[tuple(pnclf) for pnclf in clf]
        # build multiples classifiers objects 

        rocObjs =[ROC_curve(clf=_clf,X=X,y=y, cv =cv, classe_=classe_, 
                        method =meth, cross_val_pred_kws=cross_val_pred_kws,
                        **roc_kws) 
                  for (name, _clf, meth) in clf
                  ]
        # create figure obj 
        fig = plt.figure(figsize = self.fig_size)
        ax = fig.add_subplot(1,1,1)
        D_COLORS[0] = self.lc
        D_STYLES[0]= self.ls

        for ii, (name, _clf, _)  in enumerate( clf): 
            ax.plot(rocObjs[ii].fpr, 
                    rocObjs[ii].tpr, 
                    label =name + ' (AUC={:.4f})'.format(
                        rocObjs[ii].roc_auc_score), 
                    color =D_COLORS[ii],
                    linestyle = D_STYLES[ii] , 
                    linewidth = self.lw)
            
            
        if self.xlabel is None: 
            self.xlabel ='False Positive Rate'
        if self.ylabel is None: 
            self.ylabel ='True Positive Rate'
        self.xlim =[0,1]
        self.ylim =[0,1]
        ax.plot(self.xlim, self.ylim, ls= '--', color ='k')
        ax.set_xlim (self.xlim)
        ax.set_ylim (self.ylim)
        ax.set_xlabel( self.xlabel,
                      fontsize= .5 * self.font_size * self.fs )
        ax.set_ylabel (self.ylabel,
                       fontsize= .5 * self.font_size * self.fs)
        ax.tick_params(axis='both', 
                       labelsize=.5 * self.font_size * self.fs)
        
        if self.show_grid is True : 
           if self.gwhich =='minor': 
                 ax.minorticks_on() 
           ax.grid(self.show_grid,
                   axis=self.gaxis,
                   which = self.gwhich, 
                   color = self.gc,
                   linestyle=self.gls,
                   linewidth=self.glw, 
                   alpha = self.galpha
                   )
        if len(self.leg_kws) ==0 or 'loc' not in self.leg_kws.keys():
             self.leg_kws['loc']='lower right'
        ax.legend(**self.leg_kws)
        
        plt.show()
        
        if self.savefig is not None :
            plt.savefig(self.savefig,
                        dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
         
    def visualizingGeographycalData(self, X=None,  X_ =None, **kws ): 
        """ Visualize dataset. 
        
        Since there is geographical information(latitude/longitude or
         eating/northing), itis a good idea to create a scatterplot of 
        all instances to visualize data.
        
        Parameters
        ---------
        X: ndarray(n, 2), pd.DataFrame
            array composed of n-instances of two features. First features is
            use for x-axis and second feature for y-axis projection. 
        X_: nadarray(n, 2), pd.DataFrame
            array composed of n_instance in test_set.
        x: array_like  
            array of x-axis plot 
        y: array_like 
            array of y_axis 
        
        Examples
        --------
            
        >>> from sklearn.linear_model import SGDClassifier
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from watex.datasets import X, XT
        >>> from watex.datasets.data_preparing import stratified_test_set
        >>> mlObj= MLPlots(fig_size=(8, 12),
        ...                 lc='k',
        ...                 marker_style ='o',
        ...                 lw =3.,
        ...                 font_size=15.,
        ...                 xlabel= 'east',
        ...                 ylabel='north' , 
        ...                 markerfacecolor ='k', 
        ...                 markeredgecolor='r',
        ...                 alpha =1., 
        ...                 markeredgewidth=2., 
        ...                 show_grid =True,
        ...                 galpha =0.2, 
        ...                 glw=.5, 
        ...                 rotate_xlabel =90.,
        ...                 fs =3.,
        ...                 s =None, 
        ...                 )
        >>> mlObj.visualizingGeographycalData(X=X, X_=stratified_test_set) 
        """
        x=kws.pop('x', None)
        y=kws.pop('y', None)
        trainlabel=kws.pop('trainlabel', 'Train set')
        testlabel=kws.pop('testlabel', 'Test set')
        
        xy_labels =[self.xlabel,self.ylabel]
         
        if X is not None: 
            if isinstance(X, pd.DataFrame): 
                if xy_labels is not None: 
                    tem =list()
                    for label in list(X.columns):
                        for lab in xy_labels : 
                            if lab ==label: 
                                tem .append(label)
                                break 
                    if len(tem) ==0: 
                        warnings.warn(' Could not find the `{0}` labels '
                                      'in dataframe columns `{1}`'.format(
                                          xy_labels,list(X.columns)))
                        self._logging.warning(' Could not find the `{0}` labels '
                                      'in dataframe columns `{1}`'.format(
                                          xy_labels,list(X.columns)))

                    xy_labels = tem
                    
                    if len( xy_labels)<2: 
                        raise ArgumentError(
                            f'Need two labels for plot. {len(xy_labels)!r}'
                            ' is given.')
                    
                    if xy_labels is [None, None]: 
                        xy_labels = list(X.columns)
                X= X[xy_labels].to_numpy()

        if X is not None: 
            if X.shape[1] > 2:
                X= X[:, 2]
                
            x= X[:, 0]
            y= X[:, 1]
   
        if x is None:
            raise PlotError (
                'NoneType could not be plotted.  need `x-axis` value for plot.')
        if y is None: 
            raise PlotError (
                'NoneType could not be plotted. Need `y-axis` value for plot.')
        if len(x) !=len(y): 
            raise TypeError('`x`and `y` must have the same length. '
                            f'{len(x)!r} and {len(y)!r} are given respectively.')
            
            
        self.xlim =[np.ceil(min(x)), np.floor(max(x))]
        self.ylim =[np.ceil(min(y)), np.floor(max(y))]   
        
        xpad = abs((x -x.mean()).min())/5.
        ypad = abs((y -y.mean()).min())/5.
 
        if  X_ is not None: 
            if isinstance(X_, pd.DataFrame): 
                X_= X_[xy_labels].to_numpy()
            x_= X_[:, 0]
            y_= X_[:, 1]
            min_x, max_x = x_.min(), x_.max()
            min_y, max_y = y_.min(), y_.max()
            
            
            self.xlim = [min([self.xlim[0], np.floor(min_x)]),
                         max([self.xlim[1], np.ceil(max_x)])]
            self.ylim = [min([self.ylim[0], np.floor(min_y)]),
                         max([self.ylim[1], np.ceil(max_y)])]
          
        self.xlim =[self.xlim[0] - xpad, self.xlim[1] +xpad]
        self.ylim =[self.ylim[0] - ypad, self.ylim[1] +ypad]
        
         # create figure obj 
        fig = plt.figure(figsize = self.fig_size)
        ax = fig.add_subplot(1,1,1)
        
        if self.xlabel is None: 
            self.xlabel =xy_labels[0]
        if self.ylabel is None: 
            self.ylabel =xy_labels[1]

        if self.s is None: 
            self.s = self.fs *40 
        ax.scatter(x, y, 
                   color = self.lc,
                    s = self.s,
                    alpha = self.alpha , 
                    marker = self.marker_style,
                    edgecolors = self.marker_edgecolor,
                    linewidths = self.lw,
                    linestyles = self.ls,
                    facecolors = self.marker_facecolor,
                    label = trainlabel 
                )
        
        if  X_ is not None:
            if self.s is not None: 
                self.s /=2 
            ax.scatter(x_, y_, 
                       color = 'b',
                        s = self.s,
                        alpha = self.alpha , 
                        marker = self.marker_style,
                        edgecolors = 'b',
                        linewidths = self.lw,
                        linestyles = self.ls,
                        facecolors = 'b',
                        label = testlabel)
        
        
        ax.set_xlim (self.xlim)
        ax.set_ylim (self.ylim)
        ax.set_xlabel( self.xlabel,
                      fontsize= self.font_size )
        ax.set_ylabel (self.ylabel,
                       fontsize= self.font_size )
        ax.tick_params(axis='both', 
                       labelsize= self.font_size )
        plt.xticks(rotation = self.rotate_xlabel)
        plt.yticks(rotation = self.rotate_ylabel)
        
        if self.show_grid is True : 
            ax.grid(self.show_grid,
                    axis=self.gaxis,
                    which = self.gwhich, 
                    color = self.gc,
                    linestyle=self.gls,
                    linewidth=self.glw, 
                    alpha = self.galpha
                    )
            if self.gwhich =='minor': 
                ax.minorticks_on()
                
        if len(self.leg_kws) ==0 or 'loc' not in self.leg_kws.keys():
             self.leg_kws['loc']='upper left'
        ax.legend(**self.leg_kws)
        
        plt.show()
        
        if self.savefig is not None :
            plt.savefig(self.savefig,
                        dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
    
    @docSanitizer()
    @docAppender(confusion_matrix_, from_='Parameters', to='Examples')
    def confusion_matrix(self, clf, X, y, cv, *, plottype ='map', ylabel=None, 
                         matshow_kws=dict(), **conf_mx_kws): 
        """ Plot confusion matrix for error analysis
        
        Look a representation of the confusion matrix using Matplotlib matshow.
        
        Parameters 
        ----------
         plottype: str 
            can be `map` or `error` to visualize the matshow of prediction 
            and errors  respectively.
            
        matshow_kws: dict 
            matplotlib additional keywords arguments. 
            
        conf_mx_kws: dict 
            Additional confusion matrix keywords arguments.
        ylabel: list 
            list of labels names  to hold the name of each categories.
            
  
        Examples
        --------
        >>> from sklearn.svm import SVC 
        >>> from watex.view.mlplot import MLPlots
        >>> from watex.datasets import fetch_data 
        >> X,y = fetch_data('Bagoue dataset prepared')
        >>> svc_clf = SVC(C=100, gamma=1e-2, kernel='rbf', 
        ...                  random_state =42) 
        >>> matshow_kwargs ={
        ...        'aspect': 'auto', # 'auto'equal
        ...        'interpolation': None, 
        ...       'cmap':'gray' }                   
        >>> plot_kws ={'lw':3, 
        ...       'lc':(.9, 0, .8), 
        ...       'font_size':15., 
        ...        'cb_format':None,
        ...        'xlabel': 'Predicted classes',
        ...        'ylabel': 'Actual classes',
        ...        'font_weight':None,
        ...        'tp_labelbottom':False,
        ...        'tp_labeltop':True,
        ...        'tp_bottom': False
        ...        }
        >>> mObj =MLPlots(**plot_kws)
        >>> mObj.confusion_matrix(svc_clf, X=X,y=y,cv=7,                                   
        ...                        ylabel=['FR0', 'FR1', 'FR2', 'FR3'], 
        ...                        plottype='error'
        ...                        matshow_kws = matshow_kwargs,
        ...                        ) 
        
        See also 
        ---------
        Refer to :meth:`watex.utils.metrics.Metrics.confusion_matrix` 
        for furthers details.
        
        """
        _check_cmap = 'cmap' in matshow_kws.keys()
        if not _check_cmap or len(matshow_kws)==0: 
            matshow_kws['cmap']= plt.cm.gray
        
        if ylabel is not None: 
            #check the length of y and compare to y unique 
            cat_y = np.unique(y)
            if isinstance(ylabel, str) or len(ylabel)==1: 
                warnings.warn(
                   f"One label is given, need {len(cat_y)!r}. Can not be"
                    f" used to format {cat_y!r}"
                    )
                self._logging.debug(
                    f"Only one label is given. Need {len(cat_y)!r}"
                    'instead as the number of categories.')
                ylabel =None 
                
            type_y= isinstance(ylabel, (list, tuple, np.ndarray))
            if type_y:
                if len(cat_y) != len(ylabel): 
                    warnings.warn(
                        f" {'are' if len(ylabel)>1 else 'is'} given."
                        f"Need {len(cat_y)!r} instead.")
                    self._logging.debug(
                        f" {'are' if len(ylabel)>1 else 'is'} given."
                        f"Need {len(cat_y)!r} instead.")
                    ylabel =None 
                    
        # get yticks one it is a classification prof
        confObj =confusion_matrix_(clf=clf,
                                X=X,
                                y=y,
                                cv=cv,
                                **conf_mx_kws)
        
        # set all attributes in the case you want to get attributes 
        # for other purposes.
        for key in confObj.__dict__.keys():
            self.__setattr__(key, confObj.__dict__[key])
            
         # create figure obj 
        fig = plt.figure(figsize = self.fig_size)
        ax = fig.add_subplot(1,1,1)
        
        if plottype in ('map', 'default') : 
            cax = ax.matshow(confObj.conf_mx,  
                        **matshow_kws)
            if self.cb_label is None: 
                self.cb_label='Items confused'
                    
        if plottype in( 'error', 'fill diagonal') or\
            plottype.find('error')>=0:
            cax = ax.matshow(confObj.norm_conf_mx, 
                         **matshow_kws) 
            if self.cb_label is None: 
                self.cb_label='Error'
                
        cbax= fig.colorbar(cax, **self.cb_props)
        
        ax.set_xlabel( self.xlabel,
              fontsize= self.font_size )
        
        if ylabel is not None: 
            ax.set_xticks(np.unique(y))
            ax.set_xticklabels(ylabel)
            ax.set_yticks(np.unique(y))
            ax.set_yticklabels(ylabel)
            
        if self.ylabel is None:
            self.ylabel ='Actual classes'
        if self.xlabel is None:
            self.xlabel = 'Predicted classes'
        
        ax.set_ylabel (self.ylabel,
                       fontsize= self.font_size )
        ax.tick_params(axis=self.tp_axis, 
                        labelsize= self.font_size, 
                        bottom=self.tp_bottom, 
                        top=self.tp_top, 
                        labelbottom=self.tp_labelbottom, 
                        labeltop=self.tp_labeltop
                        )
        if self.tp_labeltop: 
            ax.xaxis.set_label_position('top')
        
        cbax.ax.tick_params(labelsize=self.font_size ) 
        cbax.set_label(label=self.cb_label,
                       size=self.font_size,
                       weight=self.font_weight)
        
        plt.xticks(rotation = self.rotate_xlabel)
        plt.yticks(rotation = self.rotate_ylabel)
  
        plt.show ()
        if self.savefig is not None :
           plt.savefig(self.savefig,
                       dpi=self.fig_dpi,
                       orientation =self.fig_orientation)
        return self


    def model(self, y_, ypred=None,*, clf=None, X_=None, predict =False, 
              prefix=None, index =None, fill_between=False, ylabel=None ): 
        """ Plot model from test sets or using a sample of predicted test.
        
        Parameters
        ----------
        y_:array-like of test data 
            test data or sample of label to predict 
            
        y_pred:array-like 
            predicted label 
            
        clf: callable
            Estimator of classifier 
            
        X_: ndarra of (n_samples, n_features)
            Test set to predict data. If `X_` is given  turn `predict` 
            to ``True`` to predict test data.
            
        predict:bool, 
            Make a prediction if test set `X_` is given
            
        prefix: str 
            prefix to add to your index values. For instance::
            
            index =[0, 2, 4, 7]
            prefix ='b' --> index =['b0', 'b2', 'b4', 'b7'] 
            
        index: array_like 
            list of array like of indexes. it will replace the indexes of 
            pd.Series or dataframe index if `X_` is given. 
            
        fill_between: bool 
            Fill a line between the actual classes `y_` (test label)
            
         ylabel: list 
            list of labels names  to hold the name of each categories.
            
        Examples
        --------
        
        >>> from sklearn.svm import SVC 
        >>> from watex.view.mlplot import MLPlots 
        >>> from watex.datasets import fetch_data 
        >>> X,y = fetch_data('Bagoue dataset prepared')
        >>> svc_clf = SVC(C=100, gamma=1e-2, kernel='rbf', random_state =42) 
        >>> plot_kws ={'lw' :3.,                  # line width 
                     'lc':(.9, 0, .8), 
                     'ms':7.,                
                     'yp_marker_style' :'o', 
                     'fig_size':(12, 8),
                    'font_size':15.,
                    'xlabel': 'Test examples',
                    'ylabel':'Flow categories' ,
                    'marker_style':'o', 
                    'markeredgecolor':'k', 
                    'markerfacecolor':'b', 
                    'markeredgewidth':3, 
                    'yp_markerfacecolor' :'k', 
                    'yp_markeredgecolor':'r', 
                    'alpha' :1., 
                    'yp_markeredgewidth':2.,
                    'show_grid' :True,          
                    'galpha' :0.2,              
                    'glw':.5,                   
                    'rotate_xlabel' :90.,
                    'fs' :3.,                   
                    's' :20 ,                  
                    'rotate_xlabel':90
                       }
        >>> modObj = MLPlots(**plot_kws)
        >>> modObj.model(y, X_=X, clf =svc_clf, 
                      predict= True, 
                      prefix ='b' ,
                      fill_between =False, 
                      ylabel=['FR0', 'FR1', 'FR2', 'FR3']
                      )
        """
        
        if index is not None:
            #control len of index and len of y
            try : 
                mess ='Object `index` has no length.'+\
                    ' Could not be an index.'
                len(index)
            except TypeError as type_error : 
                raise TypeError(mess) from type_error 
             
            len_index=  len(y_)==len(index)
            
            if not len_index:
                warnings.warn(
                    f"Index must have the same lenght as `y`={len(y_)!r}"
                    f" but {len(index)!r} {'are' if len(index)>1 else 'is'}"
                    " given.")
                self._logging.debug(
                    f"Index must get the same lenght as `y`={len(y_)!r}"
                    f" but {len(index)!r} {'are' if len(index)>1 else 'is'}"
                    " given.")
                index =None
                
            if len_index : 
                if isinstance(y_, (pd.Series, pd.DataFrame)):
                    if not np.all(y_.index.isin(index)):
                        warnings.warn('Indexes values provided are not in'
                                      ' `y_`. Shrank index to `y`index.',
                                      UserWarning)
                        self._logging.debug('Index values are not in `y`. Index are'
                                            ' shrank to hold indexes of `y`.')
                        index =y_.index 
                        y_=y_.values()
        
                # if prefix is not None: 
                #     #add prefix to index
                #     index =np.array([f'{prefix}' +str(item) 
                #                      for item in index ])
                
                y_=pd.Series(y_, index = index )
                
        if predict: 
  
            if clf is None: 
                warnings.warn('None estimator found! Could not predict `y` ')
                self._logging.error('NoneType `clf` <estimator> could not'
                                    ' predict `y`.')
                raise ValueError('None estimator detected!'
                                 ' could not predict `y`') 
            if X_ is None: 
                raise TypeError('NoneType can not used for prediction.'
                                ' Need a test set `X`.')
  
            # check estimator as callable object or ABCMeta classes
            if not hasattr(clf, '__call__') and  not inspect.isclass(clf)\
                and  type(clf.__class__)!=ABCMeta: 
  
                raise TypeError(f"{clf.__class__.__name__!r} is not a classifier "
                                 " or an estimator. Could not use for prediction.")
            clf.fit(X_, y_)
            ypred = clf.predict(X_)
            
            if isinstance(X_, (pd.DataFrame, pd.Series)):
                if index is None:
                    index = X_.index
                
        if isinstance(y_, pd.Series): 
            index = y_.index.astype('>U12')
        
        if index is None: 
            # take default values if  indexes are not given 
            index =np.array([i for i in range(len(y_))])
            
        if prefix is not None: 
            index =np.array([f'{prefix}' +str(item) 
                                 for item in index ])
        if len(y_)!=len(ypred): 
            raise TypeError(" `y` and predicted `ypred` must have"
                            f" the same length. But {len(y_)!r} and "
                            f"{len(ypred)!r} wre given respectively.")
            
         # create figure obj 
        fig = plt.figure(figsize = self.fig_size)
        ax = fig.add_subplot(1,1,1) # create figure obj 
        
        # plot the predicted target
        if self.s is None: 
            self.s = self.fs *40 
        ax.scatter(x= index, y =ypred ,
                  color = self.yp_lc,
                   s = self.s,
                   alpha = self.alpha, 
                   marker = self.yp_marker_style,
                   edgecolors = self.yp_marker_edgecolor,
                   linewidths = self.yp_lw,
                   linestyles = self.yp_ls,
                   facecolors = self.yp_marker_facecolor,
                   label = 'Predicted'
                   )
        
        # plot obverved data (test label =actual)
        ax.scatter(x= index,
                   y =y_ ,
                    color = self.lc,
                     s = self.s/2,
                     alpha = self.alpha, 
                     marker = self.marker_style,
                     edgecolors = self.marker_edgecolor,
                     linewidths = self.lw,
                     linestyles = self.ls,
                     facecolors = self.marker_facecolor,
                     label = 'Observed'
                       )    
            
        if fill_between: 
            ax.plot(y_, 
                    c=self.lc,
                    ls=self.ls, 
                    lw=self.lw, 
                    alpha=self.alpha
                    )
        if self.ylabel is None:
            self.ylabel ='Categories '
        if self.xlabel is None:
            self.xlabel = 'Test data'
            
        if ylabel is not None: 
            mess =''.join([ 
                    'Label must have the same length with number of categories',
                    f" ={len(np.unique(y_))!r}, but{len(ylabel)!r} ",
                    f"{'are' if len(ylabel)>1 else 'is'} given."])
            if len(ylabel) != len(np.unique(y_)): 
                warnings.warn(mess
                    )
                self._logging.debug(mess)
            else:
                ax.set_yticks(np.unique(y_))
                ax.set_yticklabels(ylabel)
        ax.set_ylabel (self.ylabel,
                       fontsize= self.font_size )
        ax.set_xlabel (self.xlabel,
               fontsize= self.font_size )
   
        if self.tp_axis is None or self.tp_axis =='both': 
            ax.tick_params(axis=self.tp_axis, 
                labelsize= self.tp_labelsize, 
                )
            
        elif self.tp_axis =='x':
            param_='y'
        elif self.tp_axis =='y': 
            param_='x'
            
        if self.tp_axis in ('x', 'y'):
            ax.tick_params(axis=self.tp_axis, 
                            labelsize= self.tp_labelsize, 
                            )
            
            ax.tick_params(axis=param_, 
                    labelsize= self.font_size, 
                    )
        
        plt.xticks(rotation = self.rotate_xlabel)
        plt.yticks(rotation = self.rotate_ylabel)
        
        if self.show_grid is True : 
            ax.grid(self.show_grid,
                    axis=self.gaxis,
                    which = self.gwhich, 
                    color = self.gc,
                    linestyle=self.gls,
                    linewidth=self.glw, 
                    alpha = self.galpha
                    )
            if self.gwhich =='minor': 
                ax.minorticks_on()
                
        if len(self.leg_kws) ==0 or 'loc' not in self.leg_kws.keys():
             self.leg_kws['loc']='upper left'
        ax.legend(**self.leg_kws)
        
        plt.show()
        if self.savefig is not None :
            plt.savefig(self.savefig,
                        dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
        
    def plot_learning_curves(self, clf, X, y, test_size=0.2, scoring ='mse',
                             **split_kws) : 
        """ Plot learning curves
        Use cross validation to get an estimate of model's generalisation 
        performance. 
        
        Parameters 
        ----------
        clf: callable 
            model estimator of classifier 
        X: ndarray(m_examples, n_features)
            training data set 
        y: array-like 
            y-label for predicting purpose 
            
        split_kws: dict 
            Additional keywords arguments. Hold from scikit-learn 
            class:`~sklearn.model_selection.train_test_split`
        """ 
        from sklearn.model_selection import train_test_split 
        from sklearn.metrics import mean_squared_error 
        
        if scoring is not None:
            try :
                scoring = scoring.lower()
            except : 
                raise TypeError(f"Scoring ={scoring!r} should be a string"
                                " not a {type(scoring)!} type.")
                
        if scoring in ('mean_squared_error', 'mean squared error') :
            scoring ='mse'
        elif scoring in ('root_mean_squared_error', 'root mean squared error'):
            scoring ='rmse'
        
        if not hasattr(clf, '__class__') and not inspect.isclass(clf.__class__): 
            raise TypeError("{clf!r} is not a model estimator.")
            
        self._logging.info(
                   f"Plot learning curve with scoring ={scoring}")    
        X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                          test_size=test_size,
                                                          **split_kws)
        train_errors, val_errors = [], []
        for m in range(1, len(y_train)): 
            try:
                clf.fit(X_train[:m], y_train[:m])
            except ValueError: 
                #The number of classes has to be greater than one; got 1 class
                continue
            y_train_pred = clf.predict(X_train[:m])
            y_val_pred = clf.predict(X_val)
            if scoring in ('mse','rmse') :
                train_errors.append(mean_squared_error(
                    y_train_pred, y_train[:m]))
                val_errors.append(
                    mean_squared_error(y_val_pred, y_val))
            else:
                train_errors.append(sum(
                    y_train_pred==y_train[:m])/len(y_train_pred))
                val_errors.append(
                    sum(y_val_pred==y_val)/len(y_val_pred))
         # create figure obj 
         
        if scoring =='rmse': 
            train_errors= np.sqrt(train_errors)
            val_errors = np.sqrt(val_errors)
            
        if self.ylabel is None:
            self.ylabel = scoring.upper()
            if scoring =='accuracy': 
                self.ylabel ='Score'
                
        if self.xlabel is None: 
            self.xlabel = 'Training set size'
            
        fig = plt.figure(figsize = self.fig_size)
        ax = fig.add_subplot(1,1,1) # create figure obj 
        
        # set new attributes 
        for nv, vv in zip(('vlc', 'vls'), ('b', ':')): 
            if not hasattr(self, nv): 
                setattr(self, nv, vv)
            
        ax.plot(train_errors,
                color = self.lc, 
                linewidth = self.lw,
                linestyle = self.ls , 
                label = 'training set',
                **self.plt_kws )
        ax.plot(val_errors,
                color = self.vlc, 
                linewidth = self.lw,
                linestyle = self.vls , 
                label = 'validation set',
                **self.plt_kws )
        
        appendLineParams(self, ax)
        
    def plotModelvsCV(self, clfs, scores=None, cv =None, **lcs_kws): 
        """ Visualize model fined tuned scores vs the cross validation
        
        Parameters 
        ----------
        clfs: callable 
            list of estimators names or a pairs estaimator and validations scores.
            For instance:: 
                
                clfs =[('SVM', scores_svm), ('LogRegress', scores_logregress), ...]
                
        scores: array like 
            list of scores on different validation sets. If scores are given, set 
            differently the `clfs` like only the name of the estimators Like:: 
                
                clfs =['SVM', 'LogRegress', ...]
                errors[errors_svm, errors_logregress, ...]
    
        cv: int, 
            number of Fold to visualize. If ``None``, visualize all cross folds.
        
        lcs_kws: dict 
            Additional keywors to customize each fine-tuned estimators. 
            It  composed of the line colors `lc` and line style `ls`. 

        """
        
        if clfs is None and scores is None: 
            raise ValueError('NoneType can not be plot.')
            
        _ckeck_score = scores is not None 
        
        if _ckeck_score :
            if isinstance(clfs, str): 
            
                clfs =[(clfs, scores)] 
            elif  isinstance(clfs, (list, tuple)) and \
                isinstance(scores, (list, tuple, np.ndarray)):
                if len(clfs) != len(scores): 
                    raise TypeError('Number of model fine-tuned and scores must have the'
                                    f" same length. {len(clfs)!r} and {len(scores)!r} "
                                    " were given respectively.")
                clfs=[(bn, bscore) for bn, bscore in zip(clfs, scores)]
            
        for ii, (clf, _) in enumerate(clfs) : 
            if clf is None:
                if hasattr(clf, '__call__') or inspect.isclass(clf.__class__): 
                    clfs[ii] = clf.__class__.__name__
        
        if not isinstance(cv, (int, float) ): 
            warnings.warn(f"type {type(cv)!r} is unacceptable type for"
                          " cross-validation. Should be integer value. "
                          "Value reseting to None")
            self._logging.warning(f"Unacceptable type {type(cv)!r}. "
                                  "Value resetting to None.")
            cv =None 
            
        if cv is None: 
            cv = len(clfs[0][1])   
            
        if cv is not None: 
            # shrink to the number of validation to keep 
            clfs = [(clfname, clfval[:cv] ) for clfname, clfval in clfs]
            
         # create figure obj 
                         
        # customize plots with colors lines and 
        if len(lcs_kws)==0:
            lcs_kws = {'lc':[self.lc, self.pc, self.rc ] + D_COLORS, 
                     'ls':[self.ls, self.ps, self.rs] + D_STYLES
                     }
    
        fig = plt.figure(figsize = self.fig_size)
        ax = fig.add_subplot(1,1,1) # create figure obj 
        
        for k in range(len(clfs)): 
            ax.plot(np.array([i for i in range(cv)])+1,
                    clfs[k][1],
                    color = lcs_kws['lc'][k], 
                    linewidth = self.lw,
                    linestyle = lcs_kws['ls'][k], 
                    label = clfs[k][0],
                    **self.plt_kws 
                    )
        appendLineParams(self, ax, xlim=self.xlim, ylim=self.ylim)
        
def appendLineParams(self, ax, xlim=None, ylim=None): 
    """ DRY(Dont Repeat Yourself). So append  the remain lines configuration 
    such as xlabel, grid , legend and ticks parameters holf from `MLPlots`
    objects.
    
    :param ax: axis to plot.
    
    """
    
    if self.xlabel is None: 
        self.xlabel =''
    if self.ylabel is None: 
        self.ylabel =''
        
    if xlim is not None: 
        ax.set_xlim(xlim)

    if ylim is not None: 
        ax.set_ylim(ylim)
        
    ax.set_xlabel( self.xlabel,
                  fontsize= .5 * self.font_size * self.fs )
    ax.set_ylabel (self.ylabel,
                   fontsize= .5 * self.font_size * self.fs)
    ax.tick_params(axis='both', 
                   labelsize=.5 * self.font_size * self.fs)
    
    if self.show_grid is True : 
       if self.gwhich =='minor': 
             ax.minorticks_on() 
       ax.grid(self.show_grid,
               axis=self.gaxis,
               which = self.gwhich, 
               color = self.gc,
               linestyle=self.gls,
               linewidth=self.glw, 
               alpha = self.galpha
               )
       
    if len(self.leg_kws) ==0 or 'loc' not in self.leg_kws.keys():
         self.leg_kws['loc']='best'
    
    ax.legend(**self.leg_kws)

    plt.show()
    
    if self.savefig is not None :
        plt.savefig(self.savefig,
                    dpi=self.fig_dpi,
                    orientation =self.fig_orientation)  
        
    return self 

def plot_matshow(self, matrix, x_label=None, y_label=None, **matshow_kws): 
    """ Quick matrix visualization using matplotlib.pyplot.matshow.
    
    Parameters
    ----------
    matrix: ndarray
        matrix of n rowns and m-columns items 
    matshow_kws: dict
        Additional keywords arguments.
        
     ylabel: list 
            list of labels names  to hold the name of each categories.
    ylabel: list 
       list of labels names  to hold the name of each categories.
       
    Examples
    ---------
    >>> import numpy as np
    >>> import watex.view.mlplot as WPL 
    >>>  matshow_kwargs ={
        'aspect': 'auto',
        'interpolation': None 
       'cmap':'copper_r', 
            }
    >>> plot_kws ={'lw':3, 
               'lc':(.9, 0, .8), 
               'font_size':15., 
                'cb_format':None,
                #'cb_label':'Rate of prediction',
                'xlabel': 'Predicted flow classes',
                'ylabel': 'Geological rocks',
                'font_weight':None,
                'tp_labelbottom':False,
                'tp_labeltop':True,
                'tp_bottom': False
                }
    >>> xlabel =['FR0', 'FR1', 'FR2', 'FR3', 'Rates'] 
    >>> ylabel =['VOLCANO-SEDIM. SCHISTS', 'GEOSYN. GRANITES', 
                 'GRANITES', '1.0', 'Rates']
    >>> array = np.array([(1. , .5, 1. ,1., .9286), 
                        (.5,  .8, 1., .667, .7692),
                        (.7, .81, .7, .5, .7442),
                        (.667, .75, 1., .75, .82),
                        (.9091, 0.8064, .7, .8667, .7931)])
    >>> mObj =WPL.MLPlots(**plot_kws)
    >>> WPL.plot_matshow(mObj, array, x_label=xlabel, 
                         y_label= ylabel, **matshow_kwargs)
    """
    # create figure obj 
    fig = plt.figure(figsize = self.fig_size)
    ax = fig.add_subplot(1,1,1)

    cax = ax.matshow(matrix, 
                     **matshow_kws) 

    cbax= fig.colorbar(cax, **self.cb_props)
    
    if self.cb_label is None: 
        self.cb_label=''
    ax.set_xlabel( self.xlabel,
          fontsize= self.font_size )
    

    if y_label is not None:
        ax.set_yticks(np.arange(0, matrix.shape[1]))
        ax.set_yticklabels(y_label)
    if x_label is not None: 
        ax.set_xticks(np.arange(0, matrix.shape[1]))
        ax.set_xticklabels(x_label)
        
    if self.ylabel is None:
        self.ylabel =''
    if self.xlabel is None:
        self.xlabel = ''
    
    ax.set_ylabel (self.ylabel,
                   fontsize= self.font_size )
    ax.tick_params(axis=self.tp_axis, 
                    labelsize= self.font_size, 
                    bottom=self.tp_bottom, 
                    top=self.tp_top, 
                    labelbottom=self.tp_labelbottom, 
                    labeltop=self.tp_labeltop
                    )
    if self.tp_labeltop: 
        ax.xaxis.set_label_position('top')
    
    cbax.ax.tick_params(labelsize=self.font_size ) 
    cbax.set_label(label=self.cb_label,
                   size=self.font_size,
                   weight=self.font_weight)
    
    plt.xticks(rotation = self.rotate_xlabel)
    plt.yticks(rotation = self.rotate_ylabel)
  
    plt.show ()
    if self.savefig is not None :
       plt.savefig(self.savefig,
                   dpi=self.fig_dpi,
                   orientation =self.fig_orientation)
    return self  
   
if __name__=='__main__': 

    # from sklearn.linear_model import SGDClassifier
    # from sklearn.ensemble import RandomForestClassifier
    from watex.datasets import fetch_data 
    import  watex.utils.mlutils as mfunc
    df = mfunc.load_data('data/geo_fdata')
    X,_ = fetch_data('Bagoue stratified sets')
    X_,_= fetch_data('test sets')

    mlObj= MLPlots(lw =3., lc='k', marker_style ='o', fig_size=(8, 12),
                   font_size=15.,
                   xlabel= 'east',
                   ylabel='north' , 
                   markerfacecolor ='k', 
                   markeredgecolor='r', alpha =1., 
                   markeredgewidth=2., show_grid =True,galpha =0.2, glw=.5, 
                   rotate_xlabel =90.,fs =3.,s =None, 
                   )
    mlObj.visualizingGeographycalData(X=X, X_=X_)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        