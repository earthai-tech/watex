# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Created on Wed Jul  7 22:23:02 2022 

from __future__ import annotations 

import os
import datetime 
import warnings
import itertools 
import numpy as np
import matplotlib as mpl 
# import matplotlib.cm as cm 
from scipy.cluster.hierarchy import ( 
    dendrogram, ward 
    )
import matplotlib.pyplot as plt

from ..exceptions import ( 
    TipError, 
    PlotError, 
    )

from .funcutils import is_iterable
from .validator import _check_array_in  

is_mlxtend =False 
try : 
    from mlxtend.Plotting import ( 
        scatterplotmatrix , heatmap ) 
except : pass 
else : is_mlxtend =True 

D_COLORS =[
    'g',
    'gray',
    'y', 
    'blue',
    'orange',
    'purple',
    'lime',
    'k', 
    'cyan', 
    (.6, .6, .6),
    (0, .6, .3), 
    (.9, 0, .8),
    (.8, .2, .8),
    (.0, .9, .4)
]

D_MARKERS =[
    'o',
    '^',
    'x',
    'D',
    '8',
    '*',
    'h',
    'p',
    '>',
    'o',
    'd',
    'H'
]

D_STYLES = [
    '-',
    '-',
    '--',
    '-.',
    ':', 
    'None',
    ' ',
    '',
    'solid', 
    'dashed',
    'dashdot',
    'dotted' 
]
    

def plot_naive_dendrogram (X, *ybounds, fig_size = (12, 5 ),  **kws): 
    """ Quick plot dendrogram using the ward clustering function from Scipy.
    
    :param X: ndarray of shape (n_samples, n_features) 
        Array of features 
    :param ybounds: int, 
        integrer values to draw horizontal cluster lines that indicate the 
        number of clusters. 
    :param fig_size: tuple (width, height), default =(8,5) 
        the matplotlib figure size given as a tuple of width and height 
    :param kws: dict , 
        Addditional keyword arguments passed to 
        :func:`scipy.cluster.hierarchy.dendrogram`
    :Examples: 
        >>> from watex.datasets import fetch_data 
        >>> from watex.utils.plotutils import plot_naive_dendrogram
        >>> X, _= fetch_data('Bagoue analysed') # data is already scaled 
        >>> # get the two features 'power' and  'magnitude'
        >>> data = X[['power', 'magnitude']]
        >>> plot_naive_dendrogram(data ) 
        >>> # add the horizontal line of the cluster at ybound = (20 , 20 )
        >>> plot_naive_dendrogram(data , 20, 20 ) 
   
    """
    # assert ybounds agument if given
    msg =(". Note that the bounds in y-axis are the y-coordinates for"
          " horizontal lines regarding to the number of clusters that"
          " might be cutted.")
    try : 
        ybounds = [ int (a) for a in ybounds ] 
    except Exception as typerror: 
        raise TypeError  (str(typerror) + msg)
    else : 
        if len(ybounds)==0 : ybounds = None 
    # the scipy ward function returns 
    # an array that specifies the 
    # distance bridged when performed 
    # agglomerate clustering
    linkage_array = ward(X) 
    
    # plot the dendrogram for the linkage array 
    # containing the distances between clusters 
    dendrogram( linkage_array , **kws )
    
    # mark the cuts on the tree that signify two or three clusters
    #change the figsize 
    plt.rcParams["figure.figsize"] = fig_size
    ax= plt.gca () 
  
    if ybounds is not None: 
        if not is_iterable(ybounds): 
            ybounds =[ybounds] 
        if len(ybounds) <=1 : 
            warnings.warn(f"axis y bound might be greater than {len(ybounds)}")
        else : 
            # split ybound into sublist of pair (x, y) coordinates
            nsplits = len(ybounds)//2 
            len_splits = [ 2 for i in range (nsplits)]
            # compose the pir list (x,y )
            itb = iter (ybounds)
            ybounds = [list(itertools.islice (itb, it)) for it in len_splits]
            bounds = ax.get_xbound () 
            for i , ( x, y)  in enumerate (ybounds)  : 
                ax.plot(bounds, [x, y], '--', c='k') 
                ax.text ( bounds [1], y , f"cluster {i +1:02}",
                         va='center', 
                         fontdict ={'size': 15}
                         )
    # get xticks and format labels
    xticks_loc = list(ax.get_xticks())
    _get_xticks_formatage(ax, xticks_loc, space =14 )
    
    plt.xlabel ("Sample index ")
    plt.ylabel ("Cluster distance")
            
    
def plot_pca_components (
        components, *, feature_names = None , cmap= 'viridis' , **kws
        ): 
    """ Visualize the coefficient of principal component analysis (PCA) as 
    a heatmap  
  
    :param components: Ndarray, shape (n_components, n_features)or PCA object 
        Array of the PCA compoments or object from 
        :class:`watex.analysis.dimensionality.nPCA`. If the object is given 
        it is not usefull to set the `feature_names`
    :param feature_names: list or str, optional 
        list of the feature names to locate in the map. `Feature_names` and 
        the number of eigen vectors must be the same length. If PCA object is  
        passed as `components` arguments, no need to set the `feature_names`. 
        The name of features is retreived automatically. 
    :param cmap: str, default='viridis'
        the matplotlib color map for matshow visualization. 
    :param kws: dict, 
        Additional keywords arguments passed to 
        :class:`matplotlib.pyplot.matshow`
        
    :Examples: 
    (1)-> with PCA object 
    
    >>> from watex.datasets import fetch_data
    >>> from watex.utils.plotutils import plot_pca_components
    >>> from watex.analysis import nPCA 
    >>> X, _= fetch_data('bagoue pca') 
    >>> pca = nPCA (X, n_components=2, return_X =False)# to return object 
    >>> plot_pca_components (pca)
    
    (2)-> use the components and features individually 
    
    >>> components = pca.components_ 
    >>> features = pca.feature_names_in_
    >>> plot_pca_components (components, feature_names= features, 
                             cmap='jet_r')
    
    """
    # if pca object is given , get the features name in as features 
    if hasattr(components, "feature_names_in_"): 
        feature_names = list (getattr (components , "feature_names_in_" ) ) 
        
    if not hasattr (components , "__array__"): 
        components = _check_array_in  (components, 'components_')
        
    plt.matshow(components, cmap =cmap , **kws)
    plt.yticks ([0 , 1], ['First component', 'Second component']) 
    cb=plt.colorbar() 
    cb.set_label('Coeff value')
    if not is_iterable(feature_names ): 
        feature_names = [feature_names ]
        
    if len(feature_names)!= components.shape [1] :
        warnings.warn("Number of feature and eigenvectors might"
                      " be consistent, expect {0}, got {1}". format( 
                          components.shape[1], len(feature_names))
                      )
        feature_names=None 
    if feature_names is not None: 
        plt.xticks (range (len(feature_names)), 
                    feature_names , rotation = 60 , ha='left' 
                    )
    plt.xlabel ("Feature") 
    plt.ylabel ("Principal components") 
    
        
def plot_clusters (
        n_clusters, X, ypred, cluster_centers =None 
        ): 
    """ Visualize the cluster that k-means identified in the datasets 
    
    :param n_clusters: int, number of cluster to visualize 
    :param X: NDArray, data containing the features, expect to be a two 
        dimensional data 
    :param ypred: array-like, array containing the predicted class labels. 
    :param cluster_centers_: NDArray containg the coordinates of the 
        centroids or the similar points with continous features. 
        
    :Example: 
    >>> from watex.utils import read_data 
    >>> from watex.exlib.sklearn import KMeans, MinMaxScaler
    >>> from watex.utils.plotutils import plot_clusters
    >>> h = read_data ('data/boreholes/H502.xlsx')
    >>> # collect two features 'resistivity' and gamma-gamma logging values
    >>> h2 = h[['resistivity', 'gamma-gamma']] 
    >>> km = KMeans (n_clusters =3 , init= 'random' ) 
    >>> # scaled the data with MinMax scaler i.e. between ( 0-1) 
    >>> h2_scaled = MinMaxScaler().fit_transform(h2)
    >>> ykm = km.fit_predict(h2_scaled )
    >>> plot_clusters (3 , h2_scaled, y_km , km.cluster_centers_ )
        
    """
    try : n_clusters = int(n_clusters )
    except: 
        raise TypeError (f"n_clusters argument must be a number, "
                         f"not {type(n_clusters).__name__!r}")
    X= np.array(X) 
    if len(X.shape )!=2 or X.shape[1]==1: 
        ndim = 1 if X.shape[1] ==1 else np.ndim (X )
        raise ValueError(
            f"X is expected to be a two dimensional data. Got {ndim}!")
    # for consistency , convert y to array    
    ypred = np.array(ypred)
    
    colors = make_mpl_properties(n_clusters)
    markers = make_mpl_properties(n_clusters, 'markers')
    for n in range (n_clusters):
        plt.scatter (X[ypred ==n, 0], 
                     X[ypred ==n , 1],  
                     s= 50 , c= colors [n ], 
                     marker=markers [n], 
                     edgecolors='black', 
                     label = f'Cluster {n +1}'
                     ) 
    if cluster_centers is not None: 
        cluster_centers = np.array (cluster_centers)
        plt.scatter (cluster_centers[:, 0 ], 
                     cluster_centers [:, 1], 
                     s= 250. , marker ='*', 
                     c='red', edgecolors='black', 
                     label='centroids' 
                     ) 
    plt.legend (scatterpoints =1 ) 
    plt.grid() 
    plt.tight_layout() 
    plt.show()
    

def plot_elbow (distorsions: list  , n_clusters:int ,fig_size = (10 , 4 ),  
               marker='o', savefig =None, **kwd): 
    """ Plot the optimal number of cluster, k', for a given class 
    
    :param distorsions: list - list of values withing the ssum-squared-error 
        (SSE) also called  `inertia_` in sckit-learn. 
    
    :param n_clusters: number of clusters. where k starts and end. 
    
    :returns: ax: Matplotlib.pyplot axes objects 
    
    :Example: 
    >>> import numpy as np 
    >>> from sklearn.cluster import KMeans 
    >>> from watex.datasets import load_iris 
    >>> from watex.utils.plotutils import plot_elbow
    >>> d= load_iris ()
    >>> X= d.data [:, 0][:, np.newaxis] # take the first axis 
    >>> # compute distorsiosn for KMeans range 
    >>> distorsions =[] ; n_clusters = 11
    >>> for i in range (1, n_clusters ): 
            km =KMeans (n_clusters =i , 
                        init= 'k-means++', 
                        n_init=10 , 
                        max_iter=300, 
                        random_state =0 
                        )
            km.fit(X) 
            distorsions.append(km.inertia_) 
    >>> plot_elbow (distorsions, n_clusters =n_clusters)
        
    """
    fig, ax = plt.subplots ( nrows=1 , ncols =1 , figsize = fig_size ) 
    
    ax.plot (range (1, n_clusters), distorsions , marker = marker, 
              **kwd )
    plt.xlabel ("Number of clusters") 
    plt.ylabel ("Distorsion")
    plt.tight_layout()
    
    if savefig is not None: 
        savefigure(fig, savefig )
    plt.show() if savefig is None else plt.close () 
    
    return ax 


def plot_cost_vs_epochs(regs, *,  fig_size = (10 , 4 ), marker ='o', 
                     savefig =None, **kws): 
    """ Plot the cost agianst the number of epochs  for the two different 
    learnings rates 
    
    Parameters 
    ----------
    regs: Callable, single or list of regression estimators 
        Estimator should be already fitted.
    fig_size: tuple , default is (10, 4)
        the size of figure 
    kws: dict , 
        Additionnal keywords arguments passes to :func:`matplotlib.pyplot.plot`
    Returns 
    ------- 
    ax: Matplotlib.pyplot axes objects 
    
    Examples 
    ---------
    >>> from watex.datasets import load_iris 
    >>> from watex.base import AdelineGradientDescent
    >>> from watex.utils.plotutils import plot_cost_vs_epochs
    >>> X, y = load_iris (return_X_y= True )
    >>> ada1 = AdelineGradientDescent (n_iter= 10 , eta= .01 ).fit(X, y) 
    >>> ada2 = AdelineGradientDescent (n_iter=10 , eta =.0001 ).fit(X, y)
    >>> plot_cost_vs_epochs (reg = [ada1, ada2] ) 

    """
    if not isinstance (regs, (list, tuple, np.array)): 
        regs =[regs]
    s = set ([hasattr(o, '__class__') for o in regs ])

    if len(s) != 1: 
        raise ValueError(" All regression should be an estimator already fitted.")
    if not list(s) [0] : 
        raise TypeError(f" Need an estimator, got {type(s[0]).__name__!r}")
    
    fig, ax = plt.subplots ( nrows=1 , ncols =len(regs) , figsize = fig_size ) 
    
    for k, m in enumerate (regs)  : 
        
        ax[k].plot(range(1, len(m.cost_)+ 1 ), np.log10 (m.cost_),
                   marker =marker, **kws)
        ax[k].set_xlabel ("Epochs") 
        ax[k].set_ylabel ("Log(sum-squared-error)")
        ax[k].set_title("%s -Learning rate %.4f" % (m.__class__.__name__, m.eta )) 
        
    if savefig is not None: 
        savefigure(fig, savefig )
    plt.show() if savefig is None else plt.close () 
    
    return ax 

    
def plot_mlxtend_heatmap (df, columns =None, **kws): 
    """ Plot correlation matrix array  as a heat map 
    
    :param df: dataframe pandas  
    :param columns: list of features, 
        If given, only the dataframe with that features is considered. 
    :param kws: additional keyword arguments passed to 
        :func:`mlxtend.plotting.heatmap`
    :return: :func:`mlxtend.plotting.heatmap` axes object 
    
    """
    cm = np.corrcoef(df[columns]. values.T)
    ax= heatmap(cm, row_names = columns , column_names = columns, **kws )
    plt.show () 
    
    return ax 

def plot_mlxtend_matrix(df, columns =None, fig_size = (10 , 8 ), alpha =.5 ):
    """ Visualize the pair wise correlation between the different features in  
    the dataset in one place. 
    
    :param df: dataframe pandas  
    :param columns: list of features, 
        If given, only the dataframe with that features is considered. 
    :param fig_size: tuple of int (width, heigh) 
        Size of the displayed figure 
    :param alpha: figure transparency, default is ``.5``. 
    
    :return: :func:`mlxtend.plotting.scatterplotmatrix` axes object 
    
    """
    if not is_mlxtend: 
        warnings.warn(" 'mlxtend' package is missing. Can plot scatter matrix."
                      " install it mannually via 'pip' or 'conda'.")
        return  ModuleNotFoundError("'mlextend' package is missing. Install it" 
                                    " using 'pip' or 'conda'")
    if isinstance (columns, str): 
        columns = [columns ] 
    try: 
        iter (columns)
    except : 
        raise TypeError(" Columns should be an iterable object, not"
                        f" {type (columns).__name__!r}")
    columns =list(columns)
    
    
    if columns is not None: 
        df =df[columns ] 
        
    ax = scatterplotmatrix (
        df[columns].values , figsize =fig_size,names =columns , alpha =alpha 
        )
    plt.tight_layout()

    plt.show () 
    
    return ax 

    
def savefigure (fig: object ,
             figname: str = None,
             ext:str  ='.png',
             **skws ): 
    """ save figure from the given figure name  
    
    :param fig: Matplotlib figure object 
    :param figname: name of figure to output 
    :param ext: str - extension of the figure 
    :param skws: Matplotlib savefigure keywards additional keywords arguments 
    
    :return: Matplotlib savefigure objects. 
    
    """
    ext = '.' + str(ext).lower().strip().replace('.', '')

    if figname is None: 
        figname =  '_' + os.path.splitext(os.path.basename(__file__)) +\
            datetime.datetime.now().strftime('%m-%d-%Y %H:%M:%S') + ext
        warnings.warn("No name of figure is given. Figure should be renamed as "
                      f"{figname!r}")
        
    file, ex = os.path.splitext(figname)
    if ex in ('', None): 
        ex = ext 
        figname = os.path.join(file, f'{ext}')

    return  fig.savefig(figname, **skws)


def resetting_ticks ( get_xyticks,  number_of_ticks=None ): 
    """
    resetting xyticks  modulo , 100
    
    :param get_xyticks:  xyticks list  , use to ax.get_x|yticks()
    :type  get_xyticks: list 
    
    :param number_of_ticks:  maybe the number of ticks on x or y axis 
    :type number_of_ticks: int
    
    :returns: a new_list or ndarray 
    :rtype: list or array_like 
    """
    if not isinstance(get_xyticks, (list, np.ndarray) ): 
        warnings.warn (
            'Arguments get_xyticks must be a list'
            ' not <{0}>.'.format(type(get_xyticks)))
        raise TipError (
            '<{0}> found. "get_xyticks" must be a '
            'list or (nd.array,1).'.format(type(get_xyticks)))
    
    if number_of_ticks is None :
        if len(get_xyticks) > 2 : 
            number_of_ticks = int((len(get_xyticks)-1)/2)
        else : number_of_ticks  = len(get_xyticks)
    
    if not(number_of_ticks, (float, int)): 
        try : number_of_ticks=int(number_of_ticks) 
        except : 
            warnings.warn('"Number_of_ticks" arguments is the times to see '
                          'the ticks on x|y axis.'\
                          ' Must be integer not <{0}>.'.
                          format(type(number_of_ticks)))
            raise PlotError(f'<{type(number_of_ticks).__name__}> detected.'
                            ' Must be integer.')
        
    number_of_ticks=int(number_of_ticks)
    
    if len(get_xyticks) > 2 :
        if get_xyticks[1] %10 != 0 : 
            get_xyticks[1] =get_xyticks[1] + (10 - get_xyticks[1] %10)
        if get_xyticks[-2]%10  !=0 : 
            get_xyticks[-2] =get_xyticks[-2] -get_xyticks[-2] %10
    
        new_array = np.linspace(get_xyticks[1], get_xyticks[-2],
                                number_of_ticks )
    elif len(get_xyticks)< 2 : 
        new_array = np.array(get_xyticks)
 
    return  new_array
        
def make_mpl_properties(n ,prop ='color'): 
    """ make matplotlib property ('colors', 'marker', 'line') to fit the 
    numer of samples
    
    :param n: int, 
        Number of property that is needed to create. It generates a group of 
        property items. 
    :param prop: str, default='color', name of property to retrieve. Accepts 
        only 'colors', 'marker' or 'line'.
    :return: list of property items with size equals to `n`.
    :Example: 
        >>> from watex.utils.plotutils import make_mpl_properties
        >>> make_mpl_properties (10 )
        ... ['g',
             'gray',
             'y',
             'blue',
             'orange',
             'purple',
             'lime',
             'k',
             'cyan',
             (0.6, 0.6, 0.6)]
        >>> make_mpl_properties(100 , prop = 'marker')
        >>> ['o',
             '^',
             'x',
             'D',
              .
              .
              .
             11,
             'None',
             None,
             ' ',
             '']
        >>> make_mpl_properties(50 , prop = 'line')
        ... ['-',
             '-',
             '--',
             '-.',
               .
               .
               . 
             'solid',
             'dashed',
             'dashdot',
             'dotted']
        
    """ 

    try: n= int(n)
    except: raise TypeError(f"Expect a number, got {type(n).__name__!r}")
    
    prop = str(prop).lower().strip().replace ('s', '') 
    if prop not in ('color', 'marker', 'line'): 
        raise ValueError ("Property {prop!r} is not availabe yet. , Expect"
                          " 'colors', 'marker' or 'line'.")
    # customize plots with colors lines and styles 
    # and create figure obj
    if prop=='color': 
        d_colors =  D_COLORS 
        d_colors = mpl.colors.ListedColormap(d_colors[:n]).colors
        if len(d_colors) == n: 
            props= d_colors 
        else:
            rcolors = list(itertools.repeat(
                d_colors , (n + len(d_colors))//len(d_colors))) 
    
            props  = list(itertools.chain(*rcolors))
        
    if prop=='marker': 
        
        d_markers =  D_MARKERS + list(mpl.lines.Line2D.markers.keys()) 
        rmarkers = list(itertools.repeat(
            d_markers , (n + len(d_markers))//len(d_markers))) 
        
        props  = list(itertools.chain(*rmarkers))
    # repeat the lines to meet the number of cv_size 
    if prop=='line': 
        d_lines =  D_STYLES
        rlines = list(itertools.repeat(
            d_lines , (n + len(d_lines))//len(d_lines))) 
        # combine all repeatlines 
        props  = list(itertools.chain(*rlines))
    
    return props [: n ]
       
def resetting_colorbar_bound(cbmax ,
                             cbmin,
                             number_of_ticks = 5, 
                             logscale=False): 
    """
    Function to reset colorbar ticks more easy to read 
    
    :param cbmax: value maximum of colorbar 
    :type cbmax: float 
    
    :param cbmin: minimum data value 
    :type cbmin: float  minimum data value
    
    :param number_of_ticks:  number of ticks should be 
                            located on the color bar . Default is 5.
    :type number_of_ticks: int 
    
    :param logscale: set to True if your data are lograith data . 
    :type logscale: bool 
    
    :returns: array of color bar ticks value.
    :rtype: array_like 
    """
    def round_modulo10(value): 
        """
        round to modulo 10 or logarithm scale  , 
        """
        if value %mod10  == 0 : return value 
        if value %mod10  !=0 : 
            if value %(mod10 /2) ==0 : return value 
            else : return (value - value %mod10 )
    
    if not(number_of_ticks, (float, int)): 
        try : number_of_ticks=int(number_of_ticks) 
        except : 
            warnings.warn('"Number_of_ticks" arguments '
                          'is the times to see the ticks on x|y axis.'
                          ' Must be integer not <{0}>.'.format(
                              type(number_of_ticks)))
            raise TipError('<{0}> detected. Must be integer.')
        
    number_of_ticks=int(number_of_ticks)
    
    if logscale is True :  mod10 =np.log10(10)
    else :mod10 = 10 
       
    if cbmax % cbmin == 0 : 
        return np.linspace(cbmin, cbmax , number_of_ticks)
    elif cbmax% cbmin != 0 :
        startpoint = cbmin + (mod10  - cbmin % mod10 )
        endpoint = cbmax - cbmax % mod10  
        return np.array([round_modulo10(ii) for
                         ii in np.linspace(startpoint, 
                                           endpoint, number_of_ticks)])
    

            
def controle_delineate_curve(res_deline =None , phase_deline =None ): 
    """
    fonction to controle delineate value given  and return value ceilling .
    
    :param  res_deline:  resistivity  value todelineate. unit of Res in `ohm.m`
    :type  res_deline: float|int|list  
    
    :param  phase_deline:   phase value to  delineate , unit of phase in degree
    :type phase_deline: float|int|list  
    
    :returns: delineate resistivity or phase values 
    :rtype: array_like 
    """
    fmt=['resistivity, phase']
 
    for ii, xx_deline in enumerate([res_deline , phase_deline]): 
        if xx_deline is  not None  : 
            if isinstance(xx_deline, (float, int, str)):
                try :xx_deline= float(xx_deline)
                except : raise TipError(
                        'Value <{0}> to delineate <{1}> is unacceptable.'\
                         ' Please ckeck your value.'.format(xx_deline, fmt[ii]))
                else :
                    if ii ==0 : return [np.ceil(np.log10(xx_deline))]
                    if ii ==1 : return [np.ceil(xx_deline)]
  
            if isinstance(xx_deline , (list, tuple, np.ndarray)):
                xx_deline =list(xx_deline)
                try :
                    if ii == 0 : xx_deline = [
                            np.ceil(np.log10(float(xx))) for xx in xx_deline]
                    elif  ii ==1 : xx_deline = [
                            np.ceil(float(xx)) for xx in xx_deline]
                        
                except : raise TipError(
                        'Value to delineate <{0}> is unacceptable.'\
                         ' Please ckeck your value.'.format(fmt[ii]))
                else : return xx_deline


def fmt_text (data_text, fmt='~', leftspace = 3, return_to_line =77) : 
    """
    Allow to format report with data text , fm and leftspace 

    :param  data_text: a long text 
    :type  data_text: str  
        
    :param fmt:  type of underline text 
    :type fmt: str

    :param leftspae: How many space do you want before starting wrinting report .
    :type leftspae: int 
    
    :param return_to_line: number of character to return to line
    :type return_to_line: int 
    """

    return_to_line= int(return_to_line)
    begin_text= leftspace *' '
    text= begin_text + fmt*(return_to_line +7) + '\n'+ begin_text

    
    ss=0
    
    for  ii, num in enumerate(data_text) : # loop the text 
        if ii == len(data_text)-1 :          # if find the last character of text 
            #text = text + data_text[ss:] + ' {0}\n'.format(fmt) # take the 
            #remain and add return chariot 
            text = text+ ' {0}\n'.format(fmt) +\
                begin_text +fmt*(return_to_line+7) +'\n' 
      
 
            break 
        if ss == return_to_line :                       
            if data_text[ii+1] !=' ' : 
                text = '{0} {1}- \n {2} '.format(
                    text, fmt, begin_text + fmt ) 
            else : 
                text ='{0} {1} \n {2} '.format(
                    text, fmt, begin_text+fmt ) 
            ss=0
        text += num    # add charatecter  
        ss +=1

    return text 


# Plotting functions

def plotvec1(u, z, v):
    """
    Plot tips function with  three vectors. 
    
    :param u: vector u - a vector 
    :type u: array like  
    
    :param z: vector z 
    :type z: array_like 
    
    :param v: vector v 
    :type v: array_like 
    
    return: plot 
    
    """
    
    ax = plt.axes()
    ax.arrow(0, 0, *u, head_width=0.05, color='r', head_length=0.1)
    plt.text(*(u + 0.1), 'u')
    
    ax.arrow(0, 0, *v, head_width=0.05, color='b', head_length=0.1)
    plt.text(*(v + 0.1), 'v')
    ax.arrow(0, 0, *z, head_width=0.05, head_length=0.1)
    plt.text(*(z + 0.1), 'z')
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)

def plotvec2(a,b):
    """
    Plot tips function with two vectors
    Just use to get the orthogonality of two vector for other purposes 

    :param a: vector u 
    :type a: array like  - a vector 
    :param b: vector z 
    :type b: array_like 
    
    *  Write your code below and press Shift+Enter to execute
    
    :Example: 
        
        >>> import numpy as np 
        >>> from watex.utils.plotutils import plotvec2
        >>> a=np.array([1,0])
        >>> b=np.array([0,1])
        >>> Plotvec2(a,b)
        >>> print('the product a to b is =', np.dot(a,b))

    """
    ax = plt.axes()
    ax.arrow(0, 0, *a, head_width=0.05, color ='r', head_length=0.1)
    plt.text(*(a + 0.1), 'a')
    ax.arrow(0, 0, *b, head_width=0.05, color ='b', head_length=0.1)
    plt.text(*(b + 0.1), 'b')
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)  

def ploterrorbar(
        ax,
        x_array,
        y_array,
        y_error=None,
        x_error=None,
        color='k',
        marker='x',
        ms=2, 
        ls=':', 
        lw=1, 
        e_capsize=2,
        e_capthick=.5,
        picker=None,
        **kws
 )-> object:
    """
    convinience function to make an error bar instance
    
    Parameters
    ------------
    
    ax: matplotlib.axes 
        instance axes to put error bar plot on

    x_array: np.ndarray(nx)
        array of x values to plot
                  
    y_array: np.ndarray(nx)
        array of y values to plot
                  
    y_error: np.ndarray(nx)
        array of errors in y-direction to plot
    
    x_error: np.ndarray(ns)
        array of error in x-direction to plot
                  
    color: string or (r, g, b)
        color of marker, line and error bar
                
    marker: string
        marker type to plot data as
                 
    ms: float
        size of marker
             
    ls: string
        line style between markers
             
    lw: float
        width of line between markers
    
    e_capsize: float
        size of error bar cap
    
    e_capthick: float
        thickness of error bar cap
    
    picker: float
          radius in points to be able to pick a point. 
        
        
    Returns:
    ---------
    errorbar_object: matplotlib.Axes.errorbar 
           error bar object containing line data, errorbars, etc.
    """
    # this is to make sure error bars 
    #plot in full and not just a dashed line
    if x_error is not None:
        x_err = x_error
    else:
        x_err = None

    if y_error is not None:
        y_err = y_error
    else:
        y_err = None

    eobj = ax.errorbar(
        x_array,
        y_array,
        marker=marker,
        ms=ms,
        mfc='None',
        mew=lw,
        mec=color,
        ls=ls,
        xerr=x_err,
        yerr=y_err,
        ecolor=color,
        color=color,
        picker=picker,
        lw=lw,
        elinewidth=lw,
        capsize=e_capsize,
        # capthick=e_capthick
        **kws
         )
    
    return eobj

def get_color_palette (RGB_color_palette): 
    """
    Convert RGB color into matplotlib color palette. In the RGB color 
    system two bits of data are used for each color, red, green, and blue. 
    That means that each color runson a scale from 0 to 255. Black  would be
    00,00,00, while white would be 255,255,255. Matplotlib has lots of
    pre-defined colormaps for us . They are all normalized to 255, so they run
    from 0 to 1. So you need only normalize data, then we can manually  select 
    colors from a color map  

    :param RGB_color_palette: str value of RGB value 
    :type RGB_color_palette: str 
        
    :returns: rgba, tuple of (R, G, B)
    :rtype: tuple
     
    :Example: 
        
        >>> from watex.utils.plotutils import get_color_palette 
        >>> get_color_palette (RGB_color_palette ='R128B128')
    """  
    
    def ascertain_cp (cp): 
        if cp >255. : 
            warnings.warn(
                ' !RGB value is range 0 to 255 pixels , '
                'not beyond !. Your input values is = {0}.'.format(cp))
            raise ValueError('Error color RGBA value ! '
                             'RGB value  provided is = {0}.'
                            ' It is larger than 255 pixels.'.format(cp))
        return cp
    if isinstance(RGB_color_palette,(float, int, str)): 
        try : 
            float(RGB_color_palette)
        except : 
              RGB_color_palette= RGB_color_palette.lower()
             
        else : return ascertain_cp(float(RGB_color_palette))/255.
    
    rgba = np.zeros((3,))
    
    if 'r' in RGB_color_palette : 
        knae = RGB_color_palette .replace('r', '').replace(
            'g', '/').replace('b', '/').split('/')
        try :
            _knae = ascertain_cp(float(knae[0]))
        except : 
            rgba[0]=1.
        else : rgba [0] = _knae /255.
        
    if 'g' in RGB_color_palette : 
        knae = RGB_color_palette .replace('g', '/').replace(
            'b', '/').split('/')
        try : 
            _knae =ascertain_cp(float(knae[1]))
        except : 
            rgba [1]=1.
            
        else :rgba[1]= _knae /255.
    if 'b' in RGB_color_palette : 
        knae = knae = RGB_color_palette .replace('g', '/').split('/')
        try : 
            _knae =ascertain_cp(float(knae[1]))
        except :
            rgba[2]=1.
        else :rgba[2]= _knae /255.
        
    return tuple(rgba)       

def _get_xticks_formatage ( ax,  xtick_range, space= 14 ):
    """ Skip xticks label at every number of spaces 
    :param ax: matplotlib axes 
    :param xtick_range: list of the xticks values 
    :param space: interval that the label must be shown.
    """
    def format_ticks (ind, x):
        """ Format thick parameter with 'FuncFormatter(func)'
        rather than using:: 
            
        axi.xaxis.set_major_locator (plt.MaxNLocator(3))
        
        ax.xaxis.set_major_formatter (plt.FuncFormatter(format_thicks))
        """
        if ind % 7 ==0: 
            return '{}'.format (ind)
        else: None 
    # show label every 'space'samples 
    if len(xtick_range) >= space : 
        ax.xaxis.set_major_formatter (plt.FuncFormatter(format_ticks))   

    
    

    
    
    
    