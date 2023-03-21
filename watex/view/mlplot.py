# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
:mod:`~watex.view.mlplot` is a set of plot templates for visualising and 
inspecting the learning models.  It gives a quick depiction for users for 
models visualization and evaluation with : :class:`~watex.view.EvalPlot`
"""
from __future__ import annotations 
import re
import warnings
import inspect 
from abc import ABCMeta 
import copy 
import numpy as np 
import pandas as pd
import seaborn as sns 
from scipy.cluster.hierarchy import dendrogram 

import matplotlib as mpl 
import matplotlib.pyplot  as plt
import matplotlib.ticker as mticker
from matplotlib import cm 
from matplotlib.colors import BoundaryNorm

from .._watexlog import watexlog
from .._docstring import ( 
    _core_docs, 
    _baseplot_params, 
    DocstringComponents, 
    )
from ..analysis.dimensionality import nPCA
from ..decorators import  docSanitizer 
from ..exlib.sklearn import  ( 
    learning_curve , 
    silhouette_samples , 
    SimpleImputer, 
    StandardScaler, 
    MinMaxScaler, 
    train_test_split, 
    mean_squared_error, 
    KMeans
    ) 
from ..exceptions import ( 
    NotFittedError , 
    LearningError, 
    EstimatorError, 
    PlotError
    )
from ..metrics import ( 
    precision_recall_tradeoff, 
    ROC_curve, 
    confusion_matrix
    )
from ..property import BasePlot 
from .._typing import ( 
    Optional, 
    Tuple, 
    F,
    List,
    ArrayLike, 
    NDArray,
    DataFrame, 
    Series
    )
from ..utils.exmath import linkage_matrix 
from ..utils.hydroutils import check_flow_objectivity 
from ..utils.coreutils import _is_readable 
from ..utils.funcutils import ( 
    is_iterable,
    reshape, 
    to_numeric_dtypes, 
    smart_strobj_recognition, 
    repr_callable_obj , 
    str2columns, 
    make_ids
    )
from ..utils.mlutils import ( 
    exporttarget , 
    selectfeatures, 
    cattarget, 
    projection_validator, 
    )
from ..utils.plotutils import (
    _get_xticks_formatage, 
    # _format_ticks, 
    make_mpl_properties, 
    
    )
from ..utils.validator import ( 
    _check_consistency_size, 
    get_estimator_name , 
    array_to_frame, 
    check_array, 
    check_X_y, 
    check_y,
    )

_logger=watexlog.get_watex_logger(__name__)

#-----
# Add specific params to Evaldocs 

_eval_params = dict( 
    objective="""
objective: str, default=None, 
    The purpose of dataset; what probem do we intend to solve ?  
    Originally the package was designed for flow rate prediction. Thus,  
    if the `objective` is set to ``flow``, plot will behave like the flow 
    rate prediction purpose and in that case, some condition of target   
    values need to be fullfilled.  Furthermore, if the objective 
    is set to ``flow``, `label_values`` as well as the `litteral_classes`
    parameters need to be supplied to right encode the target according 
    to the hydraulic system requirement during the campaign for drinking 
    water supply. For any other purpose for the dataset, keep the objective  
    to ``None``. Default is ``None``.    
    """, 
    yp_ls="""
yp_ls: str, default='-', 
    Line style of `Predicted` label. Can be [ '-' | '.' | ':' ] 
    """, 
    yp_lw="""
yp_lw: str, default= 3
    Line weight of the `Predicted` plot
    """,
    yp_lc ="""
yp_lc: str or :func:`matplotlib.cm`, default= 'k'
    Line color of the `Prediction` plot. *default* is ``k``
    """, 
    yp_marker="""
yp_marker: str or :func:`matplotlib.markers`, default ='o'
    Style of marker in  of `Prediction` points. 
    """, 
    yp_markerfacecolor="""
yp_markerfacecolor: str or :func:`matplotlib.cm`, default='k'
    Facecolor of the `Predicted` label marker.
    """, 
    yp_markeredgecolor="""
yp_markeredgecolor: stror :func:`matplotlib.cm`,  default= 'r' 
    Edgecolor of the `Predicted` label marker.
    """, 
    yp_markeredgewidth="""
yp_markeredgewidth: int, default=2
    Width of the `Predicted`label marker.
    """, 
    rs="""
rs: str, default='--'
    Line style of `Recall` metric 
    """, 
    ps="""
ps: str, default='-'
    Line style of `Precision `metric
    """, 
    rc="""
rc: str, default=(.6,.6,.6)
    Recall metric colors 
    """, 
    pc="""
pc: str or :func:`matplotlib.cm`, default='k'
    Precision colors from Matplotlib colormaps. 
    """
    )

_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"], 
    base=DocstringComponents(_baseplot_params), 
    evdoc=DocstringComponents(_eval_params), 
    )
#-------

class EvalPlot(BasePlot): 
    def __init__(self, 
                 tname:str =None, 
                 encode_labels: bool=False,
                 scale: str = None, 
                 cv: int =None, 
                 objective:str=None, 
                 prefix: str=None, 
                 label_values:List[int]=None, 
                 litteral_classes: List[str]=None, 
                 **kws 
                 ): 
        self._logging= watexlog().get_watex_logger(self.__class__.__name__)
        
        self.tname=tname
        self.objective=objective
        self.scale=scale
        self.cv=cv
        self.prefix=prefix 
        self.encode_labels=encode_labels 
        self.litteral_classes=litteral_classes 
        self.label_values=label_values 
        # precision(p) and recall(r) 
        # properties
        self.rs =kws.pop('rs', '--')
        self.ps =kws.pop('ps', '-')
        self.rc =kws.pop('rc', (.6, .6, .6))
        self.pc =kws.pop('pc', 'k')
        # predicted properties 
        self.yp_lc =kws.pop('yp_lc', 'k') 
        self.yp_marker= kws.pop('yp_marker', 'o')
        self.yp_marker_edgecolor = kws.pop('yp_markeredgecolor', 'r')
        self.yp_lw = kws.pop('yp_lw', 3.)
        self.yp_ls=kws.pop('yp_ls', '-')
        self.yp_marker_facecolor =kws.pop('yp_markerfacecolor', 'k')
        self.yp_marker_edgewidth= kws.pop('yp_markeredgewidth', 2.)
        
        super().__init__(**kws) 
        
        self.data_ =None 
        self.X=None 
        self.y= None 
        self.target_=None 
        
     
    @property 
    def inspect(self): 
        """ Inspect data and trigger plot after checking the data entry. 
        Raises `NotFittedError` if `ExPlot` is not fitted yet."""
        
        msg = ( "{expobj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if self.X is None: 
            raise NotFittedError(msg.format(
                expobj=self)
            )
        return 1 
     
    def save (self, fig): 
        """ savefigure if figure properties are given. """
        if self.savefig is not None: 
            fig.savefig (self.savefig,dpi = self.fig_dpi , 
                         bbox_inches = 'tight', 
                         orientation=self.fig_orientation 
                         )
        plt.show() if self.savefig is None else plt.close () 
        
    def fit(self, X: NDArray |DataFrame =None, y:ArrayLike =None, 
            **fit_params ): 
        """
        Fit data and populate the attributes for plotting purposes. 
        
        There is no conventional procedure for checking if a method is fitted. 
        However, an class that is not fitted should raise 
        :class:`watex.exceptions.NotFittedError` when a method is called.
        
        Parameters
        ------------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.
        y: array-like, shape (M, ) ``M=m-samples``, 
            train target; Denotes data that may be observed at training time 
            as the dependent variable in learning, but which is unavailable 
            at prediction time, and is usually the target of prediction. 
            
        data: Filepath or Dataframe or shape (M, N) from 
            :class:`pandas.DataFrame`. Dataframe containing samples M  
            and features N

        fit_params: dict Additional keywords arguments from 
            :func:watex.utils.coreutils._is_readable`
           
        Return
        -------
        ``self``: `EvalPlot` instance 
            returns ``self`` for easy method chaining.
        """
        data = fit_params.pop('data', None)
        columns = fit_params.pop ('columns', None)
        
        if data is not None: 
            self.data_ = _is_readable(data)
            
        if self.data_ is not None:
            if self.tname is not None: 
                self.target_, X  = exporttarget(
                    self.data_ , self.tname, inplace= True ) 
            y = reshape (self.target_.values ) # for consistency 
            
        if X is None:
            raise TypeError(
                "X array must not be None, or pass a filepath or "
                "dataframe object as keyword data argument to set 'X'.")
        # Create a pseudo frame"
        # if 'X' is not a dataframe
        X= array_to_frame(X, to_frame= True, input_name="X", force =True )
        X = to_numeric_dtypes(X , columns = columns )
        X = selectfeatures( X, include ='number')
        
        if len ( X.columns) ==0 : 
            raise TypeError(
                " The module {self.__class__.__name__!r } expects dataframe "
                " 'X' with numerical features only. ")
            
        self.X = X 
        self.y = np.array (y) 
        
        return self 
    
    def transform (self, X, **t_params): 
        """ Transform the data and imputs the numerical features. 
        
        It is not convenient to use `transform` if user want to keep 
        categorical values in the array 
        
        Parameters
        ------------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.
            
        t_params: dict, 
            Keyword arguments passed to :class:`sklearn.impute.SimpleImputer` 
            for imputing the missing data; default strategy is 'most_frequent'
            or keywords arguments passed to
            :func:watex.utils.funcutils.to_numeric_dtypes`
            
        Return
        -------
        X: NDArray |Dataframe , shape (M x N )
            The transformed array or dataframe with numerical features 
            
        """
        self.X = X 
        self.inspect 
        strategy = t_params.pop('strategy', 'most_frequent')
        columns = list(self.X.columns )

        imp = SimpleImputer(strategy = strategy,  **t_params ) 
        # create new dataframe 
        X= imp.fit_transform(self.X )
        if self.scale: 
            if str(self.scale).find ('minmax') >=0 : 
                sc = MinMaxScaler() 
                
            else:sc =StandardScaler()
            
            X = sc.fit_transform(X)
            
        self.X = pd.DataFrame( X , columns = columns ) 
        
        return self.X 
    
    def fit_transform (self, X, y= None , **fit_params ): 
        """ Fit and transform at once. 
        
        Parameters
        ------------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.
            
        Return
        -------
        X: NDArray |Dataframe , shape (M x N )
            The transformed array or dataframe with numerical features 
        
        """
        self.X = self.fit(X, y, **fit_params).transform(self.X )
        
        return self.X 
   
     
    def _cat_codes_y (self, prefix:str =None ,values:List[int]=None, 
                      classes: List[str]=None,  objective:str =None ): 
        """ Encode y to hold the categorical values. 
        
        Note that if objective is set to 'flow', the `values` need to  be 
        supplied, otherwise an error will raises. 
        
        :param values: list of values to encoding the numerical target `y`. 
            for instance ``values=[0, 1, 2]`` 
        :param objective: str, relate to the flow rate prediction. Set to 
            ``None`` for any other predictions. 
        :param prefix: the prefix to add to the class labels. For instance, if 
            the `prefix` equals to ``FR``, class labels will become:: 
                
                [0, 1, 2] => [FR0, FR1, FR2]
                
        :classes: list of classes names to replace the default `FR` that is 
            used to specify the flow rate. For instance, it can be:: 
                
                [0, 1, 2] => [sf0, sf1, sf2]
        Returns 
        --------
        (self.y, classes): Array-like, list[int|str]
            Array of encoded labels and list of unique class label identifiers 
            
        """

        y =copy.deepcopy(self.y) 
        
        if y is None : 
            warnings.warn("Expect a target array. Missing y(target)"
                          " is not allowed.")
            raise TypeError (" NoneType 'y' (target) can be categorized.")
            
        if objective =='flow':
            y, classes = check_flow_objectivity(y,values, classes) 
        else : 
            
            if self.target_ is not None: 
                y = self.target_ 
            else: y = pd.Series (y, name='none')
            
            values = values or self.label_values 
            if values is not None: 
                y =  cattarget(y , labels = values, 
                               rename_labels= classes or self.litteral_classes
                               )
            else: 
                y = y.astype('category').cat.codes
                
            # add prefix 
            y = y.map(lambda o: prefix + str(o) 
                                ) if prefix else y 
                
            classes = np.unique (y)
            
        return y , classes 


    def plotPCA(
            self,
            n_components:int =None, 
            *, 
            n_axes: int= None, #2,
            biplot:bool =False, 
            pc1_label:str ='Axis 1',
            pc2_label:str='Axis 2',
            plot_dict:dict= None,
            **pca_kws
    )->'EvalPlot': 
        """ Plot PCA component analysis using :class:`~.sklearn.decomposition`. 
        
        PCA identifies the axis that accounts for the largest amount of 
        variance in the train set `X`. It also finds a second axis orthogonal 
        to the first one, that accounts for the largest amount of remaining 
        variance.
        
        Parameters 
        -----------
        n_components: Number of dimension to preserve. If`n_components` 
                is ranged between float 0. to 1., it indicates the number of 
                variance ratio to preserve. If ``None`` as default value 
                the number of variance to preserve is ``95%``.
                
        n_axes: Number of importance components to retrieve the 
            variance ratio. Default is ``2``. The first two importance 
            components with most variance ratio.
            
        biplot: bool, 
            biplot plots PCA features importance (pc1 and pc2) and visualize 
            the level of variance and direction of components for different 
            variables. Refer to `Serafeim Loukas`_
            
        pc1_label:str, default ='Axis 1'
            the first component with most variance held in 'Axis 1'. Can be 
            modified to any other axis for instance 'Axis 3' to replace the 
            component in 'Axis 1' to the one in `Axis 3` and so one. This will 
            allow to visualize the position of each level of variance 
            for each variable. 
            
        pc2_label:str, default ='Axis 2',
            the second component with most variance held in 'Axis 2'. Can be 
            modified to any other axis for instance 'Axis 6' to replace the 
            component in 'Axis 2' to the one in `Axis 6` and so one. 
         
        plot_dict: dict, 
            dictionnary of font and properties for markers for each sample 
            corresponding to the `label_values`.
        
        pca_kws: dict, 
            additional  keyword arguments passed to 
            :class:`watex.analysis.dimensionality.nPCA`

        Return
        -------
        ``self``: `EvalPlot` instance
            ``self`` for easy method chaining.
             
        Notes 
        -------
        By default, `nPCA` methods plots the first two principal components 
        named `pc1_label` for axis 1 and `pc2_label` for axis 2. If you want 
        to plot the first component `pc1` vs the third components`pc2` set 
        the `pc2_label` to `Axis 3` and set the `n_components` to 3 that is 
        the max reduced columns to retrieve, otherwise an users warning will 
        be displayed.  Commonly Algorithm should automatically detect the 
        digit ``3`` in the litteral `pc1_labels` including Axis (e.g. 'Axis 3`)
        and will consider as  the third component `pc3 `. The same process is 
        available for other axis. 
        
        
        Examples 
        ---------
        >>> from watex.datasets import load_bagoue 
        >>> from watex.view.mlplot import EvalPlot 
        >>> X , y = load_bagoue(as_frame =True )
        >>> b=EvalPlot(tname ='flow', encode_labels=True ,
                          scale = True )
        >>> b.fit_transform (X, y)
        >>> b.plotPCA (n_components= 2 )
        ... 
        >>> # pc1 and pc2 labels > n_components -> raises user warnings
        >>> b.plotPCA (n_components= 2 , biplot=False, pc1_label='Axis 3',
                       pc2_label='axis 4')
        ... UserWarning: Number of components and axes might be consistent;
            '2'and '4 are given; default two components are used.
        >>> b.plotPCA (n_components= 8 , biplot=False, pc1_label='Axis3',
                       pc2_label='axis4')
            # works fine since n_components are greater to the number of axes
        ... EvalPlot(tname= None, objective= None, scale= True, ... , 
                     sns_height= 4.0, sns_aspect= 0.7, verbose= 0)
        """
        self.inspect 
        
        classes , y  = self.litteral_classes, self.y 
        classes = classes or np.unique (y)
        
        if plot_dict is None: 
            D_COLORS = make_mpl_properties(1e3)
            plot_dict ={'y_colors': D_COLORS,
                        's':100.}
            
        if self.encode_labels: 
            y, classes = self._cat_codes_y(
                self.prefix, self.label_values, self.litteral_classes, 
                self.objective 
                )
        # go for PCA analysis 
        pca= nPCA(self.X, n_components, n_axes =n_axes, return_X= False, 
                            **pca_kws)
        feature_importances_ = pca.feature_importances_
        X_reduced = pca.X 
        # for consistency
        # Get axis for plots from pca_labels
        n_axes = n_axes or pca.n_axes 
        
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
 
            if (pca1_ix >= n_axes) or  (pca2_ix >= n_axes)  : 
                warnings.warn(
                    "Number of components and axes might be"
                    f" consistent; '{n_axes!r}'and '{max(lbls)!r}"
                    " are given; default two components are used."
                                )
                pca1_ix =0
                pca2_ix = pca1_ix+1
                pc1_label , pc2_label = 'Axis 1', 'Axis 2'
                
            # if pca1_ix  or pca2
            X_= np.c_[X_reduced[:, pca1_ix],
                      X_reduced[:, pca2_ix]]
            
        # prepared defaults colors and defaults markers 
        y_palettes = plot_dict ['y_colors']
        if classes  is not None:
            if len(y_palettes) > len(classes): 
                # reduce the last colors 
                y_palettes =y_palettes[:len(classes)]
            if len(y_palettes) < len(classes): 
                # add black colors  by default
                y_palettes += ['k' for k in range(
                    len(classes) - len(y_palettes))]
            
            
        # --Plot Biplot
        if biplot: 
            
            mpl.rcParams.update(mpl.rcParamsDefault) 
            # reset ggplot style
            # Call the biplot function for only the first 2 PCs
            cmp_= np.concatenate((pca.components_[pca1_ix, :], 
                                  pca.components_[pca2_ix, :]))
            try: 
                biPlot(self, self.X, np.transpose(cmp_), y,
                        classes=classes, 
                        colors=y_palettes )
            except : 
                # plot defaults configurations  
                biPlot(self, X_reduced[:,:2],
                        np.transpose(pca.components_[0:2, :]),
                        y, 
                        classes=classes, 
                        colors=y_palettes )
                plt.show()
            else : 
                plt.show()
            
            return  
        # concatenate reduced dataframe + y_target
        try: 
            df_pca =pd.concat([
                    pd.DataFrame(X_,columns =[pc1_label, pc2_label]),
                    pd.Series(y, name=self.tname)],
                axis =1)
        except TypeError: 
            # force plot using the defauts first two componnets if 
            # something goes wrong
             df_pca =pd.concat([
                    pd.DataFrame(X_reduced[:,:2],
                                 columns =[pc1_label, pc2_label]),
                    pd.Series(y, name=self.tname)],
                axis =1)
             pca1_ix , pca2_ix =0,1
      
        # Extract the name of the first components 
        # and second components
        pca_axis_1 = feature_importances_[pca1_ix][1][0] 
        pca_axis_2 = feature_importances_[pca2_ix][1][0]
        # Extract the name of the  values of the first 
        # component and second components in percentage.
        pca_axis_1_ratio = np.around(
            abs(feature_importances_[pca1_ix][2][0]),2) *1e2
        pca_axis_2_ratio = np.around(
            abs(feature_importances_[pca2_ix][2][0]),2) *1e2
     
        # create figure obj 
        fig = plt.figure(figsize = self.fig_size)
        ax = fig.add_subplot(1,1,1)
        
        y_label = df_pca.iloc [:, -1].name # get the name of label
        
        for target , color in zip(classes, y_palettes): 
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
                           marker = self.marker,
                           markeredgecolor = self.marker_edgecolor,
                           markeredgewidth = self.marker_edgewidth,
                           markerfacecolor = self.marker_facecolor ,
                           markersize = self.ms * self.fs
                           )
        
        lineh =plt.Line2D ((-max_lim, max_lim), (0, 0),
                           color = self.lc, 
                           linewidth = self.lw,
                           linestyle = self.ls ,
                           marker = self.marker,
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
        ax.legend(classes)
        ax.grid(color=self.lc,
                linestyle=self.ls,
                linewidth=self.lw/10
                )
        
        self.save(fig)
    
        return self 

    @docSanitizer()       
    def plotPR(
        self,
        clf:F,
        label:int|str,
        kind:Optional[str]=None,
        method:Optional[str]=None,
        cvp_kws =None,
        **prt_kws
    )->'EvalPlot': 
        """ 
        Precision/recall (PR) and tradeoff plots. 
        
        PR computes a score based on the decision function and plot the result
        as a score vs threshold. 
        
        Parameters
        -----------
        clf :callable, always as a function, classifier estimator
            A supervised predictor with a finite set of discrete possible 
            output values. A classifier must supports modeling some of binary, 
            targets. It must store a classes attribute after fitting.
        label: int, 
            Specific class to evaluate the tradeoff of precision 
            and recall. `label`  needs to be specified and a value within the 
            target.     
            
        kind: str, ['threshold|'recall'], default='threshold' 
            kind of PR plot. If kind is 'recall', method plots the precision 
            VS the recall scores, otherwiwe the PR tradeoff is plotted against 
            the 'threshold.'
            
        method: str
            Method to get scores from each instance in the trainset. 
            Could be ``decison_funcion`` or ``predict_proba``. When using the  
            scikit-Learn classifier, it generally has one of the method. 
            Default is ``decision_function``.   
        
        cvp_kws: dict, optional
            The :func:`sklearn.model_selection.cross_val_predict` keywords 
            additional arguments 
            
        prt_kws:dict, 
            Additional keyword arguments passed to 
            func:`watex.exlib.sklearn.precision_recall_tradeoff`
            
        Return
        -------
        ``self``: `EvalPlot` instance
            ``self`` for easy method chaining.
             
        Examples
        ---------
        >>> from watex.exlib.sklearn import SGDClassifier
        >>> from watex.datasets.dload import load_bagoue 
        >>> from watex.utils import cattarget 
        >>> from watex.view.mlplot import EvalPlot 
        >>> X , y = load_bagoue(as_frame =True )
        >>> sgd_clf = SGDClassifier(random_state= 42) # our estimator 
        >>> b= EvalPlot(scale = True , encode_labels=True)
        >>> b.fit_transform(X, y)
        >>> # binarize the label b.y 
        >>> ybin = cattarget(b.y, labels= 2 ) # can also use labels =[0, 1]
        >>> b.y = ybin 
        >>> # plot the Precision-recall tradeoff  
        >>> b.plotPR(sgd_clf , label =1) # class=1
        ... EvalPlot(tname= None, objective= None, scale= True, ... , 
                     sns_height= 4.0, sns_aspect= 0.7, verbose= 0)

        """
        msg = ("Precision recall metric works for classification "
                "task; labels must be encoded refering to a particular"
                " class; set 'encode_labels' param to 'True' and "
                "provided a list of class integer unique identifier."
                      )
        
        kind = kind or 'threshold'
        kind=str(kind).lower().strip() 
        
        if kind.lower().find('thres')>=0: 
            kind = 'threshold' 
        elif kind.lower().find('rec')>=0: 
            kind = 'recall'
            
        if kind not in ('threshold', 'recall'): 
            raise ValueError ("Invalid kind={0!r}. Expect {1!r} or {2!r}".
                              format(kind, *('threshold', 'recall'))
                )
    
        self.inspect 
        # call precision 
        if self.y is None: 
            warnings.warn("Precision-recall deals with supervising learning"
                          " methods which expects a target to be categorized."
                          " Missing target is not allowed.")
            raise TypeError("Missing target 'y' is not allowed. Can not used"
                            " the 'precision-recall' metric.")
            
        if not self.encode_labels : 
            warnings.warn(
                msg + " Refer to <https://en.wikipedia.org/wiki/Machine_learning>"
                " for deep understanding."    
                )
            raise LearningError (msg)
            
        prtObj = precision_recall_tradeoff( 
            clf, self.X,  self.y, cv =self.cv, 
            label=label, method =method, 
            cvp_kws=cvp_kws,
            **prt_kws)
        
        # create figure obj 
        fig = plt.figure(figsize = self.fig_size)
        ax = fig.add_subplot(1,1,1)

        # for consistency set xlabel and ylabel 
        xlabel = None 
        ylabel = None 
        if kind=='threshold': 
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
            
            xlabel = self.xlabel or 'Threshold'
            ylabel =self.ylabel or 'Score'


        elif kind =='recall': 
            ax.plot(prtObj.recalls[:-1],
                    prtObj.precisions[:-1], 
                    color = self.lc, 
                    linewidth = self.lw,
                    linestyle = self.ls , 
                    label = 'Precision vs Recall',
                    **self.plt_kws )
        
            
            xlabel = self.xlabel or 'Recall'
            ylabel =self.ylabel or 'Precision'
 
            self.xlim =[0,1]
            
        ax.set_xlabel( xlabel,
                      fontsize= .5 * self.font_size * self.fs )
        ax.set_ylabel (ylabel,
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
        if kind =='recall':
            ax.set_xlim (self.xlim)

        self.save(fig)
            
        return self 
    
    def plotROC(
        self, 
        clfs,
        label: int |str, 
        method: Optional[str]=None,
        cvp_kws:dict=None,
        **roc_kws
        )-> 'EvalPlot':
        """
        Plot receiving operating characteric (ROC) classifiers. 
        
        Can plot multiple classifiers at once. If multiple classifiers are 
        given, each classifier must be a tuple of  
        ``( <name>, classifier>, <method>)``. For instance, to plot the both 
        :class:`sklearn.ensemble.RandomForestClassifier` and 
        :class:`sklearn.linear_model.SGDClassifier` classifiers, they must be
        ranged as follow::
            
             clfs =[
                 ('sgd', SGDClassifier(), "decision_function" ),
                 ('forest', RandomForestClassifier(), "predict_proba") 
                 ]
        It is important to know whether the method 'predict_proba' is valid for 
        the scikit-learn classifier, we want to plot its ROC curve. 
        
        Parameters 
        -----------
        clfs :callables, always as a function, classifier estimators
            A supervised predictor with a finite set of discrete possible 
            output values. A classifier must supports modeling some of binary, 
            targets. It must store a classes attribute after fitting.
        label: int, 
            Specific class to evaluate the tradeoff of precision 
            and recall. `label`  needs to be specified and a value within the 
            target.     
            
        kind: str, ['threshold|'recall'], default='threshold' 
            kind of PR plot. If kind is 'recall', method plots the precision 
            VS the recall scores, otherwiwe the PR tradeoff is plotted against 
            the 'threshold.'
            
        method: str
            Method to get scores from each instance in the trainset. 
            Could be ``decison_funcion`` or ``predict_proba``. When using the  
            scikit-Learn classifier, it generally has one of the method. 
            Default is ``decision_function``.   
        
        cvp_kws: dict, optional
            The :func:`sklearn.model_selection.cross_val_predict` keywords 
            additional arguments 
            
        prt_kws:dict, 
            Additional keyword arguments passed to 
            func:`watex.exlib.sklearn.precision_recall_tradeoff`
            
        roc_kws: dict 
            roc_curve additional keywords arguments.
            
        Return
        -------
        ``self``: `EvalPlot` instance
            ``self`` for easy method chaining.
            
        Examples 
        --------
        (1) Plot ROC for single classifier 
        
        >>> from watex.exlib.sklearn import ( SGDClassifier, 
                                             RandomForestClassifier
                                             )
        >>> from watex.datasets.dload import load_bagoue 
        >>> from watex.utils import cattarget 
        >>> from watex.view.mlplot import EvalPlot 
        >>> X , y = load_bagoue(as_frame =True )
        >>> sgd_clf = SGDClassifier(random_state= 42) # our estimator 
        >>> b= EvalPlot(scale = True , encode_labels=True)
        >>> b.fit_transform(X, y)
        >>> # binarize the label b.y 
        >>> ybin = cattarget(b.y, labels= 2 ) # can also use labels =[0, 1]
        >>> b.y = ybin 
        >>> # plot the ROC 
        >>> b.plotROC(sgd_clf , label =1) # class=1
        ... EvalPlot(tname= None, objective= None, scale= True, ... , 
                     sns_height= 4.0, sns_aspect= 0.7, verbose= 0)
        
        (2)-> Plot ROC for multiple classifiers 
      
        >>> b= EvalPlot(scale = True , encode_labels=True, 
                        lw =3., lc=(.9, 0, .8), font_size=7 )
        >>> sgd_clf = SGDClassifier(random_state= 42)
        >>> forest_clf =RandomForestClassifier(random_state=42)
        >>> b.fit_transform(X, y)
        >>> # binarize the label b.y 
        >>> ybin = cattarget(b.y, labels= 2 ) # can also use labels =[0, 1]
        >>> b.y = ybin 
        >>> clfs =[('sgd', sgd_clf, "decision_function" ), 
               ('forest', forest_clf, "predict_proba")]
        >>> b.plotROC (clfs =clfs , label =1 )
        ... EvalPlot(tname= None, objective= None, scale= True, ... , 
                     sns_height= 4.0, sns_aspect= 0.7, verbose= 0)
        
        """
       
        # if method not given as tuple
        if not isinstance(clfs, (list, tuple)):
            try : 
                clfs =[(clfs.__name__, clfs, method)]
            except AttributeError: 
                # type `clf` is ABCMeta 
                 clfs =[(clfs.__class__.__name__, clfs, method)]
                 
        # loop and set the tuple of  (clfname , clfvalue, clfmethod)
        # anc convert to list to support item assignments
        clfs = [list(pnclf) for pnclf in clfs]
        for i, (clfn, _clf, _) in enumerate(clfs) :
        
            if  clfn is None  or clfn =='': 
                try: 
                    clfn = _clf.__name__
                except AttributeError: 
                    # when type `clf` is ABCMeta 
                    clfn= _clf.__class__.__name__
                clfs[i][0] = clfn 
                
        # reconvert to tuple values 
        clfs =[tuple(pnclf) for pnclf in clfs]
        # build multiples classifiers objects 
        rocObjs =[ROC_curve(
            clf=_clf,X=self.X,y=self.y, cv =self.cv, 
            label=label, method =meth, cvp_kws=cvp_kws,**roc_kws) 
            for (name, _clf, meth) in clfs
                  ]
        # create figure obj 
        fig = plt.figure(figsize = self.fig_size)
        ax = fig.add_subplot(1,1,1)
        
        D_COLORS = make_mpl_properties(len(clfs))
        D_STYLES= make_mpl_properties(len(clfs), prop= 'line')
        D_COLORS[0] = self.lc
        D_STYLES[0]= self.ls

        for ii, (name, _clf, _)  in enumerate( clfs): 
            ax.plot(rocObjs[ii].fpr, 
                    rocObjs[ii].tpr, 
                    label =name + ' (AUC={:.4f})'.format(
                        rocObjs[ii].roc_auc_score), 
                    color =D_COLORS[ii],
                    linestyle = D_STYLES[ii] , 
                    linewidth = self.lw
                    )
            
        xlabel = self.xlabel or 'False Positive Rate'
        ylabel = self.ylabel or 'True Positive Rate'
        
        self.xlim =[0,1]
        self.ylim =[0,1]
        ax.plot(self.xlim, self.ylim, ls= '--', color ='k')
        ax.set_xlim (self.xlim)
        ax.set_ylim (self.ylim)
        ax.set_xlabel( xlabel,
                      fontsize= .5 * self.font_size * self.fs )
        ax.set_ylabel (ylabel,
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
        
        self.save(fig)
        
        return self 

    @docSanitizer()
    def plotConfusionMatrix(
        self, 
        clf:F, 
        *, 
        kind:str =None, 
        labels:List[int]=None, 
        matshow_kws: dict=None, 
        **conf_mx_kws
        )-> 'EvalPlot': 
        """ Plot confusion matrix for error evaluation.
        
        A representation of the confusion matrix for error visualization. If 
        kind is set ``map``, plot will give the number of confused 
        instances/items. However when `kind` is set to ``error``, the number 
        of items confused is explained as a percentage. 
        
        Parameters 
        -----------
        clf :callable, always as a function, classifier estimator
            A supervised predictor with a finite set of discrete possible 
            output values. A classifier must supports modeling some of binary, 
            targets. It must store a classes attribute after fitting.
            
        labels: int, or list of int, optional
            Specific class to evaluate the tradeoff of precision 
            and recall. `label`  needs to be specified and a value within the 
            target.     
            
         plottype: str 
            can be `map` or `error` to visualize the matshow of prediction 
            and errors  respectively.
            
        matshow_kws: dict 
            matplotlib additional keywords arguments. 
            
        conf_mx_kws: dict 
            Additional confusion matrix keywords arguments.
        ylabel: list 
            list of labels names  to hold the name of each categories.
            
        Return
        -------
        ``self``: `EvalPlot` instance
            ``self`` for easy method chaining.

        Examples
        --------
        >>> from watex.datasets import fetch_data
        >>> from watex.utils.mlutils import cattarget 
        >>> from watex.exlib.sklearn import SVC 
        >>> from watex.view.mlplot import EvalPlot
        >>> X, y = fetch_data ('bagoue', return_X_y=True, as_frame =True)
        >>> # partition the target into 4 clusters-> just for demo 
        >>> b= EvalPlot(scale =True, label_values = 4 ) 
        >>> b.fit_transform (X, y) 
        >>> # prepare our estimator 
        >>> svc_clf = SVC(C=100, gamma=1e-2, kernel='rbf', random_state =42)
        >>> matshow_kwargs ={
                'aspect': 'auto', # 'auto'equal
                'interpolation': None, 
               'cmap':'jet }                   
        >>> plot_kws ={'lw':3, 
               'lc':(.9, 0, .8), 
               'font_size':15., 
                'cb_format':None,
                'xlabel': 'Predicted classes',
                'ylabel': 'Actual classes',
                'font_weight':None,
                'tp_labelbottom':False,
                'tp_labeltop':True,
                'tp_bottom': False
                }
        >>> b.plotConfusionMatrix(clf=svc_clf, 
                                  matshow_kws = matshow_kwargs, 
                                  **plot_kws)
        >>> svc_clf = SVC(C=100, gamma=1e-2, kernel='rbf', 
        ...                  random_state =42) 
        >>> # replace the integer identifier with litteral string 
        >>> b.litteral_classes = ['FR0', 'FR1', 'FR2', 'FR3']
        >>> b.plotConfusionMatrix(svc_clf, matshow_kws=matshow_kwargs, 
                                  kind='error', **plot_kws) 
        
        """
        self.inspect
        
        kind = str (kind).lower().strip() 
        if kind.find ('error')>=0 or kind.find('fill diagonal')>=0 : 
            kind ='error'
        else: kind ='map'
        
        matshow_kws= matshow_kws or dict() 
        # gives a gray color to matshow
        # if is given as matshow keywords arguments 
        # then remove it 
        _check_cmap = 'cmap' in matshow_kws.keys()
        if not _check_cmap or len(matshow_kws)==0: 
            matshow_kws['cmap']= plt.cm.gray
        
        labels = labels or self.label_values 
        y = self.y 
        if labels is not None: 
            # labels = labels_validator(self.y, labels)
            y, labels =self._cat_codes_y(values = labels, 
                                         ) 
        # for plotting purpose, change the labels to hold 
        # the string litteral class names. 
        labels = self.litteral_classes or labels 

        # get yticks one it is a classification prof
        confObj =confusion_matrix(clf=clf,
                                X=self.X,
                                y=y,
                                cv=self.cv,
                                # **conf_mx_kws
                                )
    
         # create figure obj 
        fig = plt.figure(figsize = self.fig_size)
        ax = fig.add_subplot(1,1,1)
        
        if kind =='map' : 
            cax = ax.matshow(confObj.conf_mx,  
                        **matshow_kws)
            if self.cb_label is None: 
                self.cb_label='Items confused'
                    
        if kind in ('error', 'fill diagonal'): 
            cax = ax.matshow(confObj.norm_conf_mx, 
                         **matshow_kws) 
            self.cb_label = self.cb_label or 'Error'
      
        cbax= fig.colorbar(cax, **self.cb_props)
        ax.set_xlabel( self.xlabel,
              fontsize= self.font_size )
        
        if labels is not None: 
            xticks_loc = list(ax.get_xticks())
            yticks_loc = list(ax.get_yticks())
            ax.xaxis.set_major_locator(mticker.FixedLocator(xticks_loc))
            ax.xaxis.set_major_formatter(mticker.FixedFormatter(
                [''] + list (labels)))
            ax.yaxis.set_major_locator(mticker.FixedLocator(yticks_loc))
            ax.yaxis.set_major_formatter(mticker.FixedFormatter(
                [''] + list (labels)))
        self.ylabel = self.ylabel or 'Actual classes'
        self.xlabel = self.xlabel or 'Predicted classes'
        ax.set_ylabel (self.ylabel,
                       fontsize= self.font_size *3 )
        ax.set_xlabel (self.xlabel,
                       fontsize= self.font_size *3 )
        ax.tick_params(axis=self.tp_axis, 
                        labelsize= self.font_size *3 , 
                        bottom=self.tp_bottom, 
                        top=self.tp_top, 
                        labelbottom=self.tp_labelbottom, 
                        labeltop=self.tp_labeltop
                        )
        if self.tp_labeltop: 
            ax.xaxis.set_label_position('top')
        cbax.ax.tick_params(labelsize=self.font_size * 3 ) 
        cbax.set_label(label=self.cb_label,
                       size=self.font_size * 3 ,
                       weight=self.font_weight)
        
        plt.xticks(rotation = self.rotate_xlabel)
        plt.yticks(rotation = self.rotate_ylabel)

        self.save(fig)
        
        return self
  
    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        return repr_callable_obj  (self, skip = ('y', 'X') ) 
       
    def __getattr__(self, name):
        if name.endswith ('_'): 
            if name not in self.__dict__.keys(): 
                if name in ('data_', 'X_'): 
                    raise NotFittedError (
                        f'Fit the {self.__class__.__name__!r} object first'
                        )
                
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        appender  = "" if rv is None else f'. Do you mean {rv!r}'
        
        raise AttributeError (
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{appender}{"" if rv is None else "?"}'
            )        

EvalPlot.__doc__ ="""\
Metrics, dimensionality and model evaluatation plots.  

Inherited from :class:`BasePlot`. Dimensional reduction and metric 
plots. The class works only with numerical features. 

.. admonition:: Discouraged
    
    Contineous target values for plotting classification metrics is 
    discouraged. However, We encourage user to prepare its dataset 
    before using the :class:`EvalPlot` methods. This is recommended to have 
    full control of the expected results. Indeed, the most metrics plot 
    implemented here works with supervised methods especially deals 
    with the classification problems. So, the convenient way is for  
    users to discretize/categorize (class labels) before the `fit`. 
    If not the case, as the examples of demonstration  under each method 
    implementation, we first need to categorize the continue labels. 
    The choice is twofolds: either providing individual class label 
    as a list of integers using the method :meth:`EvalPlot._cat_codes_y` 
    or by specifying the number of clusters that the target must hold. 
    Commonly the latter choice is usefull for a test or academic 
    purpose. In practice into a real dataset, it is discouraged 
    to use this kind of target partition since, it is far away of the 
    reality and will yield unexpected misinterpretation. 
    
Parameters 
-----------
{params.core.X}
{params.core.y}
{params.core.tname}
{params.evdoc.objective}
    
encode_labels: bool, default=False,  
    label encoding works with `label_values` parameter. 
    If the `y` is a continous numerical values, we could turn the 
    regression to classification by setting `encode_labels` to ``True``.
    if value is set to ``True`` and values of labels is not given, an 
    unique identifier is created which can not fit the exact needs of the 
    users. So it is recommended to set this parameters in combinaison with 
    the`label_values`.  For instance:: 
        
        encode_labels=True ; label_values =3 
        
    indicates that the target `y` values should be categorized to hold 
    the integer identifier equals to ``[0 , 1, 2]``. `y` are splitted into 
    three subsets where::
        
        classes (c) = [ c{{0}} <= y. min(), y.min() < c {{1}}< y.max(),
                         >=y.max {{2}}]
        
    This auto-splitting could not fit the exact classification of the 
    target so it is recommended to set the `label_values` as a list of 
    class labels. For instance `label_values=[0 , 1, 2]` and else. 
   
scale: str, ['StandardScaler'|'MinMaxScaler'], default ='StandardScaler'
   kind of feature scaling to apply on numerical features. Note that when 
   using PCA, it is recommended to turn `scale` to ``True`` and `fit_transform`
   rather than only fit the method. Note that `transform` method also handle 
   the missing nan value in the data where the default strategy for filling 
   is ``most_frequent``.
   
{params.core.cv}
    
prefix: str, optional 
    litteral string to prefix the integer identical labels. 
    
label_values: list of int, optional 
    works with `encode_labels` parameters. It indicates the different 
    class labels. Refer to explanation of `encode_labels`. 
    
Litteral_classes: list or str, optional 
    Works when objective is ``flow``. Replace class integer names by its 
    litteral strings. For instance:: 
        
            label_values =[0, 1, 3, 6]
            Litteral_classes = ['rate0', 'rate1', 'rate2', 'rate3']

{params.evdoc.yp_ls}
{params.evdoc.yp_lw}
{params.evdoc.yp_lc}
{params.evdoc.rs}
{params.evdoc.ps}
{params.evdoc.rc}
{params.evdoc.pc}
{params.evdoc.yp_marker}
{params.evdoc.yp_markerfacecolor}
{params.evdoc.yp_markeredgecolor}
{params.evdoc.yp_markeredgewidth}
{params.base.savefig}
{params.base.fig_dpi}
{params.base.fig_num}
{params.base.fig_size}
{params.base.fig_orientation}
{params.base.fig_title}
{params.base.fs}
{params.base.ls}
{params.base.lc}
{params.base.lw}
{params.base.alpha}
{params.base.font_weight}
{params.base.font_style}
{params.base.font_size}
{params.base.ms}
{params.base.marker}
{params.base.marker_facecolor}
{params.base.marker_edgecolor}
{params.base.marker_edgewidth}
{params.base.xminorticks}
{params.base.yminorticks}
{params.base.bins}
{params.base.xlim}
{params.base.ylim}
{params.base.xlabel}
{params.base.ylabel}
{params.base.rotate_xlabel}
{params.base.rotate_ylabel}
{params.base.leg_kws}
{params.base.plt_kws}
{params.base.glc}
{params.base.glw}
{params.base.galpha}
{params.base.gaxis}
{params.base.gwhich}
{params.base.tp_axis}
{params.base.tp_labelsize}
{params.base.tp_bottom}
{params.base.tp_labelbottom}
{params.base.tp_labeltop}
{params.base.cb_orientation}
{params.base.cb_aspect}
{params.base.cb_shrink}
{params.base.cb_pad}
{params.base.cb_anchor}
{params.base.cb_panchor}
{params.base.cb_label}
{params.base.cb_spacing}
{params.base.cb_drawedges} 

Notes 
--------
This module works with numerical data  i.e if the data must contains the 
numerical features only. If categorical values are included in the 
dataset, they should be  removed and the size of the data should be 
chunked during the fit methods. 

""".format(
    params=_param_docs,
)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# create a shadow class to hold the font and matplotlib properties
# from 'EvalPlot` and giving an option for saving figure
_b= EvalPlot () 
pobj = type ('Plot', (BasePlot, ), {**_b.__dict__} ) 
setattr(pobj, 'save', _b.save )
# redefine the pobj doc 
pobj.__doc__="""\
Shadow plotting class that holds the :class:`~watex.property.BasePlot`
parameters. 

Each matplotlib properties can be modified as  :class:`~watex.view.pobj`
attributes object. For instance:: 
    
    >>> pobj.ls ='-.' # change the line style 
    >>> pobj.fig_Size = (7, 5) # change the figure size 
    >>> pobj.lw=7. # change the linewidth 
    
.. seealso:: 
    
    Refer to :class:`~watex.property.BasePlot` for parameter details. 
    
"""
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def plotProjection(
    X: DataFrame | NDArray, 
    Xt: DataFrame | NDArray =None, *, 
    columns: List[str] =None, 
    test_kws: dict =None,  
    **baseplot_kws 
    ): 
    """ Visualize train and test dataset based on 
    the geographical coordinates.
    
    Since there is geographical information(latitude/longitude or
    easting/northing), it is a good idea to create a scatterplot of 
    all instances to visualize data.
    
    Parameters 
    ---------
    X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        training set; Denotes data that is observed at training and prediction 
        time, used as independent variables in learning. The notation 
        is uppercase to denote that it is ordinarily a matrix. When a matrix, 
        each sample may be represented by a feature vector, or a vector of 
        precomputed (dis)similarity with each training sample. 

    Xt: Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        Shorthand for "test set"; data that is observed at testing and 
        prediction time, used as independent variables in learning. The 
        notation is uppercase to denote that it is ordinarily a matrix.
    columns: list of str or index, optional 
        columns is usefull when a dataframe is given  with a dimension size 
        greater than 2. If such data is passed to `X` or `Xt`, columns must
        hold the name to considered as 'easting', 'northing' when UTM 
        coordinates are given or 'latitude' , 'longitude' when latlon are 
        given. 
        If dimension size is greater than 2 and columns is None , an error 
        will raises to prevent the user to provide the index for 'y' and 'x' 
        coordinated retrieval. 
        
    test_kws: dict, 
        keywords arguments passed to :func:`matplotlib.plot.scatter` as test
        location font and colors properties. 
        
    baseplot_kws: dict, 
        All all  the keywords arguments passed to the peroperty  
        :class:`watex.property.BasePlot` class. 
        
    Examples
    --------
    >>> from watex.datasets import fetch_data 
    >>> from watex.view.mlplot import plotProjection 
    >>> # Discard all the non-numeric data 
    >>> # then inut numerical data 
    >>> from watex.utils import to_numeric_dtypes, naive_imputer
    >>> X, Xt, *_ = fetch_data ('bagoue', split_X_y =True, as_frame =True) 
    >>> X =to_numeric_dtypes(X, pop_cat_features=True )
    >>> X= naive_imputer(X)
    >>> Xt = to_numeric_dtypes(Xt, pop_cat_features=True )
    >>> Xt= naive_imputer(Xt)
    >>> plot_kws = dict (fig_size=(8, 12),
                     lc='k',
                     marker='o',
                     lw =3.,
                     font_size=15.,
                     xlabel= 'easting (m) ',
                     ylabel='northing (m)' , 
                     markerfacecolor ='k', 
                     markeredgecolor='r',
                     alpha =1., 
                     markeredgewidth=2., 
                     show_grid =True,
                     galpha =0.2, 
                     glw=.5, 
                     rotate_xlabel =90.,
                     fs =3.,
                     s =None )
    >>> plotProjection( X, Xt , columns= ['east', 'north'], 
                        trainlabel='train location', 
                        testlabel='test location', **plot_kws
                       )
    """
    
    trainlabel =baseplot_kws.pop ('trainlabel', None )
    testlabel =baseplot_kws.pop ('testlabel', None  )
    
    for k  in list(baseplot_kws.keys()): 
        setattr (pobj , k, baseplot_kws[k])
        
    #check array
    X=check_array (
        X, 
        input_name="X", 
        to_frame =True, 
        )
    Xt =check_array (
        Xt, 
        input_name="Xt", 
        to_frame =True, 
        )
    # validate the projections.
    xy , xynames = projection_validator(X, Xt, columns )
    x, y , xt, yt =xy 
    xname, yname, xtname, yname=xynames 

    pobj.xlim =[np.ceil(min(x)), np.floor(max(x))]
    pobj.ylim =[np.ceil(min(y)), np.floor(max(y))]   
    
    xpad = abs((x -x.mean()).min())/5.
    ypad = abs((y -y.mean()).min())/5.
 
    if  Xt is not None: 

        min_x, max_x = xt.min(), xt.max()
        min_y, max_y = yt.min(), yt.max()
        
        
        pobj.xlim = [min([pobj.xlim[0], np.floor(min_x)]),
                     max([pobj.xlim[1], np.ceil(max_x)])]
        pobj.ylim = [min([pobj.ylim[0], np.floor(min_y)]),
                     max([pobj.ylim[1], np.ceil(max_y)])]
      
    pobj.xlim =[pobj.xlim[0] - xpad, pobj.xlim[1] +xpad]
    pobj.ylim =[pobj.ylim[0] - ypad, pobj.ylim[1] +ypad]
    
     # create figure obj 
    fig = plt.figure(figsize = pobj.fig_size)
    ax = fig.add_subplot(1,1,1)
    
    xname = pobj.xlabel or xname 
    yname = pobj.ylabel or yname 
    
    if pobj.s is None: 
        pobj.s = pobj.fs *40 
    ax.scatter(x, y, 
               color = pobj.lc,
                s = pobj.s if not pobj.s else pobj.fs * pobj.s, 
                alpha = pobj.alpha , 
                marker = pobj.marker,
                edgecolors = pobj.marker_edgecolor,
                linewidths = pobj.lw,
                linestyles = pobj.ls,
                facecolors = pobj.marker_facecolor,
                label = trainlabel 
            )
    
    if  Xt is not None:
        if pobj.s is not None: 
            pobj.s /=2 
        test_kws = test_kws or dict (
            color = 'r',s = pobj.s, alpha = pobj.alpha , 
            marker = pobj.marker, edgecolors = 'r',
            linewidths = pobj.lw, linestyles = pobj.ls,
            facecolors = 'k'
            )
        ax.scatter(xt, yt, 
                    label = testlabel, 
                    **test_kws
                    )

    ax.set_xlim (pobj.xlim)
    ax.set_ylim (pobj.ylim)
    ax.set_xlabel( xname,
                  fontsize= pobj.font_size )
    ax.set_ylabel (yname,
                   fontsize= pobj.font_size )
    ax.tick_params(axis='both', 
                   labelsize= pobj.font_size )
    plt.xticks(rotation = pobj.rotate_xlabel)
    plt.yticks(rotation = pobj.rotate_ylabel)
    
    if pobj.show_grid is True : 
        ax.grid(pobj.show_grid,
                axis=pobj.gaxis,
                which = pobj.gwhich, 
                color = pobj.gc,
                linestyle=pobj.gls,
                linewidth=pobj.glw, 
                alpha = pobj.galpha
                )
        if pobj.gwhich =='minor': 
            ax.minorticks_on()
            
    if len(pobj.leg_kws) ==0 or 'loc' not in pobj.leg_kws.keys():
         pobj.leg_kws['loc']='upper left'
    ax.legend(**pobj.leg_kws)
    pobj.save(fig)

              
def plotModel(
    yt: ArrayLike |Series, 
    ypred:ArrayLike |Series=None,
    *, 
    clf:F=None, 
    Xt:DataFrame|NDArray=None, 
    predict:bool =False, 
    prefix:Optional[bool]=None, 
    index:List[int|str] =None, 
    fill_between:bool=False, 
    labels:List[str]=None, 
    return_ypred:bool=False, 
    **baseplot_kws 
    ): 
    """ Plot model 'y' (true labels) versus 'ypred' (predicted) from test 
    data.
    
    Plot will allow to know where estimator/classifier fails to predict 
    correctly the target 
    
    Parameters
    ----------
    yt:array-like, shape (M, ) ``M=m-samples``,
        test target; Denotes data that may be observed at training time 
        as the dependent variable in learning, but which is unavailable 
        at prediction time, and is usually the target of prediction. 
        
    ypred:array-like, shape (M, ) ``M=m-samples``
        Array of the predicted labels. It has the same number of samples as 
        the test data 'Xt' 
        
    clf :callable, always as a function, classifier estimator
        A supervised predictor with a finite set of discrete possible 
        output values. A classifier must supports modeling some of binary, 
        targets. It must store a classes attribute after fitting.
        
    Xt: Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        Shorthand for "test set"; data that is observed at testing and 
        prediction time, used as independent variables in learning. The 
        notation is uppercase to denote that it is ordinarily a matrix.
        
    prefix: str, optional 
        litteral string to prefix the samples/examples considered as 
        tick labels in the abscissa. For instance:: 
            
            index =[0, 2, 4, 7]
            prefix ='b' --> index =['b0', 'b2', 'b4', 'b7']

    predict: bool, default=False, 
        Expected to be 'True' when user want to predict the array 'ypred'
        and plot at the same time. Otherwise, can be set to 'False' and use 
        the'ypred' data already predicted. Note that, if 'True', an  
        estimator/classifier must be provided as well as the test data 'Xt', 
        otherwise an error will occur. 
        
    index: array_like, optional
        list integer values or string expected to be the index of 'Xt' 
        and 'yt' turned into pandas dataframe and series respectively. Note 
        that one of them has already and index and new index is given, the 
        latter must be consistent. This is usefull when data are provided as
        ndarray rathern than a dataframe. 
        
    fill_between: bool 
        Fill a line between the actual classes i.e the true labels. 
        
    labels: list of str or int, Optional
       list of labels names  to hold the name of each category.
       
    return_pred: bool, 
        return predicted 'ypred' if 'True' else nothing. 
    
    baseplot_kws: dict, 
        All all  the keywords arguments passed to the peroperty  
        :class:`watex.property.BasePlot` class. 
 
    Examples
    --------
    (1)-> Prepare our data - Use analysis data of Bagoue dataset 
            since data is alread scaled and imputed
            
    >>> from watex.exlib.sklearn  import SVC 
    >>> from watex.datasets import fetch_data 
    >>> from watex.view import plotModel 
    >>> from watex.utils.mlutils import split_train_test_by_id
    >>> X, y = fetch_data('bagoue analysis' ) 
    >>> _, Xtest = split_train_test_by_id(X, 
                                          test_ratio=.3 ,  # 30% in test set 
                                          keep_colindex= False
                                        )
    >>> _, ytest = split_train_test_by_id(y, .3 , keep_colindex =False) 
    
   (2)-> prepared our demo estimator and plot model predicted 
   
    >>> svc_clf = SVC(C=100, gamma=1e-2, kernel='rbf', random_state =42) 
    >>> base_plot_params ={
                        'lw' :3.,                  # line width 
                        'lc':(.9, 0, .8), 
                        'ms':7.,                
                        'yp_marker' :'o', 
                        'fig_size':(12, 8),
                        'font_size':15.,
                        'xlabel': 'Test examples',
                        'ylabel':'Flow categories' ,
                        'marker':'o', 
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
    >>> plotModel(yt= ytest ,
                   Xt=Xtest , 
                   predict =True , # predict the result (estimator fit)
                   clf=svc_clf ,  
                   fill_between= False, 
                   prefix ='b', 
                   labels=['FR0', 'FR1', 'FR2', 'FR3'], # replace 'y' labels. 
                   **base_plot_params 
                   )
    >>> # plot show where the model failed to predict the target 'yt'
    
    """
    def format_ticks (ind, tick_number):
        """ Format thick parameter with 'FuncFormatter(func)'
        rather than using:: 
            
        axi.xaxis.set_major_locator (plt.MaxNLocator(3))
        
        ax.xaxis.set_major_formatter (plt.FuncFormatter(format_thicks))
        """
        if ind % 7 ==0: 
            return '{}'.format (index[ind])
        else: None 
        
    #xxxxxxxxxxxxxxxx update base plot keyword arguments xxxxxxxxxxxxxx
    for k  in list(baseplot_kws.keys()): 
        setattr (pobj , k, baseplot_kws[k])

    # index is used for displaying the examples label in x-abscissa  
    # for instance index = ['b4, 'b5', 'b11',  ... ,'b425', 'b427', 'b430'] 
    
    Xt, yt,index, clf, ypred= _chk_predict_args (
        Xt, yt,index, clf, ypred , predict= predict 
        )
    if prefix is not None: 
        index =np.array([f'{prefix}' +str(item) for item in index ])        
        
    # create figure obj 
    fig = plt.figure(figsize = pobj.fig_size)
    ax = fig.add_subplot(1,1,1) # create figure obj 
    # control the size of predicted items 
    pobj.s = pobj.s or pobj.fs *30 
    # plot obverved data (test label =actual)
    ax.scatter(x= index,
               y =yt ,
                color = pobj.lc,
                s = pobj.s*10,
                alpha = pobj.alpha, 
                marker = pobj.marker,
                edgecolors = pobj.marker_edgecolor,
                linewidths = pobj.lw,
                linestyles = pobj.ls,
                facecolors = pobj.marker_facecolor,
                label = 'Observed'
                   )   
    # plot the predicted target
    ax.scatter(x= index, y =ypred ,
              color = pobj.yp_lc,
               s = pobj.s/2,
               alpha = pobj.alpha, 
               marker = pobj.yp_marker,
               edgecolors = pobj.yp_marker_edgecolor,
               linewidths = pobj.yp_lw,
               linestyles = pobj.yp_ls,
               facecolors = pobj.yp_marker_facecolor,
               label = 'Predicted'
               )
  
    if fill_between: 
        ax.plot(yt, 
                c=pobj.lc,
                ls=pobj.ls, 
                lw=pobj.lw, 
                alpha=pobj.alpha
                )
    if pobj.ylabel is None:
        pobj.ylabel ='Categories '
    if pobj.xlabel is None:
        pobj.xlabel = 'Test data'
        
    if labels is not None: 
        if not  is_iterable(labels): 
            labels =[labels]

        if len(labels) != len(np.unique(yt)): 
            warnings.warn(
                "Number of categories in 'yt' and labels must be consistent."
                f" Expected {len(np.unique(yt))}, got {len(labels)}")
        else:
            ax.set_yticks(np.unique(yt))
            ax.set_yticklabels(labels)
            
    ax.set_ylabel (pobj.ylabel,
                   fontsize= pobj.font_size  )
    ax.set_xlabel (pobj.xlabel,
           fontsize= pobj.font_size  )
   
    if pobj.tp_axis is None or pobj.tp_axis =='both': 
        ax.tick_params(axis=pobj.tp_axis, 
            labelsize= pobj.tp_labelsize *5 , 
            )
        
    elif pobj.tp_axis =='x':
        param_='y'
    elif pobj.tp_axis =='y': 
        param_='x'
        
    if pobj.tp_axis in ('x', 'y'):
        ax.tick_params(axis=pobj.tp_axis, 
                        labelsize= pobj.tp_labelsize *5 , 
                        )
        
        ax.tick_params(axis=param_, 
                labelsize= pobj.font_size, 
                )
    # show label every 14 samples 
    if len(yt ) >= 14 : 
        ax.xaxis.set_major_formatter (plt.FuncFormatter(format_ticks))

    plt.xticks(rotation = pobj.rotate_xlabel)
    plt.yticks(rotation = pobj.rotate_ylabel)
    
    if pobj.show_grid: 
        ax.grid(pobj.show_grid,
                axis=pobj.gaxis,
                which = pobj.gwhich, 
                color = pobj.gc,
                linestyle=pobj.gls,
                linewidth=pobj.glw, 
                alpha = pobj.galpha
                )
        if pobj.gwhich =='minor': 
            ax.minorticks_on()
            
    if len(pobj.leg_kws) ==0 or 'loc' not in pobj.leg_kws.keys():
         pobj.leg_kws['loc']='upper left'
    ax.legend(**pobj.leg_kws)
    
    pobj.save(fig)
    
    return ypred if return_ypred else None   

def plot_reg_scoring(
    reg, X, y, test_size=None, random_state =42, scoring ='mse',
    return_errors: bool=False, **baseplot_kws
    ): 
    #xxxxxxxxxxxxxxxx update base plot keyword arguments
    for k  in list(baseplot_kws.keys()): 
        setattr (pobj , k, baseplot_kws[k])
        
    scoring = scoring or 'mse'
    scoring = str(scoring).lower().strip() 
    if scoring not in ('mse', 'rme'): 
        raise ValueError ("Acceptable scorings are'mse' are 'rmse'"
                          f" got {scoring!r}")
    if not hasattr(reg, '__class__') and not inspect.isclass(reg.__class__): 
        raise TypeError(f"{reg!r} isn't a model estimator.")
         
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    
    train_errors, val_errors = [], []
    for m in range(1, len(y_train)): 
        try:
            reg.fit(X_train[:m], y_train[:m])
        except ValueError: # value_error 
            # raise ValueError (msg) from value_error 
            # skip the valueError
            # <The number of classes has to be greater 
            # than one; got 1 class>
            continue
        
        y_train_pred = reg.predict(X_train[:m])
        y_val_pred = reg.predict(X_val)
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
        
    if pobj.ylabel is None:
            pobj.ylabel ='Score'
            
    if pobj.xlabel is None: 
        pobj.xlabel = 'Training set size'
        
    fig = plt.figure(figsize = pobj.fig_size)
    ax = fig.add_subplot(1,1,1) # create figure obj 
    
    # set new attributes 
    for nv, vv in zip(('vlc', 'vls'), ('b', ':')): 
        if not hasattr(pobj, nv): 
            setattr(pobj, nv, vv)
        
    ax.plot(train_errors,
            color = pobj.lc, 
            linewidth = pobj.lw,
            linestyle = pobj.ls , 
            label = 'training set',
            **pobj.plt_kws )
    ax.plot(val_errors,
            color = pobj.vlc, 
            linewidth = pobj.lw,
            linestyle = pobj.vls , 
            label = 'validation set',
            **pobj.plt_kws )
    
    _remaining_plot_roperties(pobj, ax,  fig=fig )
    
    return (train_errors, val_errors) if return_errors else None 

plot_reg_scoring.__doc__ ="""\
Plot regressor learning curves using root-mean squared error scorings. 

Use the hold-out cross-validation technique for score evaluation [1]_. 

Parameters 
-----------
reg: callable, always as a function
    A regression estimator; Estimators must provide a fit method, and 
    should provide `set_params` and `get_params`, although these are usually 
    provided by inheritance from `base.BaseEstimator`. The estimated model 
    is stored in public and private attributes on the estimator instance, 
    facilitating decoding through prediction and transformation methods. 
    The core functionality of some estimators may also be available as 
    a ``function``.
     
{params.core.X}
{params.core.y}
scoring: str, ['mse'|'rmse'], default ='mse'
    kind of error to visualize on the regression learning curve. 
{params.core.test_size}
{params.core.random_state}

return_errors: bool, default='False'
    returns training eror and validation errors. 
    
baseplot_kws: dict, 
    All all  the keywords arguments passed to the peroperty  
    :class:`watex.property.BasePlot` class. 
    
Returns 
--------
(train_errors, val_errors): Tuple, 
    training score and validation scores if `return_errors` is set to 
    ``True``, otherwise returns nothing   
    
Examples 
--------- 
>>> from watex.datasets import fetch_data 
>>> from watex.view.mlplot import plot_reg_scoring 
>>> # Note that for the demo, we import SVC rather than LinearSVR since the 
>>> # problem of Bagoue dataset is a classification rather than regression.
>>> # if use regression instead, a convergence problem will occurs. 
>>> from watex.exlib.sklearn import SVC 
>>> X, y = fetch_data('bagoue analysed')# got the preprocessed and imputed data
>>> svm =SVC() 
>>> t_errors, v_errors =plot_reg_scoring(svm, X, y, return_errors=True)


Notes  
------
The hold-out technique is the classic and most popular approach for 
estimating the generalization performance of the machine learning. The 
dataset is splitted into training and test sets. The former is used for the 
model training whereas the latter is used for model performance evaluation. 
However in typical machine learning we are also interessed in tuning and 
comparing different parameter setting for futher improve the performance 
for the name refering to the given classification or regression problem for 
which we want the optimal values of tuning the hyperparameters. Thus, reusing 
the same datset over and over again during the model selection is not 
recommended since it will become a part of the training data and then the 
model will be more likely to overfit. From this issue, the hold-out cross 
validation is not a good learning practice. A better way to use the hold-out 
method is to separate the data into three parts such as the traing set, the 
the validation set and the test dataset. See more in [2]_. 

References 
------------
.. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
    Grisel, O., Blondel, M., et al. (2011) Scikit-learn: Machine learning in 
    Python. J. Mach. Learn. Res., 12, 2825–2830.
.. [2] Raschka, S. & Mirjalili, V. (2019) Python Machine Learning. 
    (J. Malysiak, S. Jain, J. Lovell, C. Nelson, S. D’silva & R. Atitkar, Eds.), 
    3rd ed., Packt.
""".format(params = _param_docs)    


def plot_model_scores(models, scores=None, cv_size=None, **baseplot_kws): 
    #xxxxxxxxxxxxxxxx set base plot keywords arguments
    for k  in list(baseplot_kws.keys()): 
        setattr (pobj , k, baseplot_kws[k])
        
    # if scores is None: 
    #     raise ValueError('NoneType can not be plot.')
    if isinstance(models, str): 
        models = str2columns (models)

    if not is_iterable(models): 
        models =[models]
        
    _ckeck_score = scores is not None 
    if _ckeck_score :
        scores = is_iterable(scores, exclude_string=True, transform= True )
        # if is_iterable(models) and is_iterable(scores): 
        if len(models) != len(scores): 
            raise TypeError(
                "Fined-tuned model and scores sizes must be consistent;"
                f" got {len(models)!r} and {len(scores)} respectively.")
            
    elif scores is None: 
        # check wether scores are appended to model
        try : 
            scores = [score for _, score in models]
        except: 
            raise TypeError (
                "Missing score(s). Scores are needed for each model.")
        models= [model for model, _  in models ]
    # for item assigments, use list instead. 
    models=[[bn, bscore] for bn, bscore in zip(models, scores)]

    for ii, (model, _) in enumerate(models) : 
        model = model or 'None'
        if not isinstance (model, str): 
            if inspect.isclass(model.__class__): 
                models[ii][0] = model.__class__.__name__
            else: 
                models[ii][0] = type(model).__name__
                
    # get_the minimal size from cv if not isinstance(cv, (int, float) ):
    cv_size_min = min (
        [ len(models[i][1]) for i in range (len(models))])

    if cv_size is None: 
        cv_size = cv_size_min

    if cv_size is not None: 
        try : 
            cv_size = int(cv_size)
        except: 
            raise ValueError(
                f"Expect a number for 'cv', got {type(cv_size).__name__!r}.")
            
        if cv_size < 1 : 
            raise ValueError (
                f"cv must contain at least one positivevalue, got {cv_size}")
        elif cv_size > cv_size_min : 
            raise ValueError(f"Size for cv is too large; expect {cv_size_min}"
                             f" as a maximum size, got {cv_size}")
        # shrink to the number of validation to keep the same size for all 
        # give model 
        models = [(modelname, modelval[:cv_size] ) 
                  for modelname, modelval in models]
    # customize plots with colors lines and styles 
    # and create figure obj 
    lcs_kws = {'lc': make_mpl_properties(cv_size), 
             'ls':make_mpl_properties(cv_size, 'line')
             }
    lcs_kws ['ls']= [pobj.ls] + lcs_kws['ls']
    lcs_kws ['lc']= [pobj.lc] + lcs_kws['lc']
    # create figure obj and change style
    # if sns_style is passed as base_plot_params 
    fig = plt.figure(figsize = pobj.fig_size)
    ax = fig.add_subplot(1,1,1) 
    if pobj.sns_style is not None: 
       sns.set_style(pobj.sns_style)
       
    for k in range(len(models)): 
        ax.plot(
            # np.array([i for i in range(cv_size)]) +1,
                np.arange (cv_size) +1, 
                models[k][1],
                color = lcs_kws['lc'][k], 
                linewidth = pobj.lw,
                linestyle = lcs_kws['ls'][k], 
                label = models[k][0],
                )
    # appendLineParams(pobj, ax, xlim=pobj.xlim, ylim=pobj.ylim)
    _remaining_plot_roperties(pobj, ax, xlim=pobj.xlim, 
                              ylim=pobj.ylim, fig=fig 
                       )
    pobj.save(fig)
    
plot_model_scores.__doc__="""\
uses the cross validation to get an estimation of model performance 
generalization.

It Visualizes model fined tuned scores vs the cross validation

Parameters 
----------
models: list of callables, always as a functions,   
    list of estimator names can also be  a pair estimators and validations 
    scores.For instance estimators and scores can be arranged as:: 
        
        models =[('SVM', scores_svm), ('LogRegress', scores_logregress), ...]
        
    If that arrangement is passed to `models` parameter then no need to pass 
    the score values of each estimators in `scores`. 
    Note that a model is an object which manages the estimation and 
    decoding. The model is estimated as a deterministic function of:

        * parameters provided in object construction or with set_params;
        * the global numpy.random random state if the estimator’s random_state 
            parameter is set to None; and
        * any data or sample properties passed to the most recent call to fit, 
            fit_transform or fit_predict, or data similarly passed in a sequence 
            of calls to partial_fit.
            
    list of estimators names or a pairs estimators and validations scores.
    For instance:: 
        
        clfs =[('SVM', scores_svm), ('LogRegress', scores_logregress), ...]
        
scores: array like 
    list of scores on different validation sets. If scores are given, 
    set only the name of the estimators passed to `models` like:: 
        
        models =['SVM', 'LogRegress', ...]
        scores=[scores_svm, scores_logregress, ...]

cv_size: float or int,
    The number of fold used for validation. If different models have different 
    cross validation values, the minimum size of cross validation is used and the 
    scored of each model is resized to match the minimum size number. 
    
baseplot_kws: dict, 
    All all  the keywords arguments passed to the peroperty  
    :class:`watex.property.BasePlot` class.  
    
Examples 
---------
(1) -> Score is appended to the model 
>>> from watex.exlib.sklearn import SVC 
>>> from watex.view.mlplot import plot_model_scores
>>> import numpy as np 
>>> svc_model = SVC() 
>>> fake_scores = np.random.permutation (np.arange (0, 1,  .05))
>>> plot_model_scores([(svc_model, fake_scores )])
... 
(2) -> Use model and score separately 

>>> plot_model_scores([svc_model],scores =[fake_scores] )# 
>>> # customize plot by passing keywords properties 
>>> base_plot_params ={
                    'lw' :3.,                  
                    'lc':(.9, 0, .8), 
                    'ms':7.,                
                    'fig_size':(12, 8),
                    'font_size':15.,
                    'xlabel': 'samples',
                    'ylabel':'scores' ,
                    'marker':'o', 
                    'alpha' :1., 
                    'yp_markeredgewidth':2.,
                    'show_grid' :True,          
                    'galpha' :0.2,              
                    'glw':.5,                   
                    'rotate_xlabel' :90.,
                    'fs' :3.,                   
                    's' :20 ,
                    'sns_style': 'darkgrid', 
               }
>>> plot_model_scores([svc_model],scores =[fake_scores] , **base_plot_params ) 
"""
def plotDendroheat(
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
    Attaches dendrogram to a heat map. 
    
    Hierachical dendrogram are often used in combination with a heat map which
    allows us to represent the individual value in data array or matrix 
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
        
    Examples
    ---------
    >>> # (1) -> Use random data
    >>> import numpy as np 
    >>> from watex.view.mlplot import plotDendroheat
    >>> np.random.seed(123) 
    >>> variables =['X', 'Y', 'Z'] ; labels =['ID_0', 'ID_1', 'ID_2',
                                             'ID_3', 'ID_4']
    >>> X= np.random.random_sample ([5,3]) *10 
    >>> df =pd.DataFrame (X, columns =variables, index =labels)
    >>> plotDendroheat (df)
    >>> # (2) -> Use Bagoue data 
    >>> from watex.datasets import load_bagoue  
    >>> X, y = load_bagoue (as_frame=True )
    >>> X =X[['magnitude', 'power', 'sfi']].astype(float) # convert to float
    >>> plotDendroheat (X )
    
    
    """
 
    df=check_array (
        df, 
        input_name="Data 'df' ", 
        to_frame =True, 
        )
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
    df_rowclust = df.iloc [r['leaves'][::-1]] if hasattr(
        df, 'columns') else df  [r['leaves'][::-1]]
    
    # 3. construct the heatmap from the reordered dataframe and position 
    # in the next ro the dendrogram 
    axm = fig.add_axes ([.23, .1, .63, .6]) #.6 # [.23, .1, .2, .6]
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
    xticks_loc = list(axm.get_xticks())
    yticks_loc = list(axm.get_yticks())

    df_rowclust_cols = df_rowclust.columns if hasattr (
        df_rowclust , 'columns') else [f"{i+1}" for i in range (df.shape[1])]
    axm.xaxis.set_major_locator(mticker.FixedLocator(xticks_loc))
    axm.xaxis.set_major_formatter(mticker.FixedFormatter(
        [''] + list (df_rowclust_cols)))
    
    df_rowclust_index = df_rowclust.index if hasattr(
        df_rowclust , 'columns') else [f"{i}" for i in range (df.shape[0])]
    axm.yaxis.set_major_locator(mticker.FixedLocator(yticks_loc))
    axm.yaxis.set_major_formatter(mticker.FixedFormatter(
        [''] + list (df_rowclust_index)))
    
    plt.show () 
    
    
def plotDendrogram (
    df:DataFrame, 
    columns:List[str] =None, 
    labels: ArrayLike =None,
    metric:str ='euclidean',  
    method:str ='complete', 
    kind:str = None,
    return_r:bool =False, 
    verbose:bool=False, 
    **kwd ): 
    r""" Visualizes the linkage matrix in the results of dendrogram. 
    
    Note that the categorical features if exist in the dataframe should 
    automatically be discarded. 
    
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
        return r-dictionnary if set to 'True' otherwise returns nothing 
    
    verbose: int, bool, default='False' 
        If ``True``, output message of the name of categorical features 
        dropped.
    
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
    if hasattr (df, 'columns') and columns is not None: 
        df = df [columns ]
        
    df = to_numeric_dtypes(df, pop_cat_features= True, verbose =verbose )
    
    df=check_array (
        df, 
        input_name="Data 'df' ", 
        to_frame =True, 
        )

    kind:str = kind or 'design'
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

def plotSilhouette (
    X:NDArray |DataFrame, 
    labels:ArrayLike=None, 
    prefit:bool=True, 
    n_clusters:int =3,  
    n_init: int=10 , 
    max_iter:int=300 , 
    random_state:int=None , 
    tol:float=1e4 , 
    metric:str='euclidean', 
    **kwd 
 ): 
    r"""
    quantifies the quality  of clustering samples. 
    
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training instances to cluster. It must be noted that the data
        will be converted to C ordering, which will cause a memory
        copy if the given data is not C-contiguous.
        If a sparse matrix is passed, a copy will be made if it's not in
        CSR format.
        
    labels : array-like 1d of shape (n_samples,)
        Label values for each sample.
         
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.
        
    prefit : bool, default=False
        Whether a prefit `labels` is expected to be passed into the function
        directly or not.
        If `True`, `labels` must be a fit predicted values target.
        If `False`, `labels` is fitted and updated from `X` by calling
        `fit_predict` methods. Any other values passed to `labels` is 
        discarded.
         
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=42
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
    
    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        
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
        
    Note
    -------
    The sihouette coefficient is bound between -1 and 1 
    
    See More
    ---------
    Silhouette is used as graphical tools,  to plot a measure how tighly is  
    grouped the examples of the clusters are.  To calculate the silhouette 
    coefficient, three steps is allows: 
        
    * calculate the **cluster cohesion**, :math:`a(i)`, as the average 
        distance between examples, :math:`x^{(i)}`, and all the others 
        points
    * calculate the **cluster separation**, :math:`b^{(i)}` from the next 
        average distance between the example , :math:`x^{(i)}` amd all 
        the example of nearest cluster 
    * calculate the silhouette, :math:`s^{(i)}`, as the difference between 
        the cluster cohesion and separation divided by the greater of the 
        two, as shown here: 
    
        .. math:: 
            
            s^{(i)}=\frac{b^{(i)} - a^{(i)}}{max {{b^{(i)},a^{(i)} }}}
                
    Examples 
    --------
    >>> from watex.datasets import load_hlogs 
    >>> from watex.view.mlplot import plotSilhouette
    >>> # use resistivity and gamma for this demo
    >>> X_res_gamma = load_hlogs().frame[['resistivity', 'gamma_gamma']]  
    
    (1) Plot silhouette with 'prefit' set to 'False' 
    >>> plotSilhouette (X_res_gamma, prefit =False)
    
    """
    if  ( 
        not prefit 
        and labels is not None
        ): 
        warnings.warn("'labels' is given while 'prefix' is 'False'"
                      "'prefit' will set to 'True'")
        prefit=True 
        
    if labels is not None: 
        if not hasattr (labels, '__array__'): 
            raise TypeError( "Labels (target 'y') expects an array-like: "
                            f"{type(labels).__name__!r}")
        labels=check_y (
            labels, 
            to_frame =True, 
            )
        if len(labels)!=len(X): 
            raise TypeError("X and labels must have a consistency size."
                            f"{len(X)} and {len(labels)} respectively.")
            
    if prefit and labels is None: 
        raise TypeError ("Labels can not be None, while 'prefit' is 'True'"
                         " Turn 'prefit' to 'False' or provide the labels "
                         "instead.")
    if not prefit : 
        km= KMeans (n_clusters =n_clusters , 
                    init='k-means++', 
                    n_init =n_init , 
                    max_iter = max_iter , 
                    tol=tol, 
                    random_state =random_state
                        ) 
        labels = km.fit_predict(X ) 
        
    return _plotSilhouette(X, labels, metric = metric , **kwd)
    
    
def _plotSilhouette (X, labels, metric ='euclidean', **kwds ):
    r"""Plot quantifying the quality  of clustering silhouette 
    
    Parameters 
    ---------
    X : array-like of shape (n_samples_a, n_samples_a) if metric == \
            "precomputed" or (n_samples_a, n_features) otherwise
        An array of pairwise distances between samples, or a feature array.

    labels : array-like of shape (n_samples,)
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
    >>> from watex.exlib.sklearn import KMeans 
    >>> from watex.datasets import load_iris 
    >>> from watex.view.mlplot import plotSilhouette
    >>> d= load_iris ()
    >>> X= d.data [:, 0][:, np.newaxis] # take the first axis 
    >>> km= KMeans (n_clusters =3 , init='k-means++', n_init =10 , 
                    max_iter = 300 , 
                    tol=1e-4, 
                    random_state =0 
                    )
    >>> y_km = km.fit_predict(X) 
    >>> plotSilhouette (X, y_km)
  
    See also 
    ---------
    watex.utils.plotutils.plot_silhouette: Plot naive silhouette 
    
    Notes
    ------ 
    
    Silhouette is used as graphical tools,  to plot a measure how tighly is  
    grouped the examples of the clusters are.  To calculate the silhouette 
    coefficient, three steps is allows: 
        
    * calculate the **cluster cohesion**, :math:`a(i)`, as the average 
        distance between examples, :math:`x^{(i)}`, and all the others 
        points
    * calculate the **cluster separation**, :math:`b^{(i)}` from the next 
        average distance between the example , :math:`x^{(i)}` amd all 
        the example of nearest cluster 
    * calculate the silhouette, :math:`s^{(i)}`, as the difference between 
        the cluster cohesion and separation divided by the greater of the 
        two, as shown here: 
            
        .. math:: 
            
            s^{(i)}=\frac{b^{(i)} - a^{(i)}}{max {{b^{(i)},a^{(i)} }}}
    
    Note that the sihouette coefficient is bound between -1 and 1 
    
    """
    cluster_labels = np.unique (labels) 
    n_clusters = cluster_labels.shape [0] 
    silhouette_vals = silhouette_samples(X, labels= labels, metric = metric ,
                                         **kwds)
    y_ax_lower , y_ax_upper = 0, 0 
    yticks =[]
    
    for i, c  in enumerate (cluster_labels ) : 
        c_silhouette_vals = silhouette_vals[labels ==c ] 
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
    
def plotLearningInspections (
        models:List[object] , 
        X:NDArray, y:ArrayLike,  
        fig_size:Tuple[int] = ( 22, 18 ) , 
        cv: int = None, 
        savefig:Optional[str] = None, 
        titles = None, 
        subplot_kws =None, 
        **kws 
  ): 
    """ Inspect multiple models from their learning curves. 
    
    Mutiples Inspection plots that generate the test and training learning 
    curve, the training  samples vs fit times curve, the fit times 
    vs score curve for each model.  
    
    Parameters
    ----------
    models : list of estimator instances
        Each estimator instance implements `fit` and `predict` methods which
        will be cloned for each validation.
    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

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

        Refer Sckikit-learn :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
        
    savefig: str, default =None , 
        the path to save the figures. Argument is passed to matplotlib.Figure 
        class. 
        
    kws: dict, 
        Additional keywords argument passed to :func:`plotLearningInspection`. 
        
    Returns
    ----------
    axes: Matplotlib axes
    
    See also 
    ---------
    plotLearningInspection:  Inspect single model 
    
    Examples 
    ---------
    >>> from watex.datasets import fetch_data
    >>> from watex.models.premodels import p 
    >>> from watex.view.mlplot import plotLearningInspections 
    >>> # import sparse  matrix from Bagoue dataset 
    >>> X, y = fetch_data ('bagoue prepared') 
    >>> # import the two pretrained models from SVM 
    >>> models = [p.SVM.rbf.best_estimator_ , p.SVM.poly.best_estimator_]
    >>> plotLearningInspections (models , X, y, ylim=(0.7, 1.01) )
    
    """
    if not is_iterable( models) : 
        models =[models ] 
    if not is_iterable (titles ): 
        titles =[titles] 
    if len(titles ) != len(models): 
        titles = titles + [None for i in range (len(models)- len(titles))]
    # set the cross-validation to 4 
    cv = cv or 4  
    #set figure and subplots 
    if len(models)==1:
        msg = ( f"{plotLearningInspection.__module__}."
               f"{plotLearningInspection.__qualname__}"
               ) 
        raise PlotError ("For a single model inspection, use the"
                         f" function {msg!r} instead."
                         )
        
    fig , axes = plt.subplots (3 , len(models), figsize = fig_size )
    subplot_kws = subplot_kws or  dict(
        left=0.0625, right = 0.95, wspace = 0.1, hspace = .5 )
    
    fig.subplots_adjust(**subplot_kws)
    
    if not is_iterable( axes) :
        axes =[axes ] 
    for kk, model in enumerate ( models ) : 
        title = titles[kk] or  get_estimator_name (model )
        plotLearningInspection(model, X=X , y=y, axes = axes [:, kk], 
                               title =title, 
                               **kws)
        
    if savefig : 
        fig.savefig (savefig , dpi = 300 )
    plt.show () if savefig is None else plt.close () 
    
def plotLearningInspection(
    model,  
    X,  
    y, 
    axes=None, 
    ylim=None, 
    cv=5, 
    n_jobs=None,
    train_sizes=None, 
    display_legend = True, 
    title=None,
):
    """Inspect model from its learning curve. 
    
    Generate 3 plots: the test and training learning curve, the training
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
    
    Examples 
    ----------
    >>> from watex.datasets import fetch_data
    >>> from watex.models import p 
    >>> from watex.view.mlplot import plotLearningInspection 
    >>> # import sparse  matrix from Bagoue datasets 
    >>> X, y = fetch_data ('bagoue prepared') 
    >>> # import the  pretrained Radial Basis Function (RBF) from SVM 
    >>> plotLearningInspection (p.SVM.rbf.best_estimator_  , X, y )
    
    """ 
    train_sizes = train_sizes or np.linspace(0.1, 1.0, 5)
    
    X, y = check_X_y(
        X, 
        y, 
        accept_sparse= True,
        to_frame =True 
        )
    
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    axes[0].set_title(title or get_estimator_name(model))
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

    # set title name
    title_name = ( 
        f"{'the model'if title else get_estimator_name(model)}"
        )
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
    axes[1].set_title(f"Scalability of {title_name}")

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
    axes[2].set_title(f"Performance of {title_name}")

    return axes
#XXX
def plot_matshow(
    arr, / , labelx:List[str] =None, labely:List[str]=None, 
    matshow_kws=None, **baseplot_kws
    ): 
    
    #xxxxxxxxx update base plot keyword arguments
    for k  in list(baseplot_kws.keys()): 
        setattr (pobj , k, baseplot_kws[k])
        
    arr= check_array(
        arr, 
        to_frame =True, 
        input_name="Array 'arr'"
        )
    matshow_kws= matshow_kws or dict()
    fig = plt.figure(figsize = pobj.fig_size)

    ax = fig.add_subplot(1,1,1)
    
    cax = ax.matshow(arr, **matshow_kws) 
    cbax= fig.colorbar(cax, **pobj.cb_props)
    
    if pobj.cb_label is None: 
        pobj.cb_label=''
    ax.set_xlabel( pobj.xlabel,
          fontsize= pobj.font_size )
    
    
    # for label in zip ([labelx, labely]): 
    #     if label is not None:
    #         if not is_iterable(label):
    #             label = [label]
    #         if len(label) !=arr.shape[1]: 
    #             warnings.warn(
    #                 "labels and arr dimensions must be consistent"
    #                 f" Expect {arr.shape[1]}, got {len(label)}. "
    #                 )
                #continue
    if labelx is not None: 
        ax = _check_labelxy (labelx , arr, ax )
    if labely is not None: 
        ax = _check_labelxy (labely, arr, ax , axis ='y')
    
    if pobj.ylabel is None:
        pobj.ylabel =''
    if pobj.xlabel is None:
        pobj.xlabel = ''
    
    ax.set_ylabel (pobj.ylabel,
                   fontsize= pobj.font_size )
    ax.tick_params(axis=pobj.tp_axis, 
                    labelsize= pobj.font_size, 
                    bottom=pobj.tp_bottom, 
                    top=pobj.tp_top, 
                    labelbottom=pobj.tp_labelbottom, 
                    labeltop=pobj.tp_labeltop
                    )
    if pobj.tp_labeltop: 
        ax.xaxis.set_label_position('top')
    
    cbax.ax.tick_params(labelsize=pobj.font_size ) 
    cbax.set_label(label=pobj.cb_label,
                   size=pobj.font_size,
                   weight=pobj.font_weight)
    
    plt.xticks(rotation = pobj.rotate_xlabel)
    plt.yticks(rotation = pobj.rotate_ylabel)

    pobj.save(fig)

plot_matshow.__doc__ ="""\
Quick matrix visualization using matplotlib.pyplot.matshow.

Parameters
----------
arr: 2D ndarray, 
    matrix of n rowns and m-columns items 
matshow_kws: dict
    Additional keywords arguments for :func:`matplotlib.axes.matshow`
    
labelx: list of str, optional 
        list of labels names that express the name of each category on 
        x-axis. It might be consistent with the matrix number of 
        columns of `arr`. 
        
label: list of str, optional 
        list of labels names that express the name of each category on 
        y-axis. It might be consistent with the matrix number of 
        row of `arr`.
    
Examples
---------
>>> import numpy as np
>>> from watex.view.mlplot import plot_matshow 
>>> matshow_kwargs ={
    'aspect': 'auto',
    'interpolation': None,
   'cmap':'copper_r', 
        }
>>> baseplot_kws ={'lw':3, 
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
>>> labelx =['FR0', 'FR1', 'FR2', 'FR3', 'Rates'] 
>>> labely =['VOLCANO-SEDIM. SCHISTS', 'GEOSYN. GRANITES', 
             'GRANITES', '1.0', 'Rates']
>>> array2d = np.array([(1. , .5, 1. ,1., .9286), 
                    (.5,  .8, 1., .667, .7692),
                    (.7, .81, .7, .5, .7442),
                    (.667, .75, 1., .75, .82),
                    (.9091, 0.8064, .7, .8667, .7931)])
>>> plot_matshow(array2d, labelx, labely, matshow_kwargs,**baseplot_kws )  

"""
  
def biPlot(
    self, 
    Xr: NDArray,
    components:NDArray,
    y: ArrayLike,
    classes: ArrayLike=None,
    markers:List [str]=None, 
    colors: List [str ]=None, 
 ):
    """
    The biplot is the best way to visualize all-in-one following a PCA analysis.
    
    There is an implementation in R but there is no standard implementation
    in Python. 

    Parameters  
    -----------
    self: :class:`watex.property.BasePlot`. 
        Matplotlib property from `BasePlot` instances. Default `BasePlot`  
        instance is given as a `pobj` instance and can be loaded for plotting 
        purpose as:: 
            
            >>> from watex.view import pobj 
        
        To change some default plot properties like line width or style, both 
        can be set before running the script as follow :: 
            
            >>> pobj.lw = 2. ; pobj.ls=':' # and so on 
            
    Xr: NDArray of transformed X. 
        the PCA projected data scores on n-given components.The reduced  
        dimension of train set 'X' with maximum ratio as sorted eigenvectors 
        from first to the last component. 
    components: NDArray, shape (n_components, n_eigenvectors ), 
        the eigenvectors of the PCA. The shape in axis must much the number 
        of component computed using PCA. If the `Xr` shape 1 equals to the 
        shape 0 of the component matrix `components`, it will be transposed 
        to fit `Xr` shape 1. 
    y: Array-like, 
        the target composing the class labels.
    classes: list or int, 
        class categories or class labels 
    markers: str, 
        Matplotlib list of markers for plotting  classes.
    colors: str, 
        Matplotlib list of colors to customize plots 
    
    Examples 
    ---------
    >>> from watex.analysis import nPCA
    >>> from watex.datasets import fetch_data
    >>> from watex.view import biPlot, pobj  # pobj is Baseplot instance 
    >>> X, y = fetch_data ('bagoue pca' )  # fetch pca data 
    >>> pca= nPCA (X, n_components= 2 , return_X= False ) # return PCA object 
    >>> components = pca.components_ [:2, :] # for two components 
    >>> biPlot (pobj, pca.X, components , y ) # pca.X is the reduced dim X 
    >>> # to change for instance line width (lw) or style (ls) 
    >>> # just use the baseplotobject (pobj)
    
    References 
    -----------
    Originally written by `Serafeim Loukas`_, serafeim.loukas@epfl.ch 
    and was edited to fit the :term:`watex` package API. 
    
    .. _Serafeim Loukas: https://towardsdatascience.com/...-python-7c274582c37e
    
    """
    Xr = check_array(
        Xr, 
        to_frame= False, 
        input_name="X reduced 'Xr'"
        )
    components = check_array(
        components, 
        to_frame =False ,
        input_name="PCA components"
        )
    Xr = np.array (Xr); components = np.array (components )
    xs = Xr[:,0] # projection on PC1
    ys = Xr[:,1] # projection on PC2
    
    if Xr.shape[1]==components.shape [0] :
        # i.e components is not transposed 
        # transposed then 
        components = components.T 
    n = components.shape[0] # number of variables
    
    fig = plt.figure(figsize=self.fig_size, #(10,8),
               dpi=self.fig_dpi #100
               )
    if classes is None: 
        classes = np.unique(y)
    if colors is None:
        # make color based on group
        # to fit length of classes
        colors = make_mpl_properties(
            len(classes))
        
    colors = [colors[c] for c in range(len(classes))]
    if markers is None:
        markers= make_mpl_properties(len(classes), prop='marker')
        
    markers = [markers[m] for m in range(len(classes))]
    
    for s,l in enumerate(classes):
        plt.scatter(xs[y==l],ys[y==l], 
                    c = colors[s], 
                    marker=markers[s]
                    ) 
    for i in range(n):
        # plot as arrows the variable scores 
        # (each variable has a score for PC1 and one for PC2)
        plt.arrow(0, 0, components[i,0], components[i,1], 
                  color = self.lc, #'k', 
                  alpha = self.alpha, #0.9,
                  linestyle = self.ls, # '-',
                  linewidth = self.lw, #1.5,
                  overhang=0.2)
        plt.text(components[i,0]* 1.15, components[i,1] * 1.15, 
                 "Var"+str(i+1),
                 color = 'k', 
                 ha = 'center',
                 va = 'center',
                 fontsize= self.font_size
                 )
    plt.tick_params(axis ='both', labelsize = self.font_size)
    
    plt.xlabel(self.xlabel or "PC1",size=self.font_size)
    plt.ylabel(self.ylabel or "PC2",size=self.font_size)
    limx= int(xs.max()) + 1
    limy= int(ys.max()) + 1
    plt.xlim([-limx,limx])
    plt.ylim([-limy,limy])
    plt.grid()
    plt.tick_params(axis='both',
                    which='both', 
                    labelsize=self.font_size
                    )
    
    self.save(fig)
    # if self.savefig is not None: 
    #     savefigure (plt, self.savefig, dpi = self.fig_dpi )
    
def _remaining_plot_roperties (self, ax, xlim=None, ylim=None, fig=None ): 
    """Append the remaining lines properties such as xlabel, grid , 
    legend and ticks parameters. Relevant idea to not 
    DRY(Don't Repeat Yourself). 
    :param ax: matplotlib.pyplot.axis 
    :param (xlim, ylim): Limit of x-axis and y-axis 
    :param fig: Matplotlib.figure name. 
    
    :return: self- Plot object. 
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

    self.save(fig)
        
    return self 


def _chk_predict_args (Xt, yt, *args,  predict =False ): 
    """ Validate arguments passed  for model prediction 
    
    :param Xt: ndarray|DataFrame, test data 
    :param yt: array-like, pandas serie for test label 
    :param args: list of other keyword arguments which seems to be usefull. 
    :param predict: bool, expect a prediction or not. 
    :returns: Tuple (Xt, yt, index , clf ,  ypred )- tuple of : 
        * Xt : test data 
        * yt : test label data 
        * index :index to fit the samples in the dataframe or the 
            shape [0] of ndarray 
        * clf: the predictor or estimator 
        * ypred: the estimator predicted values 
        
    """
    # index is used for displayed the examples label in x-abscissa  
    # for instance index = ['b4, 'b5', 'b11',  ... ,'b425', 'b427', 'b430']
    
    index , clf ,  ypred = args 
    if index is not None:
        #control len of index and len of y
        if not is_iterable (index): 
            raise TypeError("Index is an iterable object with the same length"
                            "as 'y', got '{type (index).__name__!r}'") 
        len_index= len(yt)==len(index)
        
        if not len_index:
            warnings.warn(
                "Expect an index size be consistent with 'y' size={len(yt)},"
                  " got'{len(index)}'. Given index can not be used."
                  )
            index =None
            
        if len_index : 
            if isinstance(yt, (pd.Series, pd.DataFrame)):
                if not np.all(yt.index.isin(index)):
                    warnings.warn(
                        "Given index values are mismatched. Note that for "
                        "overlaying the model plot, 'Xt' indexes must be "
                        "identical to the one in target 'yt'. The indexes"
                        " provided are wrong and should be resetted."
                        )
                    index =yt.index 
                    yt=yt.values()
            yt= pd.Series(yt, index = index )
            
    if predict: 
        if clf is None: 
            warnings.warn("An estimator/classifier is needed for prediction."
                          " Got Nonetype.")
            raise EstimatorError("No estimator detected. Could not predict 'y'") 
        if Xt is None: 
            raise TypeError(
                "Test data 'Xt' is needed for prediction. Got nothing")
  
        # check estimator as callable object or ABCMeta classes
        if not hasattr(clf, '__call__') and  not inspect.isclass(clf)\
            and  type(clf.__class__)!=ABCMeta: 
            raise EstimatorError(
                f"{clf.__class__.__name__!r} is not an estimator/classifier."
                " 'y' prediction is aborted!")
            
        clf.fit(Xt, yt)
        ypred = clf.predict(Xt)
        
        if isinstance(Xt, (pd.DataFrame, pd.Series)):
            if index is None:
                index = Xt.index
                
    if isinstance(yt, pd.Series): 
        index = yt.index.astype('>U12')
    
    if index is None: 
        # take default values if  indexes are not given 
        index =np.array([i for i in range(len(yt))])

    if len(yt)!=len(ypred): 
        raise TypeError("'ypred'(predicted) and 'yt'(true target) sizes must"
                        f" be consistent. Expected {len(yt)}, got {len(ypred)}")
        
    return Xt, yt, index , clf ,  ypred 

def _check_labelxy (lablist, ar, ax, axis = 'x' ): 
    """ Assert whether the x and y labels given for setting the ticklabels 
    are consistent. 
    
    If consistent, function set x or y labels along the x or y axis 
    of the given array. 
    
    :param lablist: list, list of the label to set along x/y axis 
    :param ar: arraylike 2d, array to set x/y axis labels 
    :param ax: matplotlib.pyplot.Axes, 
    :param axis: str, default="x", kind of axis to set the label. 
    
    """
    warn_msg = ("labels along axis {axis} and arr dimensions must be"
                " consistent. Expects {shape}, got {len_label}")
    ax_ticks, ax_labels  = (ax.set_xticks, ax.set_xticklabels 
                         ) if axis =='x' else (
                             ax.set_yticks, ax.set_yticklabels )
    if lablist is not None: 
        lablist = is_iterable(lablist, exclude_string=True, 
                              transform =True )
        if not _check_consistency_size (
                lablist , ar[0 if axis =='x' else 1], error ='ignore'): 
            warnings.warn(warn_msg.format(
                axis = axis , shape=ar.shape[0 if axis =='x' else 1], 
                len_label=len(lablist))
                )
        else:
            ax_ticks(np.arange(0, ar.shape[0 if axis =='x' else 1]))
            ax_labels(lablist)
        
    return ax         
        
def plot2d(
    ar, 
    y=None,  
    x =None,  
    distance=50., 
    stnlist =None, 
    prefix ='S', 
    how= 'py',
    to_log10=False, 
    plot_contours=False,
    top_label='', 
    **baseplot_kws
    ): 
    """Two dimensional template for visualization matrices.
    
    It is a wrappers that can plot any matrice by customizing the position 
    X and y. By default X is considering as stations  and y the resistivity 
    log data. 
    
    Parameters 
    -----------
    ar: Array-like 2D, shape (M, N) 
        2D array for plotting. For instance, it can be a 2D resistivity 
        collected at all stations (N) and all frequency (M) 
    y: array-like, default=None
        Y-coordinates. It should have the length N, the same of the ``arr2d``.
        the rows of the ``arr2d``.
    x: array-like, default=None,  
        X-coordinates. It should have the length M, the same of the ``arr2d``; 
        the columns of the 2D dimensional array.  Note that if `x` is 
        given, the `distance is not needed. 

    distance: float 
        The step between two stations. If given, it creates an array of  
        position for plotting purpose. Default value is ``50`` meters. 
        
    stnlist: list of str 
        List of stations names. If given,  it should have the same length of 
        the columns M, of `arr2d`` 
       
    prefix: str 
        string value to add as prefix of given id. Prefix can be the site 
        name. Default is ``S``. 
        
    how: str 
        Mode to index the station. Default is 'Python indexing' i.e. 
        the counting of stations would starts by 0. Any other mode will 
        start the counting by 1.
     
    to_log10: bool, default=False 
       Recompute the `ar`  in logarithm  base 10 values. Note when ``True``, 
       the ``y`` should be also in log10. 
    plot_contours: bool, default=True 
       Plot the contours map. Is available only if the plot_style is set to 
       ``pcolormesh``. 
       
    baseplot_kws: dict, 
       All all  the keywords arguments passed to the property  
       :class:`watex.property.BasePlot` class. 
       
    Returns 
    -------
    axe: <AxesSubplot> object 
    
    Examples 
    -------- 
    >>> import numpy as np
    >>> import watex 
    >>> np.random.seed (42) 
    >>> data = np.random.randn ( 15, 20 )
    >>> data_nan = data.copy() 
    >>> data_nan [2, 1] = np.nan; data_nan[4, 2]= np.nan;  data_nan[6, 3]=np.nan
    >>> watex.view.mlplot.plot2d (data )
    <AxesSubplot:xlabel='Distance(m)', ylabel='log10(Frequency)[Hz]'>
    >>> watex.view.mlplot.plot2d (data_nan ,  plt_style = 'imshow', 
                                  fig_size = (10, 4))
    """
    #xxxxxxxxx update base plot keyword arguments
    for k  in list(baseplot_kws.keys()): 
        setattr (pobj , k, baseplot_kws[k])
        
    if y is not None: 
        if len(y) != ar.shape [0]: 
            raise ValueError ("'y' array must have an identical number " 
                              f" of row of 2D array: {ar.shape[0]}")
            
    if x is not None: 
        if len(x) != ar.shape[1]: 
            raise ValueError (" 'x' array must have the same number " 
                              f" of columns of 2D array: {ar.shape[1]}")

    d= distance or 1.
    try : 
         distance = float(distance) 
    except : 
        raise TypeError (
             f'Expect a float value not {type(distance).__name__!r}')
        
    # put value to log10 if True 
    if to_log10: 
        ar = np.log10 (ar ) # assume the resistivity data 
        y = np.log10(y) if y is not None else y # assume the frequency data 

    y = np.arange(ar.shape [0]) if y is None else y 
    x=  x  or np.arange(ar.shape[1]) * d
         
    stn = stnlist or make_ids ( x , prefix , how = how) 
    #print(stnlis)
    if stn is not None: 
        stn = np.array(stn)
        
    if not _check_consistency_size(stn, x, error ="ignore"): 
        raise ValueError("The list of stations and positions must be"
                         f" consistent. {len(stnlist)} and {len(x)}"
                         " were given respectively")
            
    # make figure 
    fig, axe = plt.subplots(1,figsize = pobj.fig_size, 
                            num = pobj.fig_num,
                            dpi = pobj.fig_dpi
                            )
    
    cmap = plt.get_cmap( pobj.cmap)
    
    if pobj.plt_style not in ('pcolormesh','imshow' ): 
        warnings.warn(f"Unrecognized plot style {pobj.plt_style!r}."
                      " Expect ['pcolormesh'|'imshow']."
                      " 'pcolormesh' ( default) is used instead.")
        pobj.plt_style= 'pcolormesh'
        
    if pobj.plt_style =='pcolormesh': 
        X, Y = np.meshgrid (x, y)
        # ar = np.ma.masked_where(np.isnan(ar), ar)
        #Zm = ma.array(Z,mask=np.isnan(Z))
        pkws = dict (vmax = np.nanmax (ar),
                     vmin = np.nanmin (ar), 
                     ) 
        
        if plot_contours: 
            levels = mticker.MaxNLocator(nbins=15).tick_values(
                    np.nanmin (ar), np.nanmax(ar) )
            # delete vmin and Vmax : not supported 
            # when norm is passed 
            del pkws ['vmin'] ; del pkws ['vmax']
            pkws ['norm'] = BoundaryNorm(
                levels, ncolors=plt.colormaps[pobj.cmap].N, clip=True)
            
        
        ax = axe.pcolormesh ( X, Y, np.flipud (ar),
                    shading= pobj.plt_shading, 
                    cmap =cmap, 
                    **pkws 
            )
        if plot_contours: 
             # contours are *point* based plots, so convert 
             # our bound into point centers
            dx, dy = 0.05, 0.05
            axe.contourf(X+ dx/2.,
                         Y + dy/2., np.flipud (ar) , levels=levels,
                         cmap=plt.colormaps[pobj.cmap]
                         )
    if pobj.plt_style =='imshow': 
        ax = axe.imshow (ar,
                    interpolation = pobj.imshow_interp, 
                    cmap =cmap,
                    aspect = pobj.fig_aspect ,
                    origin= 'lower', 
                    extent=(  np.nanmin(x),
                              np.nanmax (x), 
                              np.nanmin(y), 
                              np.nanmax(y)
                              )
            )
    # set axis limit 
    axe.set_ylim(np.nanmin(y), 
                 np.nanmax(y))
    axe.set_xlim(np.nanmin(x), 
                 np.nanmax (x))

    cbl = 'log_{10}' if to_log10 else ''
    axe.set_xlabel(pobj.xlabel or 'Distance(m)', 
                 fontdict ={
                  'size': 1.5 * pobj.font_size ,
                  'weight': pobj.font_weight}
                 )
      
    axe.set_ylabel(pobj.ylabel or  f"{cbl}Frequency$[Hz]$",
             fontdict ={
                     #'style': pobj.font_style, 
                    'size':  1.5 * pobj.font_size ,
                    'weight': pobj.font_weight})
    if pobj.show_grid is True : 
        axe.minorticks_on()
        axe.grid(color='k', ls=':', lw =0.25, alpha=0.7, 
                     which ='major')
    
   
    labex = pobj.cb_label or f"{cbl}App.Res$[Ω.m]$" 
    
    cb = fig.colorbar(ax , ax= axe)
    cb.ax.yaxis.tick_left()
    cb.ax.tick_params(axis='y', direction='in', pad=2., 
                      labelsize = pobj.font_size )
    
    cb.set_label(labex,fontdict={'size': 1.2 * pobj.font_size ,
                              'style':pobj.font_style})
    #--> set second axis 
    axe2 = axe.twiny() 
    axe2.set_xticks(range(len(x)),minor=False )
    
    # set ticks params to reformat the size 
    axe.tick_params (  labelsize = pobj.font_size )
    axe2.tick_params (  labelsize = pobj.font_size )
    # get xticks and format labels using the auto detection 
    _get_xticks_formatage(axe2, stn, fmt = 'S{:02}',  auto=True, 
                          rotation=pobj.rotate_xlabel )
    
    axe2.set_xlabel(top_label, fontdict ={
        'style': pobj.font_style,
        'size': 1.5 * pobj.font_size ,
        'weight': pobj.font_weight}, )
      
    fig.suptitle(pobj.fig_title,ha='left',
                 fontsize= 15* pobj.fs, 
                 verticalalignment='center', 
                 style =pobj.font_style,
                 bbox =dict(boxstyle='round',
                            facecolor ='moccasin')
                 )
   
    #plt.tight_layout(h_pad =1.8, w_pad =2*1.08)
    plt.tight_layout()  
    if pobj.savefig is not None :
        fig.savefig(pobj.savefig, dpi = pobj.fig_dpi,
                    orientation =pobj.orient)
 
    plt.show() if pobj.savefig is None else plt.close(fig=fig) 
    
    
    return axe        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        