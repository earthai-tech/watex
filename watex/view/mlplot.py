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
from scipy.cluster.hierarchy import dendrogram # set_link_color_palette 

import matplotlib as mpl 
import matplotlib.pyplot  as plt
import matplotlib.ticker as mticker
from matplotlib import cm 

from .._watexlog import watexlog
from .._docstring import ( 
    _core_docs, 
    DocstringComponents
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
    mean_squared_error
    ) 
from ..exceptions import ( 
    # PlotError, 
    NotFittedError , 
    LearningError
    )
from ..metrics import ( 
    precision_recall_tradeoff, 
    ROC_curve, 
    confusion_matrix
    )
from ..property import BasePlot 
from ..utils.exmath import linkage_matrix 
from ..utils.hydroutils import check_flow_objectivity 
from ..utils.coreutils import _is_readable 
from ..utils.funcutils import ( 
    # _assert_all_types,
    # is_iterable,
    reshape, 
    to_numeric_dtypes, 
    smart_strobj_recognition, 
    repr_callable_obj , 
    # str2columns
    )
from ..utils.mlutils import ( 
    exporttarget , 
    selectfeatures, 
    cattarget, 
    # existfeatures, 
    projection_validator 
    )
from ..utils.plotutils import  ( 
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
    F,
    List,
    ArrayLike, 
    NDArray,
    DType, 
    DataFrame, 
    )
_logger=watexlog.get_watex_logger(__name__)

#-----
# Add specific params to docs 
_eval_params = dict( 
    objective="""
objective: str, default=None, 
    The purpose of dataset, what probem do we intend to solve ?  
    Originally the package was designed for flow rate prediction. Thus,  
    if the `objective`, plot will behave like the flow rate prediction 
    purpose and in this case, some condition of target values need to  
    be fullfilled. default is ``None``. Furthermore, if the objective 
    is set to ``flow``, `label_values`` as well as `litteral_classes`` 
    need to be supplied to right encoded the target according to the 
    hydraulic system requirement during the campaign for drinking water 
    supply. For any other purpose for the dataset, keep the objective to 
    ``None``.     
    """
    )

_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"], 
    base=DocstringComponents(_eval_params), 
    )
#-------


class EvalPlot(BasePlot): 

    def __init__(self, tname:str =None, 
                 encode_labels: bool=False,
                 scale: str = None, 
                 cv: int =None, 
                 objective:str=None, 
                 prefix: str=None, 
                 label_values:List[int]=None, 
                 litteral_classes: List[str]=None, 
                 **kws ): 
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
        # styles properties
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
                         bbox_inches = 'tight'
                         )
        plt.show() if self.savefig is None else plt.close () 
        
    def fit(self, X: NDArray |DataFrame =None, y:ArrayLike =None, 
            **fit_params ): 
        """
        Fit data and populate the arguments for plotting purposes. 
        
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
        """ Transform the data and keep only the numerical features. 
        
        It is not convenient to use `transform` if user want to keep 
        categorical values in the Array 
        
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
    
        # self.X = to_numeric_dtypes(self.X , columns = columns )
        # self.X = selectfeatures(self.X, include ='number')
        # if len (self.X.columns) ==0 : 
        #     raise TypeError(
        #         " The module {self.__class__.__name__!r } expects dataframe "
        #         " 'X' with numerical features only. ")
        # keep columns 
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

        y =self.y.copy()  
        
        if y is None : 
            warnings.warn("Expect a target array. Missing y(target)"
                          " is not allowed.")
            raise TypeError (" NoneType y (target) can be categorized.")
            
        
        if objective =='flow':
            y, classes = check_flow_objectivity(y,values, classes) 
        else : 
            
            if self.target_ is not None: 
                y = self.target_ 
            else: y = pd.Series (y, name='none')
            
            if values is not None: 
                y =  cattarget(y , labels = values )
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
            plot_dict:Generic[V] = None,
            **pca_kws
    ): 
        """ Plot PCA component analysis using :class:`~.sklearn.decomposition`. 
        
        PCA indentifies the axis that accounts for the largest amount of 
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
            plot_dict ={'y_colors':D_COLORS, 
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
        X_reduced = pca.X # the components
        n_axes = n_axes or pca.n_axes # for consistency
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
            
            mpl.rcParams.update(mpl.rcParamsDefault) # reset ggplot style
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
        # created a dataframe concatenate reduced dataframe + y_target
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
        # ranged like [('pc1',['shape', 'power',...],
        #   [-0.85927608, -0.35507183,...] ),
                    # ('pc2', ['sfi', 'power', ...],
                    #[ 0.50104756,  0.4565256 ,... ), ...]
        # print('pc1axes =', pca1_ix, 'pc1_label=', pc1_label)
        # print('pc2axes =', pca2_ix, 'pc2_label=', pc2_label)
        pca_axis_1 = feature_importances_[pca1_ix][1][0] 
        pca_axis_2 = feature_importances_[pca2_ix][1][0]
        # Extract the name of the  values of the first 
        # components and second components in percentage.
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
        
        plt.show()
        
        if self.savefig is not None :
            plt.savefig(self.savefig,
                        dpi=self.fig_dpi,
                        orientation =self.fig_orientation
                        )  
    
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
    ): 
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
        elif kind.lower().find('vsrec')>=0: 
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

        plt.show()
        
        if self.savefig is not None :
            plt.savefig(self.savefig,
                        dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
            
        return self 
    
    def plotROC(
        self, 
        clfs,
        label: int |str, 
        method: Optional[str]=None,
        cvp_kws:dict=None,
        **roc_kws
        ):
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
        It is import to know whether the method 'predict_proba' is valid for 
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
        >>> # plot the Precision-recall tradeoff  
        >>> b.plotPR(sgd_clf , label =1) # class=1
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
            label=label, method =meth, cvp_kws=cvp_kws,**roc_kws
            ) 
                  for (name, _clf, meth) in clfs
                  ]
        # create figure obj 
        fig = plt.figure(figsize = self.fig_size)
        ax = fig.add_subplot(1,1,1)
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
        
        plt.show()
        
        if self.savefig is not None :
            plt.savefig(self.savefig,
                        dpi=self.fig_dpi,
                        orientation =self.fig_orientation)
        return self 

    @docSanitizer()
    def plotConfusionMatrix(
        self, 
        clf, 
        *, 
        plottype ='map', 
        labels=None, 
        matshow_kws=dict(), 
        **conf_mx_kws
        ): 
        """ Plot confusion matrix for error analysis
        
        Look of a representation of the confusion matrix using 
        Matplotlib matshow.
        
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
        
        labels = labels or self.litteral_classes 
        
        if labels is not None: 
            #check the length of y and compare to y unique 
            cat_y = np.unique(self.y)
            if isinstance(labels, str) or len(labels)==1: 
                warnings.warn(
                   f"One label is given, need {len(cat_y)!r}. Can not be"
                    f" used to format {cat_y!r}"
                    )
                self._logging.debug(
                    f"Only one label is given. Need {len(cat_y)!r}"
                    'instead as the number of categories.')
                ylabel =None 
                
            type_y= isinstance(labels, (list, tuple, np.ndarray))
            if type_y:
                if len(cat_y) != len(labels): 
                    warnings.warn(
                        f" {'are' if len(ylabel)>1 else 'is'} given."
                        f"Need {len(cat_y)!r} instead.")
                    self._logging.debug(
                        f" {'are' if len(ylabel)>1 else 'is'} given."
                        f"Need {len(cat_y)!r} instead.")
                    ylabel =None 
                    
        # get yticks one it is a classification prof
        confObj =confusion_matrix(clf=clf,
                                X=self.X,
                                y=self.y,
                                cv=self.cv,
                                **conf_mx_kws
                                )
        
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
        
        if labels is not None: 
            ax.set_xticks(np.unique(self.y))
            ax.set_xticklabels(labels)
            ax.set_yticks(np.unique(self.y))
            ax.set_yticklabels(labels)
            
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

EvalPlot.__doc__="""\
Metric and dimensionality Evaluatation Plots  

Inherited from :class:`BasePlot`. Dimensional reduction and metrics 
plots. The class works only with numerical features. 

Parameters 
-----------
{params.core.X}
{params.core.y}
{params.core.tname}
{params.base.objective}
    
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
    target so it is recommended to set the `label_values` as list of 
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

Returns 
---------
{returns.self}


Attributes 
----------- 

Hold others optional attributes as: 
    
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
yp_marker           style  of marker in  of `Prediction` points. 
                        *default* is ``o``.
yp_markerfacecolor  facecolor of the `Predicted` label marker. 
                    *default* is ``k``
yp_markeredgecolor  edgecolor of the `Predicted` label marker. 
                    *default* is ``r``.
yp_markeredgewidth  width of the `Predicted`label marker. *default* is ``2``.  
==================  =======================================================
   
Notes 
--------
This module works with numerical data  i.e if the data must contains the 
numerical features only. If categorical values are included in the 
dataset, they should be  removed and the size of the data should be 
chunked during the fit methods. 

""".format(
    params=_param_docs,
    returns= _core_docs["returns"],
)

# create a show class to 
_b= EvalPlot () 
pobj = type ('Plot', (BasePlot, ), {**_b.__dict__} ) 
 
def plotProjection(
    X: DataFrame | NDArray, 
    Xt: DataFrame | NDArray =None, *, 
    columns: List[str] =None, 
    test_kws: dict =None,  
    **baseplot_kws 
    ): 
    """ Visualize dataset. 
    
    Since there is geographical information(latitude/longitude or
    eating/northing), it is a good idea to create a scatterplot of 
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
        columns is usefull what a dataframe is given  with a dimension size 
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
        
    bplot_kws: dict, 
        All all  the keywords arguments passed to the peroperty  
        :class:`watex.property.BasePlot` class. 
        
    Examples
    --------
        
    >>> from watex.datasets import fetch_data 
    >>> from watex.view.mlplot import plotProjection 
    >>> X, Xt, *_ = fetch_data ('bagoue', split_X_y =True, as_frame =True) 
    >>> plot_kws = dict (fig_size=(8, 12),
                     lc='k',
                     marker='o',
                     lw =3.,
                     font_size=15.,
                     xlabel= 'east',
                     ylabel='north' , 
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
    >>>  plotProjection( X, Xt , columns= ['east', 'north'], 
                        trainlabel='train location', 
                        testlabel='test location', **plot_kws
                       )

    """
    
    trainlabel =baseplot_kws.pop ('trainlabel', None )
    testlabel =baseplot_kws.pop ('testlabel', None  )
    
    for k  in list(baseplot_kws.keys()): 
        setattr (pobj , k, baseplot_kws[k])
        
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
    
    xname = xname or pobj.xlabel 
    yname =yname or pobj.ylabel 
    
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
    
    plt.show()
    
    if pobj.savefig is not None :
        plt.savefig(pobj.savefig,
                    dpi=pobj.fig_dpi,
                    orientation =pobj.fig_orientation)

                
def model(
    self, 
    y_, 
    ypred=None,
    *, 
    clf=None, 
    X_=None, 
    predict =False, 
    prefix=None, 
    index =None, 
    fill_between=False, 
    ylabel=None 
    ): 
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
        

    
def plotLearningCurve(self, clf, X, y, test_size=0.2, scoring ='mse',
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

    cv: cv: float or int,    
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
        
    Examples
    ---------
    (1) -> Use random data
    >>> import numpy as np 
    >>> >>> from watex.view.mlplot import plotBindDendro2Heatmap
    >>> np.random.seed(123) 
    >>> variable =['X', 'Y', 'Z'] ; labels =['ID_0', 'ID_1', 'ID_2',
                                             'ID_3', 'ID_4']
    >>> X= np.random.random_sample ([5,3]) *10 
    >>> df =pd.DataFrame (X, columns =variable, index =labels)
    >>> plotBindDendro2Heatmap (df, )
    
    (2) -> Use Bagoue data 
    >>> from watex.datasets import load_bagoue  
    >>> X, y = load_bagoue (as_frame=True )
    >>> X =X[['magnitude', 'power', 'sfi']].astype(float) # 
    >>> plotBindDendro2Heatmap (X )
    
    
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

    axm.xaxis.set_major_locator(mticker.FixedLocator(xticks_loc))
    axm.xaxis.set_major_formatter(mticker.FixedFormatter(
        [''] + list (df_rowclust.columns)))
    
    axm.yaxis.set_major_locator(mticker.FixedLocator(yticks_loc))
    axm.yaxis.set_major_formatter(mticker.FixedFormatter(
        [''] + list (df_rowclust.index)))
    
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

    
def plotSilhouette (X, labels, metric ='euclidean', **kwds ):
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
        classes: List | ArrayLike [str] =None,
        markers: str | List [str]  =None, 
        colors: str | List [str ] =None, 
        **baseplot_kws 
)-> None :
    """
    The biplot is the best way to visualize all-in-one following a PCA analysis.
    There is an implementation in R but there is no standard implementation
    in python. 

    Parameters  
    -----------
    score: NDAarray 
        the projected data scores 
    
    coeff: Array-like 
        the eigenvectors of the PCA .
    
    y: Array-like, 
        the target composing the class labels.
    
    classes: list or int, 
        class categories or class labels 
        
    markers: str, 
        Matplotlib list of markers for plotting  classes.
    
    colors: str, 
        Matplotlib list of colors to customize plots 
    
    References 
    -----------
    Originally written by `Serafeim Loukas`_, serafeim.loukas@epfl.ch and 
    edited for plot customizing. 
    
    .. _Serafeim Loukas: https://towardsdatascience.com/...-python-7c274582c37e>
    
    """
   
    xs = score[:,0] # projection on PC1
    ys = score[:,1] # projection on PC2
    n = coeff.shape[0] # number of variables
    plt.figure(figsize=self.fig_size, #(10,8),
               dpi=self.fig_dpi #100
               )
    if classes is None: 
        classes = np.unique(y)
    if colors is None:
        colors = D_COLORS
        colors = [colors[c] for c in range(len(classes))]
    if markers is None:
        markers=D_MARKERS 
        markers = [markers[m] for m in range(len(classes))]
    for s,l in enumerate(classes):
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
    
    if self.ssavefig is not None: 
        savefigure (plt, self.savefig, dpi = self.fig_dpi )
    


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
 


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        