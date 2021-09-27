# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Sep 16 11:31:38 2021
# This module is a set of plot for module viewer
# released under a MIT- licence.
"""
Created on Thu Sep 16 11:31:38 2021

@author: @Daniel03
"""

# import os
import re
import warnings
from typing import Generic, TypeVar, Iterable
 
import numpy as np 
import pandas as pd
import matplotlib as mpl 
import  matplotlib.pyplot  as plt
# from matplotlib.lines import Line2D 
# from sklearn.decomposition import PCA

# from .utils._watexlog import watexlog
from ..utils._watexlog import watexlog
from ..analysis.basics import categorize_flow 
from ..analysis.dimensionality import Reducers
import watex.utils.ml_utils as MLU 
import watex.utils.exceptions as Wex
# import watex.hints as Hints
import watex.utils.decorator as deco

T=TypeVar('T')
V=TypeVar('V', list, tuple, dict)
Array =  Iterable[float]
_logger=watexlog.get_watex_logger(__name__)


DEFAULTS_COLORS =[ 'g','r','y', 'blue','orange','purple', 'lime','k', 'cyan', 
                  (.6, .6, .6),
                  (0, .6, .3), 
                  (.9, 0, .8),
                  (.8, .2, .8),
                  (.0, .9, .4)
                 ]

DEFAULTS_MARKERS =['o','^','x', 'D', '8', '*', 'h', 'p', '>', 'o', 'd', 'H']

def biPlot(self, score, coeff, y, y_classes=None, markers=None, colors=None):
    """
    The biplot is the best way to visualize all-in-one following a PCA analysis.
    There is an implementation in R but there is no standard implementation
    in python. 
    
    Originally written by 
        
        Author: Serafeim Loukas, serafeim.loukas@epfl.ch
    
    and referenced to :href:`<https://towardsdatascience.com/...-python-7c274582c37e>`
    Func is edited and add some new parameters to customize plots: 
        
    :param score: the projected data
    :param coeff: the eigenvectors (PCs)
    :param y: the class labels
    :param y_classes: class categories
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
        colors = DEFAULTS_COLORS
        colors = [colors[c] for c in range(len(y_classes))]
    if markers is None:
        markers=DEFAULTS_MARKERS 
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
    
    
class MLPlots: 
    """ Mainly deals with Machine learning metrics, 
    dimensional reduction plots and else. 
    
    Composed of decomposition tips, metrics and else.
 
    Hold others optionnal informations: 
        
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
                        *default* is r"$\blacktriangledown$"
    ms                  size of marker in points. *default* is 5
    mstyle              style  of marker in points. *default* is ``o``.
    x_minorticks        minortick according to x-axis size and *default* is 1.
    y_minorticks        minortick according to y-axis size and *default* is 1.
    font_size           size of font in inches (width, height)
                        *default* is 3.
    font_style          style of font. *default* is ``italic``
                        
    bins                histograms element separation between two bar. 
                         *default* is ``10``. 
    xlim                limit of x-axis in plot. *default* is None 
    ylim                limit of y-axis in plot. *default* is None 
    ==================  =======================================================
    """
    def __init__(self, savefig =None, figsize =None, **kws ): 
        self._logging= watexlog().get_watex_logger(self.__class__.__name__)
        
        self.savefig = kws.pop('savefig', None)
        
        self.fig_num= kws.pop('fig_num', 1)
        self.fig_size = kws.pop('fig_size', (12, 8))
        self.fig_dpi =kws.pop('fig_dpi', 300)
        self.fig_legend= kws.pop('fig_legend_kws', None)
        self.fig_orientation =kws.pop('fig_orientation','landscape')
        self.fig_title =kws.pop('title', None)
        
        self.font_size =kws.pop('font_size',3.)
        self.font_style=kws.pop('font_style', 'italic')
        self.font_weight=kws.pop('font_weight', 'bold')
        
        self.fs =kws.pop('fs', 5.)
        
        self.ms =kws.pop('ms', 3.)
        self.marker_style =kws.pop('marker_style', 'D')
        self.marker_facecolor=kws.pop('markerfacecolor', 'yellow')
        self.marker_edgecolor=kws.pop('markeredgecolor', 'cyan')
        self.marker_edgewidth = kws.pop('markeredgewidth', 3.)
        
        self.lc = kws.pop('lc', 'k')
        self.font_weight =kws.pop('font_weight', 'bold')
        self.ls= kws.pop('ls', '-')
        self.lw =kws.pop('lw', 1)
        self.alpha = kws.pop('alpha', 0.5)
        
        self.bins = kws.pop('bins', 10)
        
        self.xlim =kws.pop('xlim', None )
        self.ylim=kws.pop('y_lim', None) 
        self.xlabel = kws.pop('xlabel', None)
        self.ylabel =kws.pop('ylabel', None)
        self.rotate_xlabel =kws.pop('rotate_xlabel', None)
        self.rotate_ylabel=kws.pop('rotate_ylabel',None )
        
        self.leg_kws = kws.pop('leg_kws', dict())
        self.plt_kws = kws.pop('plt_kws', dict())
        
        # precision(p) and recall(r) style(s) and color (c)
        self.rs =kws.pop('rs', '--')
        self.ps =kws.pop('ps', '-')
        self.rc =kws.pop('rc', (.6, .6, .6))
        self.pc =kws.pop('pc', 'k')
    
        self.s = kws.pop('s', None)
        #show grid 
        self.show_grid = kws.pop('show_grid',False)
        self.galpha =kws.pop('galpha', 0.5)
        self.gaxis =kws.pop('gaxis', 'both')
        self.gc =kws.pop('gc', 'k')
        self.gls =kws.pop('gls', '--')
        self.glw =kws.pop('glw', 2.)
        self.gwhich = kws.pop('gwhich', 'major')
        
        
    def PCA_(self,
             X:[Array],
             y:Array,
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
            
        """ Plot PCA component analysis using :class:`~sklearn.decomposition`. 
        
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
            plot_dict ={'y_colors':DEFAULTS_COLORS, 
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
            
    @deco.docstring(MLU.Metrics.precisionRecallTradeoff, start='Parameters', 
                    end = 'Examples')        
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
        
        ..see also::  For parameter definitions, please refer to
            :doc:`~watex.utils.ml_utils.Metrics.PrecisionRecallTradeoff`
            for further details.
            
        Parameters
        ---------
        kind: str 
            kind of plot. Plot precision-recall vs thresholds (``vsThreshod``)
            or precision vs recall (``vsThreshod``). Default is 
            ``vsThreshod``
            
        Examples
        ---------

            >>> from sklearn.linear_model import SGDClassifier
            >>> from watex.datasets.data_preparing import X_train_2
            >>> from watex.datasets import y_prepared
            >>> sgd_clf = SGDClassifier(random_state= 42)
            >>> mlObj= MLPlots(lw =3., pc = 'k', rc='b', ps='-', rs='--')
            >>> mlObj.PrecisionRecall(clf = sgd_clf,  X= X_train_2, 
            ...                y = y_prepared, classe_=1, cv=3,
            ...                 kind='vsRecall')
        """
        # call precision 
        prtObj = MLU.Metrics().precisionRecallTradeoff(
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
        """ Plot receiving operating characteric(ROC) classifiers. 
        
        To plot multiples classifiers, provide a list of classifiers. 
        
        Parameters 
        ----------
        clf: callables
            classifier or estimators. To use multiple classifier 
            set a list of classifer with their specific mmethods. 
            For instance::
                
                [('SDG', SGDClassifier,"decision_function" ), 
                 ('FOREST',RandomForestClassifier,"predict_proba")]
                
        X: ndarray, 
            Training data (trainset) composed of n-features.
            
        y: array_like 
            labelf for prediction. `y` is binary label by defaut. 
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
            Ciuld be ``decison_funcion`` or ``predict_proba`` so 
            Scikit-Learn classifuier generally have one of the method. 
            Default is ``decision_function``.
            
        roc_kws: dict 
            roc_curve additional keywords arguments
            
        See also
        --------
        
            `ROC_curve` deals wuth optional and positionals keywords arguments 
            of :meth:`~watex.utlis.ml_utils.Metrics.precisionRecallTradeoff`
            
        Examples 
        --------
        
            >>> from sklearn.linear_model import SGDClassifier
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from watex.datasets.data_preparing import X_train_2
            >>> from watex.datasets import  y_prepared
            >>> sgd_clf = SGDClassifier(random_state= 42)
            >>> forest_clf =RandomForestClassifier(random_state=42)
            >>> mlObj= MLPlots(lw =3., lc=(.9, 0, .8), font_size=7
            >>> clfs =[('sgd', sgd_clf, "decision_function" ), 
             ...      ('forest', forest_clf, "predict_proba")]
            >>> mlObj.ROC_curve_(clf = clfs,  X= X_train_2, 
            ...                      y = y_prepared, classe_=1, cv=3,)
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
        rocObjs =[ MLU.Metrics().ROC_curve( 
                                clf=_clf,
                                X=X, 
                                y=y, 
                                cv =cv, 
                                classe_=classe_, 
                                method =meth, 
                                cross_val_pred_kws=cross_val_pred_kws,
                                **roc_kws) for (name, _clf, meth) in clf]
        # create figure obj 
        fig = plt.figure(figsize = self.fig_size)
        ax = fig.add_subplot(1,1,1)
        DEFAULTS_COLORS[0] = self.lc 

        for ii, (name, _clf, _)  in enumerate( clf): 
            ax.plot(rocObjs[ii].fpr, 
                    rocObjs[ii].tpr, 
                    label =name, 
                    color =DEFAULTS_COLORS[ii], 
                    linewidth = self.lw)
            
            
        if self.xlabel is None: self.xlabel ='False Positive Rate'
        if self.ylabel is None: self.ylabel ='True Positive Rate'
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
             self.leg_kws['loc']='upper left'
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
        all instances to visaulize data.
        
        Parameters
        ---------
        X: ndarray(n, 2), pd.DataFrame
            array composed of n-isnatnces of two features. First features is
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
        >>> from watex.datasets import X, X_test
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
                        raise Wex.WATexError_plot_featuresinputargument(
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
            raise Wex.WATexError_value('NoneType could not be plotted.'
                                       ' need `x-axis` value for plot.')
        if y is None: 
            raise Wex.WATexError_value('NoneType could not be plotted.'
                                       ' Need `y-axis` value for plot.')
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
              
if __name__=='__main__': 

    # from sklearn.linear_model import SGDClassifier
    # from sklearn.ensemble import RandomForestClassifier
    from watex.datasets import fetch_data 
    import  watex.utils.ml_utils as mfunc
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

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        