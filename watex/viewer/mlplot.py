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
 

from .utils._watexlog import watexlog
from .utils.ml_utils import DimensionReduction
from .analysis.features import categorize_flow 

T=TypeVar('T')
V=TypeVar('V', list, tuple, dict)
Array =  Iterable[float]
_logger=watexlog.get_watex_logger(__name__)

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
        colors = ['g','r','y', 'blue', 'orange', 'purple', 'lime', 'k', 'cyan']
        colors = [colors[c] for c in range(len(y_classes))]
    if markers is None:
        markers=['o','^','x', 'D', '8', '*', 'h', 'p', '>'] 
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
                 fontsize=10)

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
    """ Mainly deals with Machine learning plots. 
    
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
        self.marker_style =kws.pop('maker_style', 'D')
        self.marker_facecolor=kws.pop('markefacecolor', 'yellow')
        self.marker_edgecolor=kws.pop('markeredgecolor', 'cyan')
        self.marker_edgewidth = kws.pop('markeredgewidth', 3.)
        
        self.lc = kws.pop('color', 'k')
        self.font_weight =kws.pop('font_weight', 'bold')
        self.ls= kws.pop('ls', '-')
        self.lw =kws.pop('lw', 1)
        self.alpha = kws.pop('alpha', 0.5)
        
        self.bins = kws.pop('bins', 10)
        
        self.xlim =kws.pop('xlim', None )
        self.ylim=kws.pop('y_lim', None) 
        
    
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
        
        PCA indetifies the axis that accounts for the largest amount of 
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
            
        :param y_type: type of features `y`. Could be ``cat`` for catgorial 
                features or ``num`` for numerical features. If `y` is numerical 
                features and need to be converted into a cetegorial features 
                set `y_type` to ``num`` to force the conversion of `y` into 
                a categorial features.
        
        :param replace_y: customize the encoded values by providing a new list
                of vategorized values.
                
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
                                    y_classes =['FR0', 'FR1', 'FR2', 'FR3'],
                                    biplot =False)
        """
        if plot_dict is None: 
            plot_dict ={'y_colors':['navy',
                                    'g',
                                    'r',
                                    'orange',
                                    'purple',
                                    (.9, 0., .8), 
                                    'bleue', 
                                    'cyan', 
                                    'lime', 
                                    (.0, .9, .4)], 
                        's':100.}
            
        def mere_replace(y_, y_val, y_clas): 
            """ Replace the numerical values (generaly encoded values) to 
            desired categorial features names."""
            ynew = np.zeros_like(y, dtype ='>U12')
            for i, val in enumerate(y_):
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
                
            y = mere_replace(y_=y, y_val=y_values, y_clas=y_classes)                    
            
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
        pca= DimensionReduction().PCA(X,
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
        pca_axis_1 = feature_importances_[0][1][0] 
        pca_axis_2 = feature_importances_[1][1][0]
        # Extract the name of the  values of the first components 
        # and second components in percentage.
        pca_axis_1_ratio = np.around( feature_importances_[0][2][0],2) *1e2
        pca_axis_2_ratio = np.around(feature_importances_[1][2][0],2) *1e2
     
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
                           markersize = self.ms * self.fs)
        
        lineh =plt.Line2D ((-max_lim, max_lim), (0, 0),
                           color = self.lc, 
                           linewidth = self.lw,
                           linestyle = self.ls ,
                           marker = self.marker_style,
                           markeredgecolor = self.marker_edgecolor,
                           markeredgewidth = self.marker_edgewidth,
                           markerfacecolor = self.marker_facecolor,
                           markersize = self.ms * self.fs)
        
        #Create string label from pca_axis_1
        x_axis_str = pc1_label +':'+ str(pca_axis_1) +' ({}%)'.format(
            pca_axis_1_ratio )
        y_axis_str = pc2_label +':' + str(pca_axis_2) +' ({}%)'.format(
            pca_axis_2_ratio )
        
        ax.set_xlabel( x_axis_str,
                      color='k', 
                      fontsize = self.font_size * self.fs)
        ax.set_ylabel(y_axis_str,
                      color='k',
                      fontsize = self.font_size * self.fs)
        ax.set_title('ACP', 
                     fontsize = (self.font_size +1) * self.fs)
        ax.add_artist(linev)
        ax.add_artist(lineh)
        ax.legend(y_classes)
        ax.grid(color=self.lc, linestyle=self.ls, linewidth=self.lw/10)
        
        plt.show()
        
        if self.savefig is not None :
            plt.savefig(self.savefig,
                        dpi=self.fig_dpi,
                        orientation =self.fig_orientation)    

if __name__=='__main__': 

    from .datasets.data_preparing import X_train_2
    from .datasets import y_prepared   
    
    pcaObj= MLPlots()
    pcaObj.PCA_(X= X_train_2, y=y_prepared, replace_y=True, 
                            y_classes =['FR0', 'FR1', 'FR2', 'FR3'],
                            biplot =False)
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        