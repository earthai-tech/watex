# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Additional plot utilities. 
"""
from __future__ import annotations 
import os
import re 
import copy 
import datetime 
import warnings
import itertools 
import numpy as np
import pandas as pd 
import matplotlib as mpl 
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors
import matplotlib.transforms as transforms 
from matplotlib import gridspec 
import seaborn as sns 
from scipy.cluster.hierarchy import ( 
    dendrogram, ward 
    )
import scipy.sparse as sp
import matplotlib.pyplot as plt

from ..exceptions import ( 
    TipError, 
    PlotError, 
    )
from .baseutils import adjust_phase_range 
from .funcutils import  ( 
    _assert_all_types,
    is_iterable, 
    to_numeric_dtypes, 
    make_obj_consistent_if, 
    str2columns, 
    is_in_if, 
    is_depth_in, 
    reshape, 
    interpolate_grid, 
    )
from .validator import  ( 
    _check_array_in  , 
    _is_cross_validated,
    assert_xy_in, 
    get_estimator_name,
    check_array, 
    check_X_y,
    check_y,
    check_consistent_length, 
    check_is_fitted , 
    _assert_z_or_edi_objs, 
    )
from ._dependency import import_optional_dependency 
from ..decorators import nullify_output
try: 
    from ..exlib.sklearn import ( 
        learning_curve ,   
        confusion_matrix, 
        RandomForestClassifier, 
        LogisticRegression, 
        MinMaxScaler, 
        SimpleImputer, 
        KMeans, 
        silhouette_samples, 
        roc_curve, 
        roc_auc_score, 
        ) 
except : pass 
 
try : 
    from yellowbrick.classifier import ConfusionMatrix 
except: pass 

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
#----

def plot_tensors2(
    z_or_edis_obj_list, /, 
    station='S00', 
    plot_z=False, 
    show_error_bars=True,  
    **kwargs
   ):
    """
    Plot resistivity and phase tensors or the real and imaginary impedance.
    
    This function plots the apparent resistivity and phase or the real and imaginary
    parts of impedance tensors for a given station from a list of Z or EDI objects. 
    It supports extensive customization for the plots including the option to show or 
    hide error bars, control over color schemes, marker styles, and much more.
    
    Parameters
    ----------
    z_or_edis_obj_list : list of :class:`watex.edi.Edi` or :class:`watex.externals.z.Z`
        A collection of EDI- or Impedance tensor objects. The list can contain objects
        directly representing impedance tensors or EDI objects from which impedance
        tensors can be extracted.
    station : int or str, default 'S00'
        The station to visualize. Can be specified as an index (int) or as a string
        including the station name or number. For example, 'S00' or 0 for the first
        station. The counting starts from 0.
    plot_z : bool, default False
        If True, visualize the real and imaginary parts of the impedance tensors (Z).
        If False, visualize the apparent resistivity and phase tensors.
    show_error_bars : bool, default True
        Whether to show error bars in the plots. If False, error bars are omitted
        for a cleaner visualization.
    **kwargs : dict
        Additional keyword arguments for plot customization. These can include
        matplotlib parameters for markers, lines, colors, and other plot attributes.
    
    Returns
    -------
    object
        The Z object for the specified station, containing the impedance tensor data
        and any computed properties like resistivity and phase.
    
    Examples
    --------
    Plotting the apparent resistivity and phase for the fourth station from a list
    of EDI objects:
    
    >>> import watex as wx
    >>> edi_objects = wx.fetch_data('edis', samples=17, return_data=True)
    >>> wx.utils.plotutils.plot_tensors(edi_objects, station=3)
    
    Plotting the real and imaginary parts of the impedance tensor for the first station,
    without error bars:
    
    >>> wx.utils.plotutils.plot_tensors(edi_objects, station='S00', zplot=True,
                                        show_error_bars=False)
    
    Notes
    -----
    This function is a part of the watex visualization utilities and requires a
    matplotlib environment to display the plots. Ensure that your environment
    supports graphical output or adjust your environment accordingly.
    
    See Also
    --------
    watex.methods.EM : Class for electromagnetic method processing.
    watex.utils.plotutils.plot_errorbar : Helper function to plot error bars.
    """
    station_index = _get_station_index(station)
    obj_type, z_obj = _get_obj_type_and_data(z_or_edis_obj_list, station_index)
    fig, ax_list = _initialize_plot_layout(**kwargs)
    plot_res, plot_res_err, plot_phase, plot_phase_err = _prepare_plot_data(
        z_obj, plot_z)
    
    freq, plot_res, plot_res_err, plot_phase, plot_phase_err= _filter_and_adjust_data(
        z_obj._freq, plot_res, plot_res_err, plot_phase, plot_phase_err, **kwargs)
    
    _plot_data(fig, ax_list, freq, plot_res, plot_res_err, 
               plot_phase, plot_phase_err, show_error_bars,
               plot_z=plot_z,  **kwargs)
    
    plt.show()
    
    return z_obj

def _get_station_index(station):
    """Extract and return the station index from the station identifier."""
    match = re.search(r'\d+', str(station), flags=re.IGNORECASE)
    if match is None:
        raise TypeError("Station should be or include a position number.")
    return int(match.group())

def _get_obj_type_and_data(z_or_edis_obj_list, station_index):
    """Determine the object type (EDI or Z) and retrieve the relevant data object."""
    if station_index >= len(z_or_edis_obj_list):
        raise ValueError(f"Station index out of range. Only {len(z_or_edis_obj_list)}"
                         " stations available.")
    
    data_obj = z_or_edis_obj_list[station_index]
    obj_type = 'EDI' if hasattr(data_obj, 'Z') else 'Z'
    z_obj = data_obj.Z if obj_type == 'EDI' else data_obj
    
    return obj_type, z_obj

def _initialize_plot_layout(**kwargs):
    """Initialize and return the figure and axes list for plotting."""
    fig_size = kwargs.pop('fig_size', [6, 6])
    fig_dpi = kwargs.pop('dpi', 300)
    fig = plt.figure(figsize=fig_size, dpi=fig_dpi)
    plt.clf()
    
    gs = gridspec.GridSpec(2, 4, wspace=kwargs.get('subplot_wspace', .3))
    ax_list = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(4)]
    return fig, ax_list

def _prepare_plot_data(z_obj, plot_z):
    """Prepare plot data for resistivity, phase, and their errors."""
    # Placeholder function for computing resistivity and phase
    z_obj.compute_resistivity_phase()  # Assume this function exists within z_obj
    
    plot_res = z_obj.resistivity if not plot_z else abs(z_obj.z.real)
    plot_res_err = z_obj.resistivity_err if not plot_z else abs(z_obj.z_err.real)
    plot_phase = z_obj.phase if not plot_z else abs(z_obj.z.imag)
    plot_phase_err = z_obj.phase_err if not plot_z else abs(z_obj.z_err.imag)
    
    return plot_res, plot_res_err, plot_phase, plot_phase_err

def _filter_and_adjust_data(freq, plot_res, plot_res_err, plot_phase, plot_phase_err, **kwargs):
    """ filters and adjusts the data according to specified limits."""
    phase_limits = kwargs.pop('phase_limits', None)
    period_limits = kwargs.pop('period_limits', None)
    freq_limits = kwargs.pop('freq_limits', None)
    mod_base = kwargs.pop("mod_base", 360)  

    # Validate and prioritize frequency limits over period limits
    if freq_limits is not None and period_limits is not None:
        print("Both freq_limits and period_limits are provided. Using freq_limits for filtering.")
        period_limits = None  # Ignore period_limits if freq_limits is provided
    
    if period_limits is not None:
        if not isinstance(period_limits, (list, tuple)) or len(period_limits) != 2 or not all(
                isinstance(x, (int, float)) for x in period_limits):
            raise ValueError("period_limits must be a tuple or list of two numerical"
                             " values (min_period, max_period).")
        
        min_period, max_period = sorted(period_limits)  # Ensure min < max
        # Convert period limits to frequency limits
        freq_limits = 1/max_period, 1/min_period

    # Filter data by frequency limits
    if freq_limits is not None:
        if not isinstance(freq_limits, (list, tuple)) or len(freq_limits) != 2 or not all(
                isinstance(x, (int, float)) for x in freq_limits):
            raise ValueError(
                "freq_limits must be a tuple or list of two numerical values (min_freq, max_freq).")
        
        min_freq, max_freq = sorted(freq_limits)  # Ensure min < max
        freq_mask = (freq >= min_freq) & (freq <= max_freq)
        
        # Select frequencies within the limits
        new_freq = freq[freq_mask]
        # Use the mask to select corresponding data
        indices = np.where(freq_mask)[0]
    else:
        new_freq = freq
        indices = np.arange(len(freq))
    
    # Adjust phase data according to phase_limits and mod_base
    if phase_limits is not None:
        new_plot_phase = adjust_phase_range(
            plot_phase[indices, :, :], value_range=phase_limits, mod_base=mod_base)
    else:
        new_plot_phase = plot_phase[indices, :, :]
    
    # Select corresponding data for resistivity and errors
    new_plot_res = plot_res[indices, :, :]
    new_plot_res_err = plot_res_err[indices, :, :]
    new_plot_phase_err = plot_phase_err[indices, :, :]
    
    return new_freq, new_plot_res, new_plot_res_err, new_plot_phase, new_plot_phase_err

def _plot_data(
    fig, 
    ax_list, 
    freq,
    plot_res, 
    plot_res_err, 
    plot_phase, 
    plot_phase_err, 
    show_error_bars, 
    plot_z,  
    **kwargs
    ):
    """
    Plots the resistivity and phase data on the provided axes, with extensive 
    customization options.

    This function uses matplotlib to plot the provided electromagnetic tensor
    data, including resistivity and phase, on the given axes. It offers 
    customization for marker styles, line widths, error bars, and more.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object where the plots will be drawn.
    ax_list : list of matplotlib.axes.Axes
        A list of axes objects on which the data will be plotted. The list 
        should contain 8 axes, corresponding to xx, xy, yx, yy components 
        for resistivity and phase plots.
    freq : numpy.ndarray
        1D array containing the frequencies at which measurements were taken.
    plot_res : numpy.ndarray
        3D array containing the resistivity values, shaped as (n_freq, 2, 2) 
        for xx, xy, yx, yy components.
    plot_res_err : numpy.ndarray
        3D array containing the resistivity errors, shaped as (n_freq, 2, 2).
    plot_phase : numpy.ndarray
        3D array containing the phase values, shaped as (n_freq, 2, 2).
    plot_phase_err : numpy.ndarray
        3D array containing the phase errors, shaped as (n_freq, 2, 2).
    show_error_bars : bool
        If True, error bars will be displayed on the plots.
    plot_z : bool
        If True, plots the real and imaginary parts of the impedance tensor 
        instead of resistivity and phase.
    **kwargs : dict
        Additional keyword arguments for further customization, including 
        marker sizes, line widths, and color modes.

    Returns
    -------
    None
        This function does not return any value. It directly modifies the
        provided figure and axes objects.

    Examples
    --------
    Assuming `fig` and `ax_list` have been properly initialized, and `freq`,
    `plot_res`, `plot_res_err`, `plot_phase`, `plot_phase_err` have been loaded:

    >>> _plot_data(fig, ax_list, freq, plot_res, plot_res_err, plot_phase, 
    ...            plot_phase_err, show_error_bars=True, plot_z=False, color_mode='color')

    This will plot the resistivity and phase data on the given axes with color
    markers and error bars.
    """
    # Attributes 
    ms = kwargs.pop('ms', 1.5)
    ms_r = kwargs.pop('ms_r', 3)
    lw = kwargs.pop('lw', .5)
    lw_r = kwargs.pop('lw_r', 1.0)
    e_capthick = kwargs.pop('e_capthick', .5)
    e_capsize = kwargs.pop('e_capsize', 2)
    color_mode = kwargs.pop('color_mode', 'color')
    leg_style=kwargs.pop ("leg_style", '2')
    # --> set default font size
    font_size = kwargs.pop('font_size', 6)
    tick_label_size=kwargs.pop ("tick_label_size", 8 )
    plt.rcParams['font.size'] = font_size

    fontdict = {'size': font_size + 2, 
                'weight': 'bold'} 
    # arrange colors plot 
    # color mode
    if color_mode == 'color':
        # color for data
        cted = kwargs.pop('cted', (0, 0, 1))
        ctmd = kwargs.pop('ctmd', (1, 0, 0))
        mted = kwargs.pop('mted', 's')
        mtmd = kwargs.pop('mtmd', 'o')
        ctem = kwargs.pop('ctem', (0, .6, .3))
        ctmm = kwargs.pop('ctmm', (.9, 0, .8))
        mtem = kwargs.pop('mtem', '+')
        mtmm = kwargs.pop('mtmm', '+')

    # black and white mode
    elif color_mode == 'bw':
        # color for data
        cted = kwargs.pop('cted', (0, 0, 0))
        ctmd = kwargs.pop('ctmd', (0, 0, 0))
        mted = kwargs.pop('mted', 's')
        mtmd = kwargs.pop('mtmd', 'o')
        ctem = kwargs.pop('ctem', (0.6, 0.6, 0.6))
        ctmm = kwargs.pop('ctmm', (0.6, 0.6, 0.6))
        mtem = kwargs.pop('mtem', '+')
        mtmm = kwargs.pop('mtmm', 'x')
        
     # --> make key word dictionaries for plotting
    kw_xx = {'color': cted,
             'marker': mted,
             'ms': ms,
             'ls': ':',
             'lw': lw,
             'e_capsize': e_capsize,
             'e_capthick': e_capthick}

    kw_yy = {'color': ctmd,
             'marker': mtmd,
             'ms': ms,
             'ls': ':',
             'lw': lw,
             'e_capsize': e_capsize,
             'e_capthick': e_capthick}
    
    period = 1 / freq  # Convert frequency to period
    
    # Correctly mapping components to their indices in the
    # 2x2 matrix and respective colors
    components_info = {
        'xx': {'index': (0, 0), 'kw': 'kw_xx'},
         # xx and xy share the same color and marker style
        'xy': {'index': (0, 1), 'kw': 'kw_xx'}, 
        'yx': {'index': (1, 0), 'kw': 'kw_yy'},
         # yx and yy share the same color and marker style
        'yy': {'index': (1, 1), 'kw': 'kw_yy'}  
    }
    
    res_labels = [] 
    phase_labels =[] 
    res_leg_objs =[]
    phase_leg_objs=[]
    for i, component in enumerate(components_info):
        ax_res = ax_list[i]  # Axes for resistivity plots
        ax_phase = ax_list[i + 4]  # Axes for phase plots
        index = components_info[component]['index']
        kw_arg = components_info[component]['kw']
        # Accessing the 3D array for each component
        res = plot_res[:, index[0], index[1]]
        res_err = plot_res_err[:, index[0], index[1]]
        phase = plot_phase[:, index[0], index[1]]
        phase_err = plot_phase_err[:, index[0], index[1]]

        # Keyword arguments for plotting, differentiating 
        # between xx/xy and yx/yy components
        # Dynamically access the appropriate kwargs based on the component
        plot_kwargs = locals()[kw_arg]  

        # Plot resistivity
        er_res=plot_errorbar(
            ax_res, period, res, res_err if show_error_bars else None,
            **plot_kwargs
        )
        # Plot phase
        er_phase=plot_errorbar(
            ax_phase, period, phase, phase_err if show_error_bars else None,
            **plot_kwargs
        )

        ax_res.set_xscale('log')
        ax_res.set_yscale('log')
        ax_phase.set_xscale('log')

        # Customizations
        ax_res.grid(True,  which="both",  ls="--", linewidth=0.5, 
                    color='gray', alpha=0.5
            )
        ax_phase.grid( True,  which="both", ls="--", linewidth=0.5, 
                      color='gray', alpha=0.5
            )
        # Axis limit customization 
        ax_res.set_ylim([min(res) / 2, max(res) * 2])
        ax_phase.set_ylim([min(phase) - 5, max(phase) + 5])

        # Optional: Tick Labels Formatting
        ax_res.tick_params(
            axis='both', 
            which='major', 
            labelsize=kwargs.get('tick_label_size', tick_label_size)
            )
        ax_phase.tick_params(
            axis='both', 
            which='major', 
            labelsize=kwargs.get('tick_label_size', tick_label_size)
            )

        if i == 0:
            ax_res.set_ylabel(
                "Re[Z (mV/km nT)]" if plot_z else 'App. Res.($\Omega \cdot m $)' , 
                fontsize=kwargs.get('font_size', font_size)
                )
            ax_phase.set_ylabel(
                'Phase (deg.)', 
                fontsize=kwargs.get('font_size', font_size)
                )
        
        # collect object and legend labels
        res_label =  f"$z_{{{component}}}$"  if plot_z else f"$\\rho_{{{component}}}$" 
        res_labels.append (res_label)
        phase_label =  f"$\phi_{{{component}}}$" 
        phase_labels.append ( phase_label)
        
        # collect legend object 
        res_leg_objs.append (er_res )
        phase_leg_objs.append ( er_phase)
        
    # Adjust tick labels and visibility for a cleaner look
    for ax in ax_list[:4]:  # For resistivity plots
        ax.set_xlabel('')
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    for ax in ax_list[4:]:  # For phase plots
        ax.set_xlabel('Period (s)', fontsize=kwargs.get('font_size', font_size))

    if str(leg_style).lower() =='2': 
        legend_loc = 'upper center'
        legend_pos = (.5, 1.18)
        legend_marker_scale = 1
        legend_border_axes_pad = .01
        legend_label_spacing = 0.07
        legend_handle_text_pad = .2
        legend_border_pad = .15
        for aa, ax in enumerate(ax_list[:4]):
            ax.legend([res_leg_objs[aa]],
                      [res_labels[aa]],
                      loc=legend_loc,
                      bbox_to_anchor=legend_pos,
                      markerscale=legend_marker_scale,
                      borderaxespad=legend_border_axes_pad,
                      labelspacing=legend_label_spacing,
                      handletextpad=legend_handle_text_pad,
                      borderpad=legend_border_pad,
                      framealpha=1,
                      prop={'size': max([font_size, 5])})
            
    else: 
        # Customize legend for each plot considering the plotting style
        # and the mathematical symbols
        for i, ax in enumerate(ax_list[:4]):
           # Resistivity plots legend
           ax.legend([res_labels[i]], loc='best', fontsize=kwargs.get('font_size', 8),
                     frameon=True, edgecolor='black')

        for i, ax in enumerate(ax_list[4:]):

            ax.legend([phase_labels[i]], loc='best', fontsize=kwargs.get('font_size', 8),
                      frameon=True, edgecolor='black')
        
        
    fig.subplots_adjust(hspace=0.1, wspace=0.3)

def plot_logging ( 
    X, 
    y=None, 
    zname = None, 
    tname = None, 
    labels=None,
    impute_nan=True , 
    normalize = False, 
    log10=False, 
    columns_to_skip =None, 
    pattern = None, 
    strategy='mean',  
    posiy= None, 
    fill_value = None,  
    fig_size = (16, 7),
    fig_dpi = 300, 
    colors = None,
    cs4_colors=False, 
    sns_style =False, 
    savefig = None,
    draw_spines=False, 
    seed=None, 
    verbose=0, 
    **kws
    ): 
    """ Plot logging data  
    
    Plot expects a collection of logging data. Each logging data composes a 
    column of data collected on the field.Note that can also plot anykind of 
    data related that it contains numerical values. The function does not 
    accept categorical data.   If categorical data are given, they should be 
    discarded. 
    
    Parameters 
    -----------
    X : Dataframe of shape (n_samples, n_features)
         where `n_samples` is the number of data, expected to be the data 
         collected at different depths and `n_features` is the number of 
         columns (features) that supposed to be plot. 
         Note that `X` must include the ``depth`` columns. If not given a 
         relative depth should be created according to the number of sample 
         that composes `X`.
 
    y : array-like or series of shape (n_samples,), optional
        Target relative to X for classification or regression; If given, by 
        default the target plot should be located at the last position. 
        However with the argument of `posiy` , target plot can be toggled to  
        the desired position. 

    zname: str, default='depth' or 'None'
        The name of the depth column in `X`. If the name 'depth' is not  
        specified as the main depth columns, an other name in the columns 
        that matches the depth can also be indicated so the function will put 
        aside this columm as depth column for plot purpose. If set to ``None``, 
        `zname` holds the name ``depth`` and assumes that depth exists in 
        `X` columns.
    tname: str, optional, 
        name of the target. This can rename of the target name if given `y`
        as a pandas series  or add the name of target if given as an array-like. 
        If not provided, it should use the name of the target series if `y` is
        not None. 
        
    normalize: bool, default = False
        Normalize all the data to be range between (0, 1) except the `depth`,    
        
    labels: list or str, optional
        If labels are given, they should fit the size of the number of 
        columns. The given labels should replace the old columns in `X` and 
        should figue out in the plot. This is usefull to change the columns 
        labels in the dataframe to a new labels that describe the best the 
        plot ; for instance by inluding the units in the new labels. Note that 
        if the labels do not match the size of the old columns in `X` a warning 
        should be let to the user and none operation will be performed. 
        
    impute_nan: bool, default=True, 
        Replace the NaN values in the dataframe. Note that the default 
        behaviour for replacing NaN is the ``mean``. However if the argument 
        of `fill_value` is provided,the latter should be used to replace 'NaN' 
        in `X`. 
        
    log10: bool, default=False
        Convert values to log10. This can be usefull when using the logarithm 
        data. However, it seems not all the data can be used this operation, 
        for instance, a negative data. In that case, `column_to_skip` argument
        is usefull to provide so to skip that columns when converting values 
        to log10. 
 
    columns_to_skip: list or str, optional, 
        Columns to skip when performing some operation like 'log10'. These 
        columns with not be affected by the 'log10' operations. Note that 
       `columns_to_skip` can also gives as litteral string. In that case, the 
       `pattern` is need to parse the columns into a list of string. 
       
    pattern: str, default = '[#&*@!,;\s]\s*'
        Regex pattern to parse the `columns_to_skip` into a list of string 
        where each item is a column name especially when the latter is given 
        as litteral text string. For instance:: 
            
            columns_to_skip='depth_top, thickness, sp, gamma_gamma'  
            -> ['depth_top', 'thickness', 'sp', 'gamma_gamma']
            
        by using the default pattern. To have full control of columns splitted
        it is recommended to provided your own pattern to avoid wrong parsing 
        and can lead to an error. 
        
    strategy : str, default='mean'
        The imputation strategy.

        - If "mean", then replace missing values using the mean along
          each column. Can only be used with numeric data.
        - If "median", then replace missing values using the median along
          each column. Can only be used with numeric data.
        - If "most_frequent", then replace missing using the most frequent
          value along each column. Can be used with strings or numeric data.
          If there is more than one such value, only the smallest is returned.
        - If "constant", then replace missing values with fill_value. Can be
          used with strings or numeric data.

    fill_value : str or numerical value, optional
        When strategy == "constant", fill_value is used to replace all
        occurrences of missing_values.
        If left to the default, fill_value will be 0 when imputing numerical
        data and "missing_value" for strings or object data types. If not 
        given and `impute_nan` is ``True``, the mean strategy is used instead.

    posiy: int, optional 
        the position to place the target plot `y` . By default the target plot 
        if given is located at the last position behind the logging plots. 
    
    colors: str, list of Matplotlib.colors map, optional 
        The colors for plotting each columns of `X` except the depth. If not
        given, default colors are auto-generated.
        
        If `colors` is string and 'cs4'or 'xkcd' is included. 
        Matplotlib.colors.CS4_COLORS or Matplotlib.colors.XKCD_COLORS 
        should be used instead. In addition if the `'cs4'` or `'xkcd'` is  
        suffixed by colons and integer value like ``cs4:4`` or ``xkcd:4``, the 
        CS4 or XKCD colors should be used from index equals to ``4``. 
        
        .. versionadded:: 0.2.3 
           Matplotlib.colors.CS4_COLORS or Matplotlib.colors.XKCD_COLORS can 
           be used by setting `colors` to ``'cs4'`` or ``'xkcd'``. To reproduce 
           the same CS4 or XKCD colors, set the `seed` parameter to a 
           specific value. 
        
    draw_spines: bool, tuple (-lim, +lim), default= False, 
        Only draw spine between the y-ticks. ``-lim`` and ``+lim`` are lower 
        and upper bound i.e. a range to draw the spines in y-axis. 
        
    fig_size : tuple (width, height), default =(8, 6)
        the matplotlib figure size given as a tuple of width and height
        
    fig_dpi: float or 'figure', default: rcParams["savefig.dpi"] \
        (default: 'figure')
        The resolution in dots per inch. If 'figure', use the figure's dpi value.
        
    savefig: str, default =None , 
        the path to save the figure. Argument is passed to 
        :class:`matplotlib.Figure` class. 

    sns_style: str, optional, 
        the seaborn style.
        
    seed: int, optional 
       Allow to reproduce the Matplotlib.colors.CS4_COLORS if `colors` is 
       set to ``cs4``. 
       
       .. versionadded:: 0.2.3 
       
    verbose: int, default=0 
        Output the number of categorial features dropped in the dataframe.  
        
    kws: dict, 
        Additional keyword arguments passed to :func:`matplotlib.axes.plot`
        
    Examples
    ---------
    >>> from watex.datasets import load_hlogs 
    >>> from watex.utils.plotutils import plot_logging
    >>> X0, y = load_hlogs (as_frame =True) # get the frames rather than object 
    >>> # plot the default logging with Normalize =True 
    >>> plot_logging (X0, normalize =True) 
    >>> # Include the target in the plot 
    >>> plot_logging ( X0,  y = y.kp , posiy = 0, 
                      columns_to_skip=['thickness', 'sp'], 
                      log10 =True, 
                      )
    >>> # draw spines and limit plot from (0, 700) m depth 
    >>> plot_logging (X0 , y= y.kp, draw_spines =(0, 700) )
    """
    
    X = _assert_all_types(X, pd.DataFrame, pd.Series , np.ndarray ) 
    X= check_array (
        X, 
        dtype =object, 
        force_all_finite="allow-nan", 
        input_name ="Logging dataset",
        to_frame =True  
        )
    # Discard all categorical values and 
    # keep only the numerical features.
    # drop the complete Nan columns and rows
    X = to_numeric_dtypes(X, pop_cat_features=True, verbose = verbose ) 

    if y is not None: 
       if isinstance (y, (list, tuple)): 
           # in the case a lst is given 
           y = np.array (y) 
       if not is_iterable (y): 
           raise TypeError ("y expects an iterable object."
                              f" got {type(y).__name__!r}")
       y = _assert_all_types(y, pd.Series, pd.DataFrame, np.ndarray)
       
       y=check_y (
            y, 
            to_frame =True, 
            allow_nan= True,
            )
       
       if len(y) !=len(X): 
           raise ValueError ("y and X sizes along axis 0 must be consistent;"
                             f" {len(y)} and {len(X)} are given.")
    # return X and depth 
    X, depth = is_depth_in(X, zname or 'depth', columns = labels 
                            )
    # fetch target if is given  
    X, y   = _is_target_in(X, y = y , tname = tname )
    
    # skip log10 columns if log 10 is set to True 
    if log10: 
        X = _skip_log10_columns (X, column2skip = columns_to_skip , 
                                  pattern= pattern, inplace =False) 
    # if normalize then  
    if normalize:
        msc = MinMaxScaler()
        Xsc = msc.fit_transform (X)
        # set a new dataframe with features
        if hasattr (msc , 'feature_names_in_'): 
            X = pd.DataFrame (Xsc , columns = list(msc.feature_names_in_ )
                                )
        else : X = pd.DataFrame(Xsc, columns =list(X.columns )) 
        # set the x axis and delete the normalize from X 
        # at index 0 supposed to be the x axis 
        # Xsc.iloc [:, 0 ] = x_ser 
        # X= Xsc.copy()  
    # impute_nan 
    if impute_nan: 
        # check whether there is a Nan value  in the data 
        # impute data using mean values
        if X.isnull().values.any(): 
            Xi= SimpleImputer(strategy= strategy if not fill_value else None, 
                             fill_value= fill_value
                             ).fit_transform(X)
            X = pd.DataFrame(Xi, columns= X.columns)
            
    # toggle y 
    if y is not None: 
        X = _toggle_target_in(X, y, pos = posiy)
        
    #manage colors along colors 
    colors = make_plot_colors (
        X, colors = colors , axis = 1, seed = seed , chunk=False )

    fig, ax = plt.subplots (1, ncols = X.shape [1], sharey = True , 
                            figsize = fig_size )
    
    # customize bound and set spines 
    for k in range (X.shape [1]): 
     
        ax[k].plot ( X.iloc[:, k], 
                    depth, 
                    color = colors[k], 
                    **kws
                    )
        ax[k].tick_params(top=True, 
                          labeltop=True, 
                          bottom=False, 
                          labelbottom=False
                       )
        ax[k].set_title (X.columns [k])
        ax[k].spines['right'].set_visible(False)
        ax[k].spines['bottom'].set_visible(False)
        # only show tick on the top and left 
        ax[k].xaxis.set_ticks_position('top')
        if y is not None: 
            # make X axis of the target to red 
            # for differenciation from features. 
            if X.columns [k] ==y.name: 
                ax[k].spines['top'].set_color('red')  
         
        if draw_spines: 
            # Only draw spine between the y-ticks
            if is_iterable(draw_spines): 
                # for consistency check whether values 
                # are numeric
                draw_spines = sorted (
                    list(map (lambda x: float (x) , draw_spines[:2])) 
                    ) 
                if len(draw_spines) <2: 
                    warnings.warn(
                        "Spine bounds is a tuple of (startpoint, endpoint)"
                         " Single limit value is not allowed."
                         )
            else: 
                # in case only True is given 
                # use the default plot
                ytv= ax[0].get_yticks () 
                spacing = (ytv[-1] - ytv[0] )/(len(ytv)-1) 
                # commonly matplotlib axis extrapoled the limit so 
                # start with the first and last index 
                draw_spines=  (ytv[0] + spacing/2 , ytv[-1] - spacing/2 ) 
                
            ax[k].spines['left'].set_bounds(*draw_spines )
     
    # set labels
    ax[0].set_ylabel ("Depth (m)")
        # Tweak spacing between subplots to prevent labels 
        # from overlapping 
        # plt.subplots_adjust(hspace=0.5)-> removed
    plt.gca().invert_yaxis()
    
    if savefig is not None:
        plt.savefig(savefig, dpi = fig_dpi )
        
    plt.close () if savefig is not None else plt.show() 
    
def make_plot_colors(d , / , colors:str | list[str]=None , axis:int = 0, 
                     seed:int  =None, chunk:bool =... ): 
    """ Select colors according to the data size along axis 
    
    Parameters 
    ----------
    d: Arraylike 
       Array data to select colors according to the axis 
    colors: str, list of Matplotlib.colors map, optional 
        The colors for plotting each columns of `X` except the depth. If not
        given, default colors are auto-generated.
        If `colors` is string and 'cs4'or 'xkcd' is included. 
        Matplotlib.colors.CS4_COLORS or Matplotlib.colors.XKCD_COLORS 
        should be used instead. In addition if the `'cs4'` or `'xkcd'` is  
        suffixed by colons and integer value like ``cs4:4`` or ``xkcd:4``, the 
        CS4 or XKCD colors should be used from index equals to ``4``. 
        
        .. versionadded:: 0.2.3 
           Matplotlib.colors.CS4_COLORS or Matplotlib.colors.XKCD_COLORS can 
           be used by setting `colors` to ``'cs4'`` or ``'xkcd'``. To reproduce 
           the same CS4 or XKCD colors, set the `seed` parameter to a 
           specific value. 
           
    axis: int, default=0 
       Axis along with the colors must be generated. By default colors is 
       generated along the row axis 
       
    seed: int, optional 
       Allow to reproduce the Matplotlib.colors.CS4_COLORS if `colors` is 
       set to ``cs4``. 
       
    chunk: bool, default=True 
       Chunk generated colors to fit the exact length of the `d` size 
       
    Returns 
    -------
    colors: list 
       List of new generated colors 
       
    Examples 
    --------
    >>> import numpy as np 
    >>> from watex.utils.plotutils import make_plot_colors
    >>> ar = np.random.randn (7, 2) 
    >>> make_plot_colors (ar )
    ['g', 'gray', 'y', 'blue', 'orange', 'purple', 'lime']
    >>> make_plot_colors (ar , axis =1 ) 
    Out[6]: ['g', 'gray']
    >>> make_plot_colors (ar , axis =1 , colors ='cs4')
    ['#F0F8FF', '#FAEBD7']
    >>> len(make_plot_colors (ar , axis =1 , colors ='cs4', chunk=False))
    150
    >>> make_plot_colors (ar , axis =1 , colors ='cs4:4')
    ['#F0FFFF', '#F5F5DC']
    """
    
    # get the data size where colors must be fitted. 
    # note colors should match either the row axis or colurms axis 
    axis = str(axis).lower() 
    if 'columns1'.find (axis)>=0: 
        axis =1 
    else: axis =0
    
    # manage the array 
    d= is_iterable( d, exclude_string=True, transform=True)
    if not hasattr (d, '__array__'): 
        d = np.array(d, dtype =object ) 
    
    axis_length = len(d) if len(d.shape )==1 else d.shape [axis]
    m_cs = make_mpl_properties(axis_length )
    
     #manage colors 
    # we assume the first columns is dedicated for 
    if colors ==...: colors =None 
    if ( 
            isinstance (colors, str) and 
            ( 
                "cs4" in str(colors).lower() 
                 or 'xkcd' in str(colors).lower() 
                 )
            ): 
        #initilize colors infos
        c = copy.deepcopy(colors)
        if 'cs4' in str(colors).lower() : 
            DCOLORS = mcolors.CSS4_COLORS
        else: 
            # remake the dcolors my removing the xkcd: in the keys: 
            DCOLORS = dict(( (k.replace ('xkcd:', ''), c) 
                            for k, c in mcolors.XKCD_COLORS.items()))  
        
        key_colors = list(DCOLORS.keys ())
        colors = list(DCOLORS.values() )
        
        shuffle_cs4=True 
        
        cs4_start= None
        #------
        if ':' in str(c).lower():
            cs4_start = str(c).lower().split(':')[-1]
        #try to converert into integer 
        try: 
            cs4_start= int (cs4_start)
        except : 
            if str(cs4_start).lower() in key_colors: 
                cs4_start= key_colors.index (cs4_start)
                shuffle_cs4=False
            else: 
                pass 
        
        else: shuffle_cs4=False # keep CS4 and dont shuffle 
        
        cs4_start= cs4_start or 0
        
        if shuffle_cs4: 
            np.random.seed (seed )
            colors = list(np.random.choice(colors  , len(m_cs)))
        else: 
            if cs4_start > len(colors)-1: 
                cs4_start = 0 
    
            colors = colors[ cs4_start:]
    
    if colors is not None: 
        if not is_iterable(colors): 
            colors =[colors]
        colors += m_cs 
    else :
        colors = m_cs 
        
    # shrunk data to map the exact colors 
    chunk =True if chunk is ... else False 
    return colors[:axis_length] if chunk else colors 


def plot_silhouette (X, labels, metric ='euclidean',savefig =None , **kwds ):
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
        
    savefig: str, default =None , 
        the path to save the figure. Argument is passed to 
        :class:`matplotlib.Figure` class. 
        
    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a ``scipy.spatial.distance`` metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
        
        
    See Also
    --------
    watex.view.mlplot.plotSilhouette: 
        Gives consistency plot as the use of `prefit` parameter which checks 
        whether`labels` are expected to be passed into the function 
        directly or not. 
    
    Examples
    ---------
    >>> import numpy as np 
    >>> from watex.exlib.sklearn import KMeans 
    >>> from watex.datasets import load_iris 
    >>> from watex.utils.plotutils import plot_silhouette
    >>> d= load_iris ()
    >>> X= d.data [:, 0][:, np.newaxis] # take the first axis 
    >>> km= KMeans (n_clusters =3 , init='k-means++', n_init =10 , 
                    max_iter = 300 , 
                    tol=1e-4, 
                    random_state =0 
                    )
    >>> y_km = km.fit_predict(X) 
    >>> plot_silhouette (X, y_km)

    """
    X, labels = check_X_y(
        X, 
        labels, 
        to_frame= True, 
        )
    cluster_labels = np.unique (labels) 
    n_clusters = cluster_labels.shape [0] 
    silhouette_vals = silhouette_samples(
        X, labels= labels, metric = metric ,**kwds)
    y_ax_lower , y_ax_upper = 0, 0 
    yticks =[]
    
    for i, c  in enumerate (cluster_labels ) : 
        c_silhouette_vals = silhouette_vals[labels ==c ] 
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color =mpl.cm.jet (float(i)/n_clusters )
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

    if savefig is not None:
        plt.savefig(savefig, dpi = 300 )
        
    plt.close () if savefig is not None else plt.show() 
    

def plot_sbs_feature_selection (
        sbs_estimator,/,  X=None, y=None ,fig_size=(8, 5), 
        sns_style =False, savefig = None, verbose=0 , 
        **sbs_kws
        ): 
    """plot Sequential Backward Selection (SBS) for feature selection.  
    
    SBS collects the scores of the  best feature subset at each stage. 
    
    Parameters 
    ------------
    sbs_estimator : :class:`~.watex.base.SequentialBackwardSelection`\
        estimator object
        The Sequential Backward Selection estimator can either be fitted or 
        not. If not fitted. Please provide the training `X` and `y`, 
        otherwise an error will occurs.
        
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target relative to X for classification or regression;
        None for unsupervised learning.
       
    n_estimators : int, default=500
        The number of trees in the forest.
        
    fig_size : tuple (width, height), default =(8, 6)
        the matplotlib figure size given as a tuple of width and height
        
    savefig: str, default =None , 
        the path to save the figures. Argument is passed to matplotlib.Figure 
        class. 
    sns_style: str, optional, 
        the seaborn style.
    verbose: int, default=0 
        print the feature labels with the rate of their importances. 
    sbs_kws: dict, 
        Additional keyyword arguments passed to 
        :class:`~.watex.base.SequentialBackwardSelection`
        
    Examples 
    ----------
    (1)-> Plot fitted SBS in action 
    >>> from watex.exlib.sklearn import KNeighborsClassifier , train_test_split
    >>> from watex.datasets import fetch_data
    >>> from watex.base import SequentialBackwardSelection
    >>> from watex.utils.plotutils import plot_sbs_feature_selection
    >>> X, y = fetch_data('bagoue analysed') # data already standardized
    >>> Xtrain, Xt, ytrain,  yt = train_test_split(X, y)
    >>> knn = KNeighborsClassifier(n_neighbors=5)
    >>> sbs= SequentialBackwardSelection (knn)
    >>> sbs.fit(Xtrain, ytrain )
    >>> plot_sbs_feature_selection(sbs, sns_style= True) 
    
    (2)-> Plot estimator with no prefit SBS. 
    >>> plot_sbs_feature_selection(knn, Xtrain, ytrain) # yield the same result

    """
    from ..base import SequentialBackwardSelection as SBS 
    if ( 
        not hasattr (sbs_estimator, 'scores_') 
        and not hasattr (sbs_estimator, 'k_score_')
            ): 
        if ( X is None or y is None ) : 
            clfn = get_estimator_name( sbs_estimator)
            raise TypeError (f"When {clfn} is not a fitted "
                             "estimator, X and y are needed."
                             )
        sbs_estimator = SBS(estimator = sbs_estimator, **sbs_kws)
        sbs_estimator.fit(X, y )
        
    k_feat = [len(k) for k in sbs_estimator.subsets_]
    
    if verbose: 
        flabels =None 
        if  ( not hasattr (X, 'columns') and X is not None ): 
            warnings.warn("None columns name is detected."
                          " Created using index ")
            flabels =[f'{i:>7}' for i in range (X.shape[1])]
            
        elif hasattr (X, 'columns'):
            flabels = list(X.columns)  
        elif hasattr ( sbs_estimator , 'feature_names_in'): 
            flabels = sbs_estimator.feature_names_in 
            
        if flabels is not None: 
            k3 = list (sbs_estimator.subsets_[X.shape[1]])
            print("Smallest feature for subset (k=3) ")
            print(flabels [k3])
            
        else : print("No column labels detected. Can't print the "
                     "smallest feature subset.")
        
    if sns_style: 
        _set_sns_style (sns_style)
        
    plt.figure(figsize = fig_size)
    plt.plot (k_feat , sbs_estimator.scores_, marker='o' ) 
    plt.ylim ([min(sbs_estimator.scores_) -.25 ,
               max(sbs_estimator.scores_) +.2 ])
    plt.ylabel (sbs_estimator.scorer_name_ )
    plt.xlabel ('Number of features')
    plt.tight_layout() 
    
    if savefig is not None:
        plt.savefig(savefig )
        
    plt.close () if savefig is not None else plt.show() 
    

def plot_regularization_path ( 
        X, y , c_range=(-4., 6. ), fig_size=(8, 5), sns_style =False, 
        savefig = None, **kws 
        ): 
    r""" Plot the regularisation path from Logit / LogisticRegression 
    
    Varying the  different regularization strengths and plot the  weight 
    coefficient of the different features for different regularization 
    strength. 
    
    Note that, it is recommended to standardize the data first. 
    
    Parameters 
    -----------
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features. X is expected to be 
        standardized. 

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target relative to X for classification or regression;
        None for unsupervised learning.
    c_range: list or tuple [start, stop] 
        Regularization strength list. It is a range from the strong  
        strong ( start) to lower (stop) regularization. Note that 'C' is 
        the inverse of the Logistic Regression regularization parameter 
        :math:`\lambda`. 
    fig_size : tuple (width, height), default =(8, 6)
        the matplotlib figure size given as a tuple of width and height
        
    savefig: str, default =None , 
        the path to save the figures. Argument is passed to matplotlib.Figure 
        class. 
    sns_style: str, optional, 
        the seaborn style.
        
    kws: dict, 
        Additional keywords arguments passed to 
        :class:`sklearn.linear_model.LogisticRegression`
    
    Examples
    --------
    >>> from watex.utils.plotutils import plot_regularization_path 
    >>> from watex.datasets import fetch_data
    >>> X, y = fetch_data ('bagoue analysed' ) # data aleardy standardized
    >>> plot_regularization_path (X, y ) 

    """
    X, y = check_X_y(
        X, 
        y, 
        to_frame= True, 
        )
    
    if not is_iterable(c_range): 
        raise TypeError ("'C' regularization strength is a range of C " 
                         " Logit parameter: (start, stop).")
    c_range = sorted (c_range )
    
    if len(c_range) < 2: 
        raise ValueError ("'C' range expects two values [start, stop]")
        
    if len(c_range) >2 : 
        warnings.warn ("'C' range expects two values [start, stop]. Values"
                       f" are shrunk to the first two values: {c_range[:2]} "
                       )
    weights, params = [], []    
    for c in np.arange (*c_range): 
        lr = LogisticRegression(penalty='l1', C= 10.**c, solver ='liblinear', 
                                multi_class='ovr', **kws)
        lr.fit(X,y )
        weights.append (lr.coef_[1])
        params.append(10**c)
        
    weights = np.array(weights ) 
    colors = make_mpl_properties(weights.shape[1])
    if not hasattr (X, 'columns'): 
        flabels =[f'{i:>7}' for i in range (X.shape[1])] 
    else: flabels = X.columns   
    
    # plot
    fig, ax = plt.subplots(figsize = fig_size )
    if sns_style: 
        _set_sns_style (sns_style)

    for column , color in zip( range (weights.shape [1]), colors ): 
        plt.plot (params , weights[:, column], 
                  label =flabels[column], 
                  color = color 
                  )

    plt.axhline ( 0 , color ='black', ls='--', lw= 3 )
    plt.xlim ( [ 10 ** int(c_range[0] -1), 10 ** int(c_range[1]-1) ])
    plt.ylabel ("Weight coefficient")
    plt.xlabel ('C')
    plt.xscale( 'log')
    plt.legend (loc ='upper left',)
    ax.legend(
            loc ='upper right', 
            bbox_to_anchor =(1.38, 1.03 ), 
            ncol = 1 , fancybox =True 
    )
    
    if savefig is not None:
        plt.savefig(savefig, dpi = 300 )
        
    plt.close () if savefig is not None else plt.show() 
    
def plot_rf_feature_importances (
    clf, 
    X=None, 
    y=None, 
    fig_size = (8, 4),
    savefig =None,   
    n_estimators= 500, 
    verbose =0 , 
    sns_style =None,  
    **kws 
    ): 
    """
    Plot features importance with RandomForest.  
    
    Parameters 
    ----------
    clf : estimator object
        The base estimator from which the transformer is built.
        This can be both a fitted (if ``prefit`` is set to True)
        or a non-fitted estimator. The estimator should have a
        ``feature_importances_`` or ``coef_`` attribute after fitting.
        Otherwise, the ``importance_getter`` parameter should be used.
        
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target relative to X for classification or regression;
        None for unsupervised learning.
       
    n_estimators : int, default=500
        The number of trees in the forest.
        
    fig_size : tuple (width, height), default =(8, 6)
        the matplotlib figure size given as a tuple of width and height
        
    savefig: str, default =None , 
        the path to save the figures. Argument is passed to matplotlib.Figure 
        class. 
    sns_style: str, optional, 
        the seaborn style.
    verbose: int, default=0 
        print the feature labels with the rate of their importances. 
    kws: dict, 
        Additional keyyword arguments passed to 
        :class:`sklearn.ensemble.RandomForestClassifier`
        
    Examples
    ---------
    >>> from watex.datasets import fetch_data
    >>> from watex.exlib.sklearn import RandomForestClassifier 
    >>> from watex.utils.plotutils import plot_rf_feature_importances 
    >>> X, y = fetch_data ('bagoue analysed' ) 
    >>> plot_rf_feature_importances (
        RandomForestClassifier(), X=X, y=y , sns_style=True)

    """
    if not hasattr (clf, 'feature_importances_'): 
        if ( X is None or y is None ) : 
            clfn = get_estimator_name( clf)
            raise TypeError (f"When {clfn} is not a fitted "
                             "estimator, X and y are needed."
                             )
        clf = RandomForestClassifier(n_estimators= n_estimators , **kws)
        clf.fit(X, y ) 
        
    importances = clf.feature_importances_ 
    indices = np.argsort(importances)[::-1]
    if hasattr( X, 'columns'): 
        flabels = X.columns 
    else : flabels =[f'{i:>7}' for i in range (X.shape[1])]
    
    if verbose : 
        for f in range(X.shape [1]): 
            print("%2d) %-*s %f" %(f +1 , 30 , flabels[indices[f]], 
                                   importances[indices[f]])
                  )
    if sns_style: 
        _set_sns_style (sns_style)

    plt.figure(figsize = fig_size)
    plt.title ("Feature importance")
    plt.bar (range(X.shape[1]) , 
             importances [indices], 
             align='center'
             )
    plt.xticks (range (X.shape[1]), flabels [indices], rotation =90 , 
                ) 
    plt.xlim ([-1 , X.shape[1]])
    plt.ylabel ('Importance rate')
    plt.xlabel ('Feature labels')
    plt.tight_layout()
    
    if savefig is not None:
        plt.savefig(savefig )

    plt.close () if savefig is not None else plt.show() 
    
        
def plot_confusion_matrix (yt, y_pred, view =True, ax=None, annot=True,  **kws ):
    """ plot a confusion matrix for a single classifier model.
    
    :param yt : ndarray or Series of length n
        An array or series of true target or class values. Preferably, 
        the array represents the test class labels data for error evaluation.
    
    :param y_pred: ndarray or Series of length n
        An array or series of the predicted target. 
    :param view: bool, default=True 
        Option to display the matshow map. Set to ``False`` mutes the plot. 
    :param annot: bool, default=True 
        Annotate the number of samples (right or wrong prediction ) in the plot. 
        Set ``False`` to mute the display.
    param kws: dict, 
        Additional keyword arguments passed to the function 
        :func:`sckitlearn.metrics.confusion_matrix`. 
    :returns: mat- confusion matrix bloc matrix 
    
    :example: 
    >>> #Import the required models and fetch a an Ababoost model 
    >>> # for instance then plot the confusion metric 
    >>> import matplotlib.pyplot as plt 
    >>> plt.style.use ('classic')
    >>> from watex.datasets import fetch_data
    >>> from watex.exlib.sklearn import train_test_split 
    >>> from watex.models import pModels 
    >>> from watex.utils.plotutils import plot_confusion_matrix
    >>> # split the  data . Note that fetch_data output X and y 
    >>> X, Xt, y, yt  = train_test_split (* fetch_data ('bagoue analysed'),
                                          test_size =.25  )  
    >>> # train the model with the best estimator 
    >>> pmo = pModels (model ='ada' ) 
    >>> pmo.fit(X, y )
    >>> print(pmo.estimator_ )
    >>> #%% 
    >>> # Predict the score using under the hood the best estimator 
    >>> # for adaboost classifier 
    >>> ypred = pmo.predict(Xt) 
    >>> # now plot the score 
    >>> plot_confusion_matrix (yt , ypred )
    """
    check_consistent_length (yt, y_pred)
    mat= confusion_matrix (yt, y_pred, **kws)
    if ax is None: 
        fig, ax = plt.subplots ()
        
    if view: 
        sns.heatmap (
            mat.T, square =True, annot =annot, cbar=False, ax=ax)
        # xticklabels= list(np.unique(ytrue.values)), 
        # yticklabels= list(np.unique(ytrue.values)))
        ax.set_xlabel('true labels' )
        ax.set_ylabel ('predicted labels')
    return mat 

def plot_yb_confusion_matrix (
        clf, Xt, yt, labels = None , encoder = None, savefig =None, 
        fig_size =(6, 6), **kws
        ): 
    """ Confusion matrix plot using the 'yellowbrick' package.  
    
    Creates a heatmap visualization of the sklearn.metrics.confusion_matrix().
    A confusion matrix shows each combination of the true and predicted
    classes for a test data set.

    The default color map uses a yellow/orange/red color scale. The user can
    choose between displaying values as the percent of true (cell value
    divided by sum of row) or as direct counts. If percent of true mode is
    selected, 100% accurate predictions are highlighted in green.

    Requires a classification model.
    
    Be sure 'yellowbrick' is installed before using the function, otherwise an 
    ImportError will raise. 
    
    Parameters 
    -----------
    clf : classifier estimator
        A scikit-learn estimator that should be a classifier. If the model is
        not a classifier, an exception is raised. If the internal model is not
        fitted, it is fit when the visualizer is fitted, unless otherwise specified
        by ``is_fitted``.
        
    Xt : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features. Preferably, matrix represents 
        the test data for error evaluation.  

    yt : ndarray or Series of length n
        An array or series of target or class values. Preferably, the array 
        represent the test class labels data for error evaluation.  

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If not specified the current axes will be
        used (or generated if required).

    sample_weight: array-like of shape = [n_samples], optional
        Passed to ``confusion_matrix`` to weight the samples.
        
    encoder : dict or LabelEncoder, default: None
        A mapping of classes to human readable labels. Often there is a mismatch
        between desired class labels and those contained in the target variable
        passed to ``fit()`` or ``score()``. The encoder disambiguates this mismatch
        ensuring that classes are labeled correctly in the visualization.
        
    labels : list of str, default: None
        The class labels to use for the legend ordered by the index of the sorted
        classes discovered in the ``fit()`` method. Specifying classes in this
        manner is used to change the class names to a more specific format or
        to label encoded integer classes. Some visualizers may also use this
        field to filter the visualization for specific classes. For more advanced
        usage specify an encoder rather than class labels.
        
    fig_size : tuple (width, height), default =(8, 6)
        the matplotlib figure size given as a tuple of width and height
        
    savefig: str, default =None , 
        the path to save the figures. Argument is passed to matplotlib.Figure 
        class. 
          
    Returns 
    --------
    cmo: :class:`yellowbrick.classifier.confusion_matrix.ConfusionMatrix`
        return a yellowbrick confusion matrix object instance. 
    
    Examples 
    --------
    >>> #Import the required models and fetch a an extreme gradient boosting 
    >>> # for instance then plot the confusion metric 
    >>> import matplotlib.pyplot as plt 
    >>> plt.style.use ('classic')
    >>> from watex.datasets import fetch_data
    >>> from watex.exlib.sklearn import train_test_split 
    >>> from watex.models import pModels 
    >>> from watex.utils.plotutils import plot_yb_confusion_matrix
    >>> # split the  data . Note that fetch_data output X and y 
    >>> X, Xt, y, yt  = train_test_split (* fetch_data ('bagoue analysed'),
                                          test_size =.25  )  
    >>> # train the model with the best estimator 
    >>> pmo = pModels (model ='xgboost' ) 
    >>> pmo.fit(X, y )
    >>> print(pmo.estimator_ ) # pmo.XGB.best_estimator_
    >>> #%% 
    >>> # Predict the score using under the hood the best estimator 
    >>> # for adaboost classifier 
    >>> ypred = pmo.predict(Xt) 
    
    >>> # now plot the score 
    >>> plot_yb_confusion_matrix (pmo.XGB.best_estimator_, Xt, yt  )
    """
    import_optional_dependency('yellowbrick', (
        "Cannot plot the confusion matrix via 'yellowbrick' package."
        " Alternatively, you may use ufunc `~.plot_confusion_matrix`,"
        " otherwise install it mannually.")
        )
    fig, ax = plt.subplots(figsize = fig_size )
    cmo= ConfusionMatrix (clf, classes=labels, 
                         label_encoder = encoder, **kws
                         )
    cmo.score(Xt, yt)
    cmo.show()

    if savefig is not None: 
        fig.savefig(savefig, dpi =300)

    plt.close () if savefig is not None else plt.show() 
    
    return cmo 

def plot_confusion_matrices (
    clfs, 
    Xt, 
    yt,  
    annot =True, 
    pkg=None, 
    normalize='true', 
    sample_weight=None,
    encoder=None, 
    fig_size = (22, 6),
    savefig =None, 
    subplot_kws=None,
    **scorer_kws
    ):
    """ 
    Plot inline multiple model confusion matrices using either the sckitlearn 
    or 'yellowbrick'
    
    Parameters 
    -----------
    clfs : list of classifier estimators
        A scikit-learn estimator that should be a classifier. If the model is
        not a classifier, an exception is raised. Note that the classifier 
        must be fitted beforehand.
        
    Xt : ndarray or DataFrame of shape (M X N)
        A matrix of n instances with m features. Preferably, matrix represents 
        the test data for error evaluation.  

    yt : ndarray of shape (M, ) or Series oF length (M, )
        An array or series of target or class values. Preferably, the array 
        represent the test class labels data for error evaluation.  
    
    pkg: str, optional , default ='sklearn'
        the library to handle the plot. It could be 'yellowbrick'. The basic 
        confusion matrix is handled by the scikit-learn package. 

    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
        
    encoder : dict or LabelEncoder, default: None
        A mapping of classes to human readable labels. Often there is a mismatch
        between desired class labels and those contained in the target variable
        passed to ``fit()`` or ``score()``. The encoder disambiguates this mismatch
        ensuring that classes are labeled correctly in the visualization.
        
        
    annot: bool, default=True 
        Annotate the number of samples (right or wrong prediction ) in the plot. 
        Set ``False`` to mute the display. 
    
    fig_size : tuple (width, height), default =(8, 6)
        the matplotlib figure size given as a tuple of width and height
        
    savefig: str, default =None , 
        the path to save the figures. Argument is passed to matplotlib.Figure 
        class. 
        
    Examples
    ----------
    >>> import matplotlib.pyplot as plt 
    >>> plt.style.use ('classic')
    >>> from watex.datasets import fetch_data
    >>> from watex.exlib.sklearn import train_test_split 
    >>> from watex.models.premodels import p
    >>> from watex.utils.plotutils import plot_confusion_matrices 
    >>> # split the  data . Note that fetch_data output X and y 
    >>> X, Xt, y, yt  = train_test_split (* fetch_data ('bagoue analysed'), test_size =.25  )  
    >>> # compose the models 
    >>> # from RBF, and poly 
    >>> models =[ p.SVM.rbf.best_estimator_,
             p.LogisticRegression.best_estimator_,
             p.RandomForest.best_estimator_ 
             ]
    >>> models 
    [SVC(C=2.0, coef0=0, degree=1, gamma=0.125), LogisticRegression(), 
     RandomForestClassifier(criterion='entropy', max_depth=16, n_estimators=350)]
    >>> # now fit all estimators 
    >>> fitted_models = [model.fit(X, y) for model in models ]
    >>> plot_confusion_matrices(fitted_models , Xt, yt)
    """
    pkg = pkg or 'sklearn'
    pkg= str(pkg).lower() 
    assert pkg in {"sklearn", "scikit-learn", 'yellowbrick', "yb"}, (
        f" Accepts only 'sklearn' or 'yellowbrick' packages, got {pkg!r}") 
    
    if not is_iterable( clfs): 
        clfs =[clfs]

    model_names = [get_estimator_name(name) for name in clfs ]
    # create a figure 
    subplot_kws = subplot_kws or dict (left=0.0625, right = 0.95, 
                                       wspace = 0.12)
    fig, axes = plt.subplots(1, len(clfs), figsize =(22, 6))
    fig.subplots_adjust(**subplot_kws)
    if not is_iterable(axes): 
       axes =[axes] 
    for kk, (model , mname) in enumerate(zip(clfs, model_names )): 
        ypred = model.predict(Xt)
        if pkg in ('sklearn', 'scikit-learn'): 
            plot_confusion_matrix(yt, ypred, annot =annot , ax = axes[kk], 
                normalize= normalize , sample_weight= sample_weight ) 
            axes[kk].set_title (mname)
            
        elif pkg in ('yellowbrick', 'yb'):
            plot_yb_confusion_matrix(
                model, Xt, yt, ax=axes[kk], encoder =encoder )
    if savefig is not None:
        plt.savefig(savefig, dpi = 300 )
        
    plt.close () if savefig is not None else plt.show() 
    
def plot_learning_curves(
    models, 
    X ,
    y, 
    *, 
    cv =None, 
    train_sizes= None, 
    baseline_score =0.4,
    scoring=None, 
    convergence_line =True, 
    fig_size=(20, 6),
    sns_style =None, 
    savefig=None, 
    set_legend=True, 
    subplot_kws=None,
    **kws
    ): 
    """ 
    Horizontally visualization of multiple models learning curves. 
    
    Determines cross-validated training and test scores for different training
    set sizes.
    
    Parameters 
    ----------
    models: list or estimators  
        An estimator instance or not that implements `fit` and `predict` 
        methods which will be cloned for each validation. 
        
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target relative to X for classification or regression;
        None for unsupervised learning.
   
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        ``cv`` default value if None changed from 3-fold to 4-fold.
        
     train_sizes : array-like of shape (n_ticks,), \
             default=np.linspace(0.1, 1, 50)
         Relative or absolute numbers of training examples that will be used to
         generate the learning curve. If the dtype is float, it is regarded as a
         fraction of the maximum size of the training set (that is determined
         by the selected validation method), i.e. it has to be within (0, 1].
         Otherwise it is interpreted as absolute sizes of the training sets.
         Note that for classification the number of samples usually have to
         be big enough to contain at least one sample from each class.
         
    baseline_score: floatm default=.4 
        base score to start counting in score y-axis  (score)
        
    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        
    convergence_line: bool, default=True 
        display the convergence line or not that indicate the level of bias 
        between the training and validation curve. 
        
    fig_size : tuple (width, height), default =(14, 6)
        the matplotlib figure size given as a tuple of width and height
        
    sns_style: str, optional, 
        the seaborn style . 
        
    set_legend: bool, default=True 
        display legend in each figure. Note the default location of the 
        legend is 'best' from :func:`~matplotlib.Axes.legend`
        
    subplot_kws: dict, default is \
        dict(left=0.0625, right = 0.95, wspace = 0.1) 
        the subplot keywords arguments passed to 
        :func:`matplotlib.subplots_adjust` 
    kws: dict, 
        keyword arguments passed to :func:`sklearn.model_selection.learning_curve`
        
    Examples 
    ---------
    (1) -> plot via a metaestimator already cross-validated. 
    
    >>> from watex.models.premodels import p 
    >>> from watex.datasets import fetch_data 
    >>> from watex.utils.plotutils import plot_learning_curves
    >>> X, y = fetch_data ('bagoue prepared') # yields a sparse matrix 
    >>> # let collect 04 estimators already cross-validated from SVMs
    >>> models = [ p.SVM.linear , p.SVM.rbf , p.SVM.sigmoid , p.SVM.poly ]
    >>> plot_learning_curves (models, X, y, cv=4, sns_style = 'darkgrid')
    
    (2) -> plot with  multiples models not crossvalidated yet.
    
    >>> from watex.exlib.sklearn import (LogisticRegression, 
                                         RandomForestClassifier, 
                                         SVC , KNeighborsClassifier 
                                         )
    >>> models =[LogisticRegression(), RandomForestClassifier(), SVC() ,
                 KNeighborsClassifier() ]
    >>> plot_learning_curves (models, X, y, cv=4, sns_style = 'darkgrid')
    
    """
    if not is_iterable(models): 
        models =[models]
    
    subplot_kws = subplot_kws or  dict(
        left=0.0625, right = 0.95, wspace = 0.1) 
    train_sizes = train_sizes or np.linspace(0.1, 1, 50)
    cv = cv or 4 
    if ( 
        baseline_score >=1 
        and baseline_score < 0 
        ): 
        raise ValueError ("Score for the base line must be less 1 and "
                          f"greater than 0; got {baseline_score}")
    
    if sns_style: 
        _set_sns_style (sns_style)
        
    mnames = [get_estimator_name(n) for n in models]

    fig, axes = plt.subplots(nrows=1, ncols=len(models), figsize =fig_size)
    # for consistency, put axes on list when 
    # a single model is provided 
    if not is_iterable(axes): 
        axes =[axes] 
    fig.subplots_adjust(**subplot_kws)

    for k, (model, name) in enumerate(zip(models,  mnames)):
        cmodel = model.best_estimator_  if _is_cross_validated(
            model ) else model 
        ax = list(axes)[k]

        N, train_lc , val_lc = learning_curve(
            cmodel , 
            X, 
            y, 
            train_sizes = np.linspace(0.1, 1, 50),
            cv=cv, 
            scoring=scoring, 
            **kws
            )
        ax.plot(N, np.mean(train_lc, 1), 
                   color ="blue", 
                   label ="train score"
                   )
        ax.plot(N, np.mean(val_lc, 1), 
                   color ="r", 
                   label ="validation score"
                   )
        if convergence_line : 
            ax.hlines(np.mean([train_lc[-1], 
                                  val_lc[-1]]), 
                                 N[0], N[-1], 
                                 color="k", 
                                 linestyle ="--"
                         )
        ax.set_ylim(baseline_score, 1)
        #ax[k].set_xlim (N[0], N[1])
        ax.set_xlabel("training size")
        ax.set_title(name, size=14)
        if set_legend: 
            ax.legend(loc='best')
    # for consistency
    ax = list(axes)[0]
    ax.set_ylabel("score")
    
    if savefig is not None:
        plt.savefig(savefig, dpi = 300 )
        
    plt.close () if savefig is not None else plt.show() 
        
def plot_naive_dendrogram (
        X, 
        *ybounds, 
        fig_size = (12, 5 ), 
        savefig=None,  
        **kws
        ): 
    """ Quick plot dendrogram using the ward clustering function from Scipy.
    
    :param X: ndarray of shape (n_samples, n_features) 
        Array of features 
    :param ybounds: int, 
        integrer values to draw horizontal cluster lines that indicate the 
        number of clusters. 
    :param fig_size: tuple (width, height), default =(12,5) 
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
        >>> # add the horizontal line of the cluster at ybounds = (20 , 20 )
        >>> # for a single cluster (cluser 1)
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
    # change the gca figsize 
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
            
    if savefig is not None:
        plt.savefig(savefig, dpi = 300 )
        
    plt.close () if savefig is not None else plt.show() 
    
def plot_pca_components (
        components, *, feature_names = None , cmap= 'viridis', 
        savefig=None, **kws
        ): 
    """ Visualize the coefficient of principal component analysis (PCA) as 
    a heatmap  
  
    :param components: Ndarray, shape (n_components, n_features)or PCA object 
        Array of the PCA compoments or object from 
        :class:`watex.analysis.dimensionality.nPCA`. If the object is given 
        it is not necessary to set the `feature_names`
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
    if sp.issparse (components): 
        raise TypeError ("Sparse array is not supported for PCA "
                         "components visualization."
                         )
    # if pca object is given , get the features names
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
        warnings.warn("Number of features and eigenvectors might"
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
    
    if savefig is not None:
        plt.savefig(savefig, dpi = 300 )
        
    plt.close () if savefig is not None else plt.show() 
    
        
def plot_clusters (
        n_clusters, X, y_pred, cluster_centers =None , savefig =None, 
        ): 
    """ Visualize the cluster that k-means identified in the dataset 
    
    :param n_clusters: int, number of cluster to visualize 
    :param X: NDArray, data containing the features, expect to be a two 
        dimensional data 
    :param y_pred: array-like, array containing the predicted class labels. 
    :param cluster_centers_: NDArray containg the coordinates of the 
        centroids or the similar points with continous features. 
        
    :Example: 
    >>> from watex.exlib.sklearn import KMeans, MinMaxScaler
    >>> from watex.utils.plotutils import plot_clusters
    >>> from watex.datasets import fetch_data 
    >>> h= fetch_data('hlogs').frame 
    >>> # collect two features 'resistivity' and gamma-gamma logging values
    >>> h2 = h[['resistivity', 'gamma_gamma']] 
    >>> km = KMeans (n_clusters =3 , init= 'random' ) 
    >>> # scaled the data with MinMax scaler i.e. between ( 0-1) 
    >>> h2_scaled = MinMaxScaler().fit_transform(h2)
    >>> ykm = km.fit_predict(h2_scaled )
    >>> plot_clusters (3 , h2_scaled, ykm , km.cluster_centers_ )
        
    """
    n_clusters = int(
        _assert_all_types(n_clusters, int, float,  objname ="'n_clusters'" )
        )
    X, y_pred = check_X_y(
        X, 
        y_pred, 
        )

    if len(X.shape )!=2 or X.shape[1]==1: 
        ndim = 1 if X.shape[1] ==1 else np.ndim (X )
        raise ValueError(
            f"X is expected to be a two dimensional data. Got {ndim}!")
    # for consistency , convert y to array    
    y_pred = np.array(y_pred)
    
    colors = make_mpl_properties(n_clusters)
    markers = make_mpl_properties(n_clusters, 'markers')
    for n in range (n_clusters):
        plt.scatter (X[y_pred ==n, 0], 
                     X[y_pred ==n , 1],  
                     s= 50 , c= colors [n ], 
                     marker=markers [n], 
                     edgecolors=None if markers [n] =='x' else 'black', 
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
    
    if savefig is not None:
         savefigure(savefig, savefig )
    plt.close () if savefig is not None else plt.show() 
    
    
def plot_elbow (
        X,  n_clusters , n_init = 10 , max_iter = 300 , random_state=42 ,
        fig_size = (10, 4 ), marker = 'o', savefig= None, 
        **kwd): 
    """ Plot elbow method to find the optimal number of cluster, k', 
    for a given data. 
    
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training instances to cluster. It must be noted that the data
        will be converted to C ordering, which will cause a memory
        copy if the given data is not C-contiguous.
        If a sparse matrix is passed, a copy will be made if it's not in
        CSR format.

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

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
        
    savefig: str, default =None , 
        the path to save the figure. Argument is passed to 
        :class:`matplotlib.Figure` class. 
    marker: str, default='o', 
        cluster marker point. 
        
    kwd: dict
        Addionnal keywords arguments passed to :func:`matplotlib.pyplot.plot`
        
    Returns 
    --------
    ax: Matplotlib.pyplot axes objects 
    
    Example
    ---------
    >>> from watex.datasets import load_hlogs 
    >>> from watex.utils.plotutils import plot_elbow 
    >>> # get the only resistivy and gamma-gama values for example
    >>> res_gamma = load_hlogs ().frame[['resistivity', 'gamma_gamma']]  
    >>> plot_elbow(res_gamma, n_clusters=11)
    
    """
    distorsions =[] ; n_clusters = 11
    for i in range (1, n_clusters ): 
        km =KMeans (n_clusters =i , init= 'k-means++', 
                    n_init=n_init , max_iter=max_iter, 
                    random_state =random_state 
                    )
        km.fit(X) 
        distorsions.append(km.inertia_) 
            
    ax = _plot_elbow (distorsions, n_clusters =n_clusters,fig_size = fig_size ,
                      marker =marker , savefig =savefig, **kwd) 

    return ax 
    
def _plot_elbow (distorsions: list  , n_clusters:int ,fig_size = (10 , 4 ),  
               marker='o', savefig =None, **kwd): 
    """ Plot the optimal number of cluster, k', for a given class 
    
    :param distorsions: list - list of values withing the sum-squared-error 
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
    """ Plot the cost against the number of epochs  for the two different 
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
    >>> from watex.base import AdalineGradientDescent
    >>> from watex.utils.plotutils import plot_cost_vs_epochs
    >>> X, y = load_iris (return_X_y= True )
    >>> ada1 = AdalineGradientDescent (n_iter= 10 , eta= .01 ).fit(X, y) 
    >>> ada2 = AdalineGradientDescent (n_iter=10 , eta =.0001 ).fit(X, y)
    >>> plot_cost_vs_epochs (regs = [ada1, ada2] ) 
    """
    if not isinstance (regs, (list, tuple, np.array)): 
        regs =[regs]
    s = set ([hasattr(o, '__class__') for o in regs ])

    if len(s) != 1: 
        raise ValueError("All regression models should be estimators"
                         " already fitted.")
    if not list(s) [0] : 
        raise TypeError(f"Needs an estimator, got {type(s[0]).__name__!r}")
    
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

def plot_mlxtend_heatmap (df, columns =None, savefig=None,  **kws): 
    """ Plot correlation matrix array  as a heat map 
    
    :param df: dataframe pandas  
    :param columns: list of features, 
        If given, only the dataframe with that features is considered. 
    :param kws: additional keyword arguments passed to 
        :func:`mlxtend.plotting.heatmap`
    :return: :func:`mlxtend.plotting.heatmap` axes object 
    
    :example: 
        
    >>> from watex.datasets import load_hlogs 
    >>> from watex.utils.plotutils import plot_mlxtend_heatmap
    >>> h=load_hlogs()
    >>> features = ['gamma_gamma', 'sp',
                'natural_gamma', 'resistivity']
    >>> plot_mlxtend_heatmap (h.frame , columns =features, cmap ='PuOr')
    """
    import_optional_dependency('mlxtend', extra=(
        "Can't plot heatmap using 'mlxtend' package."))
  
    from mlxtend.plotting import (  heatmap 
            ) 
    cm = np.corrcoef(df[columns]. values.T)
    ax= heatmap(cm, row_names = columns , column_names = columns, **kws )
    
    if savefig is not None:
         savefigure(savefig, savefig )
    plt.close () if savefig is not None else plt.show() 
    
    return ax 

def plot_mlxtend_matrix(df, columns =None, fig_size = (10 , 8 ),
                        alpha =.5, savefig=None  ):
    """ Visualize the pair wise correlation between the different features in  
    the dataset in one place. 
    
    :param df: dataframe pandas  
    :param columns: list of features, 
        If given, only the dataframe with that features is considered. 
    :param fig_size: tuple of int (width, heigh) 
        Size of the displayed figure 
    :param alpha: figure transparency, default is ``.5``. 
    
    :return: :func:`mlxtend.plotting.scatterplotmatrix` axes object 
    :example: 
    >>> from watex.datasets import load_hlogs 
    >>> from watex.utils.plotutils import plot_mlxtend_matrix
    >>> import pandas as pd 
    >>> import numpy as np 
    >>> h=load_hlogs()
    >>> features = ['gamma_gamma', 'natural_gamma', 'resistivity']
    >>> data = pd.DataFrame ( np.log10 (h.frame[features]), columns =features )
    >>> plot_mlxtend_matrix (data, columns =features)

    """
    import_optional_dependency("mlxtend", extra = (
        "Can't plot the scatter matrix using 'mlxtend' package.") 
                               )
    from mlxtend.plotting import scatterplotmatrix
                                       
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

    if savefig is not None:
         savefigure(savefig, savefig )
    plt.close () if savefig is not None else plt.show() 
    
    return ax 

    
def savefigure (fig: object ,
             figname: str = None,
             ext:str ='.png',
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
        ... ['o',
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
    n=int(_assert_all_types(n, int, float, objname ="'n'"))
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
        return np.array(
            [round_modulo10(ii) for ii in np.linspace(
                             startpoint,endpoint, number_of_ticks)]
            )
    

            
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

def plot_errorbar0(
        ax,
        x_ar,
        y_ar,
        y_err=None,
        x_err=None,
        color='k',
        marker='x',
        ms=2, 
        ls=':', 
        lw=1, 
        e_capsize=2,
        e_capthick=.5,
        picker=None,
        **kws
 ):
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
    eobj = ax.errorbar(
        x_ar,
        y_ar,
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

def plot_errorbar(
    ax,
    x_ar,
    y_ar,
    y_err=None,
    x_err=None,
    color='k',
    marker='x',
    ms=2, 
    ls=':', 
    lw=1, 
    e_capsize=2,
    e_capthick=.5,
    picker=None,
    show_error_bars=True,  
    **kws
 ):
    """
    Convenience function to make an error bar instance.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes instance to put error bar plot on.
    x_ar : np.ndarray
        Array of x values to plot.
    y_ar : np.ndarray
        Array of y values to plot.
    y_err : np.ndarray, optional
        Array of errors in y-direction to plot.
    x_err : np.ndarray, optional
        Array of errors in x-direction to plot.
    color : str or tuple
        Color of marker, line, and error bar.
    marker : str
        Marker type to plot data as.
    ms : float
        Size of marker.
    ls : str
        Line style between markers.
    lw : float
        Width of line between markers.
    e_capsize : float
        Size of error bar cap.
    e_capthick : float
        Thickness of error bar cap.
    picker : float, optional
        Radius in points to be able to pick a point.
    show_error_bars : bool, default True
        If False, skip plotting the error bars.
    **kws : dict
        Additional keyword arguments passed to `ax.errorbar`.

    Returns
    -------
    errorbar_object : matplotlib.container.ErrorbarContainer
        Error bar object containing line data, error bars, etc.
    """
    if show_error_bars:
        yerr = y_err
        xerr = x_err
    else:
        # Skip error bars by setting them to None
        yerr = None
        xerr = None

    eobj = ax.errorbar(
        x_ar, y_ar, marker=marker, ms=ms, mfc='None', mew=lw, mec=color,
        ls=ls, xerr=xerr, yerr=yerr, ecolor=color, color=color,
        picker=picker, lw=lw, elinewidth=lw, capsize=e_capsize,
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

def _get_xticks_formatage (
        ax,  xtick_range, space= 14 , step=7,
        fmt ='{}',auto = False, ticks ='x', **xlkws):
    """ Skip xticks label at every number of spaces 
    :param ax: matplotlib axes 
    :param xtick_range: list of the xticks values 
    :param space: interval that the label must be shown.
    :param step: the number of label to skip.
    :param fmt: str, formatage type. 
    :param ticks: str, default='x', the ticks axis to format the labels. 
      can be ``'y'``. 
    :param auto: bool , if ``True`` a dynamic tick formatage will start. 
    
    """
    def format_ticks (ind, x):
        """ Format thick parameter with 'FuncFormatter(func)'
        rather than using:: 
            
        axi.xaxis.set_major_locator (plt.MaxNLocator(3))
        
        ax.xaxis.set_major_formatter (plt.FuncFormatter(format_thicks))
        """
        if ind % step ==0: 
            return fmt.format (ind)
        else: None 
        
    # show label every 'space'samples 
    if auto: 
        space = 10.
        step = int (np.ceil ( len(xtick_range)/ space )) 
        
    rotation = xlkws.get('rotation', 90 ) if 'rotation' in xlkws.keys (
        ) else xlkws.get('rotate_xlabel', 90 )
    
    if len(xtick_range) >= space :
        if ticks=='y': 
            ax.yaxis.set_major_formatter (plt.FuncFormatter(format_ticks))
        else: 
            ax.xaxis.set_major_formatter (plt.FuncFormatter(format_ticks))

        plt.setp(ax.get_yticklabels() if ticks=='y' else ax.get_xticklabels(), 
                 rotation = rotation )
    else: 
        
        # ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        # # ticks_loc = ax.get_xticks().tolist()
        # ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks_loc))
        # ax.set_xticklabels([fmt.format(x) for x in ticks_loc])
        tlst = [fmt.format(item) for item in  xtick_range]
        ax.set_yticklabels(tlst, **xlkws) if ticks=='y' \
            else ax.set_xticklabels(tlst, **xlkws) 
  
def _set_sns_style (s, /): 
    """ Set sns style whether boolean or string is given""" 
    s = str(s).lower()
    s = re.sub(r'true|none', 'darkgrid', s)
    return sns.set_style(s) 

def _is_target_in (X, y=None, tname=None): 
    """ Create new target name for tname if given 
    
    :param X: dataframe 
        dataframe containing the data for plotting 
    :param y: array or series
        target data for plotting. Note that multitarget outpout is not 
        allowed yet. Moroever, it `y` is given as a dataframe, 'tname' must 
        be supplied to retrive y as a pandas series object, otherwise an 
        error will raise. 
    :param tname: str,  
        target name. If given and `y` is ``None``, Will try to find `tname`
        in the `X` columns. If 'tname' does not exist, plot for target is 
        cancelled. 
        
    :return y: Series 
    """
    _assert_all_types(X, pd.DataFrame)
    
    if y is not None: 
        y = _assert_all_types(y , pd.Series, pd.DataFrame, np.ndarray)
        
        if hasattr (y, 'columns'): 
            if tname not in (y.columns): tname = None 
            if tname is None: 
                raise TypeError (
                    "'tname' must be supplied when y is a dataframe.")
            y = y [tname ]
        elif hasattr (y, 'name'): 
            tname = tname or y.name 
            # reformat inplace the name of series 
            y.name = tname 
            
        elif hasattr(y, '__array__'): 
            y = pd.Series (y, name = tname or 'target')
            
    elif y is None: 
        if tname in X.columns :
            y = X.pop(tname)

    return X, y 

def _toggle_target_in  (X , y , pos=None): 
    """ Toggle the target in the convenient position. By default the target 
    plot is the last subplots 
    
    :param X: dataframe 
        dataframe containing the data for plotting 
    :param y: array or series
        the target for  plotting. 
    :param pos: int, the position to insert y in the dataframe X 
        By default , `y` is located at the last position 
        
    :return: Dataframe 
        Dataframe containing the target 'y'
        
    """
    
    pos =  0 if pos ==0  else ( pos or X.shape [1])

    pos= int ( _assert_all_types(pos, int, float ) ) 
    ms= ("The positionning of the target is out of the bound."
         "{} position is used instead.")
    
    if pos > X.shape[1] : 
        warnings.warn(ms.format('The last'))
        pos=X.shape[1]
    elif pos < 0: 
        warnings.warn(ms.format(
            " Negative index is not allowed. The first")
                      )
        pos=0 
 
    X.insert (pos, y.name, y )
    
    return X
    
def _skip_log10_columns ( X, column2skip, pattern =None , inplace =True): 
    """ Skip the columns that dont need to put value in logarithms.
    
    :param X: dataframe 
        pandas dataframe with valid columns 
    :param column2skip: list or str , 
        List of columns to skip. If given as string and separed by the default
        pattern items, it should be converted to a list and make sure the 
        columns name exist in the dataframe. Otherwise an error with 
        raise. 
    :param pattern: str, default = '[#&*@!,;\s]\s*'
        The base pattern to split the text in `column2skip` into a columns
        
    :return X: Dataframe
        Dataframe modified inplace with values computed in log10 
        except the skipped columns. 
        
    :example: 
       >>> from watex.datasets import load_hlogs 
       >>> from watex.utils.plotutils import _skip_log10_columns 
       >>> X0, _= load_hlogs (as_frame =True ) 
       >>> # let visualize the  first3 values of `sp` and `resistivity` keys 
       >>> X0['sp'][:3] , X0['resistivity'][:3]  
       ... (0   -1.580000
            1   -1.580000
            2   -1.922632
            Name: sp, dtype: float64,
            0    15.919130
            1    16.000000
            2    24.422316
            Name: resistivity, dtype: float64)
       >>> column2skip = ['hole_id','depth_top', 'depth_bottom', 
                         'strata_name', 'rock_name', 'well_diameter', 'sp']
       >>> _skip_log10_columns (X0, column2skip)
       >>> # now let visualize the same keys values 
       >>> X0['sp'][:3] , X0['resistivity'][:3]
       ... (0   -1.580000
            1   -1.580000
            2   -1.922632
            Name: sp, dtype: float64,
            0    1.201919
            1    1.204120
            2    1.387787
            Name: resistivity, dtype: float64)
      >>> # it is obvious the `resistiviy` values is log10 
      >>> # while `sp` still remains the same 
      
    """
    X0 = X.copy () 
    if not is_iterable( column2skip): 
        raise TypeError ("Columns  to skip expect an iterable object;"
                         f" got {type(column2skip).__name__!r}")
        
    pattern = pattern or r'[#&*@!,;\s]\s*'
    
    if isinstance(column2skip, str):
        column2skip = str2columns (column2skip, pattern=pattern  )
    #assert whether column to skip is in 
    if column2skip:
        cskip = copy.deepcopy (column2skip) 
        column2skip = is_in_if(X.columns, column2skip, return_diff= True)
        if len(column2skip) ==len (X.columns): 
            warnings.warn("Value(s) to skip are not detected.")
        if inplace : 
            X[column2skip] = np.log10 ( X[column2skip] ) 
            X.drop (columns =cskip , inplace =True )
            return  
        else : 
            X0[column2skip] = np.log10 ( X0[column2skip] ) 
            
    return X0
    
def plot_bar(x, y, wh= .8,  kind ='v', fig_size =(8, 6), savefig=None,
             xlabel =None, ylabel=None, fig_title=None, **bar_kws): 
    """
    Make a vertical or horizontal bar plot.

    The bars are positioned at x or y with the given alignment. Their dimensions 
    are given by width and height. The horizontal baseline is left (default 0)
    while the vertical baseline is bottom (default=0)
    
    Many parameters can take either a single value applying to all bars or a 
    sequence of values, one for each bar.
    
    Parameters 
    -----------
    x: float or array-like
        The x coordinates of the bars. is 'x' for vertical bar plot as `kind` 
        is set to ``v``(default) or `y` for horizontal bar plot as `kind` is 
        set to``h``. 
        See also align for the alignment of the bars to the coordinates.
    y: float or array-like
        The height(s) for vertical and width(s) for horizonatal of the bars.
    
    wh: float or array-like, default: 0.8
        The width(s) for vertical or height(s) for horizaontal of the bars.
        
    kind: str, ['vertical', 'horizontal'], default='vertical'
        The kind of bar plot. Can be the horizontal or vertical bar plots. 
    bar_kws: dict, 
        Additional keywords arguments passed to : 
            :func:`~matplotlib.pyplot.bar` or :func:`~matplotlib.pyplot.barh`. 
    """
    
    assert str(kind).lower().strip() in ("vertical", 'v',"horizontal", "h"), (
        "Support only the horizontal 'h' and vertical 'v' bar plots."
        " Got {kind!r}")
    kind =str(kind).lower().strip()
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize =fig_size)
    if kind in ("vertical", "v"): 
        ax.bar (x, height= y, width =  wh , **bar_kws)
    elif kind in ("horizontal", "h"): 
        ax.barh (x , width =y , height =wh, **bar_kws)
        
    ax.set_xlabel (xlabel )
    ax.set_ylabel(ylabel) 
    ax.set_title (fig_title)
    if savefig is not  None: 
        savefigure (fig, savefig, dpi = 300)
        
    plt.close () if savefig is not None else plt.show() 
    
def plot_profiling (
    erp, 
    /, 
    station = None,  
    cz=None, 
    *, 
    style = 'classic', 
    fig_size = (10, 4), 
    cz_plot_kws= None,
    marker_kws= None, 
    savefig =None,
    ax =None,
    fig=None, 
    **plot_kws
    ): 
    """ 
    Visualizes  the resistivity profiling of  ERP data. 
    
    Function can overlain the selected conductive zone to the ERP if `cz` is 
    given. 
    
    Parameters 
    -----------
    erp: array_like 1d
        The electrical resistivity profiling array. If dataframe is passed, 
        `resistivity` column must be included. 
        
        .. versionchanged:: 0.2.1 
           Can henceforth accept dataframe that contains resistivity values. 
 
    station: str, int, optional 
        Station is used to visualize the conductive zone in the `erp` profile. 
        This seems useful if `cz` is not given. 
        When `station='auto'` it automatically detect the best conductive zone 
        assuming the very low resistivity in the profile and plot the 
        conductive zone. To have the expected results, `station` position or 
        `cz` must be given or the . 
        
       .. versionadded:: 0.2.1 
           Can henceforth pass the station to plot the conductive zone. 
           
    cz: array_like, optional, 
        The selected conductive zone. If ``None``, `cz` should not be plotted.
        
    style: str, default='classic'
        Matplotlib plottings style.
        
    fig_size: tuple, default= (10, 4) 
        Matplotlib figure size. 
        
    marker_kws: dict, default = {'marker':'o', 'c':'#9EB3DD' }
        The dictionnary to customize marker in the plot 
        
    cz_plot_kws: dict, default = {'ls':'-','c':'#0A4CEE', 'lw'L2 }
        The dictionnary to customize the conductize zone in the plot.
        
    savefig: str, optional 
        Save figure name. The default resolution dot-per-inch is ``300``. 
        
    ax: Matplotlib.pyplot.Axes, optional 
       Axe to collect the figure.
       
       .. versionadded:: 0.2.8 
          
       
    fig: Matplotlib.pyplot.figure, optional 
        Supply fig to save automatically the plot, otherwise, keep it 
        to ``None``.
          
    plot_kws: dict, 
        Additional keyword arguments passed to :func:`matplotlib.pyplot.plot` 
        function 
        
    Return
    --------
    ax: Matplotlib.pyplot.Axis 
        Return axis  
        
    Examples 
    ----------
    >>> from watex.datasets import make_erp 
    >>> from watex.utils.plotutils import plot_profiling 
    >>> d= make_erp (n_stations =56, seed = 42)
    >>> plot_profiling  (d.resistivity)
    >>> # read the frame and get the resistivity values 
    >>> plot_profiling (d.frame, station ='s07' ) 
    <AxesSubplot:xlabel='Stations', ylabel='App.resistivity ($\\Omega.m$)'>
    """
    plt.style.use (style )
    
    if hasattr ( erp , 'columns') and hasattr ( erp , '__array__'): 
        if 'resistivity' not in  erp.columns : 
            raise TypeError ("Missing resistivity column in the data.")
        
        erp = erp.resistivity 
    
    erp = check_y (erp , input_name ="sample of ERP data")
   
    if station is not None: 
        from .coreutils import defineConductiveZone 
        
        auto =False 
        if str(station).lower().strip () =='auto': 
            auto = True ; station =None 
        cz, *_  = defineConductiveZone(
            erp , station = station , auto= auto )
    
    if ax is None: 
        fig, ax = plt.subplots(1,1, figsize =fig_size)
        
    leg =[]
    
    zl, = ax.plot(np.arange(len(erp)), erp, 
                  label ='Electrical resistivity profiling', 
                  **plot_kws 
                  )
    marker_kws = marker_kws or dict (marker ='o', c='#9EB3DD' )
    ax.scatter (np.arange(len(erp)), erp, **marker_kws )
    
    leg.append(zl)    
        
    if cz is not None: 
        cz= check_y (cz, input_name ="Conductive zone 'cz'")
        z = np.ma.masked_values (erp, np.isin(erp, cz ))
        sample_masked = np.ma.array(
            erp, mask = ~z.fill_value.astype('bool') )
    
        cz_plot_kws = cz_plot_kws or dict (ls='-',c='#0A4CEE', lw =2 )
        czl, = ax.plot(
            np.arange(len(erp)), sample_masked, 
            label ='Conductive zone', 
            **cz_plot_kws
            )
        leg.append(czl)

    ax.set_xticks(range(len(erp)))
    
    if len(erp ) >= 14 : 
        ax.xaxis.set_major_formatter (plt.FuncFormatter(_format_ticks))
    else : 
        ax.set_xticklabels(
            ['S{:02}'.format(int(i)+1) for i in range(len(erp))]) 
         
    ax.set_xlabel('Stations')
    ax.set_ylabel('App.resistivity ($\Omega.m$)')
    ax.legend( handles = leg, loc ='best')
    ax.set_xlim ([-1, len(erp)])

    if savefig is not  None: 
        savefigure (fig, savefig, dpi = 300)
        
    plt.close () if savefig is not None else plt.show() 
    
    return ax 

def plot_skew2d (
    edis_list, /, 
    method ='Bahr',
    sensitivity='skew',
    mode ="frequency", 
    interpolate=True, 
    show_skewness=...,
    tolog10 =True, 
    interp_method ='cubic', 
    fill_value ='auto',
    get_sites_by=None, 
    top_label ='Stations', 
    cb_label ="Sensitivity (S)",
    spacing=80, 
    fig=None , 
    fig_size = (6, 3 ), 
    dpi = 300 , 
    font_size =5.,
    cmap='jet_r',
    plot_style =None,
    rotate_xlabel=0.,
    plot_contours=..., 
    ax=None, 
    savefig=..., 
   ) : 
    """
    Plot phase sensitive skew visualization. 
    
    Phase Sensitivity Skew (:math:`\eta`) is a dimensionality tool that 
    represents a measure of the skew of the  phases of the impedance 
    tensor. The parameter is thus unaffected by the distortion 
    effect, unlike the Swift-skew and ellipticity dimensionality 
    tools [1]_. 
    
    Values of :math:`\eta` > 0.3 are considered to represent 3D data. 
    Phase-sensitive skews less than 0.1 indicate 1D, 2D or distorted 
    2D (3-D /2-D) cases. Values of :math:`\eta` between 0.1 and 0.3 indicates 
    modified 3D/2D structures [2]_ according to `Bahr' methods. However,
    values :math:`\eta >=0.2` using the `Swift` methods, the smaller the value 
    :math:`\eta` ( close to :math:`0.`), the closer the structure to 2D 
    structure and vice versa.However, it is generally considered that 
    an electrical structure of :math:`\eta < 0.4` can be treated as a 2D 
    medium. 

    Parameters
    -----------
    edis_list: str, :class:`watex.edi.Edi` 
        Full path to edifiles.
        
        .. versionchanged:: 0.3.1 
           The parameter `edi_obj` is replaced by `edis_list` which  
           indicate  a collection of :term:`EDI`files. 
        
    method: str, default='Bahr': 
        Kind of correction. Can be ``swift`` for the remove distorsion proposed 
        by Swift in 1967 [3]_. The value close to 0. assume the 1D and 2D 
        structures, and 3D otherwise. Conversly to ``bahr`` for the remove 
        distorsion proposed  by Bahr in 1991 [2]_. The latter threshold is set 
        to 0.3. Above this value the structures is 3D. 
      
    sensitivity: str, default='skew'
       phase sensistive visualization. Can be rotational invariant 
       ``invariant``. Note that setting to ``mu`` or ``invariant`` does 
       not change any interpretation since the distortion of Z are all 
       rotational invariant whatever we're using the ``Bahr`` or ``swift``
       method. 
       
       .. versionchanged:: 0.3.1 
          The parameter `view` is deprecated and replaced with `sensitivity`. 
          
    mode:str, optional 
       X-axis coordinates for visualisation. plot either ``'frequency'`` or
       ``'periods'``.  The default is ``'frequency'`` 
       
    interpolate: bool, default=True
       Interpolate the data if NaN is found. 
     
    show_skewness: bool,default=False 
       Display the average skewness value. 
       
       .. versionadded:: 0.3.1 
          `show_skewness` display the average value of the whole Z tensor 
          at each frequency. 
          
    tolog10: bool, default=True 
       Compute the the logarithm base 10 of the frequency array. If the 
        frequency data is passed as log10 values, it should be turned to 
        ``False``. 
      
    interp_method: bool,default='cubic' 
       Data interpolation method. It could be ['nearest'|'linear'|'cubic']. 
        
    fill_value: float, str, default='auto' 
       Fill the interpolated grid at the egdes or surrounding NaN with 
       a filled value. The ``auto`` uses the forward and backward 
       fill strategy. 
       
    get_sites_by: str, optional
      Fetch the sites and place names on the map. It should be 
      [``'dataid'``|``'name'``]. The former  uses the names collected in 
      :term:`EDI` data id whereas the latter generates new names from the 
      sites id and the survey name. In that case, it expects the survey name  
      to be specified. By default, it merely use the sites id. 

    top_label: str, default='Stations' 
       Label  used to name the xticks in upper. 
       
    cb_label: str, default='Sensitivity (S)'
       The colorbar label.
       
    spacing: float, default=80. 
        The step in meters between two stations/sites. If given, 
        it creates an array of positions. 
    
    fig_size: tuple, default= (6, 2) 
        Matplotlib figure size. 
      
    dpi: int, default=300 
       Image resolution in dot-per-inch 
       
    cmap: str, default='jet_r' 
      Matplotlib colormap 
    
    plot_style: str, optional
       The kind of plot. It could be ['pcolormesh'|'imshow']. The default is 
       ``pcolormesh``. 
       
    rotate_xlabel: float, Optional 
      The degree angle to rotate the station/site label accordingly. 
      
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

    savefig: str, optional 
         Save figure name. The default resolution dot-per-inch is ``300``. 
         
    Return
    --------
    ax: Matplotlib.pyplot.Axis 
        Return axis  
        
    See Also 
    ---------
    watex.methods.em.Processing.skew: 
        Skew equation formulations. 
    watex.view.TPlot.plotSkew: 
        Give a consistent plot where user can customize the plot using the 
        plot parameter of :class:`watex.property.BasePlot` class.
        
    References 
    -----------
    .. [1] Bahr, K. (1988) Interpretation of the magnetotelluric impedance 
           tensor: regional induction 395 and local telluric distortion. J. 
           Geophys. Res., 62, 119–127.
    .. [2] Bahr, K. (1991) Geological noise in magnetotelluric data: 
           a classification of distortion types. 397 Phys. Earth Planet. 
           Inter., 66, 24–38.
    .. [3] Bahr, K., 1991. Geological noise in magnetotelluric data: a 
           classification of distortion types. Physics of the Earth and 
           Planetary Interiors 66 (1–2), 24–38.  
           
    Example
    ---------
    >>> import watex as wx 
    >>> from watex.utils.plotutils import plot_skew2d 
    >>> edi_sk = wx.fetch_data ("edis", return_data =True , samples = 20 ) 
    >>> plot_skew2d (edi_sk, show_skewness=True, interpolate=True, 
                     get_sites_by='name', mode='periods')  
    
    """
    from ..view.mlplot import plot2d 
    # validate and  the phase sensitivity skew 
    skew, mu, freqs, ymat , mode, sites = _validate_sensitivity_s(
        edis_list, 
        mode =mode, 
        sensitivity=sensitivity, 
        method =method,
        get_sites_by= get_sites_by, 
        interpolate = interpolate,  
        interp_method =interp_method, 
        fill_value =fill_value,
        )
    # ------------------------------------------------------
    # Create a pcolormesh plot with the mock data and the colormap
    # Assuming frequency on y-axis is on a log scale
    plot_style = ( 'imshow' if str(plot_style).lower() =='imshow' 
                  else 'pcolormesh' 
                  )
    ax = plot2d (
          ymat,
          y = np.log10 ( freqs) if tolog10 else freqs , 
          cmap =cmap, 
          cb_label =cb_label, 
          top_label =top_label , 
          rotate_xlabel =rotate_xlabel, 
          distance = spacing , 
          fig_size  = fig_size ,
          fig_dpi = dpi , 
          font_size =font_size,
          plt_style = plot_style,
          stnlist= sites, 
          plot_contours =False if plot_contours in (
              False, ...) else True, 
          
          )
    show_skewness = False if show_skewness in (False, ...) else True 
    # view skewness 
    if show_skewness: 
        aver_skewn=np.around (np.average(ymat[ ~np.isnan(ymat)]), 3)
        ax.text(x= 0. , y= np.nanmax(freqs)/2, 
                 s="aver.-shewness:{}={}".format(str(method).capitalize(), 
                aver_skewn),  
                 fontdict= dict (style ='italic',  bbox =dict(
                     boxstyle='round',
                     facecolor ='#FFFFFF'#CED9EF'
                     ))
                 ) 
    
    ylabel = ('Log(f) [Hz]' if tolog10 else 'Frequency [Hz]'
              ) if mode=='frequency'else ( 
                  'Log10Period [s]' if tolog10 else 'Period [s]'
                  )
    ax.set_ylabel(ylabel)
     
    if savefig is not  None: 
        savefigure (fig, savefig, dpi = 300)
        
    plt.close () if savefig is not None else plt.show() 
    
    return ax 

def _validate_sensitivity_s (
        edis_list, /, mode ="frequency", sensitivity='skew', method ='Bahr',
        get_sites_by=None, interpolate =..., interp_method ='cubic', 
        fill_value ='auto', ): 
    """Compute the sensitivity S and return appropriate argument for 
    plot. 
    
    An isolate part of  :func:`plot_skew1d` or :func:`plot_skew2d`. 
    """
    from watex.methods import EMAP 
    
    sensitivity = sensitivity or 'skew'
    
    if ('inv'  in str (sensitivity).lower()
        or 'rot' in str (sensitivity).lower()
        or 'mu' in str (sensitivity).lower()
        ) : 
        sensitivity ='mu'
        
    if 'period' in str(mode).lower(): 
        mode ='period'
        
    # if str(threshold_line).lower()=='true': 
    #     threshold_line = str(method).lower() 
    po =  EMAP().fit(edis_list)
    
    # remove the outliers in the data
    # and filled with NaN 
    skew, mu =po.skew(method = method, suppress_outliers = True  )
    freqs =  1/ po.freqs_ if mode =='period' else po.freqs_ 
    ymat = skew if sensitivity =='skew' else mu 
    
    interpolate = False if interpolate in (False, ...) else True 
    if interpolate: 
        ymat = interpolate_grid(ymat, method = interp_method , 
                                fill_value= fill_value, view=False, 
                                )
    #---- get the station names
    sites = po.id
    if get_sites_by: 
        regex = re.compile('\d+', re.IGNORECASE)
        if hasattr (po, 'edifiles'): 
            if str(get_sites_by).lower()=='name':
                # get the first name of dataId of the EDI ediObjs  and filled
                # the rename dataId. remove the trail'_'  
                name = po.survey_name or  regex.sub(
                    '', po.ediObjs_[0].Head.dataid).replace('_', '') 
                # remove prefix )'S' and keep only the digit 
                dataid = list(map(lambda n: name + n, regex.findall(
                    ''.join(po.id)) ))
                
                sites = dataid
            else: 
                sites = list(
                    map(lambda obj: obj.Head.dataid, po.ediObjs_))

    return skew, mu, freqs, ymat , mode,  sites 

def plot_skew1d (
    edis_list,
    /, 
    method='Bahr', 
    sensitivity=None,
    mode=None, 
    threshold_line =None, 
    show_skewness: bool=..., 
    fig_size = (7, 5), 
    savefig = None, 
    dpi=300, 
    style=None, 
    ax=None, 
    **kws 
    ):
    """ Plot phase sensitive skew visualization. 
    
    Phase Sensitivity Skew (:math:`\eta`) is a dimensionality tool that 
    represents a measure of the skew of the  phases of the impedance 
    tensor. The parameter is thus unaffected by the distortion 
    effect, unlike the Swift-skew and ellipticity dimensionality 
    tools [1]_. 
    
    Values of :math:`\eta` > 0.3 are considered to represent 3D data. 
    Phase-sensitive skews less than 0.1 indicate 1D, 2D or distorted 
    2D (3-D /2-D) cases. Values of :math:`\eta` between 0.1 and 0.3 indicates 
    modified 3D/2D structures [2]_ according to `Bahr' methods. However,
    values :math:`\eta >=0.2` using the `Swift` methods, the smaller the value 
    :math:`\eta` ( close to :math:`0.`), the closer the structure to 2D 
    structure and vice versa.However, it is generally considered that 
    an electrical structure of :math:`\eta < 0.4` can be treated as a 2D 
    medium. Here as the ``threshold_line`` for :meth:`\eta` using the 
    Swift method should be set as `0.4`. 
    
    
    Parameters
    -----------
    edis_list: str, :class:`watex.edi.Edi` 
        Full path to edifiles.
        
        .. versionchanged:: 0.3.1 
           The parameter `edi_obj` is replaced by `edis_list` which  
           indicate  a collection of :term:`EDI`files. 
        
    method: str, default='Bahr': 
        Kind of correction. Can be ``swift`` for the remove distorsion proposed 
        by Swift in 1967 [3]_. The value close to 0. assume the 1D and 2D 
        structures, and 3D otherwise. Conversly to ``bahr`` for the remove 
        distorsion proposed  by Bahr in 1991 [2]_. The latter threshold is set 
        to 0.3. Above this value the structures is 3D. 
      
    sensitivity: str, default='skew'
       phase sensistive visualization. Can be rotational invariant 
       ``invariant``. Note that setting to ``mu`` or ``invariant`` does 
       not change any interpretation since the distortion of Z are all 
       rotational invariant whatever we're using the ``Bahr`` or ``swift``
       method. 
       
       .. versionchanged:: 0.3.1 
          The parameter `view` is deprecated and replaced with `sensitivity`. 
          
    mode:str, optional 
       X-axis coordinates for visualisation. plot either ``'frequency'`` or
       ``'periods'``.  The default is ``'frequency'`` 
       
    threshold_line: float, optional
       Visualize th threshold line. Can be ['bahr', 'swift', 'both']:
           
       - Note that when method is set to ``swift``, the value close to close 
         to :math:`0.` assume the 1D and 2D structures,  and 3D otherwise. 
       - when method is set to ``Bahr``, :math:`\mu > 0.3``  is 3D structures, 
         between :math:`[0.1 - 0.3]` assumes modified 3D/2D structures whereas 
         :math:`<0.1` 1D, 2D or distorted 2D. 
      
    show_skewness: bool,default=False 
       Display the average skewness value. 
       
       .. versionadded:: 0.3.1 
          `show_skewness` display the average value of the whole Z tensor 
          at each frequency. 
 
    fig_size: tuple, default= (10, 4) 
        Matplotlib figure size. 
        
    savefig: str, optional 
         Save figure name. The default resolution dot-per-inch is ``300``. 
         
    dpi: float, default=300. 
       Dot-per-inch for image resolution. 
       
    style: str, default='classic'
        Matplotlib plottings style.
       
    ax: Matplotlib.pyplot.Axes, optional 
       Axe to collect the figure. Could be used to support other axes. 
       
    kws: dict, 
       Matplotlib Axes scatterplot additional keywords arguments. 
        
    Return
    --------
    ax: Matplotlib.pyplot.Axis 
        Return axis  
        
    See Also 
    ---------
    watex.methods.em.Processing.skew: 
        Skew equation formulations. 
    watex.view.TPlot.plotSkew: 
        Give a consistent plot where user can customize the plot using the 
        plot parameter of :class:`watex.property.BasePlot` class.
        
    References 
    -----------
    .. [1] Bahr, K. (1988) Interpretation of the magnetotelluric impedance 
           tensor: regional induction 395 and local telluric distortion. J. 
           Geophys. Res., 62, 119–127.
    .. [2] Bahr, K. (1991) Geological noise in magnetotelluric data: 
           a classification of distortion types. 397 Phys. Earth Planet. 
           Inter., 66, 24–38.
    .. [3] Bahr, K., 1991. Geological noise in magnetotelluric data: a 
           classification of distortion types. Physics of the Earth and 
           Planetary Interiors 66 (1–2), 24–38.  
    Examples 
    ---------
    >>> import watex as wx 
    >>> from watex.utils.plotutils import plot_skew1d 
    >>> edi_sk = wx.fetch_data ("edis", return_data =True , samples = 20 ) 
    >>> plot_skew1d (edi_sk) 
    >>> plot_skew1d (edi_sk, threshold_line= True) 
    """
    if style is not None:
        plt.style.use (style )
        
    # validate and  the phase sensitivity skew 
    skew, mu, freqs, ymat , mode, _ = _validate_sensitivity_s(
        edis_list, 
        mode =mode, 
        sensitivity=sensitivity, 
        method =method,
        )
        
    if str(threshold_line).lower()=='true': 
        threshold_line = str(method).lower() 
            
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize =fig_size)

    #---manage threshold line ------
    thr_code = {"bahr": [1] , "swift":[ 2] , 'both':[1, 2] }

    if threshold_line is not None: 
        if str(threshold_line).lower() in ("*", "both"): 
            threshold_line = 'both'
            
    ct = thr_code.get(str(threshold_line).lower(), None ) 
    
    for i in range (skew.shape[1]): 
        ax.scatter ( freqs, reshape (ymat[:, i])/1.8,**kws )
        
    if ct: 
        for m in ct: 
            plt.axhline(y=0.4 if m==2 else 0.3 , color="r" if m==1 else "r",
                        linestyle="-",
                        label=f'threshold: $\mu={0.4 if m==2 else 0.3}$'
                        )
           
            ax.legend() 
            
    show_skewness = False if show_skewness  in (False, ...) else True 
    # view skewness 
    if show_skewness: 
        ymat=reshape (ymat[:, i])
        aver_skewn=np.around (np.average(ymat[ ~np.isnan(ymat)]), 3)
        ax.text(x= np.nanmin(freqs) , y= np.nanmax(ymat), 
                 s="aver.-shewness:{}={}".format(str(method).capitalize(), 
                aver_skewn),  
                 fontdict= dict (style ='italic',  bbox =dict(
                     boxstyle='round',
                     facecolor ='#FFFFFF'#CED9EF'
                     ))
                 ) 
    ax.set_xscale('log')
    
    ax.set_xlabel('Period ($s$)' if mode=='period' 
                  else 'Frequency ($H_z$)')
    ax.set_ylabel(f"{'Skew' if sensitivity =='skew' else 'Rot.Invariant'}" + "($\mu$)")

    plt.xlim ([ freqs.min() , freqs.max()])
    
    #plt.xlim() 
    if savefig is not  None: 
        savefigure (fig, savefig, dpi = dpi)
        
    plt.close () if savefig is not None else plt.show() 
    
    return ax 

def _format_ticks (value, tick_number, fmt ='S{:02}', nskip =7 ):
    """ Format thick parameter with 'FuncFormatter(func)'
    rather than using `axi.xaxis.set_major_locator (plt.MaxNLocator(3))`
    ax.xaxis.set_major_formatter (plt.FuncFormatter(format_thicks))
    
    :param value: tick range values for formatting 
    :param tick_number: number of ticks to format 
    :param fmt: str, default='S{:02}', kind of tick formatage 
    :param nskip: int, default =7, number of tick to skip 
    
    """
    if value % nskip==0: 
        return fmt.format(int(value)+ 1)
    else: None 
    
#XXX OPTIMIZE 
def plot_confidence (
    data = None, 
    *,  
    y=None, 
    x=None, 
    ci =.95 ,  
    kind ='line', 
    b_samples = 1000, 
    **sns_kws
    ): 
    """ Plot confidence data 
    
    Confidence Interval (CI)  is a type of estimate computed from the statistics 
    of the observed data which gives a range of values that’s likely to 
    contain a population parameter with a particular level of confidence.
    CI as a concept was put forth by Jerzy Neyman in a paper published 
    in 1937. There are various types of the confidence interval, some of 
    the most commonly used ones are: CI for mean, CI for the median, CI for 
    the difference between means, CI for a proportion and CI for the difference 
    in proportions.
    
    Parameters 
    ------------
    data: pandas.DataFrame, numpy.ndarray, mapping, or sequence
       Input data structure. Either a long-form collection of vectors 
       that can be assigned to named variables or a wide-form dataset 
       that will be internally reshaped.

    x, y: vectors or keys in data
       Variables that specify positions on the x and y axes.
       
    ci: float, default=.95 
       Confidence value. 
       
    kind: str, default='line' 
       kind of confidence intervval plot. 
      
    b_samples: int, default=1000
        Number of bootstraps to use for computing the confidence interval.
        
    sns_kws: dict, 
       Keywords arguments passed to the `sns.lineplot` or `sns.regplot`
       
    Returns 
    ----------
    ax: matplotlib.axes.Axes
       The matplotlib axes containing the plot.
       
    """   
    #y = np.array (y) 
    #x= x or ( np.arange (len(y)) if 
    ax=None 
    if 'lin' in str(kind).lower(): 
        ax = sns.lineplot(data= data, x=x, y=y, ci=ci, **sns_kws)
    elif 'reg' in  str(kind).lower(): 
        ax = sns.regplot(data = data, x=x, y=y, ci=ci, **sns_kws ) 
    else: 
        if not y: 
            raise ValueError("y should not be None when using the boostrapping"
                             " for plotting the confidence interval.")
        b_samples = _assert_all_types(
            b_samples, int, float, objname="Bootstrap samples `b_samples`")
        
        from sklearn.metrics import resample 
        # configure bootstrap
        n_iterations = 1000 # here k=no. of bootstrapped samples
        n_size = int(len(y))
          
        # run bootstrap
        medians = list()
        for i in range(n_iterations):
           s = resample(y, n_samples=n_size);
           m = np.median(s);
           medians.append(m)
          
        # plot scores
        plt.hist(medians)
        plt.show()
          
        # confidence intervals
        p = ((1.0-ci)/2.0) * 100
        lower =  np.percentile(medians, p)
        p = (ci+((1.0-ci)/2.0)) * 100
        upper =  np.percentile(medians, p)
  
        print(f"\n{ci*100} confidence interval {lower} and {upper}")
    
    return ax 

def plot_confidence_ellipse (x, y ): 
    """ Plot a confidence ellipse of a two-dimensional dataset 
    
    This function plots the confidence ellipse of the covariance of 
    the given array-like variables x and y. The ellipse is plotted 
    into the given axes-object ax.
    
    The approach that is used to obtain the correct geometry 
    is explained and proved here:
      https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html
      
    The method avoids the use of an iterative eigen decomposition 
    algorithm and makes use of the fact that a normalized covariance 
    matrix (composed of pearson correlation coefficients and ones) is 
    particularly easy to handle.
    
    """
    fig, ax_nstd = plt.subplots(figsize=(6, 6))
    # dependency_nstd = [[0.8, 0.75],
    #                    [-0.2, 0.35]]
    mu = 0, 0
    # scale = 8, 5
    ax_nstd.axvline(c='grey', lw=1)
    ax_nstd.axhline(c='grey', lw=1)
    
    #x, y = get_correlated_dataset(500, dependency_nstd, mu, scale)
    ax_nstd.scatter(x, y, s=0.5)
    
    confidence_ellipse(x, y, ax_nstd, n_std=1,
                       label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(x, y, ax_nstd, n_std=2,
                       label=r'$2\sigma$', edgecolor='fuchsia',
                       linestyle='--')
    confidence_ellipse(x, y, ax_nstd, n_std=3,
                       label=r'$3\sigma$', edgecolor='blue', 
                       linestyle=':')
    
    ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
    ax_nstd.set_title('Different standard deviations')
    ax_nstd.legend()
    plt.show()
    
def confidence_ellipse(
    x, 
    y, 
    ax, 
    n_std=3.0, 
    facecolor='none', 
    **kwargs
    ):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    mpl.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)  
   
def plot_strike (
    list_of_edis, /, 
    kind = 2,
    period_tolerance=.05, 
    text_pad =1.65 , 
    rot_z=0. , 
    **kws
    ): 
    
    extra =("PlotStrike uses 'mtpy' or 'pycsamt' as dependency."
            )
    import_optional_dependency ('mtpy', extra = extra )
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    from mtpy.imaging.plotstrike import PlotStrike
    from ..property import IsEdi
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    if isinstance ( list_of_edis, str): 
        if os.path.isdir ( list_of_edis ): 
            list_of_edis = [
                os.path.join(list_of_edis,  f) for f in os.listdir (list_of_edis)
                            if str(f).lower().endswith ('.edi')]
        elif os.path.isfile (list_of_edis): 
            list_of_edis =[list_of_edis ]
          
    # now check whether is valid EDI     
    # list comprehension faster than
    # tuple (map (lambda f:  IsEdi._assert_edi (f ), list_of_edis ) )
    [ IsEdi._assert_edi (f ) for f in list_of_edis ] 
    # suppress third party verbosity 
    with nullify_output():
        PlotStrike(
            fn_list=list_of_edis,
            plot_type=kind,
            **kws
            )
        
plot_strike.__doc__="""\ 
Plot the strike estimated from the invariants and phase tensor. 
in a rose diagram of xy plot. 

Parameters 
------------
list_of_edis: list, 
    full paths to .edi files to plot or list of :term:`EDI` files. 
    
   .. versionchanged:: 0.2.0 
      No need to provide a list of term:`EDI` files. Henceforth `list_of_edis`
      accepts the EDI path-like object of single EDI file then asserts 
      the validity of the EDI files afterward. 

kind: int, default=2 
    Can be [ 1 | 2 ] where:
       
   - *1* to plot individual decades in one plot
   - *2* to plot all period ranges into one polar diagram for each 
     strike angle estimation
         
   One could try also plot_type = 1 to plot by decade
                                                                   
fig_num: int, default=1, 
   figure number to be plotted. *Default* is 1
        
font_size: float, default=10, 
   Figure size 
   
rot_z: float, default=0., 
   angle of rotation clockwise positive. 

period_tolerance: float, default=.05 
    Tolerance level to match periods from different edi files.
    *Default* is 0.05

text_pad: float, default=1.65
   padding of the angle label at the bottom of each
                 polar diagram.  *Default* is 1.65
plot_range:  str, tuple 
   The period range to estimate the strike angle. It can be 
   [ 'data' | (period_min,period_max) ].  Options are:
       
   * *'data'* for estimating the strike for all periods
     in the data.
   * (pmin,pmax) for period min and period max, input as
     (log10(pmin),log10(pmax))

plot_tipper: [ True | False ]
    - True to plot the tipper strike
    - False to not plot tipper strike
   
pt_error_floor: int, optional 
   Maximum error in degrees that is allowed to 
   estimate strike. *Default* is None allowing all 
   estimates to be used.

fold: [ True | False ]
   * True to plot only from 0 to 180
   * False to plot from 0 to 360
            
plot_orthogonal: [ True | False]
    * True to plot the orthogonal strike directions
    * False to not

color: [ True | False ]
    * True to plot shade colors
    * False to plot all in one color
              
color_inv:str, 
   color of invariants plots

color_pt: str, 
   color of phase tensor plots

color_tip: str 
   color of tipper plots

ring_spacing: float, optional 
    spacing of rings in polar plots

ring_limits: tuple of int, 
   plot limits (min count, max count) set each plot have these limits 
                    
plot_orientation: str, [ 'h' | 'v' ] 
   horizontal or vertical plots


See More
--------
Plots the strike angle as determined by invariants of the impedance tensor 
(Weaver et al. [2003] [1]_) and phase tensor azimuth 
(Caldwell et al. [2004] [2]_) 

The data is split into decades where the histogram for each is plotted in
the form of a rose diagram with a range of 0 to 180 degrees.
Where 0 is North and 90 is East.  The median angle of the period band is
set in polar diagram.  The top row is the strike estimated from
the invariants of the impedance tensor.  The bottom row is the azimuth
estimated from the phase tensor.  If tipper is 'y' then the 3rd row is the
strike determined from the tipper, which is orthogonal to the induction
arrow direction.

References 
----------
.. [1] Weaver J.T, Lilley F.E.M.(2003)  Invariants of rotation of axes and indicators of
       dimensionality in magnetotellurics, Australian National University,
       University of Victoria; http://bib.gfz-potsdam.de/emtf/2007/pdf/lilley.pdf
.. [2] T. Grant Caldwell, Hugh M. Bibby, Colin Brown, The magnetotelluric phase tensor, 
       Geophysical Journal International, Volume 158, Issue 2, August 2004, 
       Pages 457–469, https://doi.org/10.1111/j.1365-246X.2004.02281.x
Examples 
----------
>>> import os 
>>> from watex.datasets import fetch_data 
>>> from watex.utils.plotutils import plot_strike 
>>> from watex.datasets._io import get_data # get edidata in cache  
>>> fetch_data ( 'huayuan', samples = 25 ) # store edi in cache 
>>> # get the edi in cache and plotStrike 
>>> edi_fn_lst = [os.path.join(get_data(),ff) for ff in os.listdir(get_data()) 
...         if ff.endswith('.edi')] 
>>> plot_strike(edi_fn_lst )

"""
def plot_text (
    x, y, 
    text=None , 
    data =None, 
    coerce =False, 
    basename ='S', 
    fig_size =( 7, 7 ), 
    show_line =False, 
    step = None , 
    xlabel ='', 
    ylabel ='', 
    color= 'k', 
    mcolor='k', 
    lcolor=None, 
    show_leg =False,
    linelabel='', 
    markerlabel='', 
    ax=None, 
    **text_kws
    ): 
    """ Plot text(s) indicating each position in the line. 
    
    Parameters 
    -----------
    x, y: str, float, Array-like 
        The position to place the text. By default, this is in data 
        coordinates. The coordinate system can be changed using the 
        transform parameter.
        
    text: str, 
        The text
        
    data: pd.DataFrame, 
       Data containing x and y names. Need to be supplied when x and y 
       are given as string names. 
       
    coerce:bool, default=False 
       Force the plot despite the given textes do not match the number of  
       positions `x` and `y`. If ``False``, number of positions must be 
       consistent with x and y, otherwise error raises. 
       
    basename: str, default='S' 
       the text to prefix the position when the text is not given. 
       
    fig_size: tuple, default=(7, 7) 
       Matplotlib figure size.
       
    show_line: bool, default=False 
       Display the line from x, y. 
       
    step: int,Optional 
       The number of intermediate positions to skip in the plotting text. 
       
    xlabel, ylabel: str, Optional, 
       The labels of x and y. 
       
    color: str, default='k', 
       Text color.
       
    mcolor: str, default='k', 
       Marker color. 
       
    lcolor: str, Optional 
       Line color if `show_line` is set to ``True``. 
       
    show_leg: bool, default=False 
       Display the legend of line and marker labels. 
       
    linelabel, markerlabel: str, Optional 
        The labels of the line and marker. 
       
    ax: Matplotlib.Axes, optional 
       Support plot to another axes 
       
       .. versionadded:: 0.2.5 
       
    text_kws: dict, 
       Keyword arguments passed to :meth:`matplotlib.axes.Axes.text`. 

    Return 
    -------
    ax: Matplotlib axes 
    
    Examples 
    --------
    >>> import watex as wx 
    >>> data =wx.make_erp (as_frame =True, n_stations= 7 )
    >>> x , y =[ 0, 1, 3 ], [2, 3, 6] 
    >>> texto = ['AMT-E1147', 'AMT-E1148',  'AMT-E180']
    >>> plot_text (x, y , text = texto)# no need to set  coerce, same length 
    >>> data =wx.make_erp (as_frame =True, n_stations= 20 )
    >>> x , y = data.easting, data.northing
    >>> text1 = ['AMT-E1147', 'AMT-E1148',  'AMT-E180'] 
    >>> plot_text (x, y , coerce =True , text = text1 , show_leg= True, 
                   show_line=True, linelabel='E1-line', markerlabel= 'Site', 
               basename ='AMT-E0' 
               )
    """
    # assume x, y  series are passed 
    if isinstance(x, str) or hasattr ( x, 'name'): 
        xlabel = x  if isinstance(x, str) else x.name 
        
    if isinstance(y, str) or hasattr ( y, 'name'): 
        ylabel = y  if isinstance(y, str) else y.name 
        
    if x is None and  y is None:
        raise TypeError("x and y are needed for text plot. NoneType"
                        " cannot be plotted.")    
        
    x, y = assert_xy_in(x, y, data = data ) 

    if text is None and not coerce: 
       raise TypeError ("Text cannot be plotted. To force plotting text with"
                        " the basename, set ``coerce=True``.")

    text = is_iterable(text , exclude_string= True , transform =True )
    
    if ( len(text) != len(y) 
        and not coerce) : 
        raise ValueError("In principle text array and x/y must be consistent."
                         f" Got {len(text)} and {len(y)}. To plot anyway,"
                         " set ``coerce=True``.")
    if coerce : 
        basename =str(basename)
        text += [f'{basename}{i+len(text):02}' for i in range (len(y) )]

    if step is not None: 
        step = _assert_all_types(step , float, int , objname ='Step') 
        for ii in range(len(text)): 
            if not ii% step ==0: 
                text[ii]=''

    if ax is None: 
        
        fig, ax = plt.subplots(1,1, figsize =fig_size)
    
    # plot = ax.scatter if show_line else ax.plot 
    ax_m = None 
    if show_line: 
        ax.plot (x, y , label = linelabel, color =lcolor 
                 ) 
        
    for ix, iy , name in zip (x, y, text ): 
        ax.text ( ix , iy , name , color = color,  **text_kws)
        if name !='':
           ax_m  = ax.scatter ( [ix], [iy] , marker ='o', color =mcolor, 
                       )
  
    ax.set_xlabel (xlabel)
    ax.set_ylabel (ylabel) 
    
    ax_m.set_label ( markerlabel) if ax_m is not None else None 
    
    if show_leg : 
        ax.legend () 
        
    return ax 

def plot_voronoi(
    X, y, *, 
    cluster_centers,
    ax= None,
    show_vertices=False, 
    line_colors='k',
    line_width=1. ,
    line_alpha=1.,   
    fig_size = (7, 7), 
    cmap='Set1', 
    show_grid=True, 
    alpha=0.2, 
    fig_title = ''
    ):
    """Plots the Voronoi diagram of the k-means clusters overlaid with 
    the data
    
    Parameters 
    -----------
    X, y : NDarray, Arraylike 1d 
      Data training X and y. Must have the same length 
    cluster_center: int, 
       Cluster center. Cluster center can be obtain withe KMeans algorithms 
    show_vertices : bool, optional
        Add the Voronoi vertices to the plot.
    line_colors : string, optional
        Specifies the line color for polygon boundaries
    line_width : float, optional
        Specifies the line width for polygon boundaries
    line_alpha : float, optional
        Specifies the line alpha for polygon boundaries
    point_size : float, optional
        Specifies the size of points
    ax: Matplotlib.Axes 
       Maplotlib axes. If `None`, a axis is created instead. 
       
    fig_size: tuple, default = (7, 7) 
       Size of the figures. 
       
    Return
    -------
    ax: Matplotlib.Axes 
       Axes to support the figure
       
    Examples 
    ---------
    >>> from sklearn.datasets import make_moons
    >>> from sklearn.cluster import KMeans 
    >>> from watex.utils.plotutils import plot_voronoi
    >>> X, y = make_moons(n_samples=2000, noise=0.2)
    >>> km = KMeans (n_init ='auto').fit(X, y ) 
    >>> plot_voronoi ( X, y , cluster_centers = km.cluster_centers_) 
    """
    X, y = check_X_y(X, y, )
    cluster_centers = check_array(cluster_centers )
    
    if ax is None: 
        fig, ax = plt.subplots(1,1, figsize =fig_size)
        
    from scipy.spatial import Voronoi, voronoi_plot_2d
    
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, alpha=0.2, 
               label = 'Voronoi plot')
    vor = Voronoi(cluster_centers)
    voronoi_plot_2d(vor, ax=ax, show_vertices=show_vertices, 
                    alpha=alpha, 
                    line_colors=line_colors,
                    line_width=line_width ,
                    line_alpha=line_alpha,  
                    )
    #ax.legend() 
    ax.set_title (fig_title , fontsize=20)
    #fig.suptitle(fig_title, fontsize=20) 
    # Make the right and bottom spines thicker and black
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color('black')
    
    if show_grid: plt.grid() 
    else: ax.grid(False)
    
    return ax 
 
    
def _make_axe_multiple ( n, ncols = 3 , fig_size =None, fig =None, ax= ... ): 
    """ Make multiple subplot axes from number of objects. """
    if is_iterable (n): 
       n = len(n) 
     
    nrows = n // ncols + ( n % ncols ) 
    if nrows ==0: 
       nrows =1 
       
    if ax in ( ... , None) : 
        fig, ax = plt.subplots (nrows, ncols, figsize = fig_size )  
    
    return fig , ax 
    
def plot_roc_curves (
   clfs, /, 
   X, y, 
   names =..., 
   colors =..., 
   ncols = 3, 
   score=False, 
   kind="inone",
   ax = None,  
   fig_size=( 7, 7), 
   **roc_kws ): 
    """ Quick plot of Receiving Operating Characterisctic (ROC) of fitted models 
    
    Parameters 
    ------------
    clfs: list, 
       list of models for ROC evaluation. Model should be a scikit-learn 
       or  XGBoost estimators 
       
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training instances to cluster. It must be noted that the data
        will be converted to C ordering, which will cause a memory
        copy if the given data is not C-contiguous.
        If a sparse matrix is passed, a copy will be made if it's not in
        CSR format.
    
    y : ndarray or Series of length (n_samples, )
        An array or series of target or class values. Preferably, the array 
        represent the test class labels data for error evaluation.  
        
    names: list, 
       List of model names. If not given, a raw name of the model is passed 
       instead.
     
    kind: str, default='inone' 
       If ``['individual'|'2'|'single']``, plot each ROC model separately. 
       Any other value, group of ROC curves into a single plot. 
       
       .. versionchanged:: 0.2.5 
          Parameter `all` is deprecated and replaced by `kind`. It henceforth 
          accepts arguments ``allinone|1|grouped`` or ``individual|2|single``
          for plotting mutliple ROC curves in one or separate each ROC curves 
          respecively. 
          
    colors : str, list 
       Colors to specify each model plot. 
       
    ncols: int, default=3 
       Number of plot to be placed inline before skipping to the next column. 
       This is feasible if `many` is set to ``True``. 
       
    score: bool,default=False
      Append the Area Under the curve score to the legend. 
      
      .. versionadded:: 0.2.4 
      
    kws: dict,
        keyword argument of :func:`sklearn.metrics.roc_curve 
        
    Return
    -------
    ax: Axes.Subplot. 
    
    Examples 
    --------
    >>> from watex.utils.plotutils import plot_roc_curves 
    >>> from sklearn.datasets import make_moons 
    >>> from watex.exlib import ( train_test_split, KNeighborsClassifier, SVC ,
    XGBClassifier, LogisticRegression ) 
    >>> X, y = make_moons (n_samples=2000, noise=0.2)
    >>> X, Xt, y, yt = train_test_split (X, y, test_size=0.2) 
    >>> clfs = [ m().fit(X, y) for m in ( KNeighborsClassifier, SVC , 
                                         XGBClassifier, LogisticRegression)]
    >>> plot_roc_curves(clfs, Xt, yt)
    Out[66]: <AxesSubplot:xlabel='False Positive Rate (FPR)', ylabel='True Positive Rate (FPR)'>
    >>> plot_roc_curves(clfs, Xt, yt,kind='2', ncols = 4 , fig_size = (10, 4))
    """
    from .validator import  get_estimator_name
    
    kind = '2' if str(kind).lower() in 'individual2single' else '1'

    def plot_roc(model, data, labels, score =False ):
        if hasattr(model, "decision_function"):
            predictions = model.decision_function(data)
        else:
            predictions = model.predict_proba(data)[:,1]
            
        fpr, tpr, _ = roc_curve(labels, predictions, **roc_kws )
        auc_score = None 
        if score: 
            auc_score = roc_auc_score ( labels, predictions,)
            
        return fpr, tpr , auc_score
    
    if not is_iterable ( clfs): 
       clfs = is_iterable ( clfs, exclude_string =True , transform =True ) 
       
    # make default_colors 
    colors = make_plot_colors(clfs, colors = colors )
    # save the name of models 
    names = make_obj_consistent_if (
        names , [ get_estimator_name(m) for m in clfs ]) 

    # check whether the model is fitted 
    if kind=='2': 
        fig, ax = _make_axe_multiple ( 
            clfs, ncols = ncols , ax = ax, fig_size = fig_size 
                                  ) 
    else: 
        if ax is None: 
            fig, ax = plt.subplots (1, 1, figsize = fig_size )  
    
    for k, ( model, name)  in enumerate (zip (clfs, names )): 
        check_is_fitted(model )
        fpr, tpr, auc_score = plot_roc(model, X, y, score)

        if hasattr (ax, '__len__'): 
            if len(ax.shape)>1: 
                i, j  =  k // ncols , k % ncols 
                axe = ax [i, j]
            else: axe = ax[k]
        else: axe = ax 

        axe.plot(fpr, tpr, label=name + ('' if auc_score is None 
                                         else f"AUC={round(auc_score, 3) }") , 
                 color = colors[k]  )
        
        if kind=='2': 
            axe.plot([0, 1], [0, 1], 'k--') 
            axe.legend ()
            axe.set_xlabel ("False Positive Rate (FPR)")
            axe.set_ylabel ("True Positive Rate (FPR)")
        # else: 
        #     ax.plot(fpr, tpr, label=name, color = colors[k])
            
    if kind!='2': 
        ax.plot([0, 1], [0, 1], 'k--') # AUC =.5 
        ax.set_xlabel ("False Positive Rate (FPR)")
        ax.set_ylabel ("True Positive Rate (FPR)")
        ax.legend() 
        
    return ax 

def plot_tensors (
    z_or_edis_obj_list, /, 
    station:int|str= 'S00', 
    zplot:bool=False, 
    show_error_bars=False, 
    **kwargs
)-> object:
    #---------------------------------------
    # Get station index.
    get_station_group = re.search ('\d+', str(station), flags=re.IGNORECASE)
    if get_station_group is None: 
        raise TypeError ("Station should be or include a position number.")
    else : station = int(get_station_group.group()) 
    
    obj_type  = _assert_z_or_edi_objs (z_or_edis_obj_list)
    
    # Assert station index to be in the range of EDIlist 
    if station >=len( z_or_edis_obj_list): 
        raise ValueError (f"Expect {len(z_or_edis_obj_list)} stations."
                          f" Got {station}.")
    # Get z objets. 
    if obj_type =='EDI': 
        z_obj = z_or_edis_obj_list[station].Z
    else: 
        z_obj= z_or_edis_obj_list[station]
        
    #-------------------------------------------
    # Attributes 
    ms = kwargs.pop('ms', 1.5)
    ms_r = kwargs.pop('ms_r', 3)
    lw = kwargs.pop('lw', .5)
    lw_r = kwargs.pop('lw_r', 1.0)
    e_capthick = kwargs.pop('e_capthick', .5)
    e_capsize = kwargs.pop('e_capsize', 2)
    color_mode = kwargs.pop('color_mode', 'color')
    plot_style = kwargs.pop('plot_style', 1)
 
    # color mode
    if color_mode == 'color':
        # color for data
        cted = kwargs.pop('cted', (0, 0, 1))
        ctmd = kwargs.pop('ctmd', (1, 0, 0))
        mted = kwargs.pop('mted', 's')
        mtmd = kwargs.pop('mtmd', 'o')
        # color for occam2d model
        if plot_style == 3:
            # if plot_style is 3, set default color 
            #for model response to same as data
            ctem = kwargs.pop('ctem',cted)
            ctmm = kwargs.pop('ctmm',ctmd)
        else:
            ctem = kwargs.pop('ctem', (0, .6, .3))
            ctmm = kwargs.pop('ctmm', (.9, 0, .8))
        mtem = kwargs.pop('mtem', '+')
        mtmm = kwargs.pop('mtmm', '+')

    # black and white mode
    elif color_mode == 'bw':
        # color for data
        cted = kwargs.pop('cted', (0, 0, 0))
        ctmd = kwargs.pop('ctmd', (0, 0, 0))
        mted = kwargs.pop('mted', 's')
        mtmd = kwargs.pop('mtmd', 'o')
        # color for occam2d model
        ctem = kwargs.pop('ctem', (0.6, 0.6, 0.6))
        ctmm = kwargs.pop('ctmm', (0.6, 0.6, 0.6))
        mtem = kwargs.pop('mtem', '+')
        mtmm = kwargs.pop('mtmm', 'x')
    
    phase_limits_d = kwargs.pop('phase_limits_d', None)
    res_limits_d = kwargs.pop('res_limits_d', None)
    res_limits_od = kwargs.pop('res_limits_od', None)
    period_limits = kwargs.pop('period_limits', None)
    subplot_wspace = kwargs.pop('subplot_wspace', .3)
    subplot_hspace = kwargs.pop('subplot_hspace', .0)
    subplot_right = kwargs.pop('subplot_right', .98)
    subplot_left = kwargs.pop('subplot_left', .08)
    subplot_top = kwargs.pop('subplot_top', .85)
    subplot_bottom = kwargs.pop('subplot_bottom', .1)
        
    fig_size = kwargs.pop('fig_size', [6, 6])
    fig_dpi = kwargs.pop('dpi', 300)
    ylabel_pad = kwargs.pop('ylabel_pad', 1.25)
    # --> set default font size
    font_size = kwargs.pop('font_size', 6)
    plt.rcParams['font.size'] = font_size

    fontdict = {'size': font_size + 2, 
                'weight': 'bold'}
        
    legend_loc = 'upper center'
    legend_pos = (.5, 1.18)
    legend_marker_scale = 1
    legend_border_axes_pad = .01
    legend_label_spacing = 0.07
    legend_handle_text_pad = .2
    legend_border_pad = .15
    
    h_ratio = [1.5, 1, .5]
    
    gs = gridspec.GridSpec(2, 4,
        wspace=subplot_wspace,
        left=subplot_left,
        top=subplot_top,
        bottom=subplot_bottom,
        right=subplot_right,
        hspace=subplot_hspace,
        height_ratios=h_ratio[:2])
    
    #------------------------------------------
    # Plot data 
    fig = plt.figure(station, fig_size, dpi= fig_dpi)
    plt.clf()
    fig.suptitle("Station {}".format(str(station)), fontdict=fontdict)
            
    axrxx = fig.add_subplot(gs[0, 0], #yscale='log'
                            )
    axrxy = fig.add_subplot(gs[0, 1], sharex=axrxx, 
                            #yscale='log'
                            )
    axryx = fig.add_subplot(gs[0, 2], sharex=axrxx, sharey=axrxy, 
                           # yscale='log'
                            )
    axryy = fig.add_subplot(gs[0, 3], sharex=axrxx, sharey=axrxx, 
                           # yscale='log'
                            )

    axpxx = fig.add_subplot(gs[1, 0])
    axpxy = fig.add_subplot(gs[1, 1], sharex=axrxx)
    axpyx = fig.add_subplot(gs[1, 2], sharex=axrxx)
    axpyy = fig.add_subplot(gs[1, 3], sharex=axrxx)
            
    # convert to apparent resistivity and phase
    z_obj.compute_resistivity_phase()
    period = 1/z_obj._freq
    
    # find locations where points have been masked
    nzxx = np.nonzero(z_obj.z[:, 0, 0])[0]
    nzxy = np.nonzero(z_obj.z[:, 0, 1])[0]
    nzyx = np.nonzero(z_obj.z[:, 1, 0])[0]
    nzyy = np.nonzero(z_obj.z[:, 1, 1])[0]

    # convert to apparent resistivity and phase
    if zplot:
        scaling = np.zeros_like(z_obj.z)
        for ii in range(2):
            for jj in range(2):
                scaling[:, ii, jj] = 1. / np.sqrt(z_obj.freq)
        plot_res = abs(z_obj.z.real * scaling)
        plot_res_err = abs(z_obj.z_err * scaling)
        plot_phase = abs(z_obj.z.imag * scaling)
        plot_phase_err = abs(z_obj.z_err * scaling)
        h_ratio = [1.5, 1, .5]

    elif not zplot:
        plot_res = z_obj.resistivity
        plot_res_err = z_obj.resistivity_err
        plot_phase = z_obj.phase
        plot_phase_err = z_obj.phase_err
        h_ratio = [1.5, 1, .5]
    
        try:
            res_limits_d = (10 ** (np.floor(np.log10(
                min([plot_res[nzxx, 0, 0].min(),
                plot_res[nzyy, 1, 1].min()])))),
                                 10 ** (np.ceil(np.log10(
                                     max([plot_res[nzxx, 0, 0].max(),
                                    plot_res[nzyy, 1, 1].max()])))))
        except ValueError:
            res_limits_d = None
        try:
            res_limits_od = (10 ** (np.floor(np.log10(
                min([plot_res[nzxy, 0, 1].min(),
                  plot_res[nzyx, 1, 0].min()])))),
                                  10 ** (np.ceil(np.log10(
                                      max([plot_res[nzxy, 0, 1].max(),
                                      plot_res[nzyx, 1, 0].max()])))))
        except ValueError:
            res_limits_od = None

    # --> make key word dictionaries for plotting
    kw_xx = {'color': cted,
             'marker': mted,
             'ms': ms,
             'ls': ':',
             'lw': lw,
             'e_capsize': e_capsize,
             'e_capthick': e_capthick}

    kw_yy = {'color': ctmd,
             'marker': mtmd,
             'ms': ms,
             'ls': ':',
             'lw': lw,
             'e_capsize': e_capsize,
             'e_capthick': e_capthick}

    # ---------plot the apparent resistivity-----------------------------------
            # plot each component in its own subplot
            # plot data response
    erxx = plot_errorbar(
        axrxx,
        period[nzxx],
        plot_res[nzxx, 0, 0],
        plot_res_err[nzxx, 0, 0],
        show_error_bars=show_error_bars, 
        **kw_xx, 
        )
    erxy = plot_errorbar(
        axrxy,
        period[nzxy],
        plot_res[nzxy, 0, 1],
        plot_res_err[nzxy, 0, 1],
        show_error_bars=show_error_bars, 
        **kw_xx)
    eryx = plot_errorbar(
        axryx,
        period[nzyx],
        plot_res[nzyx, 1, 0],
        plot_res_err[nzyx, 1, 0],
        show_error_bars=show_error_bars, 
        **kw_yy)
    eryy = plot_errorbar(
        axryy,
        period[nzyy],
        plot_res[nzyy, 1, 1],
        plot_res_err[nzyy, 1, 1],
        show_error_bars=show_error_bars, 
        **kw_yy)
    # plot phase

    plot_errorbar(
        axpxx,
        period[nzxx],
        plot_phase[nzxx, 0, 0],
        plot_phase_err[nzxx, 0, 0],
        show_error_bars=show_error_bars, 
        **kw_xx)
    plot_errorbar(
        axpxy,
        period[nzxy],
        plot_phase[nzxy, 0, 1],
        plot_phase_err[nzxy, 0, 1],
        show_error_bars=show_error_bars, 
        **kw_xx)
    plot_errorbar(
        axpyx,
        period[nzyx],
        plot_phase[nzyx, 1, 0],
        plot_phase_err[nzyx, 1, 0],
        show_error_bars=show_error_bars, 
        **kw_yy)
    plot_errorbar(
        axpyy,
        period[nzyy],
        plot_phase[nzyy, 1, 1],
        plot_phase_err[nzyy, 1, 1],
        show_error_bars=show_error_bars, 
        **kw_yy)

    # get error bar list for editing later
    #_err_list = 
    try:
        [[erxx[1][0], erxx[1][1], erxx[2][0]],
        [erxy[1][0], erxy[1][1], erxy[2][0]],
        [eryx[1][0], eryx[1][1], eryx[2][0]],
        [eryy[1][0], eryy[1][1], eryy[2][0]]]
        line_list = [[erxx[0]], [erxy[0]], [eryx[0]], [eryy[0]]]
    except IndexError:
        print('Found no Z components for {0}'.format(station))
        line_list = [[None], [None],
                     [None], [None]]

     # ------------------------------------------
    # # make things look nice
    # # set titles of the Z components
    ax_list = [axrxx, axrxy, axryx, axryy,
               axpxx, axpxy, axpyx, axpyy]
    label_list = [['$Z_{xx}$'], ['$Z_{xy}$'],
                  ['$Z_{yx}$'], ['$Z_{yy}$']]
    # for ax, label in zip(ax_list[0:4], label_list):
    #     ax.set_title(label[0], fontdict={'size': font_size + 2,
    #                                       'weight': 'bold'})
    #     # set axis properties
    for aa, ax in enumerate(ax_list):
        ax.tick_params(axis='y', pad=ylabel_pad)
        # if self.plot_tipper==False:
        if aa < 4:
            if zplot == True:
                ax.set_yscale('log',
                              #nonposy='clip'
                              )
        else:
            ax.set_xlabel('Period (s)', 
                          fontdict=fontdict
                          )

        if aa < 8:
            if zplot == True:
                ax.set_yscale('log', 
                              # nonposy='clip'
                              )
        else:
            ax.set_xlabel('Period (s)', fontdict=fontdict)

        if aa < 4 and zplot is False:
            ylabels = ax.get_yticks().tolist()
            ylabels[0] = ''
            ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ylabels))
            ax.set_yticklabels([ str(f) for f in ylabels])
            ax.set_yscale('log', 
                          #nonposy='clip'
                          )
            try: 
                # skip setting the axis limits
                if aa == 0 or aa == 3:
                    ax.set_ylim(res_limits_d)
                elif aa == 1 or aa == 2:
                    ax.set_ylim(res_limits_od)
            except: pass 

        if aa > 3 and aa < 8 and zplot is False:
            #ax.yaxis.set_major_locator(MultipleLocator(10.0))
            if phase_limits_d is not None:
                ax.set_ylim(phase_limits_d)
        # set axes labels
        if aa == 0:
            if zplot == False:
                ax.set_ylabel('App. Res. ($\mathbf{\Omega \cdot m}$)',
                              fontdict=fontdict)
            elif zplot == True:
                ax.set_ylabel('Re[Z (mV/km nT)]',
                              fontdict=fontdict)
        elif aa == 4:
            if zplot == False:
                ax.set_ylabel('Phase (deg)',
                              fontdict=fontdict)
            elif zplot == True:
                ax.set_ylabel('Im[Z (mV/km nT)]',
                              fontdict=fontdict)
        elif aa == 8:
            ax.set_ylabel('Tipper',
                          fontdict=fontdict)
        if aa > 7:
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(.1))
  
        ax.set_xscale('log', 
                     # nonposx='clip'
                      )
        # set period limits
        if period_limits is None:
            period_limits = (10 ** (np.floor(np.log10(period[0]))) * 1.01,
                                  10 ** (np.ceil(np.log10(period[-1]))) * .99)
        ax.set_xlim(xmin=period_limits[0],
                    xmax=period_limits[1])
        ax.grid(True, alpha=.25)

        ylabels = ax.get_yticks().tolist()
        if aa < 8:
            ylabels[-1] = ''
            ylabels[0] = ''

        if aa < len(ax_list)//2: 
            plt.setp(ax.get_xticklabels(), visible=False)
   
        # --> make key word dictionaries for plotting
        kw_xx = {'color': ctem,
                  'marker': mtem,
                  'ms': ms_r,
                  'ls': ':',
                  'lw': lw_r,
                  'e_capsize': e_capsize,
                  'e_capthick': e_capthick}

        kw_yy = {'color': ctmm,
                  'marker': mtmm,
                  'ms': ms_r,
                  'ls': ':',
                  'lw': lw_r,
                  'e_capsize': e_capsize,
                  'e_capthick':e_capthick}

        
        legend_ax_list = ax_list[0:4]
        for aa, ax in enumerate(legend_ax_list):
            ax.legend(line_list[aa],
                      label_list[aa],
                      loc=legend_loc,
                      bbox_to_anchor=legend_pos,
                      markerscale=legend_marker_scale,
                      borderaxespad=legend_border_axes_pad,
                      labelspacing=legend_label_spacing,
                      handletextpad=legend_handle_text_pad,
                      borderpad=legend_border_pad,
                      framealpha=1,
                      prop={'size': max([font_size, 5])})
  
    plt.show()
    
    return z_obj 
  
plot_tensors.__doc__="""\

Plot resistivity and phase tensors or the real and imaginary impedance.

Plots the real and imaginary impedance and induction vector if present.

Parameters 
------------
z_or_edis_obj_list: list of :class:`watex.edi.Edi` or \
        :class:`watex.externals.z.Z` 
        A collection of EDI- or Impedances tensors objects.
        
station: int, default='S00'
   Station to visualize the resistivity, phases or impendances tensors. 
   Default is the first station. Note that station counting start from index 
   equal to ``0``.
    
zplot: bool, default=False, 
   Visualize the impedance tensors values `Z`. 
   

kwargs: Additional keywords arguments 


To get further details about the way to control the plot, refer to the 
following attributes. 

======================== ==================================================
Attributes               Description
======================== ==================================================
color_mode               [ 'color' | 'bw' ] color or black and white plots
cted                     color for data Z_XX and Z_XY mode
ctem                     color for model Z_XX and Z_XY mode
ctmd                     color for data Z_YX and Z_YY mode
ctmm                     color for model Z_YX and Z_YY mode
data_fn                  full path to data file
data_object              WSResponse instance
e_capsize                cap size of error bars in points (*default* is .5)
e_capthick               cap thickness of error bars in points (*default*
                         is 1)
fig_dpi                  resolution of figure in dots-per-inch (300)
fig_list                 list of matplotlib.figure instances for plots
fig_size                 size of figure in inches (*default* is [6, 6])
font_size                size of font for tick labels, axes labels are
                         font_size+2 (*default* is 7)
legend_border_axes_pad   padding between legend box and axes
legend_border_pad        padding between border of legend and symbols
legend_handle_text_pad   padding between text labels and symbols of legend
legend_label_spacing     padding between labels
legend_loc               location of legend
legend_marker_scale      scale of symbols in legend
lw                       line width data curves (*default* is .5)
ms                       size of markers (*default* is 1.5)
lw_r                     line width response curves (*default* is .5)
ms_r                     size of markers response curves (*default* is 1.5)
mted                     marker for data Z_XX and Z_XY mode
mtem                     marker for model Z_XX and Z_XY mode
mtmd                     marker for data Z_YX and Z_YY mode
mtmm                     marker for model Z_YX and Z_YY mode
phase_limits             limits of phase
plot_component           [ 2 | 4 ] 2 for TE and TM or 4 for all components
plot_style               [ 1 | 2 ] 1 to plot each mode in a seperate
                         subplot and 2 to plot xx, xy and yx, yy in same
                         plots
plot_type                [ '1' | list of station name ] '1' to plot all
                         stations in data file or input a list of station
                         names to plot if station_fn is input, otherwise
                         input a list of integers associated with the
                         index with in the data file, ie 2 for 2nd station
plot_z                   [ True | False ] *default* is True to plot
                         impedance, False for plotting resistivity and
                         phase
plot_yn                  [ 'n' | 'y' ] to plot on instantiation
res_limits               limits of resistivity in linear scale
resp_fn                  full path to response file
resp_object              WSResponse object for resp_fn, or list of
                         WSResponse objects if resp_fn is a list of
                         response files
station_fn               full path to station file written by WSStation
subplot_bottom           space between axes and bottom of figure
subplot_hspace           space between subplots in vertical direction
subplot_left             space between axes and left of figure
subplot_right            space between axes and right of figure
subplot_top              space between axes and top of figure
subplot_wspace           space between subplots in horizontal direction
======================== ==================================================   
    
Examples 
---------
>>> import watex as wx  
>>> edi_data = wx.fetch_data ('edis', samples= 17 , return_data =True ) 
>>> wx.utils.plotutils.plot_tensors ( edi_data, station =4 )
""" 
    
def plot_sounding (
    ves, /, 
    style = 'bmh', 
    fig_size = (10, 4), 
    cz_plot_kws= None,
    marker_kws= None, 
    savefig =None, 
    ax=None, 
    fig=None,
    **plot_kws ): 
    """ Visualize the vertical electrical sounding. 
    
    Function plots the sounding curve from AB/2 sounding points. 
    
    Parameters 
    -----------
    ves: array_like 1d
        The vertical electrical resistivity sounding array. 
        If dataframe is passed,`resistivity` column must be included. 
        
    style: str, default='bmh'
        Matplotlib plottings style.
        
    fig_size: tuple, default= (10, 4) 
        Matplotlib figure size. 
        
    marker_kws: dict, default = {'marker':'o', 'c':'#9EB3DD' }
        The dictionnary to customize marker in the plot 
        
    cz_plot_kws: dict, default = {'ls':'-','c':'#0A4CEE', 'lw'L2 }
        The dictionnary to customize the conductize zone in the plot.
        
    savefig: str, optional 
        Save figure name. The default resolution dot-per-inch is ``300``. 
      
    ax: Matplotlib.pyplot.Axes, optional 
       Axe to collect the figure. 
       
    fig: Matplotlib.pyplot.figure, optional 
        Supply fig to save automatically the plot, otherwise, keep it 
        to ``None``.
        
    plot_kws: dict, 
        Additional keyword arguments passed to :func:`matplotlib.pyplot.plot` 
        function 
        
    Return
    --------
    ax: Matplotlib.pyplot.Axis 
        Return axis  
        
    See also
    ---------
    watex.utils.exmath.plotOhmicArea: 
        plot the Ohmic Area including the computed fracture zone. 
        
    Examples 
    ----------
    >>> from watex.datasets import make_ves
    >>> from watex.utils.plotutils import plot_sounding
    >>> import matplotlib.pyplot as plt 
    >>> fig, ax = plt.subplots ( 2, 1, figsize = (10, 10))
    >>> d= make_ves (samples =56, seed = 42)
    >>> plot_sounding  (d.resistivity, ax =ax [0], color ='k', marker ='D', )
    >>> ax[0].set_title ("VES: samples=56, seed =42")
    >>> # read the frame and get the resistivity values 
    >>> ax[1] = plot_sounding(make_ves (order ='+', max_rho =1e4, seed =65 , 
                                        as_frame=True,iorder =5), 
                              ax= ax[1], ls=':', marker ='o', color ='blue')
    >>> ax[1].set_title ("VES:samples=41, order='+', iorder=5,"
                         " max_rho=10000.$\Omega.m$, seed=65")
    """
    plt.style.use (style )
    
    if hasattr ( ves , 'columns') and hasattr ( ves , '__array__'): 
        if 'resistivity' not in  ves.columns : 
            raise TypeError ("Missing resistivity column in the data.")
        
        ves = ves.resistivity 
    
    ves = check_y (ves , input_name ="sample of VES data")
    
    if ax is None: 
        fig, ax = plt.subplots(1,1, figsize =fig_size)
        
    leg =[]
    
    zl, = ax.semilogy(np.arange(len(ves)), ves, 
                  label ='Vertical Electrical Resistivity', 
                  **plot_kws 
                  )
    marker_kws = marker_kws or dict (marker ='o', c='#9EB3DD' )
    ax.scatter (np.arange(len(ves)), ves, **marker_kws )
    
    leg.append(zl)    
        
    ax.set_xticks(range(len(ves)))
    
    _get_xticks_formatage (ax, ax.get_xticks() , auto =True, 
                           rotation=0)
        # for label in ax.xaxis.get_ticklabels()[::7]:
        #     label.set_visible(False)
         
    ax.set_xlabel('AB/2(m)')
    ax.set_ylabel('App.resistivity ($\Omega.m$)')
    ax.legend( handles = leg, loc ='best')
    ax.set_xlim ([-1, len(ves)])

    if savefig is not  None: savefigure (fig, savefig, dpi = 300)
        
    plt.close () if savefig is not None else plt.show() 
    
    return ax 

def plot_l_curve(
    rms, 
    roughness, 
    tau=None, 
    hansen_point=None, 
    rms_target=None,
    view_tline=False,
    hpoint_kws=dict(), 
    fig_size = (10, 4),
    ax=None,
    fig=None, 
    style = 'classic', 
    savefig=None, 
    **plot_kws
    ):
    """
    Plot the Hansen L-curve.
    
    The L-curve criteria is used to determine the suitable model 
    after runing multiple inversions with different :math:`\tau` values. 
    The function plots RMS vs. Roughness with an option to highlight a 
    specific point named Hansen point [1]_.
    
    The :math:`\tau` represents the measure of compromise between data fit and 
    model smoothness. To find out an appropriates-value, the inversion was 
    carried out with differents-values. The RMS error obtained from each 
    inversion is plotted against model roughnes
    
    Plots RMS vs. Roughness with an option to highlight the Hansen point.
    
    Parameters 
    ------------
    
    rms: ArrayLike, list, 
       Corresponding list pr Arraylike of RMS values.
       
    roughness: Arraylike, list, 
       List or ArratLike of roughness values. 
       
    tau: Arraylike or list, optional 
       List of tau values to visualize as text mark in the plot. 

    hansen_point: A tuple (roughness_value, RMS_value) , optional 
       The Hansen point to visualize in the plot. It can be determine 
       automatically if ``highlight_point='auto'``.
       
    rms_target: float, optional 
      The root-mean-squared target. If set, and `view_tline` is ``False``, 
      the target value should be axis limit. 
      
     view_tline: bool, default=False 
       Display the target line should be  displayed.
       
    hpoint_kws: dict, optional 
      Keyword argument to highlight the hansen point in the figure. 
     
    ax: Matplotlib.pyplot.Axes, optional 
       Axe to collect the figure. Could be used to support other axes. 
       
    fig: Matplotlib.pyplot.figure, optional 
        Supply fig to save automatically the plot, otherwise, keep it 
        to ``None``.

    savefig: str, optional 
        Save figure name. The default resolution dot-per-inch is ``300``. 
         
    Return 
    ------
    ax: Matplotlib.pyplot.Axis 
        Return axis  
        
    References
    -----------
    [1] Hansen, P. C., & O'Leary, D. P. (1993). The use of the L-Curve in
        the regularization of discrete ill-posed problems. SIAM Journal
        on Scientific Computing, 14(6), 1487–1503. https://doi.org/10.1137/0914086.
         
    Examples
    ---------
    >>> from watex.utils.plotutils import plot_l_curve
    >>> # Test the function with the provided data points and 
    >>> # highlighting point (50, 3.12)
    >>> roughness_data = [0, 50, 100, 150, 200, 250, 300, 350]
    >>> RMS_data = [3.16, 3.12, 3.1, 3.08, 3.06, 3.04, 3.02, 3]
    >>> highlight_data = (50, 3.12)
    >>> plot_l_curve(roughness_data, RMS_data, highlight_data)
    """
    
    rms= np.array (
        is_iterable(rms, exclude_string= True, transform =True ), 
                   dtype =float) 
    roughness = np.array( 
        is_iterable(roughness , exclude_string= True, transform =True 
                            ), dtype = float) 
    
    # Create the plot
    plt.style.use (style )
    
    if ax is None: 
        fig, ax = plt.subplots(1,1, figsize =fig_size)
    
    # manage the plot keyword argument and remove 
    # the default is given.
    plot_kws = _manage_plot_kws (plot_kws, dict(
        marker='o',linestyle='-', color='black')
                    )
    ax.plot(roughness, rms, **plot_kws
             )
    # Highlight the specific hansen point if "auto" 
    # option is provided.
    if str(hansen_point).lower().strip() =="auto": 
        hansen_point = _get_hansen_point(roughness, rms)
        
    if hansen_point is not None:
        if len(hansen_point)!=2: 
            raise ValueError("Hansen knee point needs a tuple of (x, y)."
                             f" Got {hansen_point}")
        hpoint_kws = _manage_plot_kws(hpoint_kws, 
                                         dict(marker='o', color='red'))
        ax.plot(hansen_point[0], hansen_point[1], **hpoint_kws)
        ax.annotate(str(hansen_point[0]), 
                     hansen_point, textcoords="offset points",
                     xytext=(0,10), ha='center'
                     )
    if tau is not None: 
        tau = is_iterable(tau, exclude_string= True, transform =True )
        rough_rms = np.hstack ((roughness, rms))
        for tvalues, text in zip ( rough_rms, tau): 
            if ( 
                    tvalues[0]==hansen_point[0] 
                    and tvalues[1]==hansen_point[1]
                    ): 
                # hansen point is found then skip it
                continue 
            ax.annotate(str(text), tvalues, textcoords="offset points",
                         xytext=(0,10), ha='center'
                         )
    if rms_target: 
        rms_target = float( _assert_all_types(
            rms_target, float, int, objname='RMS target')) 
        
    if view_tline: 
        if rms_target is None: 
            warnings.warn("Missing RMS target value. Could not display"
                          " the target line.")
        else:
            ax.axhline(y=float(rms_target), color='k', linestyle=':')

    if rms_target: 
        # get the extent value from the min 
        extent_value = abs ( rms_target - min(rms )) 
        ax.set_ylim ( [rms_target - extent_value  if view_tline else 0  ,
                       max(rms)+ extent_value]) # .3 extension limit

    # Setting the labels and title
    ax.set_xlabel('Roughness')
    ax.set_ylabel('RMS')
    ax.set_title('RMS vs. Roughness')

    # savefig 
    if savefig is not  None: savefigure (fig, savefig, dpi = 300)
    # Show the plot    
    plt.close () if savefig is not None else plt.show() 
    
    return ax 

def _get_hansen_point ( roughness, RMS): 
    """ Get the Hansen point automatically.
    
    An isolated part of :func:`~plot_l_curve`. 
    
    The L-curve criteria proposed by Hansen and O'Leary (1993)[1]_ and 
    Hansen (1998) [2]_, which suggests that the s value at the knee of 
    the curve is most appropriate, have been adopted.

    References
    -----------
    [1] Hansen, P. C., & O'Leary, D. P. (1993). The use of the L-Curve in
        the regularization of discrete ill-posed problems. SIAM Journal
        on Scientific Computing, 14(6), 1487–1503. https://doi.org/10.1137/0914086.
        
    [2] Hansen, P. C. (1998). Rank deficient and discrete Ill: Posed problems, 
        numerical aspects of linear inversion (p. 247). Philadelphia: SIAM
    """
    # Calculate the curvature of the plot
    # Using a simple method to estimate the 'corner' of the L-curve
    curvature = []
    for i in range(1, len(roughness) - 1):
        # Triangle area method to calculate curvature
        x1, y1 = roughness[i-1], RMS[i-1]
        x2, y2 = roughness[i], RMS[i]
        x3, y3 = roughness[i+1], RMS[i+1]

        # Lengths of the sides of the triangle
        a = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        b = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
        c = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)

        # Semiperimeter of the triangle
        s = (a + b + c) / 2

        # Area of the triangle
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        # Curvature is 4 * area divided by product of the sides
        # (Heron's formula for the area of a triangle)
        if a * b * c == 0:
            k = 0
        else:
            k = 4 * area / (a * b * c)
        curvature.append(k)

    # Find the index of the point with the maximum curvature
    # +1 due to curvature array being shorter
    max_curvature_index = np.argmax(curvature) + 1  
    return roughness[max_curvature_index], RMS[max_curvature_index]

def _manage_plot_kws ( kws, dkws = dict () ): 
    """ Check whether the default values are in plot_kws then pop it"""
    
    kws = dkws or kws 
    for key in dkws.keys(): 
        # if key not in then add it. 
        if key not in kws.keys(): 
            kws[key] = dkws.get(key)
            
    return kws 

def plot_skew ( 
     edis_list,
     /, 
     method ='Bahr',
     sensitivity='skew',
     mode ="frequency", 
     show_skewness=...,
     view='1d',
     interpolate=True, 
     threshold_line =None,  
     tolog10 =True, 
     interp_method ='cubic', 
     fill_value ='auto',
     get_sites_by=None, 
     top_label ='Stations', 
     cb_label ="Sensitivity (S)",
     spacing=80, 
     fig=None , 
     fig_size = (7, 5 ), 
     dpi = 300 , 
     font_size =5.,
     cmap='jet_r',
     plot_style =None,
     rotate_xlabel=0.,
     plot_contours=..., 
     style=None,
     savefig=None, 
     ax=None, 
     **plot_kws 
    ): 
    
    if str(view).lower()=='2d': 
        return plot_skew2d ( 
            edis_list, 
            method =method, 
            sensitivity = sensitivity, 
            mode=mode, 
            interpolate=interpolate, 
            show_skewness=show_skewness,
            tolog10=tolog10, 
            fill_value=fill_value, 
            interp_method =interp_method, 
            get_sites_by=get_sites_by, 
            top_label=top_label, 
            cb_label =cb_label, 
            spacing =spacing, 
            fig=fig, 
            fig_size =fig_size, 
            dpi=dpi, 
            font_size=font_size, 
            cmap=cmap, plot_style=plot_style, 
            rotate_xlabel=rotate_xlabel, 
            plot_contours=plot_contours,
            savefig=savefig, 
            ax=ax 
            )
        
    return plot_skew1d (
        edis_list,
        method =method, 
        sensitivity=sensitivity, 
        mode=mode, 
        threshold_line=threshold_line, 
        show_skewness=show_skewness, 
        fig_size=fig_size, 
        savefig=savefig, 
        style =style, 
        ax=ax ,
        dpi=dpi, 
        **plot_kws 
        )
    
plot_skew.__doc__="""\
    
Visualize the phase sensitive skew in one or two dimensional. 

Phase Sensitivity Skew (:math:`\eta`) is a dimensionality tool that 
represents a measure of the skew of the  phases of the impedance 
tensor. The parameter is thus unaffected by the distortion 
effect, unlike the Swift-skew and ellipticity dimensionality 
tools [1]_. 

Values of :math:`\eta` > 0.3 are considered to represent 3D data. 
Phase-sensitive skews less than 0.1 indicate 1D, 2D or distorted 
2D (3-D /2-D) cases. Values of :math:`\eta` between 0.1 and 0.3 indicates 
modified 3D/2D structures [2]_ according to `Bahr' methods. However,
values :math:`\eta >=0.2` using the `Swift` methods, the smaller the value 
:math:`\eta` ( close to :math:`0.`), the closer the structure to 2D 
structure and vice versa.However, it is generally considered that 
an electrical structure of :math:`\eta < 0.4` can be treated as a 2D 
medium. Here as the ``threshold_line`` for :meth:`\eta` using the 
Swift method should be set as `0.4`. 


Parameters
-----------
edis_list: str, :class:`watex.edi.Edi` 
    Full path to edifiles.
    
    .. versionchanged:: 0.3.1 
       The parameter `edi_obj` is replaced by `edis_list` which  
       indicate  a collection of :term:`EDI`files. 
    
method: str, default='Bahr': 
    Kind of correction. Can be ``swift`` for the remove distorsion proposed 
    by Swift in 1967 [3]_. The value close to 0. assume the 1D and 2D 
    structures, and 3D otherwise. Conversly to ``bahr`` for the remove 
    distorsion proposed  by Bahr in 1991 [2]_. The latter threshold is set 
    to 0.3. Above this value the structures is 3D. 
  
sensitivity: str, default='skew'
   phase sensistive visualization. Can be rotational invariant 
   ``invariant``. Note that setting to ``mu`` or ``invariant`` does 
   not change any interpretation since the distortion of Z are all 
   rotational invariant whatever we're using the ``Bahr`` or ``swift``
   method. 
   
   .. versionchanged:: 0.3.1 
      The parameter `view` is deprecated and replaced with `sensitivity`. 
      
mode:str, optional 
   X-axis coordinates for visualisation. plot either ``'frequency'`` or
   ``'periods'``.  The default is ``'frequency'`` 

show_skewness: bool,default=False 
   Display the average skewness value. 
   
   .. versionadded:: 0.3.1 
      `show_skewness` display the average value of the whole Z tensor 
      at each frequency. 
     
view: str, ['1D', '2D'], default ='1D'
   Type of skewness visualisation. 
   
interpolate: bool, default=True
   Interpolate the data if NaN is found. 

tolog10: bool, default=True 
   Compute the the logarithm base 10 of the frequency array. If the 
    frequency data is passed as log10 values, it should be turned to 
    ``False``. 
  
interp_method: bool,default='cubic' 
   Data interpolation method. It could be ['nearest'|'linear'|'cubic']. 
    
fill_value: float, str, default='auto' 
   Fill the interpolated grid at the egdes or surrounding NaN with 
   a filled value. The ``auto`` uses the forward and backward 
   fill strategy. 
   
get_sites_by: str, optional
  Fetch the sites and place names on the map. It should be 
  [``'dataid'``|``'name'``]. The former  uses the names collected in 
  :term:`EDI` data id whereas the latter generates new names from the 
  sites id and the survey name. In that case, it expects the survey name  
  to be specified. By default, it merely use the sites id. 
  
threshold_line: float, optional
   Visualize th threshold line. Can be ['bahr', 'swift', 'both']:
       
   - Note that when method is set to ``swift``, the value close to close 
     to :math:`0.` assume the 1D and 2D structures,  and 3D otherwise. 
   - when method is set to ``Bahr``, :math:`\mu > 0.3``  is 3D structures, 
     between :math:`[0.1 - 0.3]` assumes modified 3D/2D structures whereas 
     :math:`<0.1` 1D, 2D or distorted 2D. 
 
top_label: str, default='Stations' 
   Label  used to name the xticks in upper. 
   
cb_label: str, default='Sensitivity (S)'
   The colorbar label.
   
spacing: float, default=80. 
    The step in meters between two stations/sites. If given, 
    it creates an array of positions. 

fig_size: tuple, default= (6, 2) 
    Matplotlib figure size. 
  
dpi: int, default=300 
   Image resolution in dot-per-inch 
   
cmap: str, default='jet_r' 
  Matplotlib colormap 

plot_style: str, optional
   The kind of plot. It could be ['pcolormesh'|'imshow']. The default is 
   ``pcolormesh``. 
   
rotate_xlabel: float, Optional 
  The degree angle to rotate the station/site label accordingly. 
  
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

savefig: str, optional 
     Save figure name. The default resolution dot-per-inch is ``300``. 
     
plot_kws:  dict, 
   Matplotlib Axes scatterplot additional keywords arguments.

ax: Matplotlib.pyplot.Axes, optional 
   Axe to collect the figure. Could be used to support other axes. 
     
Return
--------
ax: Matplotlib.pyplot.Axis 
    Return axis  
    
See Also 
---------
watex.methods.em.Processing.skew: 
    Skew equation formulations. 
watex.view.TPlot.plotSkew: 
    Give a consistent plot where user can customize the plot using the 
    plot parameter of :class:`watex.property.BasePlot` class.
    
References 
-----------
.. [1] Bahr, K. (1988) Interpretation of the magnetotelluric impedance 
       tensor: regional induction 395 and local telluric distortion. J. 
       Geophys. Res., 62, 119–127.
.. [2] Bahr, K. (1991) Geological noise in magnetotelluric data: 
       a classification of distortion types. 397 Phys. Earth Planet. 
       Inter., 66, 24–38.
.. [3] Bahr, K., 1991. Geological noise in magnetotelluric data: a 
       classification of distortion types. Physics of the Earth and 
       Planetary Interiors 66 (1–2), 24–38.  
       
Example
---------
>>> import watex as wx 
>>> from watex.utils.plotutils import plot_skew
>>> edi_sk = wx.fetch_data ("edis", return_data =True , samples = 20 )
>>> # Get 1d visualization with Swift skewness method
>>> plot_skew (edi_sk, threshold_line= True, method ='Swift', 
               fig_size =( 12, 4))  
>>> # plot the 2D with Bahr method with period in y-axis 
>>> plot_skew (edi_sk, view='2d', show_skewness=True, interpolate=True, 
                 get_sites_by='name', mode='periods', fig_size =(6, 2))  
 
"""


    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
    
    
    
