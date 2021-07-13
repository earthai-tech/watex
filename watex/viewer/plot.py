# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Wed Jul  7 22:23:02 2021 hz
# This module is part of the WATex viewer package, which is released under a
# MIT- licence.

import os ,re, warnings
import functools 
import numpy as np 
import matplotlib as mpl 
import  matplotlib.pyplot  as plt

import matplotlib.cm as cm 
import matplotlib.colorbar as mplcb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, NullLocator
import matplotlib.gridspec as gspec


from watex.utils._watexlog import watexlog
_logger=watexlog.get_watex_logger(__name__)




class Quick_plot : 
    """
    Special class deals with analusis modules. To quick plot diagrams, 
    histograms and bar plots.
    
    """