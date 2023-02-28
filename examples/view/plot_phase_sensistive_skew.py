"""
=================================================
Plot Skew
=================================================

Phase sensitive skew visualization

"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

# %%
# 'Skew' is also knwown as the conventional asymmetry parameter 
# based on the Z magnitude. 
#
# Mosly, the :term:`EM` signal is influenced by several factors such 
# as the dimensionality of the propagation medium and the physical 
# anomalies, which can distort theEM field both locally and regionally. 
# The distortion of Z was determined from the quantification of its 
# asymmetry and the deviation from the conditions that define its 
# dimensionality. The parameters used for this purpose are all rotational 
# invariant because the Z components involved in its definition are
# independent of the orientation system used. The conventional asymmetry
# parameter based on the Z magnitude is the skew defined by Swift (1967)
# [1]_ and Bahr (1991) [2]_.
#
# - ``swift`` for the remove distorsion proposed by Swift in 1967. 
#   if the value close to 0., it assumes the 1D and 2D structures, and 3D 
#   otherwise. 
# - ``bahr`` for the remove distorsion proposed  by Bahr in 1991. 
#   The threshold is set to 0.3 and above this value the 
#   structures is 3D. However Values of :math:`\mu` > 0.3 are considered to 
#   represent 3D data. 
#   Phase-sensitive skews less than 0.1 indicate 1D, 2D or distorted 
#   2D (3-D /2-D) cases. Values of :math:`mu` between 0.1 and 0.3 indicates 
#   modified 3D/2D structures. 
# Here is an example of implementation using the :class:`watex.view.TPlot` class 
# of module :mod:`watex.view`. 
# we start by importing ``watex`` as: 
import watex 

# * `Swift method` 
test_data = watex.fetch_data ('edis', samples =37, return_data =True )
tplot = watex.TPlot(fig_size =(10,  4), marker ='x').fit(test_data)
tplot.plt_style='classic'
tplot.plotSkew(method ='swift', threshold_line=True)

# %%
# For any specific reasons, user can check the influence of the existing 
# outliers in the data. This is possible by turning off  the parameter 
# ``suppress_outliers`` to ``False`` like 
tplot.plotSkew(method ='swift', threshold_line=True, suppress_outliers=False )

# %% 
# * `Bahr method (default)`
# 
tplot.plotSkew(threshold_line=True, suppress_outliers=False )

#%%
# .. topic:: References 
#
#    .. [1] Swift, C., 1967. A magnetotelluric investigation of an 
#           electrical conductivity  anomaly in the southwestern United 
#           States. Ph.D. Thesis, MIT Press. Cambridge.
#    .. [2] Bahr, K., 1991. Geological noise in magnetotelluric data: a 
#           classification of distortion types. Physics of the Earth and 
#           Planetary Interiors 66 (1–2), 24–38.