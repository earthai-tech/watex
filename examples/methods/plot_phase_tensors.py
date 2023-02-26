"""
===================================
Plot Pseudosection Phase tensors 
===================================

Give an ellipse representation of 
phase tensors in pseudo-section format.
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

# %%
# Plot the phase tensors with differents parameters for ellipses
# such as ``[ 'phimin' | 'phimax' | 'skew' |'skew_seg' | 'phidet' |'ellipticity' ]``
# For demonstration, we will use the data collected from Huayuan (:func:`watex.datasets.load_huayuan`)
# area with 27 samples. Here, we will give a three representations of phase tensors ellipsis. 
# 
# Import required modules as: 
import watex as wx 
# fetch the huayuan data 
edi_samples = wx.fetch_data ('huayuan', samples =27 , return_data = True ) 
# can also be 
# >>> from watex.datasets import load_huayuan 
# >>> edi_samples = load_huayuan ( samples =27, return_data =True ) 

# %%
# *  Ellipse of ``'phimin'``  
tplot= wx.TPlot (fig_size =( 5, 2 )).fit(edi_samples )
tplot.plot_phase_tensors ()
# by default the color limit for 'phmin' is [0, 90] 
# %%
# * Ellipse of ``'skew'`` visualization 
# we can setup the ellipses dictionnary and passed to the 
# parameter `ellipse_dict`: 
ellip_dic = {'ellipse_colorby':'skew',
            'ellipse_range':[-1 , 1], # Color limits for skew 
            'ellip_size': 2, 
            'ellipse_cmap':'jet_r' # or color defined in `cmap` parameter such as `mt_bl2wh2rd`
            }
#>>> tplot.plot_phase_tensor (ellipse_dict=ellip_dic )
# or simply turn on `tensor` parameter to ``skew`` like: 
tplot.plot_phase_tensors (tensor='skew' )

# %% 
# * Determinant of phase tensor ``'phidet'``
tplot.plot_phase_tensors (tensor='phidet' )

