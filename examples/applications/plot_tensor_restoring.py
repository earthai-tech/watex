"""
=================================================
Restoring tensor with noised :term:`AMT` data
=================================================

Gives an step-by-step guide for restoring Z tensors 
using  real noised data containing missing 
and weak frequency signals. 
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

# %%
# This is an example of restoring tensors when data contains missing frequency.
# The data used for the demonstration is Huayuan data , Hunan province, China.
# The survey is :term:`AMT` and data is collected alongside two long lines 
# `E1` (2.5km) and `E2` (1.8km) . Unfortunately, because of strong 
# interferences and the human-made made noises (agglomerations, power-lines)  
# and many factories in this area,  signals were strongly corrupted and greatly 
# affected with strong interferences . 
# 
# * Why recovering tensors is important ? 
#
# Commonly, missing and weak signal can just be removed in the whole data and 
# keep only the valid signals from interpolation. However, in the area with 
# strong intereferences when data is too much noised, suppressing the weak 
# signals abd missing frequencies will greatly affect the quality of data 
# thereby leading a misinterpretation of the structures found in this area
# after inversion. 
# 
# For demonstration, we collected 27 samples of raw-data stored as inner dataset 
# and plot the raw data of impedance tensor z in TM mode to visualize the 
# missing tensors. This is the code snippet: 
import watex  as wx 
data = wx.fetch_data('huayuan', return_data =True, samples =27 ,
                     key ='raw', clear_cache=True) # clear watex cache data to save new EDIs
tro = wx.EMProcessing().fit(data) 
#%%
# * Output the impedance tensor in TM mode (`yx`)
# 
# Here we ouput the modulus of complex number of tensor Z. We can also output 
# z as complex number by setting ``out=z`` since the `kind` param  is by 
# default set to ``complex``. Use  'real' or 'imag'part for real and imaginary 
# part instead. Refer to method documentation for more details. 
z_yx = tro.make2d(out= 'zyx', kind ='modulus' )  
 
#%%
# * Visualize the plot using the template 2D from :func:`watex.view.plot2d`.  
wx.view.plot2d(z_yx, 
               y = tro.freqs_,
               to_log10= True,
               plot_style ='imshow', 
               fig_size =(10, 4 ), 
               font_size =7, 
               ylabel ='TM mode (raw): $Z_yx$', 
               cb_label ='TM mode: $Z_yx$', 
               distance =50., # distance between stations
               cmap = 'terrain'
               ) 

#%% 
# As a comment: one can visualize the blank line in the data wich indicates 
# the missing signal at this points. If we try to remove from each missing 
# signal data wich correspond to a data at a certain frequency, we will loose 
# lot of useful informations. Mostly the idea is to interpolate data. The 
# strategy proposed by :code:`watex` for restoring tensor consists to interpolate 
# in two directions (along x-axis and y-axis ) at the same time and decide the 
# best value that approximatively fit the surrounding resistivity values( because 
# the demonstration refer to resistivity tensor in TM mode.). This improve a 
# bit the reality of the resistivity distribution of the area. 
#
# There is a way to do the quality control of the data before restoring using 
# the quality control method :meth:`watex.methods.em.Processing.qc` by 
# specifying the tolerance parameter. By default the data is considered good 
# above 50%. Note that this threshold can be severe with 30%. Here is an example 
# to get the quality control of data 
tro.qc (tol =.4 , return_ratio = True ) # we consider good data from .60% 

# %% 
# The output shows `75%` of data goodness. This score can be improved if the 
# recovers the losses signal is triggered. However, If the user approves this 
# ratio, there is a possibility of output the valid tensors  using the 
# getValid Tensors from method :meth:`watex.methods.em.Processing.getValidTensors` 
# by keeping the same tolerance parameters as:  
#
# .. code-block: python 
#
#    >>> tro.getValidTensors (tol = .4 ) # uncomment this will output new edi with valid tensors
 
# %% 
# The next step consists to recover the missing signals since we assume that 
# we are not satisfy with our qc value = 75% 

# * Recovering missing signals  
Z = tro.zrestore ( ) 

#%% 
# Note that Z is  three dimensional array ( n_freqs, 2 , 2 ), To ferand we can 
# collect the TM restoring tensors using the function :func:`watex.utils.get2dtensor` 
z_yx_restored = wx.get2dtensor(Z, tensor ='z', component='yx') 

#%%
# The impendance tensor z in TM mode can also be output using: 
#
# .. code-block:: python 
#     
#    >>> tro.component ='yx' 
#    >>> z_yx_restored = tro.zrestore ( tensor ='z')

#%% 
# * Plot the recovering tensors 
wx.view.plot2d(z_yx_restored,
               y = tro.freqs_,
               to_log10= True,
               plot_style ='imshow', 
               fig_size =(10, 4 ), 
               font_size =7, 
               ylabel ='TM mode (restored): $Z_yx$', 
               cb_label ='TM mode: $Z_yx$', 
               distance =50.,cmap = 'terrain'
               )
#%%
# The plot below indicates the full strength amplitudes of restored data. 
# Tensors are recovered at all frequencies. 


