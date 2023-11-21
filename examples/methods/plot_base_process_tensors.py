"""
================
Plot tensors
================

plot the resistivity/phase and Impendance 
# real and imaginary parts. 
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

# %%
# :func:`~watex.utils.plot_tensors` plots resistivity and phase tensors  

#%% 
# We start by importing the required modules as: 
from watex.datasets import fetch_data 
from watex.utils.plotutils import plot_tensors 
from watex.methods import MTProcess 

#%%
# Before we'll fetch 17 samples of EDI objets 
edi_data = fetch_data ("edis", samples =17, return_data =True )
# then we will do remove the noises using the 
# adaptative -moving-average spatial filter as 
amt = MTProcess (verbose =True).fit( edi_data).remove_noises (method ='ama')

#%% 
# * Plot the resistivity/phase 
plot_tensors(amt.ediObjs_, station =7 )

#%%
# * Plot the impedances Z real/imaginary parts. 

# Setting the parameter `zplot` to ``True`` plots only the 
# impedance tensor. 
plot_tensors(amt.ediObjs_, station =7 , zplot =True )
