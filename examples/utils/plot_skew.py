"""
===================================
Plot phase sensitive skew 
===================================

shows the phase sensitivity Skew that 
represents a measure of the skew of the  
phases of the impedance tensor
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

# %%
# Skew is a dimensionality tool or a conventional asymmetry parameter based 
# on the Z magnitude.  
# Indeed, the EM signal is influenced by several factors such as the dimensionality
# of the propagation medium and the physical anomalies, which can distort the
# EM field both locally and regionally. The distortion of Z was determined 
# from the quantification of its asymmetry and the deviation from the conditions 
# that define its dimensionality. The parameters used for this purpose are all 
# rotational invariant because the Z components involved in its definition are
# independent of the orientation system used. The conventional asymmetry
# parameter based on the Z magnitude is the skew defined by Swift (1967) as
# follows:
#
# .. math:: 
#
#     skew_{swift}= |\frac{Z_{xx} + Z_{yy}}{ Z_{xy} - Z_{yx}}| 
#    
# When the :math:`skew_{swift}`  is close to ``0.``, we assume a 1D or 2D model
# when the :math:`skew_{swift}` is greater than ``>=0.2``, we assume 3D local 
# anomaly (Bahr, 1991; Reddy et al., 1977).
#
# Furthermore, Bahr (1988) proposed the phase-sensitive skew which calculates
# the skew taking into account the distortions produced in Z over 2D structures
# by shallow conductive anomalies and is defined as follows:
#
# .. math::
#   
#     skew_{Bahr} & = & \sqrt{ \frac{|[D_1, S_2] -[S_1, D_2]|}{|D_2|}} \quad \text{where} 
#    
#     S_1 & = & Z_{xx} + Z_{yy} \quad ; \quad  S_2 = Z_{xy} + Z_{yx} 
#    
#     D_1 & = &  Z_{xx} - Z_{yy} \quad ; \quad  D_2 = Z_{xy} - Z_{yx}
#    
# Note that The phase differences between two complex numbers :math:`C_1` and 
# :math:`C_2` and the corresponding amplitude  products are now abbreviated 
# by the commutators:
#    
# .. math:: 
#  
#     \[C_1, C_2] & = & I_m  C_2*C_1^*
#    
#     \[C_1, C_2] & = & R_e C_1 * I_m C_2  - R_e(C_2)* I_m C_1
#                
# Indeed, :math:`skew_{Bahr}` measures the deviation from the symmetry condition
# through the phase differences between each pair of tensor elements, considering
# that phases are less sensitive to surface distortions(i.e. galvanic distortion).
# The :math:`skew_{Bahr}` threshold is set at ``0.3`` and higher values mean 
# 3D structures (Bahr, 1991).
#
# In this demonstration, we will plot skew with a sample of 20 files and 
# we start by importing the required modules 
import watex as wx 

# %% 
# * Barh method 
edi_sk = wx.fetch_data ("edis", return_data =True , samples = 20 ) 
# we can set the Barh threshold line by setting the threshold line to 'Barh' 
wx.utils.plot_skew (edi_sk, method ='bahr', threshold_line='bahr', fig_size = (11, 5), style ='classic')  

#%% 
# * Swift method 
wx.utils.plot_skew (edi_sk, method ='swift', threshold_line='swift', fig_size = (11, 5), 
                    mode='periods')  


