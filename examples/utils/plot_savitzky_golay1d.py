"""
================================================
Plot Savitzky Golay 1D filter 
================================================

plot savitzky golay 1d from noise signal compared to filtered signal.
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause 

#%%
# Import required modules 
import numpy as np 
import matplotlib.pyplot as plt 
from watex.utils.exmath import savitzky_golay1d 

#%%
# Generate signal for filtering and add Gaussian noises 
t = np.linspace(-3, 3, 500)
y = np.exp( -t**2 ) + np.random.normal(0, 0.1, t.shape)
# compute filtered signal 
ysg = savitzky_golay1d(y, window_size=45, order=3, mode ='valid')
# plot noise signal vs filtered signal 
plt.figure(figsize =(10, 4)) 
plt.plot(t, y, label='Noisy signal')
plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
plt.plot(t, ysg, 'r', label='Filtered signal')
plt.ylabel('precomputed signal') 
plt.xlabel ('signal bandwidth') 
plt.legend()
plt.show()