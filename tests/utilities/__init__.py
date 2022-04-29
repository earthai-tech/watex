
import numpy as np 

rangn = np.random.RandomState(42) 
array1D = np.abs(rangn.randn(7))  # for res values  
# for [stations, easting, northing , resistivity] 
# 10 m is used as dipole length value 
array2D = np.abs(rangn.randn(21, 4))
dipoleLength = 10. 
array2D[:, 0] = np.arange(0 , array2D.shape[0] * dipoleLength  , dipoleLength  )
#  make a copy of arrayx with position start with 150 m  
# dipole length is 50 m 
dipoleLengthX = 50.
array2DX = array2D.copy() 
array2DX[:, 0] = np.arange(
    3*dipoleLengthX , array2DX.shape[0]  * dipoleLengthX  + 3*dipoleLengthX , dipoleLengthX  )
# extra -data 
extraarray2D =  np.abs(rangn.randn(21, 7)) 
extraarray2D [:, 0] = array2DX[:, 0]
