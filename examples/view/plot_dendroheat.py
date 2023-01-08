"""
=================================================
Plot dendrogram combined with heatmap
=================================================

visualize model fined tuned scores vs the cross validation 
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

# * Plot using random data
import numpy as np 
import pandas as pd 
from watex.view.mlplot import plotDendroheat
np.random.seed(123) 
variables =['X', 'Y', 'Z'] ; labels =['ID_0', 'ID_1', 'ID_2',
                                         'ID_3', 'ID_4']
X= np.random.random_sample ([5,3]) *10 
df =pd.DataFrame (X, columns =variables, index =labels)
plotDendroheat (df, )

# (2) -> Use Bagoue data 
# from watex.datasets import load_bagoue  
# X, y = load_bagoue (as_frame=True )
# X =X[['magnitude', 'power', 'sfi']].astype(float) # convert to float
# plotDendroheat (X )