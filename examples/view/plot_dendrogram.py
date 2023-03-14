"""
=================================================
Plot dendrogram 
=================================================

visualizes specific features on a dendrogram diagram 
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

#%%
#Use the Iris dataset from :func:`~watex.datasets.load_iris` and 
# return :class:`~watex.utils.box.Boxspace` objects where the 
# the frame , feature_names and target_names are the attributes. 
# Thus, rather than creating new columns to pass as`colums` arguments, 
# we uses the `feature_names` attribute instead: 
from watex.datasets import load_iris 
from watex.view import plotDendrogram
data = load_iris () # return a box data objet  
# print the five row of the iris dataframe 
print(data.frame.head()) 

#%% 
# Print the feature names 
print(data.feature_names ) 

#%%
# Plot the dendrogram  
plotDendrogram (data.frame, columns =data.feature_names[:2] ) 