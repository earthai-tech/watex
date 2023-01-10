"""
=================================================
Plot projection
=================================================

create a scatterplot of all instances to visualize data if there is 
there is geographical information(latitude/longitude or
easting/northing) in the data. It alows to show the distributions of the data 
in the survey area.
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

#%%
from watex.datasets import fetch_data 
from watex.view.mlplot import plotProjection 
# Discard all the non-numeric data 
# then inut numerical data 
from watex.utils import to_numeric_dtypes, naive_imputer
X, Xt, *_ = fetch_data ('bagoue', split_X_y =True, as_frame =True) 
X =to_numeric_dtypes(X, pop_cat_features=True )
X= naive_imputer(X)
Xt = to_numeric_dtypes(Xt, pop_cat_features=True )
Xt= naive_imputer(Xt)
plot_kws = dict (fig_size=(8, 12),
                 lc='k',
                 marker='o',
                 lw =3.,
                 font_size=15.,
                 xlabel= 'easting (m) ',
                 ylabel='northing (m)' , 
                 markerfacecolor ='k', 
                 markeredgecolor='r',
                 alpha =1., 
                 markeredgewidth=2., 
                 show_grid =True,
                 galpha =0.2, 
                 glw=.5, 
                 rotate_xlabel =90.,
                 fs =3.,
                 s =None )
plotProjection( X, Xt , columns= ['east', 'north'], 
                    trainlabel='train location', 
                    testlabel='test location', **plot_kws
                   )