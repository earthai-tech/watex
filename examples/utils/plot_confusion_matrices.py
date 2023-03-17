"""
=================================================
Plot confusion matrices  
=================================================

plots inline multiple confusion matrices from  selected models  
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause 

# %% 
# Plot Radial basis function from the SVM kernel machines, the logit and the 
# randomforest  stored as premodels in `p` objects and plot their 
# confusion matrices 

# Import the required models and fetch 4 models 
# derived from RBF SVM kernel machines 
# for instance then plot the confusion metric 
import matplotlib.pyplot as plt 
plt.style.use ('classic')
from watex.datasets import fetch_data
from watex.exlib.sklearn import train_test_split 
from watex.models.premodels import p
from watex.utils.plotutils import plot_confusion_matrices 
# split the  data . Note that fetch_data output X and y 
X, Xt, y, yt  = train_test_split (* fetch_data ('bagoue analysed'), test_size =.25  )  
# compose the models 
# from RBF, and poly 
models =[ p.SVM.rbf.best_estimator_,
         p.LogisticRegression.best_estimator_,
         p.RandomForest.best_estimator_ 
         ]
print(models )
# now fit all estimators 
fitted_models = [model.fit(X, y) for model in models ]
# %% 
# * Plot the confusions metrics with the selected estimators 
plot_confusion_matrices(fitted_models , Xt, yt)


# %% 
# * Plot  using the yellowbrick packages 

plot_confusion_matrices(fitted_models , Xt, yt, pkg ='yb')