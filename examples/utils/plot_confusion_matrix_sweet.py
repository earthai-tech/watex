"""
=================================================
Plot confusion matrix  sweet
=================================================

plots a confusion matrix using the 'yellowbrick' package. 
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause 

#%%
#Import the required models and fetch an extreme gradient boosting 
# for instance then plot the confusion metric 
import matplotlib.pyplot as plt 
plt.style.use ('classic')
from watex.datasets import fetch_data
from watex.exlib.sklearn import train_test_split 
from watex.models import pModels 
from watex.utils.plotutils import plot_yb_confusion_matrix
# split the  data . Note that fetch_data output X and y 
X, Xt, y, yt  = train_test_split (* fetch_data ('bagoue analysed'), test_size =.25  )  
# train the model with the best estimator 
pmo = pModels (model ='xgboost' ) 
pmo.fit(X, y )
print(pmo.estimator_ ) # pmo.XGB.best_estimator_
#%% 
# Predict the score using under the hood the best estimator 
# for adaboost classifier 
ypred = pmo.predict(Xt) 

# now plot the score 
plot_yb_confusion_matrix (pmo.XGB.best_estimator_, Xt, yt  )
