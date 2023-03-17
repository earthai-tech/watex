"""
=====================================================
Plot Stochastic Linear Adaptative Neuron Classifier  
=====================================================

visualizes the mini-batch by applying batch gradient descent to a 
smaller subset of test data.
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

#%%
# fetch data and uses the preprocessed  
import matplotlib.pyplot as plt 
from watex.base import AdalineStochasticGradientDescent 
from watex.analysis import decision_region 
from watex.datasets import fetch_data 
X, y = fetch_data ('bagoue prepared data') 

fig, axe = plt.subplots (1, 2)
asgd= AdalineStochasticGradientDescent (n_iter=15, eta=.01).fit(X.toarray(), y )
decision_region(X.toarray(), y, clf=asgd, return_axe=True, axe= axe[0]) # test set view
axe[0].set_title ("Adaline - Stochastic Gradient descent")

axe[1].plot(range (1, len(asgd.cost_) +1 ), asgd.cost_, marker ="o")
axe[1].set_xlabel ("Epochs")
axe[1].set_ylabel ("Sum-squared-error")
plt.tight_layout()
plt.show() 