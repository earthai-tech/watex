"""
=================================================
Plot Precision-Recall (PR)
=================================================

computes the score based on the decision function
 and plot the result as a score vs threshold.
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

#%%
from watex.exlib.sklearn import SGDClassifier
from watex.datasets.dload import load_bagoue 
from watex.utils import cattarget 
from watex.view.mlplot import EvalPlot 
X , y = load_bagoue(as_frame =True )
sgd_clf = SGDClassifier(random_state= 42) # our estimator 
b= EvalPlot(scale = True , encode_labels=True)
b.fit_transform(X, y)
# binarize the label b.y 
ybin = cattarget(b.y, labels= 2 ) # can also use labels =[0, 1]
b.y = ybin 
# plot the Precision-recall tradeoff  
b.plotPR(sgd_clf , label =1) # class=1