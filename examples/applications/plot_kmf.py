"""
=======================
K-Means Featurization
=======================
Featurize data for boosting the prediction and 
force model to generalization 

"""
# License: BSD-3-clause
# Author: L.Kouadio 
#%% 
# KMeans Featurisation ( KMF) is a surrogate booster to predict permeability 
# coefficient (k) before any drilling construction. Indeed, KMF creates a 
# compressed spatial index of the data which can be fed into the model for ease 
# of learning and enforce the model capability of generalization. A new 
# predictor based on model stacking technique is built with full target k 
# which balances the spatial distribution of k-labels by clustering the original data. 

#%% 
# We start by importing the required modules 
import matplotlib.pyplot as plt
import scipy 
import sklearn 
from sklearn.datasets import make_moons
from watex.transformers import featurize_X,  KMeansFeaturizer
from watex.exlib import train_test_split, XGBClassifier
from watex.utils import plot_voronoi 

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from watex.datasets import load_mxs 

# %% 
# Use common Moon datasets 
# ============================================
# * Build models with a common data sets. (e.g. Moons dataset) 
# We generate 8000 samples of dataset where we divided as 50% training and 50% testing 
# For reproducing the same samples of data, we fixed the `seed`. 
seed = 1 # to reproduce the dataset
X0 , y0 = make_moons(n_samples = 8000, noise= 0.2 , random_state= seed ) 
X0_train, X0_test, y0_train, y0_test = train_test_split (
    X0, y0 , test_size =.5, random_state = seed )

# %% 
# Here, there is a shortcut way to featurize data at once by calling  the 
# transformer :func:`watex.transformers.featurize_X` to transform X data. It 
# could also returns  KMF_model (kmf_hint) if the parameter `return_model` is set 
# to ``True``.  
X0_train_kmf, y0_train_kmf  = featurize_X(
    X0_train, y0_train , n_clusters =200, target_scale =10) 

#%% 
# * Voronoi plot 2D 
# Veronoi plot can be used to visualize the model using hint ( target associated) and without. 
# For a human representation ( 2D), we used the most two features importances 
# of the consistent data set.
 
# Xpca = nPCA( X0_train, n_components= 2 ) # reduce the data set for two most feature components 
# X0_test, y0_test = make_moons(n_samples=2000, noise=0.3)
fig, ax = plt.subplots(2,1, figsize =(7, 7)) 
kmf_hint = KMeansFeaturizer(n_clusters=200, target_scale=10).fit(X0_train,y0_train)
kmf_no_hint = KMeansFeaturizer(n_clusters=200, target_scale=0).fit(X0_train, y0_train)
plot_voronoi ( X0_train, y0_train ,cluster_centers=kmf_hint.cluster_centers_, 
                  fig_title ='KMF with hint', ax = ax [0] )
plot_voronoi ( X0_train, y0_train,cluster_centers=kmf_no_hint.cluster_centers_, 
                  fig_title ='KMF No hint' , ax = ax[1])

# %%
# As shown in the figure above. The number of clusters when target information is 
# missed span too much of the space between the two classes. Commonly KMF demonstrates 
# its usefulness when cluster boundaries align with class boundaries much more closely. 

#%% 

# Use concrete log data for KMF implementation 
# ============================================ 
# In this test case for implementing the K-Featurization (KMF) surrogate booster 
# we use the hongliu 11 borehole data for application. We will separate the data into 
# 80% training and 20% testing . Note that the hyperparameter should be fixed to 
# avoid the learning process. We used the LogisticRegression, and Gradient 
# boosting as the model for testing. We also fixed the random scale for 
# reproducing the same data.  
seed =25
X, y = load_mxs ( key ='scale', return_X_y= True, samples ="*" , seed = seed)

# *  make binary datasets 
# construct the binary data sets from the target mapping 
# then splitting the data. 
# {0: '1', 1: '11*', 2: '2', 3: '2*', 4: '3', 5: '33*'}
y [y <=2]= 0 ; y [y >0 ]=1 
training_data, test_data, training_labels, test_labels = train_test_split (
    X, y , test_size =.2 ) 

# * Call KMeansFeaturizer to featurize the data X 
kmf_hint = KMeansFeaturizer(n_clusters=100, target_scale=10).fit(training_data,
training_labels)
kmf_no_hint = KMeansFeaturizer(n_clusters=100, target_scale=0).fit(training_data,
training_labels
)
# Use the k-means featurizer to generate cluster features
training_cluster_features = kmf_hint.transform(training_data)
test_cluster_features = kmf_hint.transform(test_data)
# Form new input features with cluster features
training_with_cluster = scipy.sparse.hstack(
    (training_data,scipy.sparse.coo_matrix(training_cluster_features)))
test_with_cluster = scipy.sparse.hstack(
    (test_data, scipy.sparse.coo_matrix(test_cluster_features)))

# %% Build models 
# * Build the classifiers as example of Logistic regression ad XGBClassifiers 
lr_cluster = LogisticRegression(random_state=seed).fit(training_with_cluster,
training_labels)
xgb_cluster = XGBClassifier(max_depth =13 , 
                            n_estimators =50 , 
                            learning_rate = 0.09, 
               booster = 'gbtree').fit(training_with_cluster,training_labels  )
classifier_names = [
                'LR',
                'Boosted Trees']
classifiers = [LogisticRegression(random_state=seed),
            GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,
            max_depth=5)]

# * Plot ROC curves for demonstration 
for model in classifiers:
    model.fit(training_data, training_labels)
# Helper function to evaluate classifier performance using ROC
def test_roc(model, data, labels):
    if hasattr(model, "decision_function"):
        predictions = model.decision_function(data)
    else:
        predictions = model.predict_proba(data)[:,1]
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, predictions)
    return fpr, tpr

plt.figure()
fpr_cluster, tpr_cluster = test_roc(xgb_cluster, test_with_cluster, test_labels)
plt.plot(fpr_cluster, tpr_cluster, 'r-', label='LR with KMF')
for i, model in enumerate(classifiers):
    fpr, tpr = test_roc(model, test_data, test_labels)
    plt.plot(fpr, tpr, label=classifier_names[i])
 
plt.plot([0, 1], [0, 1], 'k--')
plt.legend() 
























