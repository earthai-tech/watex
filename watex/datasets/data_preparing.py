# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 16:47:58 2021

@author: @Daniel03

 Usage 
 -----
     This module is a part of case history proccessing in Bagoue Area. It is 
     not a perfect processing by give some steps of data preparing and 
     pipeline creation.
    
"""
# import modules 
import numpy as np 
import pandas as pd
# import matplotlib.pyplot as plt 
from pandas.plotting import scatter_matrix 

from sklearn.pipeline import Pipeline, FeatureUnion  
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

import watex.utils.ml_utils as mfunc
from watex.utils.transformers import StratifiedWithCategoryAdder, CategorizeFeatures 
from watex.utils.transformers import CombinedAttributesAdder, DataFrameSelector 
# from watex.analysis.features import sl_analysis 


# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')


""" 
--------------------------------(1) DATA PREPARING----------------------------- 
Load data and stratified data by generating a trainset and test 
 set straified.In fact, set `return-train`to ``False`` to only get the 
 first stratified data. """
# Load data 

# df =pd.read_csv('data/geo_fdata/_bagoue_civ_loc_ves&erpdata.csv')
# print(df)
df = mfunc.load_data('data/geo_fdata')
bagdataset = df.copy()
#=============stratification and categorization values ========================

#  Stratified data n_splits =1. and return the overall trainset stratified.

base_colum_for_stratification='flow'
stratifiedNumObj= StratifiedWithCategoryAdder(base_colum_for_stratification,
                                              return_train=True)
strat_train_set , strat_test_set = stratifiedNumObj.fit_transform(X=df)
# keep a copy of the data 
stratified_test_set = strat_test_set.copy()
bag_train_set = strat_train_set.copy()
raw_X = strat_train_set.drop('flow', axis =1).copy()
raw_y = strat_train_set['flow'].copy()
# visualize the data and get insights 

# bag_train_set.plot(kind='scatter', x='east', y='north', alpha=0.4, 
#                    s=bag_train_set['sfi']*50, label ='flow m3/h', 
#                    c= 'flow', cmap= plt.get_cmap('jet'), colorbar =True)
# plt.legend()
# Drop `numbering column ['num', 'east', 'north', 'name'] from data  
bag_train_set.drop(['num', 'east', 'north', 'name' ], inplace =True, axis =1)
strat_test_set.drop(['num', 'east', 'north', 'name'], inplace=True, axis =1)

#visualize correlation data 
corr_matrix = bag_train_set.corr()
# print(corr_matrix['flow'].sort_values(ascending =False))
# Plot numerical attributes using pandas.plotting.scatter_matrix 
# attributes =['flow','sfi', 'ohmS', 'power', 'magnitude']
# scatter_matrix(bag_train_set[attributes], figsize=(12,8))

# Categorize the target flow into FR0, FR1, FR2, FR3

""" Because we throw to classification problem i.e. predicted flow classes 
we firstly categorize the flow (labels) into the corresponding classes. 
Categorizes attributes doesant nt only concerns the `flow`. It can be onother 
features in the dataframe. Refer to 
:class:`~utils.transformers.CategorizeFeatures """

# slObj =sl_analysis(df =bag_train_set, set_index =False, 
# drop_columns=None, col_id ='name',
#                      flow_classes =[0., 1., 3.])
catObj = CategorizeFeatures(num_columns_properties=[
                    ('flow', ([0., 1., 3.], ['FR0', 'FR1', 'FR2', 'FR3']))
                    ])
bag_train_set_cat = catObj.fit_transform(bag_train_set)
bag_train_set_cat = pd.DataFrame(data =bag_train_set_cat ,
                                 columns =bag_train_set.columns)

# bag_train_set['flow']= bag_train_set['flow'].apply(lambda id_: map_flow(id_)) 
# strat_test_set['flow']= strat_test_set['flow'].apply(lambda id_: map_flow(id_)) 

#======Remove, encoded labels (y) and convert dataframe to numerical values====
 
#PREPARING THE data for machine learning 
# Let revert to a clean training set and let's separate the predictors and the
 # labels since we dont necessary want to apply the same transformations 
# separate the `the labels ``flow`` to the attributes 
bagoue_train_set = bag_train_set_cat.drop('flow', axis =1)
bagoue_labels = bag_train_set_cat['flow'].copy()

""" Labels are categorial featuresm, so let try to encode values using 
sklearn.LabelEncoder. """
train_encoder =LabelEncoder()
bagoue_label_encoded = train_encoder.fit_transform(bagoue_labels)
# to decode 
# bagoue_label_decoded = encoder.inverse_transform(bagoue_label_encoded)

# bagoue_train_set= bagoue_trainset.drop('name', axis =1)
# print(bagoue_train_set.shape)
# DATA cleaning 
# fill values using median values 
# create a pipelines 
# numerical pipelines 
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# (6, 5) = lwi_per_Ohms 
# NUMERICAL STEPS 
# convert pandas dataframe to numeric values using astype but can also 
# use pd.to_numeric(errors='coerce') to convert non mumerical values to NaN. 
# for instance:
# for colums in ['east', 'north', 'power', 'magnitude',  'sfi', 'ohmS', 'lwi']: 
    # pd.to_numeric(bagoue_train_set[columns], errors='coerce')
bagoue_train_set= bagoue_train_set.astype(
                        {'power':np.int32, 
                        'magnitude': np.float64, 
                        'sfi':np.float64, 
                        'lwi':np.float64, 
                        # 'east':np.float64, 
                        # 'north':np.float64, 
                        'ohmS':np.float64,
                        })

default_X= bagoue_train_set.copy()
default_y = bagoue_labels.copy() 

""" 
-------------------------(2) PIPELINE CREATING --------------------------------

Try to create a pipe"""
#=================Pilepine creation of trainset evaluation ====================

# selectorObj =  DataFrameSelector(select_type='num')
# selectorObj.fit(bagoue_train_set)
# X = selectorObj.fit_transform(bagoue_train_set)
# print(X)
# combObj = CombinedAttributesAdder(add_attributes=True, attributes_ix=[(6, 5)])
# X = combObj.fit_transform(X)
# impObj = SimpleImputer(missing_values=np.nan, strategy='median')
# X = impObj.fit_transform(X)
# stanObj = StandardScaler()
# X= stanObj.fit_transform(X) 
# (1, 0), (5,4) magnitude/power and ohms/sfi

num_pipeline =Pipeline([
    ('selector', DataFrameSelector(select_type='num')),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
    ('attribs_adder',CombinedAttributesAdder(add_attributes=True, 
                                              attributes_ix=[(1, 0)])), 
    # ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')), 
    ('std_scaler', StandardScaler())
    ])
# categorical pipelines 

cat_pipeline =Pipeline ([
    ('selector', DataFrameSelector(select_type='cat')),
    ('one_hot_encoder',OneHotEncoder())
    ])
full_pipeline =FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline), 
    ('cat_pipeline', cat_pipeline)
    ])
# print(full_pipeline)
# # now run to the whole pipeline: 
bagoue_train_set_prepared =full_pipeline.fit_transform(bagoue_train_set)
# print(bagoue_train_set_prepared )
# print(print(bagoue_train_set_prepared.shape ))
"""
---------------------------------(3)TEST SET SAVING -------------------------
 Apply the same change  as sata transformation above to the test set separely 
 and keep the safe stratified  test set with encoded label. 
 """
# TESTSET 
bagoue_strat_test_set = strat_test_set.copy()
# call catObj from class CategorizeFeatures
bagoue_strat_test_set_cat = catObj.fit_transform(bagoue_strat_test_set)
bagoue_strat_test_set_cat = pd.DataFrame(data =bagoue_strat_test_set_cat  , 
                                     columns =bagoue_strat_test_set.columns)
bagoue_testset_stratified = bagoue_strat_test_set_cat.drop('flow', axis =1)
bagoue_test_set_labels = bagoue_strat_test_set_cat['flow'].copy()

test_encoder =LabelEncoder()
bagoue_testset_label_encoded = test_encoder.fit_transform(bagoue_test_set_labels)
# bagoue_testset_stratified = bagoue_testset_stratified.drop('name', axis =1)
bagoue_testset_stratified = bagoue_testset_stratified.astype({
                        'power':np.int32, 
                        'magnitude': np.float64, 
                        'sfi':np.float64, 
                        'lwi':np.float64, 
                        # 'east':np.float64, 
                        # 'north':np.float64, 
                        'ohmS':np.float64,
                        })

#---Aplied PCA to drop somefeatures 
""" -------------------------(4)Optional:Applied PCA--------------------------- 
    Select componentd with many variances and get the features importances.
"""
trainset2 = bagoue_train_set.copy()

from sklearn.preprocessing import OrdinalEncoder 
datafameObj =  DataFrameSelector(select_type='num')
X_num = datafameObj.fit_transform(trainset2 )
X_num_columns =datafameObj.attribute_names 
X_num_df = pd.DataFrame(data = X_num, columns=X_num_columns)
# replace nan by np.nam 
imputer_obj = SimpleImputer(missing_values=np.nan, strategy='median')
X_num_imp =imputer_obj.fit_transform(X_num_df)
# scale the dataset 
scaler =StandardScaler()
X_num_scaled = scaler.fit_transform(X_num_imp)
X_num_df = pd.DataFrame(data = X_num_scaled, columns=X_num_columns)
# retrive categorial values and encode data using the ordinal Encoder 
cat_datafameObj =  DataFrameSelector(select_type='cat')
X_cat= cat_datafameObj.fit_transform(trainset2)
X_cat_columns =cat_datafameObj.attribute_names 
X_cat_df = pd.DataFrame(data = X_cat, columns=X_cat_columns )
X_ordinal_encoded = OrdinalEncoder()
X_ord= X_ordinal_encoded.fit_transform(X_cat_df )
X_cat_df = pd.DataFrame(data = X_ord, columns=X_cat_columns )

# concat the datafrane into one.
X_train_2 = pd.concat([X_num_df, X_cat_df], axis =1)

# strat component analysis 
from sklearn.decomposition import PCA 

# pca = PCA (n_components =2)
# X2D = pca.fit_transform(X_train_2)
# print(pca.explained_variance_ratio_)


pca2 = PCA()
pca2.fit(X_train_2)
cumsum = np.cumsum(pca2.explained_variance_ratio_)
d= np.argmax(cumsum>=95)+1 

pca =PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train_2)
# print(pca.explained_variance_ratio_)


# pca3 = PCA(n_components=2)
# X_pca_reduced = pca3.fit_transform(X_train_2)
# X_pca_inversed = pca3.inverse_transform(X_pca_reduced )

# X_pca =  pd.DataFrame(data = X_pca_inversed, columns=X_train_2.columns )
# plot the variances vs ndimensions 
# import matplotlib.pyplot as plt 
# plt.plot(cumsum, label ='pca explainedvariance vs dimensions')
# plt.xlabel('Ndimensions')
# plt.ylabel('ExplainedVariance')
# # print(cumsum )
# # print(X_reduced)
# print(list(X_train_2.columns)) # get the features list 
# print(pca.components_) # get the components [n_components, n_features]
# print(pca.n_components_) # number of components after set the variance ration to >=0.95
# print(pca.explained_variance_)














