# -*- coding: utf-8 -*-
#       Author: Kouadio K.Laurent<etanoyau@gmail.con>
#       Create:on Fri Sep 10 15:37:59 2021
#       Licence: MIT

# transform test set using the default pipeline 
# type of encoder use to encode label of training set
from sklearn.preprocessing import LabelEncoder 
# from watex.processing.prepare import BasicSteps  
from watex.datasets import  fetch_data 
from watex.utils.ml_utils import loadDumpedOrSerializedData
from watex.processing.transformers import CategorizeFeatures

__all__=['XT_prepared', 'yT_prepared']

# fetch test data and default pipeline used to prepared training set.
# XT, yT = fetch_data ('Bagoue untouched test sets')
_pipeline = fetch_data('Bagoue default pipeline') 

XT, yT = loadDumpedOrSerializedData('watex/datasets/__XTyT.pkl')

feature_props_to_categorize =[
    ('flow', ([0., 1., 3.], ['FR0', 'FR1', 'FR2', 'FR3']))
    ]
XT__, yT__= XT.copy(), yT.copy()
XT_prepared = _pipeline.transform(XT__)

cObj = CategorizeFeatures(
                   num_columns_properties=feature_props_to_categorize)
y = cObj.fit_transform(X=yT__)
yT_prepared =LabelEncoder ().fit_transform(y)  


# if test data are serialized or dumped , you can loaed file using 
# `loadDumpedOrSerializedData` like: 
#   >>> from watex.utils.ml_utils import loadDumpedOrSerializedData
#   >>> XT_prepared, yT_prepared = loadDumpedOrSerializedData(
#                           filename ='Watex/datasets/__XTyT.pkl')

# target or label name. 
# target ='flow'
# # drop useless features
# drop_features= ['num',
#                 'east', 
#                 'north',
#                 'name', 
#                 'lwi', 
#                 'type' 
#                 ]

# # experiences attributes combinaisions 
# add_attributes =False
# # add attributes indexes to create a new features. 
# attributesIndexes = [
#                     (0, 1),
#                     # (3,4), 
#                     # (1,4), 
#                     # (0, 4)
#                     ] 

# categorize a features on the trainset or label                
# createObjects without passing the data_fn. 
# # readfile and set dataframe
# pTObj = BasicSteps(drop_features = conf_kws['drop_features'],
#                     categorizefeature_props = conf_kws['feature_props_to_categorize'],
#                     target=conf_kws['target'], 
#                     add_attributes = conf_kws['add_attributes'], 
#                     attributes_ix = conf_kws['attributesIndexes'])
#OR
# pTObj = BasicSteps(drop_features = drop_features,
#                     categorizefeature_props = feature_props_to_categorize,
#                     target=target, 
#                     add_attributes = add_attributes, 
#                     attributes_ix = attributesIndexes)
# _, yT_prepared = pTObj.fit_transform(XT, yT, on_testset=True)
# pTObj.fit(X=XT, y=yT)


























