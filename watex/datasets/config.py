# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent,Thu Sep 23 16:19:52 2021
# automate datapreparation.

from watex.processing.prepare import BasicSteps 

__all__ = ['X', 'y', 'X0','y0', 'XT', 'yT','_X',
         'X_prepared', 'y_prepared',  '_pipeline','df0', 'df1', 'conf_kws'] 


# path to dataset 
data_fn ='data/geo_fdata/main.bagciv.data.csv'
#-------------------------------------------------------------------------
# target or label name. 
target ='flow'
# drop useless features
drop_features= ['num',
                'east', 
                'north',
                'name', 
                # 'lwi', 
                # 'type' 
                ]
# experiences attributes combinaisions 
add_attributes =False
# add attributes indexes to create a new features. 
attributesIndexes = [
                    (0, 1),
                    # (3,4), 
                    # (1,4), 
                    # (0, 4)
                    ] 
# categorize a features on the trainset or label 
feature_props_to_categorize =[
    ('flow', ([0., 1., 3.], ['FR0', 'FR1', 'FR2', 'FR3']))
    ]
                        
# bring your own pipelines .if None, use default pipeline.
ownPipeline =None 
conf_kws = {'target':target, 
            'drop_features':drop_features, 
            'add_attributes':add_attributes, 
            'attributesIndexes':attributesIndexes, 
            'feature_props_to_categorize':feature_props_to_categorize,
            }
# createOnjects. 
# readfile and set dataframe
prepareObj =BasicSteps(data = data_fn,
                        drop_features = drop_features,
                        categorizefeature_props = feature_props_to_categorize,
                        target=target, 
                        add_attributes = add_attributes, 
                        attributes_ix = attributesIndexes
                        )
data= prepareObj.data 
X =prepareObj.X             # strafified training set 
y =prepareObj.y             # stratified label 

prepareObj.fit_transform(X, y)
# --> Data sanitize but keep categorical features not encoded.
#   Text attributes not encoded remains safe. 
X0 = prepareObj.X0          # cleaning and attr combined training set 
y0= prepareObj.y0           # cleaning and attr combined label 

X_prepared = prepareObj.X_prepared  # Train Set prepared 
y_prepared= prepareObj.y_prepared   # label encoded (prepared)
_X = prepareObj._Xpd                # training categorical ordinal encoded features.
_pipeline = prepareObj.pipeline 

df0 = prepareObj._df0
df1 = prepareObj._df1
#-------------------------------------------------------------------------    

# test set stratified data. Untouchable unless the best model is found.
XT = prepareObj.X_
yT= prepareObj.y_
