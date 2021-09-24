# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent,Thu Sep 23 16:19:52 2021
# automate datapreparation.

from sklearn.model_selection import cross_val_score 
from watex.processing.process import PrepareDATA 

__all__ = ['X', 'y', 'X0','y0', 'XT', 'yT','_X',
         'X_prepared', 'y_prepared',  '_pipeline', ] 


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
                    (1, 0),
                    # (4,3)
                    ] 
# categorize a features on the trainset or label 
feature_props_to_categorize =[
    ('flow', ([0., 1., 3.], ['FR0', 'FR1', 'FR2', 'FR3']))
    ]
                        
# bring your own pipelines .if None, use default pipeline.
ownPipeline =None 
# createOnjects. 
# readfile and set dataframe
prepareObj =PrepareDATA(data = data_fn,
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

def quickscoring_evaluation_using_cross_validation(
        clf, X, y, cv=7, scoring ='accuracy', display='off'): 
    scores = cross_val_score(clf , X, y, cv = cv, scoring=scoring)
                         
    if display or display =='on':
        
        print('clf=:', clf.__class__.__name__)
        print('scores=:', scores )
        print('scores.mean=:', scores.mean())
    
    return scores , scores.mean()


#-------------------------------------------------------------------------    

# test set stratified data. Untouchable unless the best model is found.
XT = prepareObj.X_
yT= prepareObj.y_