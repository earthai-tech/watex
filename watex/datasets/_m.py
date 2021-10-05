# -*- coding: utf-8 -*-
#       Author: Kouadio K.Laurent<etanoyau@gmail.con>
#       Create:on Fri Sep 10 15:37:59 2021
#       Licence: MIT

from watex.processing.prepare import BasicSteps  
from watex.datasets.config import XT, yT#,conf_kws 

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
                 
# createOnjects. 
# # readfile and set dataframe
# pTObj = BasicSteps(drop_features = conf_kws['drop_features'],
#                     categorizefeature_props = conf_kws['feature_props_to_categorize'],
#                     target=conf_kws['target'], 
#                     add_attributes = conf_kws['add_attributes'], 
#                     attributes_ix = conf_kws['attributesIndexes'])
pTObj = BasicSteps(drop_features = drop_features,
                    categorizefeature_props = feature_props_to_categorize,
                    target=target, 
                    add_attributes = add_attributes, 
                    attributes_ix = attributesIndexes)
pTObj.fit(X=XT, y=yT)
XT_prepared, yT_prepared = pTObj.transform(on_testset=True)
























