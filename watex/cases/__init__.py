# -*- coding: utf-8 -*-
"""
Cases: Pratical cases 
=======================
the 'cases' subpackage implements function and modules already validated and 
used to solve real engineering problems like the FR predictions and boosting 
using the bases learners , SVC  and  ensembles paradigms. 

'features', 'processing', 'modeling' and 'prepare' modules are the bases steps  
and can be used for processing and analyses to give quick depiction of how data 
is look like. This can figure out the next processing steps for solving the 
evidence problem. 

"""
from .prepare import ( 
    BaseSteps, 
    default_pipeline, 
    default_preparation, 
    base_transform 
    )
from .processing import ( 
    Preprocessing , 
    Processing, 
)
from .modeling import ( 
    BaseModel 
    )
from .features import ( 
    GeoFeatures,
    FeatureInspection, 
)

__all__=[
    "BaseSteps", 
    "default_pipeline",
    "default_preparation", 
    "base_transform" , 
    "Preprocessing" , 
    "Processing", 
    "BaseModel",
    "GeoFeatures",
    "FeatureInspection", 
    ]