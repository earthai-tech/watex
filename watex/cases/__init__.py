# -*- coding: utf-8 -*-
"""
'cases' subpackage implements function and modules already validated and 
used to solve real engineering problems such as the flow rate prediction and boosting 
using the base learners, SVC  and  ensemble paradigms. 

'features', 'processing', 'modeling' and 'prepare' modules have base steps  
and can be used for processing and analyses to give a quick depiction of how data 
looks like. This can figure out the next processing steps for solving the 
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
    FeatureInspection
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