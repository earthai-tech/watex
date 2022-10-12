# -*- coding: utf-8 -*-
"""
Cases: Pratical cases 
=======================
the 'cases' subpackages implements function and modules already validated and 
used to solve real ingeneering problems like the FR predictions and boosting 
using the bases learners , SVC  and  ensembles paradigms. 

'Features', 'processing', 'modeling' and 'prepare' modules and bases steps can 
be used for processing and analyses to give quick depiction of how data 
is look like and and can figure out the next processing steps for solving the 
evidence problem. 

"""
from .prepare import ( 
    BaseSteps, 
    defaultPipeline
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
    "defaultPipeline",
    "Preprocessing" , 
    "Processing", 
    "BaseModel",
    "GeoFeatures",
    "FeatureInspection", 
    ]