# -*- coding: utf-8 -*-
"""
'cases' subpackage implements functions and modules already used to solve real 
engineering problems such as the flow rate prediction and boosting 
using the base learners and  an ensemble paradigms. 

:mod:`~watex.cases.features`, :mod:`~watex.cases.processing`, 
:mod:`~watex.cases.modeling` and :mod:`~watex.cases.prepare` modules have 
base step procedures and can be used for  processing and analyses to 
give a quick depiction of how data looks like and model performance estimation. 
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