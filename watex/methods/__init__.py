"""
Methods sub-package is composed of DC-Resistivity, EM, and hydro-geological 
methods for prediction parameter computations as well as exporting filtering 
tensors for 1D/2D modeling purpose.
"""
from .electrical import (
    ResistivityProfiling ,
    VerticalSounding, 
    DCProfiling, 
    DCSounding,
)

from .em import ( 
    EM, 
    EMAP,
    MT,
    filter_noises,
    drop_frequencies, 
    )
from .erp import ( 
    ERPCollection ,
    ERP , 
    DCMagic
    )
from .hydro import ( 
    Hydrogeology, 
    AqGroup, 
    AqSection, 
    MXS, 
    Logging, 
    )

__all__=[
    "EM", 
    "ResistivityProfiling" ,
    "VerticalSounding", 
    "DCProfiling", 
    "DCSounding",
    "EMAP", 
    "MT", 
    "filter_noises", 
    "ERPCollection", 
    "ERP", 
    "Hydrogeology", 
    "AqGroup", 
    "AqSection", 
    "MXS", 
    "Logging", 
    "DCMagic", 
    "filter_noises",
    "drop_frequencies"
    ]