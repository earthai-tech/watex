"""
External sub-package is an impedance `Z`, resistivity and phase tensors 
manipulation. It also includes some third-party codes and other scripts 
elaborated to interact between the former and the :code:`watex` package. 
"""
from .z import ( 
    ResPhase, 
    Z 
    ) 

__all__=["ResPhase", "Z"]
