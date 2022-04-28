# -*- coding: utf-8 -*-

from typing import (
    TypeVar, 
    List,
    Tuple,
    Sequence, 
    Dict, 
    Iterable, 
    Callable, 
    Union, 
    Any , 
    # Shape, 
    Generic,
    # NDArray, 
    # IntVar,
    Optional,
    Union,
    Type  

)


T = TypeVar('T')
S = TypeVar('S', bound=Union[int, Type[int]])
# N= IntVar('N')
# M=IntVar ('M')

class NDArray(Generic[T, S]): ...
class Array(List[T]): ...
