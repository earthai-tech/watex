# -*- coding: utf-8 -*-

""" 
Exceptions 
=============

Each exception is related to its modules following py the suffix `Error`. It 
inherits from top BaseExceptiom of Python build-in functions. To drop or change 
any exception here, after any change, move to its reference  module to change 
it so.  


"""
class ArgumentError(Exception): 
    pass 
class SiteError(Exception): 
    pass 

class DatasetError(Exception): 
    """ Raises exception when mutiple data are passed as list of arguments 
    where shape , columns and sizee are compared. If one aforementionned 
    details does not fit all the data. An error raises. Furthermore, an 
    error also raises when some items in the data are not valid according to 
    the prescipted details beforehand."""
    pass 

class EDIError(Exception):
    pass 

class HeaderError(Exception):
    pass 

class ConfigError(Exception):
    pass

class FileHandlingError(Exception):
    pass

class TipError(Exception):
    pass 

class PlotError(Exception): 
    pass 

class ParameterNumberError(Exception):
    pass

class ProcessingError(Exception):
    pass

class ResistivityError(Exception):
    pass

class StationError(Exception):
    pass

class FeatureError(Exception):
    pass

class EstimatorError(Exception):
    pass

class GeoPropertyError(Exception): 
    pass
class GeoArgumentError(Exception): 
    pass

class HintError(Exception): 
    pass

class SQLError(Exception): 
    pass
class StrataError(Exception): 
    pass

class SQLManagerError(Exception): 
    pass

class GeoDatabaseError(Exception): 
    pass

class ERPError(Exception):
    pass

class VESError(Exception):
    pass

class ExtractionError(Exception): 
    pass 

class EMError (Exception): 
    pass 

class FrequencyError (Exception): 
    pass 

class CoordinateError (Exception): 
    pass 

class TopModuleError (Exception): 
    pass 
class NotFittedError (Exception): 
    pass 

class ScikitLearnImportError(Exception ): 
    pass 

class GISError (Exception): 
    pass 
class LearningError(Exception): 
    pass 

class DepthError (Exception):
    """ Raises exception with everything that does not support the depth 
    line a multidimensional array. Commonly Depth is a one-dimensional array 
    and its atttribute name when pandas daframe of series is given must 
    contain at least the name 'z'. If such name does not exist in the 
    pandas dataframe or serie name, specify other name using the parameter
    'zname'  througthout the package. Be aware to not confuse
    the parameters 'depth' with 'zname' . While the former specifies the 1d
    array, the latter both are expected the name of the depth as a pandas 
    series."""
    pass 

class AquiferGroupError (Exception):
    """ Raises exception with everything that does not relate to the aquifer 
    like a multidimensional array. Commonly Aquifer is a one-dimensional array 
    composed of categorical values expected  to be the layer/rock name where 
    the pumping is performed. Note aquifer is composed of categorical data 
    at last labels are encoded into a numerical values. If that is the 
    case, values should an integer not any other types.  Furthermore, it is 
    better to not confused the permeanility coefficient 'k' with aquifer 
    even the latter is tied to the former. In the hydrogeology modules, 
    aquifer works apart as condirering a single column preferably located in 
    the in a target dafarame. However, it does not a matter whether it 
    is considered as a feature provided that the purpose fits exactly the 
    objective of the users."""
    pass 

