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
