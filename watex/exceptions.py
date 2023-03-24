# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
""" 
List of :code:`watex` exceptions for warning users. 
"""
class ArgumentError(Exception): 
    """
    Raises an Exception when values passed to the callable is not the expected 
    one. Is deprecated and should be replaced by 'TypeError' or 'ValueError' 
    for the next release."""
    pass 
class SiteError(Exception): 
    """Raises exception to everything related to the Site, Location. For 
    instance, inappropriate coordinates values."""
    pass 

class DatasetError(Exception): 
    """ Raises exception when mutiple data are passed as list of arguments 
    where shape , columns and sizee are compared. If one aforementionned 
    details does not fit all the data. An error raises. Furthermore, an 
    error also raises when some items in the data are not valid according to 
    the prescipted details beforehand."""
    pass 

class DCError (Exception): 
    """Raises exception when data passed to DC base class are not consistent.
    DCType expect D-type for Dataframe , F-type for file object or P-type of 
    pathlike object. 
    
    """
    
class EDIError(Exception):
    """Raises an Exception if the given SEG-Electrical Data Interchange data 
    does not fit the appropriate EDI-format. The correct SEG-EDI file building
    can be found in `Wight, D.E., Drive, B., 1988.` and can be donwloaded
    at <https://www.mtnet.info/docs/seg_mt_emap_1987.pdf> """
    pass 

class HeaderError(Exception):
    """ Raises an Exception if the file/data header is missing, commonly the 
    appropriate requested columns"""
    pass 

class ConfigError(Exception):
    """ Raises an Exception if configuration (file or not ) failed to be 
    executed correctly."""
    pass

class FileHandlingError(Exception):
    """Raises an Exception if failed to manipulate the files properly. Also, 
    occurs if there is no permissions for user to open, read and write 
    the files"""
    pass

class TipError(Exception):
    """Raises an Exception if the tip proposed to shortcut the plot 
    visualization isn't appropriate"""
    pass 

class PlotError(Exception): 
    """Raises an Exception if the plot cannot be run sucessffully."""
    pass 

class ParameterNumberError(Exception):
    """Raises an Exception if the given parameters are not the parameters 
    expected for a proper computation."""
    pass

class ProcessingError(Exception):
    """Raises an Exception if the auto/data processing failed to be executed 
    properly."""
    pass
class ProfileError(Exception):
    """Raises an Exception if the arguments passed to the Profile object are 
    mismatched or wrong."""
    pass
class ResistivityError(Exception):
    """Raises an Exception if the resistivity array is missing in the dataset 
    or the column name/index for restrieving the resistivity data is wrong."""
    pass

class StationError(Exception):
    """Raises an Exception if the station position or index is out of the 
    number of sites collected during the survey. Moreover, this error occurs
    if the given station does not respect the station naming proposed througout
    the`watex`_ package or unable to find the station position in the dataset.
    """
    pass

class FeatureError(Exception):
    """Raises an Exception if the features handling failed to be executed 
    properly."""
    pass

class EstimatorError(Exception):
    """Raises an Exception if the estimator or assessor passed is wrong."""
    pass

class GeoPropertyError(Exception): 
    """ Raises an Exception if the Geological property objects are trying to 
    be modified externally."""
    pass
class GeoArgumentError(Exception): 
    """Raises an Exception if the arguments passed for Geology modules are
    unappropriate."""
    pass

class HintError(Exception): 
    """Raises an Exception if the hint proposed to shortcut the processing 
    isn't appropriate."""
    pass

class SQLError(Exception): 
    """Raises an Exception if the SQL request is unappropriate. """
    pass

class StrataError(Exception):
    """Raises an Exception if the value of stratum passes is wrong. Also, this 
    error occurs if the 'sname' is missing in the Hydro-log dataset while 
    it is mandatory to provide.'sname' is the name of column that fit the 
    strata. """
    pass

class SQLManagerError(Exception): 
    """Raises an Exception if the SQL request transfer failed to be 
    executed properly."""
    pass

class GeoDatabaseError(Exception): 
    """ Raises an Exception if the database failed to respond. The request is 
    aborted. """
    pass

class ERPError(Exception):
    """Raises an Exception if data passed is not a  valid Electrical Resistivity
    Profiling. Note that 'station' and 'resistivity' must figure out in the 
    ERP data. Station is the position where the measurement is done. By 
    convention is the center of MN and AB, the potential and current electrodes
    respectively."""
    pass

class VESError(Exception):
    """Raises an Exception if data passed is not a  valid Vertical Electrical 
    Sounding. Note that 'AB' and 'resistivity' must figure out in the VES data. 
    AB, the current electrodes position,  is by convention AB/2 """
    pass

class ExtractionError(Exception): 
    """Raises an Exception if value of extration in path-like object failed,
    or *.json ,*.yml or other files formats gives an expected result of data 
    extraction."""
    pass 

class EMError (Exception): 
    """Raises an Exception is EM method failed to run successfully or 
    a wrong argument passed to parameters computation failed """
    pass 

class FrequencyError (Exception): 
    """Raises an Exception for wrong given frequencies. `watex`_ recomputes 
    frequency when is needed and sorts frequency from highest to lowest."""
    pass 

class CoordinateError (Exception): 
    """ Raises an Exception for mismatched coordinates or if coordinates 
    failed to be recomputed properly. """
    pass 

class TopModuleError (Exception): 
    """ Raises an Exception if the top module failed to be installed. Note 
    that, understanding by top module, the most dependency package that 
    is used by `watex`_ for running successfully. For instance, `scikit_learn`_  
    is the top module for modeling and predition. """
    pass 
class NotFittedError (Exception): 
    """ Raise an Exception if the 'fit' method is not called yet. Note 
    most of `watex`_ classes implements 'fit' methods for attributes 
    populating and parameter init computations. Even the plotting 
    classes need also to be fitted. """
    pass 

class ScikitLearnImportError(Exception ): 
    """ Raises Exception if failed to import scikit-learn. Refer to 
    :class:`~.watex._docstring.scikit_learn_doc` for documentation.  
    Commonly, can get scikit-learn at https://scikit-learn.org/ ."""
    pass 

class GISError (Exception): 
    """ Raises an Exception if the GIS parameters failed to be calculated 
    successfully. 
    """
    pass 
class LearningError(Exception): 
    """ Raises an Exception if the learning Inspection failed during the 
    training phase. """
    pass 

class kError (Exception):
    """ Raises exception if the array of permeability coefficient is missing 
    or the 'kname' is not specified as the name of the column that fits the 
    permeability coefficient in the hydro-log data. """
    
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

class ZError (BaseException ):
    """ Raises an Exception if the Impedance Z tensor parameters failed to 
    be calculated properly. """
    pass 


