from __future__ import print_function 

__docformat__='restructuredtext'

import functools
import inspect
import os
import shutil

import datetime 
# import pandas as pd 
import numpy as np
from typing import Iterable, Optional  

from watex.utils._watexlog import watexlog
from watex.utils.__init__ import savepath as savePath 

__logger = watexlog().get_watex_logger(__name__)




class deprecated(object):
    """
        Description:
            used to mark functions, methods and classes deprecated, and prints 
            warning message when it called
            decorators based on https://stackoverflow.com/a/40301488

        Usage:
            todo: write usage

        Author: YingzhiGou
        Date: 20/06/2017
    """
    def __init__(self, reason):  # pragma: no cover
        if inspect.isclass(reason) or inspect.isfunction(reason):
            raise TypeError("Reason for deprecation must be supplied")
        self.reason = reason

    def __call__(self, cls_or_func):  # pragma: no cover
        if inspect.isfunction(cls_or_func):
            if hasattr(cls_or_func, 'func_code'):
                _code = cls_or_func.__code__
            else:
                _code = cls_or_func.__code__
            fmt = "Call to deprecated function or method {name} ({reason})."
            filename = _code.co_filename
            lineno = _code.co_firstlineno + 1

        elif inspect.isclass(cls_or_func):
            fmt = "Call to deprecated class {name} ({reason})."
            filename = cls_or_func.__module__
            lineno = 1

        else:
            raise TypeError(type(cls_or_func))

        msg = fmt.format(name=cls_or_func.__name__, reason=self.reason)

        @functools.wraps(cls_or_func)
        def new_func(*args, **kwargs):  # pragma: no cover
            import warnings
            warnings.simplefilter('always', DeprecationWarning)  # turn off filter
            warnings.warn_explicit(msg, category=DeprecationWarning, 
                                   filename=filename, lineno=lineno)
            warnings.simplefilter('default', DeprecationWarning)  # reset filter
            return cls_or_func(*args, **kwargs)

        return new_func


class gdal_data_check(object):
    _has_checked = False
    _gdal_data_found = False
    _logger = watexlog.get_watex_logger(__name__)

    def __init__(self, func, raise_error=False):
        """
        this decorator should only be used for the function that requres 
        gdal and gdal-data to function correctly.

        the decorator will check if the GDAL_DATA is set and the path
         in GDAL_DATA is exist. If GDAL_DATA is not set, then try to
         use external program "gdal-config --datadir" to
        findout where the data files are installed.

        If failed to find the data file, then ImportError will be raised.

        :param func: function to be decorated
        """
        self._func = func
        if not self._has_checked:
            self._gdal_data_found = self._check_gdal_data()
            self._has_checked = True
        if not self._gdal_data_found:
            if(raise_error):
                raise ImportError("GDAL  is NOT installed correctly")
            else:
                print ("Ignore GDAL as it is not working. Will use pyproj")

    def __call__(self, *args, **kwargs):  # pragma: no cover
        return self._func(*args, **kwargs)

    def _check_gdal_data(self):
        if 'GDAL_DATA' not in os.environ:
            # gdal data not defined, try to define
            from subprocess import Popen, PIPE
            self._logger.warning("GDAL_DATA environment variable is not set "
                                 " Please see https://trac.osgeo.org/gdal/wiki/FAQInstallationAndBuilding#HowtosetGDAL_DATAvariable ")
            try:
                # try to find out gdal_data path using gdal-config
                self._logger.info("Trying to find gdal-data path ...")
                process = Popen(['gdal-config', '--datadir'], stdout=PIPE)
                (output, err) = process.communicate()
                exit_code = process.wait()
                output = output.strip()
                if exit_code == 0 and os.path.exists(output):
                    os.environ['GDAL_DATA'] = output
                    self._logger.info("Found gdal-data path: {}".format(output))
                    return True
                else:
                    self._logger.error(
                        "\tCannot find gdal-data path. Please find the"
                        " gdal-data path of your installation and set it to"
                        "\"GDAL_DATA\" environment variable. Please see "
                        "https://trac.osgeo.org/gdal/wiki/FAQInstallationAndBuilding#HowtosetGDAL_DATAvariable for "
                        "more information.")
                    return False
            except Exception:
                return False
        else:
            if os.path.exists(os.environ['GDAL_DATA']):
                self._logger.info("GDAL_DATA is set to: {}".
                                  format(os.environ['GDAL_DATA']))

                try:
                    from osgeo import osr
                    from osgeo.ogr import OGRERR_NONE
                except:
                    self._logger.error("Failed to load module osgeo; "
                                       "looks like GDAL is NOT working")
                    # print ("Failed to load module osgeo !!! ")

                    return False
                # end try

                return True
            else:
                self._logger.error("GDAL_DATA is set to: {},"
                                   " but the path does not exist.".
                                   format(os.environ['GDAL_DATA']))
                return False

class redirect_cls_or_func(object) :
    """
        Description:
            used to redirected functions or classes. Deprecated functions 
            or class can call others use functions or classes.
            
        Usage:
            .. todo:: use new function or class to replace old function 
                method or class with multiple parameters.

        Author: @Daniel03
        Date: 18/10/2020
    """
    
    _logger = watexlog.get_watex_logger(__name__)
    
    def __init__(self, *args, **kwargs) :
        """
        self.new_func_or_cls is just a message of deprecating 
        warning . It could be a name of new function  to let user 
        tracking its code everytime he needs . 

        """
        
        self._reason=[func_or_reason for func_or_reason in args \
                      if type(func_or_reason)==str][0]
        if self._reason is None :
            
            raise TypeError(" Redirected reason must be supplied")
        

        self._new_func_or_cls = [func_or_reason for func_or_reason in \
                                 args if type(func_or_reason)!=str][0]

        if self._new_func_or_cls is None:
            raise Exception(
                " At least one argument must be a func_method_or class."
                            "\but it's %s."%type(self._new_func_or_cls))
            self._logger.warn("\t first input argument argument must"\
                              " be a func_method_or class."
                            "\but it's %s."%type(self._new_func_or_cls))
            

    def __call__(self, cls_or_func)  : #pragma :no cover

        if inspect.isfunction(self._new_func_or_cls) : 
            if hasattr(self._new_func_or_cls, 'func_code'):
                _code =self._new_func_or_cls.__code__
                lineno=_code.co_firstlineno+1
            else :
                # do it once the method is decorated method like staticmethods
                try:
                    _code =self._new_func_or_cls.__code__ 
                except : 
                    pass

            lineno=self._new_func_or_cls.__code__.co_firstlineno
            
            fmt="redirected decorated func/methods .<{reason}> "\
                "see line {lineno}."
            
        elif inspect.isclass(self._new_func_or_cls): 
            _code=self._new_func_or_cls.__module__
            # filename=os.path.basename(_code.co_filename)
            lineno= 1
            
            fmt="redirected decorated class :<{reason}> "\
                "see line {lineno}."
        else :
            # lineno=cls_or_func.__code__.co_firstlineno
            lineno= inspect.getframeinfo(inspect.currentframe())[1]
            fmt="redirected decorated method :<{reason}> "\
                "see line {lineno}."
        
        msg=fmt.format(reason = self._reason, lineno=lineno)
        # print(msg)
        self._logger.info(msg)
            #count variables : func.__code__.co_argscounts
            #find variables in function : func.__code__.co_varnames
        @functools.wraps(cls_or_func)
        def new_func (*args, **kwargs):
            
            return cls_or_func(*args, **kwargs)
        return self._new_func_or_cls
        

class writef(object): 
    """
    Description:
            used to redirected functions or classes. Deprecated functions 
            or class can call others use functions or classes.
            
        Usage:
            .. todo:: Decorate function or class to replace old function 
                method or class with multiple parameters and export files
                into many other format. `.xlsx` , `.csv` or regular format.

        Author: @Daniel03
        Date: 09/07/2021
        
    Decorator mainly focus to export data to other files. Exported file 
    can `regular` file or excel sheets. 
    
    
    :param reason: 
        Explain the "What to do?". Can be `write` or `convert`
    :param from_: 
        
         Can be ``df`` or ``regular``. If ``df``, `func` is called and collect 
         its input argguments and write to appropriate extension. If `from_`is 
         ``regular``, Can be a simple data put on list of string ready 
         to output file into other format. 
         
    :type from_: str ``df`` or ``regular`` 
    
    :param to_: 
                Exported file extension. Can be excel sheeet (`.xlsx`, `csv`)
                or other kind of format. 
                
    :param savepath: 
        Give the path to save the new file written. 
    
    """
    
    def __init__(self, reason=None,  from_=None,
                 to=None, savepath =None, **kws): 
        self._logging =watexlog().get_watex_logger(self.__class__.__name__)
        
        self.reason = reason 
        self.from_=from_ 
        self.to= to
        
        self.refout =kws.pop('refout', None)
        self.writedfIndex =kws.pop('writeindex', False)
        
        self.savepath =savepath 
        
        
        for key in list(kws.keys()): 
            setattr(self, key, kws[key])

    def __call__(self, func):
        """ Call function and return new function decorated"""
        
        @functools.wraps(func)
        def decorated_func(*args, **kwargs): 
            """
            New decorated function and holds `func` args and kwargs arguments.
            :params args: arguments of `func`
            :param kwargs: `keywords arguments of `func`. 
            
            """
            self._logging.info('Func <{}> decorated !'.format(func.__name__))
            
            cfw = 0     # write file type 
            
            for addf in ['savepath', 'filename']: 
                if not hasattr(self, addf): 
                    setattr(self, addf, None)
                    
            erp_time = '{0}_{1}'.format(datetime.datetime.now().date(), 
                            datetime.datetime.now().time())
            
            if self.refout is None : 
               self.refout = 'w-{0}'.format(
                   erp_time )
               
            if self.reason is None : 
                print('--> No reason is set. What do you want to do?'
                      ' `write` file or `convert` file into other format?.')
                return func(*args, **kwargs)
            
            if self.reason is not None : 
                if self.reason.lower().find('write')>=0 : 
                    cfw = 1 
                    if self.from_=='df': 
                        self.df , to_, refout_, savepath_, windex = func(*args,
                                                                 **kwargs)
                        fromdf =True
                        self.writedfIndex = windex
                         
            if fromdf is True and cfw ==1 : 
                if to_ is not None : 
                    self.to= '.'+ to_.replace('.','')
     
                else: 
                    self.to = '.csv'
                if refout_ is not None : 
                    self.refout =refout_
            
                self.refout = self.refout.replace(':','-') + self.to
                
                if savepath_ is not None: 
                    self.savepath =savepath_
                if self.to =='.csv': 
                    self.df.to_csv(self.refout, header=True,
                          index =self.writedfIndex)
                elif self.to =='.xlsx':
    
                    self.df.to_excel(self.refout , sheet_name='{0}'.format(
                        self.refout[: int(len(self.refout)/2)]),
                            index=self.writedfIndex) 
                             
                         
            # savepath 
            generatedfile = '_watex{}_'.format(
                    datetime.datetime.now().time()).replace(':', '.')
            if self.savepath is None :
                self.savepath = savePath(generatedfile)
            if self.savepath is not None :
                if not os.path.isdir(self.savepath): 
                    self.savepath = savePath(generatedfile)
                try : 
                    shutil.move(os.path.join(os.getcwd(),self.refout) ,
                            os.path.join(self.savepath , self.refout))
                except : 
                    self.logging.debug("We don't find any path to save file.")
                else: 
                    print(
                    '--> reference output  file <{0}> is well exported to {1}'.
                          format(self.refout, self.savepath))
                    
            return func(*args, **kwargs)
        return decorated_func 
        
    
def catmapflow(cat_classes: Iterable[str]=['FR0', 'FR1', 'FR2', 'FR3', 'FR4']): 
    """
    Decorator function  collected  from the `func`the `target_values` to be 
    categorized and the `cat_range_values` to change 
    into `cat_classes` like:: 
          
          cat_range_values= [0.0, [0.0, 3.0], [3.0, 6.0], [6.0, 10.0], 10.0]
          target_values =[1, 2., 3., 6., 7., 9., 15., 25, ...]
          
    Decorated Fonction returns the  new function decorated holding  
    values  categorized into categorial `cat_classes`.
    For instance in groundwater exploration::
        
        - FR0 --> `flow` is equal to ``0.``m3/h
        - FR1 --> `flow` is ``0 < FR ≤ 3`` m3/h
        - FR2 --> `flow` is ``3 < FR ≤ 6`` m3/h 
        - FR3 --> `flow` is ``6 < FR ≤ 10`` m3/h
        - FR4 --> `flow` is ``10.+`` in m3/h

    :return: Iterable object with new categorized values converted 
    into `cat_classes`. 
    
    Author: @Daniel03
    Date: 13/07/2021
    """

    def categorized_dec(func):
        """
        Decorator can be adapted  to other categorized problem by changing the 
        `cat_classes` arguments to another categorized classes 
        for other purposes like ::
            
         cat_classes=['dry', 'HV', 'IHV', 'IVH+', 'UH']   
            
        Where ``IVHU`` means I:mproved V:Village H:Hydraulic and U:Urban. 
        
        :Note: 
            If `func` to be decorated contains ` cat_classes` arguments, 
            the `cat_classes` argument should be erased by the given
            from `func`. 
        
        """
        @functools.wraps(func)
        def  wrapper(*args, **kwargs): 
            """
            Function deals with the categorized flow values. 
            
            :param args: positional argumnent of `func`
            :param kwargs: Optional argument of `func`
    
            :return `new_target_array`: Iterable object categorized.
            """
            def mapf(crval,  nfval, fc):
                """
                Categorizing loop to hold the convenient classes according 
                to the `cat_range_value` provided. Come as supplement 
                tools when ``maping`` object doesnt work properly.
                
                :param crval: value to be categorized 
                :param nfval: array of `cat_range_values`
                :param fc: Object to replace the`crval` belonging 
                    to `cat_classes`
                """
                for ii, val in enumerate(nfval):
                    try : 
                        if len(val)>1: 
                            if  val[0] < crval <= val[-1] : 
                                return fc[ii]
                    except : 
                        if crval ==0.: 
                            return fc[0]
                        elif crval>= nfval[-1] : 
                            return fc[-1]
      
            cat_range_values, target_array, catfc = func(*args, **kwargs)
                
            if len(cat_range_values) != len(cat_classes): 
                __logger.error(
                    'Length of `cat_range_values` and `cat_classes` provided '
                    'must be the same length not ``{0}`` and'
                    ' ``{1}`` respectively.'.format(len(cat_range_values),
                                                    len(cat_classes)))
            try : 
                
                new_target_array = np.array(list(map( mapf, target_array)))
            except : 
                
                new_target_array = np.zeros_like(target_array)
                for ii, ff in enumerate(target_array) : 
                    new_target_array[ii] = mapf(crval=ff, 
                                            nfval=cat_range_values, 
                                           fc=cat_classes)
            return new_target_array
        return wrapper  
    return  categorized_dec
    


    







































            

        
            
    