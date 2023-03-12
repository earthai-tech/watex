# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio alias @Daniel <etanoyau@gmail.com>

from __future__ import print_function 
import functools
import inspect
import os
import sys
import copy 
import shutil
import warnings
import datetime 
from contextlib import contextmanager
import numpy as np
import  matplotlib.pyplot as plt 
import matplotlib as mpl 
import matplotlib.cm as cm 
import matplotlib.colorbar as mplcb

from ._typing import (
    Iterable,
    Optional,
    Callable,
    T,
    F
)
from ._watexlog import watexlog
_logger = watexlog.get_watex_logger(__name__)

__docformat__='restructuredtext'

      
class temp2d: 
    """ Two dimensional plot template 
    
    Parameters 
    ----------
    reason: str, Any 
        Does nothing. But if supplied, it should be the purpose of the 
        plot. 
    
    Note
    ------
    For customizing the plot, `_temp2d` uses at the last parameter of 
    the function to be decorated, the plotting arguments from 
    :class:`watex.property.BasePlot` parameters. If not given, an 
    atttribute errors will raise.

    """
    def __init__(self, reason =None, **kws):
        self.reason=reason 
    def __call__(self, func) : 
        self._func =func 
        
        @functools.wraps(self._func ) 
        def new_func (*args, **kwargs ):
            
            _args = self. _func (*args, **kwargs) 
            base_plot_kws = _args[-1]
            for key in base_plot_kws.keys () :
                # if attribute exist arase it 
                # if ( key in self.__dict__.keys() 
                #     and base_plot_kws[key] is not None
                #     ): 
                #     self.__dict__[key] = base_plot_kws[key] 
                # else:
                setattr (self, key , base_plot_kws[key] )
            return self.plot2d(*_args[:-1] )
        
        return new_func  
    
    def __getattr__(self, name): 
        msg = ("{0!r} has no attribute {1!r}. Note that {0!r} uses the"
               " plot arguments from `watex.property.BasePlot`. Plot arguments"
               " must be supplied as a keyword argument at the last parameters"
               " i.e the last value of return (output) of the function to be"
               " decorated."
               )
        raise AttributeError (msg.format(self.__class__.__name__, name))
        
    def plot2d (self, arr2d, y=None , x=None,posix =None) :
        """ Template for 2D plot. Basically if use the stations and positions 
        as `xlabel` and `positions` i explicitly both are not supplied. 
        
        Parameters 
        ------------
        arr2d : ndarray , shape (N, M) 
            2D array for plotting. For instance, it can be a 2D resistivity 
            collected at all stations (N) and all frequency (M) 
        y: array-like 
            Y-coordinates. It should have the length N, the same of the ``arr2d``.
            the rows of the ``arr2d``.
        x: array-like 
            X-coordinates. It should have the length M, the same of the ``arr2d``; 
            the columns of the 2D dimensional array.  Note that if `x` is 
            given, the `distance is not needed. 
            
        posix: list of str 
            List of stations names. If given,  it should have the same length of 
            the columns M, of `arr2d`` 
        
        Returns 
        -------
        axe: Matplotllib axis 
        
        """
        def _format_ticks (value, tick_number, fmt ='S{:02}', nskip =3 ):
            """ Format thick parameter with 'FuncFormatter(func)'
            rather than using `axi.xaxis.set_major_locator (plt.MaxNLocator(3))`
            ax.xaxis.set_major_formatter (plt.FuncFormatter(format_thicks))
            
            :param value: tick range values for formatting 
            :param tick_number: number of ticks to format 
            :param fmt: str, default='S{:02}', kind of tick formatage 
            :param nskip: int, default =7, number of tick to skip 
            
            """
            if value % nskip==0: 
                return fmt.format(int(value)+ 1)
            else: None
            
        fig, axe = plt.subplots(
            1, 
            figsize = self.fig_size, 
            num = self.fig_num,
            dpi = self.fig_dpi, 
                    )
        cmap = plt.get_cmap( self.cmap)
        
        if self.plt_style =='pcolormesh': 
            X, Y = np.meshgrid (x, y)
            axr = axe.pcolormesh ( X, Y, arr2d,
                            # for consistency check whether array does not 
                            # contain any NaN values 
                            vmax = arr2d[ ~np.isnan(arr2d)].max(), 
                            vmin = arr2d[ ~np.isnan(arr2d)].min(), 
                            shading= 'gouraud', 
                            cmap =cmap, 
                                  )

        if  self.plt_style =='imshow': 
            axr= axe.imshow (arr2d,
                            interpolation = self.imshow_interp, 
                            cmap =cmap,
                            aspect = self.fig_aspect ,
                            origin= 'upper', 
                            extent=( x[~np.isnan(x)].min(),
                                      x[~np.isnan(x)].max(), 
                                      y[~np.isnan(y)].min(), 
                                      y[~np.isnan(y)].max())
                                              )
            axe.set_ylim(y[~np.isnan(y)].min(), y[~np.isnan(y)].max())
        
        axe.set_xlabel(self.xlabel or 'Distance(m)', 
                     fontdict ={
                      'size': self.font_size ,
                      'weight': self.font_weight} )
      
        axe.set_ylabel(self.ylabel or 'log10(Frequency)[Hz]',
                 fontdict ={'size': self.font_size ,
                                  'weight': self.font_weight})

        axe.tick_params (axis ='both', labelsize = self.font_size 
                         )
        
        if self.show_grid: 
            axe.minorticks_on()
            # axe.grid(color='k', ls=':', lw =0.25, alpha=0.7, 
            #              which ='major')
            axe.grid(
                color= self.gc, 
                ls=self.gls, 
                lw =self.glw, 
                alpha=self.galpha,  
                which =self.gwhich
          )
            
        labex , cf = self.cb_label or '$log10(App.Res)[Ω.m]$', axr
    
        cb = fig.colorbar(cf , ax= axe)
        cb.ax.yaxis.tick_left()
        cb.ax.tick_params(axis='y', direction='in', pad=2.,
                          labelsize = self.font_size 
                          )
        
        cb.set_label(labex,fontdict={'size': 1.5 * self.font_size ,
                                  'style':self.font_style})
        #--> set second axis 
        axe2 = axe.twiny() 
        axe2.set_xticks(range(len(x)), minor=False, 
                        fontsize = self.font_size 
                        )
        #axe2.set_xticks(range(len(x)),minor=False )

        if len(x ) >= 12 : 
            axe2.xaxis.set_major_formatter (plt.FuncFormatter(_format_ticks))
        else : 
            axe2.set_xticklabels(posix, rotation=self.rotate_xlabel, 
                             fontsize = self.font_size )
     
        axe2.set_xlabel('Stations', 
                        fontdict ={'style': self.font_style, 
                                   'size': 1.5 * self.font_size ,
                                   'weight': self.font_weight},
                        )
        fig.suptitle(self.fig_title,
                     ha='left',
                     fontsize= 15* self.fs, 
                     verticalalignment='center', 
                    style =self.font_style,
                    bbox =dict(boxstyle='round',
                               facecolor ='moccasin'))

        plt.tight_layout()  
        if self.savefig is not None :
            fig.savefig(self.savefig, dpi = self.fig_dpi,
                        orientation =self.orient)
        plt.show() if self.savefig is None else plt.close(fig=fig) 
        
        return axe 


class donothing : 
    """ Decorator to do nothing. Just return the func as it was. 
    The `param` reason is just used to specify the skipping reason. """
    def __init__(self, reason = None ):
        self.reason = reason 
        
    def __call__(self, cls_or_func) :
        @functools.wraps (cls_or_func)
        def new_func (*args, **kwargs): 
            return cls_or_func (*args, **kwargs)
        return new_func 
    
class refAppender (object): 
    """ Append the module docstring with reStructured Text references. 
    
    Indeed, when a `func` is decorated, it will add the reStructured Text 
    references as an appender to its reference docstring. So, sphinx 
    can auto-retrieve some replacing values found inline  from the 
    :doc:`watex.documentation`. 

    Parameters
    ----------
    docref: str 
        Reference of the documentation for appending.
        
    .. |VES| replace:: Vertical Electrical Sounding 
    .. |ERP| replace:: Electrical Resistivity Profiling          
    
    Examples
    ---------
    >>> from watex.documentation import __doc__ 
    >>> from watex.tools import decorators
    >>> def donothing (): 
            ''' Im here to just replace the `|VES|` and `|RES|` values by their
            real meanings.'''
            pass 
    >>> decorated_donothing = decorators.refAppender(__doc__)(donothing) 
    >>> decorated_donothing.__doc__ 
    ... #new doctring appended and `|VES|` and `|ERP|` are replaced by 
    ... #Vertical Electrical Sounding and Electrical resistivity profiling 
    ... #during compilation in ReadTheDocs.

    """
    
    def __init__(self, docref= None ): 
        self.docref = docref 

    def __call__(self, cls_or_func): 
        return self.nfunc (cls_or_func)
    def nfunc (self, f):
        f.__doc__ += "\n" + self.docref or '' 
        setattr(f , '__doc__', f.__doc__)
        return  f 
  
class deprecated(object):
    """
    Used to mark functions, methods and classes deprecated, and prints 
    warning message when it called
    decorators based on https://stackoverflow.com/a/40301488 .

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
    _gdal_data_variable_resources = 'https://trac.osgeo.org/gdal/wiki/FAQInstallationAndBuilding#HowtosetGDAL_DATAvariable '
    _gdal_wheel_resources ='https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal'
    _gdal_installation_guide = 'https://opensourceoptions.com/blog/how-to-install-gdal-for-python-with-pip-on-windows/'


    def __init__(self, func, raise_error=False, verbose = 0):
        """
        The decorator should only be used for the function that requires 
        gdal and gdal-data correctly.

        GDAL standas for Geospatial Data Abstraction Library. 
        It is a translator library for raster geospatial data formats.
        Its distribution includes a complete GDAL installation
        It will check whether the GDAL_DATA is set and the path
         in GDAL_DATA exists. If GDAL_DATA is not set, then try to
         use external program "gdal-config --datadir" to
        findout where the data files are installed.

        If failed to find the data file, then ImportError will be raised.

        :param func: function to be decorated
        
        """
  
        self._func = func
        self.verbose= verbose 
        if not self._has_checked:
            self._gdal_data_found = self._check_gdal_data()
            self._has_checked = True
        if not self._gdal_data_found:
            if(raise_error):
                raise ImportError(
                    "GDAL  is NOT installed correctly. "
                    f"GDAL wheel can be downloaded from {self._gdal_wheel_resources}"
                    " and use `pip install <path-to-wheel-file.whl>`"
                    "for installing. Get more details here: "
                    f" {self._gdal_installation_guide}."
                                  )
            else:
                pass 
            
    def __call__(self, *args, **kwargs):  # pragma: no cover
        return self._func(*args, **kwargs)

    def _check_gdal_data(self):
        if 'GDAL_DATA' not in os.environ:
            # gdal data not defined, try to define
            from subprocess import Popen, PIPE 
            if self.verbose : 
                _logger.warning("GDAL_DATA environment variable is not set "
                                f" Please see {self._gdal_data_variable_resources}")
            try:
                # try to find out gdal_data path using gdal-config
                if self.verbose: 
                    _logger.info("Trying to find gdal-data path ...")
                process = Popen(['gdal-config', '--datadir'], stdout=PIPE)
                (output, err) = process.communicate()
                exit_code = process.wait()
                output = output.strip()
                if exit_code == 0 and os.path.exists(output):
                    os.environ['GDAL_DATA'] = output
                    _logger.info("Found gdal-data path: {}".format(output))
                    return True
                else:
                    _logger.error(
                        "\tCannot find gdal-data path. Please find the"
                        " gdal-data path of your installation and set it to"
                        "\"GDAL_DATA\" environment variable. Please see "
                        f"{self._gdal_data_variable_resources} for "
                        "more information.")
                    return False
            except Exception:
                return False
        else:
            if os.path.exists(os.environ['GDAL_DATA']):
                if self.verbose: 
                    _logger.info("GDAL_DATA is set to: {}".
                                      format(os.environ['GDAL_DATA']))

                try:
                    from .utils._dependency import import_optional_dependency
                    import_optional_dependency ('osgeo')
                    # from osgeo import osr
                    # from osgeo.ogr import OGRERR_NONE
                except: # if failed to import GDAl 
                    return False
                return True
            else:
                if self.verbose: _logger.error("GDAL_DATA is set to: {},"
                                   " but the path does not exist.".
                                   format(os.environ['GDAL_DATA']))
                return False

class redirect_cls_or_func(object) :
    """Used to redirected functions or classes. Deprecated functions  or class 
    can call others use functions or classes.
    
    Use new function or class to replace old function method or class with 
    multiple parameters.

    Author: LKouadio~@Daniel03
    Date: 18/10/2020
    
    """
    def __init__(self, *args, **kwargs) :
        """
        self.new_func_or_cls is just a message of deprecating 
        warning . It could be a name of new function  to let user 
        tracking its code everytime he needs . 

        """
        
        self._reason=[func_or_reason for func_or_reason in args
                      if type(func_or_reason)==str][0]
        if self._reason is None :
            
            raise TypeError(" Redirected reason must be supplied")
        

        self._new_func_or_cls = [func_or_reason for func_or_reason in 
                                 args if type(func_or_reason)!=str][0]

        if self._new_func_or_cls is None:
            raise Exception(
                " At least one argument must be a func_method_or class."
                            "\but it's %s."%type(self._new_func_or_cls))
            _logger.warn("\t first input argument argument must"
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
        _logger.info(msg)
            #count variables : func.__code__.co_argscounts
            #find variables in function : func.__code__.co_varnames
        @functools.wraps(cls_or_func)
        def new_func (*args, **kwargs):
            
            return cls_or_func(*args, **kwargs)
        return self._new_func_or_cls
        
class writef2(object): 
    """
    Used to redirected functions or classes. Deprecated functions  or class can
    call others use functions or classes.
             
    Decorate function or class to replace old function method or class with 
    multiple parameters and export files into many other format. `.xlsx` ,
    `.csv` or regular format. Decorator mainly focus to export data to other
    files. Exported file can `regular` file or excel sheets. 
    
    :param reason: 
        Explain the "What to do?". Can be `write` or `convert`.
        
    :param from_: 
        Can be ``df`` or ``regular``. If ``df``, `func` is called and collect 
        its input arguments and write to appropriate extension. If `from_`is 
        ``regular``, Can be a simple data put on list of string ready 
        to output file into other format. 
    :type from_: str ``df`` or ``regular`` 
    
    :param to_: 
        Exported file extension. Can be excel sheeet (`.xlsx`, `csv`)
        or other kind of format. 
            
    :param savepath: 
        Give the path to save the new file written.
        
    *Author: LKouadio ~ @Daniel03*
    *Date: 09/07/2021*
        
    """
    
    def __init__(
        self, 
        reason:Optional[str]=None,  
        from_:Optional[str]=None,
        to:Optional[str]=None, 
        savepath:Optional[str] =None, 
        **kws
        ): 
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
            :params args: positional arguments of `func`
            :param kwargs: keywords arguments of `func`. 
            
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
                      ' `write` file or `convert` file into other format?')
                return func(*args, **kwargs)
            
            if self.reason is not None : 
                if self.reason.lower().find('write')>=0 : 
                    cfw = 1 
                    if self.from_=='df': 
                        self.df , to_, refout_, savepath_, windex = func(*args,
                                                                 **kwargs)
                        fromdf =True
                        self.writedfIndex = windex
                         
            if fromdf  and cfw ==1 : 
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
                self.savepath = savepath_(generatedfile)
            if self.savepath is not None :
                if not os.path.isdir(self.savepath): 
                    self.savepath = savepath_(generatedfile)
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
        
class writef(object): 
    """
    Used to redirected functions or classes. Deprecated functions  or class can
    call others use functions or classes.
             
    Decorate function or class to replace old function method or class with 
    multiple parameters and export files into many other format. `.xlsx` ,
    `.csv` or regular format. Decorator mainly focus to export data to other
    files. Exported file can `regular` file or excel sheets. 
    
    :param reason: 
        Explain the "What to do?". Can be `write` or `convert`.
        
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
        
    *Author: LKouadio ~ @Daniel03*
    *Date: 09/07/2021*
        
    """
    
    def __init__(
        self, 
        reason:Optional[str]=None,  
        from_:Optional[str]=None,
        to:Optional[str]=None, 
        savepath:Optional[str] =None, 
        **kws
        ): 
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
            :params args: positional arguments of `func`
            :param kwargs: keywords arguments of `func`. 
            
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
                      ' `write` file or `convert` file into other format?')
                return func(*args, **kwargs)
            
            if self.reason is not None : 
                if self.reason.lower().find('write')>=0 : 
                    cfw = 1 
                    if self.from_=='df': 
                        self.df , to_, refout_, savepath_, windex = func(*args,
                                                                 **kwargs)
                        fromdf =True
                        self.writedfIndex = windex
                         
            if fromdf  and cfw ==1 : 
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
                self.savepath = savepath_(generatedfile)
            if self.savepath is not None :
                if not os.path.isdir(self.savepath): 
                    self.savepath = savepath_(generatedfile)
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
        
@deprecated('Replaced by :class:`watex.utils.decorators.catmapflow2`')   
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
    
    Author: LKouadio ~ @Daniel03
    Date: 13/07/2021
    """

    def categorized_dec(func):
        """
        Decorator can be adapted  to other categorized problem by changing the 
        `cat_classes` arguments to another categorized classes 
        for other purposes like ::
            
         cat_classes=['dry', 'HV', 'IHV', 'IVH+', 'UH']   
            
        Where ``IVHU`` means I:improved V:village H:hydraulic and U:urban. 
        
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
            cat_range_values, target_array, catfc = func(*args, **kwargs)
            
            if catfc is not None : 
                cat_classes = catfc
            # else:
            #     cat_classes: Iterable[str]=['FR0', 'FR1', 'FR2', 'FR3', 'FR4']
            
            def mapf(crval,  nfval=cat_range_values , fc=cat_classes):
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
                        if isinstance(val, (float, int)): 
                            if crval ==nfval[0]: 
                                return fc[0]
                            elif crval>= nfval[-1] : 
                                return fc[-1]
                        elif isinstance(val, (list, tuple)):
                            if len(val)>1: 
                                if  val[0] < crval <= val[-1] : 
                                    return fc[ii]
                    except : 
                        
                        if crval ==0.: 
                            return fc[0]
                        elif crval>= nfval[-1] : 
                            return fc[-1]
       
            if len(cat_range_values) != len(cat_classes): 
                
                _logger.error(
                    'Length of `cat_range_values` and `cat_classes` provided '
                    'must be the same length not ``{0}`` and'
                    ' ``{1}`` respectively.'.format(len(cat_range_values),
                                                    len(cat_classes)))
            try : 

                new_target_array = np.array(list(map( mapf, target_array)))
                # new_target_array = np.apply_along_axis(
                #     lambda if_: mapf(crval= if_),0, target_array)
            except : 
                new_target_array = np.zeros_like(target_array)
                for ii, ff in enumerate(target_array) : 
                    new_target_array[ii] = mapf(crval=ff, 
                                            nfval=cat_range_values, 
                                            fc=cat_classes)
            return new_target_array
        return wrapper  
    return  categorized_dec
    

class visualize_valearn_curve : 
    """
    Decorator to visualize the validation curve and learning curve 
    Once called, will  quick plot the `validation curve`.
            
    Quick plot the validation curve 
        
    :param reason: what_going_there? validation cure or learning curve.
        - ``val`` for validation curve 
        -``learn`` for learning curve 
    :param turn:  Continue the plotting or switch off the plot and return 
        the function. default is `off` else `on`.
    :param kwargs: 
        Could be the keywords arguments for `matplotlib.pyplot` library:: 
            
            train_kws={c:'r', s:10, marker:'s', alpha :0.5}
            val_kws= {c:'blue', s:10, marker:'h', alpha :1}

    """
    
    def __init__(self, reason ='valcurve', turn ='off', **kwargs): 
        self.reason =reason
        self.turn =turn 
        self.fig_size =kwargs.pop('fig_size', (8,6))
        self.font_size =kwargs.pop('font_size', 18.)
        self.plotStyle =kwargs.pop('plot_style', 'scatter')
        self.train_kws=kwargs.pop('train_kws',{'c':'r',  'marker':'s', 
                                               'alpha' :0.5,
                                               'label':'Training curve'})
        self.val_kws= kwargs.pop('val_kws', {'c':'blue', 'marker':'h','alpha' :1,
                   'label':'Validation curve' })
        self.k = kwargs.pop('k', np.arange(1, 220, 20))
        self.xlabel =kwargs.pop('xlabel', {'xlabel':'Evaluation of parameter', 
                                           'fontsize': self.font_size}
                                )
        self.ylabel =kwargs.pop('ylabel', {'ylabel':'Performance in %', 
                                           'fontsize': self.font_size}
                                )
        self.savefig =kwargs.pop('savefig', None)
        self.grid_kws = kwargs.pop('grid_kws', {
                       'galpha' :0.2,              # grid alpha 
                       'glw':.5,                   # grid line width 
                       'gwhich' :'major',          # minor ticks
                        })
        self.show_grid = kwargs.pop('show_grid', False)
        self.error_plot =False 
        self.scatterplot = True 
        self.lineplot =False

        if self.plotStyle.lower()=='both': 
            self.lineplot = True 
        elif self.plotStyle.lower().find('line')>=0 : 
            self.lineplot = True 
            self.scatterplot =False
        elif self.plotStyle =='scatter': 
            self.scatterplot =True 
        
        for key in kwargs.keys(): 
            setattr(self, key, kwargs[key])
            
    def __call__(self, func): 
        """ Call function and decorate `validation curve`"""
        
        @functools.wraps(func) 
        def viz_val_decorated(*args, **kwargs): 
            """ Decorated function for vizualization """

            if self.reason.lower().find('val')>=0: 
                self.reason ='val'
                train_score , val_score, switch, param_range,\
                     pname, val_kws, train_kws =func(*args, **kwargs)
                    
            elif self.reason.lower().find('learn')>=0: 
                self.reason ='learn'
                param_range,  train_score , val_score, switch,\
                         pname, val_kws, train_kws=func(*args, **kwargs)
                    
            if val_kws is not None : 
                self.val_kws =val_kws 
            if train_kws is not None: 
                self.train_kws = train_kws
                
            # add the name of parameters.
            if pname  !='': 
                self.xlabel = {'xlabel':'Evaluation of parameter %s'%pname , 
                                'fontsize': self.font_size}

            if switch  is not None :
                self.turn =switch 
                
            # if param_range is not None :
            #     k= param_range 
                
            plt.figure(figsize=self.fig_size)

            if self.turn in ['on', 1, True]: 
                # if not isinstance(param_range, bool): 
                self._plot_train_val_score(train_score, val_score, k=param_range )
      
                if self.savefig is not None: 
                    if isinstance(self.savefig, dict):
                        plt.savefig(**self.savefig)
                    else : 
                        plt.savefig(self.savefig)
                        
            # initialize the trainscore_dict  
            train_score=dict()  
            return func(*args, **kwargs)
        
        return viz_val_decorated
    
    def _plot_train_val_score (self, train_score, val_score, k): 
        """ loop to plot the train val score""" 
        if not isinstance(train_score, dict):
            train_score={'_':train_score}
            val_score = {'_': val_score}

        for trainkey, trainval in train_score.items(): 
            if self.reason !='learn':
                trainval*=100
                val_score[trainkey] *=100 
            try: 
                if self.scatterplot: 
                    plt.scatter(k,
                                val_score[trainkey].mean(axis=1),
                                **self.val_kws 
                                )
                    plt.scatter(k,
                                trainval.mean(axis=1) ,
                                **self.train_kws
                           )
            except : 
                # if exception occurs maybe from matplotlib properties 
                # then run the line plot 
                plt.plot(k, 
                         val_score[trainkey].mean(axis=1),
                         **self.val_kws
                         )

                plt.plot(k,
                         trainval.mean(axis=1),
                         **self.train_kws
                         )
            try : 
                if self.lineplot : 
                    plt.plot(k, 
                             val_score[trainkey].mean(axis=1),
                             **self.val_kws
                             )

                    plt.plot(k, 
                             trainval.mean(axis=1),
                             **self.train_kws
                             )
            except : 
            
                plt.scatter(k, val_score[trainkey].mean(axis=1),
                            **self.val_kws 
                            )
                plt.scatter(k,
                            trainval.mean(axis=1) ,
                       **self.train_kws
                       )
                
            
        if isinstance(self.xlabel, dict):
            plt.xlabel(**self.xlabel)
        else :  plt.xlabel(self.xlabel)
        
        if isinstance(self.ylabel, dict):
            plt.ylabel(**self.ylabel)
        else :  plt.ylabel(self.ylabel)
        
        plt.tick_params(axis='both', 
              labelsize= self.font_size )
        
        if self.show_grid is True:
            plt.grid(self.show_grid, **self.grid_kws
                    )
        
        plt.legend()
        plt.show()
                
        
                
class predplot: 
    """ 
    Decorator to plot the prediction.
     
    Once called, will  quick plot the `prediction`. Quick plot the prediction 
    model. Can be customize using the multiples keywargs arguments.
         
    :param turn:  Continue the plotting or switch off the plot and return 
        the function. default is `off` else `on`.
    :param kws: 
        Could be the keywords arguments for `matplotlib.pyplot`
        library 
                
    Author: LKouadio alias @Daniel
    Date: 23/07/2021
    """
    def __init__(self, turn='off', **kws): 

        self.turn =turn 
        self.fig_size =kws.pop('fig_size', (16,8))
        self.yPred_kws = kws.pop('ypred_kws',{'c':'r', 's':200, 'alpha' :1,
                                        'label':'Predicted flow:y_pred'})
        self.yObs_kws = kws.pop('ypred_kws',{'c':'blue', 's':100, 'alpha' :0.8,
                                        'label':'Observed flow:y_true'})
        self.tick_params =kws.pop('tick_params', {'axis':'x','labelsize':10, 
                                                  'labelrotation':90})
        self.xlab = kws.pop('xlabel', {'xlabel':'Boreholes tested'})
        self.ylab= kws.pop('ylabel', {'ylabel':'Flow rates(FR) classes'})
        self.obs_line=kws.pop('ObsLine', None)
        self.l_kws=kws.pop('l_', {'c':'blue', 'ls':'--', 'lw':1, 'alpha':0.5})
        self.savefig =kws.pop('savefig', None)
        if self.obs_line is None : 
            self.obs_line = ('off', 'Obs')
            
    
    def __call__(self, func:Callable[..., T]):
        """ Call the function to be decorated """
        
        @functools.wraps(func)
        def pred_decorated(*args, **kwargs): 
            """Function to be decorated"""
    
            y_true,  y_pred, switch = func(*args, **kwargs)
  
            if switch is None : self.turn ='off'
            if switch is not None : self.turn =switch
            
            if self.turn == ('on' or True or 1):
                
                plt.figure(figsize=self.fig_size)
                
                plt.scatter(y_pred.index, y_pred,**self.yPred_kws )
                plt.scatter(y_true.index,y_true, **self.yObs_kws )
                
                if self.obs_line[0] == ('on' or True or 1): 
                    if self.obs_line[1].lower().find('true') >=0 or\
                        self.obs_line[1].lower()=='obs': 
                        plt.plot(y_true, **self.l_kws)
                    elif self.obs_line[1].lower().find('pred') >=0:
                        plt.plot(y_pred, **self.l_kws)
                        
                # plt.xticks(rotation = 'vertical')
                plt.tick_params(**self.tick_params)
                plt.xlabel(**self.xlab)
                plt.ylabel(**self.ylab)
                plt.legend()
                
                if self.savefig is not None: 
                        if isinstance(self.savefig, dict):
                            plt.savefig(**self.savefig)
                        else : 
                            plt.savefig(self.savefig)
                            
            return func(*args, **kwargs)
        return pred_decorated


class pfi: 
    """ 
    Decorator to plot Permutation future importance. 
    
    Can also plot dendrogram figure by setting `reason` to 'dendro`.  Quick 
    plot the permutation  importance diagram. Can be customize using the 
    multiples keywargs arguments.
                    
    :param reason: what_going_there? validation curve or learning curve.
                    - ``pfi`` for permutation feature importance before
                        and after sguffling trees  
                    -``dendro`` for dendrogram plot  
    :param turn:  Continue the plotting or switch off the plot and return 
                the function. default is `off` else `on`.
    :param kws: 
        Could be the keywords arguments for `matplotlib.pyplot`
        library.
        
    :param barh_kws: matplotlib.pyplot.barh keywords arguments. 
        Refer to https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html
        
    :param box_kws: :ref:`plt.boxplot` keyword arguments.
        Refer to <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html>` 
    :param dendro_kws: scipy.cluster.hierarchy.dendrogram diagram 
    
    .. see also:: `<https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html>`
   
    """
    def __init__(self, reason ='pfi', turn ='off', **kwargs): 
        self._logging=watexlog().get_watex_logger(self.__class__.__name__)
        
        self.reason = reason 
        self.turn = turn 
        
        self.fig_size = kwargs.pop('fig_size',(9, 3) )
        self.savefig = kwargs.pop('savefig', None)
        
        self.xlab = kwargs.pop('xlabel', {'xlabel':'Importance'})
        self.ylab= kwargs.pop('ylabel', {'ylabel':'Features'})
        self.barh_kws=kwargs.pop('barh_kws', {'color':'blue',
                                              'edgecolor':'k', 'linewidth':2})
        self.box_kws=kwargs.pop('box_kws', {'vert':False, 'patch_artist':False})
        self.dendro_kws=kwargs.pop('dendro_kws',{'leaf_rotation':90,
                                                 # 'orientation':'right'
                                                 } )
        self.fig_title =kwargs.pop('fig_title', 'matplotlib.axes.Axes.barh Example')
        
    def __call__(self, func:Callable[..., T]): 
 
        @functools.wraps(func)
        def feat_importance_dec (*args, **kwargs): 
            """ Decorated pfi and dendrogram diagram """
            
            X, result, tree_indices, clf, tree_importance_sorted_idx,\
            data_columns, perm_sorted_idx, pfi_type, switch, savefig =func(
                *args, **kwargs)
            
            if pfi_type is not None : self.reason = pfi_type
            if switch is not None : self.turn = switch 
            if savefig is not None: self.savefig = savefig 
            
            if self.turn ==('on' or True or 1): 
                fig, axes = plt.subplots(1, 2,figsize=self.fig_size)
                    
                self._plot_barh_or_spearman(
                    X, clf, func, fig, axes, self.reason,
                    tree_indices, tree_importance_sorted_idx, data_columns, \
                        result , perm_sorted_idx, **kwargs)   

                plt.show()   
                    
                if self.savefig is not None: 
                    if isinstance(self.savefig, dict):
                        plt.savefig(**self.savefig)
                    else : 
                        plt.savefig(self.savefig)

            return func(*args, **kwargs)
        
        return  feat_importance_dec
    
    
    def _plot_barh_or_spearman (self, X, clf, func,  fig, axes , reason,
                                *args, **kwargs): 
        """ Plot bar histogram and spearmean """
        
        ax1, ax2 = axes 
        
        tree_indices, tree_importance_sorted_idx, data_columns, \
            result , perm_sorted_idx = args 
        
        if reason == 'pfi': 
            
            ax1.barh(tree_indices,
                     clf.feature_importances_[tree_importance_sorted_idx] *100,
                     height=0.7, **self.barh_kws)
            ax1.set_yticklabels(data_columns[tree_importance_sorted_idx])
            ax1.set_yticks(tree_indices)
            ax1.set_ylim((0, len(clf.feature_importances_)))
            ax2.boxplot(result.importances[perm_sorted_idx].T *100,
                        labels=data_columns[perm_sorted_idx], **self.box_kws)
            
            ax1.set_xlabel(**{k:v +' before shuffling (%)' 
                              for k, v in self.xlab.items()} )
            ax1.set_ylabel(**self.ylab)
            ax2.set_xlabel(**{k:v +' after shuffling (%)' 
                              for k, v in self.xlab.items()} )
            try : 
                
                ax1.set_title(self.fig_title + ' using '+\
                              clf.__class__.__name__)
            except : 
                ax1.set_title(self.fig_title)

            fig.tight_layout()
            
        if reason == 'dendro': 
            from scipy.stats import spearmanr
            from scipy.cluster import hierarchy
            
            if X is None : 
                self._logging.debug('Please provide the train features !')
                warnings.warn(
                    ' Parameter `X` is missing. '
                    ' Could not plot the dendromarc diagram')
                return func(*args, **kwargs)
            
            elif X is not None: 

                corr = spearmanr(X).correlation *100
                corr_linkage = hierarchy.ward(corr)
                dendro = hierarchy.dendrogram(corr_linkage,
                                        labels=data_columns, ax=ax1,
                                        **self.dendro_kws)
                dendro_idx = np.arange(0, len(dendro['ivl']))
                
                ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
                ax2.set_xticks(dendro_idx)
                ax2.set_yticks(dendro_idx)
                ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
                ax2.set_yticklabels(dendro['ivl'])
                ax1.set_ylabel(**{k:v +' linkage matrix' 
                                    for k, v in self.ylab.items()} )
                fig.tight_layout()

class catmapflow2: 
    """
    Decorator function  collected  from the `func`the `target_values` to be 
    categorized and the `cat_range_values` to change 
    into `cat_classes` like::
          
          cat_range_values= [0.0, [0.0, 3.0], [3.0, 6.0], [6.0, 10.0], 10.0]
          target_values =[1, 2., 3., 6., 7., 9., 15., 25, ...]
          
    Decorated Fonction returns the  new function decorated holding  
    values  categorized into categorial `cat_classes`.
    For instance in groundwater exploration:
        
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
    
    def __init__(self, cat_classes: Iterable[str]=['FR0', 'FR1', 'FR2', 'FR3', 'FR4']):
        self._logging= watexlog().get_watex_logger(self.__class__.__name__)
        self.cat_classes = cat_classes 

    def __call__(self, func): 
        self._func = func 
        
        return  self.categorized_dec(self._func)
    
    def categorized_dec(self, func):
        """
        Decorator can be adapted  to other categorized problem by changing the 
        `cat_classes` arguments to another categorized classes 
        for other purposes like ::
            
         cat_classes=['dry', 'HV', 'IHV', 'IVH+', 'UH']   
            
        Where ``IVHU`` means I:improved V:village H:hydraulic and U:urban. 
        
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
            self.cat_range_values, target_array, catfc = func(*args, **kwargs)
            if catfc is not None : 
                self.cat_classes = catfc

            if len(self.cat_range_values) != len(self.cat_classes): 
            
                self._logging.error(
                    "Length of  categorical `values` and `classes` provided"
                    " must be consistent; '{0}' and '{1}' are given "
                    "respectively.".format(len(self.cat_range_values),
                                                    len(self.cat_classes)))
            try : 

                new_target_array = np.array(list(map( self.mapf, target_array)))
   
            except : 
                new_target_array = np.zeros_like(target_array)
                for ii, ff in enumerate(target_array) : 
                    new_target_array[ii] = self.mapf(crval=ff)
                    
            return new_target_array
        return wrapper  

    def mapf(self, crval):
        """
        Categorizing loop to hold the convenient classes according 
        to the `cat_range_value` provided. Come as supplement 
        tools when ``maping`` object doesnt work properly.
        
        :param crval: value to be categorized 
        :param nfval: array of `cat_range_values`
        :param fc: Object to replace the`crval` belonging 
            to `cat_classes`
        """
        nfval = self.cat_range_values.copy()  
        for ii, val in enumerate(nfval):
            try : 
                if isinstance(val, (float, int)): 
                    if crval ==nfval[0]: 
                        return self.cat_classes[0]
                    elif crval>= nfval[-1] : 
                        return self.cat_classes[-1]
                elif isinstance(val, (list, tuple)):
                    if len(val)>1: 
                        if  val[0] < crval <= val[-1] : 
                            return self.cat_classes[ii]
            except : 
                
                if crval ==0.: 
                    return self.cat_classes[0]
                elif crval>= nfval[-1] : 
                    return self.cat_classes[-1]

class docstring:
    """ Generate new doctring of a function or class by appending the doctring 
    of another function from the words considered as the startpoint `start` 
    to endpoint `end`.
    
    Sometimes two functions inherit the same parameters. Repeat the writing 
    of the same parameters is redundancy. So the most easier part is to 
    collect the doctring of the inherited function and paste to the new 
    function from the `startpoint`. 
    
    Parameters
    -----------
    func0: callable, 
        function to use its doctring 
    
    start: str 
        Value from which the new docstring should be start. 
    
    end: str 
        endpoint Value of the doctring. Stop considering point.
    

    Examples
    --------
    
    .. In the followings examples let try to append the `writedf` function
       from ``param reason`` (start) to `param to_` (end) to the 
       dostring to `predPlot` class. `predPlot` class class will holds new 
       doctring with writedf.__doc__ appended from `param reason` to 
       `param to_`.
        
    >>> from watex.decorators import writedf , predPlot, docstring 
    >>> docs = doctring(writedf, start ='param reason', end='param to_')(predPlot)
    >>> docs.__doc__
    >>> predPlot.__doc__ # doc modified and holds the writedf docstring too.
    
    *Author: @Daniel03*
    *Date: 18/09/2021*
    """
    def __init__(self, func0, start='Parameters', end=None ):
        
        self.func0 = func0
        self.start =start 
        self.end =end 
        
    def __call__(self, func): 
        self._func =func 
        return self._decorator(self._func )
    
    def _decorator(self, func): 
        """ Collect the doctring of `func0` from `start` to `end` and 
        add to a new doctring of wrapper`.
        """
        func0_dstr = self.func0.__doc__ 
        # keet the only part you need
        if self.start is None: 
            start_ix =0
        else: 
            start_ix = func0_dstr.find(self.start) # index of start point
            
        if self.end is not None: 
            end_ix = func0_dstr.find(self.end)
            # remain_end_substring = func0_dstr[end_ix:]
            substring = func0_dstr[start_ix :end_ix]
        else : 
            substring = func0_dstr[start_ix :]
            end_ix = -1 
            
        if start_ix <0 : 
            warnings.warn(f'`{self.start}` not find in the given '
                          f'{self.func0.__name__!r} doctring` function will '
                          f'append all the doctring of {self.func0.__name__!r}'
                          ' by default.')
            start_ix =0 

        if end_ix <0 : 
            warnings.warn(f'`{self.end} not found in the given'
                      f' {self.func0.__name__!r} doctring` function will '
                      f'append all the doctring of {self.func0.__name__!r}'
                      ' thin the end by default.')
        
        if self.start is not None: 
            try:
                param_ix = func.__doc__.find(self.start)
            except AttributeError: 
                if inspect.isclass(func): 
                    fname = func.__class__.__name__
                else: fname = func.__name__
                # mean there is no doctrings.
                # but silent the warnings  
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    warnings.warn(" Object `%s` has none doctrings!`NoneType`"
                                  " object has no attribute `find`."%fname)
                return func
            # find end _ix and remove 
            if func.__doc__.find(self.end)>=0: 
                example_ix = func.__doc__.find(self.end)
   
                str_betw_param_example = func.__doc__[
                    param_ix:example_ix]
            else : 
                str_betw_param_example= func.__doc__[param_ix:]
                example_ix =None
             # remove --- `start`value and `\n` at the end of 
             # in func substring      
            str_betw_param_example = str_betw_param_example.replace(
                self.start +'\n', '').replace('-\n', '').replace('-', '')
            # now remove start point in 
            for i, item in enumerate(str_betw_param_example): 
                if item !=' ': 
                    str_betw_param_example= str_betw_param_example[i:]
                    break 
            # in the concat string to new docstring of func.
            func.__doc__ = func.__doc__[:param_ix] + f'{substring}'+\
                str_betw_param_example 
                
            if example_ix is not None: 
                func.__doc__+=  func.__doc__[example_ix:]
            # set new_attributes 
            setattr(func, '__doc__', func.__doc__)

        return func
            

class docAppender: 
    """
    Decorator to generate a new doctring from appending the other class docstrings. 
    
    Indeed from the startpoint <`from_`> and  the endpoint<`to`>, one can select 
    the part of the any function or class doctrings to append to the existing 
    doctring for a new doctring creation. This trip is useful to avoid 
    redundancing parameters definitions everywhere in the scripts.
    
    Parameters 
    -----------
    func0: callable, 
        Function or class to collect the doctring from. 
    from_: str 
        Reference word or expression to start the collection of the 
        necessary doctring from the `func0`. It is the startpoint. The 
        *default* is ``Parameters``. 
        
    to: str 
        Reference word to end the collection of the necessary part of the 
        docstring  of `func0`. It is the endpoint. The *default* is ``Returns``.
        
    insert: str, 
        Reference word or expression to insert the collected doctring from 
        the `func0` and append of the index of the `insert` word in `func`. 
        If not found in the `func` doctring, it should retun None so nothing 
        should be appended.  The *default* is ``Parameters``. 
    
    Examples
    ---------
    >>> from watex.decorators import docAppender 
    >>> def func0 (*args, **kwargs): 
    ...        '''Im here so share my doctring. 
    ...        
    ...        Parameters 
    ...        -----------
    ...        * args: list, 
    ...            Collection of the positional arguments 
    ...        ** kwargs: dict 
    ...            Collection of keywords arguments 
    ...        Returns 
    ...        -------
    ...             None: nothing 
    ...        '''
    ...        pass 
    >>> def func(s, k=0): 
    ...        ''' Im here to append the docstring from func0
    ...        Parameters 
    ...        ----------
    ...        s: str , 
    ...            Any string value 
    ...        k: dict, 
    ...            first keyword arguments 
    ...            
    ...        Returns 
    ...        --------
    ...            None, I return nothing 
    ...        '''
    >>> deco = docAppender(func0 , from_='Parameters',
    ...                        to='Returns', insert ='---\\n')(func)
    >>> deco.__doc__
    ...
    
    Warnings 
    --------
    Be sure to append two doctrings with the same format. One may choose 
    either the sphinx or the numpy  doc formats. Not Mixing the both.  
    
    """
    insert_=('parameters',
            'returns',
            'raises', 
            'examples',
            'notes',
            'references', 
            'see also', 
            'warnings'
            )
    
    def __init__ (self,
                  func0: Callable[[F], F] ,
                  from_: str ='Parameters',
                 to: str ='Returns',
                 insertfrom: str = 'Parameters',
                 remove =True ): 
        self.func0 = func0 
        self.from_=from_ 
        self.to=to 
        self.remove= remove
        self.insert = insertfrom 
        
    def __call__(self, func): 
        self._func = copy.deepcopy(func )
        return self.make_newdoc (self._func)
    
    def  make_newdoc(self, func): 
        """ make a new docs from the given class of function """
 
        def sanitize_docstring ( strv): 
            """Sanitize string values and force the string to be 
            on the same level for parameters and the arguments of the 
            parameters. 
            :param strv: str 
            
            return a new string sanitized that match the correct spaces for 
            the sphinx documentation.
            
            """
            if isinstance(strv, str): 
                strv = strv.split('\n')
            # remove the ''  in the first string
            if strv[0].strip() =='':strv=strv[1:] 
            # get the first occurence for parameters definitions 
            ix_ = 0 ; 
            for ix , value in enumerate (strv): 
                if (value.lower().find(':param') >=0) or (value.lower(
                        ).find('parameters')>=0): 
                    ix_ = ix ; break 
            # Put all explanations in the same level 
            # before the parameters 
            for k in range(ix_ +1): 
                strv[k]= strv[k].strip() 
        
            for ii, initem in enumerate (strv): 
                for v in self.insert_: 
                    if initem.lower().find(v)>=0: 
                        initem= initem.strip() 
                        strv[ii]= initem
                        break 
    
                if '--' in initem or (':' in initem and len(initem) < 50) : 
                    strv[ii]= initem.strip() 
                elif (initem.lower() not in self.insert_) and ii > ix_:  
                    strv[ii]='    ' + initem.strip() 
            
            return '\n'.join(strv)  
 
        # get the doctring from the main func0 
        func0_dstr = self.func0.__doc__ 
        # select the first occurence and remove '----' if exists 
        if self.from_ is None: 
            warnings.warn('Argument `from_` is missing. Should be the first'
                          f' word of {self.func0.__name__!r} doctring.')
            self.from_ = func0_dstr.split()[0]
            
        from_ix = func0_dstr.find(self.from_)
        func0_dstr = func0_dstr [from_ix:]
        # remove the first occurence of the from_ value and --- under if exists. 
        # in the case where from =':param' remove can be set to False 
        if self.remove: 
            func0_dstr = func0_dstr.replace(self.from_, '', 1).replace('-', '')
        # get the index of 'to' or set None if not given   
        # now we are selected the part and append to the 
        # existing doc func where do you want to insert 
        to_ix = func0_dstr.find (self.to ) if self.to is not None else None 
        func0_dstr= func0_dstr [:to_ix if to_ix >=0 else None]
       
        if self.insert.lower() not in (self.insert_): 
            warnings.warn(f"It's seems the given  {self.insert!r} for docstring"
                          f" insertion is missing to {self.insert_} list")
        
        in_ix =  self._func.__doc__.lower().find(self.insert.lower())
        # assert  whether the given value insert from exists . 
        if in_ix < 0 : 
            warnings.warn(f"Insert {self.insert!r} value is not found in the "
                          "{'class' if inspect.isclass(self._func) else 'function'")
        # split the string with `\n` 
        # and loop to find the first occurence 
        # by default skip the next item which could be '----' 
        # and insert to the list next point 
        func0_dstr = func0_dstr.split('\n')
        finalstr = self._func.__doc__.split('\n') 
        
        rpop(func0_dstr) 
        func0_dstr =  '\n'.join(func0_dstr)    
        for ii, oc in enumerate(finalstr) : 
            if oc.lower().find(self.insert.lower()) >=0 : 
                finalstr.insert (ii+2, func0_dstr)
                finalstr = '\n'.join(finalstr);break 
        
        setattr(func, '__doc__', sanitize_docstring (finalstr))
        
        return func 
    
class docSanitizer: 
    """Decorator to clean the doctring and  set all values of sections to 
    the same level. 
    
    It sanitizes the doctring for the use of sphinx documentation. 
    
    Examples
    --------
    >>> from watex.decorators import docSanitizer 
    >>> def messdocfunc(): 
    ...        '''My doctring is mess. I need to be polished and well arranged.
    ...        
    ...        Im here to sanitize the mess doctring. 
    ...        
    ...        Parameters
    ...        ----------
    ...                * args: list, 
    ...                    Collection of the positional arguments 
    ...                ** kwargs: dict 
    ...                    Collection of keywords arguments 
    ...
    ...        * kwargs: list,
    ...        Collection of the keyword arguments
    ...        
    ...        Warnings
    ...        --------
    ...        Let check for warnings string ... 
    ...        
    ...       '''
    ...       pass
    >>> cleandocfunc = docSanitizer()(messfocfunc)
    >>> print(cleandocfunc.__doc__)
    ... '''
    ...    My doctring is mess. I need to be polished and well arranged.
    ...
    ...    Parameters
    ...    ----------
    ...    * args: list,
    ...       Collection of the positional arguments
    ...    ** kwargs: dict
    ...        Collection of keywords arguments
    ...    * kwargs: list,
    ...        Collection of the keyword arguments
    ...    '''
    
    """
    
    insert_= ('parameters','returns','raises', 'examples','notes',
            'references', 'see also', 'warnings', ':param', ':rtype', 
            )
    
    def __call__(self, func): 
        
        func =copy.deepcopy(func)
        docstring = copy.deepcopy(func.__doc__) 
        
        if isinstance(docstring , str): 
            docstring = docstring .split('\n')
        # remove the ''  in the first string
        if docstring [0].strip() =='':docstring =docstring [1:] 
        # get the first occurence for parameters definitions 
        # and separate the doctring into two parts: descriptions 
        #and corpus doctring as the remainings 
        
        ix_ = 0  
        for ix , value in enumerate (docstring ): 
            if (value.lower().find(':param') >=0) or (value.lower(
                    ).find('parameters')>=0): 
                ix_ = ix ; break 
            
        #-->  sanitize the descriptions part 
        description =docstring [: ix_] ; 
        # before the parameters 
        for k in range(len(description)): 
            description [k]= description [k].strip() 
         # remove at the end of description the blanck space '\n' 
        description = description[:-1] if  description[-1].strip(
            )== ''  else description
      
        # --> work with the corpus docstrings 
        # get indexes for other sections and removes spaces 
        docstring = docstring [ix_:]
        rpop (docstring)
        ixb = len(docstring)
        for ind , values in enumerate (docstring): 
            if values.lower().strip() in (
                    'examples', 'see also', 'warnings', 
                     'notes', 'references'): 
                ixb = ind ; break 
        # all values in same level 
        for k in range(ixb, len(docstring)): 
            docstring [k]= docstring [k].strip() 
        for ii, initem in enumerate (docstring ): 
            for v in self.insert_: 
                if initem.lower().find(v)>=0: 
                    initem= initem.strip() 
                    docstring [ii]= initem
                    break 
            if '--' in initem or (
                    ':' in initem and len(initem) < 50
                    ) or ix_>=ixb : 
                docstring [ii]= initem.strip() 
            elif (initem.lower() not in self.insert_
                  ) and ix_< ii < ixb:  
                docstring [ii]='    ' + initem.strip() 
        # add  blanck line from indexes list ixs 
        ixs=list()
        for k, item in enumerate (docstring): 
            for param in self.insert_[:-2]: 
                if item.lower().strip() == param:  
                    ixs.append(k)
                    break   
        ki =0  
        for k in ixs : 
            docstring.insert (k+ki, '')  
            ki+=1 # add number of insertions 
            
        # --> combine the descriptions and docstring and set attributes 
        setattr(func, '__doc__' , '\n'.join(description + docstring ))  
          
        return  func

class gplot2d(object): 
    """
    Decorator class to plot geological models. 
    
    Arguments
    ----------
    
    **reason**: type of plot, can be `misfit` or `model`. If `` None``, 
        will plot `model`.
    reason: str
        related to the kind of plot 
    
    kws: Matplotlib properties and model properties

    Additional keywords attributes and descriptions
    
    ======================  ===============================================
    keywords                Description
    ======================  ===============================================
    cb_pad                  padding between axes edge and color bar 
    cb_shrink               percentage to shrink the color bar
    climits                 limits of the color scale for resistivity
                            in log scale (min, max)
    cmap                    name of color map for resistivity values
    fig_aspect              aspect ratio between width and height of 
                            resistivity image. 1 for equal axes
    fig_dpi                 resolution of figure in dots-per-inch
    fig_num                 number of figure instance
    fig_size                size of figure in inches (width, height)
    font_size               size of axes tick labels, axes labels is +2
    grid                    [ 'both' | 'major' |'minor' | None ] string 
                            to tell the program to make a grid on the 
                            specified axes.
    ms                      size of station marker 
    plot_yn                 [ 'y' | 'n']
                            'y' --> to plot on instantiation
                            'n' --> to not plot on instantiation
    station_color           color of station marker
    station_font_color      color station label
    station_font_pad        padding between station label and marker
    station_font_rotation   angle of station label in degrees 0 is 
                            horizontal
    station_font_size       font size of station label
    station_font_weight     font weight of station label
    station_id              index to take station label from station name
    station_marker          station marker.  if inputing a LaTex marker
                            be sure to input as r"LaTexMarker" otherwise
                            might not plot properly
    title                   title of plot.  If None then the name of the
                            iteration file and containing folder will be
                            the title with RMS and Roughness.
    xlimits                 limits of plot in x-direction in (km) 
    xminorticks             increment of minor ticks in x direction
    xpad                    padding in x-direction in km
    ylimits                 depth limits of plot positive down (km)
    yminorticks             increment of minor ticks in y-direction
    ypad                    padding in negative y-direction (km)
    yscale                  [ 'km' | 'm' ] scale of plot, if 'm' everything
                            will be scaled accordingly.
    ======================  ===============================================
    
    """
    def __init__(self, reason =None , **kws):
        self._logging= watexlog().get_watex_logger(self.__class__.__name__)
        self.reason =reason 
        self.fs =kws.pop('fs', 0.7)
        self.fig_num = kws.pop('fig_num', 1)
        self.fig_size = kws.pop('fig_size', [7,7])
        self.fig_aspect = kws.pop('fig_aspect','auto')
        self.fig_dpi =kws.pop('fig_dpi', 300)
        self.font_size = kws.pop('font_size', 7)
        self.aspect = kws.pop('aspect', 'auto')
        self.font_style =kws.pop('font_style', 'italic')
        self.orient=kws.pop('orientation', 'landscape')
        self.cb_pad = kws.pop('cb_pad', .0375)
        self.cb_orientation = kws.pop('cb_orientation', 'vertical')
        self.cb_shrink = kws.pop('cb_shrink', .75)
        self.cb_position = kws.pop('cb_position', None)
        self.climits = kws.pop('climits', (0, 4))
        self.station_label_rotation = kws.pop('station_label_rotation',45)
        self.imshow_interp = kws.pop('imshow_interp', 'bicubic')
        self.ms = kws.pop('ms', 2)
        self.lw =kws.pop('lw', 2)
        self.fw =kws.pop('font_weight', 'bold')
        self.station_font_color = kws.pop('station_font_color', 'k')
        self.station_marker = kws.pop('station_marker',r"$\blacktriangledown$")
        self.station_color = kws.pop('station_color', 'k')
        self.xpad = kws.pop('xpad', 1.0)
        self.ypad = kws.pop('ypad', 1.0)
        self.cmap = kws.pop('cmap', 'jet_r')
        self.depth_scale =kws.pop('depth_scale', None)
        self.doi = kws.pop('doi', 1000)
        self.savefig =kws.pop('savefig', None)
        self.model_rms =kws.pop('model_rms', None)
        self.model_roughness =kws.pop('model_roughness', None)
        self.plot_style =kws.pop( 'plot_style', 'pcolormesh') 
        self.grid_alpha =kws.pop('alpha', 0.5)
        self.show_grid = kws.pop('show_grid',True)
        self.set_station_label=kws.pop('show_station_id', True)
        
        for keys in list(kws.keys()): 
            setattr(self, keys, kws[keys])
            

    def __call__(self, func):  
        """
        Model decorator to hold the input function with arguments 
        :param func: function to be decorated 
        :type func: object 
        """

        return self.plot2DModel(func)
        
    def plot2DModel(self, func):
        @functools.wraps(func)
        def new_func (*args, **kwargs): 
            """
            new decorated function . Plot model data and misfit data 
            
            :args: arguments of  function  to be decorated 
            :type args: list 
        
            :param kwargs: positional arguments of decorated function
            :type kwargs: dict 
            :return: function decorated after visualisation
      
            """
            self._logging.info(
                ' Plot decorated {0}.'.format(func.__name__))
    
            _f=0 # flag to separated strata model misfit and occam model misfit
                #   from occamResponse file 
                
            if self.depth_scale is not None :
                self.depth_scale= str(self.depth_scale).lower() 

            if self.depth_scale not in ["km", "m"]: 
                mess ="Depth scale =`{}` is unacceptable value."\
                    " Should be convert to 'm'.".format(self.depth_scale)
                warnings.warn(mess)
                self.depth_scale= "m"
                self._logging.debug (mess)
            
            if self.depth_scale == 'km':
                dz  = 1000.
            elif self.depth_scale == 'm': 
                dz = 1.
    
            # figure configuration 
            
            self.fig = plt.figure(self.fig_num, self.fig_size, dpi=self.fig_dpi)
            plt.clf()
            self.fig_aspect ='auto'
            axm = self.fig.add_subplot(1, 1, 1, aspect=self.fig_aspect)
    
            # get geomodel data 
            if self.reason is None: 
                self.reason = 'model'# by default
                
            # ----populate special attributes from model or misfit ------------
            if self.reason =='model': 
                occam_model_resistiviy_obj, occam_data_station_names, *m = func(
                    *args, **kwargs)
                occam_data_station_offsets, occam_model_depth_offsets, *ddrms= m
                self.doi, self.depth_scale, self.model_rms, *rmisf = ddrms 
                self.model_roughness, plot_misfit = rmisf
                
                self.doi = occam_model_depth_offsets.max()
                #     self.doi = occam_model_depth_offsets.max()
                # --> check doi value provided, and convert to default unit {meters}  
                self.doi =assert_doi(doi=self.doi)
                
                # set boundaries of stations offsets and depth 
                spec_f = -(self.doi/5)/dz  # assume that depth will  start by 
                #0 then substract add value so 
                # to get space for station names text
                if self.climits is None :
                    self.climits =(0,4)  
                if plot_misfit is True : 
                    _f=2
                self.ylimits =(spec_f, self.doi/dz)  
                
            if self.reason =='misfit': 
                occam_model_resistiviy_obj, occam_data_station_names, *m = func(
                    *args, **kwargs)     
                occam_data_station_offsets, occam_model_depth_offsets, *rg=m
                self.model_rms, self.model_roughness= rg
                
                # check if "plotmisfit refers to 'geoStrata model 'geodrill
                # module then keep the doi and set `spec_f
                if 'geodtype' in list(kwargs.keys()): 
                    # means plot `misfit` from geostrata model
                    spec_f = -(self.doi/5)/dz 
                    self.ylimits =(spec_f, self.doi/dz) 
                    
                else :
                    # frequency are in log10 new doi is set according to 
                    self.doi =occam_model_depth_offsets.max()
                    #spec_f = (self.doi/5)/dz # o.8
                    spec_f = - 0.
                    _f=1 
       
                    self.ylimits = (self.doi, occam_model_depth_offsets.min())
            
            #------------- manage stations and climits ------------------------  
            occam_data_station_offsets =np.array(occam_data_station_offsets)
            # station separation and get xpad . ex dl=50 then xpad =25 
            dl = occam_data_station_offsets.max()/ (len(
                occam_data_station_offsets)-1)
            self.xpad = (dl/2)/dz 

                                   
            self.xlimits=(occam_data_station_offsets.min()/dz -self.xpad  , 
                      occam_data_station_offsets.max()/dz + self.xpad )
            
            # configure climits 
            if self.reason =='misfit':
                if self.climits is None : 
                        self.climits =(-3, 3)
                        
                elif 'min' in self.climits or 'max' in self.climits : 
                            self.climits = (occam_model_resistiviy_obj.min(), 
                                            occam_model_resistiviy_obj.max())
    
             
            if _f==2 : 
                self.reason = 'misfit' 
            self._logging.info ('Ready to plot {0}'
                                ' with matplotlib "{1}" style.'.
                                format(self.reason, self.plot_style))  
            
            # -------------- check dimensionnality ---------------------------
            occam_model_resistiviy_obj, *dm= self._check_dimensionality (
                        occam_model_resistiviy_obj,
                        occam_model_depth_offsets,
                          occam_data_station_offsets
                          )
            occam_model_depth_offsets, occam_data_station_offsets = dm

            
            if self.plot_style.lower() =='pcolormesh':
                mesh_x  , mesh_z= np.meshgrid(occam_data_station_offsets,
                                              occam_model_depth_offsets )
     
                vmin = self.climits[0]
                vmax = self.climits[1] 
    
                axm.pcolormesh (mesh_x/dz  , 
                                mesh_z/dz ,
                                  occam_model_resistiviy_obj,
                                      vmin = vmin,
                                      vmax = vmax,  
                                      shading= 'auto', 
                                      cmap =self.cmap, 
                                      alpha = None, 
                                     
                                      )
         

            if self.plot_style.lower() =='imshow': 
    
                mesh_x  , mesh_z= np.meshgrid(occam_data_station_offsets,
                                              occam_model_depth_offsets  )
    
                axm.imshow (occam_model_resistiviy_obj,
                                    vmax = self.climits[1], 
                                    vmin =self.climits[0], 
                                    interpolation = self.imshow_interp, 
                                    cmap =self.cmap,
                                    aspect = self.fig_aspect,
                                    origin= 'upper', 
                                    extent=( self.xlimits[0],
                                            self.xlimits[1],
                                            self.ylimits[1], 
                                            self.ylimits[0] - spec_f),
                                        )
    
                
            # get colormap for making a colorbar 
            if type(self.cmap) == str:
                self.cmap = cm.get_cmap(self.cmap)
            
            axm.set_xlim( [self.xlimits[0],  self.xlimits[1]])
            axm.set_ylim ([self.ylimits[1], self.ylimits[0]]) 
    
            # create twin axis to set ticks to the top station
            axe2=axm.twiny()
            axe2.xaxis.set_visible(False) # let keep only the axe lines 
            #set axis and set boundaries 
            if self.reason =='model' or _f==2 : 
                ydown_stiteslbls = self.ylimits[0]/5 
                ydown_stationlbls = self.ylimits[0] -(self.ylimits[0]/3)
                xhorizontal_lbs = (occam_data_station_offsets.max()/dz)/2
    
            elif self.reason =='misfit': 
                ydown_stiteslbls = self.ylimits[0] + 0.1 * self.ylimits[1]
                ydown_stationlbls= self.ylimits[0] +\
                    self.ylimits[1]/self.ylimits[0]
                xhorizontal_lbs = (occam_data_station_offsets.max()- 
                                   occam_data_station_offsets.min())/2
               
            for offset , names in zip (occam_data_station_offsets,
                                       occam_data_station_names):
                # plot the station marker ' black triangle down ' 
                # always plots at the surface.
                axm.text(offset/dz  ,
                        self.ylimits[0] - spec_f,  
                        s= self.station_marker,
                        horizontalalignment='center',
                        verticalalignment='baseline',
                        fontdict={'size': self.ms*5, 
                                  'color': self.station_color},
                        )
                
                if self.set_station_label is True :  # then plot label id 
                    axm.text(offset/dz ,
                            ydown_stiteslbls,  
                            s= names,
                            horizontalalignment='center',
                            verticalalignment='baseline',
                            fontdict={'size': self.ms*3, 
                                      'color': self.station_color},
                            rotation = self.station_label_rotation,
                                )
         
               
            if self.set_station_label is True : 
                axm.text (xhorizontal_lbs, 
                            ydown_stationlbls,  
                            s= 'Stations',
                            horizontalalignment='center',
                            verticalalignment='baseline',
                            fontdict={'size': self.ms*5, 
                                      'color':'k', 
                                      'style': self.font_style,
                                      'weight': self.fw},
                            )
    
            #-------------------- manage grid and colorbar -------------------- 
            self.g2dgridandcbManager(axm, _f)
            #------------------------------------------------------------------
            # initialize the reason to keep the default reason  
            self.reason = None
            
            if self.savefig is not None : 
                plt.savefig(self.savefig, dpi = self.fig_dpi)
            
            plt.show()
            
            return func(*args, **kwargs)
        
        return new_func
    
    def _check_dimensionality(self, data, z, x):
        """ Check dimensionality of data and fix it"""

        def reduce_shape(Xshape, x, axis_name =None): 
            """ Reduce shape to keep the same shape"""
            mess ="`{0}` shape({1}) {2} than the data shape `{0}` = ({3})."
            ox = len(x) 
            dsh = Xshape 
            if len(x) > Xshape : 
                x = x[: int (Xshape)]
                self._logging.debug(''.join([
                    f"Resize {axis_name!r}={ox!r} to {Xshape!r}.", 
                    mess.format(axis_name, len(x),'more',Xshape)])) 
                                        
            elif len(x) < Xshape: 
                Xshape = len(x)
                self._logging.debug(''.join([
                    f"Resize {axis_name!r}={dsh!r} to {Xshape!r}.",
                    mess.format(axis_name, len(x),'less', Xshape)]))
                
            return int(Xshape), x 
        
        sz0, z = reduce_shape(data.shape[0],
                              x=z, axis_name ='Z')
        sx0, x =reduce_shape (data.shape[1], 
                              x=x, axis_name ='X')
        # resize theshape 
        # data  = np.resize(data, (sz0, sx0))
        data = data [:sz0, :sx0]
        
        return data , z, x 
                
                
    def g2dgridandcbManager(self, axm, _f=None) :
        """ Plot2d model by configure grid and colorbar. 
        :param axm: 2d axis plot 
        :param _f: resize flag; misfit =2 and model =1 """
        # put a grid on if set to True 
        if self.show_grid is True:
            axm.minorticks_on()
            axm.grid(color='k', ls=':', lw =0.5, 
                      alpha=self.grid_alpha, which ='major')
        

          #set color bar properties 
        cbx = mplcb.make_axes(axm,  shrink=self.cb_shrink,
                              pad=self.cb_pad , location ='right' )
        cb = mplcb.ColorbarBase(cbx[0],
                        cmap=self.cmap,
                        norm=mpl.colors.Normalize(vmin=self.climits[0],
                                        vmax=self.climits[1]))
        
        cb.set_label('Resistivity ($\Omega \cdot$m)',
                  fontdict={'size': self.font_size + 1, 
                            'weight': 'bold'})
        
        if self.reason == 'model' : 
            cb.set_ticks(np.arange(int(self.climits[0]), 
                                   int(self.climits[1]) + 1))
            cb.set_ticklabels(['10$^{0}$'.format('{' + str(nn) + '}') 
                               for nn in np.arange(int(self.climits[0]),
                                          int(self.climits[1]) + 1)])
            
                                
        else : 
            cb.set_ticks(np.linspace(self.climits[0], self.climits[1],5))
            cb.set_ticklabels(['{0}'.format(str(round(nn,2))) for nn in
                                np.linspace(self.climits[0],
                                          self.climits[1],5)])
            cb.set_label('misfitvalue(%)',
                  fontdict={'size': self.font_size + 1, 
                            'weight': 'bold'})
       
        # set axes labels
        axm.set_xlabel('Distance ({0})'.format(self.depth_scale),
                      fontdict={'size': self.font_size + 2,
                                'weight': 'bold'})
        
        if self.reason =='misfit':
            if _f ==2: ylabel = 'Depth ({0})'.format(self.depth_scale)
            else : ylabel= 'Log10Frequency(Hz)'
            mesT ='Plot Misfit'
            
        elif self.reason =='model':
            ylabel = 'Depth ({0})'.format(self.depth_scale)
            mesT = 'Plot strata model' 
        
        axm.set_ylabel(ylabel,fontdict={
            'size': self.font_size + 2, 'weight': 'bold'})
       
       
        self.fig.suptitle('{0}- DataType = {1} :RMS={2}, Roughness={3}'.\
                          format(mesT, self.reason, self.model_rms, 
                                self.model_roughness),
                  ha='center',
          fontsize= 7* self.fs, 
          verticalalignment='center', 
          style =self.font_style,
          bbox =dict(boxstyle='round',facecolor ='moccasin'), 
          y=0.95 if self.reason =='model' else 0.98)
      
        return self 
    
class _M:
    def _m(self): pass
MethodType = type(_M()._m)

class _AvailableIfDescriptor:
    """Implements a conditional property using the descriptor protocol.

    Using this class to create a decorator will raise an ``AttributeError``
    if check(self) returns a falsey value. Note that if check raises an error
    this will also result in hasattr returning false.

    See https://docs.python.org/3/howto/descriptor.html for an explanation of
    descriptors.
    """

    def __init__(self, fn, check, attribute_name):
        self.fn = fn
        self.check = check
        self.attribute_name = attribute_name

        # update the docstring of the descriptor
        functools.update_wrapper(self, fn)

    def __get__(self, obj, owner=None):
        attr_err = AttributeError(
            f"This {repr(owner.__name__)} has no attribute {repr(self.attribute_name)}"
        )
        if obj is not None:
            # delegate only on instances, not the classes.
            # this is to allow access to the docstrings.
            if not self.check(obj):
                raise attr_err
            out = MethodType(self.fn, obj)

        else:
            # This makes it possible to use the decorated method as an unbound method,
            # for instance when monkeypatching.
            @functools.wraps(self.fn)
            def out(*args, **kwargs):
                if not self.check(args[0]):
                    raise attr_err
                return self.fn(*args, **kwargs)

        return out
    
@contextmanager
def nullify_output(suppress_stdout=True, suppress_stderr=True):
    """
    suppress stdout and stderr messages using context manager. 
    https://www.codeforests.com/2020/11/05/python-suppress-stdout-and-stderr/ 
    """
    stdout = sys.stdout
    stderr = sys.stderr
    devnull = open(os.devnull, "w")
    try:
        if suppress_stdout:
            sys.stdout = devnull
        if suppress_stderr:
            sys.stderr = devnull
        yield
    finally:
        if suppress_stdout:
            sys.stdout = stdout
        if suppress_stderr:
            sys.stderr = stderr
            
class suppress_output:
    """ 
    Python recipes- suppress stdout and stderr messages
    
    If you have worked on some projects that requires API calls to the 
    external parties or uses 3rd party libraries, you may sometimes run 
    into the problem that you are able to get the correct return results 
    but it also comes back with a lot of noises in the stdout and stderr. 
    For instance, the developer may leave a lot of “for your info” messages 
    in the standard output or some warning or error messages due to the 
    version differences in some of the dependency libraries.

    All these messages would flood your console and you have no control on 
    the source code, hence you cannot change its behavior. To reduce these 
    noises, one option is to suppress stdout and stderr messages during 
    making the function call. In this article, we will discuss about some 
    recipes to suppress the messages for such scenarios.
    
    """
    def __init__(self, suppress_stdout=False, suppress_stderr=False):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        devnull = open(os.devnull, "w")
        if self.suppress_stdout:
            self._stdout = sys.stdout
            sys.stdout = devnull

        if self.suppress_stderr:
            self._stderr = sys.stderr
            sys.stderr = devnull

    def __exit__(self, *args):
        if self.suppress_stdout:
            sys.stdout = self._stdout
        if self.suppress_stderr:
            sys.stderr = self._stderr
            
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            
def available_if(check):
    """An attribute that is available only if check returns a truthy value

    Parameters
    ----------
    check : callable
        When passed the object with the decorated method, this should return
        a truthy value if the attribute is available, and either return False
        or raise an AttributeError if not available.

    Examples
    --------
    >>> from sklearn.utils.metaestimators import available_if
    >>> class HelloIfEven:
    ...    def __init__(self, x):
    ...        self.x = x
    ...
    ...    def _x_is_even(self):
    ...        return self.x % 2 == 0
    ...
    ...    @available_if(_x_is_even)
    ...    def say_hello(self):
    ...        print("Hello")
    ...
    >>> obj = HelloIfEven(1)
    >>> hasattr(obj, "say_hello")
    False
    >>> obj.x = 2
    >>> hasattr(obj, "say_hello")
    True
    >>> obj.say_hello()
    Hello
    """
    return lambda fn: _AvailableIfDescriptor(fn, check, attribute_name=fn.__name__)


# decorators utilities 
def rpop(listitem): 
    """ remove all blank line in the item list. 
    :param listitem: list- list of the items and pop all 
    the existing blanck lines. """
    # now pop all the index for blanck line 
    isblanck = False 
    for ii, item  in enumerate (listitem) : 
        if item.strip()=='': 
            listitem.pop(ii)
            isblanck =True 
    return rpop(listitem) if isblanck else False  

def assert_doi(doi): 
    """
     assert the depth of investigation Depth of investigation converter 

    :param doi: depth of investigation in meters.  If value is given as string 
        following by yhe index suffix of kilometers 'km', value should be 
        converted instead. 
    :type doi: str|float 
    
    :returns doi:value in meter
    :rtype: float
           
    """
    if isinstance (doi, str):
        if doi.find('km')>=0 : 
            try: doi= float(doi.replace('km', '000')) 
            except :TypeError (" Unrecognized value. Expect value in 'km' "
                           f"or 'm' not: {doi!r}")
    try: doi = float(doi)
    except: TypeError ("Depth of investigation must be a float number "
                       "not: {str(type(doi).__name__!r)}")
    return doi
    


    