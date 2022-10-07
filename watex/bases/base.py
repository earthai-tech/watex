# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Created:on Tue Oct 12 15:37:59 2021
#   Edited:on Fri Sep 10 15:37:59 2022


from __future__ import annotations 
  
import re 
import sys 
from warnings import warn

from ..tools.mlutils import ( 
    existfeatures,
    selectfeatures 
    )
from ..tools.coreutils import ( 
    _is_readable 
    )

from .._docstring import ( 
    DocstringComponents,
    _core_docs,
    ) 
from ..typing import (
    List, 
    Optional, 
    DataFrame, 
    )
from ..tools.funcutils import (
    _assert_all_types, 
    is_installing, 
    repr_callable_obj,
    smart_strobj_recognition,
)
 
from ..exceptions import (
    NotFittedError
    )
from ..view.plot import (
   ExPlot
    )

from .._watexlog import  watexlog

# +++ add base documentations +++
_base_params = dict ( 
    axis="""
axis: {0 or 'index', 1 or 'columns'}, default 0
    Determine if rows or columns which contain missing values are 
    removed.
    * 0, or 'index' : Drop rows which contain missing values.
    * 1, or 'columns' : Drop columns which contain missing value.
    Changed in version 1.0.0: Pass tuple or list to drop on multiple 
    axes. Only a single axis is allowed.    
    """, 
    columns="""
columns: str or list of str 
    columns to replace which contain the missing data. Can use the axis 
    equals to '1'.
    """, 
    name="""
name: str, :attr:`pandas.Series.name`
    A singluar column name. If :class:`pandas.Series` is given, 'name'  
    denotes the attribute of the :class:`pandas.Series`. Preferably `name`
    must correspond to the label name of the target. 
    """, 
    sample="""
sample: int, Optional, 
    Number of row to visualize or the limit of the number of sample to be 
    able to see the patterns. This is usefull when data is composed of 
    many rows. Skrunked the data to keep some sample for visualization is 
    recommended.  ``None`` plot all the samples ( or examples) in the data     
    """, 
    kind="""
kind: str, Optional 
    type of visualization. Can be ``dendrogramm``, ``mbar`` or ``bar``. 
    ``corr`` plot  for dendrogram , :mod:`msno` bar,  :mod:`plt`
    and :mod:`msno` correlation  visualization respectively: 
        * ``bar`` plot counts the  nonmissing data  using pandas
        *  ``mbar`` use the :mod:`msno` package to count the number 
            of nonmissing data. 
        * dendrogram`` show the clusterings of where the data is missing. 
            leaves that are the same level predict one onother presence 
            (empty of filled). The vertical arms are used to indicate how  
            different cluster are. short arms mean that branch are 
            similar. 
        * ``corr` creates a heat map showing if there are correlations 
            where the data is missing. In this case, it does look like 
            the locations where missing data are corollated.
        * ``None`` is the default vizualisation. It is useful for viewing 
            contiguous area of the missing data which would indicate that 
            the missing data is  not random. The :code:`matrix` function 
            includes a sparkline along the right side. Patterns here would 
            also indicate non-random missing data. It is recommended to limit 
            the number of sample to be able to see the patterns. 
    Any other value will raise an error. 
    """, 
    inplace="""
inplace: bool, default False
    Whether to modify the DataFrame rather than creating a new one.    
    """

 )

_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"],
    base = DocstringComponents(_base_params)
    )
# +++ end base documentations +++

_logger = watexlog().get_watex_logger(__name__)

class Data: 
    def __init__ (self, 
                  verbose: int =0, 
                  ): 
        self._logging= watexlog().get_watex_logger(self.__class__.__name__)

        self.verbose=verbose 
        self.data_=None 
        
        
    @property 
    def data (self ):
        """ return verified data """
        return self.data_ 
    @data.setter 
    def data (self, d):
        """ Read and parse the data"""
        self.data_ = _is_readable (d) 
        
    @property 
    def describe (self): 
        """ Get summary stats  as well as see the cound of non-null data.
        Here is the default behaviour of the method i.e. it is to only report  
        on numeric columns. To have have full control, do it manually by 
        yourself. 
        
        """
        return self.data.describe() 
    
    def fit(self, data: str | DataFrame=None):
        """ Read, assert and fit the data 
        
        Parameters 
        ------------
        data: Dataframe or shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N
        
        Returns 
        ---------
        :class:`Data` instance
            Returns ``self`` for easy method chaining.
            
        """ 
        
        
        if data is not None: 
            self.data = data 

        return self 
    
    def skrunk (self, 
                columns: list[str], 
                data: str | DataFrame = None, 
                **kwd 
                ):
        """ Reduce the data with importance features
        
        Parameters 
        ------------
        data: Dataframe or shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N
        
        columns: str or list of str 
            Columns or features to keep in the datasets

        kwd: dict, 
        additional keywords arguments from :func:`watex.tools.mlutils.selectfeatures`
 
        Returns 
        ---------
        :class:`Data` instance
            Returns ``self`` for easy method chaining.
        
        """ 
        if data is not None: 
            self.data = data 
        self.data = selectfeatures(
            self.data , features = columns, **kwd)
  
        return self 
    

    def profilingReport (self, data: str | DataFrame = None, **kwd):
        """Generate a report in a notebook. 
        
        It will summarize the types of the columns and allow yuou to view 
        details of quatiles statistics, a histogram, common values and extreme 
        values. 
        
        Parameters 
        ------------
        data: Dataframe or shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N
        
        Returns 
        ---------
        :class:`Data` instance
            Returns ``self`` for easy method chaining.
        
        Examples 
        ---------
        >>> from watex.bases.base import Data 
        >>> Data().fit(data).profilingReport()
        
        """
        
        if data is not None: 
            self.data = data 
 
        try : 
           import pandas_profiling 
           
        except ImportError:
            warn("Missing 'pandas_profiling` library."
                 " auto-installation is triggered. please wait ... ")
            if self.verbose: 
                
                print("### -> Libray 'pandas_profiling is missing."
                      " subprocess installation is triggered. Please wait ...")
            
            is_success = is_installing('pandas_profiling', DEVNULL=True, 
                                       verbose = self.verbose )
            if not is_success : 
                warn ("'pandas_profiling' auto-installation failed. "
                       " Try a mannual installation.")
            if self.verbose: 
                print("+++ -> Installation complete!") if is_success else print(
                    "--- > Auto-installation failed. Try the mannual way.")
                
        if 'pandas_profiling' in sys.modules: 
            
            pandas_profiling.ProfilingReport( self.data , **kwd)
             
        return self 
    
    def rename_columns (self, 
                        data: str | DataFrame= None, 
                        columns: List[str]=None, 
                        pattern:Optional[str] = None
                        ): 
        """ 
        rename columns of the dataframe with columns in lowercase and spaces 
        replaced by underscores. 
        
        Parameters 
        -----------
        data: Dataframe of shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N
        
        columns: str or list of str, Optional 
            the  specific columns in dataframe to renames. However all columns 
            is put in lowecase. if columns not in dataframe, error raises.  
            
        pattern: str, Optional, 
            Regular expression pattern to strip the data. By default, the 
            pattern is ``'[ -@*#&+/]'``.
        
        Return
        -------
        ``self``: :class:`watex.bases.base.Data` instance 
            returns ``self`` for easy method chaining.
        
        """
        pattern = str (pattern)
        
        if pattern =='None': 
            pattern =  r'[ -@*#&+/]'
        regex =re.compile (pattern, flags=re.IGNORECASE)
        
        if data is not None: 
            self.data = data 
            
        self.data.columns= self.data.columns.str.strip() 
        if columns is not None: 
            existfeatures(self.data, columns, 'raise')
            
        if columns is not None: 
            self.data[columns].columns = self.data[columns].columns.str.lower(
                ).map(lambda o: regex.sub('_', o))
        if columns is None: 
            self.data.columns = self.data.columns.str.lower().map(
                lambda o: regex.sub('_', o))
        
        return self 
    
    def merge (self) : 
        """ Merge two series whatever the type with operator `&&`. 
        
        When series as dtype object as non numeric values, dtypes should be 
        change into a object 
        """
        # try : 
        #     self.data []
        
    __and__= __rand__ = merge 
    
    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        return repr_callable_obj  (self, skip ='y') 
       
    def __getattr__(self, name):
        if name.endswith ('_'): 
            if name not in self.__dict__.keys(): 
                if name in ('data_', 'X_'): 
                    raise NotFittedError (
                        f'Fit the {self.__class__.__name__!r} object first'
                        )
                
        rv = smart_strobj_recognition(name, self.__dict__, deep =True)
        appender  = "" if rv is None else f'. Do you mean {rv!r}'
        
        raise AttributeError (
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{appender}{"" if rv is None else "?"}'
            )        
        
        

class Missing (Data) : 
    """ Deal with missing in Data 
    
    Most algorithms will not work with missing data. Notable exceptions are the 
    recent boosting libraries such as the XGBoost (:doc:`watex.documentation.xgboost.__doc__`) 
    CatBoost and LightGBM. 
    As with many things in machine learning , there are no hard answaers for how 
    to treat a missing data. Also, missing data could  represent different 
    situations. There are three warious way to handle missing data:: 
        
        * Remove any row with missing data 
        * Remove any columns with missing data 
        * Impute missing values 
        * Create an indicator columns to indicator data was missing 
    
    Arguments
    ----------- 
    in_percent: bool, 
        give the statistic of missing data in percentage if ser to ``True``. 
        
    sample: int, Optional, 
        Number of row to visualize or the limit of the number of sample to be 
        able to see the patterns. This is usefull when data is composed of 
        many rows. Skrunked the data to keep some sample for visualization is 
        recommended.  ``None`` plot all the samples ( or examples) in the data 
    kind: str, Optional 
        type of visualization. Can be ``dendrogramm``, ``mbar`` or ``bar``. 
        ``corr`` plot  for dendrogram , :mod:`msno` bar,  :mod:`plt`
        and :mod:`msno` correlation  visualization respectively: 
            
            * ``bar`` plot counts the  nonmissing data  using pandas
            *  ``mbar`` use the :mod:`msno` package to count the number 
                of nonmissing data. 
            * dendrogram`` show the clusterings of where the data is missing. 
                leaves that are the same level predict one onother presence 
                (empty of filled). The vertical arms are used to indicate how  
                different cluster are. short arms mean that branch are 
                similar. 
            * ``corr` creates a heat map showing if there are correlations 
                where the data is missing. In this case, it does look like 
                the locations where missing data are corollated.
            * ``None`` is the default vizualisation. It is useful for viewing 
                contiguous area of the missing data which would indicate that 
                the missing data is  not random. The :code:`matrix` function 
                includes a sparkline along the right side. Patterns here would 
                also indicate non-random missing data. It is recommended to limit 
                the number of sample to be able to see the patterns. 
   
        Any other value will raise an error 
    
    Examples 
    --------
    >>> from watex.bases.base import Missing
    >>> data ='../../data/geodata/main.bagciv.data.csv' 
    >>> ms= Missing().fit(data) 
    >>> ms.plot_.fig_size = (12, 4 ) 
    >>> ms.plot () 
    
    """
    def __init__(self,
                   in_percent = False, 
                   sample = None, 
                   kind = None, 
                   drop_columns: List[str]=None,
                   **kws): 
  
        self.in_percent = in_percent
        self.kind = kind  
        self.sample= sample
        self.drop_columns=drop_columns 
        self.isnull_ = None
        
        super().__init__(**kws)
        
    @property 
    def isnull(self):
        """ Check the mean values  in the data  in percentge"""
        self.isnull_= self.data.isnull().mean(
            ) * 1e2  if self.in_percent else self.data.isnull().mean()
        
        return self.isnull_


    def plot(self, data: str | DataFrame=None , **kwd ):
        """
        Vizualize patterns in the missing data.
        
        Parameters 
        ------------
        data: Dataframe of shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N
        
        kind: str, Optional 
            kind of visualization. Can be ``dendrogramm``, ``mbar`` or ``bar`` plot 
            for dendrogram , :mod:`msno` bar and :mod:`plt` visualization 
            respectively: 
                
                * ``bar`` plot counts the  nonmissing data  using pandas
                *  ``mbar`` use the :mod:`msno` package to count the number 
                    of nonmissing data. 
                * dendrogram`` show the clusterings of where the data is missing. 
                    leaves that are the same level predict one onother presence 
                    (empty of filled). The vertical arms are used to indicate how  
                    different cluster are. short arms mean that branch are 
                    similar. 
                * ``corr` creates a heat map showing if there are correlations 
                    where the data is missing. In this case, it does look like 
                    the locations where missing data are corollated.
                * ``None`` is the default vizualisation. It is useful for viewing 
                    contiguous area of the missing data which would indicate that 
                    the missing data is  not random. The :code:`matrix` function 
                    includes a sparkline along the right side. Patterns here would 
                    also indicate non-random missing data. It is recommended to limit 
                    the number of sample to be able to see the patterns. 
       
                Any other value will raise an error 
            
        sample: int, Optional
            Number of row to visualize. This is usefull when data is composed of 
            many rows. Skrunked the data to keep some sample for visualization is 
            recommended.  ``None`` plot all the samples ( or examples) in the data 
            
        kws: dict 
            Additional keywords arguments of :mod:`msno.matrix` plot. 

        Return
        -------
        ``self``: :class:`watex.bases.base.Missing` instance 
            returns ``self`` for easy method chaining.
            
        
        Examples 
        --------
        >>> from watex.bases.base import Missing
        >>> data ='../../data/geodata/main.bagciv.data.csv' 
        >>> ms= Missing().fit(data) 
        >>> ms.plot_.fig_size = (12, 4 ) 
        >>> ms.plot () 
    
        """
        if data is not None: 
            self.data = data 
            
        ExPlot().missing( self.data, kind =  self.kind, 
                sample = self.sample, **kwd )
        return  self 

    @property 
    def get_missing_columns(self): 
        """ return columns with Nan Values """
        return list(self.data.columns [self.data.isna().any()]) 
    

    def drop (self, 
              data : str | DataFrame =None,  
              columns: List[str] = None, 
              inplace = False, 
              axis = 1 , 
              **kwd
              ): 
        """Remove missing data 
        
        Parameters 
        -----------
        data: Dataframe of shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N
        
        columns: str or list of str 
            columns to drop which contain the missing data. Can use the axis 
            equals to '1'.
            
        axis: {0 or 'index', 1 or 'columns'}, default 0
            Determine if rows or columns which contain missing values are 
            removed.
            * 0, or 'index' : Drop rows which contain missing values.
        
            * 1, or 'columns' : Drop columns which contain missing value.
            Changed in version 1.0.0: Pass tuple or list to drop on multiple 
            axes. Only a single axis is allowed.
        
        how: {'any', 'all'}, default 'any'
            Determine if row or column is removed from DataFrame, when we 
            have at least one NA or all NA.
            
            * 'any': If any NA values are present, drop that row or column.
            * 'all' : If all values are NA, drop that row or column.
            
        thresh: int, optional
            Require that many non-NA values. Cannot be combined with how.
        
        subset: column label or sequence of labels, optional
            Labels along other axis to consider, e.g. if you are dropping rows 
            these would be a list of columns to include.
        
        inplace: bool, default False
            Whether to modify the DataFrame rather than creating a new one.
            
        Returns 
        -------
        ``self``: :class:`watex.bases.base.Missing` instance 
            returns ``self`` for easy method chaining.
            
        """
        if data is not None: 
            self.data = data 
        if columns is not None: 
            self.drop_columns = columns 
            
        existfeatures(self.data , self.drop_columns, error ='raise')
        
        if self.drop_columns is None: 
            if inplace : 
                self.data.dropna (axis = axis , inplace = True, **kwd )
            else :  self.data = self.data .dropna (
                axis = axis , inplace = False, **kwd )
            
        elif self.drop_columns is not None: 
            if inplace : 
                self.data.drop (columns = self.drop_columns , 
                                axis = axis, inplace = True, 
                                **kwd)
            else : 
                self.data.drop (columns = self.columns , axis = axis , 
                                inplace = False , **kwd)

        
        return self 
    
    @property 
    def sanity_check (self): 
        """Ensure that we have deal with all missing values. The following 
        code returns a single boolean if there is any cell that is missing 
        in a DataFrame """
        
        return self.data.isna().any().any() 
    

    def replace (self, 
                 data:str |DataFrame = None , 
                 columns: List[str] = None,
                 fill_value: float = None , 
                 new_column_name: str= None, 
                 return_non_null: bool = False, 
                 **kwd): 
        """ 
        Replace the missing values to consider. 
        
        Use the :code:`coalease` function of :mod:`pyjanitor`. It takes a  
        dataframe and a list of columns to consider. This is a similar to 
        functionality found in Excel and SQL databases. It returns the first 
        non null value of each row. 
        
        Parameters 
        -----------
        data: Dataframe of shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N
        
        columns: str or list of str 
            columns to replace which contain the missing data. Can use the axis 
            equals to '1'.
            
        axis: {0 or 'index', 1 or 'columns'}, default 0
            Determine if rows or columns which contain missing values are 
            removed.
            * 0, or 'index' : Drop rows which contain missing values.
        
            * 1, or 'columns' : Drop columns which contain missing value.
            Changed in version 1.0.0: Pass tuple or list to drop on multiple 
            axes. Only a single axis is allowed.
            
         Returns 
         -------
         ``self``: :class:`watex.bases.base.Missing` instance 
             returns ``self`` for easy method chaining.
             
        """
        
        if data is not None: 
            self.data = data 
        existfeatures(self.data , columns )
        
        if return_non_null : 
            new_column_name = _assert_all_types(new_column_name, str  )
            
            if 'pyjanitor' not in sys.modules: 
                raise ModuleNotFoundError(" 'pyjanitor' is missing.Install it"
                                          " mannualy using conda or pip.")
            import pyjanitor as jn 
            return jn.coalease (self.data , 
                                columns = columns, 
                                new_column_name = new_column_name, 
                                )
        if fill_value is not None: 
            # fill missing values with a particular values. 
            
            try : 
                self.data = self.data .fillna(fill_value , **kwd)
            except : 
                if 'pyjanitor'  in sys.modules:
                    import pyjanitor as jn 
                    jn.fill_empty ( 
                        self.data , columns = columns or list(self.data.columns), 
                        value = fill_value 
                        )
            
        return self 
    

class Explore (Data): 
    """
    Exploratory data analysis 
    
    It is needed to create a model since it gives a feel for the data and also 
    at great excuses to meet and discuss issues with business units that 
    controls the data. 
    
    """
    
    def __init__( self, **kwd): 
        
        super().__init__(*kwd)



Data.__doc__="""\
Data base class

Typically, we train a model with a matrix of data. Note that pandas Dataframe 
is the most used because it is very nice to have columns lables even though 
Numpy arrays work as well. 

For supervised Learning for instance, suc as regression or clasification, our 
intent is to have a function that transforms features into a label. If we 
were to write this as an algebra formula, it would be look like:: 
    
    .. math::
        
        y = f(X)

:code:`X` is a matrix. Each row represent a `sample` of data or information 
about individual. Every columns in :code:`X` is a `feature`.The output of 
our function, :code:`y`, is a vector that contains labels (for classification)
or values (for regression). 

In Python, by convention, we use the variable name :code:`X` to hold the 
sample data even though the capitalization of variable is a violation of  
standard naming convention (see PEP8). 

Parameters 
-----------
{params.core.data}
{params.base.columns}
{params.base.axis}
{params.base.sample}
{params.base.kind}
{params.base.inplace}
{params.core.verbose}

Returns
-------
{returns.self}
   
Examples
--------

.. include:: ../docs/data.rst



""".format(
    params=_param_docs,
    returns=_core_docs["returns"],
)








































        