# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations 
import re 
import sys 
import inspect 
import subprocess
# import operator 
import numpy as np
from collections import defaultdict
from warnings import warn

from ._watexlog import  watexlog
from ._docstring import DocstringComponents, _core_docs
from .typing import List, Optional, DataFrame 
from .utils.coreutils import _is_readable 
from .utils.funcutils import (_assert_all_types,  repr_callable_obj, 
                              smart_strobj_recognition, smart_format )
from .exlib.sklearn import ( clone, LabelEncoder, _name_estimators , 
                            BaseEstimator, ClassifierMixin )  
from .exceptions import NotFittedError
from .view.plot import ExPlot

__all__=[
    "Data", 
    "Missing", 
    "AdelineGradientDescent", 
    "AdelineStochasticGradientDescent", 
    "MajorityVoteClassifier", 
    "Perceptron", 
    "existfeatures", 
    "is_installing", 
    "selectfeatures" , 
    "get_params" 
    ]

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
    def __init__ (self, verbose: int =0): 
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
        additional keywords arguments from :func:`watex.utils.mlutils.selectfeatures`
 
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
    
    #XXX TODO # use logical and to quick merge two frames 
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
 
class Missing (Data) : 
    """ Deal with missing in Data 
    
    Most algorithms will not work with missing data. Notable exceptions are the 
    recent boosting libraries such as the XGBoost 
    (:doc:`watex.documentation.xgboost.__doc__`) CatBoost and LightGBM. 
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
    >>> from watex.base import Missing
    >>> data ='data/geodata/main.bagciv.data.csv' 
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
        >>> from watex.base import Missing
        >>> data ='data/geodata/main.bagciv.data.csv' 
        >>> ms= Missing().fit(data) 
        >>> ms.plot_.fig_size = (12, 4 ) 
        >>> ms.plot () 
    
        """
        if data is not None: 
            self.data = data 
            
        ExPlot().plotmissing( self.data, kind =  self.kind, 
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
    
class _Base:
    """Base class for all classes in watex for parameters retrievals

    Notes
    -----
    All class defined should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "watex classes should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this class and
            contained subobjects.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple classes as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

class Perceptron (_Base): 
    r""" Object oriented perceptron API class. Perceptron classifier 
    
    Inspired from Rosenblatt concept of perceptron rules. Indeed, Rosenblatt 
    published the first concept of perceptron learning rule based on the MCP 
    (McCulloth-Pitts) neuron model. With the perceptron rule, Rosenblatt 
    proposed an algorithm thar would automatically learn the optimal weights 
    coefficients that would them be multiplied by the input features in order 
    to make the decision of whether a neuron fires (transmits a signal) or not. 
    In the context of supervised learning and classification, such algirithm 
    could them be used to predict whether a new data points belongs to one 
    class or the other. 
    
    Rosenblatt initial perceptron rule and the perceptron algorithm can be 
    summarized by the following steps: 
        - initialize the weights at 0 or small random numbers. 
        - For each training examples, :math:`x^{(i)}`:
            - Compute the output value :math:`\hat{y}`. 
            - update the weighs. 
    the weights :math:`w` vector can be fromally written as: 
        .. math:: 
            
            w := w_j + \delta w_j
            
    Parameters 
    -----------
    eta: float, 
        Learning rate between (0. and 1.) 
    n_iter: int , 
        number of iteration passes over the training set 
    random_state: int, default is 42
        random number generator seed for random weight initialization.
        
    Attributes 
    ----------
    w_: Array-like, 
        Weight after fitting 
    errors_: list 
        Number of missclassification (updates ) in each epoch
    
        
    References
    ------------
    .. [1] Rosenblatt F, 1957, The perceptron:A perceiving and Recognizing
        Automaton,Cornell Aeoronautical Laboratory 1957
    .. [2] McCulloch W.S and W. Pitts, 1943. A logical calculus of Idea of 
        Immanent in Nervous Activity, Bulleting of Mathematical Biophysics, 
        5(4): 115-133, 1943.
    
    """
    def __init__(self, eta:float = .01 , n_iter: int = 50 , 
                 random_state:int = 42 ) :
        super().__init__()
        self.eta=eta 
        self.n_iter=n_iter 
        self.random_state=random_state 
        
    def fit(self , X, y ): 
        """ Fit the training data 
        
        Parameters 
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.
        y: array-like, shape (M, ) ``M=m-samples``, 
            train target; Denotes data that may be observed at training time 
            as the dependent variable in learning, but which is unavailable 
            at prediction time, and is usually the target of prediction. 
        
        Returns 
        --------
        self: `Perceptron` instance 
            returns ``self`` for easy method chaining.
        """
        rgen = np.random.RandomState(self.random_state)
        
        self.w_ = rgen.normal(loc=0. , scale =.01 , size = 1 + X.shape[1]
                              )
        self.erros_ =list() 
        for _ in range (self.n_iter):
            errors =0 
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi 
                self.w_[0] += update 
                errors  += int(update !=0.) 
            self.errors_.append(errors)
        
        return self 
    
    def net_input(self, X) :
        """ Compute the net input """
        return np.dot (X, self.w_[1:]) + self.w_[0] 

    def predict (self, X): 
        """
       Predict the  class label after unit step
        
        Parameters
        ----------
        X : Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.

        Returns
        -------
        ypred: predicted class label after the unit step  (1, or -1)

        """             
        return np.where (self.net_input(X) >=.0 , 1 , -1 )
    
    def __repr__(self): 
        """ Represent the output class """
        
        tup = tuple (f"{key}={val}".replace ("'", '') for key, val in 
                     self.get_params().items() )
        
        return self.__class__.__name__ + str(tup).replace("'", "") 
    

class MajorityVoteClassifier (BaseEstimator, ClassifierMixin ): 
    """
    A majority vote Ensemble classifier 
    
    Combine different classification algorithms associate with individual 
    weights for confidence. The goal is to build a stronger meta-classifier 
    that balance out of the individual classifiers weaknes on a particular  
    datasets. In more precise in mathematical terms, the weighs majority 
    vote can be expressed as follow: 
        
        .. math:: 
            
            \hat{y} = arg \max{i} \sum {j=1}^{m} w_j\chi_A (C_j(x)=1)
    
    where :math:`w_j` is a weight associated with a base classifier, C_j; 
    :math:`\hat{y}` is the predicted class label of the ensemble. :math:`A` is 
    the set of the unique class label; :math:`\chi_A` is the characteristic 
    function or indicator function which returns 1 if the predicted class of 
    the jth clasifier matches i (C_j(x)=1). For equal weights, the equation 
    is simplified as follow: 
        
        .. math:: 
            
            \hat{y} = mode {{C_1(x), C_2(x), ... , C_m(x)}}
            
    Parameters 
    ------------
    
    clfs: {array_like}, shape (n_classifiers)
        Differents classifier for ensembles 
        
    vote: str , ['classlabel', 'probability'], default is {'classlabel'}
        If 'classlabel' the prediction is based on the argmax of the class 
        label. Otherwise, if 'probability', the argmax of the sum of the 
        probabilities is used to predict the class label. Note it is 
        recommended for calibrated classifiers. 
        
    weights:{array-like}, shape (n_classifiers, ), Optional, default=None 
        If a list of `int` or `float`, values are provided, the classifier 
        are weighted by importance; it uses the uniform weights if 'weights' is
        ``None``.
        
    Attributes 
    ------------
    classes_: array_like, shape (n_classifiers) 
        array of classifiers withencoded classes labels 
    
    classifiers_: list, 
        list of fitted classifiers 
        
    Examples 
    ---------
    >>> from watex.exlib.sklearn import (
        LogisticRegression,DecisionTreeClassifier ,KNeighborsClassifier, 
         Pipeline , cross_val_score , train_test_split , StandardScaler , 
         SimpleImputer )
    >>> from watex.datasets import fetch_data 
    >>> from watex.base import MajorityVoteClassifier 
    >>> from watex.base import selectfeatures 
    >>> data = fetch_data('bagoue original').get('data=dfy1')
    >>> X0 = data.iloc [:, :-1]; y0 = data ['flow'].values  
    >>> # exclude the categorical value for demonstration 
    >>> # binarize the target y 
    >>> y = np.asarray (list(map (lambda x: 0 if x<=1 else 1, y0))) 
    >>> X = selectfeatures (X0, include ='number')
    >>> X = SimpleImputer().fit_transform (X) 
    >>> X, Xt , y, yt = train_test_split(X, y)
    >>> clf1 = LogisticRegression(penalty ='l2', solver ='lbfgs') 
    >>> clf2= DecisionTreeClassifier(max_depth =1 ) 
    >>> clf3 = KNeighborsClassifier( p =2 , n_neighbors=1) 
    >>> pipe1 = Pipeline ([('sc', StandardScaler()), 
                           ('clf', clf1)])
    >>> pipe3 = Pipeline ([('sc', StandardScaler()), 
                           ('clf', clf3)])
    
    (1) -> Test the each classifier results taking individually 
    
    >>> clf_labels =['Logit', 'DTC', 'KNN']
    >>> # test the results without using the MajorityVoteClassifier
    >>> for clf , label in zip ([pipe1, clf2, pipe3], clf_labels): 
            scores = cross_val_score(clf, X, y , cv=10 , scoring ='roc_auc')
            print("ROC AUC: %.2f (+/- %.2f) [%s]" %(scores.mean(), 
                                                     scores.std(), 
                                                     label))
    ... ROC AUC: 0.91 (+/- 0.05) [Logit]
        ROC AUC: 0.73 (+/- 0.07) [DTC]
        ROC AUC: 0.77 (+/- 0.09) [KNN]
    
    (2) _> Implement the MajorityVoteClassifier
    
    >>> # test the resuls with Majority vote  
    >>> mv_clf = MajorityVoteClassifier(clfs = [pipe1, clf2, pipe3])
    >>> clf_labels += ['Majority voting']
    >>> all_clfs = [pipe1, clf2, pipe3, mv_clf]
    >>> for clf , label in zip (all_clfs, clf_labels): 
            scores = cross_val_score(clf, X, y , cv=10 , scoring ='roc_auc')
            print("ROC AUC: %.2f (+/- %.2f) [%s]" %(scores.mean(), 
                                                     scores.std(), label))
    ... ROC AUC: 0.91 (+/- 0.05) [Logit]
        ROC AUC: 0.73 (+/- 0.07) [DTC]
        ROC AUC: 0.77 (+/- 0.09) [KNN]
        ROC AUC: 0.92 (+/- 0.06) [Majority voting] # give good score & less errors 
    """     
    
    def __init__(self, clfs, weights = None , vote ='classlabel'):
        
        self.clfs=clfs 
        self.weights=weights
        self.vote=vote 
        
        self.classifier_names_={}
  
    def fit(self, X, y):
        """
        Fit classifiers 
        
        Parameters
        ----------

        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.
        y: array-like, shape (M, ) ``M=m-samples``
            train target; Denotes data that may be observed at training time 
            as the dependent variable in learning, but which is unavailable 
            at prediction time, and is usually the target of prediction. 
        
        Returns 
        --------
        self: `MajorityVoteClassifier` instance 
            returns ``self`` for easy method chaining.
        """

        self._check_clfs_vote_and_weights ()
        
        # use label encoder to ensure that class start by 0 
        # which is important for np.argmax call in predict 
        self._labenc = LabelEncoder () 
        self._labenc.fit(y)
        self.classes_ = self._labenc.classes_ 
        
        self.classifiers_ = list()
        for clf in self.clfs: 
            fitted_clf= clone (clf).fit(X, self._labenc.transform(y))
            self.classifiers_.append (fitted_clf ) 
            
        return self 
    
    def predict(self, X):
        """
        Predict the class label of X 
        
        Parameters
        ----------
        
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.
            
        Returns
        -------
        maj_vote:{array_like}, shape (n_examples, )
            Predicted class label array 
            

        """
        if self.vote =='proba': 
            maj_vote = np.argmax (self.predict_proba(X), axis =1 )
        if self.vote =='label': 
            # collect results from clf.predict 
            preds = np.asarray(
                [clf.predict(X) for clf in self.classifiers_ ]).T 
            maj_vote = np.apply_along_axis(
                lambda x : np.argmax( 
                    np.bincount(x , weights = self.weights )), 
                    axis = 1 , 
                    arr= preds 
                    
                    )
            maj_vote = self._labenc.inverse_transform(maj_vote )
        
        return maj_vote 
    
    def predict_proba (self, X): 
        """
        Predict the class probabilities an return average probabilities which 
        is usefull when computing the the receiver operating characteristic 
        area under the curve (ROC AUC ). 
        
        Parameters
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.

        Returns
        -------
        avg_proba: {array_like }, shape (n_examples, n_classes) 
            weights average probabilities for each class per example. 

        """
        probas = np.asarray (
            [ clf.predict_proba(X) for clf in self.classifiers_ ])
        avg_proba = np.average (probas , axis = 0 , weights = self.weights ) 
        
        return avg_proba 
    
    def get_params( self , deep = True ): 
        """ Overwrite the get params from `_Base` class  and get 
        classifiers parameters from GridSearch . """
        
        if not deep : 
            return super().get_params(deep =False )
        if deep : 
            out = self.classifier_names_.copy() 
            for name, step in self.classifier_names_.items() : 
                for key, value in step.get_params (deep =True).items (): 
                    out['%s__%s'% (name, key)]= value 
        
        return out 
        
    def _check_clfs_vote_and_weights (self): 
        """ assert the existence of classifiers, vote type and the 
         classfifers weigths """
        l = "https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html"
        if self.clfs is None: 
            raise TypeError( "Expect at least one classifiers. ")

        if hasattr(self.clfs , '__class__') and hasattr(
                self.clfs , '__dict__'): 
            self.clfs =[self.clfs ]
      
        s = set ([ (hasattr(o, '__class__') and hasattr(o, '__dict__')) for o 
                  in self.clfs])
        
        if  not list(s)[0] or len(s)!=1:
            raise TypeError(
                "Classifier should be a class object, not {0!r}. Please refer"
                " to Scikit-Convention to write your own estimator <{1!r}>."
                .format('type(self.clfs).__name__', l)
                )
        self.classifier_names_ = {
            k : v for k, v  in _name_estimators(self.clfs)
            }
        
        regex= re.compile(r'(class|label|target)|(proba)')
        v= regex.search(self.vote)
        if v  is None : 
            raise ValueError ("Vote argument must be 'probability' or "
                              "'classlabel', got %r"%self.vote )
        if v is not None: 
            if v.group (1) is not None:  
                self.vote  ='label'
            elif v.group(2) is not None: 
                self.vote  ='proba'
           
        if self.weights and len(self.weights)!= len(self.clfs): 
           raise ValueError(" Number of classifier must be consistent with "
                            " the weights. got {0} and {1} respectively."
                            .format(len(self.clfs), len(self.weights))
                            )
            
        
class AdelineStochasticGradientDescent (_Base) :
    r""" Adaptative Linear Neuron Classifier  with batch  (stochastic) 
    gradient descent 
    
    A stochastic gradient descent is apopular alternative wich is sometimes 
    also cal iterative or online gradient descent. Instead of updating the 
    weights based on the sum of accumulated erros over all training examples 
    :math:`x^{(i)}`: 
        
        .. math:: 
            
            \delta w: \sum{i} (y^{(i)} -\phi( z^{(i)}))x^(i)
            
    the weights are updated incremetally for each training examples: 
        
        .. math:: 
            
            \neta(y^{(i)} - \phi(z^{(i)})) x^{(i)}
            
    Parameters 
    -----------
    eta: float, 
        Learning rate between (0. and 1.) 
    n_iter: int, 
        number of iteration passes over the training set 
    suffle: bool, 
        shuffle training data every epoch if True to prevent cycles. 

    random_state: int, default is 42
        random number generator seed for random weight initialization.
        
    Attributes 
    ----------
    w_: Array-like, 
        Weight after fitting 
    cost_: list 
        Sum of squares cost function (updates ) in each epoch
        
    See also 
    ---------
    AdelineGradientDescent: :class:`watex.base.AdelineGradientDescent` 
    
    References 
    -----------
    .. [1] Windrow and al., 1960. An Adaptative "Adeline" Neuron Using Chemical
        "Memistors", Technical reports Number, 1553-2,B Windrow and al., 
        standford Electron labs, Standford, CA,October 1960. 
            
    """
    def __init__(self, eta:float = .01 , n_iter: int = 50 , shuffle=True, 
                 random_state:int = 42 ) :
        super().__init__()
        self.eta=eta 
        self.n_iter=n_iter 
        self.shuffle=shuffle 
        self.random_state=random_state 
        
        self.w_initialized =False 
        
    def fit(self , X, y ): 
        """ Fit the training data 
        
        Parameters 
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.
        y: array-like, shape (M, ) ``M=m-samples``, 
            train target; Denotes data that may be observed at training time 
            as the dependent variable in learning, but which is unavailable 
            at prediction time, and is usually the target of prediction. 
        
        Returns 
        --------
        self: `Perceptron` instance 
            returns ``self`` for easy method chaining.
        """   
        self._init_weights (X.shape[1])
        self.cost_=list() 
        for i in range(self.n_iter ): 
            if self.shuffle: 
                X, y = self._shuffle (X, y) 
            cost =[] 
            for xi , target in zip(X, y) :
                cost.append(self._update_weights(xi, target)) 
            avg_cost = sum(cost)/len(y) 
            self.cost_.append(avg_cost) 
        
        return self 
    
    def partial_fit(self, X, y):
        """
        Fit training data without reinitialising the weights 
        
        Parameters
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.
        y: array-like, shape (M, ) ``M=m-samples``, 
            train target; Denotes data that may be observed at training time 
            as the dependent variable in learning, but which is unavailable 
            at prediction time, and is usually the target of prediction. 
        
        Returns 
        --------
        self: `Perceptron` instance 
            returns ``self`` for easy method chaining.

        """
        if not self.w_initialized : 
           self._init_weights (X.shape[1])
          
        if y.ravel().shape [0]> 1: 
            for xi, target in zip(X, y):
                self._update_weights (xi, target) 
        else: 
            self._update_weights (X, y)
                
        return self 
    
    def _shuffle (self, X, y):
        """
        Shuffle training data 
        
        Parameters
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.
        y: array-like, shape (M, ) ``M=m-samples``, 
            train target; Denotes data that may be observed at training time 
            as the dependent variable in learning, but which is unavailable 
            at prediction time, and is usually the target of prediction. 

        Returns
        -------
        Training and target data shuffled  

        """
        r= self.rgen.permutation(len(y)) 
        return X[r], y[r]
    
    def _init_weights (self, m): 
        """
        Initialize weights with small random numbers 

        Parameters
        ----------
        m : int 
           random number for weights initialization .

        """
        self.rgen =  np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=.0 , scale=.01, size = 1+ m) 
        self.w_initialized = True 
        
    def _update_weights (self, X, y):
        """
        Adeline learning rules to update the weights 

        Parameters
        ----------
        X : Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set for initializing
        y :array-like, shape (M, ) ``M=m-samples``, 
            train target for initializing 

        Returns
        -------
        cost: list,
            sum-squared errors 

        """
        output = self.activation (self.net_input(X))
        errors =(y - output ) 
        self.w_[1:] += self.eta * X.dot(errors) 
        cost = errors **2 /2. 
        
        return cost 
    
    def net_input (self, X):
        """
        Compute the net input X 
        
        Parameters
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.

        Returns
        -------
       weight net inputs 

        """
        return np.dot (X, self.w_[1:]) + self.w_[0] 

    def activation (self, X):
        """
        Compute the linear activation 

        Parameters
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.

        Returns
        -------
        X: activate NDArray 

        """
        return X 
    
    def predict (self, X):
        """
        Predict the  class label after unit step
        
        Parameters
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.

        Returns
        -------
        ypred: predicted class label after the unit step  (1, or -1)
        """
        return np.where (self.activation(self.net_input(X))>=0. , 1, -1)
    
    def __repr__(self): 
        """ Represent the output class """
        
        tup = tuple (f"{key}={val}".replace ("'", '') for key, val in 
                     self.get_params().items() )
        
        return self.__class__.__name__ + str(tup).replace("'", "") 
    
class AdelineGradientDescent (_Base): 
    r"""Adaptative Linear Neuron Classifier 
    
    ADAptative LInear NEuron (Adaline) was published by Bernard Widrow and 
    his doctoral studentTeed Hoff only a few uears after Rosenblatt's 
    perceptron algorithm. It can be  considered as impovrment of the latter 
    Windrow and al., 1960.
    
    Adaline illustrates the key concepts of defining and minimizing continuous
    cost function. This lays the groundwork for understanding more advanced 
    machine learning algorithm for classification, such as Logistic Regression, 
    Support Vector Machines,and Regression models.  
    
    The key difference between Adaline rule (also know as the WIdrow-Hoff rule) 
    and Rosenblatt's perceptron is that the weights are updated based on linear 
    activation function rather than unit step function like in the perceptron. 
    In Adaline, this linear activation function :math:`\phi(z)` is simply 
    the identifu function of the net input so that:
        
        .. math:: 
            
            \phi (w^Tx)= w^Tx 
    
    while the linear activation function is used for learning the weights. 
    
    Parameters 
    -----------
    eta: float, 
        Learning rate between (0. and 1.) 
    n_iter: int , 
        number of iteration passes over the training set 
    random_state: int, default is 42
        random number generator seed for random weight initialization.
        
    Attributes 
    ----------
    w_: Array-like, 
        Weight after fitting 
    cost_: list 
        Sum of squares cost function (updates ) in each epoch
        
    
    References 
    -----------
    .. [1] Windrow and al., 1960. An Adaptative "Adeline" Neuron Using Chemical
        "Memistors", Technical reports Number, 1553-2,B Windrow and al., 
        standford Electron labs, Standford, CA,October 1960. 
        
    """
    def __init__(self, eta:float = .01 , n_iter: int = 50 , 
                 random_state:int = 42 ) :
        super().__init__()
        self.eta=eta 
        self.n_iter=n_iter 
        self.random_state=random_state 
        
    def fit(self , X, y ): 
        """ Fit the training data 
        
        Parameters 
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.
        y: array-like, shape (M, ) ``M=m-samples``, 
            train target; Denotes data that may be observed at training time 
            as the dependent variable in learning, but which is unavailable 
            at prediction time, and is usually the target of prediction. 
        
        Returns 
        --------
        self: `Perceptron` instance 
            returns ``self`` for easy method chaining.
        """
        rgen = np.random.RandomState(self.random_state)
        
        self.w_ = rgen.normal(loc=0. , scale =.01 , size = 1 + X.shape[1]
                              )
        self.cost_ =list()    
        
        for i in range (self.n_iter): 
            net_input = self.net_input (X) 
            output = self.activation (net_input) 
            errors =  ( y -  output ) 
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum() 
            cost = (errors **2 ).sum() / 2. 
            self.cost_.append(cost) 
        
        return self 
    
    def net_input (self, X):
        """
        Compute the net input X 
        
        Parameters
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.

        Returns
        -------
       weight net inputs 

        """
        return np.dot (X, self.w_[1:]) + self.w_[0] 

    def activation (self, X):
        """
        Compute the linear activation 

        Parameters
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.

        Returns
        -------
        X: activate NDArray 

        """
        return X 
    
    def predict (self, X):
        """
        Predict the  class label after unit step
        
        Parameters
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.

        Returns
        -------
        ypred: predicted class label after the unit step  (1, or -1)
        """
        return np.where (self.activation(self.net_input(X))>=0. , 1, -1)
    
    def __repr__(self): 
        """ Represent the output class """
        
        tup = tuple (f"{key}={val}".replace ("'", '') for key, val in 
                     self.get_params().items() )
        
        return self.__class__.__name__ + str(tup).replace("'", "") 
        
def get_params (obj: object 
                ) -> dict: 
    """
    Get object parameters. 
    
    Object can be callable or instances 
    
    :param obj: object , can be callable or instance 
    
    :return: dict of parameters values 
    
    :examples: 
    >>> from sklearn.svm import SVC 
    >>> from watex.base import get_params 
    >>> sigmoid= SVC (
        **{
            'C': 512.0,
            'coef0': 0,
            'degree': 1,
            'gamma': 0.001953125,
            'kernel': 'sigmoid',
            'tol': 1.0 
            }
        )
    >>> pvalues = get_params( sigmoid)
    >>> {'decision_function_shape': 'ovr',
         'break_ties': False,
         'kernel': 'sigmoid',
         'degree': 1,
         'gamma': 0.001953125,
         'coef0': 0,
         'tol': 1.0,
         'C': 512.0,
         'nu': 0.0,
         'epsilon': 0.0,
         'shrinking': True,
         'probability': False,
         'cache_size': 200,
         'class_weight': None,
         'verbose': False,
         'max_iter': -1,
         'random_state': None
     }
    """
    if hasattr (obj, '__call__'): 
        cls_or_func_signature = inspect.signature(obj)
        PARAMS_VALUES = {k: None if v.default is (inspect.Parameter.empty 
                         or ...) else v.default 
                    for k, v in cls_or_func_signature.parameters.items()
                    # if v.default is not inspect.Parameter.empty
                    }
    elif hasattr(obj, '__dict__'): 
        PARAMS_VALUES = {k:v  for k, v in obj.__dict__.items() 
                         if not (k.endswith('_') or k.startswith('_'))}
    
    return PARAMS_VALUES

def is_installing (
        module: str , 
        upgrade: bool=True , 
        action: bool=True, 
        DEVNULL: bool=False,
        verbose: int=0,
        **subpkws
    )-> bool: 
    """ Install or uninstall a module/package using the subprocess 
    under the hood.
    
    Parameters 
    ------------
    module: str,
        the module or library name to install using Python Index Package `PIP`
    
    upgrade: bool,
        install the lastest version of the package. *default* is ``True``.   
        
    DEVNULL:bool, 
        decline the stdoutput the message in the console 
    
    action: str,bool 
        Action to perform. 'install' or 'uninstall' a package. *default* is 
        ``True`` which means 'intall'. 
        
    verbose: int, Optional
        Control the verbosity i.e output a message. High level 
        means more messages. *default* is ``0``.
         
    subpkws: dict, 
        additional subprocess keywords arguments 
    Returns 
    ---------
    success: bool 
        whether the package is sucessfully installed or not. 
        
    Example
    --------
    >>> from watex.base import is_installing
    >>> is_installing(
        'tqdm', action ='install', DEVNULL=True, verbose =1)
    >>> is_installing(
        'tqdm', action ='uninstall', verbose =1)
    """
    #implement pip as subprocess 
    # refer to https://pythongeeks.org/subprocess-in-python/
    if not action: 
        if verbose > 0 :
            print("---> No action `install`or `uninstall`"
                  f" of the module {module!r} performed.")
        return action  # DO NOTHING 
    
    success=False 

    action_msg ='uninstallation' if action =='uninstall' else 'installation' 

    if action in ('install', 'uninstall', True) and verbose > 0:
        print(f'---> Module {module!r} {action_msg} will take a while,'
              ' please be patient...')
        
    cmdg =f'<pip install {module}> | <python -m pip install {module}>'\
        if action in (True, 'install') else ''.join([
            f'<pip uninstall {module} -y> or <pip3 uninstall {module} -y ',
            f'or <python -m pip uninstall {module} -y>.'])
        
    upgrade ='--upgrade' if upgrade else '' 
    
    if action == 'uninstall':
        upgrade= '-y' # Don't ask for confirmation of uninstall deletions.
    elif action in ('install', True):
        action = 'install'

    cmd = ['-m', 'pip', f'{action}', f'{module}', f'{upgrade}']

    try: 
        STDOUT = subprocess.DEVNULL if DEVNULL else None 
        STDERR= subprocess.STDOUT if DEVNULL else None 
    
        subprocess.check_call(
            [sys.executable] + cmd, stdout= STDOUT, stderr=STDERR,
                              **subpkws)
        if action in (True, 'install'):
            # freeze the dependancies
            reqs = subprocess.check_output(
                [sys.executable,'-m', 'pip','freeze'])
            [r.decode().split('==')[0] for r in reqs.split()]

        success=True
        
    except: 

        if verbose > 0 : 
            print(f'---> Module {module!r} {action_msg} failed. Please use'
                f' the following command: {cmdg} to manually do it.')
    else : 
        if verbose > 0: 
            print(f"{action_msg.capitalize()} of `{module}` "
                      "and dependancies was successfully done!") 
        
    return success
 
def existfeatures (df, features, error='raise'): 
    """Control whether the features exists or not  
    
    :param df: a dataframe for features selections 
    :param features: list of features to select. Lits of features must be in the 
        dataframe otherwise an error occurs. 
    :param error: str - raise if the features don't exist in the dataframe. 
        *default* is ``raise`` and ``ignore`` otherwise. 
        
    :return: bool 
        assert whether the features exists 
    """
    isf = False  
    
    error= 'raise' if error.lower().strip().find('raise')>= 0  else 'ignore' 

    if isinstance(features, str): 
        features =[features]
        
    features = _assert_all_types(features, list, tuple, np.ndarray)
    set_f =  set (features).intersection (set(df.columns))
    if len(set_f)!= len(features): 
        nfeat= len(features) 
        msg = f"Feature{'s' if nfeat >1 else ''}"
        if len(set_f)==0:
            if error =='raise':
                raise ValueError (f"{msg} {smart_format(features)} "
                                  f"{'does not' if nfeat <2 else 'dont'}"
                                  " exist in the dataframe")
            isf = False 
        # get the difference 
        diff = set (features).difference(set_f) if len(
            features)> len(set_f) else set_f.difference (set(features))
        nfeat= len(diff)
        if error =='raise':
            raise ValueError(f"{msg} {smart_format(diff)} not found in"
                             " the dataframe.")
        isf = False  
    else : isf = True 
    
    return isf  
    
def selectfeatures (
        df: DataFrame,
        features: List[str] =None, 
        include = None, 
        exclude = None,
        coerce: bool=False,
        **kwd
        ): 
    """ Select features  and return new dataframe.  
    
    :param df: a dataframe for features selections 
    :param features: list of features to select. Lits of features must be in the 
        dataframe otherwise an error occurs. 
    :param include: the type of data to retrieved in the dataframe `df`. Can  
        be ``number``. 
    :param exclude: type of the data to exclude in the dataframe `df`. Can be 
        ``number`` i.e. only non-digits data will be keep in the data return.
    :param coerce: return the whole dataframe with transforming numeric columns.
        Be aware that no selection is done and no error is raises instead. 
        *default* is ``False``
    :param kwd: additional keywords arguments from `pd.astype` function 
    
    :ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html
    """
    
    if features is not None: 
        existfeatures(df, features, error ='raise')
    # change the dataype 
    df = df.astype (float, errors ='ignore', **kwd) 
    # assert whether the features are in the data columns
    if features is not None: 
        return df [features] 
    # raise ValueError: at least one of include or exclude must be nonempty
    # use coerce to no raise error and return data frame instead.
    return df if coerce else df.select_dtypes (include, exclude) 