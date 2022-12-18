# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Created date: Sep 19 09:04:21 2022

"""
Hydrogeological module 
========================
Hydrogeological parameters of aquifer are the essential and crucial basic data 
in the designing and construction progress of geotechnical engineering and 
groundwater dewatering, which are directly related to the reliability of these 
parameters.

Created on Mon Sep 19 09:04:21 2022

"""

from __future__ import ( 
    division, 
    annotations 
    )
import warnings 
from abc import ABC, abstractclassmethod
from .._docstring import ( 
    _core_docs, 
    DocstringComponents 
    )
from ..exceptions import ( 
    NotFittedError, 
    StrataError
    )
from ..utils.hydroutils import ( 
    find_aquifer_groups, 
    find_similar_labels, 
    get_aquifer_section, 
    get_aquifer_sections, 
    # get_compressed_vector, 
    reduce_samples, 
    select_base_stratum,
    make_MXS_labels, 
    predict_NGA_labels

    )
from ..utils.funcutils import ( 
    sanitize_frame_cols, 
    to_numeric_dtypes, 
    smart_strobj_recognition, 
    repr_callable_obj, 
    is_in_if
    )
from ..utils.validator import check_array 

from .._watexlog import watexlog 

#-----------------------

_base_params = dict( 
    aqname="""
aqname: str, optional, 
    Name of aquifer group column. `aqname` allows to retrieve the 
    aquifer group `arr_aq` value in  a specific dataframe. Commonly
   `aqname` needs to be supplied when a dataframe is passed as a positional 
    or keyword argument. Note that it is not mandatory to have a group of 
    aquifer in the log data. It is needed only if the label similarity 
    needs to be calculated.    
    """, 
    sname="""
sname: str, optional 
    Name of column in the dataframe that contains the strata values. 
    Dont confuse 'sname' with 'stratum' which is the name of the valid 
    layer/rock in the array/Series of strata.     
    """, 
    )

_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"], 
    base= DocstringComponents(_base_params)
    )
#------------------------
class HGeol(ABC):
    """ 
    A branch of geology concerned with the occurrence, use, and functions of 
    surface water and groundwater. 
    
    Hydrogeology is the study of groundwater – it is sometimes referred to as
    geohydrology or groundwater hydrology. Hydrogeology deals with how water 
    gets into the ground (recharge), how it flows in the subsurface 
    (through aquifers) and how groundwater interacts with the surrounding soil 
    and rock (the geology).
    
    
    see also
    ---------

    Hydrogeologists apply this knowledge to many practical uses. They might:
        
        * Design and construct water wells for drinking water supply, irrigation 
            schemes and other purposes;
        * Try to discover how much water is available to sustain water supplies 
            so that these do not adversely affect the environment – for example, 
            by depleting natural baseflows to rivers and important wetland 
            ecosystems;
        * Investigate the quality of the water to ensure that it is fit for its 
            intended use; 
        * Where the groundwater is polluted, they design schemes to try and 
            clean up this pollution;
            Design construction dewatering schemes and deal with groundwater 
            problems associated with mining; Help to harness geothermal energy
            through groundwater-based heat pumps.
    """
    @abstractclassmethod 
    def __init__(
        self, 
        **kwd
        ): 
        self._logging = watexlog.get_watex_logger(self.__class__.__name__)


class HData: 

    def __init__(
        self,
        kname=None, 
        zname=None, 
        aqname=None, 
        sname=None, 
        verbose=0
        ): 
        self._logging = watexlog.get_watex_logger(self.__class__.__name__)
        self.kname=kname
        self.zname=zname
        self.aqname=aqname
        self.sname=sname
        self.verbose=verbose
    
    def fit(
        self, 
        data,  
        **fit_params
        ): 
        """ Fit Hydro-data and populate attributes. 
        
        Note that each column of the dataframe can be retrieved as an attribute
        value. The attribute maker replace all spaces in the items if exist
        in data columns with '_'. For instance, retrieving the 'layer thickness' 
        as an items in the data should be: 'layer_thickness' like:: 
            
            >>> from watex.datasets import load_hlogs 
            >>> from watex.methods.hydro import HData 
            >>> h=HData ().fit(load_hlogs().frame) 
            >>> h.layer_thickness # for retrieving 'layer thickness' 
            
        Parameters 
        -----------
        
        data : Dataframe of shape (n_samples, n_features)
             where `n_samples` is the number of data, expected to be the data 
             collected at different depths and `n_features` is the number of 
             columns (features) that supposed to be plot. 
             Note that `X` must include the ``depth`` columns. If not given a 
             relative depth should be created according to the number of 
             samples that composes `data`.
 
        fit_params: dict, 
            Additional keyword arguments passed to 
            :func:`~.watex.utils.funcutils.to_numeric_dtypes`. 
               
       Returns 
       -------
          self:  `HData` object instanciated for chaining methods. 
            
        """
        data = check_array (
            data, 
            force_all_finite= "allow-nan", 
            dtype =object , 
            input_name="Data", 
            to_frame=True, 
            )
        data = sanitize_frame_cols(data, fill_pattern= '_' )
        self.data_, nf, cf = to_numeric_dtypes(
            data , 
            return_feature_types= True, 
            verbose =self.verbose, 
            **fit_params 
            )
        self.feature_names_in_ = nf + cf 
        
        if len(cf )!=0:
            # sanitize the categorical values 
            for c in cf : self.data_ [c] = self.data_[c].str.strip() 
        for name, val in zip (("k", "z", "aq", "s"), (
                self.kname, self.zname, self.aqname, self.sname)): 
            if val: 
                c=val
                val = is_in_if (list(self.data_.columns), val, 
                                 error ='ignore')
                if val is None and self.verbose : 
                    warnings.warn(f" Invalid '{name}name'={c!r}. Name not "
                                  "found in the given dataset. None is set "
                                  "instead.")
                
            setattr (self, f"{name}_", 
                     self.data_[val[0]] if val else val 
                     )
            
        for name in self.data_.columns : 
            setattr (self, name, self.data_[name])
            
        return self 
    
    def squeeze_data (self, strategy="average", **rs_kws): 
        """ Compressed data by sample reducing 
        
        To compress many boreholes data, it is recommended to use 
        :func:`get_unique_section`. 
        
        Parameters 
        ---------- 
        
        sname: str, optional 
            Name of column in the dataframe that contains the strata values. 
            Dont confuse 'sname' with 'stratum' which is the name of the valid 
            layer/rock in the array/Series of strata. 
        
        strategy: str , default='average' or 'mean', 
            strategy used to select or compute the numerical data into a 
            singular series. It can be ['naive']. In that case , a single serie 
            if randomly picked up into the base strata data.
            
        rs_kws: dict, 
            keyword arguments passed to 
            :func:`~.watex.utils.hydroutils.reduce_samples`
            
        Returns 
        ----------
        sqdat: pandas.dataframes
            new dataframe with reducing samples. 
            
        """
        self.inspect 
        
        if self.sname is None: 
            raise StrataError (
                "'sname' cannot be none for data compressing. Refer to"
                " :func:`~watex.utils.hydroutils.reduce_samples` for"
                " pure examples.")
            
        sqdat = reduce_samples(
            self.data_, 
            sname= self.sname, 
            zname=self.zname, 
            kname =self.kname, 
            strategy = strategy,
            **rs_kws
            )[0]
        return sqdat 
    
    def get_base_stratum (self ): 
        """ Select the base stratum """
        self.inspect 
        
        self.base_stratum_ = select_base_stratum(
            self.data_,
            sname = self.sname , 
            stratum = None, 
            return_counts=False, 
            return_rate=False, 
            )
        return self.base_stratum_ 
    
    
    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'data_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 
    
    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        t =("kname", "zname", "aqname", "sname", "verbose" )
        outm = ( '<{!r}:' + ', '.join(
            [f"{k}={getattr(self, k)!r}" for k in t]) + '>' 
            ) 
        return  outm.format(self.__class__.__name__)
       
    
    def __getattr__(self, name):
        _getattr_(self, name)
           
HData.__doc__="""\
Hydro-Log data 

Hydro-log data is a mixed data composed of logging data, borehole data 
and geological data. To only used the logging data, it recommended to use 
:class:`~.watex.methods.hydro.Logging` instead. 


Parameters 
------------
{params.core.kname}
{params.core.zname}
{params.base.aqname}
{params.base.sname}

Examples 
----------
>>> from watex.datasets import load_hlogs 
>>> hd=HData (kname ='k', zname='depth_top', sname='strata_name', 
              aqname='aquifer_group', verbose =True ) 
>>> hd.fit(load_hlogs().frame)
>>> 
""".format (params =_param_docs , 
)     
    
    
class AqSection (HData): 
    def __init__(
            self,
            aqname=None, 
            kname=None, 
            zname= None, 
            **kws
            ): 
        super().__init__(
            kname =kname , 
            aqname= aqname, 
            zname= zname, 
            **kws
            )
    
    def findSection(self, z= None, depth_unit ="m"): 
        """ Find aquifer valid section (upper and lower section ) 
        
        Parameters 
        -----------
        z: array-like 1d, pandas.Series 
            Array of depth or a pandas series that contains the depth values. 
            Two  dimensional array or more is not allowed. However when `z` 
            is given as  a dataframe and `zname` is not supplied, an error 
            raises since `zname` is used to fetch and overwritten `z` 
            from the dataframe. 
            
        Returns 
        --------
        self.section_: list of float 
            valid upper and lower section in SI units (m) if depth values are 
            given in meters. 
        
        """
        self.inspect 
        
        self.section_ = get_aquifer_sections(
            self.data , 
            zname=self.zname, 
            kname= self.kname, 
            return_data= False, 
            return_index= False,  
            z=z, 
            )[0]
        if self.verbose: 
            print("### The valid section of aquifer is {} to {} {}."
                  .format(self.section_[0], self.section_[-1],
                          depth_unit)
                  )
        return self.section_ 
    
    
class MXS (HData): 
    def __init__(
        self, 
        kname=None, 
        aqname=None, 
        **kws
        ): 
        super().__init__(kname =kname, aqname =aqname, **kws)

    def predictNGA (
            self,
            n_components =2 ,  
            n_clusters = 3 , 
            random_state = 42, 
            keep_label_0 =False, 
            **npca_kws
            ): 
        """ Predict Naive Group of aquifer """
        self.inspect 
        
        from ..analysis.dimensionality import nPCA 
        from ..utils.mlutils import ( 
            naive_imputer, 
            naive_scaler 
        )
        
        X= to_numeric_dtypes(
            self.data_, pop_cat_features= True , verbose =self.verbose 
            )
        X= nPCA(naive_scaler (naive_imputer(X, mode = 'bi-impute' )), 
                n_components= n_components , 
                random_state=random_state ,
                view=False, 
                **npca_kws
                )
        self.yNGA_, self.cluster_centers_= predict_NGA_labels(
            X, n_clusters= n_clusters, 
            return_cluster_centers= True, 
            keep_label_0= keep_label_0 ,  
            verbose = self.verbose, 
            
            )
        return self.yNGA_
    
    
        
        
class Logging :
    """
    Logging class 
    
    Parameters 
    -----------
    zname: str, default='depth' or 'None'
        The name of the depth column in `data`. If the name 'depth' is not  
        specified as the main depth columns, an other name in the columns 
        that matches the depth can also be indicated so the function will put 
        aside this columm as depth column for plot purpose. If set to ``None``, 
        `zname` holds the name ``depth`` and assumes that depth exists in 
        `data` columns.
        
    kname: str, int
        Name of permeability coefficient columns. `kname` allows to retrieve the 
        permeability coefficient 'k' in  a specific dataframe. If integer is passed, 
        it assumes the index of the dataframe  fits the 'k' columns. Note that 
        integer value must not be out the dataframe size along axis 1. Commonly
       `kname` needs to be supplied when a dataframe is passed as a positional 
        or keyword argument. 
        
    """
    def __init__(
        self, 
        zname=None, 
        kname=None,
        verbose=0
        ):
        
        self._logging = watexlog.get_watex_logger(self.__class__.__name__)
        self.zname=zname 
        self.kname=kname
        self.verbose=verbose 
        
    def fit(self, data , **fit_params): 
        """
        Fit logging data and populate attributes 
        
        Parameters 
        -----------
        
        data : Dataframe of shape (n_samples, n_features)
             where `n_samples` is the number of data, expected to be the data 
             collected at different depths and `n_features` is the number of 
             columns (features) that supposed to be plot. 
             Note that `X` must include the ``depth`` columns. If not given a 
             relative depth should be created according to the number of 
             samples that composes `data`.
 
        fit_params: dict, 
            Additional keyword arguments passed to 
            :func:`~.watex.utils.funcutils.to_numeric_dtypes`. 
               
       Returns 
       -------
          self:  object instanciated for chaining methods. 
       
        """
        
        data = check_array (
            data, 
            force_all_finite= "allow-nan", 
            dtype =object , 
            input_name="data", 
            to_frame= True, 
            )
        self.data_= to_numeric_dtypes( 
            data , pop_cat_features= True, verbose =self.verbose, 
            **fit_params 
            )
        self.feature_names_in_ = list(self.data_ ) 
        
        return self 
    
    def plot (
        self, 
        normalize = False, 
        impute_nan= True, 
        log10=False, 
        posiy=None, 
        fill_value = None, 
        **plot_kws
        ):
        """ Plot the logging data 
        
        Parameters
        -----------
        
            
        normalize: bool, default = False
            Normalize all the data to be range between (0, 1) except the `depth`,    

        impute_nan: bool, default=True, 
            Replace the NaN values in the dataframe. Note that the default 
            behaviour for replacing NaN is the ``mean``. However if the argument 
            of `fill_value` is provided,the latter should be used to replace 'NaN' 
            in `X`. 
            
        log10: bool, default=False
            Convert values to log10. This can be usefull when using the logarithm 
            data. However, it seems not all the data can be used this operation, 
            for instance, a negative data. In that case, `column_to_skip` argument
            is usefull to provide so to skip that columns when converting values 
            to log10. 
            
        fill_value : str or numerical value, optional
            When strategy == "constant", fill_value is used to replace all
            occurrences of missing_values.
            If left to the default, fill_value will be 0 when imputing numerical
            data and "missing_value" for strings or object data types. If not 
            given and `impute_nan` is ``True``, the mean strategy is used instead.

        posiy: int, optional 
            the position to place the target plot `y` . By default the target plot 
            if given is located at the last position behind the logging plots.  
            
        """
        self.inspect 
        
        from ..utils.plotutils import plot_logging 
        
        plot_logging (
            self.data_, 
            tname = self.kname, 
            zname =self.zname,
            normalize = normalize, 
            impute_nan= impute_nan, 
            log10=log10, 
            posiy=posiy, 
            fill_value = fill_value, 
            **plot_kws
            )
        
        return self 
    
    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        return repr_callable_obj  (self)
       
    
    def __getattr__(self, name):
        _getattr_(self, name)

    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'data_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 
    
    
class AquiferGroup (HData):

    def __init__ (
            self, 
            kname =None, 
            aqname =None, 
            **kws
            ): 
        super().__init__(
            kname =kname,
            aqname=aqname, 
            **kws)
        self.kname =kname 
        self.aqname =aqname 
        
    

AquiferGroup.__doc__="""\
Group of Aquifer is mostly related to area information after multiple 
boreholes collected. 

However when predicted 'k' with a missing k-values using the Mixture 
Learning Strategy (MXS), we intend to solve this problem by creating 
a Naive Group of Aquifer (NGA) to compensate the missing k-values in the 
dataset. This could be a good idea to avoid introducing a lot of bias since 
the group of aquifer is mostly tied to the permeability coefficient 'k'. 
To do this, an unsupervised learning is used to predict the NGA labels then 
the NGA labels are used in turn to fill the missing k-values. The best 
strategy for operting this trick is to  seek for some importances between
the true k-values with their corresponding aquifer groups at each depth, 
and find the most representative group. Once the most representative group 
is found for each true label 'k', the group of aquifer can be renamed as 
the naive similarity with the true k-label. For instance if true k-value 
is the label 1 and label 1 is most representative with the group of aquifer
'IV', therefore this group can be replaced throughout the column 
with 'k1'+'IV=> i.e. 'k14'. This becomes a new label created and is used to 
fill the true label 'y_true' to become a MXS target ( include NGA label). 
Note that the true label with valid 'k-value' remained intach and unchanged.
The same process is done for label 2, 3 and so on. The selection of MXS 
label from NGA strongly depends on its preponderance or importance rate in 
the whole dataset. 

The following example is the demonstration to how to compute the group 
representativity in datasets. 

Parameters 
----------
g:dict, 
    Dictionnary compose of occurence between the true labels 
    and the group of aquifer  as a function of occurence and
    repesentativity 
Example 
--------
>>> from watex.utils import naive_imputer, read_data, reshape 
>>> from watex.datasets import load_hlogs 
>>> from watex.utils.hydroutils import classify_k, find_aquifer_groups 
>>> b= load_hlogs () #just taking the target names
>>> data = read_data ('data/boreholes/hf.csv') # read complete data
>>> y = data [b.target_names]
>>> # impute the missing values found in aquifer group columns
>>> # reshape 1d array along axis 0 for imputation 
>>> agroup_imputed = naive_imputer ( reshape (y.aquifer_group, axis =0 ) , 
...                                    strategy ='most_frequent') 
>>> # reshape back to array_like 1d 
>>> y.aquifer_group =reshape (agroup_imputed) 
>>> # categorize the 'k' continous value in 'y.k' using the default 
>>> # 'k' mapping func 
>>> y.k = classify_k (y.k , default_func =True)
>>> # get the group obj
>>> group_obj = find_aquifer_groups(y.k, y.aquifer_group) 
>>> group_obj 
_Group(Label=[' 1 ', 
             Preponderance( rate = '53.141  %', 
                           [('Groups', {'V': 0.32, 'IV': 0.266, 'II': 0.236, 
                                        'III': 0.158, 'IV&V': 0.01, 
                                        'II&III': 0.005, 'III&IV': 0.005}),
                            ('Representativity', ( 'V', 0.32)),
                            ('Similarity', 'V')])],
        Label=[' 2 ', 
              Preponderance( rate = ' 19.11  %', 
                           [('Groups', {'III': 0.274, 'II': 0.26, 'V': 0.26, 
                                        'IV': 0.178, 'III&IV': 0.027}),
                            ('Representativity', ( 'III', 0.27)),
                            ('Similarity', 'III')])],
        Label=[' 3 ', 
              Preponderance( rate = '27.749  %', 
                           [('Groups', {'V': 0.443, 'IV': 0.311, 'III': 0.245}),
                            ('Representativity', ( 'V', 0.44)),
                            ('Similarity', 'V')])],
             )
(2) Use the subjectivity and set the strata columns as default array 

>>> find_aquifer_groups(y.k, subjectivity=True, default_arr= X.strata_name ) 
_Group(Label=[' 1 ', 
             Preponderance( rate = '53.141  %', 
                           [('Groups', {'siltstone': 0.35, 'coal': 0.227, 
                                        'fine-grained sandstone': 0.158, 
                                        'medium-grained sandstone': 0.094, 
                                        'mudstone': 0.079, 
                                        'carbonaceous mudstone': 0.054, 
                                        'coarse-grained sandstone': 0.03, 
                                        'coarse': 0.01}),
                            ('Representativity', ( 'siltstone', 0.35)),
                            ('Similarity', 'siltstone')])],
        Label=[' 2 ', 
              Preponderance( rate = ' 19.11  %', 
                           [('Groups', {'mudstone': 0.288, 'siltstone': 0.205, 
                                        'coal': 0.192, 
                                        'coarse-grained sandstone': 0.137, 
                                        'fine-grained sandstone': 0.137, 
                                        'carbonaceous mudstone': 0.027, 
                                        'medium-grained sandstone': 0.014}),
                            ('Representativity', ( 'mudstone', 0.29)),
                            ('Similarity', 'mudstone')])],
        Label=[' 3 ', 
              Preponderance( rate = '27.749  %', 
                           [('Groups', {'mudstone': 0.245, 'coal': 0.226, 
                                        'siltstone': 0.217, 
                                        'fine-grained sandstone': 0.123, 
                                        'carbonaceous mudstone': 0.066, 
                                        'medium-grained sandstone': 0.066, 
                                        'coarse-grained sandstone': 0.057}),
                            ('Representativity', ( 'mudstone', 0.24)),
                            ('Similarity', 'mudstone')])],
             )                  
"""


def _getattr_(self, name):
    """ Isolated part of __getattr__ to reformat the attribute getter. """
    rv = smart_strobj_recognition(name, self.__dict__, deep =True)
    appender  = "" if rv is None else f'. Do you mean {rv!r}'
    
    if name =='table_': 
        err_msg =(". Call 'summary' method to fetch attribute 'table_'")
    else: err_msg =  f'{appender}{"" if rv is None else "?"}' 
    
    raise AttributeError (
        f'{self.__class__.__name__!r} object has no attribute {name!r}'
        f'{err_msg}'
        )











































    

