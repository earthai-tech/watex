# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
:mod:`~watex.methods.hydro` computes Hydrogeological parameters of aquifer 
that are the essential and crucial basic data in the designing and 
construction progress of geotechnical engineering and groundwater dewatering. 
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
    StrataError, 
    kError, 
    AquiferGroupError
    )
from ..utils.hydroutils import (
    find_aquifer_groups, 
    find_similar_labels, 
    get_aquifer_sections, 
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
    is_in_if, 
    )
from ..utils.validator import check_array 

from .._watexlog import watexlog 

__all__=["Hydrogeology", 
         "AqSection", 
         "AqGroup", 
         "MXS", 
         "Logging"
         ]
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

class HData(ABC):
    @abstractclassmethod 
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
        """
        Fit Hydro-data and populate attributes. 
        
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
            :func:`~watex.utils.funcutils.to_numeric_dtypes`. 
      
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
            :func:`~watex.utils.hydroutils.reduce_samples`
            
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
    
    def get_base_stratum (self , stratum=None ): 
        """Select the base stratum 
        
        Parameters
        -----------
        stratum: str, optional 
            Name of the base stratum. Must be self contain as an item of the 
            strata data. Note that if `stratum` is passed, the auto-detection of 
            base stratum is not triggered. It returns the same stratum.
        
        Returns
        ---------
        base_stratum : str
            the most recurrent stratum in the data and compute the rate of 
            occurrence. 
            
        """
        self.inspect 
        
        self.base_stratum_ = select_base_stratum(
            self.data_,
            sname = self.sname , 
            stratum =stratum, 
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
Hydro-Log data , Abstract Base class and can't be instanciated. 

Hydro-log data is a mixed data composed of logging data, borehole data 
and geological data. To only used the logging data, it recommended to use 
:class:`~.watex.methods.hydro.Logging` instead. 


Parameters 
------------
{params.core.kname}
{params.core.zname}
{params.base.aqname}
{params.base.sname}

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
    
    def findSection(
        self, 
        z= None, 
        depth_unit ="m"
        ): 
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
            self.data_ , 
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

AqSection.__doc__="""\
Aquifer section class 

Get the section of each aquifer from dataframe. 

The unique section 'upper' and 'lower' is the valid range of the whole 
data to consider as a  valid data. Indeed, the aquifer section computing 
is  necessary to shrunk the data of the whole boreholes. Mosly the data 
from the section is consided the valid data as the predictor Xr. Out of the
range of aquifers ection, data can be discarded or compressed to top Xr. 

Parameters 
------------
{params.base.aqname}
{params.core.kname}
{params.core.zname}

""".format(params =_param_docs )    

class MXS (HData): 
    def __init__(
        self, 
        kname=None, 
        aqname=None,
        threshold:float=None,
        method:str="naive", 
        trailer:str="*", 
        keep_label_0:bool=False,
        random_state:int=42,
        n_groups:int=3, 
        sep:str=None, 
        prefix=None,
        **kws
        ): 
        super().__init__(
        kname =kname, 
        aqname =aqname, 
        **kws
            )
        
        self.threshold=threshold 
        self.method=method
        self.n_groups=n_groups
        self.trailer=trailer
        self.keep_label_0=keep_label_0 
        self.random_state=random_state 
        self.sep=sep 
        self.prefix=prefix 
        
    def predictNGA (
        self,
        n_components:int=2 ,  
        return_label=False, 
        **NGA_kws
        ): 
        """ Predicts Naive Group of Aquifer from Hydro-Log data. 
        
        Parameters
        ------------
        n_components: int, default=2 
            Number of dimension to preserve. If`n_components` is ranged 
            between float 0. to 1., it indicates the number of variance 
            ratio to preserve. If ``None`` as default value the number of 
            variance to preserve is ``95%``.
        return_label: bool,default=False
            If `True`, return the NGA label predicted, otherwise return 
            :class:`~.MXS` instanciated object. if ``False``, NGA label 
            can be fetch using the attribute 
            :attr:`watex.hydro.MXS.yNGA_`
            
        NGA_kws: dict, 
            keyword argument passed to :func:`watex.utils.predict_NGA_labels`
        Returns 
        --------
        yNGA_ or self : arraylike-1d of naive group of aquifer or 
            :class:`~.MXS` instanciated object.
        
        Example 
        --------
        >>> from watex.datasets import load_hlogs 
        >>> from watex.methods.hydro import MXS 
        >>> hdata = load_hlogs ().frame 
        >>> # drop the 'remark' columns since there is no valid data 
        >>> hdata.drop (columns ='remark', inplace=True) 
        >>> mxs =MXS (kname ='k').fit(hdata) # specify the 'k' column  
        >>> y_pred = mxs.predictNGA(return_label=True )
        >>> y_pred [-12:] 
        Out[52]: array([1, 3, 1, 3, 3, 3, 3, 1, 3, 3, 3, 3])
        """
        self.inspect 
        
        from ..analysis.dimensionality import nPCA 
        from ..utils.mlutils import ( 
            naive_imputer, 
            naive_scaler 
        )
        
        X= to_numeric_dtypes(
            self.data_, 
            pop_cat_features= True , 
            verbose =self.verbose 
            )
        X= nPCA(naive_scaler (naive_imputer(X)), 
                n_components= n_components , 
                random_state=self.random_state,
                view=False, 
                return_X=True,
                plot_kws=dict(), 
               
                )
        self.yNGA_, self.cluster_centers_= predict_NGA_labels(
            X, n_clusters= self.n_groups, 
            return_cluster_centers= True, 
            keep_label_0= self.keep_label_0 ,  
            random_state= self.random_state,
            **NGA_kws
            )
        return self.yNGA_ if return_label else self 
    
    
    def makeyMXS (
        self, 
        y_pred=None, 
        func:callable=None,
        categorize_k= False, 
        default_func= False,  
        **mxs_kws
        ): 
        r""" Construct the MXS target :math:`y*`
        
        Parameters 
        -----------
        y_pred: Array-like 1d, pandas.Series
            Array composing the valid NGA labels. Note that NGA labels is  a 
            predicted labels mostly using the unsupervising learning. 
            
            :seealso: :func:`~predict_NGA_labels` for further details. 
        
        func: callable 
            Function to specifically map the permeability coefficient column 
            in the dataframe of serie. If not given, the default function can be 
            enabled instead from param `default_func`. 

        string: bool, 
            If set to "True", categorized map from 'k'  should be prefixed by "k". 
            However is string value is given , the prefix is changed according 
            to this label. 
            
        default_ufunc: bool, 
            Default function for mapping k is setting to ``True``. Note that, this 
            could probably not fitted your own data. So  it is recommended to 
            provide your own function for mapping 'k'. However the default 'k' 
            mapping is given as follow: 
                
            - k0 {0}: k = 0 
            - k1 {1}: 0 < k <= .01 
            - k2 {2}: .01 < k <= .07 
            - k3 {3}: k> .07 
            
         mxs_kws:dict, 
             Additional keyword arguments passed to 
             :func:`~.watex.utils.make_MXS_labels`. 
             
        Returns 
        --------
        MXS.mxs_labels_: array-like 1d `
             array like of MXS labels 
             
        Example 
        --------
        >>> from watex.datasets import load_hlogs 
        >>> from watex.methods.hydro import MXS 
        >>> hdata = load_hlogs ().frame 
        >>> # drop the 'remark' columns since there is no valid data 
        >>> hdata.drop (columns ='remark', inplace=True) 
        >>> mxs =MXS (kname ='k').fit(hdata) # specify the 'k'columns 
        >>> # we can predict the NGA labels and yMXS with single line 
        >>> # of code snippet using the default 'k' classification.
        >>> ymxs = mxs.predictNGA().makeyMXS(categorize_k=True, default_func=True)
        >>> mxs.yNGA_[:7] 
        ... array([2, 2, 2, 2, 2, 2, 2])
        >>> ymxs[:7]
        Out[40]: array([22, 22, 22, 22, 22, 22, 22])
        >>> mxs.mxs_group_classes_
        Out[56]: {1: 1, 2: 22, 3: 3} # transform classes 
        >>> mxs.mxs_group_labels_ 
        Out[57]: (2,)
        >>> # **comment: 
            # # only the label '2' is tranformed to '22' since 
            # it is the only one that has similariry with the true label 2 
        """
        self.inspect 
        
        if self.k_ is None: 
            raise kError ("'k' data for permeability coefficient cannot"
                        " be None. Specify the name of the column 'kname'"
                        " that fits the permeability coefficient values"
                        " in the hydro-log dataset."
            )

        if ( 
            not hasattr (self, 'yNGA_') 
            and y_pred is None
            ) : 
            raise AquiferGroupError (
                "y_pred for Naive Group of Aquifer (NGA) cannot be "
                " None. Use :meth:`~predictNGA` method or"
                " :func:`~.watex.utils.predict_NGA_labels` to"
                " predict NGA labels first."
                 )
        
        elif ( 
            hasattr (self, "yNGA_") 
            and y_pred is None 
            ): 
            y_pred = self.yNGA_ 
            
        MXS = make_MXS_labels(
            self.k_, 
            y_pred, 
            threshold= self.threshold, 
            trailer=self.trailer, 
            method=self.method, 
            return_groups=False, 
            return_obj= True, 
            kname=self.kname, 
            keep_label_0=self.keep_label_0,
            sep=self.sep, 
            prefix=self.prefix,
            inplace=False, 
            categorize_k=categorize_k, 
            default_func=default_func, 
            func=func, 
            **mxs_kws
            )
        for key in MXS.keys (): 
            setattr(self, key, MXS[key])
        return  MXS.mxs_labels_

    def labelSimilarity(
        self, 
        func:callable=None,
        categorize_k= False, 
        default_func= False, 
        **sm_kws
        ):
        """Find label similarities
        
        Parameters 
        -----------

        func: callable 
            Function to specifically map the permeability coefficient column 
            in the dataframe of serie. If not given, the default function can be 
            enabled instead from param `default_func`. 

        string: bool, 
            If set to "True", categorized map from 'k'  should be prefixed by "k". 
            However is string value is given , the prefix is changed according 
            to this label. 
            
        default_ufunc: bool, 
            Default function for mapping k is setting to ``True``. Note that, this 
            could probably not fitted your own data. So  it is recommended to 
            provide your own function for mapping 'k'. However the default 'k' 
            mapping is given as follow: 
                
            - k0 {0}: k = 0 
            - k1 {1}: 0 < k <= .01 
            - k2 {2}: .01 < k <= .07 
            - k3 {3}: k> .07 
        sm_kws:dict, 
            Additional keyword arguments passed to 
            :func:`~.watex.utils.find_similar_labels`.
            
        """
        
        self.inspect
        
        msg =("{0!r} data for {1} cannot be None. Specify the name of the "
              "column {2!r} that fits the {1} values in the hydro-log dataset."
              )
        if self.k_ is None: 
            raise kError (msg.format("k","permeability coefficient", "kname" ))
        if self.aq_ is None: 
            raise AquiferGroupError(msg.format(
                "aq", "aquifer groups", "aqname")
            )

        similar_labels= find_similar_labels(
            self.k_, 
            self.aq_, 
            threshold=self.threshold, 
            keep_label_0=self.keep_label_0, 
            method=self.method, 
            return_groups=False, 
            **sm_kws
            )
        return  similar_labels
    
MXS.__doc__="""\
Mixture Learning Strategy (MXS)    

The use of machine learning for k-parameter prediction seems an alternative
way to reduce the cost of data collection thereby saving money. However, 
the borehole data comes with a lot of missing k  since the parameter is 
strongly tied to the aquifer after the pumping test. In other words, the 
k-parameter collection is feasible if the layer in the well is an aquifer. 
Unfortunately, predicting some samples of k in a large set of missing data 
remains an issue using the classical supervised learning methods. We, 
therefore propose an alternative approach called a mixture learning 
strategy (MXS) to solve these double issues. It entails predicting upstream 
a naïve group of aquifers (NGA) combined with the real values k to 
counterbalance the missing values and yield an optimal prediction score. 
The method, first, implies the K-Means and Hierarchical Agglomerative 
Clustering (HAC) algorithms. K-Means and HAC are used for NGA label 
predicting necessary the MXS label merging. 


Parameters 
-----------

{params.core.kname} 
{params.base.aqname}

threshold: float, default=None 
    The threshold from which, label in 'k' array can be considered  
    similar than the one in NGA labels 'y_pred'. The default is 'None' which 
    means none rule is considered and the high preponderence or occurence 
    in the data compared to other labels is considered as the most 
    representative  and similar. Setting the rule instead by fixing 
    the threshold is recommended especially in a huge dataset.

n_groups : int, default=3
    The number of aquifer n_groups to form as well as the number of
    centroids to generate. If a idea about the number of aquifer group
    in the areas, it should be used instead. Hiwever, it is recommended
    to validate this number using the 'elbow plot' or the 'silhouette
    plot' or the Hierachical Agglomerative Clustering dendrogram. 
    Refer to :func:`~watex.utils.plot_elbow` or 
    :func:`~.watex.view.plotSilhouette` 
    or :func:~.watex.view.plotDendrogram` for plotting purpose. 
            
keep_label_0: bool, default=False
    The prediction already include the label 0. However, including 0 in 
    the predicted label refers to 'k=0' i.e. no permeability coefficient 
    equals to 0, which is not True in principle, because all rocks  have 
    a permeability coefficient 'k'. Here we considered 'k=0' as an undefined 
    permeability coefficient. Therefore, '0' , can be exclude since, it can 
    also considered as a missing 'k'-value. If predicted '0' is in the target 
    it should mean a missing 'k'-value rather than being a concrete label.  
    Therefore, to avoid any confusion, '0' is altered to '1' so the value 
    `+1` is used to move forward all class labels thereby excluding 
    the '0' label. To force include 0 in the label, set `keep_label_0` 
    to ``True``.
    
 sep: str, default'' 
     Separator between the true labels 'y_true' and predicted NGA labels.
     Sep is used to rewrite the MXS labels. Mostly the MXS labels is a 
     combinaison with the true label of permeability coefficient 'k' and 
     the label of NGA to compose new similarity labels. For instance 
     
     >>> true_labels=['k1', 'k2', 'k3'] ; NGA_labels =['II', 'I', 'UV']
     >>> # gives 
     >>> MXS_labels= ['k1_II', 'k2_I', 'k3_UV']
 
     where the seperator `sep` is set to ``_``. This happens especially 
     when one of the label (NGA or true_labels) is not a numeric datatype 
     and a similariy is found between 'k1' and 'II', 'k2' and 'I' and so on.
     
 prefix: str, default=''
     prefix is used to rename the true_labels i.e the true valid-k. For
     instance::
         >>> k_valid =[1, 2, ..] -> k_new = [k1, k2, ...]
     where 'k' is the prefix. 
     
 method: str ['naive', 'strict'], default='naive'
     The kind of strategy to compute the representativity of a label 
     in the predicted array 'y_pred'. It can also be 'strict'. Indeed:
     
     - ``naive`` computes the importance of the label by the number of its
         occurence for this specific label in the array 'y_true'. It does not 
         take into account of the occurence of other existing labels. This 
         is usefull for unbalanced class labels in `y_true`.
     - ``strict`` computes the importance of the label by the number of 
         occurence in the whole valid `y_true` i.e. under the total of 
         occurence of all the labels that exist in the whole 'arra_aq'. 
         This can give a suitable anaylse results if the data is not 
         unbalanced for each labels in `y_pred`.
         
 trailer: str, default='*'
     The Mixture strategy marker to differentiate the existing class label  
     in 'y_true' with the predicted labels 'y_pred' especially when  
     the the same class labels are also present the true label with the 
     same label-identifier name. This usefull  to avoid any confusion  for
     both labels  in `y_true` and `y_pred` for better demarcation and 
     distinction. Note that if the `trailer`is set to ``None`` and both 
     `y_true` and `y_pred` are numeric data, the labels in `y_pred` are 
     systematically renamed to be distinct with the ones in the 'y_true'. 
     For instance :: 
         
         >>> true_labels=[1, 2, 3] ; NGA_labels =[0, 1, 2]
         >>> # with trailer , MXS labels should be 
         >>>  MXS_labels= ['0', '1*', '2*', '3'] # 1 and 2 are in true_labels 
         >>> # with no trailer 
         >>> MXS_labels= [0, 4, 5, 3] # 1 and 2 have been changed to [4, 5]
         
{params.core.verbose}

Examples 
---------
>>> from watex.datasets import load_hlogs 
>>> from watex.methods.hydro import MXS 
>>> hdata= load_hlogs (as_frame =True) 
>>> # drop the 'remark' columns since there is no valid data 
>>> hdata.drop (columns ='remark', inplace =True)
>>> mxs = MXS (kname ='k').fit(hdata)
>>> # predict the default NGA 
>>> mxs.predictNGA() # default prediction with n_groups =3 
>>> # make MXS labels using the default 'k' categorization 
>>> ymxs=mxs.makeyMXS(categorize_k=True, default_func=True)
>>> mxs.yNGA_ [62:74] 
Out[43]: array([1, 2, 2, 2, 3, 1, 2, 1, 2, 2, 1, 2])
>>> ymxs[62:74] 
Out[44]: array([ 1, 22, 22, 22,  3,  1, 22,  1, 22, 22,  1, 22]) 
>>> # to get the label similariry , need to provide the 
>>> # the column name of aquifer group and fit again like 
>>> mxs = MXS (kname ='k', aqname ='aquifer_group').fit(hdata)
>>> sim = mxs.labelSimilarity() 
>>> sim 
Out[47]: [(0, 'II')] # group II and label 0 are very similar 
""" .format(
params =_param_docs 
)   

class Logging :
    """
    Logging class
    
    Only deal with numerical values. If categorical values are find in the 
    logging dataset, they should be discarded. 
    
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
        
    Examples 
    ----------
    >>> from watex.datasets import load_hlogs 
    >>> from watex.methods.hydro import Logging 
    >>> # get the logging data 
    >>> h = load_hlogs ()
    >>> h.feature_names
    Out[29]: 
    ['hole_id',
     'depth_top',
     'depth_bottom',
     'strata_name',
     'rock_name',
     'layer_thickness',
     'resistivity',
     'gamma_gamma',
     'natural_gamma',
     'sp',
     'short_distance_gamma',
     'well_diameter']
    >>> # we can fit to collect the valid logging data
    >>> log= Logging(kname ='k', zname='depth_top' ).fit(h.frame[h.feature_names])
    >>> log.feature_names_in_ # categorical features should be discarded.
    Out[33]: 
    ['depth_top',
     'depth_bottom',
     'layer_thickness',
     'resistivity',
     'gamma_gamma',
     'natural_gamma',
     'sp',
     'short_distance_gamma',
     'well_diameter']
    >>> log.plot ()
    Out[34]: Logging(zname= depth_top, kname= k, verbose= 0)
    >>> # plot log including the target y 
    >>> log.plot (y = h.frame.k , posiy =0 )# first position 
    Logging(zname= depth_top, kname= k, verbose= 0)
    
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
        
        
    def fit(
        self, 
        data, 
        **fit_params
        )->"Logging": 
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
        self: object instanciated for chaining methods. 
       
        """
        
        data = check_array (
            data, 
            force_all_finite= "allow-nan", 
            dtype =object , 
            input_name="data", 
            to_frame= True, 
            )
        self.data_= to_numeric_dtypes( 
            data , pop_cat_features= True, 
            verbose =self.verbose, 
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
    
class AqGroup (HData):
    def __init__ (
            self, 
            kname =None, 
            aqname =None,
            method="naive", 
            keep_label_0=False, 
            **kws
            ): 
        super().__init__(
            kname =kname,
            aqname=aqname, 
            **kws
            )
        self.method=method
        self.keep_label_0=keep_label_0 
        
    def findGroups (
        self , 
        method="naive", 
        default_arr = None, 
        **g_kws 
        ):
        """ Find the existing group between the permeability coefficient `k` 
        and the group of aquifer. 
        
        It computes the occurence between the true labels 
        and the group of aquifer  as a function of occurence and
        repesentativity.
        
        Parameters 
        ----------
        keep_label_0: bool, default=False
            The prediction already include the label 0. However, including 0 in 
            the predicted label refers to 'k=0' i.e. no permeability coefficient 
            equals to 0, which is not True in principle, because all rocks  have 
            a permeability coefficient 'k'. Here we considered 'k=0' as an undefined 
            permeability coefficient. Therefore, '0' , can be exclude since, it can 
            also considered as a missing 'k'-value. If predicted '0' is in the target 
            it should mean a missing 'k'-value rather than being a concrete label.  
            Therefore, to avoid any confusion, '0' is altered to '1' so the value 
            `+1` is used to move forward all class labels thereby excluding 
            the '0' label. To force include 0 in the label, set `keep_label_0` 
            to ``True``.
        
        method: str ['naive', 'strict'], default='naive'
            The kind of strategy to compute the representativity of a label 
            in the predicted array 'y_pred'. It can also be 'strict'. Indeed:
            
            - ``naive`` computes the importance of the label by the number of its
                occurence for this specific label in the array 'y_true'. It does not 
                take into account of the occurence of other existing labels. This 
                is usefull for unbalanced class labels in `y_true`.
            - ``strict`` computes the importance of the label by the number of 
                occurence in the whole valid `y_true` i.e. under the total of 
                occurence of all the labels that exist in the whole 'arra_aq'. 
                This can give a suitable anaylse results if the data is not 
                unbalanced for each labels in `y_pred`.
        Returns
        --------
        g: _Group: :class:`~.box._Group` class object 
            Use attribute `.groups` to find the group values. 
                 
        """
        self.inspect
        
        msg =("{0!r} data for {1} cannot be None. Specify the name of the "
              "column {2!r} that fits the {1} values in the hydro-log dataset."
              )
        if self.k_ is None: 
            raise kError (msg.format("k","permeability coefficient", "kname" ))
        if self.aq_ is None: 
            raise AquiferGroupError(msg.format(
                "aq", "aquifer groups", "aqname")
            )

        g= find_aquifer_groups(
            self.k_, self.aq_,
            kname=self.kname , 
            aqname = self.aqname,
            method=method, 
            **g_kws
            )
        return g 
    
AqGroup.__doc__="""\
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
Note that the true label with valid 'k-value' remained intact and unchanged.
The same process is done for label 2, 3 and so on. The selection of MXS 
label from NGA strongly depends on its preponderance or importance rate in 
the whole dataset. 

The following example is the demonstration to how to compute the group 
representativity in datasets. 

Parameters 
----------
{params.core.kname}
{params.base.aqname}

g:dict, 
    Dictionnary compose of occurence between the true labels 
    and the group of aquifer  as a function of occurence and
    repesentativity 
Example 
--------
>>> from watex.methods.hydro import AqGroup 
>>> hg = AqGroup (kname ='k', aqname='aquifer_group').fit(hdata ) 
>>> hg.findGroups () 
Out[25]: 
 _Group(Label=[' 0 ', 
                   Preponderance( rate = ' 100.0  %', 
                                [('Groups', {{'II': 1.0}}),
                                 ('Representativity', ( 'II', 1.0)),
                                 ('Similarity', 'II')])],
             )                 
""".format(params = _param_docs)

#XXX TODO 
class Hydrogeology(ABC):
    """ 
    A branch of geology concerned with the occurrence, use, and functions of 
    surface water and groundwater. 
    
    Hydrogeology is the study of groundwater – it is sometimes referred to as
    geohydrology or groundwater hydrology. Hydrogeology deals with how water 
    gets into the ground (recharge), how it flows in the subsurface 
    (through aquifers) and how groundwater interacts with the surrounding soil 
    and rock (the geology).
    
    Indeed, hydrogeologists apply this knowledge to many practical uses. 
    They might:
        
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
        
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def _getattr_(self, name):
    """ Isolated part of __getattr__ to reformat the attribute getter. """
    rv = smart_strobj_recognition(name, self.__dict__, deep =True)
    appender  = "" if rv is None else f'. Do you mean {rv!r}'
    
    if name =='yNGA_': 
        err_msg =(". Call 'predictNGA' method to fetch attribute 'yNGA_'")
    else: err_msg =  f'{appender}{"" if rv is None else "?"}' 
    
    raise AttributeError (
        f'{self.__class__.__name__!r} object has no attribute {name!r}'
        f'{err_msg}'
        )











































    

