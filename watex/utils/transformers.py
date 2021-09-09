# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Wed Jul 14 20:00:26 2021
# This module is a set of transformers for data preparing. It is  part of 
# the WATex preprocessing module which is released under a MIT- licence.
"""
Created on Mon Sep  6 17:53:06 2021

@author: @Daniel03
"""
# from __future__ import division 

import inspect
# import collections

import warnings 
import numpy as np 
import pandas as pd 
# from pandas.api.types import is_integer_dtype

from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.model_selection import train_test_split 
from sklearn.base import BaseEstimator, TransformerMixin 

# import watex.utils.exceptions as Wex 
from watex.utils._watexlog import watexlog 
from watex.analysis.features import categorize_flow 

import  watex.utils.ml_utils as mlfunc

__docformat__='restructuredtext'

_logger = watexlog().get_watex_logger(__name__)

class StratifiedWithCategoryAdder( BaseEstimator, TransformerMixin ): 
    """
    Stratified sampling transformer based on new generated category 
    from numerical attributes and return stratified trainset and test set.
    
    Arguments 
    ---------- 
        *base_num_feature*:str, 
            Numerical features to categorize. 
            
        *threshold_operator*: float, 
            The coefficient to divised the numerical features value to 
            normalize the data 
            
        *max_category*: Maximum value fits a max category to gather all 
            value greather than.
            
        *return_train*: bool, 
            return the whole stratified trainset if set to ``True``.
        
    Another way to stratify dataset is to get insights from the dataset and 
    to add a new category as additional mileage. From this new attributes,
    data could be stratified after categorizing numerical features. 
    Once data is tratified, the new category will be drop and return the 
    train set and testset stratified. For instance::  
        
        >>> from watex.utils.transformers import StratifiedWithCategoryAdder
        >>> stratifiedNumObj= StratifiedWithCatogoryAdder('flow')
        >>> stratifiedNumObj.fit_transform(X=df)
        >>> stats2 = stratifiedNumObj.statistics_
        
    Usage::
        
        In this example, we firstly categorize the `flow` attribute using 
        the ceilvalue (see :func:`~discretizeCategoriesforStratification`) 
        and groupby other values greater than the ``max_category`` value to the 
        ``max_category`` andput in the temporary features. From this features 
        the categorization is performed and stratified the trainset and 
        the test set.
        
    Note::
        
        If `base_num_feature` is not given, dataset will be stratified using 
        purely random sampling.
    """
    
    def __init__(self, base_num_feature=None, threshold_operator = 1., return_train=False,
                 max_category=3, n_splits=1, test_size=0.2, random_state=42):
        self._logging= watexlog().get_watex_logger(self.__class__.__name__)
        
        self.base_num_feature= base_num_feature
        self.return_train= return_train
        self.threshold_operator=  threshold_operator
        self.max_category = max_category 
        self.n_splits = n_splits 
        self.test_size = test_size 
        self.random_state = random_state 
        
        self.base_items_ =None 
        self.statistics_=None 
        
    def fit(self, X, y=None): 
        """ Fit method """
        return self
    
    def transform(self, X, y=None):
        """Transform data and populate inspections attributes 
            from hyperparameters."""

        if self.base_num_feature is not None:
            in_c= 'temf_'
            # discretize the new added category from the threshold value
            X = mlfunc.discretizeCategoriesforStratification(
                                             X,
                                            in_cat=self.base_num_feature, 
                                             new_cat=in_c, 
                                             divby =self.threshold_operator,
                                             higherclass = self.max_category
                 )

            self.base_items_ = list(
            X[in_c].value_counts().index.values)
        
            split = StratifiedShuffleSplit(self.n_splits, self.test_size, 
                                           self.random_state)
            
            for train_index, test_index  in split.split(X, X[in_c]): 
                strat_train_set = X.loc[train_index]
                strat_test_set = X.loc[test_index] 
                #keep a copy of all stratified trainset.
                strat_train_set_copy = X.loc[ np.delete(X.index, test_index)]
                
        train_set, test_set = train_test_split( X, test_size = self.test_size,
                                   random_state= self.random_state)
        
        if self.base_num_feature is None or self.n_splits==0:
            
            self._logging.info('Stratification not applied! Train and test sets'
                               'were purely generated using random sampling.')
            
            return train_set , test_set 
            
        if self.base_num_feature is not None:
            # get statistic from `in_c` category proportions into the 
            # the overall dataset 
            o_ =X[in_c].value_counts() /len(X)
            r_ = test_set[in_c].value_counts()\
                /len(test_set)
            s_ = strat_test_set[in_c].value_counts()\
                /len( strat_test_set)
            r_error , s_error = ((r_/ o_)-1)*100, ((s_/ o_)-1)*100
            
            self.statistics_ = np.c_[np.array(self.base_items_), 
                                     o_,
                                     r_,
                                     s_, 
                                     r_error,
                                     s_error
                                     ]
      
            self.statistics_ = pd.DataFrame(data = self.statistics_,
                                columns =[in_c, 
                                          'Overall',
                                          'Random', 
                                          'Stratified', 
                                          'Rand. %error',
                                          'strat. %error'
                                          ])
            
            # set a pandas dataframe for inspections attributes `statistics`.
            self.statistics_.set_index(in_c, inplace=True)
            
            # remove the add category into the set 
            for set in(strat_train_set_copy, strat_train_set, strat_test_set): 
                set.drop([in_c], axis=1, inplace =True)
                
            if self.return_train: 
                strat_train_set = strat_train_set_copy 
                
            return strat_train_set, strat_test_set 

    
class StratifiedUsingBaseCategory( BaseEstimator, TransformerMixin ): 
    """
    Transformer to stratified dataset to have data more representativce into 
    the trainset and the test set especially when data is not large enough.
    
    Arguments: 
    ----------
        *base_column*: str or int, 
            Hyperparameters and can be index of the base mileage(category)
            for stratifications. If `base_column` is None, will return 
            the purely random sampling.
        *test_size*: float 
            Size to put in the test set 
        *random_state*: shuffled number of instance in the overall dataset. 
            default is ``42``.
    
    Usage::
        
        If data is  not large enough especially relative number of attributes
        if much possible to run therisk of introducing a significant sampling 
        biais.Therefore strafied sampling is a better way to avoid 
         a significant biais of sampling survey. For instance:: 
            
            >>> from watex.utils.ml_utils import load_data 
            >>> df = load_data('data/geo_fdata')
            >>>stratifiedObj = StratifiedUsingBaseCategory(base_column='geol')
            >>> stratifiedObj.fit_transform(X=df)
            >>> stats= stratifiedObj.statistics_

    Note::
        An :attr:`~statictics_` inspection attributes is good way to observe 
        thetest set generated using purely random sampling and using the 
        stratified sampling. The stratified sampling has category 
        ``base_column``proportions almost indentical to those in the full 
        dataset whereas the testset generated using purely random sampling 
        is quite skewed. 
    """
    
    def __init__(self, base_column =None,test_size=0.2, random_state=42):
        self._logging= watexlog().get_watex_logger(self.__class__.__name__)
        
        self.base_column = base_column 
        self.test_size = test_size  
        self.random_state = random_state
        
        #create inspection attributes
        self.statistics_=None 
        self.base_flag_ =False 
        self.base_items_= None 
        
    def fit(self, X, y=None): 
        """ Fit method and populated isnpections attributes 
        from hyperparameters."""
        return self
    
    def transform(self, X, y=None):
        """ return dataset `trainset` and `testset` using stratified sampling. 
        
        If `base_column` not given will return the `trainset` and `testset` 
        using purely random sampling.
        """
        if self.base_column is None: 
            self.stratified = False 
            self._logging.debug(
                f'Base column is not given``{self.base_column}``.Test set'
                ' will be generated using purely random sampling.')
            
        train_set, test_set = train_test_split(
                X, test_size = self.test_size ,random_state= self.random_state )
        
        if self.base_column  is not None: 
            
            if isinstance(self.base_column, (int, float)):  # use index to find 
            # base colum name. 
                self.base_column = X.columns[int(self.base_column)]
                self.base_flag_ =True 
            elif isinstance(self.base_column, str):
                # check wether the column exist into dataframe
                for elm in X.columns:
                    if elm.find(self.base_column.lower())>=0 : 
                        self.base_column = elm
                        self.base_flag_=True
                        break 
                    
        if not self.base_flag_: 
            self._logging.debug(
                f'Base column ``{self.base_column}`` not found '
                f'in `{X.columns}`')
            warnings.warn(
                f'Base column ``{self.base_column}`` not found in '
                f'{X.columns}.Test set is generated using purely '
                'random sampling.')
            
        self.base_items_ = list(
            X[self.base_column].value_counts().index.values)
        
        if self.base_flag_: 
            strat_train_set, strat_test_set = \
                mlfunc.stratifiedUsingDiscretedCategories(X, self.base_column)
                
            # get statistic from `basecolumn category proportions into the 
            # the whole dataset, in the testset generated using purely random 
            # sampling and the test set generated using the stratified sampling.
            
            o_ =X[self.base_column].value_counts() /len(X)
            r_ = test_set [self.base_column].value_counts()/len(test_set)
            s_ = strat_test_set[self.base_column].value_counts()/len( strat_test_set)
            r_error , s_error = ((r_/ o_)-1)*100, ((s_/ o_)-1)*100
            
            self.statistics_ = np.c_[np.array(self.base_items_),
                                     o_,
                                     r_, 
                                     s_, 
                                     r_error,
                                     s_error]
      
            self.statistics_ = pd.DataFrame(data = self.statistics_,
                                columns =[self.base_column,
                                          'Overall', 
                                          'Random', 
                                          'Stratified',
                                          'Rand. %error',
                                          'strat. %error'
                                          ])
            
            # set a pandas dataframe for inspections attributes `statistics`.
            self.statistics_.set_index(self.base_column, inplace=True)
            
            return strat_train_set, strat_test_set 
        
        if not self.base_flag_: 
            return train_set, test_set 
        
class CategorizeFeatures(BaseEstimator, TransformerMixin ): 
    """ Transform numerical features into categorial features and return 
    a new array transformed. 
    
    Arguments 
    ----------
        *num_columns_properties*: list 
            list composed ofnumerical `features name`, list of 
            `features boundaries` with their `categorized names`
            
    Notes::
 
    From the boundaries values including, features values can be transformed.
    `num_columns_properties` is composed of::
        
        - `feature name` or index: eg:'flow`' or index of fllow ='12' 
        - `features boundaries`:eg:[0., 1., 3] which correspond to::
            
            - 0: features flow values with equal to 0. By default the begining 
                value like 0 is unranged.
            - 0-1: replace values ranged between 0 and 1. 
            - 1-3:replace values ranged between 1-3 
            - >3 : get all values greater than 3. by default categorize values 
                greater than  the last  values. 
            If the default classification is not suitable, create your own range
                values like:: 
                
                -[[0-1], [1-3], 3] (1)
        - `categorized names`: Be sure that if the value is provided as  without 
            ranging like (1). The number of `categorized values` must be 
            the size of the `features boundaries` +1. For instance, we try to 
            replace all numerical values in column `flow` by ::
                
                -FR0 : all fllow egal to 0. 
                -FR1: flow between 0-1 
                -FR2: flow between 1-3 
                -FR3: flow greater than 3. 
            As you can see the `features boundaries` [0., 1., 3]size is equal to 
            `categorized name`['FR0', 'FR1', 'FR2', 'FR3'] size +1. 
    Usage::
        
        Can categorize multiples features by setting each component explained 
        above as list of tuples. For instance we try to replace the both 
        numerical features `power` and `flow` in the dataframe by their 
        corresponding `features boundaries. Here is how to set  the 
        `num_columns_properties` like:: 
            
            num_columns_porperties =[
                ('flow', ([0, 1, 3], ['FR0', 'FR1', 'FR2', 'FR3'])),
                ('power', ([10, 30, 100], ['pw0', 'pw1', 'pw2', 'pw4']))
                ]
    :Example:
        
        >>> from watex.utils.transformers import  CategorizeFeatures
        >>> from watex.utils.ml_utils import load_data 
        >>> df= mlfunc.load_data('data/geo_fdata')
        >>> catObj = CategorizeFeatures(
            num_columns_properties=num_columns_porperties )
        >>> X= catObj.fit_transform(df)
        >>> catObj.in_values_
        >>> catObj.out_values_
    """
    def __init__(self, num_columns_properties=None): 
        self._logging= watexlog().get_watex_logger(self.__class__.__name__)
        
        self.num_columns_properties=num_columns_properties  
        
        self.base_columns_=None
        self.in_values_ = None
        self.out_values_ = None
        self.base_columns_ix_=None 
  
    def fit(self, X, y=None):
        
        self.base_columns_ = [n_[0] for  n_ in self.num_columns_properties]
        self.in_values_ = [n_[1][0] for  n_ in self.num_columns_properties]
        self.out_values_ = [n_[1][1] for  n_ in self.num_columns_properties]

        return self
    
    def transform(self, X, y=None) :
        """ Tranform the data and return new array. Can straightforwardly
        call :meth:`~TransformerMixin.fit_transform` inherited from 
        scikit_learn."""
        
        X_dtype =''
        if isinstance(self.base_columns_, (list, tuple)): 
            self.base_columns_ =np.array(self.base_columns_)
            
        if np.array(self.base_columns_).dtype in ['int', 'float']: 
            self.base_columns_.astype(np.int32)
            
            #in the case indexes are provided 
            self.base_columns_ix_ = self.base_columns_ 
            
        # check whether X is unique array or array_like.
        try: 
            X.shape[1]
        except IndexError: 
            if isinstance(X, pd.Series):
            # if X.__class__.__name__ =='Series': 
                X= X.values 
            X_dtype = 'unik__'
            
        except RuntimeError : 
            # handle other possible errors.
            X_dtype = 'unik__'
        else : 
            if X.shape[1]==1: 
                X=X.reshape((X.shape[0]),)
                X_dtype ='unik__'
                
        if X_dtype =='unik__': 
               X =  categorize_flow(X, self.in_values_[0],
                                    classes=self.out_values_[0] )
               self.base_columns_ix_ =(0,)
               
               return X
        # now 
        if isinstance(X, pd.DataFrame): 
            X= self.ascertain_mumerical_values(X)    
            
        if self.base_columns_ix_ is not None: 
            for ii, ix_ in enumerate(self.base_columns_ix_): 
                X[:, ix_]=categorize_flow(X[:, ix_], 
                                          self.in_values_[ii],
                                          classes=self.out_values_[ii]
                                          )
                
        self.base_columns_ =tuple(self.base_columns_)
        self.in_values_ = tuple(self.in_values_)
        self.out_values_ = tuple(self.out_values_)
        self.base_columns_ix_ = tuple(self.base_columns_ix_)
        
        return X 
    
    def ascertain_mumerical_values(self, X, y=None): 
        """ Retreive indexes from mumerical attributes and return a dataframe
        values especially if `X` is dataframe else returns values of array."""
        

        # ascertain dataframe whether there is an categorial values. 
        try:
            # if isinstance(X, pd.DataFrame)
            list_of_numerical_cols= X.select_dtypes(
                exclude=['object']).columns.tolist()
    
        except AttributeError: 
            # if 'numpy.ndarray' object has no attribute 'select_dtypes'
            list_of_numerical_cols= []
            
        t_=[]
        # return X if no numeruical columns found
        if len(list_of_numerical_cols) ==0 : 
            self._logging.info('`None`numerical columns detected.')
            warnings.warn('None numerical columns detected.It seems')
            
            return X.values 
        
        for bcol in self.base_columns_:
            for dfcols in list_of_numerical_cols: 
                if dfcols.lower() == bcol.lower(): 
                    t_.append(dfcols)
                    break 
           
        if len(t_) ==0: 
            self._logging.info(
                f'Numerical features `{self.base_columns_}`not found in'\
                '`{list_of_numerical_cols}`')
               
            return  X.values 
        
        # get base columns index 
        self.base_columns_ix_ =[ int(X.columns.get_loc(col_n))
                                for col_n in self.base_columns_]
        
        return X.values 


class CombinedAttributesAdder(BaseEstimator, TransformerMixin ):
    """ Combined attributes from features `ohmS` and `lwi`. 
    
    Create a new attributed using features index or litteral string operator.
    Inherits from scikit_learn `BaseEstimator`and `TransformerMixin` classes.
 
    Paramters
    ----------
    *add_attributes* : bool,
            Decide to add new features values by combining 
            numerical features operation. By default ease to 
            divided two numerical features.
                    
    *attributes_ix* : str or list of int,
            Divide two features from string litteral operation betwen or 
            list of features indexes. 
            
    Returns
    --------   
    X : np.ndarray, 
        A  new array contained the new data from the `attributes_ix` operation. 
        If `add_attributes` is set to ``False``, will return the same array like 
        beginning. 
    
    Notes
    ------
    A litteral string operator is a by default divided two numerical fetaures
    separated by the main word "_per_". For instance, to create a new
    feature based on the division of the features ``lwi`` and the feature 
    ``ohmS``, the litteral string operator is::
        
        attributes_ix='lwi_per_ohmS'
        
    Or it could be the indexes of both features in the array like:: 
        
        attributes_ix =[(10, 9)] 
    
    which means the `lwi` and `ohmS` are found at index ``10`` and ``9``
    respectively
    
    Furthermore, multiples operations can be set by adding mutiples litteral 
    string operator into a list like::
        
        attributes_ix = [ 'power_per_magnitude', 'ohmS_per_lwi']
        
    Example::
        
        >>> from watex.utils.transformers import CombinedAttributesAdder
        >>> from watex.utils.ml_utils import load_data 
        >>> df =load_data('data/geo_fdata')
        >>> addObj = CombinedAttributesAdder(add_attributes=True, 
                                     attributes_ix='lwi_per_ohmS')
        >>> addObj.fit_transform(df)
        >>> addObj.attributes_ix
        >>> addObj.attributes_names_
    """
    def __init__(self, add_attributes =False,attributes_ix = 'lwi_per_ohmS'):
        self._logging= watexlog().get_watex_logger(self.__class__.__name__)
        
        self.add_attributes = add_attributes  
        self.attributes_ix = attributes_ix

        self.attributes_names_ =[] 
        
    def fit(self, X, y=None ):
        return self 
    
    def transform(self, X, y=None):
        """ Tranform the data and return new array. Can straightforwardly
        call fit_transform method inherited from `TransformerMixin` class."""
        def weird_division(ix_):
            """ Replace 0. value to 1 in denominator for division 
            calculus."""
            return ix_ if ix_!=0. else 1
        
        if self.attributes_ix is not None: 
            if isinstance(self.attributes_ix , str): 
                self.attributes_ix = [self.attributes_ix]
                
            for attr_ in self.attributes_ix : 
                if isinstance(attr_, str):
                    # break str characters with
                    t_= attr_.replace('_', '').split('per')
                    self.attributes_names_.append(tuple(t_))
        
            
        if isinstance(X, pd.DataFrame): 
            if len(self.attributes_names_) !=0: 
                # get index from columns positions  and change the 
                # the litteral string operator to index values from 
                # columns names.
                self.attributes_ix = [(int(X.columns.get_loc(col_n[0])), 
                      int(X.columns.get_loc(col_n[1]) ))
                 for col_n in self.attributes_names_]
                
            X= X.values 
 
        if self.add_attributes: 
            for num_ix , deno_ix  in self.attributes_ix :
        
                try: 
                   
                    num_per_deno = X[:, num_ix] /X[:, deno_ix ]
                    
                except ZeroDivisionError:
                    # replace the existing 0 to 1 to operate a new division.
                    weird_divison_values = np.array(
                        list(map(weird_division, X[:, deno_ix ])))
                    num_per_deno = X[:, num_ix] /weird_divison_values
                    
                X= np.c_[X, num_per_deno ]
                   
        return X 
     
class DataFrameSelector(BaseEstimator, TransformerMixin):
    """ Select data from specific attributes for column transformer. 
    
    Select only numerical or categorial columns for operations. Work as the
    same like sckit-learn `make_colum_tranformer` 
    
    Parameters 
    ----------
        *attribute_names*: list or array_like 
            List of  the main columns to keep the data 
            
        *select_type*: str 
            Automatic numerical and categorial selector. If `select_type` is 
            ``num``, only numerical values in dataframe are retrieved else 
            ``cat`` for categorials attributes.
            
    Returns
    -------
        X: ndarray 
            New array with composed of data of selected `attribute_names`.
            
    :Example:
        
        >>> from watex.utils.transformers import DataFrameSelector 
        >>> from watex.utils.ml_utils import load_data   
        >>> df = mlfunc.load_data('data/geo_fdata')
        >>> XObj = DataFrameSelector(attribute_names=['power','magnitude','sfi'],
        ...                          select_type=None)
        >>> cdf = XObj.fit_transform(df)
    """  
    def __init__(self, attribute_names=None, select_type =None): 
        self._logging= watexlog().get_watex_logger(self.__class__.__name__)
        
        self.attribute_names = attribute_names 
        self.select_type = select_type 
        
    def fit(self, X, y=None): 
        return self
    
    def transform(self, X, y=None): 
        """ Transform data and return numerical or categorial values."""
       
        if isinstance(self.attribute_names, str): 
            self.attribute_names =[self.attribute_names]
            
        if self.attribute_names is not None: 
            t_= []
            for in_attr in self.attribute_names: 
                for attr_ in X.columns: 
                    if in_attr.lower()== attr_.lower(): 
                        t_.append(attr_)
                        break 
                    
            if len(t_)==0: 
                self._logging.warn(f' `{self.attribute_names}` not found in the'
                                   f'`{X.columns}')
                warnings.warn('None attribute in the dataframe match'
                              f'`{self.attribute_names}.')
                
            if len(t_) != len(self.attribute_names): 
                mm_= set(self.attribute_names).difference(set(t_))
                warnings.warn(
                    f'Value{"s" if len(mm_)>1 else""} {list(mm_)} not found.'
                    f" Only `{t_}`match{'es' if len(t_) <1 else ''}"
                    " the dataframe features.")
                self._logging.warning(
                    f'Only `{t_}` can be considered as dataframe attributes.')
                                   
            self.attribute_names =t_
            
            return X[self.attribute_names].values 
        
        try: 
            if self.select_type.lower().find('num')>=0:
                self.select_type =='num'
            elif self.select_type.lower().find('cat')>=0: 
                self.select_type =='cat'
            else: self.select_type =None 
            
        except:
            warnings.warn(f'`Select_type`` given argument ``{self.select_type}``'
                         ' seems to be wrong. Should defaultly return the '
                         'Dataframe value.', RuntimeWarning)
            self._logging.warnings('A given argument `select_type`seems to be'
                                   'wrong %s. Use ``cat`` or ``num`` for '
                                   'categorical or numerical attributes '
                                   'respectively.'% inspect.signature(self.__init__))
            self.select_type =None 
        
        if self.select_type is None:
            warnings.warn('Arguments of `%s` arguments %s are all None. Should'
                          ' returns the dataframe values.'% (repr(self),
                              inspect.signature (self.__init__)))
            
            self._logging.warning('Object arguments are None.'
                               ' Should return the dataframe values.')
            return X.values 
        
        if self.select_type =='num':
            obj_columns= X.select_dtypes(include='number').columns.tolist()

        elif self.select_type =='cat': 
            obj_columns= X.select_dtypes(include=['object']).columns.tolist() 
 
        self.attribute_names = obj_columns 
        
        return X[self.attribute_names].values 
        
    def __repr__(self):
        return self.__class__.__name__
        
        
                  
if __name__=='__main__': 
    # import matplotlib.pyplot as plt 
    # df =pd.read_csv('data/geo_fdata/_bagoue_civ_loc_ves&erpdata.csv')
    # print(df)
    df = mlfunc.load_data('data/geo_fdata')
    stratifiedNumObj= StratifiedWithCategoryAdder('flow', n_splits=1,
                                                  return_train=True)
    strat_train_set , strat_test_set = stratifiedNumObj.fit_transform(X=df)
    bag_train_set = strat_train_set.copy()
    test_label = bag_train_set['flow'].copy()


    # if test_label.__class__.__name__ =='Series': 
        
    #     print('yes')
    # catObj =CategorizeFeatures(num_columns_properties=[
    #                 ('flow', ([0., 1., 3.], ['FR0', 'FR1', 'FR2', 'FR3']))
    #                 ])
    # X= catObj.fit_transform(test_label)
    
    # dfObj = DataFrameSelector(attribute_names=None,
    #                           select_type='num')
    # cdf = dfObj.fit_transform(bag_train_set)
    # comObj =CombinedAttributesAdder(add_attributes=True, attributes_ix=[(6,5)])
    # xattr = comObj.fit_transform(cdf)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
