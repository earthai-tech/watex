# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   created date Wed Jul 14 20:00:26 2021
#   Edited on Mon Sep  6 17:53:06 2021
"""
Gives some efficient tools for data manipulation and transformation.
"""
from __future__ import division 

import inspect
import warnings 
import numpy as np 
import pandas as pd 
# from pandas.api.types import is_integer_dtype
from .exlib.sklearn  import ( 
    StratifiedShuffleSplit, 
    train_test_split,
    BaseEstimator,
    TransformerMixin,
    StandardScaler,
    MinMaxScaler,
    OrdinalEncoder,
    OneHotEncoder 
)
 
from ._watexlog import watexlog 
from .utils.funcutils import parse_attrs , to_numeric_dtypes
from .utils.mlutils import (  
    discretizeCategoriesforStratification, 
    stratifiedUsingDiscretedCategories, 
    existfeatures 
    )
from .utils.hydroutils import categorize_flow 

__docformat__='restructuredtext'

_logger = watexlog().get_watex_logger(__name__)

class StratifiedWithCategoryAdder( BaseEstimator, TransformerMixin ): 
    """
    Stratified sampling transformer based on new generated category 
    from numerical attributes and return stratified trainset and test set.
    
    Arguments 
    ---------- 
    *base_num_feature*: str, 
        Numerical features to categorize. 
        
    *threshold_operator*: float, 
        The coefficient to divised the numerical features value to 
        normalize the data 
        
    *max_category*: Maximum value fits a max category to gather all 
        value greather than.
        
    *return_train*: bool, 
        Return the whole stratified trainset if set to ``True``.
        usefull when the dataset is not enough. It is convenient to 
        train all the whole trainset rather than a small amount of 
        stratified data. Sometimes all the stratified data are 
        not the similar equal one to another especially when the dataset 
        is not enough.
        
    Another way to stratify dataset is to get insights from the dataset and 
    to add a new category as additional mileage. From this new attributes,
    data could be stratified after categorizing numerical features. 
    Once data is tratified, the new category will be drop and return the 
    train set and testset stratified. For instance::  
        
        >>> from watex.transformers import StratifiedWithCategoryAdder
        >>> stratifiedNumObj= StratifiedWithCatogoryAdder('flow')
        >>> stratifiedNumObj.fit_transform(X=df)
        >>> stats2 = stratifiedNumObj.statistics_
        
    Usage
    ------
    In this example, we firstly categorize the `flow` attribute using 
    the ceilvalue (see :func:`~discretizeCategoriesforStratification`) 
    and groupby other values greater than the ``max_category`` value to the 
    ``max_category`` andput in the temporary features. From this features 
    the categorization is performed and stratified the trainset and 
    the test set.
        
    Notes 
    ------
    If `base_num_feature` is not given, dataset will be stratified using 
    random sampling.
        
    """
    
    def __init__(self,
                 base_num_feature=None,
                 threshold_operator = 1.,
                 return_train=False,
                 max_category=3,
                 n_splits=1, 
                 test_size=0.2, 
                 random_state=42):
        
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
            X = discretizeCategoriesforStratification(
                                             X,
                                            in_cat=self.base_num_feature, 
                                             new_cat=in_c, 
                                             divby =self.threshold_operator,
                                             higherclass = self.max_category
                 )

            self.base_items_ = list(
            X[in_c].value_counts().index.values)
        
            split = StratifiedShuffleSplit(n_splits =self.n_splits,
                                           test_size =self.test_size, 
                                           random_state =self.random_state)
            
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
               
            # force to remove the temporary features for splitting in 
            # the original dataset

            if in_c in X.columns: 
                X.drop([in_c], axis =1, inplace=True)
               
            return strat_train_set, strat_test_set 

    
class StratifiedUsingBaseCategory( BaseEstimator, TransformerMixin ): 
    """
    Transformer to stratified dataset to have data more representativce into 
    the trainset and the test set especially when data is not large enough.
    
    Arguments 
    ----------
    *base_column*: str or int, 
        Hyperparameters and can be index of the base mileage(category)
        for stratifications. If `base_column` is None, will return 
        the purely random sampling.
        
    *test_size*: float 
        Size to put in the test set.
        
    *random_state*: shuffled number of instance in the overall dataset. 
        default is ``42``.
    
    Usage 
    ------
    If data is  not large enough especially relative number of attributes
    if much possible to run therisk of introducing a significant sampling 
    biais.Therefore strafied sampling is a better way to avoid 
     a significant biais of sampling survey. For instance:: 
        
        >>> from watex.transformers import StratifiedUsingBaseCategory 
        >>> from watex.utils.mlutils import load_data 
        >>> df = load_data('data/geo_fdata')
        >>> stratifiedObj = StratifiedUsingBaseCategory(base_column='geol')
        >>> stratifiedObj.fit_transform(X=df)
        >>> stats= stratifiedObj.statistics_

    Notes
    ------
    An :attr:`~.statictics_` inspection attribute is good way to observe 
    the test set generated using purely random and  the 
    stratified sampling. The stratified sampling has category 
    ``base_column`` proportions almost indentical to those in the full 
    dataset whereas the test set generated using purely random sampling 
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
                stratifiedUsingDiscretedCategories(X, self.base_column)
                
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
    """ Transform numerical features into categorical features and return 
    a new array transformed. 
    
    Arguments 
    ----------
    *num_columns_properties*: list 
        list composed ofnumerical `features name`, list of 
        `features boundaries` with their `categorized names`.
            
    Notes
    ------
    From the boundaries values including, features values can be transformed.
    `num_columns_properties` is composed of:
        
    - `feature name` or index equals to 'flow`' or index of flow ='12' 
    - `features boundaries` equals to ``[0., 1., 3]`` may correspond to:
        
        - 0: features flow values with equal to 0. By default the begining 
            value like 0 is unranged.
        - 0-1: replace values ranged between 0 and 1. 
        - 1-3:replace values ranged between 1-3 
        - >3 : get all values greater than 3. by default categorize values 
            greater than  the last  values. 
        If the default classification is not suitable, create your own range
            values like ``[[0-1], [1-3], 3] (1)``
            
    - `categorized names`: Be sure that if the value is provided as  without 
        ranging like (1). The number of `categorized values` must be 
        the size of the `features boundaries` +1. For instance, we try to 
        replace all numerical values in column `flow` by ::
            
            -FR0 : all fllow egal to 0. 
            -FR1: flow between 0-1 
            -FR2: flow between 1-3 
            -FR3: flow greater than 3. 
        As you can see the `features boundaries` [0., 1., 3]size is equal 
        to `categorized name`['FR0', 'FR1', 'FR2', 'FR3'] size +1. 
            
    Usage
    ------
    Can categorize multiples features by setting each component explained 
    above as list of tuples. For instance we try to replace the both 
    numerical features `power` and `flow` in the dataframe by their 
    corresponding `features` boundaries. Here is how to set  the 
    `num_columns_properties` :: 
        
        num_columns_porperties =[
            ('flow', ([0, 1, 3], ['FR0', 'FR1', 'FR2', 'FR3'])),
            ('power', ([10, 30, 100], ['pw0', 'pw1', 'pw2', 'pw4']))
            ]
            
    Examples
    --------
    >>> from watex.transformers import  CategorizeFeatures
    >>> from watex.utils.mlutils import load_data 
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

        return self
    
    def transform(self, X, y=None) :
        """ Transform the data and return new array. Can straightforwardly
        call :meth:`~.sklearn.TransformerMixin.fit_transform` inherited 
        from scikit_learn."""
        
        self.base_columns_ = [n_[0] for  n_ in self.num_columns_properties]
        self.in_values_ = [n_[1][0] for  n_ in self.num_columns_properties]
        self.out_values_ = [n_[1][1] for  n_ in self.num_columns_properties]
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
    """ Combined attributes from litteral string operators, indexes or names. 
    
    Create a new attribute using features index or litteral string operator.
    Inherits from scikit_learn `BaseEstimator`and `TransformerMixin` classes.
 
    Arguments 
    ----------
    *attribute_names* : list of str , optional
        List of features for combinaison. Decide to combine new feature
        values by from `operator` parameters. By default, the combinaison it 
        is ratio of the given attribute/numerical features. For instance, 
        ``attribute_names=['lwi', 'ohmS']`` will divide the feature 'lwi' by 
        'ohmS'.
                    
    *attributes_indexes* : list of int,
        index of each feature/feature for experience combinaison. User 
        warning should raise if any index does match the dataframe of array 
        columns.
            
    *operator*: str, default ='/' 
        Type of operation to perform. Can be ['/', '+', '-', '*', '%']  
        
    Returns
    --------   
    X : np.ndarray, 
        A  new array contained the new data from the `attrs_indexes` operation. 
        If `attr_names` and attr_indexes is ``None``, will return the same array 
        like beginning. 
    
    Notes
    ------
    A litteral string operator can be used. For instance dividing two numerical 
    features can be illustrated using the word "per" separated by underscore like 
    "_per_" For instance, to create a new feature based on the division of 
    the features ``lwi`` and ``ohmS``, the litteral string operator that holds
    the ``attribute_names`` could be::
        
        attribute_names='lwi_per_ohmS'
        
    The same litteral string is valid for multiplication (_mul_) , 
    substraction (_sub_) , modulo (_mod_) and addition (_add_). However, 
    indexes of features can also use rather than `attribute_names` providing
    the `operator` parameters. 
    
    Or it could be the indexes of both features in the array like 
    ``attributes_ix =[(10, 9)]`` which means the `lwi` and `ohmS` are
    found at index ``10`` and ``9``respectively. Furthermore, multiples 
    operations can be set by adding mutiples litteral string operator into a 
    list like ``attributes_ix = [ 'power_per_magnitude', 'ohmS_per_lwi']``.
        
    Examples 
    --------
    >>> import pandas as pd 
    >>> from watex.transformers import CombinedAttributesAdder
    >>> from watex.datasets.dload import load_bagoue 
    >>> X, y = load_bagoue (as_frame =True ) 
    >>> cobj = CombinedAttributesAdder (attribute_names='lwi_per_ohmS')
    >>> Xadded = cobj.fit_transform(X)
    >>> cobj.attribute_names_
    ... ['num',
         'name',
         'east',
         'north',
         'power',
         'magnitude',
         'shape',
         'type',
         'sfi',
         'ohmS',
         'lwi',
         'geol',
         'lwi_div_ohmS'] # new attributes with 'lwi'/'ohmS'
    >>> df0 = pd.DataFrame (Xadded, columns = cobj.attribute_names_)
    >>> df0['lwi_div_ohmS']
    ... 0           0.0
        1      0.000002
        2      0.000005
        3      0.000004
        4      0.000008
          
        426    0.453359
        427    0.382985
        428    0.476676
        429    0.457371
        430    0.379429
        Name: lwi_div_ohmS, Length: 431, dtype: object
    >>> cobj = CombinedAttributesAdder (
        attribute_names=['lwi', 'ohmS', 'power'], operator='+')
    >>> df0 = pd.DataFrame (cobj.fit_transform(X),
                            columns = cobj.attribute_names_)
    >>> df0.iloc [:, -1]
    ... 0      1777.165142
        1      1207.551531
        2         850.5625
        3      1051.943553
        4       844.095833
            
        426      1708.8585
        427      1705.5375
        428      1568.9825
        429     1570.15625
        430      1666.9185
        Name: lwi_add_ohmS_add_power, Length: 431, dtype: object
    >>> cobj = CombinedAttributesAdder (
        attribute_indexes =[1,6], operator='+')
    >>> df0 = pd.DataFrame (cobj.fit_transform(X), 
                            columns = cobj.attribute_names_)
    >>> df0.iloc [:, -1]
    ... 0        b1W
        1        b2V
        2        b3V
        3        b4W
        4        b5W
         
        426    b427W
        427    b428V
        428    b429V
        429    b430V
        430    b431V
        Name: name_add_shape, Length: 431, dtype: object
    """
    _op ={'times': ('times', 'prod', 'mul', '*', 'x'), 
            'add': ('add', '+', 'plus'), 
            'div': ('quot', '/', 'div', 'per'), 
            'sub': ('sub', '-', 'less'), 
            'mod': ('mod', '%'),
        }
    
    def __init__(
            self, 
            attribute_names =None, 
            attribute_indexes = None, 
            operator: str='/', 
        ):

        self.attribute_names=attribute_names
        self.attribute_indexes= attribute_indexes
        self.operator=operator
        self.attribute_names_=None 
        
    def fit(self, X, y=None ):
        """
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
        self: `CombinedAttributesAdder` instance 
            returns ``self`` for easy method chaining.
        
        """
        return self 
    
    def transform(self, X): 
        """ Tranform X and return new array with experience attributes
        combinaison. 
        
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
        --------
        X: NDarray,  Ndarray ( M x N+1 matrix) 
            returns X transformed (``M=m-samples``, & ``N=n+1-features``) 
            with attribute  combined. 
            
        .. versionadded:: 0.1.3
            
        """
        columns =[]
        self.operator = self._get_operator (
            self.operator or self.attribute_names)
        
        if self.operator is None: 
            warnings.warn("None or Invalid operator cannot be use for "
                          "attribute combinaisons.")
        if isinstance (self.attribute_names, str): 
            self.attribute_names_ = parse_attrs(self.attribute_names) 
        
        elif isinstance(self.attribute_names, 
                        (list, tuple, np.ndarray) ):
            self.attribute_names_ = self.attribute_names 
            
        if isinstance(X, pd.DataFrame) : 
            # asset wether attributes exists 
            # no raise errors, return the dataframe 
            if self.attribute_names_ : 
                existfeatures(X, self.attribute_names_  )
            # get the index of attributes from dataframe 
            if self.attribute_names_: 
                self.attribute_indexes = list(map (
                    lambda o: list(X.columns).index (o), self.attribute_names_)
                    ) 
                
            elif self.attribute_indexes : 
                # try :
                self.attribute_names_ = list(map (
                    lambda ix: list(X.columns)[ix], self.attribute_indexes)
                    ) 
                # except IndexError: 
                #     raise IndexError("List of index is out the range.")
                
            columns = X.columns 
            X= to_numeric_dtypes(X)
            X= X.values 
            
        if self.attribute_indexes: 
            X = self._operate(X)
        
        if self.attribute_names_ is not None: 
            self.attribute_names_ = list(columns) + ([
                f'_{self.operator}_'.join([ v for v in self.attribute_names_ ])
                ] if self._isfine else [])
        
        return X 
            
    def _get_operator (self, operator): 
        """ Get operator for combining  attribute """
        
        for k, v in self._op.items() :
            for o in v: 
                if operator.find(o) >=0 : 
                    self.operator = k 
                    return self.operator 
        return 
    
    def _operate (self,  X): 
        """ Operate data from indexes """
        def weird_division(ix_):
            """ Replace 0. value to 1 in denominator for division 
            calculus."""
            return ix_ if ix_!=0. else 1
        
        msg=("Unsupported operand type(s)! index provided {} doesn't match"
             " any numerical features. Experience combinaison attributes"
             " is not possible.")
        
        self._isfine=True 
        Xc =X[:, self.attribute_indexes]
        cb= Xc[:, 0 ] ; Xc=Xc[:,  1: ]
        
        for k in range (Xc.shape[1]): 
            try : 
                if self.operator =='mod': 
                    cb %= Xc[:, k]
                if self.operator =='add': 
                    cb += Xc[:, k]
                if self.operator =='sub': 
                    cb -= Xc[:, k]
                if self.operator =='div': 
                    # if the denominator contain nan or 0 
                    # a weird division is triggered and replace 
                    # the denominator by 1
                    try : 
                        cb /= Xc[:, k]
                    except ZeroDivisionError: 
                        wv= np.array(
                            list(map(weird_division, Xc[:, k])))
                        cb /=wv
    
                    except ( TypeError, RuntimeError, RuntimeWarning):
                        warnings.warn(msg.format(
                            self.attribute_indexes[1:][k])) 
                        
                if self.operator =='x': 
                    cb *= Xc[:, k]        
                    
            except: 
                warnings.warn(msg.format(self.attribute_indexes[1:][k])) 
                self._isfine =False          
            
        X =  np.c_[X, cb ]  if self._isfine else X 
        
        return X 

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """ Select data from specific attributes for column transformer. 
    
    Select only numerical or categorial columns for operations. Work as the
    same like sckit-learn `make_colum_tranformer` 
    
    Arguments  
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
            
    Examples 
    ---------
    >>> from watex.transformers import DataFrameSelector 
    >>> from watex.utils.mlutils import load_data   
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
                                   '`{X.columns}`.')
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
        
class FrameUnion (BaseEstimator, TransformerMixin) : 
    """ Unified categorial and numerical features after scaling and 
    and categorial features encoded.
    
    Use :class:`~watex.tranformers.DataframeSelector` class to define 
    the categorial features and numerical features.
    
    Arguments
    ---------
    num_attributes: list 
        List of numerical attributes 
        
    cat_attributes: list 
        list of categorial attributes 
        
    scale: bool 
        Features scaling. Default is ``True`` and use 
        `:class:~sklearn.preprocessing.StandarScaler` 
        
    imput_data: bool , 
        Replace the missing data. Default is ``True`` and use 
        :attr:`~sklearn.impute.SimpleImputer.strategy`. 
        
    param_search: bool, 
        If `num_attributes` and `cat_attributes`are None, the numerical 
        features and categorial features` should be found automatically.
        Default is ``True``
        
    scale_mode:bool, 
        Mode of data scaling. Default is ``StandardScaler``but can be 
        a ``MinMaxScaler`` 
        
    encode_mode: bool, 
        Mode of data encoding. Default is ``OrdinalEncoder`` but can be 
        ``OneHotEncoder`` but creating a sparse matrix. Once selected, 
        the new shape of ``X`` should be different from the original 
        shape. 
    
    Example
    ------- 
    >>> from watex.datasets import fetch_data 
    >>> from watex.utils.transformers import FrameUnion
    >>> X_= fetch_data ('Bagoue original').get('data=dfy1')
    >>> frameObj = FrameUnion(X_, encoding =OneHotEncoder)
    >>> X= frameObj.fit_transform(X_)
        
    """  
    def __init__(self,
                 num_attributes =None , 
                 cat_attributes =None,
                 scale =True,
                 imput_data=True,
                 encode =True, 
                 param_search ='auto', 
                 strategy ='median', 
                 scale_mode ='StandardScaler', 
                 encode_mode ='OrdinalEncoder' ): 
        
        self._logging = watexlog().get_watex_logger(self.__class__.__name__)
        
        self.num_attributes = num_attributes 
        self.cat_attributes = cat_attributes 
        self.param_search = param_search 
        self.imput_data = imput_data 
        self.strategy =strategy 
        self.scale = scale
        self.encode = encode 
        self.scale_mode = scale_mode
        self.encode_mode = encode_mode
        
        self.X_=None 
        self.X_num_= None 
        self.X_cat_ =None
        self.num_attributes_=None
        self.cat_attributes_=None 
        self.attributes_=None 
        
    def fit(self, X): 
        return self
    
    def transform(self, X, y=None): 
        """ Transform data and return X numerical and categorial encoded 
        values."""
        
        if self.scale_mode.lower().find('stand')>=0: 
            self.scale_mode = 'StandardScaler'
        elif self.scale_mode.lower().find('min')>=0: 
            self.scale_mode = 'MinMaxScaler'
        if self.encode_mode.lower().find('ordinal')>=0: 
            self.encode_mode = 'OrdinalEncoder'
            
        elif self.encode_mode.lower().find('hot') >=0: 
            self.encode_mode = 'OneHotEncoder'
            
        numObj = DataFrameSelector(attribute_names= self.num_attributes, 
                                         select_type='num')
        catObj =DataFrameSelector(attribute_names= self.cat_attributes, 
                                         select_type='cat')
        num_arrayObj = numObj.fit_transform(X)
        cat_arrayObj = catObj.fit_transform(X)
        self.num_attributes_ = numObj.attribute_names 
        self.cat_attributes_ = catObj.attribute_names 
        
        self.attributes_ = self.num_attributes_ + self.cat_attributes_ 
        
        self.X_num_= num_arrayObj.copy()
        self.X_cat_ =cat_arrayObj.copy()
        self.X_ = np.c_[self.X_num_, self.X_cat_]
        
        if self.imput_data : 
            from sklearn.impute import SimpleImputer
            imputer_obj = SimpleImputer(missing_values=np.nan, 
                                        strategy=self.strategy)
            num_arrayObj =imputer_obj.fit_transform(num_arrayObj)
            
        if self.scale :
            if self.scale_mode == 'StandardScaler': 
                scaler = StandardScaler()
            if self.scale_mode =='MinMaxScaler':
                scaler = MinMaxScaler()
        
            num_arrayObj = scaler.fit_transform(num_arrayObj)
            
        if self.encode : 
            if self.encode_mode =='OrdinalEncoder': 
                encoder = OrdinalEncoder()
            elif self.encode_mode =='OneHotEncoder':
                encoder = OneHotEncoder(sparse_output=True)
            cat_arrayObj= encoder.fit_transform(cat_arrayObj )
            # sparse matrix of type class <'numpy.float64'>' stored 
            # element in compressed sparses raw format . To convert the sense 
            # matrix to numpy array , we need to just call 'to_array()'.
            warnings.warn(f'Sparse matrix `{cat_arrayObj.shape!r}` is converted'
                          ' in dense Numpy array.', UserWarning)
            # cat_arrayObj= cat_arrayObj.toarray()

        try: 
            X= np.c_[num_arrayObj,cat_arrayObj]
            
        except ValueError: 
            # For consistency use the np.concatenate rather than np.c_
            X= np.concatenate((num_arrayObj,cat_arrayObj), axis =1)
        
        if self.encode_mode =='OneHotEncoder':
            warnings.warn('Use `OneHotEncoder` to encode categorial features'
                          ' generates a Sparse matrix. X is henceforth '
                          ' composed of sparse matrix. The new dimension is'
                          ' {0} rather than {1}.'.format(X.shape,
                             self.X_.shape), UserWarning)
            self._logging.info('X become a spared matrix. The new shape is'
                               '{X.shape!r} against the orignal '
                               '{self.X_shape!r}')
            
        return X
        




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
