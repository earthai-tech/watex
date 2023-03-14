# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Created on Thu Oct 13 14:52:26 2022

@author: Daniel
"""
class Boxspace(dict):  
    """Is a container object exposing keys as attributes.
    
    BowlSpace objects are sometimes used as an output for functions and methods.
    They extend dictionaries by enabling values to be accessed by key,
    `Boxspace["value_key"]`, or by an attribute, `Boxspace.value_key`.
    Another option is to use Namespace of collection modules as: 
        
        >>> from collections import namedtuple
        >>> Boxspace = namedtuple ('Boxspace', [< attribute names >] )
        
    However the explicit class that inhers from build-in dict is easy to 
    handle attributes and to avoid multiple error where the given name 
    in the `names` attributes does not match the expected attributes to fetch. 
    
    Examples
    --------
    >>> from watex.utils.box import Boxspace 
    >>> bs = Boxspace(pkg='watex',  objective ='give water', version ='0.1.dev')
    >>> bs['pkg']
    ... 'watex'
    >>> bs.pkg
    ... 'watex'
    >>> bs.objective 
    ... 'give water'
    >>> bs.version
    ... '0.1.dev'
    """

    def __init__(self, **kws):
        super().__init__(kws)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        # Overriding __setstate__ to be a noop has the effect of
        # ignoring the pickled __dict__
        pass
    
class _Group:
    """ Group of Aquifer is mostly related to area information after multiple 
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
    >>> from watex.utils import naive_imputer, read_data , reshape 
    >>> from watex.datasets import load_hlogs 
    >>> from watex.utils.hydroutils import classify_k, find_aquifer_groups 
    >>> b= load_hlogs () #just taking the target names
    >>> data = read_data ('data/boreholes/hf.csv') # read complete data
    >>> y = data [b.target_names]
    >>> # impute the missing values found in aquifer group columns
    >>> # reshape 1d array along axis 0 for imputation 
    >>> agroup_imputed = naive_imputer ( reshape (y.aquifer_group, axis =0 ) , 
                                        strategy ='most_frequent') 
    >>> # reshape back to array_like 1d 
    >>> y.aquifer_group =reshape (agroup_imputed) 
    >>> # categorize the 'k' continous value in 'y.k' using the default 
    >>> # 'k' mapping func 
    >>> y.k = classify_k (y.k , default_func =True)
    >>> # get the group obj
    >>> group_obj = find_aquifer_groups(y.k, y.aquifer_group,  ) 
    >>> group_obj 
    ... _Group(Label=[' 1 ', 
                       Preponderance( rate = '53.141  %', 
                                    [('Groups', {'V': 0.32, 'IV': 0.266, 
                                                 'II': 0.236, 'III': 0.158, 
                                                 'IV&V': 0.01, 'II&III': 0.005, 
                                                 'III&IV': 0.005}),
                                     ('Representativity', ( 'V', 0.32)),
                                     ('Similarity', 'V')])],
                 Label=[' 2 ', 
                       Preponderance( rate = ' 19.11  %', 
                                    [('Groups', {'III': 0.274, 'II': 0.26, 
                                                 'V': 0.26, 'IV': 0.178, 
                                                 'III&IV': 0.027}),
                                     ('Representativity', ( 'III', 0.27)),
                                     ('Similarity', 'III')])],
                 Label=[' 3 ', 
                       Preponderance( rate = '27.749  %', 
                                    [('Groups', {'V': 0.443, 'IV': 0.311, 
                                                 'III': 0.245}),
                                     ('Representativity', ( 'V', 0.44)),
                                     ('Similarity', 'V')])],
                 )
                                      
    """
    def __init__ (self, g=None, /  ): 
        self.g_ = g
        
    @property 
    def g(self): 
        return self.g_
    @property 
    def similarity (self): 
        """return label similarities with NGA labels  """
        return (
            (label, list(rep_val [1])[0] ) 
            for label, rep_val in self.g_.items()
                )
    @property 
    def preponderance (self): 
        """ Returns label occurences in the datasets """
        return   (
            (label, rep_val[0]) 
            for label, rep_val in self.g_.items()
             )
    @property 
    def representativity (self): 
        """ Returns the representativity of each labels"""
        return ( (label, round(rep_val[1].get(list(rep_val [1])[0]), 2))  
                    for label, rep_val in self.g_.items()
                     )
    @property 
    def groups (self): 
        """Return groups for each label """
        return ((label, {k: v for k, v in repr_val[1].items()}) 
                  for label, repr_val in self.g_.items () 
                  )

    def __repr__ (self ) :
        return  self.__class__.__name__  + "(" +  self._format (
            self.g) + "{:>13}".format(")")

    def _format (self, gdict): 
        """ Format representativity of Aquifer groups 
        Parameters 
        ----------
        gdict: dict, 
            Dictionnary compose of occurence of the group as a function
            of aquifer group repesentativity 
        """
        ag=[]
        for k, (label, repr_val ) in enumerate ( gdict.items() ): 
            prep , g  = repr_val 
            
            ag+=["{:5}=['{:^3}', \n".format(
                "Label" if k==0 else "{:>17}".format("Label"), label
                                               ) 
                ]
            ag +=["{:>32}( rate = '{:^7} %', \n".format(
                "Preponderance", round (prep *100, 3 )
                                                  )] 
            ag += ["{:>34}'Groups', {}),\n".format("[(",
                # str({ k: "{:>5}".format(round (v, 3)) for k , v in g.items()}) 
                str({ k: round (v, 3) for k , v in g.items()}) 
                    )
                ]
            ag +=["{:>34}'Representativity', ( '{}', {})),\n".format("(", 
                 list(g)[0], round ( g.get(list(g)[0]), 2))
                ]
            ag += ["{:>34}'Similarity', '{}')])],\n ".format("(", list(g)[0] )
                   ]
            # ag+=['{:>30}'.format("])],\n ")] 
        #ag+=["{:>7}".format(")")]
    
        return ''.join (ag) 
    

