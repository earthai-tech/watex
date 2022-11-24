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

from .._watexlog import watexlog 

class Hydrogeology :
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
    
    def __init__(self, **kwds): 
        self._logging = watexlog.get_watex_logger(self.__class__.__name__)


class AquiferGroup:
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
    >>> from watex.utils.hydroutils import map_k, find_aqgroup_representativity 
    >>> b= load_hlogs () #just taking the target names
    >>> data = read_data ('data/boreholes/hf.csv') # read complete data
    >>> y = data [b.target_names]
    >>> # impute the missing values found in aquifer group columns
    >>> # reshape 1d array along axis 0 for imputation 
    >>> agroup_imputed = naive_imputer ( reshape (y.aquifer_group, axis =0 ) , 
                                        strategy ='most_frequent') 
    >>> # rehape back to array_like 1d 
    >>> y.aquifer_group =reshape (agroup_imputed) 
    >>> # categorize the 'k' continous value in 'y.k' using the default 
    >>> # 'k' mapping func 
    >>> y.k = map_k (y.k , default_func =True)
    >>> # get the group obj
    >>> group_obj = find_aqgroup_representativity(y.k, y.aquifer_group,  ) 
    >>> group_obj 
    ... AquiferGroup(Label=[' 1 ', 
                       Preponderance( rate = '53.141  %', 
                                    (['Groups', {'V': 0.32, 'IV': 0.266, 'II': 0.158, 'III': 0.158, 'II ': 0.079, 'IV&V': 0.01, 'II&III': 0.005, 'III&IV': 0.005}),
                                     ('Representativity', ( 'V', 0.32)),
                                     ('Similarity', 'V')])],
                 Label=[' 2 ', 
                       Preponderance( rate = ' 19.11  %', 
                                    (['Groups', {'III': 0.274, 'V': 0.26, ' II ': 0.178, 'IV': 0.178, 'II': 0.082, 'III&IV': 0.027}),
                                     ('Representativity', ( 'III', 0.27)),
                                     ('Similarity', 'III')])],
                 Label=[' 3 ', 
                       Preponderance( rate = '27.749  %', 
                                    (['Groups', {'V': 0.443, 'IV': 0.311, 'III': 0.245}),
                                     ('Representativity', ( 'V', 0.44)),
                                     ('Similarity', 'V')])],
                 )
                                      
    """
    def __init__ (self, g=None, /  ): 
        self.groups_ = g
    @property 
    def groups (self): 
        return self.groups_
    
    def __repr__ (self ) :
        return  self.__class__.__name__  + "(" +  self._format (
            self.groups ) + "{:>13}".format(")")
        
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
            ag +=["{:>32}( rate = '{:^7} %', \n".format("Preponderance", round (prep *100, 3 )
                                                  )] 
            ag += ["{:>34}'Groups', {}),\n".format("([",
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
    

