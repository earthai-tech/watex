# -*- coding: utf-8 -*-
#  License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Make Dataset 
===============
Automate the loading of dataset   

"""
import os
from importlib import resources  
from .dload import load_bagoue 
from .rload import loadBagoueDataset
from ..cases import  (
    BaseSteps, 
   ) 
from ..utils.mlutils import (
    dumpOrSerializeData
    )

__all__ = [
    "_bagoue_data_preparer"
  ] 

try: 
    # load data from local 
    DATA = load_bagoue().frame 
except: 
    # remotely download the data 
    if not os.path.isfile(DATA): 
        DATA= loadBagoueDataset ()
    
def _bagoue_data_preparer (): 
    """ Prepare the defaults case study data using 
    :class:`watex.cases.prepare.BaseSteps` and main bagoue data file 
    collected on the local machine or the remote. 
    """
    PKGS='watex.etc'
    # target or label name. 
    nameoftarget ='flow'
    # drop useless features
    drop_features= ['num',
                    'east', 
                    'north',
                    'name', 
                    # 'lwi', 
                    # 'type' 
                    ]
    # experiences attributes combinaisions 
    add_attributes =None 
    # add attributes indexes to create a new features. 
    attributesIndexes = None #[
                            # (0, 1),
                            # (3,4), 
                            # (1,4), 
                            # (0, 4)
                            # ] 
    # categorize a features on the trainset or label 
    feature_props_to_categorize =[
        ('flow', ([0., 1., 3.], ['FR0', 'FR1', 'FR2', 'FR3'])),
        ]
                     
    # bring your own pipelines .if None, use default pipeline.
    #ownPipeline =None 
    _conf_kws = {'tname':nameoftarget, 
                'drop_features':drop_features, 
                'add_attributes':add_attributes, 
                'attributesIndexes':attributesIndexes, 
                }
    _conf_kws['feature_props_to_categorize']= feature_props_to_categorize
    # createOnjects. 
    # readfile and set dataframe
    # hash equal to ``True`` to unsure data remain consistent 
    # even mutilple runs.
    prepareObj =BaseSteps(
        # data = data_fn,
        drop_features = drop_features,
        categorizefeature_props = feature_props_to_categorize,
        tname=nameoftarget, 
        add_attributes = add_attributes, 
        attribute_indexes =attributesIndexes, 
        hash=False
        )
    
    prepareObj.stratifydata(DATA) 
    _BAGDATA= prepareObj.data 
    _X =prepareObj.X.copy()             # strafified training set 
    _y =prepareObj.y.copy()             # stratified label 
    
    prepareObj.fit_transform(_X, _y)
    
    # --> Data sanitize but keep categorical features not encoded.
    #   Text attributes not encoded remains safe. 
    _X0 = prepareObj.X0          # cleaning and attr combined training set 
    _y0= prepareObj.y0           # cleaning and attr combined label 
    
    _Xp= prepareObj.X_prepared  # Train Set prepared 
    _yp= prepareObj.y_prepared  # label encoded (prepared)
    _Xc = prepareObj._Xpd       # training categorical ordinal encoded features.
    
    _pipeline = prepareObj.pipeline 
    
    _df0 = prepareObj._df0
    _df1 = prepareObj._df1
    
    # test set stratified data. Untouchable unless the best model is found.
    _XT = prepareObj.X_
    _yT= prepareObj.y_
    
    #  keep the test sets safe using  `dumpOrSerializeData` 
    # save the test set info in a savefile for the first run like::
    with resources.path (PKGS, '__Xy.pkl') as pkl : 
        pkl_file = str(pkl)
    if not os.path.isfile (pkl_file): 
        train_data =(_Xp,_yp )#_BAGDATA
        dumpOrSerializeData(train_data, filename ='__Xy.pkl', to='joblib', 
                                  savepath='watex/etc')
    with resources.path (PKGS, '__XTyT.pkl') as pkl : 
        pkl_file = str(pkl)
        
    if not os.path.isfile(pkl_file): 
        test_data=(_XT, _yT)
        dumpOrSerializeData(test_data, filename ='__XTyT.pkl', to='joblib', 
                              savepath='watex/etc')
    
    yield ( 
        _X,
        _y,
        _X0,
        _y0,
        _XT,
        _yT,
        _Xc,
        _Xp,
        _yp,
        _pipeline,
        _df0,
        _df1,
        _BAGDATA
        )
#-------------------------------------------------------------------------    
# store files in b.pkl file
# import sklearn 
# import watex 
# from watex.datasets._p import _bagoue_data_preparer
# import sklearn._min_dependencies as min_deps 
# data = list(_bagoue_data_preparer ()) [0]
# keys = [ '_X', '_y', '_X0', '_y0', '_XT', '_yT', '_Xc', '_Xp', '_yp', 
#         '_pipeline', '_df0', '_df1', '_BAGDATA'] 
# b = {} 
# for k, v in zip ( keys, data ): 
#     b[k]= v 

# b['__sklearn_version__']=  [sklearn.__version__]
# b['min_dependencies'] =  min_deps.tag_to_packages['install'] 
# b['NUMPY_MIN_VERSION']= [min_deps.NUMPY_MIN_VERSION ]
# b['SCIPY_MIN_VERSION']= [min_deps.SCIPY_MIN_VERSION] 
# b['__watex_version__']= [watex.__version__]
# b =[b] put the dict in the list to avoid pickling mirror 
# # for name, value in zip (
# # b = dict() 
# #     #df.to_hdf ( 'watex/datasets/data/b.h5', key =name,mode ='a' )
# dumpOrSerializeData(b, filename ='b2.pkl', to='joblib', 
#                     savepath='watex/datasets/data')
#--------------------------------------------------------------------------- 
    
    
    
    
    
    
    
    
    
    
    
    