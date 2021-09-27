# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Wed Sep 15 11:39:43 2021 
# This module is a set of datasets packages
# released under a MIT- licence.

from watex.utils._watexlog import watexlog
from watex.datasets.config import (data,
                                X, y,
                                X0, y0, 
                                X_prepared, y_prepared,
                                XT, yT, 
                                _X,_pipeline, 
                                df0, df1)
                                 
__logger = watexlog().get_watex_logger(__name__)

BAGOUE_TAGS= (
        'Bagoue original', 
        'Bagoue stratified sets', 
        'Bagoue data prepared', 
        'Bagoue mid-evaluation', 
        'semi-preparing`', 
        'Bagoue data preprocessing', 
        'Bagoue default pipeline', 
        'Bagoue analysis`', 
        'Bagoue pca',
        'Bagoue dimension reduction', 
                        )

def fetch_data(param): 
    """ Fetch bagoue dataset values and details."""
    from ..utils.infos  import BagoueNotes
    
    if param.lower().find('original')>=0: 
        __logger.info('Fetching the original data. Return a dict')
        
        return {'COL_NAMES': data.columns, 
                'DESCR':'https://doi.org/10.5281/zenodo.4896758: bagoue-original',
                'data': data.values, 
                'data=df':data, 
                'data=dfy1':df1, 
                'data=dfy2':df0,
                'attrs-infos':BagoueNotes.bagattr_infos, 
                'dataset-contest':{
                    '__documentation:':'`~watex.utils.infos.BagoueNotes.__doc__`', 
                    '__area':'https://en.wikipedia.org/wiki/Ivory_Coast', 
                    '__casehistory':'https://github.com/WEgeophysics/watex/blob/WATex-process/examples/codes/pred_r.PNG',
                    '__wikipages':'https://github.com/WEgeophysics/watex/wiki',
                    }
                }
    
    elif param.lower().find('stratified')>=0: 
        __logger.info('Fetching the stratified training data `X` and `y`')
        
        return  X, y
    
    elif param.lower().find('prepared')>=0:
        __logger.info('Fetching the prepared data `X` and `y`')
        
        return X_prepared, y_prepared 
    
    elif param.lower().find('semi')>=0 or param.lower().find('fit')>=0 or \
        param.lower().find('mid')>=0 or param.lower().find('preprocess')>=0: 
        __logger.info('Fetching the mid-preparation data `X` and `y`')

        return X0, y0 
    
    elif param.lower().find('test set')>=0  or param.lower().find('x test')>=0: 
        __logger.info('Fetching the stratified test set `X` and `y`')
        
        return XT, yT
    
    elif param.lower().find('pipeline')>=0:
        __logger.info('Fetching the transformer pipeline =`defaultPipeline`')

        return _pipeline
    
    elif ('analysis' or 'pca' or 'dim' or 'reduc') in param.lower():

        __logger.info('Fetching the data for analyses. Text attributes'
                      ' are ordinarily encoded using the`defaultPipeline`')
        return _X, y0
    
    else : 
        from ..utils.exceptions import WATexError_datasets
        from ..hints import format_generic_obj 
        
        raise WATexError_datasets('Arguments ~`{0}` not found in default tags:'
                                  ' {1}. Unable to retrieve data.'.format(param, 
                                format_generic_obj (BAGOUE_TAGS)).format(
                                    *list(BAGOUE_TAGS)))
    


fetch_data.__doc__ +="""\
Parameters
----------
param: str 
    Different options to retrieve data
    Could be: 
        - `Bagoue original`: for original data 
        - `Bagoue stratified sets`: for stratification data
        - `Bagoue data prepared`: Data prepared using the default pipelines
        - `Bagoue mid-evaluation|semi-preparing|Bagoue data preprocessed|
            or Bagoue data fit`: To retrieve only the data cleaned and 
            attributes experience combinaisons.
        - `Bagoue test set` : for stratified test set data
        - `Bagoue default pipeline`: retrive the default pipeline for 
            data preparing.
        - `Bagoue analysis|pca|dimension reduction`: To retreive data with 
            text attributes only encoded using the ordinal encoder additional 
            to attributes  combinaisons. 
        
Returns
-------
    `data` : Original data 
    `X`, `y` : Stratified train set and training label 
    `X0`, `y0`: data cleaned after dropping useless features and combined 
        numerical attributes combinaisions if ``True``
    `X_prepared`, `y_prepared`: Data prepared after applying  all the 
       transformation via the transformer (pipeline). 
    `XT`, `yT` : stratified test set and test label  
    `_X`: Stratified training set for data analysis. So None sparse
        matrix is contained. The text attributes (categorical) are converted 
        using Ordianal Encoder.  
    `_pipeline`: the default pipeline. 
"""
# from .data_preparing import bagdataset as data  
# from .data_preparing import bagoue_train_set_prepared as TRAINSET_PREPARED 
# from .data_preparing import bagoue_label_encoded as TRAINSET_LABEL_ENCODED 
# from .data_preparing import raw_X as TRAINSET
# from .data_preparing import raw_y  as LABELS 
# from .data_preparing import default_X as dX
# from .data_preparing import default_y  as dy 
# from .data_preparing import full_pipeline  
# from .data_preparing import bagoue_testset_stratified as TESTSET 
# from .data_preparing import bagoue_testset_label_encoded as TESTSET_LABEL_ENCODED

# # raw dataset 
# bagoue_dataset = data 
# # raw trainset and test set 
# X, y = TRAINSET , LABELS
# # stratified trainset and testset 
# X_ , y_= dX , dy 
# # after stratificated , defaults data prepared 


# X_prepared, y_prepared = TRAINSET_PREPARED, TRAINSET_LABEL_ENCODED
# # Test set put aside and applied the transformation as above. 

# X_test, y_test  = TESTSET,  TESTSET_LABEL_ENCODED
# # default pipeline 
# # call pipeline to see all the transformation 
# default_pipeline = full_pipeline 
if __name__=='__main__':
    import numpy as np

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    