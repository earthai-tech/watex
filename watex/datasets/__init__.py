# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Wed Sep 15 11:39:43 2021 
# This module is a set of datasets packages
# released under a MIT- licence.
import warnings

from ..property  import BagoueNotes
from .._watexlog import watexlog
from ..tools.mlutils import  ( 
    loadDumpedOrSerializedData, 
    format_generic_obj, 
    )
from ..exceptions import DatasetError

__logger = watexlog().get_watex_logger(__name__)

try:
    from .config import (
        data,
        X, y,
        X0, y0,
        XT, yT, 
        X_prepared, y_prepared,
        _X,_pipeline, 
        df0, df1
        )
except : 
    
    __logger.debug("None Config file detected. Be aware that you will not able "
                    "implements the basics examples of the scripts or Basic "
                    " steps of datapreparing!")
    warnings.warn("None config file detected! Be aware you will not take into"
                  " advantage of the basics steps thoughout the scripts. "
                  " Future implementation will presume to fetch data "
                  " automatically from repository or  from zenodo record.", 
                  FutureWarning)

BAGOUE_TAGS= (
        'Bagoue original', 
        'Bagoue stratified sets', 
        'Bagoue prepared sets', 
        'Bagoue mid-evaluation', 
        'Bagoue semi-preparing`', 
        'Bagoue preprocessing sets', 
        'Bagoue default pipeline', 
        'Bagoue analyses sets`', 
        'Bagoue pca sets',
        'Bagoue reduce dimension sets', 
        'Bagoue untouched test sets'
                        )

def fetch_data(param): 
    """ Fetch bagoue dataset values and details."""
    
    if param.lower().find('original')>=0: 
        __logger.info('Fetching the Bagoue original data. Returns a dictionnary '
                      ' of area description, attributes and contest details.')
        
        return {
            'COL_NAMES': data.columns, 
            'DESCR':'https://doi.org/10.5281/zenodo.5571534: bagoue-original',
            'data': data.values, 
            'data=df':data, 
            'data=dfy1':df1, 
            'data=dfy2':df0,
            'attrs-infos':BagoueNotes.bagattr_infos, 
            'dataset-contest':{
                '__documentation:':'`watex.property.BagoueNotes.__doc__`', 
                '__area':'https://en.wikipedia.org/wiki/Ivory_Coast', 
                '__casehistory':'https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021WR031623',
                '__wikipages':'https://github.com/WEgeophysics/watex/wiki',
                '__citations': ('https://doi.org/10.1029/2021wr031623', 
                                ' https://doi.org/10.5281/zenodo.5529368')
                },
            'tags':BAGOUE_TAGS
                }
    
    elif param.lower().find('stratified')>=0: 
        __logger.info('Fetching the stratified training data `X` and `y`')
        
        return  X, y
    
    elif param.lower().find('prepared')>=0:
        __logger.info('Fetching the prepared data `X` and `y`')
        Xp, yp= loadingdefaultSerializedData ('watex/datasets/__Xy.pkl',
                                              (X_prepared, y_prepared),
                                              dtype='prepared training' )
        return Xp, yp
    
    elif param.lower().find('semi-')>=0 or param.lower().find('fit')>=0 or \
        param.lower().find('mid-')>=0 or param.lower().find('preprocess')>=0: 
        __logger.info('Fetching the mid-preparation data `X` and `y`')

        return X0, y0 
    
    elif param.lower().find('test set')>=0  or param.lower().find('x test')>=0: 
        __logger.info('Fetching the stratified test set `X` and `y`')
        
        XT0, yT0= loadingdefaultSerializedData ('watex/datasets/__XTyT.pkl',
                                              (XT, yT), dtype='test' )
        return XT0, yT0
    
    elif param.lower().find('pipeline')>=0:
        __logger.info('Fetching the transformer pipeline =`defaultPipeline`')

        return _pipeline
    
    elif _pca_set_checker(param.lower()):
        __logger.info('Fetching the data for analyses. Text attributes'
                      ' are ordinarily encoded to numerical categories.')
        return _X, y_prepared 
    
    else : 
        raise DatasetError(
            'Arguments ~`{0}` not found in default tags:'
             ' {1}. Unable to fetch data.'.format(param, 
              format_generic_obj (BAGOUE_TAGS)).format(
                *list(BAGOUE_TAGS)))
    
def loadingdefaultSerializedData (f, d0, dtype ='test'): 
    """ Retrive Bagoue data from dumped or Serialized file.
    :param f: str or Path-Like obj 
        Dumped or Serialized default data 
    :param d0: tuple 
        Return default returns wich is the data from config 
        <./datasets/config.py > 
    :param dtype:str 
        Type of data to retreive.
    """
    load_source ='Serialized'
    try : 
        X, y= loadDumpedOrSerializedData(f)
    except : 
        __logger.info(f"Fetching data from {load_source!r} source failed. "
                       " We try `Config` loading source...")
        load_source='Config'
        X, y =d0
        
    __logger.info(f"Loading the {dtype} data from <{load_source}>"
                  "successfuly done!")
    
    return X, y

def _pca_set_checker (param):
    for ix in ['analys', 'pca', 'dim','reduc']: 
        if ix in param.lower():return True 
    return False 
        
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
        numerical attributes combinaisons if ``True``
    `X_prepared`, `y_prepared`: Data prepared after applying  all the 
       transformation via the transformer (pipeline). 
    `XT`, `yT` : stratified test set and test label  
    `_X`: Stratified training set for data analysis. So None sparse
        matrix is contained. The text attributes (categorical) are converted 
        using Ordianal Encoder.  
    `_pipeline`: the default pipeline. 
"""




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    