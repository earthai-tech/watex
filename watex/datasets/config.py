# -*- coding: utf-8 -*-
#   create date: Thu Sep 23 16:19:52 2021/

"""
Dataset 
==========
Fetch data set from the local machine. It data does not exist, retrieve it 
from online (repository or zenodo record ) 
 
"""
import re

from ..property  import BagoueNotes
from ..tools.funcutils import ( 
    smart_format 
    )
from ..tools.mlutils import  ( 
    loadDumpedOrSerializedData, 
    )
from ..exceptions import DatasetError
from .._watexlog import watexlog
from ._p import ( 
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

_logger = watexlog().get_watex_logger(__name__)


__all__=['fetch_data']

_BTAGS = ( 
    'mid', 
    'semi', 
    'preprocess', 
    'fit',
    'analysis', 
    'pca',
    'reduce', 
    'dimension',
    'test',
    'pipe',
    'prepared'
    )

_msg =dict (
    origin = ("Can't fetch an original data <- dict contest details." ), 
    )

regex = re.compile ( r'|'.join(_BTAGS ) + '|origin' , re.IGNORECASE
                    ) 
for key in _BTAGS : 
    _msg[key] = (
        "Can't build default transformer pipeline: <-'default pipeline' "
        )  if key =='pipe' else (
            f"Can't fetching {key} data: <-'X' & 'y' "
            )
            
_BVAL= dict (
    origin= {
        'COL_NAMES': _BAGDATA.columns, 
        'DESCR':'https://doi.org/10.5281/zenodo.5571534: bagoue-original',
        'data': _BAGDATA.values, 
        'data=df':_BAGDATA, 
        'data=dfy1':_df1, 
        'data=dfy2':_df0,
        'attrs-infos':BagoueNotes.bagattr_infos, 
        'dataset-contest':{
            '_documentation:':'`watex.property.BagoueNotes.__doc__`', 
            '_area':'https://en.wikipedia.org/wiki/Ivory_Coast', 
            '_casehistory':'https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021WR031623',
            '_wikipages':'https://github.com/WEgeophysics/watex/wiki',
            '_citations': ('https://doi.org/10.1029/2021wr031623', 
                            ' https://doi.org/10.5281/zenodo.5529368')
            },
        'tags': ('original', 
                 'stratified',
                 'mid', 
                 'semi', 
                 'preprocess', 
                 'pipe', 
                 'analyses', 
                 'pca',
                 'reduce dimension', 
                 'test'
                 'pipe',
                 'prepared',
                 )
            }, 
    stratified= (
        _X,
        _y
        ),
    semi= (
        _X0,
         _y0 
         ), 
    pipe= _pipeline, 
    analysis= (
        _Xc,
        _yp 
        ), 
)
  
def fetch_data(tag): 
    """ Fetch dataset from 'tag'. A tag correspond to each level of data 
    processing. 
    
    An example of retrieving Bagoue dataset can be experimented.

    Parameters 
    ------------
    tag: str,  
        stage of data processing. Tthere are different options to retrieve data
        Could be:
            
        * ['original'] => original or raw data -& returns a dict of details 
            contex combine with get method to get the dataframe like::
                
                >>> fetch_data ('bagoue original').get ('data=df')
        * ['stratified'] => stratification data
        * ['mid' |'semi'|'preprocess'|'fit']=> data cleaned with 
            attributes experience combinaisons.
        * ['pipe']=>  default pipeline created during the data preparing.
        * ['analyses'|'pca'|'reduce dimension']=> data with text attributes
            only encoded using the ordinal encoder +  attributes  combinaisons. 
        * ['test'] => stratified test set data

           
    Returns
    -------
        `data`: Original data 
        `X`, `y` : Stratified train set and training target 
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
    r=None 
    pm =regex.search (tag)
    if pm is None: 
        raise DatasetError(f"Unknow tag {tag!r}. Expect 'original',"
                           f" {smart_format(_BTAGS, 'or')}") 
        
    pm= pm.group() 

    if _pca_set_checker(pm.lower()): 
        pm = 'analysis'
    
    elif pm in ('mid','semi', 'preprocess', 'fit'): 
        pm = 'semi' 
        
    if pm =='prepared': 
        r = loadingdefaultSerializedData (
            'watex/etc/__Xy.pkl',(_Xp, _yp), dtype='training' 
                )
    elif pm =='test': 
        r = loadingdefaultSerializedData (
            'watex/etc/__XTyT.pkl',(_XT, _yT), dtype='test' ),
    else : 
        try : 
            r =_BVAL[pm]
        except : 
           _logger.error (_msg[pm])
    return r 

     
def loadingdefaultSerializedData (f, d0, dtype ='test'): 
    """ Retreive Bagoue data from dumped or Serialized file.
    
    :param f: str or Path-Like obj 
        Dumped or Serialized default data 
    :param d0: tuple 
        Return default returns wich is the data from config 
        <./datasets/config.py > 
    :param dtype:str 
        Type of data to retreive.
    """
    
    
    load_source ='serialized'
    try : 
        X, y= loadDumpedOrSerializedData(f)
    except : 
        _logger.error(f"Fetch data from {load_source!r} source failed. "
                       " Use local 'config' source instead ...")
        load_source='config'
        X, y =d0

    return X, y

def _pca_set_checker (param):
    for ix in ['analys', 'pca', 'dim','reduc']: 
        if ix in param.lower():
            return True 
    return False 
   

    
