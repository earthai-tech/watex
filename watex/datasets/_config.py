# -*- coding: utf-8 -*-
#   create date: Thu Sep 23 16:19:52 2021
#   Author: L.Kouadio 
#   License: BSD-3-Clause

"""
Dataset 
==========
Fetch data set from the local machine. If data does not exist, retrieve it 
from the remote (repository or zenodo record ) 
 
"""
import re
from importlib import resources 
from .io import DMODULE 
from ..property  import BagoueNotes
from ..utils.funcutils import ( 
    smart_format 
    )
from ..utils.mlutils import  ( 
    loadDumpedOrSerializedData, 
    )
from ..exceptions import DatasetError
from .._watexlog import watexlog

_logger = watexlog().get_watex_logger(__name__)


__all__=['_fetch_data']

_BTAGS = ( 
    'semi', 
    'preprocessed', 
    'fitted',
    'stratified', 
    'analysed', 
    'pca',
    'reduced', 
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
          
_BAG=dict()

try : 
    with resources.path (DMODULE, 'b.pkl') as p : 
        data_file = str(p) # for consistency
        _BAG = loadDumpedOrSerializedData(data_file)[0] 

except : 
    from ._p import ( 
        _bagoue_data_preparer 
        ) 
    (
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
    ) = list(_bagoue_data_preparer())[0]
    
    _BAG =  {
        '_X':_X,
        '_y':_y,
        '_X0':_X0,
        '_y0':_y0,
        '_XT':_XT,
        '_yT':_yT,
        '_Xc':_Xc,
        '_Xp':_Xp,
        '_yp':_yp,
        '_pipeline':_pipeline,
        '_df0':_df0,
        '_df1':_df1,
        '_BAGDATA':_BAGDATA
            }
            
_BVAL= dict (
    origin= {
        'COL_NAMES': _BAG.get ('_BAGDATA').columns, 
        'DESCR':'https://doi.org/10.5281/zenodo.5571534: bagoue-original',
        'data': _BAG.get('_BAGDATA').values, 
        'data=df':_BAG.get('_BAGDATA'), 
        'data=dfy1':_BAG.get('_df1'), 
        'data=dfy2':_BAG.get('_df0'),
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
                 'semi', 
                 'preprocessed', 
                 'pipe', 
                 'analysed', 
                 'pca',
                 'dimension reduced', 
                 'test'
                 'pipe',
                 'prepared',
                 )
            }, 
    stratified= (
        _BAG.get('_X'), 
        _BAG.get('_y') 
        ),
    semi= (
        _BAG.get('_X0'),
         _BAG.get('_y0') 
         ), 
    pipe= _BAG.get('_pipeline'), 
    analysed= (
        _BAG.get('_Xc'),
        _BAG.get('_yp') 
        ), 
)

def _fetch_data(tag, data_names=[] ): 
    PKGS ="watex.etc"
    r=None
    tag = str(tag)
 
    if _tag_checker(tag.lower()): 
        pm = 'analysed'
    elif _tag_checker(tag.lower(), ('mid','semi', 'preprocess', 'fit')):
        pm='semi'
    else : 
        pm =regex.search (tag)
        if pm is None: 
            data_names+= list(_BTAGS)
            msg = (f"Unknow tag-name {tag!r}. None dataset is stored"
                f" under the name {tag!r}. Available tags are: "
                f"{smart_format (data_names, 'or')}"
                )
            raise DatasetError(msg)

        pm= pm.group() 
    
    if pm =='prepared': 
        with resources.path (PKGS, '__Xy.pkl') as pkl : 
            pkl_file = str(pkl)
            
        r = loadingdefaultSerializedData (
            pkl_file,  (_BAG.get('_Xp'), _BAG.get('_yp')),
            dtype='training' 
                )
    elif pm =='test': 
        with resources.path (PKGS, '__XTyT.pkl') as pkl : 
            pkl_file = str(pkl)
        r, = loadingdefaultSerializedData (
            pkl_file, ( _BAG.get('_XT'),  _BAG.get('_yT')), 
            dtype='test' ),
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
        <./datasets/_config.py > 
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

def _tag_checker (param, tag_id= ('analys', 'pca', 'dim','reduc') 
                      # out = 'analysed'
                      ):
    for ix in tag_id: 
        if param.lower().find(ix)>=0:
            return True
    return False 
   
    
_fetch_data.__doc__ ="""\
Fetch dataset from 'tag'. A tag correspond to each level of data 
processing. 

Experiment an example of retrieving Bagoue dataset.

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

# pickle bag data details:
# Python.__version__: 3.10.6 , 
# scikit_learn_vesion_: 1.1.3 
# pandas_version : 1.4.4. 
# numpy version: 1.23.3
# scipy version:1.9.3 
 
    
    
    
    
    
    
    
    
    
    
     
