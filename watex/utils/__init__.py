# -*- coding: utf-8 -*-

import os
import  re
import numpy as np

from .coreutils import ( 
    plotAnomaly, 
    vesSelector, 
    erpSelector, 
    defineConductiveZone,
    makeCoords, 
    )
from .exmath import ( 
    type_,
    shape, 
    power, 
    magnitude, 
    sfi, 
    ohmicArea, 
    vesDataOperator, 
    scalePosition,
    rhoa2z, 
    z2rhoa, 
    interpolate1d, 
    interpolate2d,
    scaley, 
    fittensor, 
    get_strike, 
    get_profile_angle, 
    moving_average, 
    linkage_matrix, 
    )
from .funcutils import ( 
    reshape, 
    to_numeric_dtypes 
    )
from .mlutils import ( 
    selectfeatures, 
    getGlobalScore, 
    discretizeCategoriesforStratification,
    stratifiedUsingDiscretedCategories, 
    split_train_test, 
    correlatedfeatures, 
    findCatandNumFeatures,
    evalModel, 
    cattarget, 
    labels_validator, 
    projection_validator, 
    rename_labels_in 
    
    )
from .plotutils import ( 
    plotmlxtendheatmap , 
    plotmlxtendmatrix, 
    plotcostvsepochs
    )
from ..decorators import gdal_data_check

__all__=[
        'plotAnomaly', 
        'vesSelector', 
        'erpSelector', 
        'defineConductiveZone',
        'makeCoords', 
        'type_',
        'shape', 
        'power', 
        'magnitude', 
        'sfi', 
        'ohmicArea', 
        'vesDataOperator', 
        'scalePosition',
        'rhoa2z', 
        'z2rhoa', 
        'interpolate1d', 
        'interpolate2d',
        'scaley', 
        'fittensor', 
        'selectfeatures', 
        'getGlobalScore', 
        'discretizeCategoriesforStratification',
        'stratifiedUsingDiscretedCategories', 
        'split_train_test', 
        'correlatedfeatures', 
        'findCatandNumFeatures',
        'evalModel',
        'plotmlxtendheatmap', 
        'plotmlxtendmatrix', 
        'plotcostvsepochs', 
        'get_strike', 
        'get_profile_angle', 
        'moving_average', 
        'linkage_matrix',
        'reshape', 
        'to_numeric_dtypes' , 
        'cattarget', 
        'labels_validator', 
        'projection_validator', 
        'rename_labels_in', 
        
        ]

HAS_GDAL = gdal_data_check(None)._gdal_data_found
NEW_GDAL = False

if (not HAS_GDAL):
    try:
        import pyproj
    except ImportError:
        raise RuntimeError("Either GDAL or PyProj must be installed")
else:
    import osgeo
    if hasattr(osgeo, '__version__') and int(osgeo.__version__[0]) >= 3:
        NEW_GDAL = True

# Import pyproj and set ESPG_DICT 
EPSG_DICT = {}
try:
    import pyproj
    epsgfn = os.path.join(pyproj.pyproj_datadir, 'epsg')
    f = open(epsgfn, 'r')
    lines = f.readlines()

    for line in lines:
        if ('#' in line): continue
        epsg_code_val = re.compile('<(\d+)>').findall(line)
        if epsg_code_val is not None and len(epsg_code_val) > 0 and \
            epsg_code_val[0].isdigit():
            epsg_code = int(epsg_code_val[0])
            epsg_string = re.compile('>(.*)<').findall(line)[0].strip()

            EPSG_DICT[epsg_code] = epsg_string
        else:
            #print("epsg_code_val NOT found for this line ", line, epsg_code_val)
            pass  
   
except Exception:
    path = os.path.dirname(os.path.abspath(__file__))
    epsg_dict_fn = os.path.join(path, 'epsg.npy')

    EPSG_DICT = np.load(epsg_dict_fn, allow_pickle=True).item()
    

