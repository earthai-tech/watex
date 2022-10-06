# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 14:22:59 2021
@author: @Daniel03

.. synopsis:: compute geelectrical features `sfi`, `magnitude`,
             power` ``shape` and `type`  from ERP survey lines. 
             If shape and type are notgiven on the excel worksheet , 
             will compute automatically. 
             ...
             
-----------------------------------------------------------------
How to arrange your sheet before calling  erp collection module. 
    refered as :mod:`watex.core.erp.ERP_collection` ?
-----------------------------------------------------------------

    Suppose an ERP survey performed in `zzegbao1_2`  location . 
    The excel whorkbook will probably called ` `zzegbao1_2.xlsx`
    or `zzegbao1_2.csv`. 
    This is how must be arrange data into the worksheet:
        
       Headers with `*` means compulsory data and others are optional  data. 
       `x`  and `y` are  utm easting and northing coordinates respectively, 
       while `rho` is the apparemt resistivity at each measurement point(`pk`).
       `sloc` is the colum of anomaly boundaries definition. The optionals 
       column's names such as `sloc`, `shape` and  `type` can be nothing.
        
    === ======== =========== ========== ======== ======= ========
    *pk  *x         *y          *rho      sloc     shape  type   
    === ======== =========== ========== ======== ======= ========
    0   790210      1093010     230        low   
    10  790214      1093016     93         se       V       CP
    20  790218      1093026     93         up
        ...
    140 790255      1093116     138        
    === ======== =========== ========== ======== ======= ========
    
    - `low` means the lower boundary of selected anomaly,  can also be '1'
    - `up` means the uper boundary of selected anomaly , can also be `2` 
    - `se` means the sounding location on the survey. can be `ves` or `0`. 
    - `V` anomaly-shape and can be 'W', 'K', 'U', 'H', 'C' and 'M' 
    - `CP` anomaly type and can be 'CB2P', 'NC' or 'EC'
    
    For further details about anomaly shape and type , please refer to the
    :doc:`watex.core.ERP.get_shape` or :doc:`watex.core.ERP.get_type`
    
    ..note::
        `se` is  presumed   to be location of the selected anomaly 
        position and if ``shape` and `type`not provided, will be get 
        automatically. However  to have full control, we strongly 
        recommend to specify the type and the shape of your selected 
        anomaly on ERP line. 
 
"""

from watex.methods.erp import ERP_collection 

# path to your erp lines 
pathtoErpfiles =  'data/Bag.main&rawds/ert_copy/an_dchar'

# export the file 
exportFile = True 
# create your output folder, if None wll create automatically

savepath = 'data/exFeatures'
# extension file . can ve .xlsx or .csv 
exportType ='.xlsx'

# name of your export file . if none will automatically create 
exporfilename ='_textfile'

erpObjs =ERP_collection(listOferpfn=pathtoErpfiles,
                        export =exportFile,
                        extension =exportType, 
                        filename = exporfilename, 
                        savepath = savepath )


