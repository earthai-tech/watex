# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Wed Jul 14 20:00:26 2021
# This module is a set of datasets packages
# released under a MIT- licence.
"""
Created on Wed Sep 15 11:39:43 2021

@author: @Daniel03

"""
from .data_preparing import bagdataset as data  
from .data_preparing import bagoue_train_set_prepared as TRAINSET_PREPARED 
from .data_preparing import bagoue_label_encoded as TRAINSET_LABEL_ENCODED 
from .data_preparing import raw_X as TRAINSET
from .data_preparing import raw_y  as LABELS 
from .data_preparing import default_X as dX
from .data_preparing import default_y  as dy 
from .data_preparing import full_pipeline  
from .data_preparing import bagoue_testset_stratified as TESTSET 
from .data_preparing import bagoue_testset_label_encoded as TESTSET_LABEL_ENCODED



dataset_infos= """"
    `Bagoue dataset` are are Bagoue region is located in WestAfrica and lies
    between longitudes 6° and 7° W and latitudes 9° and 11° N in the north of 
    Cote d’Ivoire. The average FR observed in this area fluctuates between 1
    and 3 m3/h. Refer to the link of case story paper in the repository 
    part https://github.com/WEgeophysics/watex#documentation to visualize the
    location map of the study area with the geographical distribution of the
    various boreholes in the region. The geophysical data and boreholes
    data were collected from National  Office of Drinking Water (ONEP) and
    West-Africa International Drilling  Company (FORACO-CI) during  the 
    Presidential Emergency Program (PPU) in 2012-2013 and the National Drinking 
     Water Supply Program (PNAEP) in 2014.
    
    The data are firstly composed of Electrical resistivity profile (ERP) data
    collected from geophysical survey lines with various arrays such as
    Schlumberger, gradient rectangle and Wenner (α or β) and the Vertical 
    electricalsounding (VES) data carried out on the selected anomalies.
    The configuration used during the ERP is Schlumberger with distance of
    AB is 200m and MN =20m."""

# raw dataset 
bagoue_dataset = data 
# raw trainset and test set 
X, y = TRAINSET , LABELS
# stratified trainset and testset 
X_ , y_= dX , dy 
# after stratificated , defaults data prepared 


X_prepared, y_prepared = TRAINSET_PREPARED, TRAINSET_LABEL_ENCODED
# Test set put aside and applied the transformation as above. 

X_test, y_test  = TESTSET,  TESTSET_LABEL_ENCODED
# default pipeline 
# call pipeline to see all the transformation 
default_pipeline = full_pipeline 