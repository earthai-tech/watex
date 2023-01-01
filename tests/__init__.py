# -*- coding: utf-8 -*-
# Licence:BSD 3-Clause
# Author: LKouadio

from __future__ import print_function
import os
import sys
from difflib import unified_diff
import matplotlib

from watex._watexlog import watexlog
from watex.datasets import fetch_data 

# ++++++++++++++++++++++++ configure testing path ++++++++++++++++++++++++++++

if os.path.abspath('..') not in sys.path: 
    sys.path.insert(0, os.path.abspath('..'))
    
TEST_ROOT = os.path.normpath(
    os.path.abspath(
        os.path.dirname(
            os.path.dirname(__file__)
            )
    )
)  

TEST_DIR = os.path.normpath(
    os.path.abspath(os.path.dirname(__file__)))

TEST_TEMP_DIR = os.path.normpath(os.path.join(TEST_DIR, "temp"))

ERP_PATH = os.path.normpath(
    os.path.join(
        TEST_ROOT, 'data/erp')
    )
VES_PATH= os.path.normpath(
    os.path.join(
        TEST_ROOT, 'data/ves')
    )

ORIGINAL_DATA = fetch_data(
    'Bagoue original'
    )
PREPARED_DATA = fetch_data(
    'Bagoue data prepared'
    )
SEMI_PROCESSED_DATA= fetch_data(
    'Bagoue data preprocessed'
    )
ANALYSIS_DATA = fetch_data(
    'Bagoue analysis'
    )

erp_test_location_name ='l10_gbalo.xlsx'

PREFIX = [
    'station', 
    'resistivity', 
    'longitude', 
    'latitude', 
    'easting', 
    'northing'
 ]

DATA_UNSAFE= os.path.join(
    ERP_PATH, 'testunsafedata.csv'
    )
DATA_SAFE = os.path.join(
    ERP_PATH, 'testsafedata.csv'
    )

DATA_UNSAFE_XLS = os.path.join(
    ERP_PATH, 'testunsafedata.xlsx'
    )
DATA_SAFE_XLS = os.path.join(
    ERP_PATH, 'testsafedata.xlsx'
    )

DATA_EXTRA = os.path.join(
    ERP_PATH, 'testunsafedata_extra.csv'
    )

# set test logging configure
watexlog.load_configure(
    os.path.join(
            os.path.abspath('.'), "watex/wlog.yml")
    )

#-----------------
# create a path to temp dir and recursively 
# deleted after test is passed 

import shutil
if not os.path.isdir(TEST_TEMP_DIR):
    os.mkdir(TEST_TEMP_DIR)


def make_temp_dir(dir_name, base_dir=TEST_TEMP_DIR):
    _temp_dir = os.path.normpath(os.path.join(base_dir, dir_name))
    if os.path.isdir(_temp_dir):
        shutil.rmtree(_temp_dir) # clean the existing directory 
    os.mkdir(_temp_dir)           # make a new director to collect temp files 
    return _temp_dir

# ++++++++   config matplotlib display and files controls ++++++++++++++++++++

if os.name == "posix" and 'DISPLAY' not in os.environ:

    print("MATPLOTLIB: No Display found, using non-interactive svg backend",
          file=sys.stderr)
    matplotlib.use('svg')
    import matplotlib.pyplot as plt
    
    TEST_HAS_DISPLAY = False
else:
    #matplotlib.use('svg')
    import matplotlib.pyplot as plt
    TEST_HAS_DISPLAY  = True
    plt.ion()
    
watexlog.get_watex_logger(__name__).info(
    "Testing using matplotlib backend {}".format(
        matplotlib.rcParams['backend'])
    )


def reset_matplotlib():
    """Reset figure after the plot. """
    interactive = matplotlib.rcParams['interactive']
    backend = matplotlib.rcParams['backend']
    matplotlib.rcdefaults()  # reset the rcparams to default
    matplotlib.rcParams['backend'] = backend
    matplotlib.rcParams['interactive'] = interactive
    logger = watexlog().get_watex_logger(__name__)
    
    logger.info("Testing using matplotlib backend {}".format(
        matplotlib.rcParams['backend'])
        )
    
    
def diff_files(after, before, ignores=None):
    """
    compare two files using difflib library 
    :param ignores: don't coerce and ignore term in the line. 
    :param before: the state of the file before triggering the test
    :param after: the state of the file after testing 
    :return: the number count of different lines
    """

    with open(before) as f2p:
        before_lines = f2p.readlines()
    with open(after) as f1p:
        after_lines = f1p.readlines()

    before_lines = [line.strip() for line in before_lines]
    after_lines = [line.strip() for line in after_lines]

    if ignores:
        for ignored_term in ignores:
            before_lines = [
                line for line in before_lines if ignored_term not in line
                ]
            after_lines = [
                line for line in before_lines if ignored_term not in line
                ]

    msg = "Comparing {} and {}:\n".format(before, after)

    lines = [
        line for line in unified_diff(
        before_lines,
        after_lines,
        fromfile="baseline ({})".format(before),
        tofile="test ({})".format(after),
        n=0)
        ]


    if lines:
        msg += "  Found differences:\n\t" + "\n\t".join(lines)
        is_identical = False
    else:
        msg += " NO differences found."
        is_identical = True

    return is_identical, msg  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    