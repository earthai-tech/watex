
import os
import shutil
# import sys 

from watex.tools._watexlog import watexlog
from watex.datasets import fetch_data as BAGOUE_DATASET

# sys.path.insert(0, os.path.abspath('..'))

TEST_WATex_ROOT = os.path.normpath(
    os.path.abspath(
        os.path.dirname(
            os.path.dirname(__file__)
            )
    )
)  # assume tests is on the root level of pyCSAMT goes one step backward 

TEST_DIR = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))


TEST_TEMP_DIR = os.path.normpath(os.path.join(TEST_DIR, "temp"))

if not os.path.isdir(TEST_TEMP_DIR):
    os.mkdir(TEST_TEMP_DIR)


def make_temp_dir(dir_name, base_dir=TEST_TEMP_DIR):
    _temp_dir = os.path.normpath(os.path.join(base_dir, dir_name))
    if os.path.isdir(_temp_dir):
        shutil.rmtree(_temp_dir) # clean the existing directory 
    os.mkdir(_temp_dir)             # make a new director to collect temp files 
    return _temp_dir

# declare some main data directories for test samples 

ERP_DATA_DIR = os.path.normpath(
    os.path.join(TEST_WATex_ROOT, 'data/erp'))
VES_DATA_DIR = os.path.normpath(
    os.path.join(TEST_WATex_ROOT, 'data/ves'))

DATA_ORIGINAL = BAGOUE_DATASET('Bagoue original')
DATA_PREPARED = BAGOUE_DATASET('Bagoue data prepared')
DATA_MID_PROCESSED= BAGOUE_DATASET('Bagoue data preprocessed')
DATA_ANALYSES = BAGOUE_DATASET('Bagoue analysis')

erp_test_location_name ='l10_gbalo.xlsx'
PREFIX = ['station','easting', 'northing', 'resistivity' ]

DATA_UNSAFE= os.path.join(ERP_DATA_DIR, 'testunsafedata.csv')
DATA_SAFE = os.path.join(ERP_DATA_DIR, 'testsafedata.csv')

DATA_UNSAFE_XLS = os.path.join(ERP_DATA_DIR, 'testunsafedata.xlsx')
DATA_SAFE_XLS = os.path.join(ERP_DATA_DIR, 'testsafedata.xlsx')

DATA_EXTRA = os.path.join(ERP_DATA_DIR, 'testunsafedata_extra.csv')

# set test logging configure
watexlog.load_configure(
    os.path.join(os.path.abspath('./watex'),'utils', 
                 "wlog.yml"))
