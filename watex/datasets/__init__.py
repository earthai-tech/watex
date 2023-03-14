"""
Dataset subpackage is used to fetch data from the local machine. 
If the data does not exist or deleted, the remote searching 
(repository or zenodo record ) triggers via the module 
:mod:`~watex.datasets.rload`
"""
from .sets import ( 
    load_bagoue, 
    load_gbalo, 
    load_iris, 
    load_semien, 
    load_tankesse, 
    load_boundiali,
    load_hlogs,
    load_huayuan,
    fetch_data,
    load_edis,
    make_erp , 
    make_ves, 
    DATASET
    )

__all__=[ 
         "load_bagoue",
         "load_gbalo", 
         "load_iris", 
         "load_semien", 
         "load_tankesse", 
         "load_boundiali",
         "load_hlogs",
         "load_huayuan", 
         "fetch_data",
         "load_edis",
         "make_erp" , 
         "make_ves", 
         "DATASET"
         ]