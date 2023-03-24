"""
Models sub-package focuses on  training and validation phases. It also composed 
of a set of grid-search tricks from model hyperparameters fine-tuning and 
the pretrained models fetching from :mod:`~watex.models.validation` and 
:mod:`~watex.models.premodels` respectively. Modules of 'Models' sub-package
expect the predictor :math:`X`  and the target :math:`y` to be preprocessed.  
"""
from .validation import ( 
    BaseEvaluation, 
    GridSearch, 
    GridSearchMultiple,
    get_best_kPCA_params, 
    get_scorers, 
    getGlobalScores, 
    getSplitBestScores, 
    displayCVTables, 
    displayFineTunedResults, 
    displayModelMaxDetails, 
    naive_evaluation, 

    )
from .premodels import pModels 

__all__=[
    "BaseEvaluation", 
    "GridSearch", 
    "GridSearchMultiple", 
    "get_best_kPCA_params", 
    "get_scorers", 
    "getGlobalScores", 
    "getSplitBestScores", 
    "displayCVTables", 
    "displayFineTunedResults", 
    "displayModelMaxDetails", 
    "naive_evaluation", 
    "pModels"
    ]