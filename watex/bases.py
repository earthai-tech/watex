# -*- coding: utf-8 -*-
#       Author: Kouadio K.Laurent<etanoyau@gmail.con>
#       Create:on Fri Sep 10 15:37:59 2021
#       Licence: MIT

import os 
import re
import sys 
import warnings 
import pickle 
import joblib
from pprint import pprint  
import pandas as pd
from typing import TypeVar, Callable

T= TypeVar('T')
if __name__ =='__main__' or __package__ is None: 
    sys.path.append( os.path.dirname(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.dirname(__file__))
    __package__ ='watex'
   
import watex.utils.decorator as dec 
import watex.utils.exceptions as Wex
from .utils.__init__ import savepath as savePath 
from .utils._watexlog import watexlog

__logger = watexlog().get_watex_logger(__name__)

OptsList, paramsList =[['bore', 'for'], 
                        ['x','east'], 
                        ['y', 'north'], 
                        ['pow', 'puiss', 'pa'], 
                        ['magn', 'amp', 'ma'], 
                        ['shape', 'form'], 
                        ['type'], 
                        ['sfi', 'if'], 
                        ['lat'], 
                        ['lon'], 
                        ['lwi', 'wi'], 
                        ['ohms', 'surf'], 
                        ['geol'], 
                        ['flow', 'deb']
                        ], ['id', 
                           'east', 
                           'north', 
                           'power', 
                           'magnitude', 
                           'shape', 
                           'type', 
                           'sfi', 
                           'lat', 
                           'lon', 
                           'lwi', 
                           'ohmS', 
                           'geol', 
                           'flow'
                           ]
            
def fetch_model(modelfile:str, modelpath:str =None, default:bool=True,
                modname:str =None, verbose:int =0): 
    """ Fetch your model saved using Python pickle module or 
    joblib module. 
    
    :param modelfile: str or Path-Like object 
        dumped model file name saved using `joblib` or Python `pickle` module.
    :param modelpath: path-Like object , 
        Path to model dumped file =`modelfile`
    :default: bool, 
        Model parameters by default are saved into a dictionary. When default 
        is ``True``, returns a tuple of pair (the model and its best parameters)
        . If False return all values saved from `~.MultipleGridSearch`
       
    :modname: str 
        Is the name of model to retrived from dumped file. If name is given 
        get only the model and its best parameters. 
    :verbose: int, level=0 
        control the verbosity.More message if greater than 0.
    
    :returns:
        - `model_class_params`: if default is ``True``
        - `pickledmodel`: model dumped and all parameters if default is `False`
        
    :Example: 
        >>> from watex.bases import fetch_model 
        >>> my_model = fetch_model ('SVC__LinearSVC__LogisticRegression.pkl',
                                    default =False,  modname='SVC')
        >>> my_model
    """
    
    try:
        isdir =os.path.isdir( modelpath)
    except TypeError: 
        #stat: path should be string, bytes, os.PathLike or integer, not NoneType
        isdir =False
        
    if isdir and modelfile is not None: 
        modelfile = os.join.path(modelpath, modelfile)

    isfile = os.path.isfile(modelfile)
    if not isfile: 
        raise FileNotFoundError ("File {modelfile!r} not found!")
        
    from_joblib =False 
    if modelfile.endswith('.pkl'): from_joblib  =True 
    
    if from_joblib:
       __logger.info(f"Loading models `{os.path.basename(modelfile)}`!")
       try : 
           pickledmodel = joblib.load(modelfile)
           # and later ....
           # f'{pickfname}._loaded' = joblib.load(f'{pickfname}.pkl')
           dmsg=f"Model {modelfile !r} retreived from~.externals.joblib`!"
       except : 
           dmsg=''.join([f"Nothing to retrived. It's seems model {modelfile !r}", 
                         " not really saved using ~external.joblib module! ", 
                         "Please check your model filename."])
    
    if not from_joblib: 
        __logger.info(f"Loading models `{os.path.basename(modelfile)}`!")
        try: 
           # DeSerializing pickled data 
           with open(modelfile, 'rb') as modf: 
               pickledmodel= pickle.load (modf)
           __logger.info(f"Model `{os.path.basename(modelfile)!r} deserialized"
                         "  using Python pickle module.`!")
           
           dmsg=f"Model {modelfile!r} deserizaled from  {modelfile!r}!"
        except: 
            dmsg =''.join([" Unable to deserialized the "
                           f"{os.path.basename(modelfile)!r}"])
           
        else: 
            __logger.info(dmsg)   
           
    if verbose > 0: 
        pprint(
            dmsg 
            )
           
    if modname is not None: 
        keymess = "{modname!r} not found."
        try : 
            if default:
                model_class_params  =( pickledmodel[modname]['best_model'], 
                                   pickledmodel[modname]['best_params_'], 
                                   pickledmodel[modname]['best_scores'],
                                   )
            if not default: 
                model_class_params=pickledmodel[modname]
                
        except KeyError as key_error: 
            warnings.warn(
                f"Model name {modname!r} not found in the list of dumped"
                f" models = {list(pickledmodel.keys()) !r}")
            raise KeyError from key_error(keymess + "Shoud try the model's"
                                          f"names ={list(pickledmodel.keys())!r}")
        
        if verbose > 0: 
            pprint('Should return a tuple of `best model` and the'
                   ' `model best parameters.')
           
        return model_class_params  
            
    if default:
        model_class_params =list()    
        
        for mm in pickledmodel.keys(): 
            model_class_params.append((pickledmodel[mm]['best_model'], 
                                      pickledmodel[mm]['best_params_'],
                                      pickledmodel[modname]['best_scores']))
    
        if verbose > 0: 
               pprint('Should return a list of tuple pairs:`best model`and '
                      ' `model best parameters.')
               
        return model_class_params

    return pickledmodel


def sanitize_fdataset(_df): 
    """ Sanitize the feature dataset. Recognize the columns provided 
    by the users and resset according to the features labels disposals
    :attr:`~Features.featureLabels`."""
    
    UTM_FLAG =0 
    
    def getandReplace(optionsList, params, df): 
        """
        Function to  get parames and replace to the main features params.
        
        :param optionsList: 
            User options to qualified the features headlines. 
        :type optionsList: list
        
        :param params: Exhaustive parameters names. 
        :type params: list 
        
        :param df: pd.DataFrame collected from `features_fn`. 
        
        :return: sanitize columns
        :rtype: list 
        """
        columns = [c.lower() for c in df.columns] 
        
        for ii, celemnt in enumerate(columns): 
            for listOption, param in zip(optionsList, params): 
                 for option in listOption:
                     if param =='lwi': 
                        if celemnt.find('eau')>=0 : 
                            columns[ii]=param 
                            break
                     if re.match(r'^{0}+'.format(option), celemnt):
                         columns[ii]=param
                         if columns[ii] =='east': 
                             UTM_FLAG=1
                         break

        return columns

    new_df_columns= getandReplace(optionsList=OptsList, params=paramsList,
                                  df= _df)
    df = pd.DataFrame(data=_df.to_numpy(), 
                           columns= new_df_columns)
    return df , UTM_FLAG
     
   
@dec.writef(reason='write', from_='df')
def exportdf (df =None, refout:str =None,  to:str =None, savepath:str =None,
              modname:str  ='_wexported_', reset_index:bool =True): 
    """ 
    Export dataframe ``df``  to `refout` files. `refout` file can 
    be Excell sheet file or '.json' file. To get more details about 
    the `writef` decorator , see :doc:`watex.utils.decorator.writef`. 
    
    :param refout: 
        Output filename. If not given will be created refering to the 
        exported date. 
        
    :param to: Export type; Can be `.xlsx` , `.csv`, `.json` and else.
       
    :param savepath: 
        Path to save the `refout` filename. If not given
        will be created.
    :param modname: Folder to hold the `refout` file. Change it accordingly.
        
    :returns: 
        - `df_`: new dataframe to be exported. 
        
    """
    if df is None :
        warnings.warn(
            'Once ``df`` arguments in decorator :`class:~decorator.writef`'
            ' is selected. The main type of file ready to be written MUST be '
            'a pd.DataFrame format. If not an error raises. Please refer to '
            ':doc:`~.utils.decorator.writef` for more details.')
        
        raise Wex.WATexError_file_handling(
            'No dataframe detected. Please provided your dataFrame.')

    df_ =df.copy(deep=True)
    if reset_index is True : 
        df_.reset_index(inplace =True)
    if savepath is None :
       savepath = savePath(modname)
        
    return df_, to,  refout, savepath, reset_index 

# #--------------Evaluate your model on the test data ------------------------------
# my_model, *_ = fetch_model('SVC__LinearSVC__LogisticRegression.pkl', modname ='SVC') 
# #---------------------------------------------------------------------------------

def _pred_statistics(y_true, *,  y_pred=None, X=None, X_=None, 
                     clf:Callable[..., T]=None,verbose:int=0): 
    """ Make a quick statistic after prediction. 
    
    :param y_true: array-like 
        y value (label) to predict
    :param y_pred: array_like
        y value predicted
    :pram X: ndarray(nexamples, nfeatures)
        Training data sets 
    :param X_: ndarray(nexamples, nfeatures)
        test sets 
    :param clf: callable
        Estimator or classifier object. 
    :param verbose:int, level=0 
        Control the verbosity. More than 1 more message
    """
    
    clf_name =''
    if y_pred is None: 
        if clf is None: 
            warnings.warn('None estimator found! Could not predict `y` ')
            __logger.error('NoneType `clf` <estimator> could not'
                                ' predict `y`.')
            raise ValueError('None estimator detected!'
                             ' could not predict `y`.') 
        # check whether is 
        is_clf = hasattr(clf, '__call__')
        if is_clf : clf_name = clf.__name__

        if not is_clf :
            # try whether is ABCMeta class 
            try : 
                is_clf = hasattr(clf.__class__, '__call__')
            except : 
                raise TypeError(f"{clf!r} is not a model classifier or estimator. "
                                 " Could not use for prediction.")
            clf_name = clf.__class__.__name__
        
            # check estimator 
            
        if X_ is None: 
            raise TypeError('NoneType can not used for prediction.'
                            ' Need a test set `X`.')
  
        clf.fit(X_, y_true)
        y_pred = clf.predict(X_)
        
    if len(y_true) !=len(y_pred): 
        raise TypeError("`y_true` and `y_pred` must have the same length." 
                        f" {len(y_true)!r} and {len(y_pred)!r} were given"
                        " respectively.")
        
    # get the model score apres prediction 
    clf_score = round(sum(y_true ==y_pred)/len(y_true), 4)
    dms = f"Overall model {clf_name!r} score ={clf_score *100 } % "
    
    from sklearn.metrics import confusion_matrix , mean_squared_error
    
    conf_mx =confusion_matrix(y_true, y_pred)
    dms +=f"\n Confusion matrix= \n {conf_mx}"
    mse = mean_squared_error(y_true, y_pred )
    dms += f"\n MSE error = {mse *100} %."

    return clf_score, conf_mx  
        