# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent, Wed Jul 14 20:00:26 2021
# This module is a set of utils for data prepprocessing
# released under a MIT- licence.
"""
Created on Sat Aug 28 16:26:04 2021

@author: @Daniel03

"""
import os 
import inspect
import hashlib 
import tarfile 
import warnings  
from six.moves import urllib 
from typing import TypeVar, Generic, Iterable , Callable, Text
import pandas as pd 
import numpy as np 

from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.model_selection import cross_val_predict 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import confusion_matrix , f1_score
from sklearn.metrics import roc_curve, roc_auc_score

from watex.utils._watexlog import watexlog
import watex.utils.decorator as deco
import watex.utils.exceptions as Wex
import watex.hints as Hints

T= TypeVar('T')
KT=TypeVar('KT')
VT=TypeVar('VT')

_logger = watexlog().get_watex_logger(__name__)

DOWNLOAD_ROOT = 'https://github.com/WEgeophysics/watex/master/'
#'https://zenodo.org/record/4896758#.YTWgKY4zZhE'
DATA_PATH = 'data/tar.tgz_files'
TGZ_FILENAME = '/bagoue.main&rawdata.tgz'
CSV_FILENAME = 'main.bagciv.data.csv'#'_bagoue_civ_loc_ves&erpdata4.csv'

DATA_URL = DOWNLOAD_ROOT + DATA_PATH  + TGZ_FILENAME


def read_from_excelsheets(erp_file: T = None ) -> Iterable[VT]: 
    
    """ Read all Excelsheets and build a list of dataframe of all sheets.
   
    :param erp_file:
        Excell workbooks containing `erp` profile data.
    :return: A list composed of the name of `erp_file` at index =0 and the 
            datataframes.
    """
    
    allfls:Generic[KT, VT] = pd.read_excel(erp_file, sheet_name=None)
    
    list_of_df =[os.path.basename(os.path.splitext(erp_file)[0])]
    for sheets , values in allfls.items(): 
        list_of_df.append(pd.DataFrame(values))

    return list_of_df 

def write_excel(listOfDfs: Iterable[VT], csv:bool =False , sep:T =','): 
    """ 
    Rewrite excell workbook with dataframe for :ref:`read_from_excelsheets`. 
    
    Its recover the name of the files and write the data from dataframe 
    associated with the name of the `erp_file`. 
    
    :param listOfDfs: list composed of `erp_file` name at index 0 and the
     remains dataframes. 
    :param csv: output workbook in 'csv' format. If ``False`` will return un 
     `excel` format. 
    :param sep: type of data separation. 'default is ``,``.'
    
    """
    site_name = listOfDfs[0]
    listOfDfs = listOfDfs[1:]
    for ii , df in enumerate(listOfDfs):
        
        if csv:
            df.to_csv(df, sep=sep)
        else :
            with pd.ExcelWriter(f"z{site_name}_{ii}.xlsx") as writer: 
                df.to_excel(writer, index=False)
    
   
def fetch_geo_data (data_url:str = DATA_URL, data_path:str =DATA_PATH,
                    tgz_filename =TGZ_FILENAME ) -> Text: 
    """ Fetch data from data repository in zip of 'targz_file. 
    
    I will create a `datasets/data` directory in your workspace, downloading
     the `~.tgz_file and extract the `data.csv` from this directory.
    
    :param data_url: url to the datafilename where `tgz` filename is located  
    :param data_path: absolute path to the `tgz` filename 
    :param filename: `tgz` filename. 
    """
    if not os.path.isdir(data_path): 
        os.makedirs(data_path)
    tgz_path = os.path.join(data_url, tgz_filename.replace('/', ''))
    
    urllib.request.urlretrieve(data_url, tgz_path)
    data_tgz = tarfile.open(tgz_path)
    data_tgz.extractall(path = data_path )
    data_tgz.close()
    
    
def load_data (data_path:str = DATA_PATH,
               filename:str =CSV_FILENAME, sep =',' )-> Generic[VT]:
    """ Load CSV file to pd.dataframe. 
    
    :param data_path: path to data file 
    :param filename: name of file. 
    
    """ 
    if os.path.isfile(data_path): 
        return pd.read_csv(data_path, sep)
    
    csv_path = os.path.join(data_path , filename)
    
    return pd.read_csv(csv_path, sep)


def split_train_test (data:Generic[VT], test_ratio:T)-> Generic[VT]: 
    """ Split dataset into trainset and testset from `test_ratio` 
    and return train set and test set.
        
    ..note: `test_ratio` is ranged between 0 to 1. Default is 20%.
    """
    shuffled_indices =np.random.permutation(len(data)) 
    test_set_size = int(len(data)* test_ratio)
    test_indices = shuffled_indices [:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    return data.iloc[train_indices], data.iloc[test_indices]
    
def test_set_check_id (identifier, test_ratio, hash:Callable[..., T]) -> bool: 
    """ 
    Get the testset id and set the corresponding unique identifier. 
    
    Compute the a hash of each instance identifier, keep only the last byte 
    of the hash and put the instance in the testset if this value is lower 
    or equal to 51(~20% of 256) 
    has.digest()` contains object in size between 0 to 255 bytes.
    
    :param identifier: integer unique value 
    :param ratio: ratio to put in test set. Default is 20%. 
    
    :param hash:  
        Secure hashes and message digests algorithm. Can be 
        SHA1, SHA224, SHA256, SHA384, and SHA512 (defined in FIPS 180-2) 
        as well as RSA’s MD5 algorithm (defined in Internet RFC 1321). 
        
        Please refer to :ref:`<https://docs.python.org/3/library/hashlib.html>` 
        for futher details.
    """
    return hash(np.int64(identifier)).digest()[-1]< 256 * test_ratio

def split_train_test_by_id(data, test_ratio:T, id_column:T=None,
                           hash=hashlib.md5)-> Generic[VT]: 
    """Ensure that data will remain consistent accross multiple runs, even if 
    dataset is refreshed. 
    
    The new testset will contain 20%of the instance, but it will not contain 
    any instance that was previously in the training set.

    :param data: Pandas.core.DataFrame 
    :param test_ratio: ratio of data to put in testset 
    :id_colum: identifier index columns. If `id_column` is None,  reset  
                dataframe `data` index and set `id_column` equal to ``index``
    :param hash: secures hashes algorithms. Refer to 
                :func:`~test_set_check_id`
    :returns: consistency trainset and testset 
    """
    if id_column is None: 
        id_column ='index' 
        data = data.reset_index() # adds an `index` columns
        
    ids = data[id_column]
    in_test_set =ids.apply(lambda id_:test_set_check_id(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

def discretizeCategoriesforStratification(data, in_cat:str =None,
                               new_cat:str=None, **kws) -> Generic[VT]: 
    """ Create a new category attribute to discretize instances. 
    
    A new category in data is better use to stratified the trainset and 
    the dataset to be consistent and rounding using ceil values.
    
    :param in_cat: column name used for stratified dataset 
    :param new_cat: new category name created and inset into the 
                dataframe.
    :return: new dataframe with new column of created category.
    """
    divby = kws.pop('divby', 1.5) # normalize to hold raisonable number 
    combined_cat_into = kws.pop('higherclass', 5) # upper class bound 
    
    data[new_cat]= np.ceil(data[in_cat]) /divby 
    data[new_cat].where(data[in_cat] < combined_cat_into, 
                             float(combined_cat_into), inplace =True )
    return data 

def stratifiedUsingDiscretedCategories(data:VT , cat_name:str , n_splits:int =1, 
                    test_size:float= 0.2, random_state:int = 42)-> Generic[VT]: 
    """ Stratified sampling based on new generated category  from 
    :func:`~DiscretizeCategoriesforStratification`.
    
    :param data: dataframe holding the new column of category 
    :param cat_name: new category name inserted into `data` 
    :param n_splits: number of splits 
    """
    
    split = StratifiedShuffleSplit(n_splits, test_size, random_state)
    for train_index, test_index in split.split(data, data[cat_name]): 
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index] 
        
    return strat_train_set , strat_test_set 

 
class Metrics: 
    """ Metric class.
    
    Metrics are measures of quantitative assessment commonly used for 
    assessing, comparing, and tracking performance or production. Generally,
    a group of metrics will typically be used to build a dashboard that
    management or analysts review on a regular basis to maintain performance
    assessments, opinions, and business strategies.
    
    Here we implement some Scikit-learn metrics like `precision`, `recall`
    `f1_score` , `confusion matrix`, and `receiving operating characteristic`
    (R0C)
    """ 
    
    def precisionRecallTradeoff(self, 
                                clf,
                                X,
                                y,
                                cv =7,
                                classe_ =None,
                                method="decision_function",
                                cross_val_pred_kws =None,
                                y_tradeoff =None, 
                                **prt_kws):
        """ Precision/recall Tradeoff computes a score based on the decision 
        function. 
        
        Is assign the instance to the positive class if that score on 
        the left is greater than the `threshold` else it assigns to negative 
        class. 
        
        Parameters
        ----------
        
        clf: obj
            classifier or estimator
            
        X: ndarray, 
            Training data (trainset) composed of n-features.
            
        y: array_like 
            labelf for prediction. `y` is binary label by defaut. 
            If '`y` is composed of multilabel, specify  the `classe_` 
            argumentto binarize the label(`True` ot `False`). ``True``  
            for `classe_`and ``False`` otherwise.
            
        cv: int 
            K-fold cross validation. Default is ``3``
            
        classe_: float, int 
            Specific class to evaluate the tradeoff of precision 
            and recall. If `y` is already a binary classifer, `classe_` 
            does need to specify. 
            
        method: str
            Method to get scores from each instance in the trainset. 
            Ciuld be ``decison_funcion`` or ``predict_proba`` so 
            Scikit-Learn classifuier generally have one of the method. 
            Default is ``decision_function``.
        
        y_tradeoff: float
            check your `precision score` and `recall score`  with a 
            specific tradeoff. Suppose  to get a precision of 90%, you 
            might specify a tradeoff and get the `precision score` and 
            `recall score` by setting a `y-tradeoff` value.

        Notes
        ------
            
        Contreverse to the `confusion matrix`, a precision-recall 
        tradeoff is very interesting metric to get the accuracy of the 
        positive prediction named ``precison`` of the classifier with 
        equation is:: 
            
            precision = TP/(TP+FP)
            
        where ``TP`` is the True Positive and ``FP`` is the False Positive
        A trival way to have perfect precision is to make one single 
        positive precision (`precision` = 1/1 =100%). This would be usefull 
        since the calssifier would ignore all but one positive instance. So 
        `precision` is typically used along another metric named `recall`,
         also `sensitivity` or `true positive rate(TPR)`:This is the ratio of 
        positive instances that are corectly detected by the classifier.  
        Equation of`recall` is given as:: 
            
            recall = TP/(TP+FN)
            
        where ``FN`` is of couse the number of False Negatives. 
        It's often convenient to combine `preicion`and `recall` metrics into
        a single metric call the `F1 score`, in particular if you need a 
        simple way to compared two classifiers. The `F1 score` is the harmonic 
        mean of the `precision` and `recall`. Whereas the regular mean treats 
        all  values equaly, the harmony mean gives much more weight to low 
        values. As a result, the classifier will only get the `F1 score` if 
        both `recalll` and `preccion` are high. The equation is given below::
            
            F1= 2/((1/precision)+(1/recall))
            F1= 2* precision*recall/(precision+recall)
            F1 = TP/(TP+ (FN +FP)/2)
        
        The way to increase the precion and reduce the recall and vice versa
        is called `preicionrecall tradeoff`
        
        Examples
        --------
        
        >>> from sklearn.linear_model import SGDClassifier
        >>> from watex.utils.ml_utils import Metrics 
        >>> sgd_clf = SGDClassifier()
        >>> mObj = Metrics(). precisionRecallTradeoff(clf = sgd_clf, 
        ...                                           X= X_train_2, 
        ...                                         y = y_prepared, 
        ...                                         classe_=1, cv=3 )                                
        >>> mObj.confusion_matrix 
        >>> mObj.f1_score
        >>> mObj.precision_score
        >>> mObj.recall_score
        """
        
        # check y if value to plot is binarized ie.True of false 
        y_unik = np.unique(y)
        if len(y_unik )!=2 and classe_ is None: 

            warnings.warn('Classes value of `y` is %s, but need 2.' 
                          '`PrecisionRecall Tradeoff` is used for training '
                           'binarize classifier'%len(y_unik ), UserWarning)
            self._logging.warning('Need a binary classifier(2). %s are given'
                                  %len(y_unik ))
            raise ValueError(f'Need binary classes but {len(y_unik )!r}'
                             f' {"are" if len(y_unik )>1 else "is"} given')
            
        if classe_ is not None: 
            try : 
                classe_= int(classe_)
            except ValueError: 
                raise Wex.WATexError_inputarguments(
                    'Need integer value. Could not convert to Float.')
            except TypeError: 
                raise Wex.WATexError_inputarguments(
                    'Could not convert {type(classe_)!r}') 
        
            if classe_ not in y: 
                raise Wex.WATexError_inputarguments(
                    'Value must contain a least a value of label '
                        '`y`={0}'.format(
                            Hints.format_generic_obj(y).format(*list(y))))
                                     
            y=(y==classe_)
            
        if cross_val_pred_kws is None: 
            cross_val_pred_kws = dict()
            
        self.y_scores = cross_val_predict(clf,
                                          X, 
                                          y, 
                                          cv =cv,
                                          method= method,
                                          **cross_val_pred_kws )

        y_scores = cross_val_predict(clf,
                                     X,
                                     y, 
                                     cv =cv,
                                     **cross_val_pred_kws )
        self.confusion_matrix =confusion_matrix(y, y_scores )
        
        self.f1_score = f1_score(y,y_scores)
        self.precision_score = precision_score(y, y_scores)
        self.recall_score= recall_score(y, y_scores)
            
        if method =='predict_proba': 
            # if classifier has a `predict_proba` method like `Random_forest`
            # then use the positive class probablities as score 
            # score = proba of positive class 
            self.y_scores =self.y_scores [:, 1] 
            
        if y_tradeoff is not None:
            try : 
                float(y_tradeoff)
            except ValueError: 
                raise Wex.WATexError_float(
                    f'Could not convert {y_tradeoff!r} to float.')
            except TypeError: 
                raise Wex.WATexError_inputarguments(
                    f'Invalid type `{type(y_tradeoff)}`')
                
            y_score_pred = (self.y_scores > y_tradeoff) 
            self.precision_score_tradeoff = precision_score(y,
                                                            y_score_pred)
            self.recall_score_tradeoff = recall_score(y, 
                                                      y_score_pred)
            
        self.precisions, self.recalls, self.thresholds =\
            precision_recall_curve(y,
                                   self.y_scores,
                                   **prt_kws)
            
        self.y =y
        
        return self 
    
    @deco.docstring(precisionRecallTradeoff, start ='Parameters', end ='Notes')
    def ROC_curve(self, roc_kws=None, **tradeoff_kws): 
        """The Receiving Operating Characteric (ROC) curve is another common
        tool  used with binary classifiers. 
        
        It s very similar to preicision/recall , but instead of plotting 
        precision versus recall, the ROC curve plots the `true positive rate`
        (TNR)another name for recall) against the `false positive rate`(FPR). 
        The FPR is the ratio of negative instances that are correctly classified 
        as positive.It is equal to one minus the TNR, which is the ratio 
        of  negative  isinstance that are correctly classified as negative.
        The TNR is also called `specify`. Hence the ROC curve plot 
        `sensitivity`(recall) versus 1-specifity.
        
        Parameters 
        ----------
        clf: callable
            classifier or estimator
                
        X: ndarray, 
            Training data (trainset) composed of n-features.
            
        y: array_like 
            labelf for prediction. `y` is binary label by defaut. 
            If '`y` is composed of multilabel, specify  the `classe_` 
            argumentto binarize the label(`True` ot `False`). ``True``  
            for `classe_`and ``False`` otherwise.
            
        roc_kws: dict 
            roc_curve additional keywords arguments
            
        See also
        --------
        
            `ROC_curve` deals wuth optional and positionals keywords arguments 
            of :meth:`~watex.utlis.ml_utils.Metrics.precisionRecallTradeoff`
            
        Examples
        ---------
        
            >>> from sklearn.linear_model import SGDClassifier
            >>> from watex.utils.ml_utils import Metrics 
            >>> sgd_clf = SGDClassifier()
            >>> rocObj = Metrics().ROC_curve(clf = sgd_clf,  X= X_train_2, 
            ...                                 y = y_prepared, classe_=1, cv=3 )
            >>> rocObj.fpr
        """
        self.precisionRecallTradeoff(**tradeoff_kws)
        if roc_kws is None: roc_kws =dict()
        self.fpr , self.tpr , thresholds = roc_curve(self.y, 
                                           self.y_scores,
                                           **roc_kws )
        self.roc_auc_score = roc_auc_score(self.y, self.y_scores)
        
        return self 
    
    
    def confusion_matrix(self, clf, X, y,*, cv =7, plot_conf_max=False, 
                         crossvalp_kws=dict(), **conf_mx_kws ): 
        """ Evaluate the preformance of the model or classifier by counting 
        the number of ttimes instances of class A are classified in class B. 
        
        To compute a confusion matrix, you need first to have a set of 
        prediction, so they can be compared to the actual targets. You could 
        make a prediction using the test set, but it's better to keep it 
        untouch since you are not ready to make your final prediction. Remember 
        that we use the test set only at very end of the project, once you 
        have a classifier that you are ready to lauchn instead. 
        The confusion metric give a lot of information but sometimes we may 
        prefer a more concise metric like `precision` and `recall` metrics. 
        
        Parameters 
        ----------
        clf: obj
            classifier or estimator
            
        X: ndarray, 
            Training data (trainset) composed of n-features.
            
        y: array_like 
            labelf for prediction. `y` is binary label by defaut. 
            If '`y` is composed of multilabel, specify  the `classe_` 
            argumentto binarize the label(`True` ot `False`). ``True``  
            for `classe_`and ``False`` otherwise.
            
        cv: int 
            K-fold cross validation. Default is ``7``
            
        plot_conf_max: bool, str 
            can be `map` or `error` to visualize the matshow of prediction 
            and errors 

        crossvalp_kws: dict 
            crossvalpredict additional keywords arguments 
            
        conf_mx_kws: dict 
            Additional confusion matrix keywords arguments.
        
        Example
        --------
            
            >>> from sklearn.svm import SVC 
            >>> from watex.utils.ml_utils import Metrics 
            >>> from watex.datasets import fetch_data 
            X,y = fetch_data('Bagoue dataset prepared') 
            >>> svc_clf = SVC(C=100, gamma=1e-2, kernel='rbf',
                          random_state =42) 
            >>> mObj =Metrics().confusion_matrix(svc_clf,X=X,y=y,
                                            plot_conf_max='map')
        """
        # Get all param values and set attributes 
        func_sig = inspect.signature(Metrics.confusion_matrix)
        PARAMS_VALUES = {k: v.default
            for k, v in func_sig.parameters.items()
            if v.default is not  inspect.Parameter.empty
            }
        # add positional params 
        for pname, pval in zip( ['X', 'y', 'clf'], [X, y, clf]): 
            PARAMS_VALUES[pname]=pval 
            
        # PARAMS_VALUES2 = {k: v
        #     for k, v in func_sig.parameters.items()
        #     if (v.default is inspect.Parameter.empty and k !='self')
        #     }
        # parameters = [p.name for p in func_sig.parameters.values()
               # if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for key in PARAMS_VALUES.keys(): 
            setattr(self, key, PARAMS_VALUES[key] )
            
        y_pred =cross_val_predict(clf, X, y, cv=cv, **crossvalp_kws )
        
        if y_pred.ndim ==1 : 
            y_pred.reshape(-1, 1)
        conf_mx = confusion_matrix(y, y_pred, **conf_mx_kws)
        
        for att, val in zip(['y_pred', 'conf_mx'],
                            [y_pred, conf_mx]): 
            setattr(self, att, val)
        
        # statement to plot confusion matrix errors rather than values 
        row_sums = self.conf_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = self.conf_mx / row_sums 
        # now let fill the diagonal with zeros to keep only the errors
        # and let's plot the results 
        np.fill_diagonal(norm_conf_mx, 0)
        setattr(self, 'norm_conf_mx', norm_conf_mx)
        
          
        fp =0
        if plot_conf_max =='map': 
            confmax = self.conf_mx
            fp=1
        if plot_conf_max =='error':
            confmax= norm_conf_mx
            fp =1
        if fp: 
            import matplotlib.pyplot as plt 
            plt.matshow(confmax, cmap=plt.cm.gray)
            plt.show ()
            
        return self 
            
        
# if __name__=="__main__": 
#     if __package__ is None : 
#         __package__='watex'
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.linear_model import SGDClassifier
#     from .datasets import X_, y_,  X_prepared, y_prepared, default_pipeline

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
