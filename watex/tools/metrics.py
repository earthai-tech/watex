# -*- coding: utf-8 -*-
# Copyright (c) 2021 Kouadio K. Laurent
#       Created on Tue Oct  5 15:09:46 2021
#       released under a MIT- licence.
#       <etanoyau@gmail.com>

import inspect
import warnings  

import numpy as np 
# from abc import ABC,abstractmethod
from sklearn import metrics 
from ..sklearn import ( 
    precision_recall_curve,
    precision_score,
    recall_score,
    confusion_matrix ,
    f1_score,
    roc_curve, 
    roc_auc_score,
    cross_val_predict, 
    )

from ._watexlog import watexlog
from .mlutils import format_generic_obj 
import watex.tools.decorators as deco
import watex.exceptions as Wex

_logger = watexlog().get_watex_logger(__name__)

__all__=['precision_recall_tradeoff', 'ROC_curve', 'confusion_matrix_']

class Metrics(object):
    """ Metrics pseudo class.
    
    Metrics are measures of quantitative assessment commonly used for 
    assessing, comparing, and tracking performance or production. Generally,
    a group of metrics will typically be used to build a dashboard that
    management or analysts review on a regular basis to maintain performance
    assessments, opinions, and business strategies.
    
    Here we implement some Scikit-learn metrics like `precision`, `recall`
    `f1_score` , `confusion matrix`, and `receiving operating characteristic`
    (R0C)
    """ 
    # @abstractmethod 
    def __init__(self): 
        setattr(self, 'metrics', tuple(metrics.SCORERS.keys()))
        
    @property
    def get_metrics(self): 
        """ Get the list of scikit_learn metrics"""
        return getattr(self, 'metrics')

    
def precision_recall_tradeoff(clf, X,y,*, cv =7,classe_ =None,
                            method="decision_function",cross_val_pred_kws =None,
                            y_tradeoff =None, **prt_kws):
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
    equation is:
    
    .. math:: 
        
        precision = TP/(TP+FP)
        
    where ``TP`` is the True Positive and ``FP`` is the False Positive
    A trival way to have perfect precision is to make one single 
    positive precision (`precision` = 1/1 =100%). This would be usefull 
    since the calssifier would ignore all but one positive instance. So 
    `precision` is typically used along another metric named `recall`,
     also `sensitivity` or `true positive rate(TPR)`:This is the ratio of 
    positive instances that are corectly detected by the classifier.  
    Equation of`recall` is given as:
    
    .. math::
        
        recall = TP/(TP+FN)
        
    where ``FN`` is of couse the number of False Negatives. 
    It's often convenient to combine `preicion`and `recall` metrics into
    a single metric call the `F1 score`, in particular if you need a 
    simple way to compared two classifiers. The `F1 score` is the harmonic 
    mean of the `precision` and `recall`. Whereas the regular mean treats 
    all  values equaly, the harmony mean gives much more weight to low 
    values. As a result, the classifier will only get the `F1 score` if 
    both `recalll` and `preccion` are high. The equation is given below:
    
    .. math::
        
        F1= 2/((1/precision)+(1/recall))\\
        F1= 2* precision*recall /(precision+recall)\\
        F1 = TP/(TP+ (FN +FP)/2)
    
    The way to increase the precion and reduce the recall and vice versa
    is called `preicionrecall tradeoff`.
    
    Examples
    --------
    >>> from sklearn.linear_model import SGDClassifier
    >>> from watex.tools.metrics import precision_recall_tradeoff
    >>> from watex.datasets import fetch_data 
    >>> X, y= fetch_data('Bagoue prepared')
    >>> sgd_clf = SGDClassifier()
    >>> mObj = precision_recall_tradeoff (clf = sgd_clf, X= X, y = y,
                                    classe_=1, cv=3 , y_tradeoff=0.90) 
    >>> mObj.confusion_matrix
    """
    
    # check y if value to plot is binarized ie.True of false 
    y_unik = np.unique(y)
    if len(y_unik )!=2 and classe_ is None: 

        warnings.warn('Classes value of `y` is %s, but need 2.' 
                      '`PrecisionRecall Tradeoff` is used for training '
                       'binarize classifier'%len(y_unik ), UserWarning)
        _logger.warning('Need a binary classifier(2). %s are given'
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
            raise Wex.ArgumentError(
                'Value must contain a least a value of label '
                    '`y`={0}'.format(
                        format_generic_obj(y).format(*list(y))))
                                 
        y=(y==classe_)
        
    if cross_val_pred_kws is None: 
        cross_val_pred_kws = dict()
        
    mObj = Metrics()#precision_recall_tradeoff
    
    mObj.y_scores = cross_val_predict(clf,X,y,cv =cv,
                                          method= method,
                                          **cross_val_pred_kws )
    y_scores = cross_val_predict(clf,X,y, cv =cv,
                                 **cross_val_pred_kws )
    
    mObj.confusion_matrix =confusion_matrix(y, y_scores )
    
    mObj.f1_score = f1_score(y,y_scores)
    mObj.precision_score = precision_score(y, y_scores)
    mObj.recall_score= recall_score(y, y_scores)
        
    if method =='predict_proba': 
        # if classifier has a `predict_proba` method like `Random_forest`
        # then use the positive class probablities as score 
        # score = proba of positive class 
        mObj.y_scores =mObj.y_scores [:, 1] 
        
    if y_tradeoff is not None:
        try : 
            float(y_tradeoff)
        except ValueError: 
            raise Wex.WATexError_float(
                f"Could not convert {y_tradeoff!r} to float.")
        except TypeError: 
            raise Wex.WATexError_inputarguments(
                f'Invalid type `{type(y_tradeoff)}`')
            
        y_score_pred = (mObj.y_scores > y_tradeoff) 
        mObj.precision_score_tradeoff = precision_score(y,
                                                        y_score_pred)
        mObj.recall_score_tradeoff = recall_score(y, 
                                                  y_score_pred)
        
    mObj.precisions, mObj.recalls, mObj.thresholds =\
        precision_recall_curve(y,
                               mObj.y_scores,
                               **prt_kws)
        
    mObj.y =y
    
    return mObj
    
@deco.docstring(precision_recall_tradeoff, start ='Parameters', end ='Notes')
def ROC_curve( roc_kws=None, **tradeoff_kws): 
    """The Receiving Operating Characteric (ROC) curve is another common
    tool  used with binary classifiers. 
    
    It s very similar to preicision/recall , but instead of plotting 
    precision versus recall, the ROC curve plots the `true positive rate`
    (TNR)another name for recall) against the `false positive rate`(FPR). 
    The FPR is the ratio of negative instances that are correctly classified 
    as positive.It is equal to one minus the TNR, which is the ratio 
    of  negative  isinstance that are correctly classified as negative.
    The TNR is also called `specify`. Hence the ROC curve plot 
    `sensitivity` (recall) versus 1-specifity.
    
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
    ---------
    `ROC_curve` deals wuth optional and positionals keywords arguments of
    :meth:`~.tools.mlutils.Metrics.precisionRecallTradeoff`.
        
    Examples
    --------
        >>> from sklearn.linear_model import SGDClassifier
        >>> from watex.tools.metrics import ROC_curve
        >>> from watex.datasets import fetch_data 
        >>> X, y= fetch_data('Bagoue prepared')
        >>> rocObj =ROC_curve(clf = sgd_clf,  X= X, 
                       y = y, classe_=1, cv=3 )                                
        >>> rocObj.__dict__.keys()
        >>> rocObj.roc_auc_score 
        >>> rocObj.fpr
    """
    mObj =Metrics()
    obj= precision_recall_tradeoff(**tradeoff_kws)
    for key in obj.__dict__.keys():
        setattr(mObj, key, obj.__dict__[key])
        
    if roc_kws is None: roc_kws =dict()
    mObj.fpr , mObj.tpr , thresholds = roc_curve(mObj.y, 
                                       mObj.y_scores,
                                       **roc_kws )
    mObj.roc_auc_score = roc_auc_score(mObj.y, mObj.y_scores)

    return mObj 

    
def confusion_matrix_(clf, X, y,*, cv =7, plot_conf_max=False, 
                     crossvalp_kws=dict(), **conf_mx_kws ): 
    """ Evaluate the preformance of the model or classifier by counting 
    the number of the times instances of class A are classified in class B. 
    
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
    
    Examples
    --------
    >>> from sklearn.svm import SVC 
    >>> from watex.tools.metrics import Metrics 
    >>> from watex.datasets import fetch_data 
    >>> X,y = fetch_data('Bagoue dataset prepared') 
    >>> svc_clf = SVC(C=100, gamma=1e-2, kernel='rbf',
    ...              random_state =42) 
    >>> confObj =confusion_matrix_(svc_clf,X=X,y=y,
    ...                        plot_conf_max='error')
    >>> confObj.norm_conf_mx
    >>> confObj.conf_mx
    >>> confObj.__dict__.keys()
    """
    # Get all param values and set attributes 
    func_sig = inspect.signature(confusion_matrix_)
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
    mObj = Metrics() #confusion_matrix_ 
    for key in PARAMS_VALUES.keys(): 
        setattr(mObj , key, PARAMS_VALUES[key] )
        
    y_pred =cross_val_predict(clf, X, y, cv=cv, **crossvalp_kws )
    
    if y_pred.ndim ==1 : 
        y_pred.reshape(-1, 1)
    conf_mx = confusion_matrix(y, y_pred, **conf_mx_kws)
    
    for att, val in zip(['y_pred', 'conf_mx'],
                        [y_pred, conf_mx]): 
        setattr(mObj , att, val)
    
    # statement to plot confusion matrix errors rather than values 
    row_sums = mObj .conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = mObj.conf_mx / row_sums 
    # now let fill the diagonal with zeros to keep only the errors
    # and let's plot the results 
    np.fill_diagonal(norm_conf_mx, 0)
    setattr(mObj , 'norm_conf_mx', norm_conf_mx)
    
    fp =0
    if plot_conf_max =='map': 
        confmax = mObj.conf_mx
        fp=1
    if plot_conf_max =='error':
        confmax= norm_conf_mx
        fp =1
    if fp: 
        import matplotlib.pyplot as plt 
        plt.matshow(confmax, cmap=plt.cm.gray)
        plt.show ()
        
    return mObj  
  
          