# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Created on Tue Oct  5 15:09:46 2021

"""
Metrics
==========

Metrics are measures of quantitative assessment commonly used for 
assessing, comparing, and tracking performance or production. Generally,
a group of metrics will typically be used to build a dashboard that
management or analysts review on a regular basis to maintain performance
assessments, opinions, and business strategies.

Here we implement some `Scikit-learn`_ metrics like `precision`, `recall`
`f1_score` , `confusion matrix`, and `receiving operating characteristic`
(R0C)

.. _Scikit_learn: https://scikit-learn.org/
"""
from __future__ import annotations 

import warnings  
import numpy as np 
from sklearn import metrics 

from ._docstring import ( 
    DocstringComponents,
    _core_docs,
    )
from .typing import ( 
    List, 
    Optional, 
    ArrayLike , 
    NDArray,
    F
    
    )
from ._watexlog import watexlog
from .exlib import ( 
    precision_recall_curve,
    precision_score,
    recall_score,
    confusion_matrix ,
    f1_score,
    roc_curve, 
    roc_auc_score,
    cross_val_predict, 
    )

from .exceptions import ( 
    ArgumentError 
    )

_logger = watexlog().get_watex_logger(__name__)

__all__=['precision_recall_tradeoff', 'ROC_curve', 'confusion_matrix_']

#----add metrics docs 
_metrics_params =dict (
    classe_="""
classe_: float, int 
    Specific class to evaluate the tradeoff of precision 
    and recall. If `y` is already a binary classifer, `classe_` 
    does need to specify.     
    """, 
    method="""
method: str
    Method to get scores from each instance in the trainset. 
    Ciuld be ``decison_funcion`` or ``predict_proba`` so 
    Scikit-Learn classifuier generally have one of the method. 
    Default is ``decision_function``.   
    """, 
    y_tradeoff="""
"y_tradeoff: float
    check your `precision score` and `recall score`  with a 
    specific tradeoff. Suppose  to get a precision of 90%, you 
    might specify a tradeoff and get the `precision score` and 
    `recall score` by setting a `y-tradeoff` value.
    """
    )
_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"],
    metric=DocstringComponents(_metrics_params ), 
    )

# ------

def get_metrics(): 
    """ Get the list of `Scikit_learn`_  metrics. 
    
    Metrics are measures of quantitative assessment commonly used for 
    assessing, comparing, and tracking performance or production. Generally,
    a group of metrics will typically be used to build a dashboard that
    management or analysts review on a regular basis to maintain performance
    assessments, opinions, and business strategies.
    
    """
    return tuple(metrics.SCORERS.keys())

def _assert_metrics_args(y, classe_): 
    """ Assert metrics argument 
    
    :param y: array-like, 
        label for prediction. `y` is binary label by default. 
        If `y` is composed of multilabel, specify  the `classe_` 
        argumentto binarize the label(`True` ot `False`). ``True``  
        for `classe_`and ``False`` otherwise. 
    :param classe_:float, int 
        Specific class to evaluate the tradeoff of precision 
        and recall. If `y` is already a binary classifer, `classe_` 
        does need to specify.     
    """
    # check y if value to plot is binarized ie.True of false 
    y_unik = np.unique(y)
    if len(y_unik )!=2 and classe_ is None: 
        warnings.warn('Classe values of `y` is %s, but need 2.' 
                      '`PrecisionRecall Tradeoff` is used for training '
                       'binarize classifier'%len(y_unik ), UserWarning)
        _logger.warning('Need a binary classifier(2), but %s are given'
                              %len(y_unik ))
        raise ValueError(f'Need binary classes but {len(y_unik )!r}'
                         f' {"are" if len(y_unik )>1 else "is"} given')
        
    if classe_ is not None: 
        try : 
            classe_= int(classe_)
        except ValueError: 
            raise ValueError('Need integer value; Could not convert to Float.')
        except TypeError: 
            raise TypeError('Could not convert {type(classe_)!r}') 
    
        if classe_ not in y: 
            raise ArgumentError(
                'Value must contain a least a binarize class label')
  
def precision_recall_tradeoff(
    clf:F, 
    X:NDArray,
    y:ArrayLike,
    *,
    cv:int =7,
    classe_: str | Optional[List[str]]=None,
    method:str ="decision_function",
    cross_val_pred_kws: Optional[dict]  =None,
    y_tradeoff: Optional[float] =None,
    **prt_kws
)-> object:
    #create a object to hold attributes 
    obj = type('Metrics', (), {})
    
    _assert_metrics_args(y, classe_)
    y=(y==classe_) # set boolean 
    
    if cross_val_pred_kws is None: 
        cross_val_pred_kws = dict()
        
    obj.y_scores = cross_val_predict(
        clf,
         X,
         y,
         cv =cv,
        method= method,
        **cross_val_pred_kws 
    )
    y_scores = cross_val_predict(
        clf,
        X,
        y, 
        cv =cv,
        **cross_val_pred_kws 
        )
    
    obj.confusion_matrix =confusion_matrix(y, y_scores )
    
    obj.f1_score = f1_score(y,y_scores)
    obj.precision_score = precision_score(y, y_scores)
    obj.recall_score= recall_score(y, y_scores)
        
    if method =='predict_proba': 
        # if classifier has a `predict_proba` method like 
        # `Random_forest` then use the positive class
        # probablities as score  score = proba of positive 
        # class 
        obj.y_scores =obj.y_scores [:, 1] 
        
    if y_tradeoff is not None:
        try : 
            float(y_tradeoff)
        except ValueError: 
            raise ValueError(f"Could not convert {y_tradeoff!r} to float.")
        except TypeError: 
            raise TypeError(f'Invalid type `{type(y_tradeoff)}`')
            
        y_score_pred = (obj.y_scores > y_tradeoff) 
        obj.precision_score = precision_score(y, y_score_pred)
        obj.recall_score = recall_score(y, y_score_pred)
        
    obj.precisions, obj.recalls, obj.thresholds =\
        precision_recall_curve(y, obj.y_scores,**prt_kws)
        
    obj.y =y
    
    return obj

precision_recall_tradeoff.__doc__=r"""\
Precision/recall Tradeoff computes a score based on the decision 
function. 

Is assign the instance to the positive class if that score on 
the left is greater than the `threshold` else it assigns to negative 
class. 

Parameters
----------
{params.core.clf}
{params.core.X}
{params.core.y}
{params.core.cv}

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

.. math:: precision = TP/(TP+FP)
    
where ``TP`` is the True Positive and ``FP`` is the False Positive
A trival way to have perfect precision is to make one single 
positive precision (`precision` = 1/1 =100%). This would be usefull 
since the calssifier would ignore all but one positive instance. So 
`precision` is typically used along another metric named `recall`,
 also `sensitivity` or `true positive rate(TPR)`:This is the ratio of 
positive instances that are corectly detected by the classifier.  
Equation of`recall` is given as:

.. math:: recall = TP/(TP+FN)
    
where ``FN`` is of couse the number of False Negatives. 
It's often convenient to combine `preicion`and `recall` metrics into
a single metric call the `F1 score`, in particular if you need a 
simple way to compared two classifiers. The `F1 score` is the harmonic 
mean of the `precision` and `recall`. Whereas the regular mean treats 
all  values equaly, the harmony mean gives much more weight to low 
values. As a result, the classifier will only get the `F1 score` if 
both `recalll` and `preccion` are high. The equation is given below:

.. math::
    
    F1 &= 2/((1/precision)+(1/recall))= 2* precision*recall /(precision+recall) \\ 
       &= TP/(TP+ (FN +FP)/2)
    
The way to increase the precion and reduce the recall and vice versa
is called `preicionrecall tradeoff`.

Returns 
--------
obj: object, an instancied metric tying object 
    The metric object is composed of the following attributes:: 
        * `confusion_matrix` 
        * `f1_score`
        * `precision_score`
        * `recall_score`
        * `precisions` from `precision_recall_curve` 
        * `recalls` from `precision_recall_curve` 
        * `thresholds` from `precision_recall_curve` 
        * `y` classified 
    and can be retrieved for plot purpose.    
  
Examples
--------
>>> from watex.exlib import SGDClassifier
>>> from watex.metrics import precision_recall_tradeoff
>>> from watex.datasets import fetch_data 
>>> X, y= fetch_data('Bagoue prepared')
>>> sgd_clf = SGDClassifier()
>>> mObj = precision_recall_tradeoff (clf = sgd_clf, X= X, y = y,
                                classe_=1, cv=3 , y_tradeoff=0.90) 
>>> mObj.confusion_matrix
""".format(
    params =_param_docs
)
    
def ROC_curve( 
    roc_kws:dict =None, 
    **tradeoff_kws
)-> object: 

    obj= precision_recall_tradeoff(**tradeoff_kws)
    # for key in obj.__dict__.keys():
    #     setattr(mObj, key, obj.__dict__[key])
    if roc_kws is None: roc_kws =dict()
    obj.fpr , obj.tpr , thresholds = roc_curve(obj.y, 
                                       obj.y_scores,
                                       **roc_kws )
    obj.roc_auc_score = roc_auc_score(obj.y, obj.y_scores)

    return obj 

ROC_curve.__doc__ ="""\
The Receiving Operating Characteric (ROC) curve is another common
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
{params.core.clf}
{params.core.X}
{params.core.y}
{params.core.cv}
{params.metric.classe_}
{params.metric.method}
{params.metric.y_tradeoff}

roc_kws: dict 
    roc_curve additional keywords arguments
    
See also
---------
`ROC_curve` deals with optional and positionals keywords arguments of
:meth:`watex.view.mlplot.MLPlot.precisionRecallTradeoff`.
    
Returns 
---------
obj: object, an instancied metric tying object 
    The metric object hold the following attributes additional to the return
    attributes from :func:~.precision_recall_tradeoff`:: 
        * `roc_auc_score` for area under the curve
        * `fpr` for false positive rate 
        * `tpr` for true positive rate 
        * `thresholds` from `roc_curve` 
        * `y` classified 
    and can be retrieved for plot purpose.    
    
Examples
--------
>>> from watex.exlib import SGDClassifier
>>> from watex.metrics import ROC_curve
>>> from watex.datasets import fetch_data 
>>> X, y= fetch_data('Bagoue prepared')
>>> rocObj =ROC_curve(clf = sgd_clf,  X= X, 
               y = y, classe_=1, cv=3 )                                
>>> rocObj.__dict__.keys()
>>> rocObj.roc_auc_score 
>>> rocObj.fpr

""".format(
    params =_param_docs
)   

def confusion_matrix_(
    clf:F, 
    X:NDArray, 
    y:ArrayLike,
    *, 
    cv:int =7, 
    plot_conf_max:bool =False, 
    crossvalp_kws:dict=dict(), 
    **conf_mx_kws 
)->object: 

    #create a object to hold attributes 
    obj = type('Metrics', (), dict())
    obj.y_pred =cross_val_predict(clf, X, y, cv=cv, **crossvalp_kws )
    
    if obj.y_pred.ndim ==1 : 
        obj.y_pred.reshape(-1, 1)
    obj.conf_mx = confusion_matrix(y, obj.y_pred, **conf_mx_kws)

    # statement to plot confusion matrix errors rather than values 
    row_sums = obj.conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = obj.conf_mx / row_sums 
    # now let fill the diagonal with zeros to keep only the errors
    # and let's plot the results 
    np.fill_diagonal(norm_conf_mx, 0)
    obj.norm_conf_mx= norm_conf_mx

    fp =0
    if plot_conf_max =='map': 
        confmax = obj.conf_mx
        fp=1
    if plot_conf_max =='error':
        confmax= norm_conf_mx
        fp =1
    if fp: 
        import matplotlib.pyplot as plt 
        plt.matshow(confmax, cmap=plt.cm.gray)
        plt.show ()
        
    return obj  
  
confusion_matrix_.__doc__ ="""\
Evaluate the preformance of the model or classifier by counting 
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
{params.core.clf}
{params.core.X}
{params.core.y}
{params.core.cv}
{params.metric.classe_}
{params.metric.method}
{params.metric.y_tradeoff}

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
""".format(
    params =_param_docs
)
# Get all param values and set attributes 
# func_sig = inspect.signature(confusion_matrix_)
# PARAMS_VALUES = {k: v.default
#     for k, v in func_sig.parameters.items()
#     if v.default is not  inspect.Parameter.empty
#     }
# # add positional params 
# for pname, pval in zip( ['X', 'y', 'clf'], [X, y, clf]): 
#     PARAMS_VALUES[pname]=pval 
    
# PARAMS_VALUES2 = {k: v
#     for k, v in func_sig.parameters.items()
#     if (v.default is inspect.Parameter.empty and k !='self')
#     }
# parameters = [p.name for p in func_sig.parameters.values()
       # if p.name != 'self' and p.kind != p.VAR_KEYWORD] 
# mObj = Metrics() #confusion_matrix_ 
# for key in PARAMS_VALUES.keys(): 
#     setattr(mObj , key, PARAMS_VALUES[key] )