# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
#   Created on Tue Oct  5 15:09:46 2021

"""
Metrics are measures of quantitative assessment commonly used for 
estimating, comparing, and tracking performance or production. Generally,
a group of metrics will typically be used to build a dashboard that
management or analysts review on a regular basis to maintain performance
assessments, opinions, and business strategies.
"""
from __future__ import annotations 
import copy
import warnings  
import numpy as np 
from sklearn import metrics 

from ._docstring import ( 
    DocstringComponents,
    _core_docs,
    )
from ._watexlog import watexlog
from ._typing import ( 
    List, 
    Optional, 
    ArrayLike , 
    NDArray,
    F
    )
from .exceptions import LearningError 
from .exlib.sklearn import ( 
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_curve, 
    roc_auc_score,
    cross_val_predict, 
    accuracy_score, 
    confusion_matrix as cfsmx ,
    )
from .utils.validator import get_estimator_name 

_logger = watexlog().get_watex_logger(__name__)

__all__=['precision_recall_tradeoff',
         'ROC_curve',
         'confusion_matrix', 
         "get_metrics", 
         "get_eval_scores"
         ]

#----add metrics docs 
_metrics_params =dict (
    label="""
label: float, int 
    Specific class to evaluate the tradeoff of precision 
    and recall. If `y` is already a binary classifer (0 & 1), `label` 
    does need to specify.     
    """, 
    method="""
method: str
    Method to get scores from each instance in the trainset. 
    Could be a ``decison_funcion`` or ``predict_proba``. When using the  
    scikit-Learn classifier, it generally has one of the method. 
    Default is ``decision_function``.   
    """, 
    tradeoff="""
tradeoff: float
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
    """
    Get the list of  available metrics. 
    
    Metrics are measures of quantitative assessment commonly used for 
    assessing, comparing, and tracking performance or production. Generally,
    a group of metrics will typically be used to build a dashboard that
    management or analysts review on a regular basis to maintain performance
    assessments, opinions, and business strategies.
    """
    return tuple(metrics.SCORERS.keys())

def get_eval_scores (
    model, 
    Xt, 
    yt, 
    *, 
    multi_class="raise", 
    average="binary", 
    normalize=True, 
    sample_weight=None,
    verbose = False, 
    **scorer_kws, 
    ): 
    ypred = model.predict(Xt) 
    acc_scores = accuracy_score(yt, ypred, normalize=normalize, 
                                sample_weight= sample_weight) 
    rec_scores = recall_score(
        yt, ypred, average =average, sample_weight = sample_weight, 
        **scorer_kws)
    prec_scores = precision_score(
        yt, ypred, average =average,sample_weight = sample_weight, 
        **scorer_kws)
    try:
        #compute y_score when predict_proba is available 
        # or  when  probability=True
        ypred = model.predict_proba(Xt) if multi_class !='raise'\
            else model.predict(Xt) 
    except: rocauc_scores=None 
    else :
        rocauc_scores= roc_auc_score (
            yt, ypred, average=average, multi_class=multi_class, 
            sample_weight = sample_weight, **scorer_kws)

    scores= dict ( 
        accuracy = acc_scores , recall = rec_scores, 
        precision= prec_scores, auc = rocauc_scores 
        )
    if verbose: 
        mname=get_estimator_name(model)
        print(f"{mname}:\n")
        print("accuracy -score = ", acc_scores)
        print("recall -score = ", rec_scores)
        print("precision -score = ", prec_scores)
        print("ROC AUC-score = ", rocauc_scores)
    return scores 

get_eval_scores.__doc__ ="""\
Compute the `accuracy`,  `precision`, `recall` and `AUC` scores.

Parameters 
------------
{params.core.model}
{params.core.Xt} 
{params.core.yt}

average : {{'micro', 'macro', 'samples', 'weighted', 'binary'}} or None, \
        default='binary'
    This parameter is required for multiclass/multilabel targets.
    If ``None``, the scores for each class are returned. Otherwise, this
    determines the type of averaging performed on the data:

    ``'binary'``:
        Only report results for the class specified by ``pos_label``.
        This is applicable only if targets (``y_{{true,pred}}``) are binary.
    ``'micro'``:
        Calculate metrics globally by counting the total true positives,
        false negatives and false positives.
    ``'macro'``:
        Calculate metrics for each label, and find their unweighted
        mean.  This does not take label imbalance into account.
    ``'weighted'``:
        Calculate metrics for each label, and find their average weighted
        by support (the number of true instances for each label). This
        alters 'macro' to account for label imbalance; it can result in an
        F-score that is not between precision and recall. Weighted recall
        is equal to accuracy.
    ``'samples'``:
        Calculate metrics for each instance, and find their average (only
        meaningful for multilabel classification where this differs from
        :func:`accuracy_score`).
        Will be ignored when ``y_true`` is binary.
        Note: multiclass ROC AUC currently only handles the 'macro' and
        'weighted' averages.
        
multi_class : {{'raise', 'ovr', 'ovo'}}, default='raise'
    Only used for multiclass targets. Determines the type of configuration
    to use. The default value raises an error, so either
    ``'ovr'`` or ``'ovo'`` must be passed explicitly.

    ``'ovr'``:
        Stands for One-vs-rest. Computes the AUC of each class
        against the rest [1]_ [2]_. This
        treats the multiclass case in the same way as the multilabel case.
        Sensitive to class imbalance even when ``average == 'macro'``,
        because class imbalance affects the composition of each of the
        'rest' groupings.
    ``'ovo'``:
        Stands for One-vs-one. Computes the average AUC of all
        possible pairwise combinations of classes [3]_.
        Insensitive to class imbalance when
        ``average == 'macro'``.
        
normalize : bool, default=True
    If ``False``, return the number of correctly classified samples.
    Otherwise, return the fraction of correctly classified samples.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.
    
{params.core.verbose}

scorer_kws: dict, 
    Additional keyword arguments passed to the scorer metrics: 
    :func:`~sklearn.metrics.accuracy_score`, 
    :func:`~sklearn.metrics.precision_score`, 
    :func:`~sklearn.metrics.recall_score`, 
    :func:`~sklearn.metrics.roc_auc_score`
    
Returns 
--------
scores: dict , 
    A dictionnary to retain all the scores from metrics evaluation such as 
    - accuracy , 
    - recall 
    - precision 
    - ROC AUC ( Receiving Operating Characteric Area Under the Curve)
    
Notes 
-------
Note that if `yt` is given, it computes `y_score` known as array-like of 
shape (n_samples,) or (n_samples, n_classes)Target scores following the 
scheme below: 

* In the binary case, it corresponds to an array of shape
  `(n_samples,)`. Both probability estimates and non-thresholded
  decision values can be provided. The probability estimates correspond
  to the **probability of the class with the greater label**,
  i.e. `estimator.classes_[1]` and thus
  `estimator.predict_proba(X, y)[:, 1]`. The decision values
  corresponds to the output of `estimator.decision_function(X, y)`.
  See more information in the :ref:`User guide <roc_auc_binary>`;
* In the multiclass case, it corresponds to an array of shape
  `(n_samples, n_classes)` of probability estimates provided by the
  `predict_proba` method. The probability estimates **must**
  sum to 1 across the possible classes. In addition, the order of the
  class scores must correspond to the order of ``labels``,
  if provided, or else to the numerical or lexicographical order of
  the labels in ``y_true``. See more information in the
  :ref:`User guide <roc_auc_multiclass>`;
* In the multilabel case, it corresponds to an array of shape
  `(n_samples, n_classes)`. Probability estimates are provided by the
  `predict_proba` method and the non-thresholded decision values by
  the `decision_function` method. The probability estimates correspond
  to the **probability of the class with the greater label for each
  output** of the classifier. See more information in the
  :ref:`User guide <roc_auc_multilabel>`.
      
References
----------

.. [1] Provost, F., Domingos, P. (2000). Well-trained PETs: Improving
       probability estimation trees (Section 6.2), CeDER Working Paper
       #IS-00-04, Stern School of Business, New York University.

.. [2] `Fawcett, T. (2006). An introduction to ROC analysis. Pattern
        Recognition Letters, 27(8), 861-874.
        <https://www.sciencedirect.com/science/article/pii/S016786550500303X>`_
         
.. [3] `Hand, D.J., Till, R.J. (2001). A Simple Generalisation of the Area
        Under the ROC Curve for Multiple Class Classification Problems.
        Machine Learning, 45(2), 171-186.
        <http://link.springer.com/article/10.1023/A:1010920819831>`_
See Also
--------
average_precision_score : Area under the precision-recall curve.
roc_curve : Compute Receiver operating characteristic (ROC) curve.
RocCurveDisplay.from_estimator : Plot Receiver Operating Characteristic
    (ROC) curve given an estimator and some data.
RocCurveDisplay.from_predictions : Plot Receiver Operating Characteristic
    (ROC) curve given the true and predicted values.
    
Examples
--------
Binary case:

>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.metrics import roc_auc_score
>>> X, y = load_breast_cancer(return_X_y=True)
>>> clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
>>> roc_auc_score(y, clf.predict_proba(X)[:, 1])
0.99...
>>> roc_auc_score(y, clf.decision_function(X))
0.99...

Multiclass case:

>>> from sklearn.datasets import load_iris
>>> X, y = load_iris(return_X_y=True)
>>> clf = LogisticRegression(solver="liblinear").fit(X, y)
>>> roc_auc_score(y, clf.predict_proba(X), multi_class='ovr')
0.99...

Multilabel case:

>>> import numpy as np
>>> from sklearn.datasets import make_multilabel_classification
>>> from sklearn.multioutput import MultiOutputClassifier
>>> X, y = make_multilabel_classification(random_state=0)
>>> clf = MultiOutputClassifier(clf).fit(X, y)
>>> # get a list of n_output containing probability arrays of shape
>>> # (n_samples, n_classes)
>>> y_pred = clf.predict_proba(X)
>>> # extract the positive columns for each output
>>> y_pred = np.transpose([pred[:, 1] for pred in y_pred])
>>> roc_auc_score(y, y_pred, average=None)
array([0.82..., 0.86..., 0.94..., 0.85... , 0.94...])
>>> from sklearn.linear_model import RidgeClassifierCV
>>> clf = RidgeClassifierCV().fit(X, y)
>>> roc_auc_score(y, clf.decision_function(X), average=None)
array([0.81..., 0.84... , 0.93..., 0.87..., 0.94...])
""".format(params =_param_docs
)
    
def _assert_metrics_args(y, label): 
    """ Assert metrics argument 
    
    :param y: array-like, 
        label for prediction. `y` is binary label by default. 
        If `y` is composed of multilabel, specify  the `classe_` 
        argumentto binarize the label(`True` ot `False`). ``True``  
        for `classe_`and ``False`` otherwise. 
    :param label:float, int 
        Specific class to evaluate the tradeoff of precision 
        and recall. If `y` is already a binary classifer, `classe_` 
        does need to specify.     
    """
    # check y if value to plot is binarized ie.True of false 
    msg = ("Precision-recall metrics are fundamentally metrics for"
           " binary classification. ")
    y_unik = np.unique(y)
    if len(y_unik )!=2 and label is None: 
        warnings.warn( msg + f"Classes values of 'y' is '{len(y_unik )}', "
                      "while expecting '2'. Can not set the tradeoff for "
                      " non-binarized classifier ",  UserWarning
                       )
        _logger.warning('Expect a binary classifier(2), but %s are given'
                              %len(y_unik ))
        raise LearningError(f'Expect a binary labels but {len(y_unik )!r}'
                         f' {"are" if len(y_unik )>1 else "is"} given')
        
    if label is not None: 
        try : 
            label= int(label)
        except ValueError: 
            raise ValueError('Need integer value; Could not convert to Float.')
        except TypeError: 
            raise TypeError(f'Could not convert {type(label).__name__!r}') 
    
    if label not in y: 
        raise ValueError("Value '{}' must be a label of a binary target"
                         .format(label))
  
def precision_recall_tradeoff(
    clf:F, 
    X:NDArray,
    y:ArrayLike,
    *,
    cv:int =7,
    label: str | Optional[List[str]]=None,
    method:Optional[str] =None,
    cvp_kws: Optional[dict]  =None,
    tradeoff: Optional[float] =None,
    **prt_kws
)-> object:
    
    mc= copy.deepcopy(method)
    method = method or "decision_function"
    method =str(method).lower().strip() 
    if method not in ('decision_function', 'predict_proba'): 
        raise ValueError (f"Invalid method {mc!r}.Expect 'decision_function'"
                          " or 'predict_proba'.")
        
    #create a object to hold attributes 
    obj = type('Metrics', (), {})
    
    _assert_metrics_args(y, label)
    y=(y==label) # set boolean 
    
    if cvp_kws is None: 
        cvp_kws = dict()
        
    obj.y_scores = cross_val_predict(
        clf,
         X,
         y,
         cv =cv,
        method= method,
        **cvp_kws 
    )
    y_scores = cross_val_predict(
        clf,
        X,
        y, 
        cv =cv,
        **cvp_kws 
        )
    
    obj.confusion_matrix =cfsmx(y, y_scores )
    
    obj.f1_score = f1_score(y,y_scores)
    obj.precision_score = precision_score(y, y_scores)
    obj.recall_score= recall_score(y, y_scores)
        
    if method =='predict_proba': 
        # if classifier has a `predict_proba` method like 
        # `Random_forest` then use the positive class
        # probablities as score  score = proba of positive 
        # class 
        obj.y_scores =obj.y_scores [:, 1] 
        
    if tradeoff is not None:
        try : 
            float(tradeoff)
        except ValueError: 
            raise ValueError(f"Could not convert {tradeoff!r} to float.")
        except TypeError: 
            raise TypeError(f'Invalid type `{type(tradeoff)}`')
            
        y_score_pred = (obj.y_scores > tradeoff) 
        obj.precision_score = precision_score(y, y_score_pred)
        obj.recall_score = recall_score(y, y_score_pred)
        
    obj.precisions, obj.recalls, obj.thresholds =\
        precision_recall_curve(y, obj.y_scores,**prt_kws)
        
    obj.y =y
    
    return obj

precision_recall_tradeoff.__doc__ ="""\
Precision-recall Tradeoff computes a score based on the decision function. 

Is assign the instance to the positive class if that score on 
the left is greater than the `threshold` else it assigns to negative 
class. 

Parameters
----------
{params.core.clf}
{params.core.X}
{params.core.y}
{params.core.cv}

label: float, int 
    Specific class to evaluate the tradeoff of precision 
    and recall. If `y` is already a binary classifer, `classe_` 
    does need to specify. 
method: str
    Method to get scores from each instance in the trainset. 
    Ciuld be ``decison_funcion`` or ``predict_proba`` so 
    Scikit-Learn classifier generally have one of the method. 
    Default is ``decision_function``.
tradeoff: float, optional,
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
    The metric object is composed of the following attributes:
        
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

It's very similar to precision/recall , but instead of plotting 
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
{params.metric.label}
{params.metric.method}
{params.metric.tradeoff}

roc_kws: dict 
    roc_curve additional keywords arguments
    
See also
---------
watex.view.mlplot.MLPlot.precisionRecallTradeoff:  
    plot consistency precision recall curve. 
    
    
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

def confusion_matrix(
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
    obj.conf_mx = cfsmx(y, obj.y_pred, **conf_mx_kws)

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
  
confusion_matrix.__doc__ ="""\
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
{params.metric.label}
{params.metric.method}
{params.metric.tradeoff}

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
>>> from watex.utils.metrics import Metrics 
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