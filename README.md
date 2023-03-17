<img src="docs/_static/logo_wide_rev.svg"><br>

-----------------------------------------------------

# *WATex*: machine learning research in water exploration

### *Life is much better with potable water*

 [![Documentation Status](https://readthedocs.org/projects/watex/badge/?version=latest)](https://watex.readthedocs.io/en/latest/?badge=latest)
 ![GitHub](https://img.shields.io/github/license/WEgeophysics/watex?color=blue&label=Licence&logo=Github&logoColor=blue&style=flat-square)
 ![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/WEgeophysics/watex/ci.yaml?label=CI%20-%20Build%20&logo=github&logoColor=g)
[![Coverage Status](https://coveralls.io/repos/github/WEgeophysics/watex/badge.svg?branch=master)](https://coveralls.io/github/WEgeophysics/watex?branch=master)
 ![GitHub release (latest SemVer including pre-releases)](https://img.shields.io/github/v/release/WEgeophysics/watex?color=blue&include_prereleases&logo=python)
 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7553789.svg)](https://doi.org/10.5281/zenodo.7553789)
 ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/watex?color=orange&logo=pypi)
 [![PyPI version](https://badge.fury.io/py/watex.svg)](https://badge.fury.io/py/watex)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/watex.svg)](https://anaconda.org/conda-forge/watex)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/watex/badges/platforms.svg)](https://anaconda.org/conda-forge/watex)

##  Overview

*WATex* is a Python-based library mainly focused on the groundwater exploration (GWE). It brings novel approaches 
for reducing numerous losses during the hydro-geophysical exploration projects. It encompasses 
the Direct-current (DC) resistivity ( Electrical profiling (ERP) & vertical electrical sounding (VES)), 
short-periods electromagnetic (EM), geology and hydrogeology methods. From methodologies based on Machine Learning,  
it allows to: 
- auto-detect the right position to locate the drilling operations to minimize the rate of unsuccessful drillings 
  and unsustainable boreholes;
- reduce the cost of permeability coefficient (k) data collection during the hydro-geophysical engineering projects,
- predict the water content in the well such as the groundwater flow rate, the level of water inrush, ...
- recover the EM loss signals in area with huge interferences noises ...
- etc.

## Documentation 

Visit the [library website](https://watex.readthedocs.io/en/latest/) for more resources. You can also quick browse the software [API reference](https://watex.readthedocs.io/en/latest/api_references.html)
and flip through the [examples page](https://watex.readthedocs.io/en/latest/glr_examples/index.html) to see some of expected results. Furthermore, the 
[step-by-step guide](https://watex.readthedocs.io/en/latest/glr_examples/applications/index.html#applications-step-by-step-guide) is elaborated for real-world 
engineering problems such as computing DC parameters and predicting the k-parameter... 

## Licence 

*WATex* is under [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause) License. 

## Installation 

The system requires preferably Python 3.9+. 

* from *pip*

[_WATex_](https://pypi.org/project/watex/0.1.7/) can be installed from [PyPI](https://pypi.org/) platform distribution as: 
```bash 
pip install watex
```

* from *conda* 

The [installation](https://anaconda.org/conda-forge/watex)  from [conda-forge](https://conda-forge.org/) distribution channel can be achieved with :

```bash 
conda install -c conda-forge watex
``` 
To get the latest development of the code, it is recommended to install it from source using: 

```bash
git clone https://github.com/WEgeophysics/watex.git 
```
Furthermore, for step-by-step guide about the installation and how to manage the 
dependencies, visit our [installation guide](https://watex.readthedocs.io/en/latest/installation.html) page.

## Some demos 

### 1. Drilling location auto-detection

For this example, we randomly generate 50 stations of synthetic ERP resistivity data with ``minimum`` and ``maximum ``
resistivity values equal to  ``1e1`` and ``1e4`` ohm.m  respectively as:

```python 
import watex as wx 
data = wx.make_erp (n_stations=50, max_rho=1e4, min_rho=10., as_frame =True, seed =42 ) 
``` 

* Naive auto-detection (NAD)

The NAD automatically proposes a suitable location with NO restrictions (constraints) observed in the survey site
during the GWE. We may understand by ``suitable``, a location expecting to give a flow rate greater 
than 1m3/hr at least. 

```python
robj=wx.ResistivityProfiling (auto=True ).fit(data ) 
robj.sves_ 
Out[1]: 'S025'
```
The suitable drilling location is proposed at station ``S25`` (stored in the attribute ``sves_``). 

* Auto-detection with constraints (ADC)

The constraints refer to the restrictions observed in the survey area during the DWSC. This is common
in real-world exploration. For instance, a station close to a heritage site should be discarded 
since no drilling operations are authorized at that place. When many restrictions 
are enumerated in the survey site, they must be listed in a dictionary with a reason and passed to the parameter 
``constraints`` so these stations should be ignored during the automatic detection. Here is an example of constraints
application to our example.

```python 
restrictions = {
    'S10': 'Household waste site, avoid contamination',
    'S27': 'Municipality site, no authorization to make a drill',
    'S29': 'Heritage site, drilling prohibited',
    'S42': 'Anthropic polluted place, avoid contamination within a few years',
    'S46': 'Marsh zone, borehole will dry up during the dry season'
 }
robj=wx.ResistivityProfiling (constraints= restrictions, auto=True ).fit(data ) 
robj.sves_
Out[2]: 'S033'
```
Notice, the station ``S25`` is no longer considered as the `suitable` location and henceforth, propose ``S33`` as the
priority for drilling operations. However, if the station is close to a restricted area, a warning should raise to 
inform the user to avoid taking a risk to perform a drilling location at that location.

Note that before the drilling operations commence, make sure to carry out the DC-sounding (VES) at that point. **_WATex_** computes 
another parameter called `ohmic-area` `` (ohmS)`` to detect the effectiveness of the existing fracture zone at that point. See more in 
the software [documentation](https://watex.readthedocs.io/en/latest/).
  
### 2. Predict permeability coefficient ``k`` from logging dataset using MXS approach
 
MXS stands for mixture learning strategy. It uses upstream unsupervised learning for 
``k`` -aquifer similarity label prediction and the supervising learning for 
final ``k``-value prediction. For our toy example, we use two boreholes data 
stored in the software and merge them to compose a unique dataset. In addition, we dropped the 
``remark`` observation which is subjective data not useful for ``k`` prediction as:

```python
import watex as wx
h= wx.fetch_data("hlogs", key='*', drop_observations =True ) # returns log data object.
h.feature_names
Out[3]: Index(['hole_id', 'depth_top', 'depth_bottom', 'strata_name', 'rock_name',
           'layer_thickness', 'resistivity', 'gamma_gamma', 'natural_gamma', 'sp',
           'short_distance_gamma', 'well_diameter'],
          dtype='object')
hdata = h.frame 
```
``k`` is collected as continue values (m/darcies) and should be categorized for the 
naive group of aquifer prediction (NGA). The latter is used to predict 
upstream the  MXS target ``ymxs``.  Here, we used the default categorization 
provided by the software and we assume that in the area, there are at least ``2`` 
groups of the aquifer. The code is given as: 
```python 
mxs = wx.MXS (kname ='k', n_groups =2).fit(hdata) 
ymxs=mxs.predictNGA().makeyMXS(categorize_k=True, default_func=True)
mxs.yNGA_ [62:74]
Out[4]: array([1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
ymxs[62:74]
Out[5]: array([ 0,  0,  0,  0, 12, 12, 12, 12, 12, 12, 12, 12])
```
To understand the transformation from NGA to MXS target (``ymxs``), please, have a look 
of the following [paper](http://dx.doi.org/10.2139/ssrn.4326365).
Once the MXS target is predicted, we call the ``make_naive_pipe`` function to 
impute, scale, and transform the predictor ``X`` at once into a compressed sparse 
matrix ready for final prediction using the [support vector machines](https://ieeexplore.ieee.org/document/708428) and 
[random forest](https://www.ibm.com/topics/random-forest) as examples. Here we go: 
```python 
X= hdata [h.feature_names]
Xtransf = wx.make_naive_pipe (X, transform=True) 
Xtransf 
Out[6]: 
<218x46 sparse matrix of type '<class 'numpy.float64'>'
	with 2616 stored elements in Compressed Sparse Row format> 
Xtrain, Xtest, ytrain, ytest = wx.sklearn.train_test_split (Xtransf, ymxs ) 
ypred_k_svc= wx.sklearn.SVC().fit(Xtrain, ytrain).predict(Xtest)
ypred_k_rf = wx.sklearn.RandomForestClassifier ().fit(Xtrain, ytrain).predict(Xtest)
```
We can now check the ``k`` prediction scores using ``accuracy_score`` function as: 
```python 
wx.sklearn.accuracy_score (ytest, ypred_k_svc)
Out[7]: 0.9272727272727272
wx.sklearn.accuracy_score (ytest, ypred_k_rf)
Out[8]: 0.9636363636363636
```
As we can see, the results of ``k`` prediction are quite satisfactory for our 
toy example using only two boreholes data. Note that things can become more 
interesting when using many boreholes data. For more in 
depth, visit our [examples page](https://watex.readthedocs.io/en/latest/glr_examples/index.html). 

### 3. EM tensors recovering and analyses

For a basic illustration, we fetch 20 audio-frequency magnetotelluric (AMT) data 
stored as EDI objects collected in a `huayuan` area (Hunan province, China) with 
multiple interferences noised as: 

```python 
import watex as wx
e= wx.fetch_data ('huayuan', samples =20 , key='noised') # returns an EM -objets 
edi_data = e.data # get the array  of EDI objects  
``` 
Before EM data restoration, we can analyse the quality control (QC) of the data and 
show the confidence interval that makes us confident about the data at each station. 
By default the confidence test uses the errors in the resistivity tensor. Let's getting 
started: 
```python 
po= wx.EMProcessing ().fit(edi_data)   # make a EM processing object 
r= po.qc (tol =0.2 , return_ratio = True ) # consider good data from 80% significance.  
r
Out[9]: 0.95
``` 
We can then visualizate the confidence interval at the 20 AMT stations as: 
```python 
wx.plot_confidence_in(edi_data) 

``` 
Alternatively, we can use the ``qc`` function (more consistent) to get the valid data and 
the interpolated frequencies. For instance, we want to known the number of frequencies dropped 
during the control analysis. Just do it: 
```python 
QCo= wx.qc (edi_data , tol=.2,  return_qco =True )  # returns the quality control object
len(e.emo.freqs_)   # number of frequency in noised data   
Out[10]: 56
len(QCo.freqs_)     # number of frequency in valid data after QC  
Out[11]: 53
QCo.invalid_freqs_  # get the useless frequencies based on tol param so we can drop them into the EM data 
Out[12]: array([8.19200e+04, 4.85294e+01, 5.62500e+00]) #  81920.0, 48.53 and 5.625 Hz 
```
The ``plot_confidence_in`` function allows to assert whether tensor values can be recovered 
for these three frequencies at each station. Note that the threshold for the EM data 
to be restored is set to ``50%``. Below this value, data is unrecoverable. 
Furthermore, if our QC rate ``r=95%`` is not to be yet satisfactory in our AMT data, we can 
process to the impedance tensor ``Z`` restoration as:  
```python 
Z=po.zrestore() # returns 3D tensors (Nfrequency, 2, 2), 2x2 for XX, XY, YX and YY components. 
```
Now, let's evaluate the new QC ratio to verify the recovering efficaciousness such as: 
```python 
r, =wx.qc (Z)
r
Out[13]: 1.0
``` 
Great! As we can see, the tensor is restored at each station with ``100%`` ratio and we notice 
that the confidence line is above 95% in alongside the 20 investigation sites and 
compare to the previous plot ( ``rate =75%``). The snippet below can allow to visualize 
this improvement with the confidence interval as: 
```python 
wx.plot_confidence_in(Z)  
```
Besides, user can flip through the following links for more examples about [EM tensor restoring](https://watex.readthedocs.io/en/latest/glr_examples/applications/plot_tensor_restoring.html#sphx-glr-glr-examples-applications-plot-tensor-restoring-py),  
the [sknewness](https://watex.readthedocs.io/en/latest/glr_examples/methods/plot_phase_tensors.html#sphx-glr-glr-examples-methods-plot-phase-tensors-py) analysis plots, 
the [strike](https://watex.readthedocs.io/en/latest/glr_examples/utils/plot_strike.html#sphx-glr-glr-examples-utils-plot-strike-py) plot, 
the [filtering](https://watex.readthedocs.io/en/latest/methods.html#filtering-tensors-ama-flma-tma) data, and else...

## Citations

If the [software](https://doi.org/10.1016/j.softx.2023.101367) seemed useful to you in any published work, we will appreciate to cite the paper below:

> *Kouadio, K.L., Liu, J., Liu, R., 2023. watex: machine learning research in water exploration. SoftwareX . 101367(2023). https://doi.org/10.1016/j.softx.2023.101367*

In most situations where *WATex* is cited, a citation to [scikit-learn](https://scikit-learn.org/stable/about.html#citing-scikit-learn) would also be appropriate.

See also some [case history](https://watex.readthedocs.io/en/latest/citing.html) papers using *WATex*. 

## Contributions 

1. Department of Geophysics, School of Geosciences & Info-physics, [Central South University](https://en.csu.edu.cn/), China.
2. Hunan Key Laboratory of Nonferrous Resources and Geological Hazards Exploration Changsha, Hunan, China
3. Laboratoire de Geologie Ressources Minerales et Energetiques, UFR des Sciences de la Terre et des Ressources Minières, [Université Félix Houphouët-Boigny]( https://www.univ-fhb.edu.ci/index.php/ufr-strm/), Cote d'Ivoire.

Developer: [_L. Kouadio_](https://wegeophysics.github.io/) <<etanoyau@gmail.com>>

