<img src="docs/_static/logo_wide_rev.svg"><br>

-----------------------------------------------------

# *WATex*: machine learning research in hydro-geophysics

### *Life is much better with potable water*

 [![Documentation Status](https://readthedocs.org/projects/watex/badge/?version=latest)](https://watex.readthedocs.io/en/latest/?badge=latest)
 ![GitHub](https://img.shields.io/github/license/WEgeophysics/watex?color=blue&label=Licence&style=flat-square)
  ![Libraries.io dependency status for GitHub repo](https://img.shields.io/librariesio/github/WEgeophysics/watex?logo=appveyor) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7553789.svg)](https://doi.org/10.5281/zenodo.7553789)
  ![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/WEgeophysics/watex?logo=python)


##  Mission and Goals

**_WATex_** is a Python library mainly focused on the groundwater exploration (GWE) field. It brings novel approaches 
    for reducing numerous losses during the hydro-geophysical exploration projects and encompasses 
    the DC-resistivity ( Electrical profiling (ERP) & vertical electrical sounding (VES)), short periods EM, geology 
    and hydrogeology methods. From new methods based on Machine Learning,  it allows to: 
   - auto-detect the right position to locate the drilling operations, 
   - reduce the cost of permeability coefficient (k) data collection during the hydro-geophysical engineering projects,
   - predict the water content in the well such as the groundwater flow rate, the level of water inrush, ...
   - etc.


## Documentation 

Visit the [library website](https://watex.readthedocs.io/en/latest/) for more resources. You can also quick browse the software [API reference](https://watex.readthedocs.io/en/latest/api_references.html)
and flip through the [examples page](https://watex.readthedocs.io/en/latest/glr_examples/index.html) to see some of expected results. Furthermore, the 
[step-by-step guide](https://watex.readthedocs.io/en/latest/glr_examples/applications/index.html#applications-step-by-step-guide) is elaborated for real-world 
engineering problems such as computing DC parameters and predicting the k-parameter... 

## Demo of the drilling location auto-detection 

This is a naive example (no constraints) to automatically propose the suitable location to make 
the drilling operations during the GWE. We may understand by ``suitable``, a location 
expecting to give a flow rate greater than > 3m3/hr at least. 

We randomly generate 50 stations with DC-resistivity ```min/min =1e1/1e4`` ohm.m:

```python

import watex as wx 
data = wx.make_erp (n_stations=50, max_rho=1e4, min_rho=10., as_frame =True, seed =42 ) 
robj=wx.ResistivityProfiling (auto=True ).fit(data ) 
robj.sves_ 
Out[1]: 'S025'

```
The algorithm proposes the best drilling location be made at station ``S25`` (stored in the attribute ``sves_``). Note that 
before the drilling operations commence, make sure to carry out the DC-sounding (VES) at that point. **_WATex_** computes 
another parameter called `ohmic-area` `` (ohmS)`` to detect the effectiveness of the existing fracture zone at that point. See more in 
the software [documentation](https://watex.readthedocs.io/en/latest/).

  
## Licence 

**_WATex_** is under [3-Clause BSD](https://opensource.org/licenses/BSD-3-Clause) License.

## Installation 

**_WATex_** is not available in any distribution platforms yet ( [pyPI](https://pypi.org/)  and [conda-forge](https://conda-forge.org/) ). 
However, your can install the package from source: 

```bash
git clone https://github.com/WEgeophysics/watex.git 
```
or simply visit the [installation guide](https://watex.readthedocs.io/en/latest/installation.html) page.

Installation via `pip`( [pyPI](https://pypi.org/) ) and `conda` ( [conda-forge](https://conda-forge.org/) ) is coming soon ... 

## System requirement

* Python 3.9+ 


## Citations
)
If you use the software in any published work, I will much appreciate to cite the paper or the [DOI](https://doi.org/10.5281/zenodo.7553789) below:

> *Kouadio, K. L., Kouame, L. N., Drissa, C., Mi, B., Kouamelan, K. S., Gnoleba, S. P. D., et al. (2022). Groundwater Flow Rate Prediction from Geo‐Electrical Features using Support Vector Machines. Water Resources Research, (May 2022). https://doi.org/10.1029/2021wr031623*

In most situations where **_WATex_** is cited, a citation to [scikit-learn](https://scikit-learn.org/stable/about.html#citing-scikit-learn) would also be appropriate.

## Contributions 

1. Department of Geophysics, School of Geosciences & Info-physics, [Central South University](https://en.csu.edu.cn/), China.
2. Hunan Key Laboratory of Nonferrous Resources and Geological Hazards Exploration Changsha, Hunan, China
3. Laboratoire de Geologie Ressources Minerales et Energetiques, UFR des Sciences de la Terre et des Ressources Minières, [Université Félix Houphouët-Boigny]( https://www.univ-fhb.edu.ci/index.php/ufr-strm/), Cote d'Ivoire.

Developer: [_L. Kouadio_](https://wegeophysics.github.io/) <<etanoyau@gmail.com>>



