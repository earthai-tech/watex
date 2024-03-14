<img src="docs/_static/logo_wide_rev.svg"><br>

-----------------------------------------------------

# *WATex*: machine learning research in water exploration

### *Life is much better with potable water*

 [![Documentation Status](https://readthedocs.org/projects/watex/badge/?version=latest)](https://watex.readthedocs.io/en/latest/?badge=latest)
 ![GitHub](https://img.shields.io/github/license/WEgeophysics/watex?color=blue&label=Licence&logo=Github&logoColor=blue&style=flat-square)
 ![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/WEgeophysics/watex/ci.yaml?label=CI%20-%20Build%20&logo=github&logoColor=g)
[![Coverage Status](https://coveralls.io/repos/github/WEgeophysics/watex/badge.svg?branch=master)](https://coveralls.io/github/WEgeophysics/watex?branch=master)
 ![GitHub release (latest SemVer including pre-releases)](https://img.shields.io/github/v/release/WEgeophysics/watex?color=blue&include_prereleases&logo=python)
 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7744732.svg)](https://doi.org/10.5281/zenodo.7744732)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/watex?logo=pypi)
 [![PyPI version](https://badge.fury.io/py/watex.svg)](https://badge.fury.io/py/watex)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/watex.svg)](https://anaconda.org/conda-forge/watex)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/watex/badges/platforms.svg)](https://anaconda.org/conda-forge/watex)


## Overview

*WATex* is a Python-based library primarily designed for Groundwater Exploration (GWE). It introduces innovative strategies aimed at minimizing losses encountered during hydro-geophysical exploration projects. Integrating methods from Direct-current (DC) resistivity—including Electrical Profiling (ERP) and Vertical Electrical Sounding (VES)—alongside short-period electromagnetic (EM), geology, and hydrogeology, *WATex* leverages Machine Learning techniques to enhance exploration outcomes. Key features include:
- Automating the identification of optimal drilling locations to reduce the incidence of unsuccessful drillings and unsustainable boreholes.
- Predicting well water content, including groundwater flow rates and water inrush levels.
- Restoring EM signal integrity in areas plagued by significant interference noise.
- And more.

## Documentation

For comprehensive information and additional resources, visit the [WATex library website](https://watex.readthedocs.io/en/latest/). To quickly navigate through the software's API reference, access the [API reference page](https://watex.readthedocs.io/en/latest/api_references.html). Explore the [examples section](https://watex.readthedocs.io/en/latest/glr_examples/index.html) for a preview of potential results. Additionally, a detailed [step-by-step guide](https://watex.readthedocs.io/en/latest/glr_examples/applications/index.html#applications-step-by-step-guide) is provided to tackle real-world engineering challenges, such as computing DC parameters and predicting the k-parameter.

## License

*WATex* is distributed under the [BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause).

## Installation

*WATex* is best supported on Python 3.9 or later.

### From *pip*

Install *WATex* directly from the Python Package Index (PyPI) with the following command:

```bash
pip install watex
```
### From *conda*

For users who prefer the conda ecosystem, *WATex* can be installed from the conda-forge distribution channel:

```bash
conda install -c conda-forge watex
```

### From Source

To access the most current development version of the code, installation from the source is recommended. Use the following commands to clone the repository and install:
```bash
git clone https://github.com/WEgeophysics/watex.git
```

### Additional Information

For a comprehensive installation guide, including how to manage dependencies effectively, 
please refer to our [Installation Guide](https://watex.readthedocs.io/en/latest/installation.html).


## Some Demos

### 1. Drilling Location Auto-detection

In this demonstration, we showcase the process of automatically detecting optimal locations 
for drilling by generating 50 stations of synthetic ERP resistivity data. The data is characterized 
by minimum and maximum resistivity values set at `10 ohm.m` and `10,000 ohm.m`, respectively:

```python
import watex as wx
data = wx.make_erp(n_stations=50, max_rho=1e4, min_rho=10., as_frame=True, seed=42)
```

#### Naive Auto-detection (NAD)

The NAD method identifies a suitable drilling location without considering any restrictions or 
constraints that might be present at the survey site during Groundwater Exploration (GWE). A location 
is deemed "suitable" if it is expected to yield a flow rate of at least 1m³/hr:

```python
from watex.methods import ResistivityProfiling
robj = ResistivityProfiling(auto=True).fit(data)
robj.sves_
Out[1]: 'S025'
```

The algorithm proposes station `S25` as the optimal drilling location, which is stored 
in the `sves_` attribute.

#### Auto-detection with Constraints (ADC)

In contrast, the ADC method accounts for constraints observed in the survey area during 
the Drilling Water Supply Chain (DWSC). These constraints are often encountered in real-world 
scenarios. For example, a station near a heritage site may be excluded due to drilling restrictions. 
When multiple constraints exist, they should be compiled into a dictionary detailing the reasons for 
each and passed to the `constraints` parameter. This ensures that these stations are disregarded during 
the automatic detection process:

```python
restrictions = {
    'S10': 'Household waste site, avoid contamination',

    'S27': 'Municipality site, no authorization for drilling',
    'S29': 'Heritage site, drilling prohibited',
    'S42': 'Anthropic polluted place, potential future contamination risk',
    'S46': 'Marsh zone, likely borehole dry-up during dry season'
}
robj = ResistivityProfiling(constraints=restrictions, auto=True).fit(data)
robj.sves_
# Output: 'S033'
```
This method revises the suitable drilling location to station `S33`, taking into account 
the specified constraints. Should a station be near a restricted area, the system raises a warning 
to advise against risking drilling operations at that location.

**Important Reminder:** Prior to initiating drilling operations, ensure a DC-sounding (VES) is conducted at the identified location. *WATex* calculates an additional parameter known as `ohmic-area` (ohmS) to evaluate the presence and effectiveness of fracture zones at that site. For further information, refer to the [WATex documentation](https://watex.readthedocs.io/en/latest/).


### 2. EM Tensor Recovery and Analysis

This demonstration outlines the process of recovering and analyzing electromagnetic (EM) tensor data. 
We begin by fetching 20 audio-frequency magnetotelluric (AMT) data points stored as EDI objects 
from the Huayuan area in Hunan Province, China, known for multiple interference noises:

```python
import watex as wx
e = wx.fetch_data('huayuan', samples=20, key='noised')  # Returns an EM object
edi_data = e.data  # Retrieve the array of EDI objects
```

Before restoring EM data, it's crucial to assess the data quality and evaluate the confidence 
intervals to ensure reliability at each station. Typically, this quality control (QC) analysis 
focuses on errors within the resistivity tensor:

```python
from watex.methods import EMAP
po = EMAP().fit(edi_data)  # Creates an EM Array Profiling processing object
r = po.qc(tol=0.2, return_ratio=True)  # Good data deemed from 80% significance level
r
Out[9]: 0.95
```

To visualize the confidence intervals at the 20 AMT stations:

```python
from watex.utils import plot_confidence_in
plot_confidence_in(edi_data)
```

For a more thorough quality control, we use the `qc` function to filter out invalid data and 
interpolate frequencies. To determine the number of frequencies dropped during this analysis:

```python
from watex.utils import qc
QCo = qc(edi_data, tol=.2, return_qco=True)  # Returns the quality control object
len(e.emo.freqs_)  # Original number of frequencies in noisy data
Out[10]: 56
len(QCo.freqs_)  # Number of frequencies in valid data after QC
Out[11]: 53
QCo.invalid_freqs_  # Frequencies discarded based on the tolerance parameter
Out[12]: array([81920.0, 48.53, 5.625])  # 81920.0, 48.53, and 5.625 Hz
```

The `plot_confidence_in` function is crucial for assessing whether tensor values for these 
frequencies are recoverable at each station. It's important to note that data is considered 
unrecoverable if the confidence level falls below 50%.

Should the initial QC rate of 95% not meet our standards, we can proceed to restore the 
impedance tensor `Z`:

```python
Z = po.zrestore()  # Returns 3D tensors for XX, XY, YX, and YY components
```

Evaluating the new QC ratio post-restoration confirms the effectiveness of our 
recovery efforts:

```python
r, = wx.qc(Z)
r
Out[13]: 1.0
```

As observed, the tensor restoration achieves a 100% success rate across all stations, 
significantly improving upon the initial analysis. To visualize this enhancement in 
confidence levels:

```python
plot_confidence_in(Z)
```

For further exploration on EM tensor restoration, phase tensor analysis, strike plotting, data filtering, and more, users are encouraged to visit the following links for detailed examples:
- [EM Tensor Restoring](https://watex.readthedocs.io/en/latest/glr_examples/applications/plot_tensor_restoring.html#sphx-glr-glr-examples-applications-plot-tensor-restoring-py)
- [Skewness Analysis Plots](https://watex.readthedocs.io/en/latest/glr_examples/methods/plot_phase_tensors.html#sphx-glr-glr-examples-methods-plot-phase-tensors-py)
- [Strike Plot](https://watex.readthedocs.io/en/latest/glr_examples/utils/plot_strike.html#sphx-glr-glr-examples-utils-plot-strike-py)
- [Filtering Data](https://watex.readthedocs.io/en/latest/methods.html#filtering-tensors-ama-flma-tma)


## Citations

Should you find the [WATex software](https://doi.org/10.1016/j.softx.2023.101367) beneficial 
for your research or any published work, we kindly ask you to cite the following article:

> Kouadio, K.L., Liu, J., Liu, R., 2023. watex: machine learning research in water exploration. SoftwareX, 101367(2023). [https://doi.org/10.1016/j.softx.2023.101367](https://doi.org/10.1016/j.softx.2023.101367)

In publications that mention *WATex*, acknowledging [scikit-learn](https://scikit-learn.org/stable/about.html#citing-scikit-learn) may also be relevant due to its integral role in the software's development.

For additional insights and examples, refer to our compilation of [case history papers](https://watex.readthedocs.io/en/latest/citing.html) that utilized *WATex*.

## Contributions

The development and success of *WATex* have been made possible through contributions from the following 
institutions:

1. Department of Geophysics, School of Geosciences & Info-physics, [Central South University](https://en.csu.edu.cn/), China.
2. Hunan Key Laboratory of Nonferrous Resources and Geological Hazards Exploration, Changsha, Hunan, China.
3. Laboratoire de Geologie, Ressources Minerales et Energetiques, UFR des Sciences de la Terre et des Ressources Minières, [Université Félix Houphouët-Boigny](https://www.univ-fhb.edu.ci/index.php/ufr-strm/), Côte d'Ivoire.

For inquiries, suggestions, or contributions, please reach out to the main developer, [_LKouadio_](https://wegeophysics.github.io/) at <etanoyau@gmail.com>.

