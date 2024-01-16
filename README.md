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

**WATex** is a Python-based library designed for Groundwater Exploration (GWE). It integrates cutting-edge methodologies, including Direct-Current (DC) resistivity (Electrical Profiling (ERP) & Vertical Electrical Sounding (VES)), short-period Electromagnetic (EM), geology, and hydrogeology. Leveraging Machine Learning techniques, WATex enables users to:

- Automatically identify optimal drilling locations to reduce the incidence of unsuccessful drillings and unsustainable boreholes.
- Predict well water content, encompassing groundwater flow rate and water inrush levels.
- Restore EM loss signals in areas heavily affected by interferences and noise.
- And more.

## Documentation

For additional resources, visit the [WATex library website](https://watex.readthedocs.io/en/latest/). The software's [API reference](https://watex.readthedocs.io/en/latest/api_references.html) and [examples page](https://watex.readthedocs.io/en/latest/glr_examples/index.html) offer insights into the expected results. A comprehensive [step-by-step guide](https://watex.readthedocs.io/en/latest/glr_examples/applications/index.html#applications-step-by-step-guide) is available for tackling real-world engineering problems, such as calculating DC parameters and predicting the k-parameter.

## License

WATex is distributed under the [BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause).

## Installation

WATex requires Python 3.9 or newer.

### Installing from PyPI

Install WATex from PyPI with the command:

```bash
pip install watex
```

### Installing from Conda-Forge

To install from [Conda-Forge](https://conda-forge.org/):

```bash
conda install -c conda-forge watex
```

For the latest development version, clone the repository:

```bash
git clone https://github.com/WEgeophysics/watex.git
```

Visit our [installation guide](https://watex.readthedocs.io/en/latest/installation.html) for detailed installation instructions and dependency management.

## Examples

### 1. Auto-Detection of Drilling Locations

Generate synthetic ERP resistivity data for 50 stations:

```python
import watex as wx
data = wx.make_erp(n_stations=50, max_rho=1e4, min_rho=10., as_frame=True, seed=42)
```

- Basic Detection (BD)

Automatically proposes a suitable drilling location without considering site constraints, aiming for a minimum flow rate of 1m3/hr.

```python
robj = wx.ResistivityProfiling(auto=True).fit(data)
robj.sves_
# Output: 'S025'
```

- Auto-Detection with Constraints (ADC)

Considers site-specific restrictions, such as proximity to heritage sites or pollution risks. Example with applied constraints:

```python
restrictions = {
    'S10': 'Household waste site, avoid contamination',
    'S27': 'Municipality site, no authorization to drill',
    'S29': 'Heritage site, drilling prohibited',
    'S42': 'Polluted area, contamination risk',
    'S46': 'Marsh zone, seasonal dry-up risk'
}
robj = wx.ResistifyProfiling(constraints=restrictions, auto=True).fit(data)
robj.sves_
# Output: 'S033'
```

Note: Conduct DC-sounding (VES) before drilling to assess fracture zones' effectiveness. More information is available in the [documentation](https://watex.readthedocs.io/en/latest/).

### 2. EM Tensor Recovery and Analysis

Fetching and analyzing AMT data:

```python
import watex as wx
e = wx.fetch_data('huayuan', samples=20, key='noised')
edi_data = e.data
```

Quality control and confidence interval visualization:

```python
po = wx.EMProcessing().fit(edi_data)
r = po.qc(tol=0.2, return_ratio=True)
# Output: 0.95

wx.plot_confidence_in(edi_data)
```

Further analyses and visualizations are available in the [software documentation](https://watex.readthedocs.io/en/latest/).

## Citations

If WATex contributes to your research, please cite:

> _Kouadio, K.L., Liu, J., Liu, R., 2023. WATex: Machine Learning Research in Water Exploration.
> SoftwareX. 101367(2023). [DOI](https://doi.org/10.1016/j.softx.2023.101367)_

Citing [scikit-learn](https://scikit-learn.org/stable/about.html#citing-scikit-learn) is also recommended when referencing WATex. Explore [case histories](https://watex.readthedocs.io/en/latest/citing.html) using WATex for more insights.

## Contributions

Key contributors include:

1. Department of Geophysics, School of Geosciences & Info-physics, [Central South University](https://en.csu.edu.cn/), China.
2. Hunan Key Laboratory of Nonferrous Resources and Geological Hazards Exploration, Changsha, Hunan, China.
3. Laboratoire de Geologie Ressources Minerales et Energetiques, UFR des Sciences de la Terre et des Ressources Minières, [Université Félix Houphouët-Boigny](https://www.univ-fhb.edu.ci/index.php/ufr-strm/), Cote d'Ivoire.

Developer: [_L. Kouadio_](https://wegeophysics.github.io/) (etanoyau@gmail.com)
