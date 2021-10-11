# WATex-AI: A special toolbox for WATer EXploration  using AI Learning methods

[![Build Status](https://travis-ci.com/WEgeophysics/watex.svg?branch=master)](https://travis-ci.com/WEgeophysics/watex)
 ![Requires.io (branch)](https://img.shields.io/requires/github/WEgeophysics/watex/master?style=flat-square) ![GitHub](https://img.shields.io/github/license/WEgeophysics/watex?color=blue&label=Licence&style=flat-square) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5560937.svg)](https://doi.org/10.5281/zenodo.5560937) ![GitHub release (latest by date)](https://img.shields.io/github/v/release/WEgeophysics/watex) ![GitHub repo size](https://img.shields.io/github/repo-size/WEgeophysics/watex?style=flat-square) ![GitHub issues](https://img.shields.io/github/issues/WEgeophysics/watex)


## Overview

The mission of toolbox is to bring a piece of solution in a wide program of   **_WATER4ALL_** especially in Africa and participate of [Sustanaible Development Goals N6](https://www.un.org/sustainabledevelopment/development-agenda/) achievement. 

* **Goals** 

    **WATex-AI** has four (04) objectives:
    -  Contribute to select the best anomaly presumed to give a  suitable flow rate(FR) according
         to the type of hydraulic required for the targeted population.
    -  Intend to supply drinking water for regions faced to water scarcity  by predicting FR before  drilling  
         to limit the failures drillings and dry boreholes.
    -  Minimize the risk of dry boreholes and failure drillings which lead for affordable  project budget elaboration during the water campaigns. 
         Less expensive pojects is economically profitable in term of funding-raise from partners and organizations aids.  
    -  Involve water sanitation for population welfare by bringing a piece of solution of their daily problems.
        The latter goal should not be developped for the first realease. 
   
* **Learning methods implemented**

    - Supervised learnings:  
        -  Support vector machines: [**SVMs**](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
        -  Neighbors: **KNN** 
        -  Trees: **DTC**
        -  Ensemble methods (RandomForests, Bagging and Pasting, Boosting)
    - Unsupervided learnings(in progress):
        -  Artificial neural networks **ANN** 
        -  Apriori
     - Dimensionality Reduction 
         -  Kernel Principal Component Analysis **k-PCA** 
         -  t-distributed Stochastic Neighbor Embedding **t-SNE**
         -  Randomized PCA
         -  Locally Linear Embedding (LLE)
* **Note** 

    Actually the supervised part works properly and the developement of unsupervised learning and deep 
    learning part will be added as things progress. 

## Data preparation steps

Before taking advantage of WATex algorihms especially when dealing with Electrical Resistivity Profile(ERP)
as well as the Vertical Electrical Sounding (VES) data, we need a few steps of data preparing. 
ERP and VES data straighfordwarly collected from field MUST be referenced. An example to how to
prepare ERP and VES data can be found in `data/geof_data` directory. If ERP and VES are in the same Excelworkbook in separed sheets,
use the tool in  `read_from_excelsheets` and `write_excel` from `watex.utils.ml_utils` to separate each ERP and VES 
by keeping the same location coordinate where the VES is done. 
A new directory `_anEX_` shoud be created with new built data. Once the `build` is sucessfully done, the geoelectrical 
 features shoud be computed automatically. To have full control of your selected anomaly, the
 `lower`, `upper` (anomaly boundaries) and `se` or`ves|*|0` of selected anomaly should be specified on each 
 ERP survey line in Excelsheet (see `data/geof_data/XXXXXXX.csv`) then  a new ExcelWorkbook `main.<name of survey area>.csv` shoud 
 be created. Once the features' file is generated, now enjoy your End-to-End Machine Learning(ML) project with implemented algorithms.


## Documentation 

* Toolbox mission and objectives: https://github.com/WEgeophysics/watex/wiki
* Codes implementations: https://github.com/WEgeophysics/watex/wiki/Some-functionalities
* Case history: Implementation in `Bagoue` region in north part of [Cote d'Ivoire](https://en.wikipedia.org/wiki/Ivory_Coast)
         click on the [model prediction in Bagoue region](https://github.com/WEgeophysics/watex/blob/WATex-process/examples/codes/pred_r.PNG) 
    To fetch the original data of Bagoue area, do: 
```
>>> from watex.datasets import fetch_data 
>>> data = fetch_data('Bagoue original')[data]
>>> attributes_infos = fetch_data('Bagoue original')['attrs-infos']
```

## Licence 

WATex-AI is under Massachusetts Institute of Technology License [MIT](https://www.mit.edu/~amini/LICENSE.md).


## System requirements :I
* Python 3.7+ 

## Contributors
  
1. Key Laboratory of Geoscience Big Data and Deep Resource of Zhejiang Province , School of Earth Sciences, Zhejiang University, China
2. Equipe de Recherche Géophysique Appliquée, Laboratoire de Géologie Ressources Minérales et Energétiques, UFR des Sciences de la Terre et des Ressources Minières, Université Félix Houphouët-Boigny, Cote d'Ivoire. 

* Developer: [1](http://www.zju.edu.cn/english/), [2](https://www.univ-fhb.edu.ci/index.php/ufr-strm/) [_Kouadio K. Laurent_](kkouao@zju.edu.cn), _etanoyau@gmail.com_
* Contributor: [1](http://www.zju.edu.cn/english/) [_Binbin MI_](mibinbin@zju.edu.cn)
    



	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
