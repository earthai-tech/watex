# WATex-AI: A special toolbox for WATer EXploration  using AI Learning methods

[![Build Status](https://travis-ci.com/WEgeophysics/watex.svg?branch=master)](https://travis-ci.com/WEgeophysics/watex)
 ![Requires.io (branch)](https://img.shields.io/requires/github/WEgeophysics/watex/master?style=flat-square) ![GitHub](https://img.shields.io/github/license/WEgeophysics/watex?color=blue&label=Licence&style=flat-square) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4896758.svg)](https://doi.org/10.5281/zenodo.4896758)


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
        -  Ensemble methods 
    - Unsupervided learnings(not implemented yet):
        -  Kernel Principal Component Analysis **k-PCA** 
        -  Artificial neural networks **ANN** 
        -  t-distributed Stochastic Neighbor Embeedding **t-SNE**
        -  Apriori
        
* **Note** 

    Actually only the supervised part including [SVMs](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) works properly and the developement with pure Python is still ongoing. 
    Other AI algorithms implemented will be added as things progress. To handle some functionalities before the full development, please refer to `.checkpoints ` folder.
     
## Documentation 

* Toolbox mission and objectives: https://github.com/WEgeophysics/watex/wiki
* Codes implementations: https://github.com/WEgeophysics/watex/wiki/Some-functionalities
* Case history: Implementation in `Bagoue` region in north part of [Cote d'Ivoire](https://en.wikipedia.org/wiki/Ivory_Coast)
         click on the [model prediction in Bagoue region](https://github.com/WEgeophysics/watex/blob/WATex-process/examples/codes/pred_r.PNG) 

## Licence 

WATex-AI is under Massachusetts Institute of Technology License [MIT](https://www.mit.edu/~amini/LICENSE.md).

## System requirements 
* Python 3.7+ 

## Contributors
  
1. Key Laboratory of Geoscience Big Data and Deep Resource of Zhejiang Province , School of Earth Sciences, Zhejiang University, China
2. Equipe de Recherche Géophysique Appliquée, Laboratoire de Géologie Ressources Minérales et Energétiques, UFR des Sciences de la Terre et des Ressources Minières, Université Félix Houphouët-Boigny, Cote d'Ivoire. 

* Developer: [1](http://www.zju.edu.cn/english/), [2](https://www.univ-fhb.edu.ci/index.php/ufr-strm/) [_Kouadio K. Laurent_](kkouao@zju.edu.cn), _etanoyau@gmail.com_
* Contributor: [1](http://www.zju.edu.cn/english/) [_Binbin MI_](mibinbin@zju.edu.cn)
    



	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
