# WATex: A special toolbox for WATer EXploration  using AI Learning methods

[![Build Status](https://travis-ci.com/WEgeophysics/watex.svg?branch=master)](https://travis-ci.com/WEgeophysics/watex) ![Requires.io (branch)](https://img.shields.io/requires/github/WEgeophysics/watex/master?style=flat-square) ![GitHub](https://img.shields.io/github/license/WEgeophysics/watex?color=blue&label=Licence&style=flat-square) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4896758.svg)](https://doi.org/10.5281/zenodo.4896758)


## Overview

The mission of toolbox is to bring a piece of solution in a wide program of   **_WATER4ALL_** especially in Africa and participate of [Sustanaible Development Goals N6](https://www.un.org/sustainabledevelopment/development-agenda/) achievement. 

* **Goals** 

    **WATex** has four (04) objectives:
    -  Contribute to select the best anomaly presumed to give a  suitable flow rate(FR) according
         to the type of hydraulic required for the targeted population.
    -  Intend to supply drinking water for regions faced to water scarcity  by predicting FR before  drilling to 
         to limit the failures drillings and dry boreholes.
    -  Minimize the risk of dry boreholes and failure drillings which lead for  affordable  project budget elaboration during the water campaigns. 
         Less expensive pojects is economiccaly profitable in term of funding-raise from partners and organizations aids.  
    -  Involve water sanitation for population welfare by bringing a piece of solution of their daily problems.
        The latter goal should not be developped for the first realease. 
   
* **Learning methods implemented**

    - Supervised learnings:  
        -  Support vector machines: [SVMs](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
        -  Neighbors: **KNN** 
        -  Trees: **DTC**
    - Unsupervided learnings: 
        -  Artificial neural networks **ANN** (not implemented yet)
 
* **Note** 

    Actually only [SVMs](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)) works porperly and the developement with pure Python is still ongoing. 
    Other AI algorithms implemented will be add as things progress. To handle some fonctionalities before the full development, please refer to `.checkpoints ` folder.
     
## Documentation 

* Toolbox mission and objectives: https://github.com/WEgeophysics/watex/wiki

## Licence 

WATex is under Massachusetts Institute of Technology License [MIT](https://www.mit.edu/~amini/LICENSE.md).

## Units used 

1. Apparent resistivity `rhoa` in ohm.meter. 
2. Standard fracture index `sfi`, no unit(n.u). 
3. Anomaly ratio `anr` ,  in %.
4. Anomaly power *Pa* or `power`  in meter(m). 
5. Anomaly magnitude *Ma* or `magnitude` in ohm.m. 
6. Anomaly shape - can be `V, M, K, L, H, C, V` and `W` (n.u). 
7. Anomaly type - can be `EC, NC, CB2P`and `PC` (n.u).
8. Layer thickness `thick` in m. 
9. Ohmic surface `ohmS` in ohm.m2. 
10. Station( site) or position is given as `pk` in m.

## How to get the geo-electrical features from selected anomaly?

**Geo-electrical features** are mainly used FR prediction purposes. 
 Beforehand, we refer  to the  data directory `data\erp` accordingly for this demonstration. 
 The 'ERP' data of survey line  is found on `l10_gbalo.csv`. There are two ways to get **Geolectrical features**. 
 The first option  is to provide the selected anomaly boundaries into the argument ` posMinMax` and 
  the seccond way is to let program  find automatically the *the best anomaly point*. The first option is strongly recommended. 

 Fist of all , try to import the module _ERP_ from ` watex.core.erp.ERP`  and build `erp_obj`
 as follow: 
```
>>> from watex.core.erp import ERP 
>>> erp_obj =ERP (erp_fn = data/erp/l10_gbalo.csv',  # erp_data 
...                auto=False,                        # automatic computation  option 
...                dipole_length =10.,                 # distance between measurements 
...                posMinMax= (90, 130),               # select anomaly boundaries 
...                 turn_on =True                      # display infos
                 )
```
 - To get the _best anomaly_ point from the 'erp_line' if `auto` option is enabled, try: 
```
>>> erp_obj.select_best_point_ 
Out[1]: 170 
-----------------------------------------------------------------------------
--|> The best point is found  at position (pk) = 170.0 m. ----> Station 18              
-----------------------------------------------------------------------------
>>> erp_obj.select_best_value_ 
Out[1]: 80.0
-----------------------------------------------------------------------------
--|> Best conductive value selected is = 80.0 Ω.m                    
-----------------------------------------------------------------------------
```
- To get the next geo-electrical features, considered the _prefix_`abest_+ {feature_name}`. 
For instance :

```
>>> erp_obj.abest_type         # Type of the best selected anomaly on erp line
Out[3]:  CB2P                  # is  contact between two planes "CB2P". 
>>> erp_obj.abest_shape         
Out[4]: V                       # Best selected anomaly shape is "V"
>>> erp_obj.abest_magnitude    
Out[5]: 45                     # Best anomaly magnitude is 45 Ω.m. 
>>> erp_obj.abest_power         
Out[6]: 40.0                    # Best anomaly power is 40.0 m. 
>>> erp_obj.abest_sfi          
Out[7]: 1.9394488747363936      # best anomaly standard fracturation index.
>>> erp_obj.abest_anr           # best anomaly ration the whole ERP line.
Out[8]: 50.76113145430543 % 
```
- If `auto` is enabled, the program could find additional maximum three best 
conductive points from the whole  ERP line as : 
```
>>> erp_obj.best_points 
-----------------------------------------------------------------------------
--|> 3 best points were found :
 01 : position = 170.0 m ----> rhoa = 80 Ω.m
 02 : position = 80.0 m ----> rhoa = 95 Ω.m
 03 : position = 40.0 m ----> rhoa = 110 Ω.m               
-----------------------------------------------------------------------------
```
for multiples `erp` file reading try, all `geo_electrical_features` from all 
ERP survey line are auto-computed. For example: 

```
>>> from watex.core.erp import ERP_collection
>>> erpColObj= ERP_collection(listOferpfn= 'data/erp')
>>> erpColObj.erpdf 
Out[9]:
               id      east      north  power  magnitude shape  type       sfi
0  e2059747141000  790187.0  1093022.0   40.0       45.0     V  CB2P  1.939449
1  e2059722582344  790232.0  1093057.0   50.0       17.0     V  CB2P  1.352764
2  e2059733751112  790724.0  1092789.5   30.0      211.0     V  CB2P  4.787024
```

## System requirements 
* Python 3.7+ 

## Contributors
  
1. Key Laboratory of Geoscience Big Data and Deep Resource of Zhejiang Province , School of Earth Sciences, Zhejiang University, China
2. Equipe de Recherche Géophysique Appliquée, Laboratoire de Géologie Ressources Minérales et Energétiques, UFR des Sciences de la Terre et des Ressources Minières, Université Félix Houphouët-Boigny, Cote d'Ivoire. 

* Developer's name: [1](http://www.zju.edu.cn/english/), [2](https://www.univ-fhb.edu.ci/index.php/ufr-strm/) [_Kouadio K. Laurent_](kkouao@zju.edu.cn), _etanoyau@gmail.com_
* Contributors' names: [1](http://www.zju.edu.cn/english/)[_Binbin MI_](mibinbin@zju.edu.cn)
    



	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
