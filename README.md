# WATex-AI: A special toolbox for WATer EXploration  using AI Learning methods

[![Build Status](https://travis-ci.com/WEgeophysics/watex.svg?branch=master)](https://travis-ci.com/WEgeophysics/watex) ![Requires.io (branch)](https://img.shields.io/requires/github/WEgeophysics/watex/master?style=flat-square) ![GitHub](https://img.shields.io/github/license/WEgeophysics/watex?color=blue&label=Licence&style=flat-square) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4896758.svg)](https://doi.org/10.5281/zenodo.4896758)


## Overview

The mission of toolbox is to bring a piece of solution in a wide program of   **_WATER4ALL_** especially in Africa and participate of [Sustanaible Development Goals N6](https://www.un.org/sustainabledevelopment/development-agenda/) achievement. 

* **Goals** 

    **WATex-AI** has four (04) objectives:
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
        -  Support vector machines: [**SVMs**](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
        -  Neighbors: **KNN** 
        -  Trees: **DTC**
    - Unsupervided learnings: 
        -  Artificial neural networks **ANN** (not implemented yet)
 
* **Note** 

    Actually only [SVMs](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) works porperly and the developement with pure Python is still ongoing. 
    Other AI algorithms implemented will be added as things progress. To handle some functionalities before the full development, please refer to `.checkpoints ` folder.
     
## Documentation 

* Toolbox mission and objectives: https://github.com/WEgeophysics/watex/wiki

## Licence 

WATex-AI is under Massachusetts Institute of Technology License [MIT](https://www.mit.edu/~amini/LICENSE.md).

## Units used 

1. Apparent resistivity `rhoa` in ohm.meter. 
2. Standard fracture index `sfi`, no unit(n.u). 
3. Anomaly ratio `anr` ,  in %.
4. Anomaly power *Pa* or `power`  in meter(m). 
5. Anomaly magnitude *Ma* or `magnitude` in ohm.m. 
6. Anomaly shape - can be `V, M, K, L, H, C, V` and `W` (n.u). 
7. Anomaly type - can be `EC, NC, CB2P`and `CP` (n.u).
	- `EC`: *Extensive conductive*
	- `NC`: *Narrow conductive* 
	- `CP`: *Conductive plane*
	- `CB2P`: *Conductive between two planes* 
9. Layer thickness `thick` in m. 
10. Ohmic surface `ohmS` in ohm.m2. 
11. Station( site) or position is given as `pk` in m.

## Get the geo-electrical features from selected anomaly

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
...                dipole_length =10.,                # distance between measurements 
...                posMinMax= (90, 130),              # select anomaly boundaries 
...                turn_on =True                      # display infos
                 )
```
 - To get the _best anomaly_ point from the 'erp_line' if `auto` option is enabled, try: 
```
>>> erp_obj.select_best_point_ 
Out[1]: 170 			# --|> The best point is found  at position (pk) = 170.0 m. ----> Station 18              
>>> erp_obj.select_best_value_ 
Out[1]: 80.0			# --|> Best conductive value selected is = 80.0 Ω.m                    
```
- To get the next geo-electrical features, considered the _prefix_`best_+ {feature_name}`. 
For instance :
```
>>> erp_obj.best_type         # Type of the best selected anomaly on erp line
>>> erp_obj.best_shape        # Best selected anomaly shape is "V"
>>> erp_obj.best_magnitude   # Best anomaly magnitude is 45 Ω.m. 
>>> erp_obj.best_power         # Best anomaly power is 40.0 m. 
>>> erp_obj.best_sfi     	# best anomaly standard fracturation index.
>>> erp_obj.best_anr           # best anomaly ration the whole ERP line.
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
For multiples `erp` file reading try, all `geo_electrical_features` from all 
ERP survey line are auto-computed. For example: 

```
>>> from watex.core.erp import ERP_collection
>>> erpColObj= ERP_collection(listOferpfn= 'data/erp')
>>> erpColObj.erpdf 
```
Get all features for data analysis and prediction purpose  by calling `Features`
from `~.core.geofeatures` module as **(1)** or do the same task by calling different module collections`ves`, `geol`,
considered as Python object **(2)**: 
```
(1) 							|(2)
>>> from watex.core.geofeatures import Features       	|>>> from watex.core.geofeatures import Features
>>> featurefn ='data/geo_fdata/BagoueDataset2.xlsx' 	|>>> from watex.core.erp import ERP_collection 
>>> featObj =Features(features_fn= featurefn) 		|>>> from watex.core.ves import VES_collection 
>>> featObj.site_ids 					|>>> from watex.core.geology import Geology, Borehole 
>>> featObj.site_names 					|>>> featObj =Features(ErpColObjs=ERP_collection('data/erp')
>>> featObj.df 						|... 		vesObjs=VES_collection('data/ves'),
                       					|...		geoObjs=Geology('data/geol'),
                       					|...		boreholeObjs=Borehole('data/boreh'))
							|>>> featObj.site_ids
							|>>> featObj.site_names
							|>>> featObj.df
``` 
![](https://github.com/WEgeophysics/watex/blob/WATex-process/examples/codes/features_computation.PNG)


## Data analysis and quick plot hints

 To solve the classification problem in `supervised learning`, we need to categorize  the `targetted` numerical values 
 into categorized values using the module `watex.analysis` . It's possible to export data using `~writedf` function: 
```
>>> from watex.analysis.features import sl_analysis 
>>> slObj =sl_analysis(
...   data_fn='data/geo_fdata/BagoueDataset2.xlsx',
...   set_index =True)
>>> slObj.writedf()
``` 
To quick see how data look like, call `~viewer`packages: 
```
>>> from watex.viewer.plot import QuickPlot 
>>> qplotObj = QuickPlot( df = slObj.df , lc='b') 
>>> qplotObj.hist_cat_distribution(target_name='flow')
```
If `df` is not given, It's easy to quick visualize the data setting the argument `data_fn` that match the 
to datafile  like `data_fn ='data/geo_fdata/BagoueDataset2.xlsx'`. Both will give the same result.

To draw a plot of two features with bivariate and univariate graphs, use `~.QuickPlot.joint2features` methods as
below:
```
>>> from watex.viewer.plot.QuickPlot import joint2features
>>> qkObj = QuickPlot(
...             data_fn ='data/geo_fdata/BagoueDataset2.xlsx', lc='b', 
...             target_name = 'flow', set_theme ='darkgrid', 
...             fig_title='`ohmS` and `lwi` features linked'
...             )  
>>> sns_pkws={
...            'kind':'reg' , #'kde', 'hex'
...            # "hue": 'flow', 
...               }
>>> joinpl_kws={"color": "r", 
                'zorder':0, 'levels':6}
>>> plmarg_kws={'color':"r", 'height':-.15, 'clip_on':False}           
>>> qkObj.joint2features(features=['ohmS', 'lwi'], 
...            join_kws=joinpl_kws, marginals_kws=plmarg_kws, 
...            **sns_pkws, 
...            ) 
``` 
To draw a scatter plot with possibility of several semantic features groupings, use `scatteringFeatures`
methods. Indeed this method analysis is a process of understanding  how features in a 
dataset relate to each other and how those relationships depend on other features. It easy to customize
plot if user has an experience of  `seaborn` plot styles. For instance we can visualize the relationship 
between the features `lwi` , '`flow` and the `geology(geol)`' as: 
```
>>> from watex.viewer.plot.QuickPlot import  scatteringFeatures
>>> qkObj = QuickPlot(
...    data_fn ='data/geo_fdata/BagoueDataset2.xlsx' , 
...             fig_title='Relationship between geology and level of water inflow',
...             xlabel='Level of water inflow (lwi)', 
...             ylabel='Flow rate in m3/h'
...            )  
>>> marker_list= ['o','s','P', 'H']
>>> markers_dict = {key:mv 
...               for key, mv in zip( list (
...                       dict(qkObj.df ['geol'].value_counts(
...                           normalize=True)).keys()), 
...                            marker_list)}
>>> sns_pkws={'markers':markers_dict, 
...          'sizes':(20, 200),
...          "hue":'geol', 
...          'style':'geol',
...         "palette":'deep',
...          'legend':'full',
...          # "hue_norm":(0,7)
...            }
>>> regpl_kws = {'col':'flow', 
...             'hue':'lwi', 
...             'style':'geol',
...             'kind':'scatter'
...            }
>>> qkObj.scatteringFeatures(features=['lwi', 'flow'],
...                         relplot_kws=regpl_kws,
...                         **sns_pkws, 
...                    ) 
```

## System requirements 
* Python 3.7+ 

## Contributors
  
1. Key Laboratory of Geoscience Big Data and Deep Resource of Zhejiang Province , School of Earth Sciences, Zhejiang University, China
2. Equipe de Recherche Géophysique Appliquée, Laboratoire de Géologie Ressources Minérales et Energétiques, UFR des Sciences de la Terre et des Ressources Minières, Université Félix Houphouët-Boigny, Cote d'Ivoire. 

* Developer's name: [1](http://www.zju.edu.cn/english/), [2](https://www.univ-fhb.edu.ci/index.php/ufr-strm/) [_Kouadio K. Laurent_](kkouao@zju.edu.cn), _etanoyau@gmail.com_
* Contributors' names: [1](http://www.zju.edu.cn/english/) [_Binbin MI_](mibinbin@zju.edu.cn)
    



	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
