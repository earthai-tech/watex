# WATex-AI: A special toolbox for WATer EXploration  using AI Learning methods

[![Build Status](https://travis-ci.com/WEgeophysics/watex.svg?branch=master)](https://travis-ci.com/WEgeophysics/watex) ![Requires.io (branch)](https://img.shields.io/requires/github/WEgeophysics/watex/master?style=flat-square) ![GitHub](https://img.shields.io/github/license/WEgeophysics/watex?color=blue&label=Licence&style=flat-square) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4896758.svg)](https://doi.org/10.5281/zenodo.4896758)


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

## Licence 

WATex-AI is under Massachusetts Institute of Technology License [MIT](https://www.mit.edu/~amini/LICENSE.md).

## Features and units used 

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
10. Station( site) or position is given as `pk` in m.
11. Ohmic surface `ohmS` in ohm.m2 got from the vertical electrical sounding(VES)
12. Level of water inflow `lwi` in m got from the existing boreholes.
13. Geology `geol` of the survey area got during the drilling or from the previous geology works.

## Get the geo-electrical features from selected anomaly

**Geo-electrical features** are mainly used for FR prediction purposes. 
 Beforehand, we refer  to the  data directory `data\erp` accordingly for this demonstration. 
 The electrical resistivity profile (ERP) data of survey line  is found on `l10_gbalo.csv`. There are two ways to get **Geo-electrical features**. 
 The first option  is to provide the selected anomaly boundaries into the argument ` posMinMax` and 
  the seccond way is to let program  find automatically the *the best anomaly point*. The first option is strongly recommended. 

 First of all, we import the module _ERP_ from ` watex.core.erp.ERP` to build the `erp_obj`
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
 - To get automatically the _best anomaly_ point from the 'erp_line' of survey area, enable `auto` option and try: 
```
>>> erp_obj.select_best_point_ 
Out[1]: 170 			# --|> The best point is found  at position (pk) = 170.0 m. ----> Station 18              
>>> erp_obj.select_best_value_ 
Out[1]: 80.0			# --|> Best conductive value selected is = 80.0 Ω.m                    
```
- To get the other geo-electrical features, considered the _prefix_`best_+ {feature_name}`. 
For instance :
```
>>> erp_obj.best_type         # Type of the best selected anomaly on erp line
>>> erp_obj.best_shape        # Best selected anomaly shape is "V"
>>> erp_obj.best_magnitude   # Best anomaly magnitude is 45 Ω.m. 
>>> erp_obj.best_power         # Best anomaly power is 40.0 m. 
>>> erp_obj.best_sfi     	# best anomaly standard fracturation index.
>>> erp_obj.best_anr           # best anomaly ration the whole ERP line.
```
- If `auto` is enabled, the program could find additionally three(03) maximum  best 
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
Generate multiple `Features` from different locations of `erp` survey line by computing all `geo_electrical_features` from all 
ERP survey line using the `watex.core.erp.ERP_collection` module as below: 
```
>>> from watex.core.erp import ERP_collection
>>> erpColObj= ERP_collection(listOferpfn= 'data/erp')
>>> erpColObj.erpdf 
```
Get all features for data analysis and prediction purpose  by calling `Features`
from module `~.core.geofeatures` as **(1)** or do the same task by calling different module collections from `ves`, `geol`,
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
 into categorized values using the module `watex.analysis` . It's possible to export data using the decorated `~writedf` function: 
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
It's easy to quick visualize the data by setting the argument `data_fn`, if `df` is not given, as `data_fn ='data/geo_fdata/BagoueDataset2.xlsx'`.
Both will give the same result.
To draw a plot of two features with bivariate and univariate graphs, use `~.QuickPlot.joint2features` method as
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
...                'zorder':0, 'levels':6}
>>> plmarg_kws={'color':"r", 'height':-.15, 'clip_on':False}           
>>> qkObj.joint2features(features=['ohmS', 'lwi'], 
...            join_kws=joinpl_kws, marginals_kws=plmarg_kws, 
...            **sns_pkws, 
...            ) 
``` 
To draw a scatter plot with possibility of several semantic features groupings, use `scatteringFeatures`
method. Indeed this method analysis is a process of understanding  how features in a 
dataset relate to each other and how those relationships depend on other features. It easy to customize
plot if user has an experience of  `seaborn` plot styles. For instance we can visualize the relationship 
between the mileages `lwi` , '`flow` and the `geology(geol)`' as: 
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
WATex-AI gives a piece of  mileages  discussion. Indeed, discussing about mileages seems to be a good approach to 
comprehend the features relationship, their correlation as well as their influence between each other. 
For instance, to try to discuss  about the mileages `'ohmS', 'sfi','geol' and 'flow'`, we merely 
need to import `discussingFeatures` method from `QuickPlot` class as below: 
```
>>> from viewer.plot.QuickPlot import discussingFeatures 
>>> qkObj = QuickPlot(  fig_legend_kws={'loc':'upper right'},
...          fig_title = '`sfi` vs`ohmS|`geol`',
...            )  
>>> sns_pkws={'aspect':2 , 
...          "height": 2, 
...                  }
>>> map_kws={'edgecolor':"w"}   
>>> qkObj.discussingFeatures(
...    data_fn ='data/geo_fdata/BagoueDataset2.xlsx' , 
...                         features =['ohmS', 'sfi','geol', 'flow'],
...                           map_kws=map_kws,  **sns_pkws)                          
```

## Data processing  

Processing is usefull before modeling step. To process data, a default implementation is given for 
data `preprocessing` after data sanitizing. It consists of creating a model pipeline using different `supervised learnings` methods. 
A default pipeline is created though the `prepocessor` designing. Indeed  a `preprocessor` is a set 
of `transformers + estimators` and multiple other functions to boost the prediction. WATex-AI includes
nine(09) inner defaults estimators such as `neighbors`, `trees`, `svm` and `~.ensemble` estimators category.
An example of  `preprocessing`class implementation is given below: 
```
>>> from watex.processing.sl import Preprocessing
>>> prepObj = Preprocessing(drop_features = ['lwi', 'x_m', 'y_m'],
...    data_fn ='data/geo_fdata/BagoueDataset2.xlsx')
>>> prepObj.X_train, prepObj.X_test, prepObj.y_train, prepObj.y_test
>>> prepObj.categorial_features, prepObj.numerical_features 
>>> prepObj.random_state = 25 
>>> preObj.test_size = 0.25
>>> prepObj.make_preprocessor()         # use default preprocessing
>>> prepObj.make_preprocessing_model( default_estimator='SVM')
>>> prepObj.preprocessing_model_score
>>> prepObj.preprocess_model_prediction
>>> prepObj.confusion_matrix
>>> prepObj.classification_report
```
It 's also interesting to evaluate a quick model score without any preprocessing beforehand by calling the 
 `Processing` superclass as : 
```
>>> from watex.processing.sl import Processing 
>>> processObj = Processing(
...   data_fn = 'data/geo_fdata/BagoueDataset2.xlsx')
>>> processObj.quick_estimation(estimator=DecisionTreeClassifier(
...    max_depth=100, random_state=13))
>>> processObj.model_score
0.5769230769230769                  # model score ~ 57.692   %
>>> processObj.model_prediction
```
Now let's evaluate onto the same dataset the `model_score` by reinjecting the default composite estimator 
 using `preprocessor` pipelines. We trigger the composite estimator  by switching  the `auto` option to `True`.
```
>>> processObj = Processing(data_fn = 'data/geo_fdata/BagoueDataset2.xlsx', 
...                        auto=True)
>>> processObj.preprocessor
>>> processObj.model_score
0.72487896523648201                 # new composite estimator ~ 72,49   %
>>> processObj.model_prediction
``` 
We clearly see the difference of  `14.798%` between the two options. Furthermore,  we can get the validation curve
 by callig `get_validation_curve` function using the same default composite estimator like: 

```
>>> processObj.get_validation_curve(switch_plot='on', preprocess_step=True)
```

## Modeling 

The most interesting and challenging part of modeling is the `tuning hyperparameters` after designing a composite estimator. 
Getting the best params is a better way to reorginize the created pipeline `{transformers +estimators}` so to 
have a great capability of data generalization. In the following example, we try to create a simple pipeline 
and we'll tuning its hyperparameters. Then the best parameters obtained will be reinjected into the design estimator for the next 
prediction. This is an example and the user has the ability to create its own pipelines more powerfull. 
We consider a **svc** estimator as default estimator. The process are described below: 
```
>>> from watex.modeling.sl.modeling import Modeling 
>>> from sklearn.preprocessing import RobustScaler, PolynomialFeatures 
>>> from sklearn.feature_selection import SelectKBest, f_classif 
>>> from sklearn.svm import SVC 
>>> from sklearn.compose import make_column_selector 
>>> my_own_pipelines= {
        'num_column_selector_': make_column_selector(
            dtype_include=np.number),
        'cat_column_selector_': make_column_selector(
            dtype_exclude=np.number),
        'features_engineering_':PolynomialFeatures(
            3, include_bias=False),
        'selectors_': SelectKBest(f_classif, k=3), 
        'encodages_': RobustScaler()
          }
>>> my_estimator = SVC(C=1, gamma=1e-4, random_state=7)             # random estimator 
>>> modelObj = Modeling(data_fn ='data/geo_fdata/BagoueDataset2.xlsx', 
               pipelines =my_own_pipelines , 
               estimator = my_estimator)
>>> hyperparams ={
    'columntransformer__pipeline-1__polynomialfeatures__degree': np.arange(2,10), 
    'columntransformer__pipeline-1__selectkbest__k': np.arange(2,7), 
    'svc__C': [1, 10, 100],
    'svc__gamma':[1e-1, 1e-2, 1e-3]}
>>> my_compose_estimator_ = modelObj.model_ 
>>> modelObj.tuning_hyperparameters(
                            estimator= my_compose_estimator_ , 
                            hyper_params= hyperparams, 
                            search='rand') 
>>> modelObj.best_params_
Out[7]:
{'columntransformer__pipeline-1__polynomialfeatures__degree': 2, 'columntransformer__pipeline-1__selectkbest__k': 2, 'svc__C': 1, 'svc__gamma': 0.1}
>>> modelObj.best_score_
Out[8]:
-----------------------------------------------------------------------------
> SupportVectorClassifier       :   Score  =   73.092   %
-----------------------------------------------------------------------------
```
We can now rebuild and rearrange the pipeline by specifying the best parameters values and run again so to get the
the new model_score and model prediction: 
```
>>> modelObj.model_score
Out[9]:
-----------------------------------------------------------------------------
> SupportVectorClassifier       :   Score  =   75.132   %
----------------------------------------------------------------------------- 
``` 
* **Note**: This is an illustration example, you can tuning your hyperparameters using an other 
estimators either the *supervised learning* method by handling the method
`watex.modeling.sl.modeling.Modeling.tuning_hyperparameters` parameters. You can quick have a look of your
*learning curve* by calling decorated method `get_learning_curve` as below: 
```
>>> from watex.modeling.sl.modeling import Modeling
>>> processObj = Modeling(
    data_fn = 'data/geo_fdata/BagoueDataset2.xlsx')
>>> processObj.get_learning_curve (estimator= my_compose_estimator_,
        switch_plot='on')
```

## System requirements 
* Python 3.7+ 

## Contributors
  
1. Key Laboratory of Geoscience Big Data and Deep Resource of Zhejiang Province , School of Earth Sciences, Zhejiang University, China
2. Equipe de Recherche Géophysique Appliquée, Laboratoire de Géologie Ressources Minérales et Energétiques, UFR des Sciences de la Terre et des Ressources Minières, Université Félix Houphouët-Boigny, Cote d'Ivoire. 

* Developer's name: [1](http://www.zju.edu.cn/english/), [2](https://www.univ-fhb.edu.ci/index.php/ufr-strm/) [_Kouadio K. Laurent_](kkouao@zju.edu.cn), _etanoyau@gmail.com_
* Contributors' names: [1](http://www.zju.edu.cn/english/) [_Binbin MI_](mibinbin@zju.edu.cn)
    



	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
