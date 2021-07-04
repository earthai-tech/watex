 # WATex : Water exploration using AI Learning Methods

![Requires.io (branch)](https://img.shields.io/requires/github/WEgeophysics/watex/master?style=flat-square) ![GitHub](https://img.shields.io/github/license/WEgeophysics/watex?color=blue&label=Licence&style=flat-square)

**_Special toolbox for WATer EXploration  using Artificial Intelligence Learning methods_**

## Overview

**WATex**has three main objectives. Firstly, it's an exploration open source software using AI learning methods like for water exploration like underground water research,
 and secondly intend to supply drinking water for regions faced of water scarcity  by predicting flow rate before any drilling to 
 avoid failures drillings and dry boreholes. The third objective involves water sanitation for population welfare by bringing a piece of solution of their daily problems.
 The latter should not be developped for the first realease. 
 
## Purpose 
 
 **WATex** is developed to  indirectly participate of [Sustanaible Development Goals N6](https://www.un.org/sustainabledevelopment/development-agenda/) achievement which is  `Ensure access to water and sanitation for all`.
 Designing **WATex** using AI lerning methods such as *SVM, KNN, DTC* for supervided learning and *ANN* for unsupervided lerning (not work yet) has double goals. On the one hand,
 it’s used to predict the different classes of flow rate (FR)  from an upstream geoelectrical features analysis.
 On the other hand, it will contribute to select the best anomaly presumes to give a  suitable FR according
 to the type of hydraulic required by the project with lower cost for the targeted population. Thus, for the first demonstration, **WATex** 
 will work with *SVM* to create a *SVM composite estimator (CE-SVC) * to minimize the risk of dry boreholes and failure drillings 
 using *electrical resistivity profile ( ERP)*  and * vertical electrical sounding (VES)* less expensive geophysical  methods. The developpement of the toolbox 
 will be economically profitable from organisations such as [AMCOW](https://amcow-online.org/initiatives/amcow-pan-african-groundwater-program-apagrop), [UNICEF](https://www.unicef.org/),[WHO](https://www.who.int/) and 
 governments to reduce their investments for efficiency results during the Drinking water  supply campaigns. 
 Other AI algorithms implemented will be add  as things progress. 
 
## Units used 

1. Apparent resistivity *_rhoaa_* in ohm.meter 
2. Standard fracture index *SFI*  , no unit(n.u) 
3. Anomaly ratio *ANR* ,  in %
4. Anomaly power *Pa* or *power*  in meter(m) 
5. Anomaly magnitude *Ma* or *magnitude* in ohm.m 
6. Anomaly shape - can be *_V, M, K, L, H, C, V_* and *_W_* (n.u). 
7. Anomaly type - can be *_EC, NC, CB2P_* and *_PC_* (n.u)
8. Layer thickness *_thick_* in m. 
9. Ohmic surface *_OhmS_* in ohm.m2 
10. Station( site) or position is given as *_pk_* in m.

## Compute geo-electrical features 

**Geo-electrical features** are mainly computed  FR prediction purposes. A demo to compute electrical features from _ERP_ is done as follow.  
 Beforehand, we refer  to the  data directory `data\l10_gbalo.csv`accordingly for the `gbalo` locality. There are two ways to get **Geolectrical features**. 
 First possibily is to provide the selected anomaly boundaries as argument ` posMinMax` ans seccond way is to let program to automatically find the *the best anomaly*. 
 The first option is strongly recommended. 
 Fist of all , try to import _ERP_ class ` watex.core.erp.ERP`  as follow: 
 ```
 >>> from watex.core.erp import ERP
 ```
	-  To get he *anomaly power * 
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	