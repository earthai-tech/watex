
.. _cases:

================
Case Histories  
================

.. currentmodule:: watex.cases

:mod:`~watex.cases` implements functions and modules already available and ready to use 
for solving real engineering problems like flow rate predictions and boosting using the 
bases learners, SVC, and ensemble paradigms. :mod:`watex.cases.features`, :mod:`watex.cases.processing`, 
:mod:`watex.cases.modeling` and :mod:`watex.cases.prepare` modules have base steps and can 
be used for processing and analyses to give a quick depiction of how data looks like. This 
can figure out the next processing steps for solving the evidence problem.

Features 
============

:mod:`~watex.cases.features` is a set of different manipulation that can be performed on the 
case history feature data. 


GeoFeatures
------------
:class:`~watex.cases.features.GeoFeatures` expects the geological, the boreholes and DC-electrical 
resistivity data. :class:`~watex.cases.features.GeoFeatures` set all feature values of 
different investigation sites. `GeoFeatures` class is  composed of: 

* `erp` class  get from :class:`~watex.methods.erp.ERPCollection`
* `geol`  obtained from :class:`~watex.geology.geology.Geology` 
* `boreh` get from :class:`~watex.geology.geology.Borehole` 

.. topic:: Examples: 

.. code-block:: python 

    >>> from watex.cases.features import GeoFeatures 
    >>> data ='data/geodata/main.bagciv.data.csv' 
    >>> featObj =GeoFeatures().fit(data ) 
    >>> featObj.id_
    array(['e0000001', 'e0000002', 'e0000003', 'e0000004', 'e0000005',
           'e0000006', 'e0000007'], dtype='<U8')
    >>> featObj.site_names_[:7] # view the site for borehole 
    array(['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'], dtype=object)
	

Feature Inspection
--------------------
:class:`~watex.cases.features.FeatureInspection` summarizes flow features. It deals with
data feature categorization. When numerical values are provided standard `qualitative` or 
`quantitative`  analysis is performed. 
	
.. code-block:: python 

	>>> from watex.cases.features import FeatureInspection
	>>> data = 'data/geodata/main.bagciv.data.csv'
	>>> fobj = FeatureInspection().fit(data) 
	>>> fobj.data_.columns
	Index(['num', 'name', 'east', 'north', 'power', 'magnitude', 'shape', 'type',
		   'sfi', 'ohmS', 'lwi', 'geol', 'flow'],
		  dtype='object')

Prepare 
========

:mod:`~watex.cases.prepare` base module helps to automate data preparation at once. It is created fast 
data preparation in real engineering cases study. This is a naive approach for quickly reproducing the 
published paperwork, especially for flow rate prediction. 

Base data preparation for case studies
-----------------------------------------

The base step has been used to solve flow rate prediction problems [1]_. Its steps procedure 
can straightforwardly help users to fast reach the same goal as the published paper. An example 
of a different kind of Bagoue dataset [2]_, is prepared using the `BaseSteps` module. 

.. topic:: References 

	.. [1] Kouadio, K.L., Kouame, L.N., Drissa, C., Mi, B., Kouamelan, K.S., 
		Gnoleba, S.P.D., Zhang, H., et al. (2022) Groundwater Flow Rate 
		Prediction from GeoElectrical Features using Support Vector Machines. 
		Water Resour. Res. :doi:`10.1029/2021wr031623`
		
	.. [2] Kouadio, K.L., Nicolas, K.L., Binbin, M., Déguine, G.S.P. & Serge, 
		K.K. (2021, October) Bagoue dataset-Cote d’Ivoire: Electrical profiling,
		electrical sounding and boreholes data, Zenodo. :doi:`10.5281/zenodo.5560937`

.. seealso:: 

	An example of Bagoue dataset preparation in the :mod:`~watex.datasets._p` module. 
	
	
Processing 
=============

:mod:`~watex.cases.processing` gives basic processing for achieving results. Here, we implement the 
processing step performed for predicting the flow rate prediction [1]_. 

Preprocessing 
----------------
:class:`~watex.cases.processing.Preprocessing` gives the prior steps for flow rate prediction. 

.. note:: 

	If :math:`X` and :math:`y` are provided, they are considered as a feature set
	and target respectively. They should be split into the training set 
	and test set respectively.
	
.. code-block:: python 

	>>> from watex.cases.processing import Preprocessing 
	>>> from watex.datasets import fetch_data 
	>>> data = fetch_data('bagoue original').get('data=dfy2')
	>>> pc = Preprocessing (drop_features = ['lwi', 'num', 'name']
							).fit(data =data )
	>>> len(pc.X ),  len(y), len(pc.Xt ),  len(pc.yt)
	(344, 344, 87, 87) # trainset (X,y) and testset (Xt, yt)
	
One can assemble pipes and an estimator to make a model (default) following the snippet code 
below. Indeed, the model is composed of transformers and estimators. If one is set to `None`, 
it uses the default pipe and estimator which might be not the one expected. Therefore providing 
a pipe and estimator is recommended.

.. topic:: Examples: 

* We can get the default preprocessor by merely calling: 

.. code-block:: python 

	>>> from watex.cases.processing import Preprocessing 
	>>> pc = Preprocessing (tname = 'flow', drop_features =['lwi', 'name', 'num'])
	>>> data = fetch_data ('bagoue original').get('data=dfy2')
	>>> pc.fit(data =data) 
	>>> pc.makeModel() # use default model and preprocessor 
	>>> pc.model_ 
	Pipeline(steps=[('preprocessor',
			 ColumnTransformer(transformers=[('numpipe',
					Pipeline(steps=[('imputer',
							   SimpleImputer()),
							('polynomialfeatures',
										   PolynomialFeatures(degree=10,
															  include_bias=False)),
										  ('selectors',
										   SelectKBest(k=4)),
										  ('scalers',
										   RobustScaler())]),
						  ['east', 'north', 'power',
						   'magnitude', 'sfi',
						   'ohmS']),
						 ('catpipe',
						  Pipeline(steps=[('imputer',
										   SimpleImputer()),
										  ('onehotencoder',
										   OneHotEncoder(handle_unknown='ignore'))]),
						  ['type', 'shape', 'geol'])])),
					('SVC', SVC(C=100, gamma=0.001, random_state=42))])
                 
* Or build your preprocessor object using the example below: 

.. code-block:: python 

	>>> from watex.exlib.sklearn import ( 
		Pipeline, 
		ColumnTransformer,
		SimpleImputer, 
		StandardScaler, 
		OneHotEncoder,
		LogisticRegression
		)
	>>> from watex.datasets import fetch_data 
	>>> from watex.cases.processing import Preprocessing 
	>>> pc = Preprocessing (tname = 'flow', drop_features =['lwi', 'name', 'num'])
	>>> numeric_features = ['east', 'north', 'power', 'magnitude', 'sfi', 'ohmS']
	>>> numeric_transformer = Pipeline(
		steps=[("imputer", SimpleImputer(strategy="median")), 
			   ("scaler", StandardScaler())]
		)
	>>> categorical_features = ['shape', 'geol', 'type']
	>>> categorical_transformer = OneHotEncoder(handle_unknown="ignore")
	>>> preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_transformer, numeric_features),
			("cat", categorical_transformer, categorical_features),
		])
	>>> pc.makeModel (pipe = preprocessor, 
					  estimator =  LogisticRegression())
	>>> # or pc.model_
	Pipeline(steps=[('preprocessor',
					 ColumnTransformer(transformers=[('num',
							  Pipeline(steps=[('imputer',
											   SimpleImputer(strategy='median')),
											  ('scaler',
											   StandardScaler())]),
							  ['east', 'north', 'power',
							   'magnitude', 'sfi',
							   'ohmS']),
							 ('cat',
							  OneHotEncoder(handle_unknown='ignore'),
							  ['shape', 'geol', 'type'])])),
					('LogisticRegression', LogisticRegression())])



Once a model is created,  a dummy baseline model can be evaluated from preprocessing 
pipeline; onto a model by providing an estimator. This is possible thanks to 
:meth:`~watex.cases.processing.Preprocessing.baseEvaluation`.  A code snippet is 
given below: 

.. code-block:: python 

	>>> from watex.cases.processing import Preprocessing 
	>>> pc = Preprocessing (tname = 'flow', drop_features =['lwi', 'name', 'num'])
	>>> data = fetch_data ('bagoue original').get('data=dfy2')
	>>> pc.fit(data =data)
	# (1) -> default estimator 
	>>> pc.baseEvaluation (eval_metric=True)
	0.47126436781609193 # score 
	
.. code-block:: python 

	# (2) -> multiples estimators 
	>>> from watex.exlib.sklearn import RandomForestClassifier , SGDClassifier, SimpleImputer 
	>>> estimators={'RandomForestClassifier':RandomForestClassifier
					(n_estimators=200, random_state=0), 
					'SDGC':SGDClassifier(random_state=0)}
	>>> pc.X= SimpleImputer().fit_transform(pc.X)
	>>> pc.Xt= SimpleImputer().fit_transform(pc.Xt) # remove NaN values 
	>>> pc.BaseEvaluation(estimator={
	 'RandomForestClassifier':RandomForestClassifier(
	    n_estimators=200, random_state=0), 
	  'SDGC':SGDClassifier(random_state=0)}, eval_metric =True)
	>>> pc.ypred_
	{'RandomForestClassifier': array([2, 1, 2, 2, 2, 2, 0, 1, 1, 2, 3, 1, 0, 0, 1, 1, 1, 2, 2, 3, 2, 3,
			1, 2, 1, 2, 0, 2, 2, 3, 2, 2, 1, 1, 3, 3, 0, 2, 3, 3, 2, 1, 0, 2,
			1, 1, 2, 2, 2, 2, 1, 1, 0, 2, 0, 2, 1, 2, 1, 1, 2, 0, 1, 2, 0, 2,
			2, 3, 2, 2, 3, 0, 1, 2, 2, 3, 1, 1, 0, 1, 1, 2, 0, 0, 2, 0, 1],
		   dtype=int8),
	 'SGDClassifier': array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
			3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
			3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
			3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
		   dtype=int8)} 
	>>> pc.BaseEvaluation(estimator={
		 'RandomForestClassifier':RandomForestClassifier(
			n_estimators=200, random_state=0), 
		  'SDGC':SGDClassifier(random_state=0)}, eval_metric =True)
	>>> pc.base_score_
	{'RandomForestClassifier': 0.7816091954022989,
	'SGDClassifier': 0.14942528735632185}
	
When using the estimator "randomForest", the score is a little bit improved to `78%` 
whereas it was `47.13 %` for dummy prediction. 

Processing
----------------
:class:`~watex.cases.processing.Processing` is dedicated to managing baseline model evaluation 
and learning. It also manages the validation curves after fiddling with a few estimator hyperparameters. 

.. topic:: Examples: 

.. code-block:: python 

	>>> from watex.cases.processing  import Processing
	>>> from watex.exlib.sklearn import (StandardScaler,RandomForestClassifier, make_column_selector, PolynomialFeatures, SelectKBest, f_classif)  
	>>> data = fetch_data ('bagoue original').get('data=dfy2')
	>>> my_own_pipeline= {'num_column_selector_': 
	...                       make_column_selector(dtype_include=np.number),
	...                'cat_column_selector_': 
	...                    make_column_selector(dtype_exclude=np.number),
	...                'features_engineering_':
	...                    PolynomialFeatures(3,include_bias=True),
	...                'selectors_': SelectKBest(f_classif, k=4), 
	...               'encodages_': StandardScaler()
	...                 }
	>>> my_estimator={
	...    'RandomForestClassifier':RandomForestClassifier(
	...    n_estimators=200, random_state=0)
	...    }
	>>> processObj= Processing (tname = 'flow', drop_features =['lwi', 'name', 'num'], pipeline= my_own_pipeline, estimator=my_estimator)  
	>>> processObj.fit(data=data )
	>>> processObj.baseEvaluation (eval_metric=True ) 
	0.4942528735632184 # score is an ensemble score for both model 
	>>> processObj.get_validation_curve (switch_plot='on', val_params= {'param_name': "n_estimators", "param_range": np.arange (1, 20, 5), "scoring": 'neg_mean_squared_error'} ) 
	
	
	
