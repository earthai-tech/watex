
.. _datasets:

================
Datasets 
================

.. currentmodule:: watex.datasets

:mod:`~watex.datasets` fetches data from the local machine. If data does not exist, module retrieves it from 
the remote (repository) or using  zenodo record. :code:`watex` implements three (3) kinds of datasets: 

* DC-resistivity datasets (DC-datasets) 
* Learning datasets 
* EDI datasets; EDI stands for Electrical Data Interchange, refer to :mod:`~watex.edi`. 

DC-Datasets 
==================

The DC dataset is divided into two kinds of datasets: The Electrical resistivity profiling (ERP) 
and vertical electrical sounding (VES) datasets [1]_.  

.. _Cote d'Ivoire: https://en.wikipedia.org/wiki/Ivory_Coast 
.. _pandas dataframe: https://pandas.pydata.org/docs/


.. _erp_dataset:

ERP dataset
---------------

Most of the `DC-ERP` data are collected from different survey areas during the National Drinking 
Water Supply Program (PNAEP) occurs in 2014 in `Cote d'Ivoire`_. 

This is an example of the ERP data arrangement table: 

.. table::
   :widths: auto
   :class: longtable
   
   +-----------+-----------+-----------+--------------+     
   |station    | easting   | northing  | resistivity  |             
   +===========+===========+===========+==============+ 
   |0          |382741     |896203     |79            |
   +-----------+-----------+-----------+--------------+ 
   |10         |382743     |896193     |62            |
   +-----------+-----------+-----------+--------------+ 
   |20         |382747     |896184     |51            |
   +-----------+-----------+-----------+--------------+ 
   |...        |...        |...        | ...          |
   +-----------+-----------+-----------+--------------+          
   |980        |382705     |894887     |55            |
   +-----------+-----------+-----------+--------------+ 
   |990        |382704     |895879     |58            |
   +-----------+-----------+-----------+--------------+     
 
All the DC-ERP datasets hold the following parameters: 

* `kind` : str , ['ves'|'erp'], default is {'erp'}. 
   The kind of DC data to retrieve. If `kind` is set to ``ves`` , VES data is fetched and ERP otherwise. Note that this is only valid for Gbalo locality (:func:`~watex.datasets.dload.load_gbalo`).
* `tag, data_names`: ``None``, Always None for API consistency.
* `as_frame` : bool, default=False. 
   If True, the data is a `pandas dataframe`_ including columns with appropriate types (numeric). The target is a pandas Data frame or Series depending 
   on the number of target columns. If `as_frame` is ``False``,  then returning a :class:`~watex.utils.box.Boxspace`.
* `kws` : dict, Keywords arguments pass to :func:`~watex.utils.coreutils._is_readable` function for parsing data. 
		
There are two localities for DC-ERP datasets : 

* `Tankesse` data fetches using :func:`~watex.datasets.dload.load_tankesse`  
* `Gbalo` data fetches using :func:`~watex.datasets.dload.load_gbalo`

.. topic:: Examples: 

.. code-block:: python 

	>>> from watex.datasets import load_tankesse, load_gbalo 
	>>> load_tankesse ().resistivity.max()
	224
	>>>  # To get the max station of the survey area 
	>>> load_gablo().station.max()  # in meter 
	440.0
	
.. note:: 

	The array configuration  during the PNEAP is Schlumberger and the max depth investigation is 
	in meters for :math:`AB/2` (current electrodes). The profiling step :math:`AB/2` and  :math:`MN/2` 
	(potential electrodes)  are fixed to meters [2]_. The `easting`, and `northing` are in meters and 
	`resistivity` columns are in :math:`\Omega.m` as apparent resistivity values. Furthermore, if the 
	UTM coordinates (easting and northing) data is given as well as the UTM_zone, the latitude and longitude 
	data are auto-computed and vice versa. The user does need to provide both coordinates data types
	( UTM or DD:MM.SS)

To ascertain whether the data is acceptable,  it is better to reverify the arrangement using the function 
:func:`~watex.utils.coreutils.erpSelector` for data validation. 

.. _ves_dataset:

VES dataset 
-------------------

Most of the `DC-VES` data are also collected from different survey areas during the PNAEP program. 
The following table gives an illustration of the standard data arrangement: 

.. table::
   :widths: auto
   :class: longtable

   +------+--------+----------+----------+------------+
   |AB/2  |  MN/2  |     SE1  |    SE2   |      SE... |	
   +======+========+==========+==========+============+
   |1     |0.4     |107       |93        |75          |
   +------+--------+----------+----------+------------+
   |2     |0.4     |97        |91        |49          |
   +------+--------+----------+----------+------------+
   | ...  |  ...   |   ...    |   ...    |     ...    |
   +------+--------+----------+----------+------------+
   |100   |10      |79        |96        |98          |
   +------+--------+----------+----------+------------+
   |110   |10      |84        |104       |104         |
   +------+--------+----------+----------+------------+

where :math:`AB/2`,  :math:`MN/2` and :math:`SE` are the depth measurement of the current electrodes AB, 
the spacing of the potential electrodes, and the sounding resistivity values in :math:`\Omega.m` [3]_. 
Note that many sounding data (`SE`) can be collected in the survey area. For simplifying purposes :math:`AB/2` 
and  :math:`MN/2` are kept in VES frame as :math:`AB` and  :math:`MN` respectively whereas :math:`SE` is 
renamed to :math:`resistivity`.  

The following table gives the true sanitized arrangement acceptable for all functions and 
methods that use the VES data: 

.. table::
   :widths: auto
   :class: longtable
   
   +-----------+-----------+-------------+-------------+----------------+----------------+     
   |AB         | MN        | resistivity | resistivity | resistivity    |   ...          |      
   +===========+===========+=============+=============+================+================+    
   |1          |0.4        |107          |93           | 75             |   ...          |
   +-----------+-----------+-------------+-------------+----------------+----------------+ 
   |2          |0.4        |97           |91           | 49             |   ...          |   
   +-----------+-----------+-------------+-------------+----------------+----------------+ 
   |...        |...        |...          | ...         | ...   	        |   ...          |
   +-----------+-----------+-------------+-------------+----------------+----------------+ 
   |100        |10         |79           |96           |98              |   ...          |
   +-----------+-----------+-------------+-------------+----------------+----------------+  
   |110        |10         |84           |104          |104             |   ...          |
   +-----------+-----------+-------------+-------------+----------------+----------------+ 

The following parameters are passed to the VES data to retrieve the expected data:

* `tag, data_names`: ``None`` , Always None for API consistency 
* `as_frame` : bool, default=False. 
   If True, the data is a pandas DataFrame including columns with appropriate types (numeric). The target is a panda data frame or Series depending on the number of target columns. If `as_frame` is False, then returning a :class:`~watex.utils.Boxspace` dictionary-like object.
* `index_rhoa`: int, default=0. 
   Index of the resistivity columns to retrieve. Note that this is useful in cases many sounding values are collected in the same survey area. `index_rhoa=0` fetches the first sounding values in the collection of all values. For instance `index_rhoa=0` in the raw arrangement above fetches the sounding data `SE1` i.e the first resistivity column. 
* `kws`: dict, Keywords arguments pass to :func:`~watex.utils.coreutils._is_readable` function for parsing data. 
 
There are three localities for DC-VES datasets: 

* `Gbalo` data fetches using :func:`~watex.datasets.dload.load_gbalo`  by passing argument ``ves`` to parameter `kind`. 
* `Boundiali` data fetches using :func:`~watex.datasets.dload.load_boundiali`
* `Semien` data fetches using :func:`~watex.datasets.dload.load_semien`
	
If the raw arrangement (above ) is given, it is better to reverify the arrangement using the function :func:`~watex.utils.coreutils.vesSelector` for data validation. 

.. topic:: Examples: 

.. code-block:: python 

	>>> from watex.datasets import load_boundiali , load_gbalo 
	>>> load_boundiali ().head(3) 
	   AB   MN  resistivity
	0   1  0.4          107
	1   2  0.4           97
	2   3  0.4           69
	>>> load_boundiali (index_rhoa =2 ).head(3) # third sounding data 
	   AB   MN  resistivity
	0   1  0.4           75
	1   2  0.4           49
	2   3  0.4           44
	>>> load_gbalo (kind ='ves').AB.max () # max AB/2 depth 
	100

.. note:: 
	The array configuration is Schlumberger and the max depth investigation is 
	100 meters for :math:`AB/2` (current electrodes). The  profiling step
	:math:`AB` is fixed to 100  meters whereas :math:`MN/2`  also fixed to
	(potential electrodes) to 10 meters. `station` , `easting` and `northing` are in meters and 
	`resistivity` columns are in :math:`\Omega.m` as apparent resistivity values.  
	
	
Learning Dataset
===================

The learning datasets are the data ready for predictions where the features are already precomputed.  
An example is the most popular dataset:func:`~watex.datasets.iris`. The famous example of :code:`watex` 
datasets in the Bagoue datasets. See :func:`~watex.datasets.dload.load_bagoue` for parameter definitions. 
The second sample of learning datasets is the hydrogeological dataset. The latter is composed of geology, 
boreholes, and logging data. Refer to :func:`~watex.datasets.dload.load_hlogs` for parameter explanations. 

.. topic:: Examples

.. code-block:: python 

	>>> from watex.datasets import load_bagoue
	>>> d = load_bagoue () 
	>>> d.target[[10, 25, 50]]
	array([0, 2, 0])
	>>> list(d.target_names)
	['flow']   
	>>> from watex.datasets import load_iris
	>>> data = load_iris()
	>>> data.target[[10, 25, 50]]
	array([0, 0, 1])
	>>> list(data.target_names)
	['setosa', 'versicolor', 'virginica']
	>>> from watex.datasets.dload import load_hlogs 
	>>> b= load_hlogs()
	>>> b.target_names 
	['aquifer_group',
	 'pumping_level',
	 'aquifer_thickness',
	 'hole_depth',
	 'pumping_depth',
	 'section_aperture',
	 'k',
	 'kp',
	 'r',
	 'rp',
	 'remark']
	>>> # Let's say we are interested of the targets 'pumping_level' and 
	>>> # 'aquifer_thickness' and returns `y' 
	>>> _, y = load_hlogs (as_frame=True, # return as frame X and y
						   tnames =['pumping_level','aquifer_thickness'], 
						   )
	>>> list(y.columns)
	['pumping_level', 'aquifer_thickness'] 
	
.. _em_dataset:	

EDI dataset 
===============

SEG-EDI dataset is a collection of edi-objects from :class:`~watex.edi.Edi`. Data can be restored using the 
:func:`~watex.datasets.dload.load_edis`. Refer to the function (:func:`~watex.datasets.dload.load_edis`.) 
parameters explanation for further details. 

.. topic:: Examples: 

.. code-block:: python 

	>>> from watex.datasets.dload import load_edis 
	>>> load_edis ().frame [:3]
					edi
	0  Edi( verbose=0 )
	1  Edi( verbose=0 )
	2  Edi( verbose=0 )
	>>> load_edis (as_frame =True, key='longitude latitude', samples = 7) 
		latitude   longitude
	0  26.051390  110.485833
	1  26.051794  110.486153
	2  26.052198  110.486473
	3  26.052602  110.486793
	4  26.053006  110.487113
	5  26.053410  110.487433
	6  26.053815  110.487753
	
	
Boilerplate function : :func:`~watex.datasets.fetch_data`
=========================================================== 

The boilerplate function :func:`~watex.datasets.fetch_data` accepts as `tag` argument the area 
name of all sampling datasets implemented in :mod:`~watex.datasets` and returns the return values of 
each dataset. However, there is a special case when using :func:`~watex.datasets.fetch_data` for the 
Bagoue area [4]_. Indeed, the later dataset gives multiple stages of data processing. To fetch any stage 
of the data processing, the area name must be following by the processing stage name. For instance, fetching 
the `analysed` data for PCA analysis, the `tag` should be ``Bagoue analyzed`` rather than ``Bagoue``. Refer 
to the function parameters explanation for further details as well as the processing stages [5]_. If the only 
name is given, the :func:`~watex.datasets.load_bagoue` should be enabled and will output the return accordingly. 
See the demonstration below to fetch some processing stages of `Bagoue datasets`. 
 
.. topic:: Examples: 

.. code-block:: python 

	>>> from watex.datasets import fetch_data 
	>>> fetch_data ('gbalo').head (3) 
	   station  resistivity  longitude  latitude  easting   northing
	0      0.0         1101        0.0       0.0   790752  1092750.0
	1     10.0         1147        0.0       0.0   790747  1092758.0
	2     20.0         1345        0.0       0.0   790743  1092763.0
	>>> fetch_data ('semien', index_rhoa=1).head (3) 
	   AB   MN  resistivity
	0   1  0.4           70
	1   2  0.4           82
	2   3  0.4           89
	>>> h = fetch_data ('hlogs') 
	>>> h.frame.columns[:7] 
	Index(['hole_id', 'depth_top', 'depth_bottom', 'strata_name', 'rock_name',
       'layer_thickness', 'resistivity'],
       dtype='object')
	>>> e= fetch_data ('edis', samples =7 , return_data = True) 
	array([Edi( verbose=0 ), Edi( verbose=0 ), Edi( verbose=0 ),
       Edi( verbose=0 ), Edi( verbose=0 ), Edi( verbose=0 ),
       Edi( verbose=0 )], dtype=object)
	>>> b = fetch_data('bagoue' ) # no suffix returns  object
	>>> b.tnames 
	array(['flow'], dtype='<U4')
	>>> b.feature_names 
    ['num',
	 'name',
	 'east',
	 'north',
	 'power',
	 'magnitude',
	 'shape',
	 'type',
	 'sfi',
	 'ohmS',
	 'lwi',
	 'geol']
	>>> X, y = fetch_data('bagoue prepared' ) # prepared staged 
	>>> X # is transformed  # ready for prediction 
	>>> X[0] 
	<1x18 sparse matrix of type '<class 'numpy.float64'>'
		with 8 stored elements in Compressed Sparse Row format>
	>>> y
	array([2, 1, 2, 2, 1, 0, ... , 3, 2, 3, 3, 2], dtype=int64)
	>>> fetch_data('bagoue pipe' ) # fetch the pipeline for Bagoue data processing 
	FeatureUnion(transformer_list=[('num_pipeline',
                                Pipeline(steps=[('selectorObj',
                                                 DataFrameSelector(attribute_names=['power', 'magnitude', 'sfi', 'ohmS', 'lwi'])),
                                                ('imputerObj',
                                                 SimpleImputer(strategy='median',
                                                               verbose='deprecated')),
                                                ('scalerObj',
                                                 StandardScaler())])),
                               ('cat_pipeline',
                                Pipeline(steps=[('selectorObj',
                                                 DataFrameSelector(attribute_names=['shape', 'type', 'geol'])),
                                                ('OneHotEncoder',
                                                 OneHotEncoder())]))])
												 

Generate ERP or VES data 
============================

ERP and VES data can be generated using the function :func:`~watex.datasets.gdata.make_erp` 
and :func:`~watex.datasets.gdata.make_ves` respectively. Check the function parameters for 
further details. The following code snippets gives an example of generating ERP and VES data: 

.. code-block:: python 

    >>> from watex.datasets import make_erp, make_ves 
    >>> erp_data = make_erp (n_stations =50 , step =30  , as_frame =True)
    >>> erp_data.head(3)
    Out[256]: 
       station  longitude  latitude        easting    northing  resistivity
    0        0 -13.488511  0.000997  668210.580864  110.183287   225.265306
    1       30 -13.488511  0.000997  668210.581109  110.183482   327.204082
    2       60 -13.488510  0.000997  668210.581355  110.183676   204.877551
    >>> b = make_ves (samples =50 , order ='+') # 50 measurements in deeper 
    >>> b.resistivity [:-7]
    Out[314]: 
    array([429.873 , 434.255 , 438.5707, 442.8203, 447.0042, 451.1228,
           457.5775])
    >>> b.frame.head(3)  
    Out[315]: 
        AB   MN  resistivity
    0  1.0  0.6   429.872999
    1  2.0  0.6   434.255018
    2  3.0  0.6   438.570675


.. topic:: References 

    .. [1] Kouadio, K.L., Nicolas, K.L., Binbin, M., Déguine, G.S.P. & Serge, K.K. (2021). Bagoue dataset-Cote d’Ivoire: Electrical 
		profiling,electrical sounding and boreholes data, Zenodo. https://zenodo.org/record/5560937
    
    .. [2] Koefoed, O. (1970). A fast method for determining the layer distribution 
        from the raised kernel function in geoelectrical sounding. Geophysical
        Prospecting, 18(4), 564–570. https://doi.org/10.1111/j.1365-2478.1970.tb02129.x
         
    .. [3] Koefoed, O. (1976). Progress in the Direct Interpretation of Resistivity 
        Soundings: an Algorithm. Geophysical Prospecting, 24(2), 233–240.
        https://doi.org/10.1111/j.1365-2478.1976.tb00921.x
		
    .. [4] Kouadio, K.L., Kouame, L.N., Drissa, C., Mi, B., Kouamelan, K.S., Gnoleba, S.P.D., Zhang, H., et al. (2022) Groundwater Flow Rate 
		Prediction from GeoElectrical Features using Support Vector Machines. Water Resour. Res.  https://doi.org/10.1029/2021wr031623
        
    .. [5] Biemi, J. (1992). Contribution à l’étude géologique, hydrogéologique et par télédétection
        de bassins versants subsaheliens du socle précambrien d’Afrique de l’Ouest:
        hydrostructurale hydrodynamique, hydrochimie et isotopie des aquifères discontinus
        de sillons et aires gran. In Thèse de Doctorat (IOS journa, p. 493). Abidjan, Cote d'Ivoire
		
		
