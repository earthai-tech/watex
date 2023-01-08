.. _view:

================
View 
================

.. currentmodule:: watex.view

:mod:`~watex.view` is dedicated for visualization purposes data from the local machine. The module deals with the parameters 
and processing spaces  and yields multiples plots for data exploration, features analysis, features discussion, 
tensor recovery, model inspection, and evaluation.  :mod:`~watex.view`  is divided in two sub-modules: 
* :mod:`~watex.view.plot` for handling the params space plots via  :class:`~watex.view.ExPlot`, :class:`~watex.view.QuickPlot`, and :class:`~watex.view.TPlot`. 
* :mod:`~watex.view.mlplot` for handling the processing space plot through the :class:`~watex.view.EvalPlot` as well as many other functions. 

All the classes implemented in :mod:`~watex.view` module from :class:`~watex.property.Baseplots` ABC (Abstract Base Class) objects. 
All arguments from this class can be used for customize the plots. Refer to :class:`~watex.property.Baseplots` to know the 
acceptables attributes for that class for plot customizing. 

Furthermore, note the existence of the `tname` and `pkg` parameters passed mostly to the :mod:`~watex.view`  module classes.
* `tname`: always str, is  the target name or label. In supervised learning the target name is considered as the reference name of :math:`y` or label variable:  
* `pkg`: always str, Optional by default, is the kind or library to use for visualization. can be ['yb'|'msn'|'sns'|'pd']  for 'yellowbrick'[1]_ , 'missingno', 'seaborn' or 'pandas' respectively. 
	Mosyly the default value for `pkg` is ``pd`` or ``sns``.  To install these packages, use ``pip`` or ``conda``. Note that the `pkg` parameter is specific for each plotting method, is not passed to `__init__` method. 
	
Additional to that, each module plots has some specific parameters passed to ``__init__`` methods. Refer to each plot class documentation.   


Params space plots  
====================
The `params space` plots is ensured by the modules :class:`~watex.view.ExPlot`, :class:`~watex.view.QuickPlot`, for 
data exploratory, data analysis and quick visualization,  and tensors plots for EM recovery signals. 


`Exploratory plots` 
--------------------

`ExPlot` ( :class:`~watex.view.plot.ExPlot` ) is a shadow class and explore data to create a model since 
it gives a feel for the data and also at great excuses to meet and discuss issues with business units that controls the data. Moreover 
all `ExPlot` methods return an instancied object that inherits from :class:`~watex.property.Baseplots` for visualization. Simply, `ExPlot` 
can be called as:: 
	>>> from watex.view import ExPlot # for short calling 
	>>> from watex.view.plot import ExPlot 
	
The costumizing attributes can be passed as keywords arguments before the `fit` method (:meth:`~watex.view.ExPlot.fit` as :: 

	>>> plot_kws =dict (fig_size=(12, 7), xlabel="Label for X", ylabel= "something for Y", ... ) 
	>>> ExPlot (**plot_kws) 

Here, are some examples of plots can be inferred the :class:`~watex.view.plot.ExPlot`. Refer to :class:`~watex.view.plot.ExPlot` for 
acceptable additional parameters. 

Plot parallel coordinates: :meth:`~watex.view.ExPlot.plotparallelcoords` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:meth:`~watex.view.ExPlot.plotparallelcoords` uses parallel coordinates in multivariates for clustering visualization. For examples:

.. code-block:: python 

	>>> from watex.datasets import fetch_data 
	>>> from watex.view import ExPlot 
	>>> data =fetch_data('original data').get('data=dfy1')
	>>> p = ExPlot (tname ='flow', fig_size =(7, 5)).fit(data)
	>>> p.plotparallelcoords(pkg='yb')


.. figure:: ../examples/auto_examples/view_explot_parallel_coordinates.png
   :target: ../examples/auto_examples/view_explot_parallel_coordinates.html
   :align: center
   :scale: 60%
   
Plot Radial: :meth:`~watex.view.ExPlot.radviz` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~watex.view.ExPlot.radviz`  shows each sample on circle or square, with features on the circonference to 
vizualize separately between target. Values are normalized and each figure has a spring that pulls samples 
to it based on the value. Here is an example: 

.. code-block:: python 
	
	>>> from watex.datasets import fetch_data 
	>>> from watex.view import ExPlot 
	>>> # visualization using the yellowbrick package 
	>>> data0 = fetch_data('bagoue original').get('data=dfy1')
	>>> p = ExPlot(tname ='flow').fit(data0)
	>>> p.plotradviz(classes= [0, 1, 2, 3] ) # can set to None 
	>>> # visualization using the pandas 
	>>> p.plotradviz(classes= [0, 1, 2, 3], pkg='yb' )
	
.. |pd_rv| image:: ../examples/auto_examples/view_explot_plot_rad_viz_pd.png
   :target: ../examples/auto_examples/view_explot_plot_rad_viz_pd.html
   :scale: 50%

.. |yb_rd| image:: ../examples/auto_examples/view_explot_plot_rad_viz_yb.png
   :target: ../examples/auto_examples/view_explot_plot_rad_viz_yb.html
   :scale: 50%
	
* **Radial Visualization: using pandas and yellowbrick plots** 

  ==================================== ====================================
  RadViz with Pandas 	              		RadViz with yellowbrick  
  ==================================== ====================================
  |pd_rv|                         		|yb_rd|
  ==================================== ====================================
  
Plot Pairwise comparison: :meth:`~watex.view.ExPlot.plotpairwisecomparison` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~watex.view.ExPlot.plotpairwisecomparison` creates pairwise comparison between features. It shows 
a ['pearson'|'spearman'|'covariance'] correlation. Here is a example of code snippet:

.. code-block:: python 

    >>> from watex.datasets import fetch_data 
	>>> from watex.view import ExPlot 
	>>> data = fetch_data ('bagoue original').get('data=dfy1') 
	>>> p= ExPlot(tname='flow', fig_size=(7, 5)).fit(data)
	>>> p.plotpairwisecomparison(fmt='.2f', corr='spearman', pkg ='yb',
								 annot=True, 
								 cmap='RdBu_r', 
								 vmin=-1, 
								 vmax=1 )
								 
The following outputs is given below:

.. figure:: ../examples/auto_examples/view_explot_plot_pairwise_comparison.png
   :target: ../examples/auto_examples/view_explot_plot_pairwise_comparison.html
   :align: center
   :scale: 70%
   
Plot Categorical Features Comparison :meth:`~watex.view.ExPlot.plotcutcomparison` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~watex.view.ExPlot.plotcutcomparison` compares the quantile values of ordinal categories. It simulates that the the bining of `xname` into a `q` 
quantiles, and `yname` into `bins`. Plot is normalized so its fills all the vertical area which makes easy to see that in the `4*q %` 
quantiles. Note that `xname` and `yname` are vectors or keys in data variables that specify positions on the `x` and `y` axes. Both 
are the column names to consider. Should be items in the dataframe columns. Raise an error if elements do not exist. Here is an example of 
`sfi` (:func:`~watex.utils.exmath.sfi`) and `ohmS` (:func:`~watex.utils.exmath.ohmicArea`):   

.. code-block:: python 

	>>> from watex.datasets import fetch_data 
	>>> from watex.view import ExPlot 
	>>> data = fetch_data ('bagoue original').get('data=dfy1') 
	>>> p= ExPlot(tname='flow', fig_size =(7, 5) ).fit(data)
	>>> p.plotcutcomparison(xname ='sfi', yname='ohmS') # compare 'sfi' and 'ohmS'
	<'ExPlot':xname='sfi', yname='ohmS' , tname='flow'>

.. figure:: ../examples/auto_examples/view_explot_plot_cat_comparison.png
   :target: ../examples/auto_examples/view_explot_plot_cat_comparison.html
   :align: center
   :scale: 60%
   
Plot Box: :meth:`~watex.view.ExPlot.plotbv` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 

:meth:`~watex.view.ExPlot.plotbv` allows the distributions visualization using the box, boxen or violin plots. The choice of the 
box is passed to the parameter `kind`. See :meth:`~watex.view.ExPlot.plotbv` documentation for further details. A basic example 
using the 'violin' plot is given below: 

.. code-block:: python 

	>>> from watex.datasets import fetch_data 
	>>> from watex.view import ExPlot 
	>>> data = fetch_data ('bagoue original').get('data=dfy1') 
	>>> p= ExPlot(tname='flow', fig_size =(7, 5)).fit(data)
	>>> p.plotbv(xname='flow', yname='ohmS', kind='violin')
	<'ExPlot':xname='flow', yname='ohmS' , tname='flow'>
  
Here is the given output: 

.. figure:: ../examples/auto_examples/view_explot_plot_bv_violin.png
   :target: ../examples/auto_examples/view_explot_plot_bv_violin.html
   :align: center
   :scale: 60%
   
Plot Grid in Pair: :meth:`~watex.view.ExPlot.plotpairgrid`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~watex.view.ExPlot.plotpairgrid` creates a pair grid. Plot is a matrix of columns and kernel density estimations. 
To color by a columns from a dataframe, use `hue` parameter. Refer to :meth:`~watex.view.ExPlot.plotpairgrid` for more details. 
Here is a basic example: 

.. code-block:: python 

	>>> from watex.datasets import fetch_data 
	>>> from watex.view import ExPlot 
	>>> data = fetch_data ('bagoue original').get('data=dfy1') 
	>>> p= ExPlot(tname='flow', fig_size =(7, 5)).fit(data)
	>>> p.plotpairgrid (vars = ['magnitude', 'power'] )
	<'ExPlot':xname=None, yname=None , tname='flow'>

.. figure:: ../examples/auto_examples/view_explot_plot_pair_grid.png
   :target: ../examples/auto_examples/view_explot_plot_pair_grid.html
   :align: center
   :scale: 70%   
   
Plot Joint features: :meth:`~watex.view.ExPlot.plotjoint`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~watex.view.ExPlot.plotjoint` is a fancier scatterplot that includes histogram on the edge as well as a 
regression line called a `joinplot`. Here is an example: 

.. code-block:: python 

	>>> from watex.datasets import fetch_data 
	>>> from watex.view import ExPlot 
	>>> data = fetch_data ('bagoue original').get('data=dfy1') 
	>>> p= ExPlot(tname='flow', fig_size =(7, 5)).fit(data)
	>>> p.plotjoint(xname ='type', yname='shape', corr='spearman',  pkg ='yb')
	<'ExPlot':xname='power', yname='shape' , tname='flow'>

.. figure:: ../examples/auto_examples/view_explot_plot_joint.png
   :target: ../examples/auto_examples/view_explot_plot_joint.html
   :align: center
   :scale: 60%     
   
There several options to customize the jointplot passed to :meth:`~watex.view.ExPlot.plotjoint`.  
   
.. seealso:: 
	:meth:`~watex.view.plot.QuickPlot.joint2features` 
	
Plot Scatter: :meth:`~watex.view.ExPlot.plotscatter`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~watex.view.ExPlot.plotscatter` plot numerical features and shows the relationship between two numeric columns. 


.. code-block:: python 

	>>> from watex.view import ExPlot 
	>>> ExPlot(tname='flow', fig_size =(7, 5)).fit(data).plotscatter (xname ='sfi', yname='ohmS' )
	<'ExPlot':xname='sfi', yname='ohmS' , tname='flow'>

The above code generates the following output: 

.. figure:: ../examples/auto_examples/view_explot_plot_scatter.png
   :target: ../examples/auto_examples/view_explot_plot_scatter.html
   :align: center
   :scale: 70%     

Plot Histogram - Features vs Target: :meth:`~watex.view.ExPlot.plothistvstarget`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~watex.view.ExPlot.plothistvstarget` plots a histogram of continuous values against the target of binary plot. Here is a base implementation: 

.. code-block:: python 
	
	>>> from watex.utils import read_data 
	>>> from watex.view import ExPlot
	>>> data = read_data  ( 'data/geodata/main.bagciv.data.csv' ) 
	>>> p = ExPlot(tname ='flow').fit(data)
	>>> p.fig_size = (7, 5)
	>>> p.savefig ='bbox.png'
	>>> p.plothistvstarget (xname= 'sfi', c = 0, kind = 'binarize',  kde=True, 
					  posilabel='dried borehole (m3/h)',
					  neglabel = 'accept. boreholes'
					  )
	<'ExPlot':xname='sfi', yname=None , tname='flow'>

Here is the example output: 

.. figure:: ../examples/auto_examples/view_explot_plot_histvstarget.png
   :target: ../examples/auto_examples/view_explot_plot_histvstarget.html
   :align: center
   :scale: 70%     

.. seealso:: 
	
	:meth:`~watex.view.ExPlot.plothist` for a pure histogram visualization 
	
Plot Missing: :meth:`~watex.view.ExPlot.plotmissing`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~watex.view.ExPlot.plotmissing` helps vizualizing patterns in the missing data. Here are some examples: 

.. code-block:: python 

	>>> from watex.utils import read_data
	>>> from watex.view import ExPlot
	>>> data = read_data ('data/geodata/main.bagciv.data.csv' ) 
	>>> p = ExPlot().fit(data)
	>>> p.fig_size = (7, 5)
	>>> p.plotmissing(kind ='dendrogram')
	<'ExPlot':xname=None, yname=None , tname=None>
	
The following outputs gives three kind of missing data visualization.The first outputs the base missing pattern:
 
.. figure:: ../examples/auto_examples/view_explot_plot_missing_mpattern.png
   :target: ../examples/auto_examples/view_explot_plot_missing_mpattern.html
   :align: center
   :scale: 70%

The remains two gives another representations of missing data. 

.. |dendro_mss| image:: ../examples/auto_examples/view_explot_plot_missing_dendrogram.png
   :target: ../examples/auto_examples/view_explot_plot_missing_dendrogram.html
   :scale: 60%
   
.. |corr_mss| image:: ../examples/auto_examples/view_explot_plot_missing_corr.png
   :target: ../examples/auto_examples/view_explot_plot_missing_corr.html
   :scale: 60%
	
* **Three kind of missing data visualization** 

  ==================================== ====================================
  Dendrogram missing patterns 	              Correlation missing patterns  
  ==================================== ====================================
  |dendro_mss|                         		|corr_mss|
  ==================================== ====================================

.. seealso:: 

	:class:`~watex.base.Missing` for missing data manipulating


`Analysis and Discussing plots` 
-------------------------------------

:class:`~watex.view.QuickPlot` is a special class that deals with analysis modules for quick diagrams, histograms and bar visualization. 
Originally, it was designed for the flow rate (`FR`) prediction during the drinking water supply campaign (DWSC) [2]_. The parameters `mapflow` and `classes` work together and usefull when 
flow data is passed to :meth:`~watex.view.plot.QuickPlot.fit`. Once `mapflow` is set to ``True``, the flow target :math:`y` should be 
turned to categorical values encoded  referring to the type of types of hydraulic system commonly recommended during the DWSC. Mostly 
the hydraulic system is tied to the number of living inhabitants in the survey area [3]_. For instance:: 
	* FR = 0 is for dry boreholes (FR0)
	* 0 < FR ≤ 3m3/h for village hydraulic (≤2000 inhabitants) (FR1)
	* 3 < FR ≤ 6m3/h  for improved village hydraulic(>2000-20 000inhbts) (FR2)
	* 6 <FR ≤ 10m3/h for urban hydraulic (>200 000 inhabitants)(FR3)
    
Note that this flow range passed by default when `mapflow=True` is not exhaustive and can be modified according to the type of hydraulic 
required on each project project. 
Beyond the FR objective as the first motivation design of this class, however, it still prefectly works with any other dataset 
if the appropriate arguments are passed to different methods. In the following, some examples will be displayed to give a visual depiction 
of using the :class:`~watex.view.QuickPlot`  class. 

Plot Naive Target Inspection: :meth:`~watex.view.QuickPlot.naiveviz`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:meth:`~watex.view.QuickPlot.naiveviz` generates a plot to visualize the data using  the existing coordinates `x` and `y`  by considering a special dataframe feature, mostly 
the target :math:`y`. The plot indicates the distribution of the data based on the coordinates positions. Here a demonstration for naive visualization. 
        
.. code-block:: python
 
	>>> import matplotlib.pyplot as plt 
	>>> from watex.transformers import StratifiedWithCategoryAdder
	>>> from watex.view.plot import QuickPlot
	>>> from watex.datasets import load_bagoue 
	>>> df = load_bagoue ().frame
	>>> stratifiedNumObj= StratifiedWithCategoryAdder('flow')
	>>> strat_train_set , *_= \
	...    stratifiedNumObj.fit_transform(X=df) 
	>>> pd_kws ={'alpha': 0.4, 
	...         'label': 'flow m3/h', 
	...         'c':'flow', 
	...         'cmap':plt.get_cmap('jet'), 
	...         'colorbar':True}
	>>> qkObj=QuickPlot(fs=25., fig_size = (7, 5))
	>>> qkObj.fit(strat_train_set)
	>>> qkObj.naiveviz( x= 'east', y='north', **pd_kws)
	Out[103]: QuickPlot(savefig= None, fig_num= 1, fig_size= (7, 5), ... , classes= None, tname= None, mapflow= False)
	
Here is the following output: 

.. figure:: ../examples/auto_examples/view_quickplot_naive_visualization.png
   :target: ../examples/auto_examples/view_quickplot_naive_visualization.html
   :align: center
   :scale: 70%  
   
As a brief interpretation, at a glance, the survey area is dominated by the dried boreholes :math:`FR=0` and the unsustainable 
boreholes ( :math:`0 \leq FR < 2 \quad m^3/hr`). The most productive boreholes  (:math:`FR \geq 4 \quad m^3/hr` ) are located in the 
southeastern part of survey area. 

Plot Feature Discussing: :meth:`~watex.view.QuickPlot.discussingfeatures`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~watex.view.QuickPlot.discussingfeatures` plots feature distributions along the target. For instance, ones can provide the 
features names at least 04 and discuss with their distribution. Here is a basic example of Bagoue dataset (:func:`watex.datasets.load_bagoue`) 

:meth:`~watex.view.QuickPlot.discussingfeatures` maps a dataset onto multiple axes arrayed in a grid of rows and columns that 
correspond to levels of features in the dataset. Here is a snippet codes for a concrete example: 


.. code-block:: python 

	>>> from watex.view.plot import  QuickPlot 
	>>> from watex.datasets import load_bagoue 
	>>> data = load_bagoue ().frame 
	>>> qkObj = QuickPlot(  leg_kws={'loc':'upper right'},
	...          fig_title = '`sfi` vs`ohmS|`geol`',
	...            ) 
	>>> qkObj.tname='flow' # target the DC-flow rate prediction dataset
	>>> qkObj.mapflow=True  # to hold category FR0, FR1 etc..
	>>> qkObj.fit(data) 
	>>> sns_pkws={'aspect':2 , 
	...          "height": 2, 
	...                  }
	>>> map_kws={'edgecolor':"w"}   
	>>> qkObj.discussingfeatures(features =['ohmS', 'sfi','geol', 'flow'],
	...                           map_kws=map_kws,  **sns_pkws
	...                         )
	QuickPlot(savefig= None, fig_num= 1, fig_size= (12, 8), ... , classes= None, tname= flow, mapflow= True)
	
This is the following output: 

.. figure:: ../examples/auto_examples/view_quickplot_discussingfeatures.png
   :target: ../examples/auto_examples/view_quickplot_discussingfeatures.html
   :align: center
   :scale: 50%  
   
At a glance, the figure above shows that most of the drillings carried out on granites have an `FR` of around :math:`1 \quad m^3/hr` 
(:math:`FR1:0< FR \leq 1`). With these kinds of flows, it is obvious that the boreholes will be unproductive 
(unsustainable) within a few years.  However, the volcano-sedimentary schists seem the most suitable geological structure 
with an `FR` greater than :math:`3 \quad m^3/hr`. However, the wide fractures on these formations 
(explained by :math:`ohmS > 1500 \Omega.m^2`) do not mean that they should be more productive since all the drillings performed 
on the wide fracture do not always give a productive `FR` (:math:`FR>3 \quad m^3/hr`) contrary to the narrow fractures 
(around 1000 ohmS). As a result, it is reliable to consider this approach during a future DWSC such as the geology of the 
area and also the rock fracturing ratio computed thanks to the parameters `sfi` and `ohmS`.  


Plot Features Scattering: :meth:`~watex.view.QuickPlot.scatteringfeatures`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~watex.view.QuickPlot.scatteringfeatures` draws a scatter plot with possibility of several semantic features grouping.
Indeed `scatteringfeatures` analysis is a process of understanding how features in a dataset relate to each other and how those 
relationships depend on other features. Visualization can be a core component of this process because, when data are visualized 
properly,the human visual system can see trends and patterns that indicate a relationship. Below is an example of feature scattering 
plots: 

.. code-block:: python 

	>>> from watex.view.plot import  QuickPlot 
	>>> from watex.datasets import load_bagoue 
	>>> data = load_bagoue ().frame
	>>> qkObj = QuickPlot(lc='b', sns_style ='darkgrid', 
	...             fig_title='geol vs level of water inrush (m) ',
	...             xlabel='Level of water inrush (lwi) in meters', 
	...             ylabel='Flow rate in m3/h'
	...            ) 
	>>>
	>>> qkObj.tname='flow' # target the DC-flow rate prediction dataset
	>>> qkObj.mapflow=True  # to hold category FR0, FR1 etc..
	>>> qkObj.fig_size=(7, 5)
	>>> qkObj.fit(data) 
	>>> marker_list= ['o','s','P', 'H']
	>>> markers_dict = {key:mv for key, mv in zip( list (
	...                       dict(qkObj.data ['geol'].value_counts(
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
	>>> qkObj.scatteringfeatures(features=['lwi', 'flow'],
	...                         relplot_kws=regpl_kws,
	...                         **sns_pkws, 
	...                    )
	QuickPlot(savefig= None, fig_num= 1, fig_size= (7, 5), ... , classes= None, tname= flow, mapflow= True)

This is the following output: 

.. figure:: ../examples/auto_examples/view_quickplot_scatteringfeatures.png
   :target: ../examples/auto_examples/view_quickplot_scatteringfeatures.html
   :align: center
   :scale: 70%  
   
In the above figure, ones can notice that the productive `FR` is mostly found between 10 -20  and 40-45 m meters deep. Some borehole 
has the two level of water inrush and mostly found in volcano-sedimentary schists. 

Plot Jointing Features: :meth:`~watex.view.QuickPlot.joint2features`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~watex.view.QuickPlot.joint2features` allows to visualize correlation of two features. It draws a plot of two features 
with bivariate and univariate graphs. Here is an example: 

.. code-block:: python 

	>>> from watex.view.plot import QuickPlot 
	>>> from watex.datasets import load_bagoue 
	>>> data = load_bagoue ().frame
	>>> qkObj = QuickPlot( lc='b', sns_style ='darkgrid', 
	...             fig_title='Quantitative features correlation'
	...             ).fit(data)  
	>>> qkObj.fig_size =(7, 5)
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
	QuickPlot(savefig= None, fig_num= 1, fig_size= (7, 5), ... , classes= None, tname= None, mapflow= False)
        
The above code snippet renders the following output: 

.. figure:: ../examples/auto_examples/view_quickplot_joint2features.png
   :target: ../examples/auto_examples/view_quickplot_joint2features.html
   :align: center
   :scale: 60%  
   
.. seealso:: 
	:meth:`~watex.view.ExPlot.plotjoint` uses additional package like `yellowbrick` [1]_. 

Plot Qualitative Features: :meth:`~watex.view.QuickPlot.numfeatures`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~watex.view.QuickPlot.numfeatures` plots qualitative (numerical) features  distribution using correlative aspect. It only works with 
sure numerical features.  Here is a concrete examples for plotting the numerical distribution according to the target passed to `tname`. For 
this demonstration, we will change a little bit the target name to ohmic-area `ohmS` and only keep the features `['ohmS', 'power', 'lwi', 'flow']` 
in the data. Note, rather than mapping the flow with parameter `mapflow`, we use the function :func:`~watex.utils.funcutils.smart_label_classifier` to 
categorize the numerical `ohmS` values into three classes called `oarea` such as :math:`oarea1 :ohmS \leq 1000 \Omega.m^2` encoded to `{0}`, 
:math:`oarea2 :1000 <ohmS \leq 2000 \Omega.m^2` encoded to `{1}` and :math:`oarea3 : oa3 > 2000 \Omega.m^2` encoded to `{2}`.  The code snippet is given below:


.. code-block:: python 

	>>> from watex.view.plot import QuickPlot 
	>>> from watex.datasets import load_bagoue 
	>>> from watex.utils import smart_label_classifier 
	>>> data = load_bagoue ().frame
	>>> demo_features =['ohmS', 'power', 'lwi', 'flow'] 
	>>> data_area=data [demo_features] 
	>>> # categorized the ohmS series into a class labels 'oarea1', 'oarea2' and 'oarea3'
	>>> data_area ['ohmS'] = smart_label_classifier (data_area.ohmS, values =[1000, 2000 ])
	>>> qkObj = QuickPlot(mapflow =False, tname="ohmS"
							  ).fit(data_area)
	>>> qkObj.sns_style ='darkgrid', 
	>>> qkObj.fig_title='Quantitative features correlation'
	>>> qkObj.fig_size =(7, 5)
	>>> sns_pkws={'aspect':2 , 
	...          "height": 2, 
	# ...          'markers':['o', 'x', 'D', 'H', 's',
	#                         '^', '+', 'S'],
	...          'diag_kind':'kde', 
	...          'corner':False,
	...          }
	>>> marklow = {'level':4, 
	...          'color':".2"}
	>>> qkObj.numfeatures(coerce=True, map_lower_kws=marklow, **sns_pkws)

Here is the following output: 

.. figure:: ../examples/auto_examples/view_quickplot_numfeatures.png
   :target: ../examples/auto_examples/view_quickplot_numfeatures.html
   :align: center
   :scale: 60%  
   
   
Plot Correlating Features: :meth:`~watex.view.QuickPlot.corrmatrix`  
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~watex.view.QuickPlot.corrmatrix`  method quickly plots the correlation between the numerical and categorical features 
by setting the parameter `cortype` to ``num`` or ``cat`` for numerical or categorical features. Here is an example 
using the `Bagoue dataset` with mixture type of features:
        
.. code-block:: python 

	>>> from watex.view.plot import QuickPlot 
	>>> from watex.datasets import load_bagoue 
	>>> data = load_bagoue ().frame
	>>> qplotObj = QuickPlot(fig_size = (7, 5)).fit(data)
	>>> sns_kwargs ={'annot': False, 
	...            'linewidth': .5, 
	...            'center':0 , 
	...            # 'cmap':'jet_r', 
	...            'cbar':True}
	>>> qplotObj.corrmatrix(cortype='cat', **sns_kwargs)
	>>> qplotObj.corrmatrix( **sns_kwargs)
	QuickPlot(savefig= None, fig_num= 1, fig_size= (7, 5), ... , classes= None, tname= None, mapflow= False)
   
Here are examples of matrix correlation between categorical and numerical features 

.. |cat_corr| image:: ../examples/auto_examples/view_quickplot_correlation_matrix_categorical_features.png
   :target: ../examples/auto_examples/view_quickplot_correlation_matrix_categorical_features.html
   :scale: 40%

.. |num_corr| image:: ../examples/auto_examples/view_quickplot_correlation_matrix_numerical_features.png
   :target: ../examples/auto_examples/view_quickplot_correlation_matrix_numerical_features.html
   :scale: 40%
 
	
* **Categorical and numerical correlation** 

  =================================    ====================================
  Categorical features correlation	    Numerical features correlation
  =================================    ====================================
  |cat_corr|    					    |num_corr|
  =================================    ====================================


Plot Multiples categorical feature distribution :meth:`~watex.view.QuickPlot.multicatdist`  
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~watex.view.QuickPlot.multicatdist` gives a figure-level interface for drawing multiple categorical distributions. It plots 
onto a FacetGrid. The following code snippet gives a basic illustration. 

.. code-block:: python 

	>>> from watex.view.plot import QuickPlot 
	>>> from watex.datasets import load_bagoue 
	>>> data = load_bagoue ().frame
	>>> qplotObj= QuickPlot(lc='b', tname='flow')
	>>> qplotObj.sns_style = 'darkgrid'
	>>> qplotObj.mapflow=True # to categorize the flow rate 
	>>> qplotObj.fit(data)
	>>> fdict={
	...            'x':['shape', 'type', 'type'], 
	...            'col':['type', 'geol', 'shape'], 
	...            'hue':['flow', 'flow', 'geol'],
	...            } 
	>>> qplotObj.multicatdist(**fdict)
        

A mutiples categorical feature distributions can give a usefull insights about the figure distributions. Let check the following outputs and 
give a brief interpretation: 

.. figure:: ../examples/auto_examples/view_quickplot_multicat_distribution_with_geol_as_target.png
   :target: ../examples/auto_examples/view_quickplot_multicat_distribution_with_geol_as_target.html
   :align: center
   :scale: 50%  

For instance the figure above shows at the glance that the granites are the most geological structures encountered in the survey area. Furthermore, the conductive zones 
found in the volcano-sedimentary schists are mostly dominated by the type `CP` and `NC` whereas the they are found everywhere in the 
granites formations especially in the anomly with shape `V` . The conductive zone with shape `C` and `K` as well as the `U` are rarely found. This can be 
explained by the difficulty to find a wide-fracture in that area. Refer to :func:`~watex.utils.exmath.type_`  and :func:`~watex.utils.exmath.shape` for 
further details about the `type` and `shape` DC parameters` computation and [4]_ for a better explanations. The next figure will try to 
give  insights about the probable relationship between the geological formations of the area and the other features. 

.. figure:: ../examples/auto_examples/view_quickplot_multicat_distribution_with_flow_as_target_vs_geol.png
   :target: ../examples/auto_examples/view_quickplot_multicat_distribution_with_flow_as_target_vs_geol.html
   :align: center
   :scale: 50%  
   
The figure validates the predominance of the granites formations in the area and indicated that at the same time, the granites are the 
most structures that yield a dried boreholes (unsucessful (FR0: FR=0)) and unsustainable boreholes (FR1). At the opposite to the granites, 
the volcano-sedimentary shists are the most productives with productive FR values (FR2 and FR3) althought there are a little bit FR1. It seems 
very interesting to consider the geology of the area when looking about the productive FR in the survey area. 


Plot Base Distributions using Histogram or Bar: :meth:`~watex.view.QuickPlot.histcatdist` | :meth:`~watex.view.QuickPlot.barcatdist`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two basic distributions plots can be visualized using the :meth:`~watex.view.QuickPlot.histcatdist` or 
:meth:`~watex.view.QuickPlot.barcatdist` for histogram and bar distribution respectively. Here are two bases examples. 
For better view, target can be mapped into a categorized labels. For instance the flow target straighforwardly fetched 
from :mod:`~watex.datasets` can be mapped using the function :func:`~watex.utils.cattarget` as follow:: 

	>>> from watex.datasets import load_bagoue 
	>>> from watex.utils import cattarget 
	>>> data = load_bagoue ().frame 
	>>> data.flow.values [:7] 
	array([2., 0., 1., 1., 1., 2., 1.])
	>>> data.flow = cattarget (data.flow , labels = [0, 1, 2,3], rename_labels= ['FR0', 'FR1', 'FR2', 'FR3'])
	>>> data.flow.values  [:7] 
	array(['FR2', 'FR0', 'FR1', 'FR1', 'FR1', 'FR2', 'FR1'], dtype=object)
	
The snippet code above can be skipped by fetching the same data using the boilerplate function :func:`~watex.datasets.fetch_data` yields the stored 
the second stored kind of flow target in the frame by passing  the `get` dictionnary method to the the `tag=original`like this::
	
	>>> from watex.datasets import fetch_data 
	>>> data= fetch_data ('Bagoue original').get('data=dfy2')
	>>> data.flow.values [:7]  # same like above even the data is shuffled.
	array(['FR2', 'FR1', 'FR1', 'FR2', 'FR2', 'FR2', 'FR1'], dtype=object)
	
The following codes gives bases distributions histogram and bar plots: 

.. code-block:: python 

	>>> from watex.view.plot import QuickPlot
	>>> from watex.datasets import load_bagoue 
	>>> data = load_bagoue ().frame 
	>>> # data = fetch_data ('Bagoue original').get ('data=dfy2') # for FR categorization instead  
	>>> qplotObj= QuickPlot(xlabel = 'Anomaly type',
							ylabel='Number of  occurence (%)',
							lc='b', tname='flow')
	>>> qplotObj.fig_size = (7, 5) 
	>>> qplotObj.sns_style = 'darkgrid'
	>>> qplotObj.fit(data)
	>>> qplotObj. barcatdist(basic_plot =False, 
						  groupby=['shape' ])
	
The following outputs gives the target numerical and  categorization histogram plots: 

.. |num_hplot| image:: ../examples/auto_examples/view_quickplot_bar_distributions_num_values.png
   :target: ../examples/auto_examples/view_quickplot_bar_distributions_num_values.html
   :scale: 40%

.. |cat_hplot| image:: ../examples/auto_examples/view_quickplot_bar_distributions.png
   :target: ../examples/auto_examples/view_quickplot_bar_distributions.html
   :scale: 40%
 
 
* **Numerical and categorization target histogram plot** 

  =================================    ========================================
  Histogram with numeric target 	        Histogram with categorical target
  =================================    ========================================
  |num_hplot|    					    	|cat_hplot|
  =================================    ========================================
 
		
For base bar distribution plots, refer to ::meth:`~watex.view.QuickPlot.barcatdist` examples below. 

`Tensor recovery  plots` 
------------------------------
:class:`~watex.view.TPlot`  gives base plots from short-periods EM processing data.
Indeed, :class:`~watex.view.TPlot` plots Tensors (Impedances , resistivity and phases ) plot class. 
:class:`~watex.view.TPlot` returns an instancied object that inherits from :class:`watex.property.Baseplots` ABC (Abstract Base Class) 
for visualization. 

.. note:: 
	:class:`~watex.view.TPlot` can straighforwardly used without explicity call the :class:`~watex.methods.Processing` beforehand. 
	When Edi data is passed to `fit` methods, it implicitely call the :class:`~watex.methods.Processing` under the hood for 
	attributes loading ready for visualization. The class does not output any restored or corrected EDI-files. To handle this 
	procedure, use :class:`~watex.methods.EM` instead.
	
Here, we will give some examples of base tensor recovery plots from EDI datasets loaded from using :func:`~watex.datasets.load_edis`. 
Note that, as well as all the :mod:`~watex.view` plotting classes, :class:`~watex.view.TPlot` inherits from a global parameters of 
:class:`~watex.view.property.BasePlot`. Thus, each plot can be flexibly customized according to the user's desire. 
For instance, to visualize the corrected 2D tensors, one can  customize the plot by settling the plot keyword arguments as::

	>>> plot_kws = dict(
		ylabel = '$Log_{10}Frequency [Hz]$', 
		xlabel = '$Distance(m)$', 
		cb_label = '$Log_{10}Rhoa[\Omega.m]$', 
		fig_size =(7, 5), 
		font_size =7) 
	>>> TPlot(**plot_kws )


Single site signal recovery visualization: :meth:`~watex.view.TPlot.plot_recovery` 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~watex.view.TPlot.plot_recovery` allows visualizing the restored tensor at each site. For instance the visualisation of 
the first site `S00` is given as: 

.. code-block:: 

	>>> from watex.view import TPlot
	>>> from watex.datasets import load_edis 
	>>> edi_data = load_edis (return_data =True, samples =7) 
	>>> plot_kws = dict( ylabel = '$Log_{10}Frequency [Hz]$', 
				xlabel = '$Distance(m)$', 
				cb_label = '$Log_{10}Rhoa[\Omega.m$]', 
				fig_size =(7, 4), 
				font_size =7. 
				) 
	>>> t= TPlot(**plot_kws ).fit(edi_data)
	>>> # plot recovery of site 'S01'
	>>> t.plot_recovery ('S01')
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	|Data collected =  7      |EDI success. read=  7      |Rate     =  100.0  %|
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	<'TPlot':survey_area=None, distance=50.0, prefix='S', window_size=5, component='xy', mode='same', method='slinear', out='srho', how='py', c=2>

The result of the abode code is given below: 

.. figure:: ../examples/auto_examples/view_tplot_recovery_tensor_site_s1.png
   :target: ../examples/auto_examples/view_tplot_recovery_tensor_site_s1.html
   :align: center
   :scale: 70% 
   
Indeed, the EDI-data stored in :code:`watex` is already preprocessed. For a concrete example, refer to :ref:`methods` page.

Mutiple sites signal recovery visualization: :meth:`~watex.view.TPlot.plot_multi_recovery`   
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~watex.view.TPlot.plot_multi_recovery` plots mutiple site/stations with signal recovery. Here is a example: 

.. code-block:: python 

	>>> from watex.view.plot import TPlot 
	>>> from watex.datasets import load_edis 
	>>> # takes the 03 samples of EDIs 
	>>> edi_data = load_edis (return_data= True, samples =21 ) 
	>>> TPlot(fig_size =(9, 7), font_size =7., show_grid =True, gwhich='both').fit(edi_data).plot_multi_recovery (
		sites =['S00', 'S07', 'S14',  'S20'], colors =['ok-',  'xr-.', '^b-', 'oc-.'])
   
The above code yields the following result. 

.. figure:: ../examples/auto_examples/view_tplot_mutiple_recovery_tensors.png
   :target: ../examples/auto_examples/view_tplot_mutiple_recovery_tensors.html
   :align: center
   :scale: 70% 
   
 
Two dimensional tensor plot: :meth:`~watex.view.TPlot.plot_tensor2d`   
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:meth:`~watex.view.TPlot.plot_tensor2d`  gives a quick visualization of tensors. In the following examples, the resistivity and 
phase tensors from `yx` are plotted using: 

.. code-block:: python 

	>>> from watex.view.plot import TPlot 
	>>> from watex.datasets import load_edis 
	>>> # get some 12 samples of EDI for demo 
	>>> edi_data = load_edis (return_data =True, samples =12)
	>>> # customize plot by adding plot_kws 
	>>> plot_kws = dict( ylabel = '$Log_{10}Frequency [Hz]$', 
						xlabel = '$Distance(m)$', 
						cb_label = '$Log_{10}Rhoa[\Omega.m$]', 
						fig_size =(6, 3), 
						font_size =7.,
						plt_style ='imshow',  
						) 
	>>> t= TPlot(component='yx', **plot_kws).fit(edi_data)
	>>> # plot recovery2d using the log10 resistivity 
	>>> t.plot_tensor2d (to_log10=True)
	
To run the phase tensor at the same component `yx`, we don't need to re-run the script above, just customize the colorbar label 
and easily set the `tensor` params to ``phase`` like below: 

.. code-block:: python 

	>>> t.cb_label= '$Phase [\degree]$' 
	>>> t.plot_tensor2d ( tensor ='phase', to_log10=True) 
	<AxesSubplot:xlabel='$Distance(m)$', ylabel='$Log_{10}Frequency [Hz]$'>
	
The following outputs is given below : 

.. |res_yx| image:: ../examples/auto_examples/view_tplot_quick_plot_tensor_resyx.png
   :target: ../examples/auto_examples/view_tplot_quick_plot_tensor_resyx.html
   :scale: 70%

.. |phase_yx| image:: ../examples/auto_examples/view_tplot_quick_plot_tensor_phase.png
   :target: ../examples/auto_examples/view_tplot_quick_plot_tensor_phase.html
   :scale: 70%
 
* **Resistivity and Phase tensors `xy` 2D plots** 

  =================================     ========================================
  Resistivity tensor yx 	            Phase tensor yx 
  =================================     ========================================
  |res_yx|    					    	|phase_yx|
  =================================     ========================================

  
Two dimensional filtered tensor plots: :meth:`~watex.view.TPlot.plot_ctensor2d`   
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~watex.view.TPlot.plot_ctensor2d` plots the filtered tensor by applying the filtered function. Refer to :ref:`methods` to 
have an idea about the different available filters. As the example below, we collected 12 EDI data and applied the fixed-dipole-length 
moving-average filter (FLMA) passed to the parameter `ffilter`. The default filter is trimming moving average (TMA)as: 

.. code-block:: python 

	>>> from watex.view.plot import TPlot 
	>>> from watex.datasets import load_edis 
	>>> # get some 12 samples of EDI for demo 
	>>> edi_data = load_edis (return_data =True, samples =12 )
	>>> # customize plot by adding plot_kws 
	>>> plot_kws = dict( ylabel = '$Log_{10}Frequency [Hz]$', 
						xlabel = '$Distance(m)$', 
						cb_label = '$Log_{10}Rhoa[\Omega.m$]', 
						fig_size =(6, 3), 
						font_size =7.,
						plt_style='imshow'
						) 
	>>> t= TPlot(**plot_kws ).fit(edi_data)
	>>> # plot filtered tensor using the log10 resistivity 
	>>> t.plot_ctensor2d (to_log10=True, ffilter='flma')

Here is the following output and can be compared with the above figure. The FLMA shows a smooth filtered resistivity tensor `yx` as possible. 


.. figure:: ../examples/auto_examples/view_tplot_quick_plot_tensor_resyx_filtered_with_fixed_dipole_length_moving_average.png
   :target: ../examples/auto_examples/view_tplot_quick_plot_tensor_resyx_filtered_with_fixed_dipole_length_moving_average.html
   :align: center
   :scale: 90% 
   

.. note:: 
	
	The params space plots give some basic plots for data exploratory and analysis. However, user can provides it own scripts for plotting.
	The module :mod:`~watex.view.plot` can not give all the possible plot that can be yield with the predicting datasets. 
	
.. seealso:: 

	`seaborn`_ provides some wonderfull statistical data visualization tools. 
	
.. _seaborn: https://seaborn.pydata.org/


.. topic:: References 
	.. [1] Bengfort, B., & Bilbro, R., 2019. Yellowbrick: Visualizing the scikit-learn model. Journal of Open Source Software, 4( 35 ), 1075. https://doi.org/10.21105/joss.01075
	.. [2] Mel, E.A.C.T., Adou, D.L., Ouattara, S., 2017. Le programme presidentiel d’urgence (PPU) et son impact dans le departement de Daloa (Cote d’Ivoire). Rev. Géographie Trop. d’Environnement 2, 10.
	.. [3] Mobio, A.K., 2018. Exploitation des systèmes d’Hydraulique Villageoise Améliorée pour un accès durable à l’eau potable des populations rurales en Côte d’Ivoire : Quelle stratégie ? Institut International d’Ingenierie de l’Eau et de l’Environnement.
	.. [3] Kouadio, K.L., Loukou, N.K., Coulibaly, D., Mi, B., Kouamelan, S.K., Gnoleba, S.P.D., Zhang, H., XIA, J., 2022. Groundwater Flow Rate Prediction from Geo‐Electrical Features using Support Vector Machines. Water Resour. Res. 1–33. https://doi.org/10.1029/2021wr031623
