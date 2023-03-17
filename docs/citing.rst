.. _citing:

=============
Citing 
=============


If the software is used in any scientific publication, we appreciate citing the paper below. 
Here is a ready-made BibTeX entry:

.. highlight:: none

::

    @article{Kouadio_Liu_Liu_2023b,
    title={watex: machine learning research in water exploration}, 
    DOI={https://doi.org/10.1016/j.softx.2023.101367}, 
    journal={SoftwareX}, 
    author={Kouadio, Kouao Laurent and Liu, Jianxin and Liu, Rong}, 
    year={2023}, 
    pages={7} 
    }

Moreover, :code:`watex` are used in real-world case study papers. For instance, it was used for  the :term:`flow` rate prediction 
in :term:`Bagoue region` applying the support vector-machines (SVMs) and provides complete details explanations about the :term:`DC-resistivity` 
parameter definitions (:func:`watex.datasets.load_bagoue`). Here is a ready-made BibTeX entry of the published papers:

.. highlight:: none

::

    @article{Kouadio2022,
    author = {Kouadio, Kouao Laurent and Loukou, Nicolas Kouame and Coulibaly, Drissa and Mi, Binbin and Kouamelan, Serge Kouamelan and Gnoleba, Serge Pac{\^{o}}me D{\'{e}}guine and Zhang, Hongyu and XIA, Jianghai},
    doi = {10.1029/2021wr031623},
    file = {:C\:/Users/Daniel/Desktop/papers/groundwater-flow-rate-prediction-from-geo-electrical-features-using-support-vector-machines.pdf:pdf},
    issn = {0043-1397},
    journal = {Water Resources Research},
    pages = {1--33},
    title = {{Groundwater Flow Rate Prediction from Geo‐Electrical Features using Support Vector Machines}},
    year = {2022}
    }

Furthermore, :code:`watex` is also used to predict and enhance the :term:`permeability coefficient k` score from geological, 
:term:`borehole`  and logging data by solving  the numerous missing :term:`k` existing in the borehole data (:func:`watex.datasets.load_hlogs`). It implements a novel 
approach called mixture learning strategy (MXS: :class:`watex.methods.MXS`) as the combinaison of the :term:`unsupervised learning` ( K-Means and HAC algorithmns) 
and  :term:`supervised learning` ( SVMs and Xtreme Gradient Boosting) methodologies for reducing the numerous unsucessful 
pumping tests. Here is a ready-made BibTeX:
  
.. highlight:: none

::
  
    @article{Kouadio_Liu_Liu_2023a, 
    title={A mixture learning strategy for predicting aquifers permeability coefficient k}, 
    DOI={http://dx.doi.org/10.2139/ssrn.4326365}, 
    journal={Engineering Geology}, 
    author={Kouadio, Kouao Laurent and Liu, Jianxin and Liu, Rong}, 
    year={2023} 
    }
  
In most situations where :code:`watex` is cited, a citation to `scikit-learn <https://scikit-learn.org/stable/about.html#citing-scikit-learn>`_ would also be appropriate.


