.. _citing:

=============
Citing 
=============


We kindly request that any scientific publications utilizing *WATex* for analysis, research, 
or case studies cite the following reference. A prepared BibTeX entry for your convenience is 
provided below:

.. highlight:: none

::

    @article{Kouadio_Liu_Liu_2023,
        title={watex: machine learning research in water exploration},
        DOI={https://doi.org/10.1016/j.softx.2023.101367},
        journal={SoftwareX},
        author={Kouadio, Kouao Laurent and Liu, Jianxin and Liu, Rong},
        year={2023},
        volume={101367},
        pages={1-7}
    }

Additional case studies and applications involving *WATex* can enrich your understanding and 
application of machine learning in water exploration and more. These real-world applications 
highlight the versatility and effectiveness of *WATex* in field conditions. We invite you to 
explore these resources for a deeper dive into the practical use of *WATex* in geophysical exploration,  
environmental tasks, and more.

Flow rate predictions 
----------------------

Moreover, :code:`watex` has been utilized in impactful real-world case study papers. For instance, 
it facilitated flow rate prediction in the Bagoue region by applying support vector machines (SVMs), 
offering comprehensive explanations about DC-resistivity parameter definitions 
(:func:`watex.datasets.load_bagoue`). Another significant application involved ensemble 
learning paradigms to enhance flow rate prediction accuracy in areas facing severe water shortages. 
These studies underscore the practical benefits and effectiveness of *WATex* in addressing geophysical 
and water exploration challenges. Below are the ready-made BibTeX entries for these published papers:

.. highlight:: none

::

    @article{Kouadio2022,
        author = {Kouadio, Kouao Laurent and Loukou, Nicolas Kouame and Coulibaly, Drissa and Mi, Binbin and Kouamelan, Serge Kouamelan and Gnoleba, Serge Pac{\^{o}}me D{\'{e}}guine and Zhang, Hongyu and XIA, Jianghai},
        doi = {10.1029/2021WR031623},
        journal = {Water Resources Research},
        pages = {1--33},
        title = {{Groundwater Flow Rate Prediction from Geo‐Electrical Features using Support Vector Machines}},
        year = {2022}
    }

    @article{Kouadio_Liu_Kouamelan_Liu_2023,
        author = {Kouadio, Kouao Laurent and Liu, Jianxin and Kouamelan, Serge Kouamelan and Liu, Rong},
        title = {Ensemble Learning Paradigms for Flow Rate Prediction Boosting},
        journal = {Water Resources Management},
        volume = {37},
        number = {11},
        pages = {4413--4431},
        year = {2023},
        doi = {10.1007/s11269-023-03562-5},
        abstract = {In response to the issue of water scarcity in recent years, international organizations, in collaboration with many governments, have initiated several drinking water supply projects carried out by geophysical and drilling companies. Unfortunately, despite the reliability of electrical resistivity profiling (ERP) and vertical electrical sounding (VES) methods, the substantial financial losses incurred due to numerous unsuccessful drillings are owing to the difficulty to emphasize the drilling location properly. Therefore, we proposed the ensemble machine learning (EML) paradigms to predict the flow rate (FR) with an optimal score before any drilling operations. The approach was experimented in a region with severe water shortages. Thus, geo-electrical features from the ERP and VES were defined and coupled with borehole data to create the binary dataset for unproductive and productive boreholes respectively. Then, the dataset is state-of-art transformed before feeding to the EML algorithms. The model performance and generalization capability were evaluated using the Matthews correlation, the accuracy, the confusion matrix, the binary predictor error, the precision-recall, and the cumulative gain plot. As a result, the benchmark, pasting, extreme gradient boosting, and stacking paradigms have built a powerful range of FR prediction scores between 90 ~ 96%. Henceforth, the robust EML paradigms can be used to identify the best location for drilling operations, lowering the repercussion of unsuccessful drillings.},
        URL = {https://doi.org/10.1007/s11269-023-03562-5}
    }



Land subsidence simulation and risk prevention 
-----------------------------------------------

In addition to facilitating groundwater exploration, *WATex* has proven its versatility across a variety 
of geophysical and environmental challenges. A recent case study demonstrates *WATex*'s application in 
the realm of urban planning and infrastructure management, focusing on the simulation of land subsidence 
using advanced machine learning techniques. This research leverages the power of eXtreme Gradient Boosting 
Regressor (XGBR) and Long Short-Term Memory (LSTM) models, highlighting the critical roles of groundwater 
level and building concentration in land subsidence scenarios. By accurately predicting subsidence, this 
study expands the potential uses of *WATex* into environmental risk assessment and impact modeling. 
Furthermore, it emphasizes the software's capacity to contribute valuable insights for policy making, 
aimed at fostering sustainable urban development strategies.

The following references highlight the impactful real-world applications of *WATex* in addressing complex 
geophysical and environmental issues:

.. highlight:: none

::

    @article{LIU2024120078,
        title = {Machine learning-based techniques for land subsidence simulation in an urban area},
        journal = {Journal of Environmental Management},
        volume = {352},
        pages = {120078},
        year = {2024},
        issn = {0301-4797},
        doi = {https://doi.org/10.1016/j.jenvman.2024.120078},
        url = {https://www.sciencedirect.com/science/article/pii/S0301479724000641},
        author = {Jianxin Liu and Wenxiang Liu and Fabrice Blanchard Allechy and Zhiwen Zheng and Rong Liu and Kouao Laurent Kouadio},
        keywords = {Land subsidence, Machine learning, Environmental risk assessment, Groundwater impact modeling}
    }



Moreover, :code:`watex` are used in real-world case study papers. For instance, it was used for  the :term:`flow` rate prediction 
in :term:`Bagoue region` applying the support vector-machines (SVMs) and provides complete details explanations about the :term:`DC-resistivity` 
parameter definitions (:func:`watex.datasets.load_bagoue`). Here is a ready-made BibTeX entry of the published papers:

Acknowledgment
--------------
It is important to recognize the synergistic relationship between *WATex* and foundational machine 
learning libraries that enhance its capabilities. Specifically, the extensive use of `scikit-learn` 
for various machine learning tasks within *WATex* merits acknowledgment. Therefore, in addition to citing
*WATex* in your scholarly work, it is also highly recommended to cite `scikit-learn` to appreciate the 
comprehensive ecosystem of tools that contribute to the effectiveness of *WATex* in geophysical 
exploration and environmental studies.

For convenient referencing, here is the BibTeX entry for `scikit-learn`, as recommended in their citation 
guidelines:

.. highlight:: none

::

    @article{scikit-learn,
     title={Scikit-learn: Machine Learning in {P}ython},
     author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
             and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
             and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
             Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
     journal={Journal of Machine Learning Research},
     volume={12},
     pages={2825--2830},
     year={2011}
    }

Here is the link to the `scikit-learn` citation guidelines for more details: `scikit-learn <https://scikit-learn.org/stable/about.html#citing-scikit-learn>`_.

This acknowledgment not only enriches the scholarly rigor of your citations but also highlights 
the collaborative nature of open-source software development in advancing scientific research and 
practical applications in the field of machine learning and geophysics.
