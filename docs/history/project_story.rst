
.. _story_ref:

================
Project Story
================

Here is a story that describes the :code:`watex` project design. 

Introductory Notes
--------------------

Water is an indispensable resource globally, facing significant challenges in 
this century due to climate change. Beyond its use in industries, electronics, 
services, and companies, water remains a scarce commodity, particularly in 
developing countries. Indeed, water treatment processes are costly, with only 
developed nations typically able to afford comprehensive treatment solutions.

The genesis of the *WATex* project dates back to 2015, within the geosciences and 
civil engineering operations company (`IBS+ <https://www.facebook.com/ibsplus.ingenierie/?locale=fr_FR>`__). 
Selected as a geophysical engineer for this venture, my mission was to facilitate 
access to potable water across various regions of Côte d'Ivoire and beyond. 
Noteworthy examples of our work outside CIV include the Community Drinking Water Supply 
(:term:`CDWS`) projects in West African sub-regions like Niger, Guinea, `Mali <https://fr.wikipedia.org/wiki/Mali>`__, 
and `Burkina <https://fr.wikipedia.org/wiki/Burkina_Faso>`__. These projects received 
partial funding and support from international organizations such as the 
World Bank and UNICEF, alongside the governments of the host countries, under 
initiatives like PNAEP and PPU in Côte d'Ivoire.

The Development Journey
------------------------

Our experiences in these regions highlighted the acute need for efficient and 
reliable methods to locate groundwater resources. Traditional approaches, while 
effective to some extent, often resulted in high rates of unsuccessful drillings, 
leading to wasted resources and diminished hope among the communities we aimed 
to serve.

Recognizing this, we embarked on the development of *WATex* as a tool to harness 
the power of machine learning for water exploration. Our goal was to significantly 
improve the accuracy of groundwater detection, thereby reducing the risk of 
unsuccessful drillings and maximizing the impact of available resources.

*WATex* was envisioned as a user-friendly, open-source platform that could 
democratize access to advanced geophysical data analysis, making it accessible 
not only to professionals in the field but also to local communities and 
small-scale operations.


Trigger Effect
---------------

Although enhancing the living conditions of the population remains one of my primary 
concerns, the catalyst for :code:`watex`'s development was the Diacohou-Nord project in 2017, 
located in the central part of Côte d'Ivoire (see Figure 1). This project was unique in 
its aim to identify the optimal location for :term:`drilling` operations to achieve a :term:`flow` 
rate (:term:`FR`) of :math:`10m^3/hr` (referred to as :term:`RFR`). The area faced significant challenges with 
drinking water availability, especially during the dry season. The topographic and 
terrain constraints, among others, were not conducive to achieving the desired FR, 
exacerbating the living conditions of the local population. 

During our visit, from 6 P.M. to 6 A.M. daily, women in the village ventured into 
the forest to find potable water for their families (Figure 1.a). The water collected 
from lowlands, marsh areas, and distant forest rocks served as an alternative 
source when local wells and boreholes dried up, despite the risks posed by wild animals.

.. figure:: ../example_thumbs/watex_summary.jpg
   :target: ../example_thumbs/watex_summary.html
   :align: center
   :scale: 40%
   
   DC-resistivity survey investigations. a) An illustration of the critical water 
   shortage issue in Diacohou-Nord. Women wait for water in the forest from night 
   until morning. b) DC survey investigations in the "Koro" locality, north area 
   of Côte d'Ivoire.

The quest for water at night in the forest led to numerous and severe consequences, 
including snakebites. On May 17, 2017, I witnessed a woman bitten by a snake, who was 
then urgently transported to the nearest city for treatment. This incident highlighted 
the common and dangerous challenges faced by the community in their daily quest for water.

Motivated by this experience, I decided to address this issue to prevent such 
incidents from recurring. Two months later, I resigned from the company and sought 
scholarship opportunities to study programming and :term:`artificial intelligence`. My goal 
was to develop a machine capable of detecting underground water reservoirs and 
estimating the FR based on the population's needs, even during dry seasons.

.. figure:: ../example_thumbs/DN_seeking_water_2.jpg
   :target: ../example_thumbs/DN_seeking_water_2.html
   :align: center
   :scale: 70%
   
   Women seeking drinking water: left-panel) The site where a lady was bitten by 
   a snake. Early in the morning, my team and I inspected the site to propose 
   alternative solutions; right-panel) Survey investigation in Diacohou-Nord 
   locality. The grass's color shows the harsh effects of the dry season in this area.

Three months later, I was fortunate to receive a scholarship from the `China Scholarship Council (CSC) <https://www.chinesescholarshipcouncil.com/>`__  
(CSC) in collaboration with the Côte d'Ivoire government 
for a Ph.D. candidacy. I enrolled at `Zhejiang University <https://www.zju.edu.cn/english/>`__ (ZJU) in 2018, where 
my research focused on computational geophysics. My projects aimed to develop 
new ML approaches for detecting fracture zones and predicting :term:`FR` efficiently 
using :term:`DC-resistivity`and electromagnetic data (notably :term:`CSAMT`). Thus, the initial 
version of :code:`watex` was conceived, centering on a case study in the Bagoue 
region (see Figure 1). The outcomes were promising, achieving a 77% accuracy 
rate in FR predictions with a reasonable amount of data.


Efficiency Test
-----------------

To evaluate the software's effectiveness in a new location outside of the :term:`Bagoue region`, 
data were acquired from a local company, `GEOTRAP SARL <https://www.piaafrica.com/fr/cote-divoire/mines-exploitations/79486-geotrap-sarl-geophysique-et-travaux-publics>`__, in 
the Tankesse area of the Indenié Djuablin region (east of :term:`CIV`, see Figure 1). The gathered data 
were processed and analyzed using :code:`watex`'s :class:`watex.methods.electrical` algorithms to 
automatically identify favorable drilling stations (highlighted in blue) and select the optimal one 
by incorporating environmental constraints, aiming to achieve a :term:`RFR` of :math:`5m^3/hr`. The 
software ultimately recommended station ``S53`` as the best drilling site.

.. figure:: ../example_thumbs/tankesse_data_processing.jpg
   :target: ../example_thumbs/tankesse_data_processing.html
   :align: center
   :scale: 40%
   
   Data processing in Tankesse. Data were collected from GEOTRAP SARL.
   
Remarkably, two months after drilling commenced, a :term:`flow rate` of :math:`9.7m^3/hr` was achieved, 
surpassing :code:`watex`'s prediction of :math:`7.3m^3/hr` (see `video <https://youtube.com/shorts/NDci9g_v01Q>`__). 
This outcome underlines the algorithms' conservative approach in estimating :term:`groundwater` flow rates 
to minimize the risk of unsuccessful drillings and reduce financial expenditures. A `YouTube video <https://youtube.com/shorts/NDci9g_v01Q>`__ 
demonstrates :code:`watex`'s application efficiency in future Community Drinking Water Supply (CDWS) 
projects.

.. raw:: html

   <div style="text-align: center; margin-bottom: 2em;">
   <iframe width="320" height="560" src="https://www.youtube.com/embed/NDci9g_v01Q" title="Geosciences computing: watex efficient test performed in Tankesse area" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
   </div>

This success was particularly significant for the population of Tankesse, an area known for its thick 
granitic layers that frequently lead to inaccurate flow rate predictions during drilling operations. 
These challenges have historically resulted in numerous failed drills and unsustainable boreholes. The 
application of :term:`artificial intelligence` (AI) through :code:`watex` has demonstrated its capability 
to overcome these obstacles, marking a significant advancement for the community's well-being in the 
region. Data for the Tankesse area can be accessed via :func:`watex.datasets.load_tankesse`.

Impact and Future Vision
-------------------------

Since its inception, *WATex* has grown from a concept to a functional tool that 
has been applied in several practical projects, demonstrating its potential to 
revolutionize water exploration practices. By integrating machine learning algorithms 
with traditional geophysical methods, *WATex* offers a novel approach that enhances 
the prediction and analysis capabilities of researchers and practitioners alike.

Looking forward, we are committed to continuous improvement and expansion of 
*WATex*'s capabilities. Our aim is not only to refine its technical aspects but 
also to foster a community of users and contributors who can share experiences, 
data, and strategies for effective water exploration.

As we navigate the challenges of climate change and water scarcity, *WATex* stands 
as a beacon of innovation, offering hope and practical solutions for sustainable 
water management across the globe.

The journey of *WATex* is a testament to the power of collaborative innovation in 
addressing some of the most pressing environmental challenges of our time. It 
underscores the critical role of technology in enhancing our understanding and 
management of natural resources, paving the way for a more sustainable and 
water-secure future.

.. _External Links:

For more information about the projects and organizations mentioned, please visit 
the following links:

- `IBS+ Engineering <https://www.facebook.com/ibsplus.ingenierie/?locale=fr_FR>`__
- `World Bank <https://www.worldbank.org/en/home>`__
- `UNICEF <https://www.unicef.org/>`__

Conclusions
--------------
:code:`watex` emerges as a cost-effective tool by leveraging economical geophysical 
methods (:term:`ERP` and :term:`VES`) to predict the expected :term:`flow rate` (FR), 
which correlates with the population size of a locality for long-term water 
exploitation. For instance, if the population of a given area increases from 
2,000 to 50,000 inhabitants over ten years, the :term:`required flow rate` (RFR) of 
:math:`3m^3/hr` suitable for 2,000 people will become insufficient in a decade 
due to population growth and climate change impacts. Hence, :code:`watex` presents 
itself as a viable solution to minimize the frequency of unsuccessful and 
unsustainable drilling efforts.

Beyond addressing issues directly tied to hydrogeological exploration, :code:`watex` 
also presents additional valuable features. Looking ahead, it aspires to become a 
key library in the :term:`groundwater exploration` (GWE) domain within the next five 
years, enriched by the collective efforts and contributions from a diverse 
group of project participants.

.. seealso::

   For a quick comprehension of the project's inception, refer to :doc:`five-minutes <five_min_understanding>`.

*Credit to the author*.


