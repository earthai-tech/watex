
.. _five_min_ref:

============================
Five Minutes Understanding
============================

This section delves into the project's origins from :term:`CDWS` and provides insights into the 
significance and straightforward understanding of the :term:`DC-resistivity` methods during the 
:term:`campaign for drinking water supply` (CDWS).

What are :term:`ERP` and :term:`VES`?
----------------------------------------

:term:`Electrical resistivity profiling` (ERP) and :term:`vertical electrical sounding` (VES) represent 
subsurface :term:`geophysical` imaging techniques favored for :term:`groundwater` discovery during CDWS, 
particularly in developing nations. These methods are chosen for their affordability and suitability in 
sedimentary environments. Typically, ERP is employed first to identify conductive zones (anomalies) based 
on criteria such as anomaly :term:`shape`, :term:`type`, and local :term:`geology` [1]_; [2]_. Following 
this, VES is used to infer the presence of fracture zones and determine layer thicknesses. 
The combination of these methods aims to locate the most favorable drilling site expected to yield 
a flow rate at or above the required :term:`flow rate` (RFR).

Why is :term:`RFR` crucial in :term:`CDWS`?
------------------------------------------------------------

The RFR is directly linked to the hydraulic system design, which depends on the population 
size within the target locality [3]_. For instance, a village hydraulic system (VH) is typically 
recommended for populations under 1000 (:math:`FR\geq 1 m^3/hr`). For populations between 1000 and 
4000, an improved village hydraulic (IVH) system (:math:`FR\geq 3 m^3/hr`) is suggested, while 
urban hydraulic systems (UH) (:math:`FR\geq 10 m^3/hr`) are designed for larger populations 
exceeding 4000 ([4]_; [5]_; [6]_). Consequently, drilling that yields a flow rate below the RFR is 
deemed unsuccessful, necessitating further geophysical surveys in the area. This process not only 
incurs additional costs but also poses significant challenges in adhering to project timelines, a 
puzzle that many local companies strive to solve.

What traditional techniques/tips are used to solve the unsuccessful :term:`drilling`?
-----------------------------------------------------------------------------------------

Local companies typically propose up to three drilling locations, ranked by priority after 
the survey, to maximize their chances of achieving the required flow rate (RFR). 
However, the strategy of suggesting multiple drilling points does not always 
guarantee success, and locating the optimal drilling site to meet the RFR 
continues to pose significant challenges.

.. figure:: ../example_thumbs/erp_scheme.png
   :target: ../example_thumbs/erp_scheme.html
   :align: center
   :scale: 60%
   
   :term:`DC-resistivity` methods. a) :term:`ERP` and :term:`VES` investigations. 
   b) Priority for drilling operations based on traditional methods.

How traditional techniques mitigate unsuccessful :term:`drilling` ?
----------------------------------------------------------------------

Traditional methods to reduce unsuccessful drilling risks typically involve a blend 
of initial surface assessments, analysis of historical data, and the incorporation 
of local geological insights. Practices such as geological mapping, soil sampling, 
and the use of anecdotal evidence from community water wells are instrumental in 
gathering crucial pre-drilling information. Furthermore, tailoring drilling strategies 
to the specific subsurface conditions encountered, alongside applying empirical formulas 
rooted in the local geological and hydrological context, has proven effective in 
improving drilling success rates.

Grasping these conventional techniques is essential to recognize the enhancements 
brought about by :code:`watex`. Merging contemporary :term:`machine learning` algorithms 
with established :term:`geophysical` approaches, :code:`watex` elevates the precision 
of groundwater discovery processes. This significant improvement in predictive 
accuracy  diminishes the chance of drilling failures and also ensures a more 
efficient use of resources, paving the way for more sustainable water supply solutions.

What's novel about using :term:`WATex` in :term:`GWE`?
-------------------------------------------------------

:code:`watex` introduces "smart" algorithms, including pre-trained :term:`machine learning` 
models from :class:`watex.models.pModels`, to predict feasible :term:`FR` before 
initiating drilling operations. This innovative approach aims to enhance 
traditional :term:`geophysical` methods, decrease unsuccessful drillings, 
minimize unsustainable boreholes, and reduce financial losses. Additionally, 
when constraints, such as site restrictions, are inputted into the 
:class:`watex.methods.electrical.ResistivityProfiling` class, :code:`watex` can 
advise whether the auto-detected station is suitable for drilling. It also alerts 
users about selected stations near restricted areas, thereby informing decision-making 
processes effectively.

Addressing the :term:`GWE` challenge during :term:`CDWS` has been a cornerstone 
in the development of the :code:`watex` project.

.. topic:: References

   .. [1] Nikiema, D.G.C., 2012. Essai d‘optimisation de l’implantation géophysique des 
      forages en zone de socle : Cas de la province de Séno, Nord Est du Burkina Faso. 
      IST / IRD Ile-de-France, Ouagadougou, Burkina Faso, West-africa.
   .. [2] Sombo, P.A., Williams, F., Loukou, K.N., Kouassi, E.G., 2011. Contribution de 
      la Prospection Électrique à L’identification et à la Caractérisation des Aquifères 
      de Socle du Département de Sikensi (Sud de la Côte d’Ivoire). Eur. J. Sci. Res. 64, 206–219.
   .. [3] CIEH, 1993. évaluation de l’aide publique française (1981-1990) / Ministère 
      de la coopération et du développement, Secrétariat permanent des études, des 
      évaluations et des statistiques, in: Evaluations / Ministère de La Coopération 
      et Du Développement ; 10). Paris : Ministère de la coopération et du développement, 
      Secrétariat permanent des études, des évaluations et des statistiques, cop. 1992, p. 139 p. : tabl., couv. ill. en coul.; 30 cm.
   .. [4] CIEH, 2001. L’utilisation des méthodes géophysiques pour la recherche 
      d’eaux dans les aquifères discontinus. Série hydrogéologie 169.
   .. [5] MHCI, 2012. Lancement des travaux de renforcement de l’alimentation en eau 
      potable de Boundiali. Minist. l’hydraulique 15.
   .. [6] Mobio, A.K., 2018. Exploitation des systèmes d’Hydraulique Villageoise Améliorée 
      pour un accès durable à l’eau potable des populations rurales en Côte d’Ivoire : 
      Quelle stratégie ? Institut International d’Ingenierie de l’Eau et de l’Environnement.

