.. _bagoue_dataset:

Bagoue DC parameters dataset
------------------------------

**Data Set Characteristics:**

    :Number of Instances: 431 
    :Number of Attributes: 12 numeric/categorical predictive. flow ( Attribute 13) is usually the target.
    :Attribute Information:
        - NUM      		number of borehole collected
        - NAME       	borehole ID 
        - EAST    		Easting coordinates in UTM (zone 27N, WGS84) (m)
        - NORTH     	Northing coordinates in in UTM (Zone 29N, WGS84) (m)
        - POWER      	power computed at the selected anomaly (conductive zone) (m)
        - MAGNITUDE     magnitude computed at the selected anomaly (conductive zone) (ohm.m)
        - SHAPE      	shape detected at the selected anomaly (conductive zone) (no unit)
        - TYPE      	type detected at the selected anomaly (conductive zone) (no unit)
        - SFI      		pseudo-fracturing index computed at the selected anomaly (no unit)
        - OHMS      	ohmic-are (presumed to reveal the existence of the fracture at the selected anomaly)  (ohm.m^2)
        - LWI  			water inrush collected after the drilling operation (m)
        - GEOL        	the geology of the exploration area
        - FLOW    		the flow rate obtained after the drilling operations (m^3/h)

    :Missing Attribute Values: 5.34 % on OHMS

    :Creator: Kouadio, L.
	
    :Date: October, 2022

The data of Bagoue area was collected in the Bagoue region (northern part of Cote d Ivoire, West Africa) 
during the campaign for Drinking Water Supply (CDWS) projects (the projects the Presidential Emergency Program in 2012-2013 and the National 
Drinking Water Supply Program (PNAEP) in 2014 (Kra et al. 2016, Mel et al. 2017)).  It is composed of 
431 DC-Electrical Resistivity Profile (ERP), 407 DC-Electrical Sounding (VES), and 431 boreholes data. Moreover, the array configurations 
of the ERP and VES methods during both programs are Schlumberger with 200 m for the current electrodes distance and 20 m for the potential electrodes.  

.. topic:: References

   - Kouadio, K.L., Kouame, L.N., Drissa, C., Mi, B., Kouamelan, K.S., Gnoleba, S.P.D., Zhang, H., et al. (2022). Groundwater Flow Rate Prediction from Geo-Electrical Features using Support Vector Machines.' Water Resour. Res. doi:10.1029/2021wr031623
   - Kra, K.J., Koffi, Y.S.K., Alla, K.A. & Kouadio, A.F. (2016) 'Projets d emergence post-crise et disparite territoriale en Cote dIvoire.' Les Cah. du CELHTO, 2, 608_624.
   - Mel, E.A.C.T., Adou, D.L. & Ouattara, S. (2017) 'Le programme presidentiel d urgence (PPU) et son impact dans le departement de Daloa (Cote dIvoire).' Rev. Geographie Trop. dEnvironnement, 2, 10. Retrieved from http://revue-geotrope.com/update/root_revue/20181202/13-Article-MEL-AL.pdf
   - Many, many more ...