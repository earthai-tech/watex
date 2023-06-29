.. _nanshang_dataset:

Nanshang drilling dataset
--------------------------

**Localisation**
    :Country: China 
    :Province: Guangdong 
    :City: Guangzhou
    :UTM zone: 49Q 
    :EPSG: 21419
    :Projection:Xian 80-94/Beijing 1954-19° Belt

**Data Set Characteristics:**

    :Number of Instances: 58 
    :Number of Attributes: 9 (3 dates, 4 numerics , 2 categoricals)
    :Attribute Information:
        - year of drilling 
        - hole id (hole_id) as the code of drilling
        - type of drilling ( 53 engineering or 5 hydrogeological)
        - longitude in degres decimal 
        - latitude in degres decimal 
        - easting coordinates in meters
        - northing coordinates in meters 
        - ground height distance in meters 
        - depth of boreholes in meters 
        - openning date of the drilling 
        - end date of the drilling

    :Summary Statistics:

      - dates 

      =================== ============== ================ ============== ================== ================ ============
      year                opening date     end date       drilling        type              easting           northing 
      =================== ============== ================ ============== ================== ================ ============
      year:               2018           2019             NSGXX-NSSXX    engineering         -                     -  
      opening date:       01/07/2018     -                NSGC25         engineering         2522589          19759356
      end date:           -              13/07/2019       19NSSW01      hydrogeological      2509081          19774075
      =================== ============== ================ ============== ================== ================ ============

      - numerics 

      ======================== =============== ============== =============== 
                                Min             Max               Mean        
      ======================== =============== ============== ===============
      year:                     2018            2019           2.018579e+03
      longitude                 -8.744506e+01   -8.668665e+01  -8.688448e+01
      latitude:                 1.864942e+00    2.187123e+00   2.025237e+00
      easting:                  2.499111e+06    2.587894e+06   2.522117e+06
      northing:                 1.974174e+07    1.977870e+07   1.976064e+07
      ground_height_distance    1.000000e-01    1.200000e+01   4.115789e+00
      depth                     2.080000e+01    2.031500e+02   8.940228e+01
      ======================== =============== ============== ===============

    :Missing Attribute Values: None
    :Creator: K.L. Laurent (lkouao@csu.edu.cn) and Liu Rong (liurongkaoyan@csu.edu.cn) 
    :Donor: Central South University - School of Geosciences and Info-physics(https://en.csu.edu.cn/)
    :Date: June, 2023


Nanshang data is collected during the Nashang project from 2018 to 2019. The main drilling performed 
is engineering drillings for controlling the soil settlement and quality. Besides, some 
hydrogeological drillings were also performed. Indeed, Nanshang project aim goal is to forecast the 
the probable land subsidence from 2024 to 2035 using many other influencal factors that encompass the
InSAR data, the highways map, the distance to read, rivers ( Pearl rives etc ). 

.. topic:: References

   - 