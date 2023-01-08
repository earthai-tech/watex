.. _data_space:

=================
Data space
=================

The data space is composed of four different data types referring to the implemented methods such as:

ERP data type
------------------

It can be arranged into several formats such as `*.csv`, `*. xlsx`, `*.xml`, `*.html`, or 
simple in `Pandas <https://pandas.pydata.org/>`_  data frame. The columns of ERP must be composed of station positions, the resistivity data, 
and the coordinates such as longitude/latitude or easting/ northing. 

VES data type
---------------

It expects the same format as ERP. However, the columns of DC- sounding must be the AB/2 depth measurements at each 
time the current electrodes are moved apart and the resistivity values collected at each sounding depth. The MN/2 values 
of the potential electrodes are not compulsory. 


EM data type
--------------

watex deals only with the SEG-Electrical Data Interchange format(*.edi). However, the EDI - object 
created from external software like `pycsamt <https://github.com/WEgeophysics/pycsamt>`_ and `MTpy <https://github.com/MTgeophysics/mtpy>`_ 
can also be read. Indeed, the watex EDI module API is designed to work with both. In addition, attributes and methods 
from EDI objects are constructed following both software structures. Boreholes and geology data type: Both can be collected 
in `*. yaml`, `*.json` or `*csv formats`. An example of data arrangement can be found in the `data/boreholes` directory of the package. 