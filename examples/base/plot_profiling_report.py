"""
============================
Plot Report Profiling 
============================

visualizes the data report.
"""
# Author: L.Kouadio 
# Licence: BSD-3-clause

#%%
# Plot reports and deals with the pandas-profiling module directly implemented   
# as a method of the assessor class :class:`~watex.base.Data`.  

from watex.datasets import load_bagoue 
from watex.base import Data 
profile= Data().fit(load_bagoue().frame ).profilingReport (title =" Profiling report")
# Export the report into html 
profile.to_file ('Bagoue_report.html') 
# to export to Json 
# profile.to_file ('Bagoue_report.json') 
#%% 
# Export the report into a wigte  
profile.to_widgets() 

#%% 
# Export to the notebook iframe 
profile.to_notebook_iframe() 

#%%
# .. seealso:: pip installation of `profiling report <https://pypi.org/project/pandas-profiling/>`_. 


