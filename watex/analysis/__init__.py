# import chart_studio if not exist , install it 
# process to install chart_studio using sub_process 
import sys
import warnings
import subprocess 
import pandas as pd

from watex.utils._watexlog import watexlog
  
PD_READ_FEATURES ={
                    ".csv":pd.read_csv, 
                     ".xlsx":pd.read_excel,
                     ".json":pd.read_json,
                     ".html":pd.read_json,
                     ".sql" : pd.read_sql
                     }  

SUCCES_IMPORT_CHARTSTUDIO=False 

try:
    import chart_studio
    import chart_studio.plotly as py 
except: 
    #implement pip as subprocess
    try: 
        
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
        'chart_studio'])
        #process output with an API in the subprocess module 
        reqs = subprocess.check_output([sys.executable,'m', 'pip', 'freeze'])
        installed_packages  =[r.decode().split('==')[0] for r in reqs.split()]
        #get the list of installed dependancies 
        watexlog.get_watex_logger().info(
            'CHART_STUDIO was successfully installed with its dependancies')
        
        SUCCESS_IMPORT_CHART_STUDIO =True
    except : 
        SUCCESS_IMPORT_CHART_STUDIO =False

else:
    # updating chart_studio 
    try : 
        chart_studio_version = [int(ss) for ss in chart_studio.__version__.split('.')]
        if chart_studio_version[0] == 1:
            if chart_studio_version[1] < 1:
                warnings.warn(
                    'Note: need chart_studio version 1.1.0 or higher to write '
                     ' to plot some analyses figures propertly.', ImportWarning)
                watexlog().get_watex_logger().warning(
                        'Note: need chart_studio version 1.14.0 to plot '
                    ' component analysis figures or might not work properly.')

            else : 
                msg = 'Plot chart is currently able to run.'
                watexlog().get_watex_logger().info( ''.join([msg, 
                                    'Chart_studio was successfully imported !']))

    except: 
        
         watexlog().get_watex_logger().debug(
            'Trouble occurs during searching of chart_studio version 1.14.0 to  '
            ' to collect the updating version. Default version will be use for'
            ' chart plot.')
  
    else: 
        
        msg = 'Plot chart is  updated  with new version and able to run.'
        watexlog().get_watex_logger().info( ''.join([msg, 
                'Chart_studio was successfully'
                ' imported with the updated version!']))
        
    SUCCESS_IMPORT_CHARTSTUDIO  =True     

