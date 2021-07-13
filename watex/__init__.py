import re
import warnings 
import pandas as pd

import watex.utils.decorator as dec 
import watex.utils.exceptions as Wex
from watex.utils.__init__ import savepath as savePath 


OptsList, paramsList =[['bore', 'for'], 
                        ['x','east'], 
                        ['y', 'north'], 
                        ['pow', 'puiss', 'pa'], 
                        ['magn', 'amp', 'ma'], 
                        ['shape', 'form'], 
                        ['type'], 
                        ['sfi', 'if'], 
                        ['lat'], 
                        ['lon'], 
                        ['lwi', 'wi'], 
                        ['ohms', 'surf'], 
                        ['geol'], 
                        ['flow', 'deb']
                        ], ['id', 
                           'east', 
                           'north', 
                           'power', 
                           'magnitude', 
                           'shape', 
                           'type', 
                           'sfi', 
                           'lat', 
                           'lon', 
                           'lwi', 
                           'ohmS', 
                           'geol', 
                           'flow'
                           ]

def sanitize_fdataset(_df): 
    """ Sanitize the feature dataset. Recognize the columns provided 
    by the users and resset according to the features labels disposals
    :attr:`~Features.featureLabels`."""
    
    UTM_FLAG =0 
    
    def getandReplace(optionsList, params, df): 
        """
        Function to  get parames and replace to the main features params.
        
        :param optionsList: 
            User options to qualified the features headlines. 
        :type optionsList: list
        
        :param params: Exhaustive parameters names. 
        :type params: list 
        
        :param df: pd.DataFrame collected from `features_fn`. 
        
        :return: sanitize columns
        :rtype: list 
        """
        columns = [c.lower() for c in df.columns] 
        
        for ii, celemnt in enumerate(columns): 
            for listOption, param in zip(optionsList, params): 
                 for option in listOption:
                     if param =='lwi': 
                        if celemnt.find('eau')>=0 : 
                            columns[ii]=param 
                            break
                     if re.match(r'^{0}+'.format(option), celemnt):
                         columns[ii]=param
                         if columns[ii] =='east': 
                             UTM_FLAG=1
                         break

        return columns

    new_df_columns= getandReplace(optionsList=OptsList, params=paramsList,
                                  df= _df)
    df = pd.DataFrame(data=_df.to_numpy(), 
                           columns= new_df_columns)
    return df , UTM_FLAG
     
   
@dec.writef(reason='write', from_='df')
def exportdf (df =None, refout:str =None,  to:str =None, savepath:str =None,
              modname:str  ='_wexported_', reset_index:bool =True): 
    """ 
    Export dataframe ``df``  to `refout` files. `refout` file can 
    be Excell sheet file or '.json' file. To get more details about 
    the `writef` decorator , see :doc:`watex.utils.decorator.writef`. 
    
    :param refout: 
        Output filename. If not given will be created refering to the 
        exported date. 
        
    :param to: Export type; Can be `.xlsx` , `.csv`, `.json` and else.
       
    :param savepath: 
        Path to save the `refout` filename. If not given
        will be created.
    :param modname: Folder to hold the `refout` file. Change it accordingly.
        
    :returns: 
        - `df_`: new dataframe to be exported. 
        
    """
    if df is None :
        warnings.warn(
            'Once ``df`` arguments in decorator :`class:~decorator.writef`'
            ' is selected. The main type of file ready to be written MUST be '
            'a pd.DataFrame format. If not an error raises. Please refer to '
            ':doc:`~.utils.decorator.writef` for more details.')
        
        raise Wex.WATexError_file_handling(
            'No dataframe detected. Please provided your dataFrame.')

    df_ =df.copy(deep=True)
    if reset_index is True : 
        df_.reset_index(inplace =True)
    if savepath is None :
       savepath = savePath(modname)
        
    return df_, to,  refout, savepath, reset_index 
        
        
        
        
        
        
        
        
        
        
        
        