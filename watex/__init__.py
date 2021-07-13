

import re 
import pandas as pd



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
        