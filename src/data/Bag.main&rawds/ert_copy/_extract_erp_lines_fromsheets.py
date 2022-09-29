# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:17:23 2021

@author: Daniel03
"""
import os 
import pandas as pd 

def read_from_excelsheets(erp_file): 
    allfls= pd.read_excel(erp_file, sheet_name=None)
    
    list_of_df =[os.path.basename(os.path.splitext(erp_file)[0])]
    for sheets , values in allfls.items(): 
        list_of_df.append(pd.DataFrame(values))
     
    
    return list_of_df 
def write_excel(listOfDfs, csv =False , sep=','): 
    
    site_name = listOfDfs[0]
    listOfDfs = listOfDfs[1:]
    for ii , df in enumerate(listOfDfs):
        
        if csv:
            df.to_csv(df, sep=sep)
        else :
            with pd.ExcelWriter(f"z{site_name}_{ii}.xlsx") as writer: 
                df.to_excel(writer, index=False)
    
    
if __name__=="__main__": 
    erp_file ='kouto-gbalo.xlsx'
    # erpPath = os.abspath()
    # print(os.path.abspath('.'))
    list_erp = [os.path.join(os.path.abspath('.'), file) 
                for file in os.listdir('.') 
                if os.path.isfile(file) and file.endswith('.xlsx')]
    # print(list_erp)
    for ffile in list_erp : 
        dictfiles = read_from_excelsheets(ffile)
        write_excel(dictfiles)