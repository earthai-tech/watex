# -*- coding: utf-8 -*-
"""
Created on Wed May  4 09:51:16 2022

@author: @Daniel03
"""
import os 
from watex.utils.func_utils import convert_csvdata_from_fr_to_en
import pandas as pd 


df = pd.read_csv(r'C:/Users\Administrator\Desktop\__elodata/pme.final.csv')

df = df.iloc [2: , ::]
df.reset_index(inplace = True , drop =True )
df_safe = df.copy()

s50 = df.sample (frac =0.5 , replace =True , random_state =42)
s33 = df.sample (frac =0.33 , replace =True , random_state =42)
s25 = df.sample (frac =0.25 , replace =True , random_state =42)

df50 = pd.concat ([df_safe , s50])
df33 = pd.concat ([df_safe , s33])
df25 = pd.concat ([df_safe , s25] )

for name , ff in zip(['sf50' , 'sf33', 'sf25'], [df50 , df33, df25]) : 
    ff.to_csv (r'C:\Users\Administrator\Desktop\__elodata\pme.data.{0}.csv'.format(name),
               index =False)
