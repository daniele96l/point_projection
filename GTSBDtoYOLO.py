#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:13:46 2022

@author: danieleligato
"""

import pandas as pd

res = [1360,800]

data = pd.read_csv('labels.txt', sep = ';', header = None)

data[1] = data[1]/res[0]
data[3] = data[3]/res[0]
data[2] = data[2]/res[1]
data[4] = data[4]/res[1]

data = data.rename(columns = {1:"X_1",2:"Y_1",3:"X_2",4:"Y_2",})
data["C_X"] = round((data["X_2"]-data["X_1"])/2 + data["X_1"],10)
data["C_Y"] = round((data["Y_2"]-data["Y_1"])/2 + data["Y_1"],10)
data["W"] = round((data["X_2"]-data["X_1"]),10)
data["H"] = round((data["Y_2"]-data["Y_1"]),10) 


data[0] = data[0].str[:-4]


data_correct = data[[5,"C_X","C_Y","W","H",0]]
'''
i=0
for index, row in data_correct.iterrows():
    if i > len(data_correct):
       break
    else:
       f = open(row[0]+'.txt', 'w')
       current_frame = row[0]
       stringa = str(row[:5].values).replace(']', '');
       stringa = stringa.replace('[', '');
       f.write(stringa)
       
       f.close()
       i+=1'''
       
for i in range(len(data_correct)-1):
    current_frame = data_correct.loc[i][0]
    f = open(current_frame+'.txt', 'a')
    stringa = str(data_correct.loc[i][:5].values).replace(']', '');
    stringa = stringa.replace('[', '');
    f.write(stringa)
    f.write("\n")
    print(stringa)
    if(current_frame != data_correct.loc[i+1][0]):
        f.close()
        print(current_frame)

        
        
    i+=1
        
       
       
       
