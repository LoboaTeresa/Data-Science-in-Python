# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 23:54:56 2020
@author: Teresa

Función que utiliza variables dummy y la introduce en un data set
"""

def createDummies(df, var_name):
    dummy_var = pd.get_dummies(df[var_name], prefix=var_name)
    df=df.drop([var_name], axis = 1)
    df=pd.concat([df, dummy_var], axis = 1) 
    return df