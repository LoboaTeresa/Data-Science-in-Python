# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 18:56:26 2020

@author: 3A

FUNCIÓN PARA CALCULAR EL COEF DE CPORRELACIÓN DE PEARSON
coef(dataframe, "nombre_variable1", "nombre variable2")
realmente no sé si funcionaría o no con:
coef(dataframe, "nombre_variable1", "nombre variable2").
    
Para hacer un estudio de las correlaciones de las variables de un df:
    for x in cols:
        for y in cols:
            print(x + "," + y + ":" + str(corr_coef(data_ads, x, y)))

También podemos hacer todo esto con una función integrada en python:
  data.corr()
"""

def corr_coef (df, var1, var2):
    #numerador del coef de correlacion Pearson
    df ["corrn"] = (data_ads[var1] - np.mean(data_ads[var1]))*(data_ads[var2]- np.mean(data_ads[var2]))
    df["corr1"] = (df[var1]-np.mean(df[var1]))**2 #**2 es al cuarado
    df["corr2"] = (df[var2]-np.mean(df[var2]))**2
    r = sum(df["corrn"])/np.sqrt(sum(df["corr1"])*sum(df["corr2"]))
    return r
