# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 18:04:24 2020

@author: 3A
"""

import pandas as pd

mainpath="C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master/datasets"
filepath = "ads/Advertising.csv"
fullpath = mainpath +"/" + filepath

data_ads = pd.read_csv(fullpath)

print(data_ads.head())
print("Filas en data_ads = %d "  %len (data_ads))

import numpy as np #para apendiczar algo a mi dataset

data_ads ["corrn"] = (data_ads["TV"] - np.mean(data_ads["TV"]))*(data_ads["Sales"]- np.mean(data_ads["Sales"])) #numerador del coef de correlacion Pearson

print(data_ads.head())

data_ads["corr1"] = (data_ads["TV"]-np.mean(data_ads["TV"]))**2 #☻**2 es al cuarado
data_ads["corr2"]= (data_ads["Sales"]-np.mean(data_ads["Sales"]))**2
corr_pearson = sum(data_ads["corrn"])/np.sqrt(sum(data_ads["corr1"])*sum(data_ads["corr2"]))
print(corr_pearson)

def corr_coef (df, var1, var2):
    #numerador del coef de correlacion Pearson
    df ["corrn"] = (data_ads[var1] - np.mean(data_ads[var1]))*(data_ads[var2]- np.mean(data_ads[var2]))
    df["corr1"] = (df[var1]-np.mean(df[var1]))**2 #**2 es al cuarado
    df["corr2"] = (df[var2]-np.mean(df[var2]))**2
    r = sum(df["corrn"])/np.sqrt(sum(df["corr1"])*sum(df["corr2"]))
    return r

print(corr_coef(data_ads, "TV", "Sales"))

#estudio la correlacion entre variables del data frame
cols = data_ads.columns.values
print (cols)
for x in cols:
    for y in cols:
        print(x + "," + y + ":" + str(corr_coef(data_ads, x, y)))

import matplotlib.pyplot as plt
plt.plot(data_ads["TV"], data_ads["Sales"],"ro")  #el ro indica que se muestren los puntos discretos en rojo
plt.title ("Gasto en TV vs Ventas del Producto")

plt.show()
plt.plot(data_ads["Newspaper"], data_ads["Sales"],"go")  #el ro indica que se muestren los puntos discretos en verde
plt.title ("Gasto en Newspapers vs Ventas del Producto")

plt.show()
plt.plot(data_ads["Radio"], data_ads["Sales"],"yo")  #el ro indica que se muestren los puntos discretos en amarillo
plt.title ("Gasto en Radio vs Ventas del Producto")

#el coef de pearson se puede calcular con una funcion de python:
data_ads = pd.read_csv(fullpath)  
print(data_ads.corr())

plt.matshow(data_ads.corr()) #pinto una matriz de correlación