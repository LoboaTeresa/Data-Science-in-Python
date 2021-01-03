# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 21:24:33 2020
@author: 3A

ÍNDICE
------------------------------------------------------------------------------
*Resumen de los datos: Dimensiones y estructuras estadísticas básicas.
*Missing Values
*Variables dummy
"""
#-----------------------------------------------------------------------------
##RESUMEN DE LOS DATOS; DIMENSIONES Y ESTRUCTURAS
#-----------------------------------------------------------------------------

import pandas as pd
import os

#especifico el path de los datos
mainpath="C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master/datasets"
filepath = "titanic/titanic3.csv"
fullpath = os.path.join(mainpath, filepath)

data = pd.read_csv(fullpath)
print(data.head(10))  #te muestra las 10 primeras filasjhh
print(data.tail()) #te muestralas últimas 5 filas

#Podemos ver las dimensiones del data sets de la siguiente manera:
dimensiones = data.shape
print(dimensiones)  #(filas,columnas)
print(data.columns) #obtenemos los nombres de las columnas
print(data.columns.values) #obtenemos los nombres de las columnas en un array

#vamos a hacer un resumen de los estadísticos básicos de las variables numéricas
print(data.describe())
"""
    count: cuántos objetos no nulos hay
    mean: media
    std: desviación estándar
    min: valor más pequeño
    max: valor más grande
    25%: quantiles- si ordenamos de menor a mayor, el valor que caería en el 20% de la ordenación
    50%: quantiles- si ordenamos de menor a mayor, el valor que caería en el 50%
    70%: quantiles- si ordenamos de menor a mayor, el valor que caería en el 70% 
"""
   
print(data.dtypes) #me muestra el tipo de data en cada columna
"""
    float64: número decimal de 64 bits
    object: string
    int64: número entero de 64 bits
"""

#-----------------------------------------------------------------------------
##MISSING VALUES
#-----------------------------------------------------------------------------
print(pd.isnull(data["body"])) # me dice si los elementos de la columna "body"son nulos (True) o no (False)
print(pd.isnull(data["body"]).values) #lo mismo que antes pero en un array
print(pd.isnull(data["body"]).values.ravel().sum()) #el ravel hace que todos los datos estén en 1 fila. En este caso no haria falta pero podría no ser el caso.
                                                    #el sum me suma los Trues (1) y así sé cuánts trues tengo.
#print(pd.notnull(data)) # me dice si los elementos de data son no nulos (True) o si lo son (False)
"""
    NaN y None indican que faltan valores. Obviamente en nuetsro dataset faltan datos 
    Los valores que faltan en un Data Set pueden deberse a dos motivos:
        *Extracción de datos
        *Recolección de los datos
    ¿Cómo gestionamos la falta de datos?
        *Borrado de datos de toda la columna/fila.
        *Cómputo de los valores faltantes: sustituir los Nan por otro valor.
"""
#Borrado de datos
data.dropna(axis=0,how="all") #borramos una fila (axis=0) solo si todos los valores son NaN (how="all")
data2=data
data.dropna(axis=0,how = "any") #borramos una fila entera si alguno de los valores son NaN (how="any")

#Computo de Nan y sustitución
data3=data
data3.fillna(0) #sustituimos los valores que faltan por 0; .fillna no sustituye datos del dataset original
data3=data3.fillna(0) # ahora si se sustituyen

data4=data
data4=data4.fillna("Desconocido") #sustituimos el NaNs/Nones por la palabra "desconocido"

data5=data
data5["body"] = data5["body"].fillna(0)
data5["home.dest"] = data5["home.dest"].fillna("Desconocido")
print(pd.isnull(data5["body"]).values.ravel().sum())

#Algo inteligente que hacer con columnas que guardan valores numéricos es sustituir
#los valores que faltan por la media de la columna.

print(pd.isnull(data["age"]).values.ravel().sum()) 
data5["age"].fillna(data5["age"].mean())

print(data5["age"][1292]) #me sale nan porque no he sustituido en data5. [columna][fila]

#otra sol. es sustituir un valor que falta por el valor más cercano a este por arriba o por abajo.
data5["age"] = data5["age"].fillna(method="ffill") #sustituye por el primer valor no Nan x encima del Nan
data5["age"] = data5["age"].fillna(method="backfill") #sustituye por el primer valor no NaN x debajo del  Nan

#-----------------------------------------------------------------------------
##VARIABLES DUMMY
#-----------------------------------------------------------------------------
#para variables catgóricas, como la que guarda la columna "sex" (female o male) puede ser 
#util crear una variable dummy, es decir, una variable con 2 columnas basadas en la original.
# Una para male con valores 1 cuando sea male y 0 cuando sea female y una segunda columna al
# revés.
dummy_sex = pd.get_dummies(data["sex"], prefix="sex") #llamará a las columnas como prefix_valor, en este caso: sex_female y sex_male
print(dummy_sex.head())
column_name = data.columns.values.tolist() # el .tolist() me da una lista de los valores, no un array
print(column_name)

data=data.drop(["sex"], axis = 1) # me cargo la columna llamada "sex". axis=0->fila; axis=1-> columna.
                            # pero no me altera data. si quiero cambiar data: data=data.drop(...)
#si queremos combinar el dataset de la variable dummy:
data=pd.concat([data, dummy_sex], axis = 1) #con axis=1 especifico que quiero que se me añada como columna.
print (data.head())

#puedo convertir esto en una función. Mirar función "createDummies"

def createDummies(df, var_name):
    dummy_var = pd.get_dummies(df[var_name], prefix=var_name)
    df=df.drop([var_name], axis = 1)
    df=pd.concat([df, dummy_var], axis = 1) 
    return df
data_good=createDummies(data3, "sex")
print(data_good.head())
