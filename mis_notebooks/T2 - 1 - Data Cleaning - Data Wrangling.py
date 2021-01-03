# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 10:59:46 2020
@author: Teresa

El data wrangling, a veces denominada data munging, es el proceso de transformar
y mapear datos de un dataset raw (en bruto) en otro formato con la intención de 
hacerlo más apropiado y valioso para una variedad de propósios posteriores, como 
el análisis. Un data wrangler es una persona que realiza estas operaciones de transformación.

Esto puede incluir mungling, visualización de datos, agregación de datos, entrenamiento
de un modelo estadístico, así como muchos otro usos potenciales. La oscilación de datos 
como proceso generalmente sigue un conjunto de pasos generales que comienzan extrayendo 
los datos de forma cruda del origen de datos, dividiendo los datos en bruto usando algoritmos
(por ejemplo, clasificación) o analizando los datos en estructuras de datos predefinidas,
y finalmente depositando el contenido resultante en un sistema de almacenamiento (o sitio)
para su uso futuro.

ÍNDICE
------------------------------------------------------------------------------
*Crear un subconjunto de datos
*Crear un subconjunto de datos con condiciones concretas
*Crear un subconjunto de columnas y filas con .iloc y .loc
*Generación de números aleatorios
    -enteros 
    -decimales
    -shuffle
    -elección aleatoria
    -seed
"""

import pandas as pd
import os

mainpath="C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master/datasets"
filepath = "customer-churn-model/Customer Churn Model.txt"
fullpath = os.path.join(mainpath, filepath)

data = pd.read_csv(fullpath)
print(data.head())

#-----------------------------------------------------------------------------
##CREAR UN SUBCONJUNTO DE DATOS
#-----------------------------------------------------------------------------
account_length=data["Account Length"]
print(account_length.head())
print(type(account_length))  #me sale que es de tipo SERIE y no Data Frame, por lo tanto se maneja de forma distinta con pandas

#podemos tambien seleccionar varias columnas:
desired_columns = ["Account Length", "Phone", "Eve Charge", "Day Calls"]
subset = data[desired_columns]
print(type(subset)) #al ser varias columnas es un de tipo Data Frame. Ya hemos trabajado antes con esto

#si el número de columnas es pequeño este metodo: ok
#si es grande, este método de seleccionar columnas no es eficiente

#puede ser más eficiente quitar las columnas que no nos interesan:
desired_columns = ["Account Length", "Phone", "Eve Charge", "Day Calls"]
all_columns_list = data.columns.values.tolist() #realmente en spider me da que el to.list() no sitve de mucho
print(all_columns_list)
sublist=[x for x in all_columns_list if x not in desired_columns] #una lista con los nombres de las columnas complementarias
subset2 = data[sublist] #obtengo el conjunto de datos complementario a las columnas deseadas.
print(subset2)

#otra forma de hacerlo:
a = set(desired_columns)
b = set(all_columns_list)
sublist2 = b - a
sublist2 = list(sublist2)
subset3 = data[sublist2] #obtengo el conjunto de datos complementario a las columnas deseadas.
print(subset3)

#-----------------------------------------------------------------------------
##CREAR UN SUBCONJUNTO DE DATOS CON CONDICIONES CONCRETAS
#-----------------------------------------------------------------------------
data[1:25] #selecciono las filas de la 1 a la 24
data[:8] #selecciono las filas de la 0 a la 7
data[60:] #selecciono las filas de la 60 en adelante.

##Usuarios con Total Mins > 500
data1 = data[data["Day Mins"]>200] #selecciono las filas con valores de la columna Day Mins > 200
print(data1)

#Usuarios de Nueva York (State = "NY")
data2 = data[data["State"] == "NY"] #selecciono las filas con valores de la columna State = NY
print(data2)

#AND-> &
data3 = data[(data["Day Mins"]>300) & (data["State"] == "NY")]

# OR-> |
data4 = data[(data["Day Mins"]>300) | (data["State"] == "NY")]

#Ejercicio usuarios estado OH con longiutud mayor que 100 y que tanto las llmadadas de día y de noche superen cada una los 100 min siendo llamadas día> llamadas noche
data4 = data[((data["Day Mins"]>200) < (data["Night Mins"] >200)) & (data["State"] == "NY") & (data["State"] == "OH")]
#-----------------------------------------------------------------------------
##CREAR UN SUBCONJUNTO DE FILAS Y COLUMNAS con ILOC Y LOC
#------------------------------------------------------------------------------
#Min de día, de noche y Longitud de la Cuenta (Columnas) de los primeros 50 infividuos (filas)
subset_first_50 = data[["Day Mins", "Night Mins", "Account Length"]][:50]

#Puedo usar el método IX para seleccionar filas y columnas dentro de un mismo corchete
data.iloc[1:10, 3:6] #primeras  filas sin contar con el header y las columnas de la 3 a la 6]
data.iloc[:, 3:6] #todas las filas y las columnas de la 3 a la 6
data.iloc[1:10, 3:6] #primeras  filas sin contar con el header y las columnas de la 3 a la 6]
data.iloc[[1,4,7,32], 3:6]
data.loc[[1,4,7,32], ["Area Code", "VMail Plan", "Day Mins"]] #lox es para etiquetas e iloc para índices

data ["Total Mins"]= data["Day Mins"]+data["Night Mins"]+ data["Eve Mins"]  #he añaido una nueva columna al set data con los valores especificados

#-----------------------------------------------------------------------------
##GENERACIÓN DE NÚMEROS ALEATORIOS
#------------------------------------------------------------------------------
import numpy as np

numrand = np.random.randint(1,100) #me genera un numero aleatorio entero entre el 1 y el 100
print(numrand)

decrand = np.random.random() #me genera un número aleatorio decimal entre 0 y 1

#Función que me genere una lista de n números aleatorios enteros dentro del intervalo [a,b]
def randint_list(n, a, b):
    x=[]
    for i in range(n):
        x.append(np.random.randint(a,b))
    return x

aleatorio = randint_list(25,1,50)
print(aleatorio)

#esta funcion es innecesaria porque ya existe un método que me hace esto:
import random
    
for i in range(10):
    random.randrange(0,100,7) #genemare números aleatorios entre 0 y 100 pero que sean múltiplos de 7
    random.randrange(1,100,7) #genemare números aleatorios entre 0 y 100 pero que sean múltiplos de 7, +1 (porque empieza en 1)

#Shuffling
a = np.arange(100) #me genera un array con números del 0 al 99 ordenados de menor a mayor
print(a)
arand = np.random.shuffle(a) #me desordena la array aleatoriamente
print(arand)

#Elección aleatoria
column_list = data.columns.values.tolist() #eeneración de un array con los nombres de las columnas de data
np.random.choice(column_list) #me elige uno de los valores del array column_list, la cual contiene los nombres de las columnas de data

## Seed: para garantizar la reproducibilidad de los resultados
np.random.seed(2018)  #realmente se suele establecer al principio del script y podemos hacerla igual a 2018 o cualquier otro número
for i in range(5):
    print(np.random.random())
    