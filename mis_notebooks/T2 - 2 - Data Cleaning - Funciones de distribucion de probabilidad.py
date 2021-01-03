# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 19:46:23 2020
@author: 3A

ÍNDICE
------------------------------------------------------------------------------
*Distribución Uniforme
*Distribución normal
*Método de montecarlo para calcular el valor de pi
*Dummy Data Sets

"""
#-----------------------------------------------------------------------------
## DISTRIBUCIÓN DE PROBABILIDAD UNIFORME
#-----------------------------------------------------------------------------
#vamos a usar esta distrib para generar números aleatorios que sigan esta distrib, es decir, que cada valor aleatorio tenga igual prob de salir.
 
import numpy as np  #para usar números random
import pandas as pd #para construir data frames
import matplotlib.pyplot as plt #cojo el subconjunto de la librería marplotlib para hacer plots
import os
np.random.seed(1997) 

a = 1 #límite inferior
b = 100 #límite superior
n = 200 #tamaño muestra
data  =np.random.uniform(a,b,n)

plt.hist(data)


#-----------------------------------------------------------------------------
## DISTRIBUCIÓN DE PROBABILIDAD NORMAL O DE GAUSS
#-----------------------------------------------------------------------------
data = np.random.randn(100)  # genero 100 valores aleatorios de tal manera que igan una distrib normal estandar (media cero y desv tipica=1)
x = range(1,101)
plt.plot(x,data)
#plt.hist(data)
#plt.plot(x, sorted(data)) #función de distrib acumulada, es decir, ploteo los números ordenados de menor a mayor

#distrib normal no estandar
mu = 5.5 #♣nota media examen
sd = 2.5 #desv estandar

data = 5.5 + 2.5 * np.random.randn(1000)
#plt.hist(data)

##TH central del límite
#puedo convertir cualquier distrib normal en una estandar tq: Z=(X-mu)/sd
#Y viceversa: X= mu + sd * Z
data1 = np.random.randn() #me da un numero aleatorio cuya prob sigue una distrib normal estandar
data2 = np.random.randn(2)  #me da un array con 2 numeros aleatorio cuya prob sigue una distrib normal estandar
data3 = np.random.randn(2,4) #me da un elemento con 2 filas/arrays con 4 elementos aleatorios cada uno que siguen una distrib normal estandar
                             #bastante útil para crear data sets
print(data1)
print(data2)
print(data3)

#-----------------------------------------------------------------------------
## SIMULACION MONTECARLO PARA OBTENER EL VALOR DE PI
#-----------------------------------------------------------------------------
#Genero dos números aleatorios entre 0 y 1 (x e y) en total 1000 veces
#Calculamos X * X + Y * Y (x e y han de seguir una distrib uniforme)
    #Si el valor de lo anterior es inferior a 1 --> estamos dentro del círculo
    #Si el valor de lo anterior es superior a 1 --> estamos fuera del círculo
#Calculamos el número total de puntos dentro del círculo y lo dividimos por el número total de intentos
#para obtener una aprox de la prob de caer dentro del círulo.
#Usamos dicha prob para aproximar el valor de pi.
#Repetimos el experimento un númiro suficiente de veces (ej 100), para obtener (100) aproximaciones diferentes de pi.
#Calculamos el promedio de los 100 experimentos anteriores para dar un valor final de pi.

def pi_montecarlo(n, n_exp):
    pi_avg = 0
    pi_value_list =[]
  
    for i in range(n_exp): # 1)
        value = 0
        x = np.random.uniform(0,1,n).tolist() #genero una lista con n valores de x aleatorios entre 0 y 1
        y = np.random.uniform(0,1,n).tolist()
        for j in range(n): # 2)
            z = np.sqrt(x[j] * x[j] + y[j] * y[j])
            if z <= 1: # 2)
                value += 1
        float_value = float(value) #número total de veces que lanzo y caigo en el círculo
        pi_value = float_value * 4 /n  # valor de pi, n es el número total de veces que lanzo --> numero veces en circulo/ numero veces totales
        pi_value_list.append(pi_value) #añado el valor de pi a la lista
        pi_avg += pi_value #voy sumando los valores de pi que me salen en cada bucle para al final calcular la media
    pi = pi_avg/n_exp
    
    print(pi)
    fig = plt.plot(pi_value_list) 
    return (pi, fig)

pi_montecarlo(1000, 100)

#-----------------------------------------------------------------------------
## DUMMY DATA SETS
#-----------------------------------------------------------------------------
n=1000000
dataF = pd.DataFrame( #creo un data frame a modo de diccionario de forma rápida usando pd
    {
         'A': np.random.randn(n),
         'B': 1.5 + 2.5 * np.random.randn(n),
         'C': np.random.uniform(5, 32, n)
     }
)
print(dataF.describe())

#voy a crear ahora un dummy data set a partir de un data set preexistente
mainpath="C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master/datasets"
filepath = "customer-churn-model/Customer Churn Model.txt"
fullpath = os.path.join(mainpath, filepath)

data = pd.read_csv(fullpath)
column_names = data.columns.values.tolist()
a=len(column_names)
dataF2 = pd.DataFrame(
    {
         'Column name': column_names,
         'A': np.random.randn(a),
         'B': np.random.uniform(0, 1, a)
     }, index = range(42, 42 + a) # esto último lo que hace es nunerar las filas del 42 hasta el 42+a
)
print(dataF2.head())

