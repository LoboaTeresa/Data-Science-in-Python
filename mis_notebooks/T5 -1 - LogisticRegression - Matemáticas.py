# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:59:50 2020

@author: 3A

ÍNDICE: Las matemáticas tras la regresión logística
------------------------------------------------------------------------------
*Las tablas de contingencia
*Porbabilidad condicional
*Ratio de probabilidades
*La regresión logística desde la regresión lineal

"""
import pandas as pd
import numpy as np


mainpath="C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master/datasets"
filepath = "gender-purchase/Gender Purchase.csv"
fullpath = mainpath +"/" + filepath

df = pd.read_csv(fullpath)
print(df.head())
#-----------------------------------------------------------------------------
## TABLAS DE CONTINGENCIA
#-----------------------------------------------------------------------------
print(df.shape) #(filas, columnas)

#TABLA DE CONTINGENCIA: filas y columnas que nos indican la frecuencia con la que ocurre un suceso
contingency_table = pd.crosstab(df["Gender"], df["Purchase"])
print("TABLA DE CONTINGENCIA: ")
print(contingency_table)


print(contingency_table.sum(axis = 1)) #número de mujeres/ hombres entran en la tienda (axis=1--> fila)
print(contingency_table.sum(axis = 0)) #número de personas que no compran (axis=0 --> columnna)

#Coge cada componente de la tabla y la divide por la suma de su correspondiente 
#fila. Así obtenemos el porcentaje de fujeres/hombres que compran y que no compran
#El axis = 0  no sé por qué es necesario pero lo es. Sirve para devolver los resultados por columnas
print(contingency_table.astype("float").div(contingency_table.sum(axis=1), axis = 0))
#-----------------------------------------------------------------------------
## PROBABILIDAD CONDICIONAL
#-----------------------------------------------------------------------------
#Probabilidad de que un suceso ocurra o no conociedo ciertas condiciones
#Ej: A= ¿Cuál es la prob. de que un cliente compre un producto si es hombre?
#  B=¿Cuál es la prob. de que sabiendo que un cliente compra un producto sea mujer?
from IPython.display import display, Math, Latex #Lo del latex es m´sa para los notebooks de Jupiter
display(Math(r'P(Purchase|Male) = \frac{Numero\total\de\compras\hechas\por\hombres}{Numero\total \de\hombres\del\grupo}=\frac{Purchase\cap Male}{Male}'))
A=121/246
display(Math(r'P(Purchase|Female) = \frac{Numero\total\de\compras\hechas\por\mujeres}{Numero\total \de\compras}=\frac{Female\cap Purchase}{Purchase}'))
B=159/280
#Prob de que no sea mujer sabiendo que compra = 1-B = 121/280
#Las prob más importantes:

display(Math(r'P(Purchase|Male)' ))
print("(Número de hombres que compran/ Número de hombres totales) = 121 / 246 = "+ str(121/246))
display(Math(r'P(No\Male) '))
print("(Número de hombres que no compran / Número de hombres total) = 125 / 246 = "+ str(125/246))
display(Math(r'P(Purchase|Female)' ))
print("(Número de Mujeres que compran / Número de mujeres total) = 159 / 265 = "+ str(159/265))
display(Math(r'P(No\Female)' ))
print("(Número de Mujeres que no compran / Número de mujeres total) = 106 / 265 = "+ str(106/265) )


#-----------------------------------------------------------------------------
## RATIO DE PROBABILIDADES
#-----------------------------------------------------------------------------
#Cociente entre los casos de éxitos sobre los de fracaso en el suceso estudiado 
#y para cada grupo.
display(Math(r'P_m = \probabilidad\de\hacer\compra\sabiendo\que\es\un\hombre'))
display(Math(r'P_f = \probabilidad\de\hacer\compra\sabiendo\que\es\una\mujer'))
display(Math(r'odds{purchase,male} = \frac{P_m}{1-P_m}' ))
display(Math(r'odds{purchase,female} = \frac{P_f}{1-P_f}' ))
pm=121/246 #Probabilidad de hacer compra sabiendo que es un hombre
pf=159/265 #Probabilidad de hacer compra sabiendo que es una mujer
odds_m = pm/(1-pm)
odds_f = pf/(1-pf)
print("ratio hombres = (Probabilidad de hacer compra sabiendo que es un hombre / 1 - Probabilidad de no hacer compra sabiendo que es un hombre) =  (Número de hombres que compran / Número de hombres que no compran)" + str(odds_m))
print("ratio mijeres = (Probabilidad de hacer compra sabiendo que es una muher / 1 - Probabilidad de no hacer compra sabiendo que es una mujer) =  (Número de mujeres que compran / Número de mujeres que no compran)" + str(odds_f))
#Cualquier cociente (ratio) tendrá un valor entre o y +inf

#Si el ratio es superior a 1, es más probable el éxito que el fracaso. Cuanto mayor 
#el ratio, mayor es la prob de éxito del suceso.

#Si el ratio es exactamente igual a 1, exito y fracaso son equiprobables, p=0.5

#Si el ratio es menor que 1, el fracaso es más probable que el éxito. Cuanto menor 
#el ratio, menor es la prob de éxito del suceso.

odds_ratio= odds_m/odds_f #Si es >0 es más probable el éxito del suceso de hombres

#-----------------------------------------------------------------------------
## LA REGRESIÓN LOGÍSTICA DESDE LA REGRESIÓN LINEAL
#-----------------------------------------------------------------------------
"""
Lineal: y = a + b · x --> (x,y) entre (-inf , +inf)
Logística: y entre (0 , 1), x entre(-inf,+inf)
          P = a + b · x --> P es la porb condicionada de éxito o fracaso
                             condicionada a la variable x. P entre (0,1)
         Nuestra expresión para P no nos pone límites entre (0,1). En lugar de 
         dar una expresión lineal a P, démosla a P/(1-P)
         
         P = a + b · x --> esto me da valores entre (-inf, +inf)  :(
         P/(1-P) = a + b · x --> esto me da valores entre (0, +inf)  :(
         ln(P/(1-P)) = a + b · x ---> me da valores entre (0,+1) bieeeeen
 
P = 1 / {1 + exp[-(a + b · x)]} 

*Si a+bx es muy pequeñp (negativo), entonces P tiende a 0
*Si a+bx =0, P=0,5
*Si a+bx es muy grande (positivo), entonces P tende a 1

"""
#-----------------------------------------------------------------------------
## LA REGRESIÓN LOGÍSTICA MÚLTIPLE
#-----------------------------------------------------------------------------
# P = 1 / {1 + exp[-(a + SUM( bi · xi))]}  *sum desde i=1 hasta i=n

