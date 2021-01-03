# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:30:32 2020

@author: 3A

ÍNDICE
------------------------------------------------------------------------------
*Dividir el dataset en conjunto de entrenamiento y de testing
*Creación del modelo con el conjunto training
*Validación del modelo con una predicción con los datos de testing
*Comentario de los estadísticos

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1997) 

mainpath="C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master/datasets"
filepath = "ads/Advertising.csv"
fullpath = mainpath +"/" + filepath

data = pd.read_csv(fullpath)
print(data.head())

#-----------------------------------------------------------------------------
## DIVIDIR EL DATASET EN CONJUNTO DE ENTRENAMIENTO Y DE TESTING
#Antes de hacer cualquier modelo lo suyo es hacer esta división de los datos
#-----------------------------------------------------------------------------

#Construyo una distribución normal de la longitud de nuestro dataset
a = np.random.randn(len(data))
check = (a<0.8) ##array de true o false en funcion de si a es mayor o menor que 0.8
training = data[check] #80% de los datos van al conjunto de training #consideto los training data aquellos que cumplen la condicion del check
testing = data[~check] #20% de los datos van al conjunto de testing #consideto los testing data aquellos que no cumplen la condicion del check

print("Longitud del conjunto training: %d" %len(training))
print("Longitud del conjunto testing: %d" %len(testing))

#Ahora crearemos el modelo con el conjunto training y comprobaremso su efectividad con el training

#-----------------------------------------------------------------------------
## CREACIÓN DEL MODEO CON LOS DATOS DEL CONJUNTO TRAINING
#-----------------------------------------------------------------------------
import statsmodels.formula.api as smf
lm = smf.ols(formula = "Sales~TV+Radio", data=training).fit()
print(lm.summary())
"""
Voy a comentar esto para saber cómo interpretar el summary:
    *R-squared y adj. R-squared > 0.9 ---> buenísimo
    *Fstatistic: 695.1 ---> grande, bueno
    *Prob(F-statistic)=P-valor: e-76: muy bajo, bueno
    *P>|t|: Los 3 (intercept, TV y Radio) p-valores son 0.000, genial
    *t: los tres estadísticos t son elevados respecto a los coefs: guay
    
Sales = 2.9465 + 0.0464*TV + 0.1828*Radio
"""
#-----------------------------------------------------------------------------
## VAIDACIÓN DEL MODELO CON EL CONJUNTO DE TESTING
#-----------------------------------------------------------------------------
#Predigo datos con los datos del testing, aunque estos datos de testing no se hayan usado para crear el modelo
sales_pred = lm.predict(testing)


#Veamos ahora cómo se diferencian esta predicción de os datos originales:
SSD=sum((testing["Sales"]-sales_pred)**2)
RSE = np.sqrt(SSD/(len(testing)-2-1))
print(RSE) #1.74258461494082 --> bajito, guay
sales_mean=np.mean(testing["Sales"])
error = RSE/sales_mean
print(error) #--> 0.12476739486449788, es decir, un 12% de los datos no se explica con el modelo.
            # en el modelo que hicimos en otro notebook con el 100% de los datos, el error también era ~12%
"""
Hemos validado con éxito nuestro modelo.
No presenta ningún problema de overfitting
Describe bien casi todo el conjunto de datos

RSE es peque ~1.7
Y el error también ~12%

En el análisis también estaría guay incluir el VIF cuando sorpechemos de muticolinealidad

"""

