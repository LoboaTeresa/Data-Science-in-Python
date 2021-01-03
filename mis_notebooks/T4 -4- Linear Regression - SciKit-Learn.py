# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 17:21:05 2020

@author: 3A

ÍNDICE
------------------------------------------------------------------------------
*Paquete SciKit-learn para la regresión lineal y selección de rasgos
    ·
"""
import pandas as pd
import numpy as np

mainpath="C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master/datasets"
filepath = "ads/Advertising.csv"
fullpath = mainpath +"/" + filepath

data = pd.read_csv(fullpath)
print(data.head())
#-----------------------------------------------------------------------------
## REGRESIÓN LINEAL MÚLTIPLE AUTOMATIZADA con libresís SkLearn
#-----------------------------------------------------------------------------
#En T4.2 hablamos de 2 métodos (constructivo y destructivo) para seleccionar las
#variables mas apropiadas para desarrollar un modelo de regresión lineal múltiple
#En T4.2 estudiamos el desarrollo manual de estos métodos. Ahora los vamos a 
#automatizar con la librería scikit-learn

from sklearn.feature_selection import RFE #(RFE-->recursive feature elimination)
from sklearn.svm import SVR #svm-->superverton machina, otra técnica para crear modelos
                            #SVR--> Para modelos lineales
                            
#Creo un array con los identificadores de las variables predictoras
feature_cols=["TV", "Radio", "Newspaper"]
X=data[feature_cols] #variables predictoras recogidas en x
Y=data["Sales"] # variable predictiva recogida en y

estimator = SVR(kernel="linear") #el kernel me indica el tipo de modelo que quiero crear, lineal
selector = RFE(estimator, 2, step=1) #quiero 2 variables predictivas para mi modelo
                                     #quiero que lo haga en 1 solo paso (Step=1)
selector=selector.fit(X,Y) #Para que cree mi modelo con las variables predictoraas y predictivas que he definido en x e y
print(selector.support_) # me sale {True, True, False}--> en mi modelo he usado las 2 primeras variables solo (TV y Radio)

print(selector.ranking_) #ordena las variables por significación.
                         #me sale {1,1,2} las variables con menor índice serán más significativas
  
#Ahora que sabemos qué variable nos conviene descartar, hacemos la regresión lineal                         
from sklearn.linear_model import LinearRegression
X_pred = X[["TV", "Radio"]] #array con las variables predictoras buenas
lm=LinearRegression() #creo el modelo lineal
lm.fit(X_pred,Y) #indico que lm debe ajustar las variables pred a la variable y
print(lm.intercept_) #alfa
print(lm.coef_) #coefs de radio y tv
print(lm.score(X_pred,Y)) #R^2 ajustado

#esto último lo podíamos hacer con la otra librería de T4-3