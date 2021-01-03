# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 23:57:03 2020
@author: 3A

ÍNDICE
------------------------------------------------------------------------------
*Plots y visualización de los datos
*Histogramas de frecuencias
*Boxplot o diagrama de caja y bigotes
"""
#-----------------------------------------------------------------------------
##PLOTS Y VISUALIZACIÓN DE LOS DATOS
#-----------------------------------------------------------------------------
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt #usaré el subpaquete pypot de la libreria matplotlib

mainpath="C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master/datasets"
filepath = "customer-churn-model/Customer Churn Model.txt"
fullpath = os.path.join(mainpath, filepath)

data = pd.read_csv(fullpath)

print(data.head())


#savefig("path_donde_guardar_imagen) ·guardar el plor como imagen

#**SCATTER PLOT
#--------------
data.plot(kind="scatter", x="Day Mins", y="Day Charge")
data.plot(kind="scatter", x="Night Mins", y="Night Charge")
data.plot(kind="scatter", x="Day Mins", y="Day Charge")

figure, axs = plt.subplots(2,2,sharey=True, sharex=True) #creo una matriz 2x2 de plots donde he especificado que comparten escalas de eje x e y
data.plot(kind="scatter",x="Day Mins", y="Day Charge", ax=axs[0][0])
data.plot(kind="scatter", x="Night Mins", y="Night Charge", ax=axs[0][1])
data.plot(kind="scatter", x="Day Calls", y="Day Charge", ax=axs[1][0])
data.plot(kind="scatter", x="Night Calls", y="Night Charge", ax=axs[1][1])
plt.show() #sin esto, por alguna razon, no se me muestra el grávico (2,2) y en su lugar se me plotea el histograma siguiente)
#**HISTOGRAMAS DE FRECUENCIAS
#----------------------------

plt.hist(data["Day Calls"], bins = 20) #20 barras
plt.xlabel("Número de llamadas al día") #Etiqueta eje X
plt.ylabel("Frecuencua") #Etiqueta eje Y
plt.title("Histograma del número de llamadas al día")

plt.show() #vuelve a ser necesario
plt.hist(data["Day Calls"], bins = [0,30,60,90,120,150,180]) #señalo donde quiero hacer las separaciones de las barras

#Mirarme la REGLA DE STURGES para saber cuántas divisiones se han de hacer en un histograma.
sturges=int(np.ceil(1+ np.log2(3333)))
#plt.hist(data["Day Calls"], bins =sturges) #número de barras recomendado por la regla de sturges. 3333=número de filas=tamaño muestra


#**BOXPLOT O DIAGRAMA DE CAJA Y BIGOTES
#--------------------------------------
#plt.boxplot(data["Day Calls"])
#plt.ylabel("Número de llamadas diarias")
#plt.title("Boxplot de las llamadas diarias")