# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 17:14:52 2020
@author: 3A

ÍNDICE
------------------------------------------------------------------------------
*Regresión lineal simple
    ·Paquete statsmodel para regresión simple
*Regresión lineal múltiple
    ·Paquete statsmodel para regresión múltiple
*Multicolinealidad: cuando 2 variabres predoras están relacionadas, malo
    ·Método VIF (Factor Inflación de la Varianza) para saber cuál d las 
     variables predictoras relacionadas elimino
"""
#-----------------------------------------------------------------------------
## REGRESIÓN LINEAL SIMPLE
#   EL PAQUETE STATSMODEL PARA REGRESIÓN LINEAL
#-----------------------------------------------------------------------------
import pandas as pd
import numpy as np

mainpath="C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master/datasets"
filepath = "ads/Advertising.csv"
fullpath = mainpath +"/" + filepath

data = pd.read_csv(fullpath)
print(data.head())

import statsmodels.formula.api as smf  #paquete para crear regresión lineal
lm = smf.ols(formula = "Sales~TV", data = data).fit()
#La formula indica que sales es la función predictora en función de Tv.
#data me dice los datos usados
#.fit da la orden de que se ajusten los datos al modelo más apropiado

print(lm.params)
#El modelo lineal predictivo sería:    Sales =  7.032594 +  0.047537 *TV
print(lm.pvalues) #me dalen los p-values del modelo. Si son muy pequeños, beta!=0 y es un muy buen modelo
print(lm.rsquared_adj) #
print(lm.summary()) #me da un análisis del modelo de regresión
                    #Un p valer superior al 0.85 esmalo (P>|t|)

#puedo usar mi modelo para hacer predicciones de las Sales.
sales_pred = lm.predict(pd.DataFrame(data["TV"])) #uso la columna de TV pero la conivierto en dataframe para usar esta función
print(sales_pred)

import matplotlib.pyplot as plt

#%matplotlib inline
data.plot(kind="scatter", x = "TV", y = "Sales")
plt.plot(pd.DataFrame(data["TV"]), sales_pred, c="red", linewidth =2)

#Error estandar residual:
data["sales_pred"] =  7.032594 +  0.047537 * data["TV"]
data["RSE"] = (data["Sales"]-data["sales_pred"])**2
SSD =sum(data["RSE"])
RSE = np.sqrt(SSD/(len(data)-2))

sales_m =np.mean(data["Sales"])
error_promedio=RSE/sales_m #--> me da el porcentaje de datos que no quedan bien explicados por el modelo


plt.show()
plt.hist((data["Sales"]-data["sales_pred"]))
    
#-----------------------------------------------------------------------------
## REGRESIÓN LINEAL MÚLTIPLE

#    El paquete statsmodel para regresión múltiple
#Sales~TV, Sales~Newspaper, Sales~Radio, Sales~TV+Newspaper, Sales~TV+Radio, Sales~Radio+Newspaper, Sales~TV+Newspaper+Radio

#Para saber qué variables son buenas se calculará el p-valor de cada variable y
#si es demasiado grande indicara que no es buena predictora y se eliminará del modelo
#Cuanto menor sea el p-valor de una variable, mayor será su su aportación al modelo
#Nos marcaremos un p-valor límite y las varioles con p-valor superior, las ignoramos en nuestro modelo

#MÉTODO CONSTRUCTIVO: es con un modelo vacío y al añadir una variable, observar si
#el error Estándar Residual (RSE) aumenta o disminuye. Si disminuye, la añadimos al 
#modelo, además de comprobar que el p-valor de esa variable es inferior a un límite 
#fijado

#MÉTODO DESTRUCTIVO: el modelo inicial tiene todas las posibles variables añadidas 
#y las va dscartando una a una. Si el p-valor es muy grande y el RSE no aumenta al 
#ser eliminada, quitamos definitivamente esa variable.

#Estos métodos están automatizados en python, pero emecemos haciéndolo manualmente
#-----------------------------------------------------------------------------

#MÉTODO CONSTRUCTIVO: mi modelo inicial tendra solo Sales ~ TV
print("Info sobre modelo Sales~TV")
print(lm.pvalues) 
print(lm.rsquared_adj) #0.6099
print(RSE) # 3.258
print(error_promedio) #0.23238--> 23%

#Añado ahora la variable newspaper

lm2 = smf.ols(formula="Sales~TV+Newspaper", data = data).fit()
print("Info sobre modelo Sales~TV+Newspaper")
print(lm2.params) #Sales = 5.77 + 0.0469TV + 0.044219Newspaper
print(lm2.pvalues) #Pvalor de nespaper es E-5, sigue siendo bastante peque 
print(lm2.rsquared_adj)  #R^2 aumenta(ok) p-value de Newspaper peque (ok)
sales_pred = lm2.predict(data[["TV", "Newspaper"]])

SSD=sum(data["Sales"]-sales_pred)**2
data["RSE"]=SSD
RSE =np.sqrt(SSD/(len(data)-2-1)) #2 variables predictoras
print(RSE) #Baja respecto al modelo con Sales~TV solo (bien)
sales_m =np.mean(data["Sales"])
error_promedio=RSE/sales_m #--> me da el porcentaje de datos que no quedan bien explicados por el modelo
#R^2 sube, p-valor sigue siendo peque, RSE baja--> todo bien pero la mejora no es muy grande


#Añado radio al modelo existente
lm3 = smf.ols(formula="Sales~TV+Radio", data = data).fit()
print("Info sobre modelo Sales~TV+Radio")
print(lm3.summary()) 
sales_pred = lm3.predict(data[["TV", "Radio"]])
SSD=sum((data["Sales"]-sales_pred)**2)
data["RSE"]=SSD
RSE =np.sqrt(SSD/(len(data)-2-1)) #2 variables predictoras
print(RSE) #Baja respecto al modelo con Sales~TV solo (bien)
error_promedio=RSE/sales_m
print(error_promedio)

#R(sales~TV)= 0.612                #R(sales~TV+RADIO)=0.897  (Bien)
#Prob(F)(sales~TV)=  1.47e-42      #Prob(F)(sales~TV+RADIO)=4.83e-98 (Muy bien)
#beta(sales~TV)= 0.475             #R(sales~TV+RADIO)=0.188 (grande)(bien)
#RSE(sales~TV)= 3.258              #RSE(sales~TV+RADIO)=1.6813 (baja, bien)
#Error promedio(sales~TV)= 0.23238 #err promedio(sales~TV+RADIO)=0.119(baja, bien)

lm4=smf.ols(formula="Sales~TV+Radio+Newspaper", data = data).fit()
sales_pred = lm4.predict(data[["TV", "Radio", "Newspaper"]])
SSD=sum((data["Sales"]-sales_pred)**2)
data["RSE"]=SSD
RSE =np.sqrt(SSD/(len(data)-3-1)) #3 variables predictoras
print(RSE) #Baja respecto al modelo con Sales~TV solo (bien)
error_promedio=RSE/sales_m
print(error_promedio)

#Pvalor de Newspaper es muy grande(0.86)(malo)
#F-stadistics baja(bien)
#El intervalo de confi incluye al 0 (malo)
#no es bueno añadir Newspaper

#-----------------------------------------------------------------------------
## MULTICOLINEALIDAD
#-----------------------------------------------------------------------------
#Alude a la correacion entre variables predictoras del modelo. Es la razón por 
#la que al añadir una nueva variable, nuestro modelo puede empeorar.
#Guay que una variable predictora tenga relacion con la predictiva. Pero que dos 
#variables predictoras interaccionen entre sí... malo. Habrá que ver cuál de
#las dos conviene incluir en el modelo, pero no las dos.

data_ads = pd.read_csv(fullpath)
plt.matshow(data_ads.corr())  #vemso que newspaper y radio están bastante relacionadas 
print(data_ads.corr()) #fíjate que la correlación entre radio y newspaper es relativamente alta

#Al añdir el periódico a mi modelo, empeora. Como hemos visto, hay una multicolinealidad
#entre las variables predictoras. Usaremos el método VIF para ver cuál de las variables
#eliminamos para resolver este problema.

#MÉTODO VIF: escribir la variable que nos da problemas en función de otras variables predictoras
#   1)Escribimos una variable predictora en función de las otras (modelo lineal)
#   2)Con el R^2 de este modelo calculamos el "factor ínflación de la varianza" o VIF
        #VIF = 1/ (1-R^2)
#   3)VIF = 1 (variables no tienen correlación, super bien)
#     VIF < 5 (Variables relacionadas moderadamente con otras variables predictoras, pero todavía pueden formar parte del modelo)
#     VIF > 5 (Variables altamente correlacionadas, hay que eliminarla del modelo)

#Newspaper ~ TV + RADIO --> VIF = 1/ (1-R^2)
lm_n = smf.ols(formula= "Newspaper~TV+Radio", data = data_ads).fit()
rsquared_n = lm_n.rsquared
VIF=1/(1-rsquared_n)
print(VIF) #1.1451873787239288

#TV ~ Newspaper + RADIO  --> VIF = 1/ (1-R^2)
lm_tv = smf.ols(formula= "TV~Newspaper+Radio", data = data_ads).fit()
rsquared_tv = lm_tv.rsquared
VIF=1/(1-rsquared_tv)
print(VIF) #1.0046107849396502 

#Radio ~ TV + Newspaper  --> VIF = 1/ (1-R^2)
lm_r = smf.ols(formula= "Radio~TV+Newspaper", data = data_ads).fit()
rsquared_r = lm_r.rsquared
VIF=1/(1-rsquared_r)
print(VIF) #1.1449519171055353

#Sí es cierto que el VIF del newspaper no es >5, pero ya vimos que el modelo con 
#las tres variables predictoras (lm4) era peor que aquel con solo radio y tele (lm3)
#además, aunque sea poco, VIF_radio<VIF_newspaper --> Quitamos newspaper y suamos lm3
