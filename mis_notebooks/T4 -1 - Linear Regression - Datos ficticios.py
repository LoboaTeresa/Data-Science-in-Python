# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:48:48 2020
@author: 3A

ÍNDICE
------------------------------------------------------------------------------
*Modelos de regresión lineak
    ·Modelo con datos simulados
"""
#-----------------------------------------------------------------------------
## MODELOS DE REGRESIÓN LINEAL
#    *Modelo con datos simulados
# y = a + b * x <-- lo que busca el modelo
#generamos dos distribuciones de números: X e Y
# X: 100 valores distribuídos según una N(1.5 , 2.5) *media 1.5 y desv est=2.5
# Yestimada = Ye = 8 + 2.8 * x + e    *e=error de la estimación
# e estará distribuído según una normal N(0,0.8) * N de e siempre tiene media 0!!!
#-----------------------------------------------------------------------------
import pandas as pd
import numpy as np

#genero una normal de media 1.5 y desv típica 2.5 para generar 100 datos para x
x = 1.5 + 2.5 * np.random.randn(100)
res = 0 + 0.8 * np.random.randn(100)
y_pred = 8 +2.8 * x
y_act = 8 + 2.8 * x + res

data = pd.DataFrame(
    {
     "x": x_list,
     "y_actual" : y_act_list,
     "y_prediccion" : y_pred_list
     }    
)

print(data.head())


data["SSR"] = (data["y_prediccion"] - np.mean(y_act))**2
data["SSD"] = (data["y_prediccion"] - data["y_actual"])**2
data["SST"] = (data["y_actual"] - np.mean(y_act))**2

print(data.head())
SSR= sum(data["SSR"])
SSD= sum(data["SSD"])
SST= sum(data["SST"])

print(SSR)
print(SSD)
print(SST)
print(SSR+SSD) # si no me coincide exactamente con SST es por erores de redondeo
R= SSR/SST
print("R1 = %d" %R)

plt.show()
plt.hist(data["y_prediccion"] - data["y_actual"]) #me deberia salir una normal

#-----------------------------------------------------------------------------
## OBTENIENDO LA RECTA DE REGRESIÓN
#   * y = a + b * x
#   * b = sum(xi-x_m)*(yi-y_m)/sum(xi-x_m)^2
#   * a = y_m - b * x_m
#-----------------------------------------------------------------------------
x_mean = np.mean(data["x"])
y_mean = np.mean(data["y_actual"])

print("la media de x es x_mean = %d y la media de y es y_mean = %d" %(x_mean, y_mean))

data["beta_numerador"] = (data["x"]-x_mean)*(data["y_actual"]-y_mean)
data["beta_denom"] = (data["x"]-x_mean)**2
beta = sum(data["beta_numerador"])/sum(data["beta_denom"])
alpha = y_mean - beta * x_mean

print("alpha = %d y beta = %d" %(alpha, beta))
print("El modelo obtenido por regresión es: y = %d + %d * x "  %(alpha, beta))
data["y_model"] = alpha + beta * data["x"]
print(data.head(4))

SSR = sum((data["y_model"]- y_mean)**2)
SSD = sum((data["y_model"]- data["y_actual"])**2)
SST = sum((data["y_actual"]- y_mean)**2)
print("SSR = %d, SSD = %d, SST = %d "  %(SSR, SSD, SST))
R2 = SSR/SST
print(R2)
print("R2 = %d "  %float(R2))

#Para la representación gráfica:
    #1er método:
x_list = x.tolist()
y_pred_list = y_pred.tolist()
y_act_list = y_act.tolist()
y_mean = [np.mean(y_act) for i in range(1, len(x_list) + 1)] #para crear un array de len = lenx con todos los valores = media de y

import matplotlib.pyplot as plt

%matplotlib inline
plt.plot(x,y_pred,"y")
plt.plot(data["x"],data["y_model"], "b")
plt.plot(x, y_act, "ro")
plt.plot(x,y_mean, "g")
plt.title("Valor actual vs Predicción")

#-----------------------------------------------------------------------------
## ERROR ESTÁNDAR RESIDUAL
#-----------------------------------------------------------------------------
RSE = np.sqrt(SSD/(len(data)-2))
print(RSE)
print(np.mean(data["y_actual"]))
print(RSE / np.mean(data["y_actual"])) #porcentaje de error