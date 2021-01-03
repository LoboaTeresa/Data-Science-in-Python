# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:07:30 2020

ÍNDICE: K-MEDOIDES
------------------------------------------------------------------------------

*Distribuciones en forma de anillo
*Algoritmo con K-means
*Algoritmo con K-medoides
*Algoritmo de clustering espectral

"""
"""-----------------------------------------------------------------------------
DISTRIBUCIONES EN FORMA DE ANILLO
-----------------------------------------------------------------------------"""
from math import sin, cos, radians, pi, sqrt
import numpy.random as rnd
import numpy as np
import matplotlib.pyplot as plt

def ring(r_min=0, r_max=1, n_samples=360): #los valores que he establecido son "por defecto
    angle = rnd.uniform(0,2*pi, n_samples) #☻creo un set de n_samples angulos aleatorios entre 0 y 2pi
    distance = rnd.uniform(r_min, r_max, n_samples) 
    data = [] #cartesianas
    for a, d in zip(angle, distance):
        data.append([d*cos(a), d*sin(a)])
    return np.array(data)

data1 = ring(3,5)
data2 = ring(24,27)

data = np.concatenate([data1,data2], axis = 0)
labels = np.concatenate([[0 for i in range(0, len(data1))], [1 for i in range(0, len(data2))]]) #concateno un array de ceros y otro de unos

plt.scatter(data[:,0], data[:,1], c=labels, s=5, cmap="autumn")
plt.show()
"""-----------------------------------------------------------------------------
ALGORITMO CON KMEANS
-----------------------------------------------------------------------------"""
from sklearn.cluster import KMeans
km = KMeans(2).fit(data) #K=2, porque yo misma he creado los datos de forma que haya 2 cliusters
clust = km.predict(data)

plt.scatter(data[:,0], data[:,1], c = clust, s=5, cmap = "autumn")
plt.show()
#Este método no me permite distingir distintos cluster basados en anillos concentricos
#falla por las métricas que usan para calcular als distancias. Estas producen objetos que son convexos, es decir,
#si trazo una línea entre dos puntos de un mismo cluster, esta línea cae dentro del cluster

"""-----------------------------------------------------------------------------
ALGORITMO CON KMEDOIDES
-----------------------------------------------------------------------------"""
#Seleccionamos k puntos iniciales dentro del conjunto inical de clusters. Para cada uno de 
#los puntos del dataset calcularemos el centro del cluster más cercano con cualquier tipo 
#de métrica. La diferencia estará en  que el centro del cluster no quedará asignado al 
#baricentro de esos puntos sino que  quedará asignado a dicho punto en cuestión.
#Para cada puntd el cluster lo que haré sera intercambiar el cntro del cluster con el punto y
#calcularé la reducción en las distancias totales con respecto al centro del cluster a través
#de todos los miembros utilizando ese swap. Si no mejora, no lo cogeré. Iré iterando para todos
#y cada uno de os puntos del dataset y me quedaré siempre como centro uno de los puntos.
#Se llama técnica de los Kmedoides porque se trata de encontrar el punto que está más en medio

#Si yo establezco que haya 2 clusters, escogeré 2 puntos y esos dos puntos actuarán de medoides.

from pyclust import KMedoids
kmed = KMedoids(2).fit_predict(data) #2medoides. fit_predict: hace el modelo+la predicción
plt.scatter(data[:,0], data[:,1], c=kmed, s=5, cmap="autumn")
#no mejora :( Tendremos que cambiar el tipo de clustering
"""-----------------------------------------------------------------------------
ALGORITMO DEL CLUSTERING ESPECTRAL
-----------------------------------------------------------------------------"""
from sklearn.cluster import SpectralClustering

clust = SpectralClustering(2).fit_predict(data)
plt.scatter(data[:,0], data[:,1], c = clust, s=5, cmap = "autumn")
plt.show()
#esto funciona muy bien

#si no conozco k, yo primero haría una propagación de la afinidad porque no necesita una k
#si puedo estimar la k yo solo y facilmente?bien. podré usar la distancia euclidea para clasificar por clusters??si--> uso kmeans
#si no puedo usar la métrica euclidea pueda quizas sirva kmedoids
#si se pueden separar por clusters mediante rectas, podremos usar alguna técnica como la de super_vectors_machine
#si los datos no son linealmente separables en clusters, usamos el clustering espectral por ejemplo. 

"""
Podemos estimar la k?
    *No: Propagación de la afinidad
    *Si: Podemos usar la distancia Euclídea?
        -Sí: Kmeans
        -No: Sirve de algo buscar los valores centrales?
            ·Sí: K medoides
            ·No: Son los datos linealmente separables?
                + Sí; Clustering aglomerativo
                +No: Clustering Espectral
                

"""