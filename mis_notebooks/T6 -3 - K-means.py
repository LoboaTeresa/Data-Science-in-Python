# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:39:15 2020

@author: 3A

ÍNDICE: EL MÉTODO DE K-MEANS
------------------------------------------------------------------------------

*Recuperar el número de clusters y sus elementos
"""
"""-----------------------------------------------------------------------------
CLUSTERING JERÁRQUICO DE DATOS ALEATORIO
-----------------------------------------------------------------------------"""

import numpy as np

data = np.random.random(90).reshape(30,3) #reorganizo en 30 filas 3 columnas
print(data)

#El método K-means tiene un problema porque hay que elegir de antemano el número de clusters que queremos.

#1) mirando el data, elegimos k puntos y los definimos como los k centroides iniciales
#por ejemplo me dice el jefe que k=2: elijo entonces 2 centroides aleatorios
c1 = np.random.choice(range(len(data))) #me elige un númer aleatorio entre el 0 y el 29
c2 = np.random.choice(range(len(data)))
clust_centers = np.vstack([data[c1], data[c2]]) #vstack me pone en un array c2 debajo de c1
print(clust_centers)

from scipy.cluster.vq import vq
vq(data, clust_centers) #esto nos da dos arrays.
#   La 1 array es de 0s(dato correspondiente al 1 cluster) y unos (datos correpondientes al segundo cluster)
#   La 2 array nos da las distancias de cada uno de los datos al centroide del cluster 1 o 2

from scipy.cluster.vq import kmeans
kmeans(data, clust_centers) #en la primera fila me salen las coordenadas del baricentro del cluster 1
#                           y en la segund ala del segundo. el último número solitario
#                            es la suma de las (distancias al baricentro)al cuadrado normalizada
kmeans(data, 2) #me hace un k means conociendo los baricentros o conociendo el k=2.

