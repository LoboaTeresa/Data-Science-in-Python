# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:01:12 2020

@author: 3A


    
"""
"""-----------------------------------------------------------------------------
PROPAGACIÓN DE LA AFINIDAD
-----------------------------------------------------------------------------"""
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs #para generar aleatoriamente puntos en forma de distribución gausiana

#Generamos unos datos aleatorios
#   -Generamos 3 centros:
centers =[[1,1],[-1,-1],[1,-1]]
#   -Genero las X y las etiquetas con la función make_blobs
#       *Se crearán 300 muestras. 
#       *Las distrib estarán centradas en los centros que hemos descrito arriba. Como son 3 centros, habrá 3 distribuciones.
#       *La desviación estándar reapecto a los centros ? 0.5
#       *random_state es como lo de fijar la semilla seed.

X, labels = make_blobs(n_samples=300, centers=centers, cluster_std=0.5, random_state=0)

import matplotlib.pyplot as plt
from itertools import cycle

#Hago un scatter plot para visualizar los datos rápidamente:
#   -Representaré todas las filas de la columna 0 de X frente a todas las filas de la columna 1 de X.
#   -Las clases que servirán para la coloración las definimos como las etiquetas c=labels
#   -s=50 tiene que ver con el tamaño de los puntos
#   - Elegimos una paleta de colores predefinida "autum" pero hay otras más
plt.scatter(X[:,0], X[:,1], c = labels, s = 50, cmap = "autumn")

#calculamos l afinidad:
af = AffinityPropagation(preference=-50).fit(X) 
cluster_center_ids = af.cluster_centers_indices_ #me dice de las distribuciones quehe generado, la posición  del punto que se supone que es el centro.
labels = af.labels_ #etiquetas de cada punto, es decri, a qué cluster pertenece cada punto
n_clust = len(cluster_center_ids) #número de clusters que se han generado. Me salen , que por la forma en que hemos creadolos datos "aleatorios" tiene sentido

#vamos a crear un algoritmo (función) que me muetsre en pantalla toda esta info últil
def report_affinity_propagation(X):
    af = AffinityPropagation(preference=-50).fit(X) 
    cluster_center_ids = af.cluster_centers_indices_ #me dice de las distribuciones que he generado, la posición  del punto que se supone que es el centro.
    clust_labels = af.labels_  #etiquetas de cada punto, es decri, a qué cluster pertenece cada punto
    n_clust = len(cluster_center_ids) #número de clusters que se han generado.
    print("Número estimado de clusters: %d" %n_clust)
    print("Homogeneidad: %0.3f" %metrics.homogeneity_score(labels, clust_labels)) #Homogeneidad del clustering (cuántas han sido correctamente clasificadas) con 3 cifras decimales (%0.3f). meto las labels originales (labels) y las que me da el modelo de los clusters (clust_labels)
    print("Completitud: %0.3f" %metrics.completeness_score(labels, clust_labels)) #Tasa de verdaderos positivos y negativos
    print("V-measure: %0.3f" %metrics.v_measure_score(labels, clust_labels))
    print("R2 ajustado: %0.3f" %metrics.adjusted_rand_score(labels, clust_labels))
    print("Información mútua ajustada: %0.3f" %metrics.adjusted_mutual_info_score(labels, clust_labels))
    print("Coeficiente de la silueta: %0.3f" %metrics.silhouette_score(X, labels, metric="sqeuclidean"))
    plt.figure(figsize=(16,9)) #Definimos que el plot sea de tamaño 16/9 para que ocupe toda la pantalla
    plt.clf() #para establecer el tipo de dibujo
    
    colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk") #como no sé cuantos clusters va a haber, genero un ciclo largo: blue green red cian magenta yellow k=negro y repito
    for k, color in zip(range(n_clust), colors):
        class_members = (clust_labels==k) #miembros del clustering k-ésimo
        clust_center = X[cluster_center_ids[k]]
        plt.plot(X[class_members,0], X[class_members,1], color + '.') #represento los datos de la fila "class_members", todos en el mismo color (col) y como puntos
        plt.plot(clust_center[0], clust_center[1], 'o', markerfacecolor = color, markeredgecolor="k", markersize=14)
        for x in X[class_members]: #para poner las flechas esas
            plt.plot([clust_center[0], x[0]], [clust_center[1], x[1]], color)
            
    plt.title("Número estimado de clusters %d" %n_clust)
    plt.show()
report_affinity_propagation(X)
    
        
    