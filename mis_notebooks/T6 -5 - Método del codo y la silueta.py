# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:33:55 2020

@author: 3A

ÍNDICE: EL MÉTODO DEL CODO Y DE LA SILUETA DEL CLUSTERING
------------------------------------------------------------------------------

*Método de la  silueta
*Representación del codo
"""
"""-----------------------------------------------------------------------------
MÉTODO DE LA SILUETA
-----------------------------------------------------------------------------"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.cluster import KMeans #para hacer el método k means
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score

#Creo mis datos sin ton ni son:
x1 = np.array([3,1,1,2,1,6,6,6,5,6,7,8,9,8,9,9,8])
x2=([5,4,5,6,5,8,6,7,6,7,1,2,1,2,3,2,3])
X=np.array(list(zip(x1,x2))).reshape(len(x1), 2) #relaciono los elementos de x1 con los de x2 y con el reshape
                                               #hago reshape para tener len(x1) filas y 2 columnas
print(tuple(X))
print("hola")
print(sum(X))
print(3+1+1+2+1+6+6+6+5+6+7+8+9+8+9+9+8)
plt.plot()
plt.xlim([0,10])
plt.ylim([0,10])
plt.title("Dataset a clasificar")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x1,x2)
plt.show()

max_k =10 #max número de iteraciones (máximo número de clusters que vamos a crear)
K = range(1, max_k) #este range va el 1 al (max_k)-1. Crearé primero 1 cluster, luego 2... hasta max_k -1
ssw = [] #suma de los cuadrados internos como un array vacío
color_palette = [plt.cm.spectral(float(i)/max_k) for i in K] # array con 10 colores, uso la funcion cm=color manager. 
#centroide del dataset completo. El for i in K replico la operacion para cada valor de K, por lo que centroids es un array con todos sus valores iguales (antes de empezar el bucle)
centroid = [sum(X)/len(X) for i in K] 
#suma de los cuadrados totales: minima distancia entre los datos X y el centroide usando la distancia euclidea. 
#axis=1 es para definir que el min se elija por filas.
#cdist me da una matriz de dimensiones (dim(X)x dim(centroid))=(len(X) x len(centroids))
sst = sum(np.min(cdist(X,centroid,"euclidean"), axis=1)) 

                                                                                                           
#Algoritmo para hace kmeans para cada k del range que he definido
for k in K:
    kmeanModel =KMeans(n_clusters = k).fit(X)
    
    #defino ciertas variables para simplificarme el script
    centers = pd.DataFrame(kmeanModel.cluster_centers_) #convierto esto en DF para porder trabajr con ello como un df
    labels = kmeanModel.labels_
    
    #calculamos la distancia de los puntos de x al centroide de sus respectivos clusters con la métrica euclídea y la suma se hace por filas (axis=1)
    ssw_k = sum(np.min(cdist(X, kmeanModel.cluster_centers_, "euclidean"),axis=1))
    ssw.append(ssw_k)
    
    #habrá tantas etiquetas de color como puntos
    label_color = [color_palette[i] for i in labels]
    
    #Fabricaremos una silueta para cada cluster. 
    if 1<k<len(X): #por seguridad, no hacemos silueta si k=1 o si k = len(x) (en esos casos el metodo silhouete no funciona)
        #crear un subplot de una fila y 2 columnas (xk voy a pintar la silueta por un lado y la segmentación del cluster por otro)
        fig, (axis1,axis2) = plt.subplots(1,2) #(subplots de 1 fila y 2 columnas)
        fig.set_size_inches(20,8) #configuro el tamaño de todo el plot en pulgadas (inches)
        
        #El primer subplot contendrá la silueta, que puede tener valores desde -1 a 1
        #El nuestro caso ya controlamos que los valores están entre -0.1 y 1 y limitare los ejes (sino, por defecto,lo limito entre -1 y 1)
        axis1.set_xlim([-0.1,1.0]) #fijamos un límite de la variable del eje x entre -0.1 y 1
        #El número de clusters a insertar determinará el tamaño de cada barra
        #El coeficiente (n_clusters+1)*10 será el espacio en blanco que dejaremos entre
        #siluetas individuales de cada cluster para separarlas.
        axis1.set_ylim([0,len(X)+ (k+1)*10]) #límites eje vertical: haremos len(x) cajitas y además (k+1)*10 las dejaremos en blanco como separación de cajitas
        
        silhouette_avg =silhouette_score(X,labels) #calculamos el promedio de las siluetas de los datos
        print("*Para k=", k, "el promedio de la silueta es de : ", silhouette_avg)
        sample_silhouette_values = silhouette_samples (X, labels) 
        #recordamos que el promedio de las siluetas se usa para ver si el número de clusters es apropiado
        #Si hay algun valor de las siluetas muy infarior al promedio, habria que aumentar el valor de k.
        #Seguiríamos aumentando k hasta que no haya ninguna silueta muy dispar.
        
        y_lower = 10 #para dentro del bucle que va a recorrer cada cluster me funcione y lo calcule ??
        for i in range (k):
            #Agregamos la silueta del cluster i-ésimo
            ith_cluster_sv = sample_silhouette_values[labels==i] # me quedo con los valores de las siluetas con la label=i, es decir, del cluster i
            print("    - Para i = ", i+1, "la silueta del cluster vale : ", np.mean(ith_cluster_sv))
            #ordenamos descendientemente las siluetas del cluster i-ésimo
            ith_cluster_sv.sort()
            
           #calculamos dónde colocar la primera silueta en el eje principal
            ith_cluster_size = ith_cluster_sv.shape [0] #el tamaño será el número de filas ([0])
            y_upper = y_lower + ith_cluster_size #voy apilano siluetas de clusters
            
            #elegimos el color del cluster
            color = color_palette[i]
            
            #pintamos la silueta del cluster i-ésimo. Representamos el valor 0 con el valor de la siluete del cluster i-ésimo
            axis1.fill_betweenx(np.arange(y_lower, y_upper),
                                0, ith_cluster_sv, facecolor = color, alpha = 0.7) #alpha define la transparencia
            
            #Etiquetamos dicho cluster con el número en el centro
            axis1.text(-0.05, y_lower + 0.5 * ith_cluster_size, str(i+1)) #escriblo en (-0.05, centro del cluster=y_lower+ith_cluster_size) la etiqueta del cluster i+1
            
            #Calculamos el nuevo y_lower para el siguiente cluster del gráfico
            y_lower = y_upper + 10 #dejamos vacías 10 posiciones sin muestra (espaciado entre silutas de clusters distintos)
       
        axis1.set_title("Representación de la silueta para k =%s" %str(k))
        axis1.set_xlabel("S(i)")
        axis1.set_ylabel("ID del Cluster")
        
        #Fin de la representación de la silueta
    #Plot de los k-means con los puntos respectivos
    plt.plot()
    plt.xlim([0,10])
    plt.ylim([0,10])
    plt.title("Clustering para k = %s" %str(k))
    plt.scatter(x1,x2, c=label_color)
    plt.scatter(centers[0], centers[1], c=color_palette, marker = "x")#scater plot con los centroides 
    plt.show()
"""-----------------------------------------------------------------------------
REPRESENTACIÓN DEL CODO
-----------------------------------------------------------------------------"""
plt.plot(K,ssw, "bx-")
plt.xlabel("SSw(k)")
plt.ylabel("La técnica del codo para encontrar el k óptimo")
plt.show()

#Según la técnica de la silueta, k óptimo es 3 (las siluetas de los clusters me salen cercanos a la silueta promedio)
#mirando el plot del codo, veo que los dos métodos coinciden en que k=3 es el número óptimo de clusters
"""-----------------------------------------------------------------------------
REPRESENTACIÓN DEL CODO NORMALIZADO
-----------------------------------------------------------------------------"""
plt.plot(K,1-ssw/sst, "bx-")
plt.xlabel("1-norm(SSw(k))")
plt.ylabel("La técnica del codo normalizado para encontrar el k óptimo")
plt.show()