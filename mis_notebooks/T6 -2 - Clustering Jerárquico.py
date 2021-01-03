# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:37:01 2020

@author: 3A

ÍNDICE: CLUSTERING JERÁRQUICO
------------------------------------------------------------------------------
Antes de empezar, pongamos un poco de notación para hablar todos el mismo idioma:
    -X: dataset (array de n x m) de puntos a clusterizar
    -n: números de datos
    -m: números de rasgos
    -Z: array de enlace del cluster con la info de las uniones
    -k: número de clusters
*Representación gráfica de un dendrograma
*Truncamiento del dendrograma
*Dendrograma personalizado
*Corte automático del dendrograma (selección del número de clusters)
    -Método del codo
    -Método del codo variante
*clustering jerárquico de datos aleatorios
*Representación gráfica de un denrograma

*Recuperar el número de clusters y sus elementos
"""
"""-----------------------------------------------------------------------------
CLUSTERING JERÁRQUICO DE DATOS ALEATORIO
-----------------------------------------------------------------------------"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage 

#En lugar de usar unos datos concretos vamos a crearlos aleatoriamente
np.random.seed(47110) #fijo la semilla
#creo un array de 2 columnas (por eso lo de multi) con datos aleatorios siguiendo una distib normal
a = np.random.multivariate_normal([10,0],[[3,1],[1,4]], size = [100,]) #array con x entre 0 y 10 e y 
b = np.random.multivariate_normal([0,20],[[3,1],[1,4]], size = [50,])
#np.random.multivariate_normal([media1, media2],[matrix de cov, ha de ser simétrica], tamaño) #buscar en gogle que viene bien explicado
X= np.concatenate((a,b)) #tendré 150 filas y 2 columnas, es decir 150 datos
print(X.shape)
plt.scatter(X[:,0], X[:,1]) #en la x mostramos todas las filas de la columna 0 y en la Y, todas las filas de la col 1
plt.show()

#ahora hacemos el clustering jerárquico
Z = linkage(X, "ward") #Matriz de enlace de dimension n-1 (n datos)

from scipy.cluster.hierarchy import cophenet #
from scipy.spatial.distance import pdist #

c, coph_dist = cophenet(Z, pdist(X)) 
#pdist me calcula la distancias entre los datos originales.
#c es el coef de cophenet. me dice c=0.98--> 98% de la conservación de las distancias originales respecto a los clusteres originales.

print(Z[0]) #me muetsran los dos datos más cercanos, la distancia entre ellos y el número de elementos del cluster
Z[:20] #20 primeros elementos
#el elemento 152 de Z ya ha incorporado 3 elementos. Dice que son el 152 y el 62. Si queremos conocer los
#datos originales que ha juntado:
print(Z[152-len(X)]) #cluster 152
#ahora sé que los datos que ha juntado son el 33, 62 y 68
print(X[[33,62,68]]) #veo que si, son datos muy parecidos

idx = [33,62,68]
plt.figure(figsize=(10,8))
plt.scatter(X[:,0],X[:,1]) #pintar todos los untos
plt.scatter(X[idx, 0], X[idx,1], c='r')#destacamos en rojo los puntos interesantes y vemos que tiene sentido qye hayan sido aglutinados
plt.show()
    
"""-----------------------------------------------------------------------------
REPRESENTACIÓN GRÁFICA DE UN DENDROGRAMA
-----------------------------------------------------------------------------"""
plt.figure(figsize=(25,10))
plt.title("Dendrograma del clustering jerárgico")
plt.xlabel("índices de la muestra")
plt.ylabel("Distancias")
dendrogram(Z,leaf_rotation=90.,leaf_font_size=8.0, color_threshold=0.1*180)
#dendrogram(Z,leaf_rotation=90.,leaf_font_size=8.0,labels=array_etiquetas)
print(Z[-4:,]) #Muestro las últimas 4 filas, todas las columnas
"""-----------------------------------------------------------------------------
TRUNCAMIENTO DE UN DENDROGRAMA
-----------------------------------------------------------------------------"""
plt.figure(figsize=(25,10))
plt.title("Dendrograma del clustering jerárgico truncado")
plt.xlabel("índices de la muestra")
plt.ylabel("Distancias")
#muestro solo los últimos p clusters que se unen en el dendrograma--> truncate_mode
#muestro los elementos de los nodos hoja: show_leaf_counts si lo ponemos en true nos dice el numero de elementos de la rama
#contraigo los clusters pequeños: show_contracted --> si ponemso true nos sirve para visualizar cuantos elementos nos hemos comido.
dendrogram(Z,leaf_rotation=90.,leaf_font_size=8.0, color_threshold=0.1*180, truncate_mode="lastp", p=10, show_leaf_counts=False, show_contracted=True)
plt.show()
"""-----------------------------------------------------------------------------
DENDROGRAMA PERSONALIZADO
-----------------------------------------------------------------------------"""
def dendrogram_tune(*args, **kwargs):
    max_d=kwargs.pop("max_d",None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold']=max_d
    annotate_above =kwargs.pop('annotate_above',0)
    
    ddata = dendrogram(*args, **kwargs)
    
    if not kwargs.get('no_plot', False):
        plt.title("Clustering jerárgico con Dendrograma truncado")
        plt.xlabel("Índice del Dataset(o tamaño del cluster)")
        plt.ylabel("Distancia")
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3]) #i=index
            y = d[1] #d=distance
            if y> annotate_above: #si el valor de y > que lo anotado, pondremos un circulito de color c y añadiremos una anotación
                plt.plot(x,y,'o', c=c)  #c=color
                plt.annotate('%.3g' %y, (x,y), xytext=(0,-5), textcoords="offset points", va="top", ha="center")
    if max_d:
        plt.axhline(y=max_d, c='k')
    return ddata

dendrogram_tune(Z,truncate_mode='lastp', p=12, leaf_rotation=90., leaf_font_size=12., show_contracted=True, annotate_above=10, max_d=23)

#el max_d me sirve para hacer un corte y definir el numero de dendrogramas que quiero. 
#¿Cómo mas o menos automatizar esto?
"""-----------------------------------------------------------------------------
CORTE AUTOMÁTICO DEL DENDROGRAMA
-----------------------------------------------------------------------------"""
# definimos: inconsistency_i = (h_i-avg(h_ij)/std(h_j)   h= altura, que es una distancia
#esto tiene python u paquete integrado
from scipy.cluster.hierarchy import inconsistent  #para calcular la inconsistencia de los clusters que se han formado
depth = 5 #sirve para determinar con cuantos niveles de union anteriores tiene que hacer la media o la std
incons = inconsistent(Z, depth)
print(incons) # array([promedio, desviacion estandar, numero de elementos, factor 
#inconsistencia para cada una de las uniones]). Lo que suele ser interesante son las 
#últimas uniones:
print(incons[-10:]) #una incosistencia de unos 5 ya es grnade.
#la inconsistencia es muy dependiente de la depth, eso es algo negativo.

"""MÉTODO DEL CODO
-----------------------------------------------------------------------------"""
#localiza la mayor altura de union. Es un método visual.
last = Z[-10:,2] #(columna 3 de Z)
last_rev =last[::-1] #me revierte el orden de los elementos de la lista
print(last)
print(last_rev)

plt.show()
idx = np.arange(1,len(last)+1) #◙arrange me da valores equiespaciados desde 1 hasta len(last)+1
plt.plot(idx, last_rev)

acc = np.diff(last,2) # se restan los valores last[i+1]-last[i] dos veces
acc_rev = acc[::-1]
plt.plot(idx[:-2]+1, acc_rev)
plt.show()
k = acc_rev.argmax() +2
print("el número óptimo de clusters es %s" %str(k))
#problema de este codo: hay que descartar que k=1
"""MÉTODO DEL CODO variante
-----------------------------------------------------------------------------"""
#puede ser que el salto mas grande no fuera al final, sino que fuera al inicio y esto sería un porblema
c= np.random.multivariate_normal([40,40], [[20,1],[1,30]],size=[200,])
d= np.random.multivariate_normal([80,80], [[30,1],[1,30]],size=[200,])
e= np.random.multivariate_normal([0,100], [[100,1],[1,100]],size=[200,])

X2= np.concatenate((X,c,d,e),)
plt.scatter(X2[:,0],X2[:,1])
plt.show()
Z2 = linkage(X2,"ward")
plt.figure(figsize=(10,10))
dendrogram_tune(
    Z2,
    truncate_mode="lastp",
    p=30,
    leaf_rotation= 90.,
    leaf_font_size=10,
    show_contracted=True,  #para que aparezcan las bolitas
    annotate_above =40, #anotar las uniones por encima de 40 puntos
    max_d=170 #corte max a 170
    )
plt.show()
#localiza la mayor altura de union. Es un método visual.
last = Z2[-10:,2] #(columna 3 de Z)
last_rev =last[::-1] #me revierte el orden de los elementos de la lista
print(last)
print(last_rev)

plt.show()
idx = np.arange(1,len(last)+1) #◙arrange me da valores equiespaciados desde 1 hasta len(last)+1
plt.plot(idx, last_rev)

acc = np.diff(last,2) # se restan los valores last[i+1]-last[i] dos veces
acc_rev = acc[::-1]
plt.plot(idx[:-2]+1, acc_rev)
plt.show()
k = acc_rev.argmax() +2
print("el número óptimo de clusters es %s" %str(k))
print(inconsistent(Z2,5)[-10:])

"""-----------------------------------------------------------------------------
RECUPERAR EL NÚMERO DE CLUSTERS Y SUS ELEMENTOS
-----------------------------------------------------------------------------"""
from scipy.cluster.hierarchy import fcluster
max_d=170 
clusters = fcluster(Z2, max_d, criterion ="distance") #Z= matriz de enlace; m_max=distancia max; criterion= criterio de división
print(clusters) #150 elementos, uno por cada fila del dataset. En este caso hay 3 clusters
#en la matriz clusters me sale 1,2 o 3 según el cluster al que pertenezca cada elemento.

#también puedo cortar por número de clusters en lugar de por distancia max
k=3
clusters2 = fcluster(Z,k,criterion = "maxclust")

#o tambien por la profundidad, como en el método que vimos de la inconsistencia del corte automático
clusters3 = fcluster(Z2,8,depth=10)

#para visualizar las matrices de una forma más visual:
plt.figure(figsize = (10,8))
plt.scatter(X2[:,0],X2[:,1],c=clusters, cmap="prism") #color=clusters