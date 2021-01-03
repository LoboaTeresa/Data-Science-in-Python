# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 17:07:22 2020

@author: 3A

ÍNDICE: CLUSTERING COMPLETO
------------------------------------------------------------------------------

*Clustering con python
"""
"""-----------------------------------------------------------------------------
CLUSTERING CON PYTHON
-----------------------------------------------------------------------------"""
import pandas as pd
import numpy as np


##1##Importamos y estudiamos el dataset
##-----------------------------------------------------------------------------"""
mainpath="C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master/datasets"
filepath = "wine/winequality-red.csv"
fullpath = mainpath +"/" + filepath

df = pd.read_csv(fullpath, sep=";")

print(df.head())
print(df.shape)

import matplotlib.pyplot as plt
plt.hist(df["quality"])
plt.show()
df.groupby("quality").mean() #agrupamos los vinos según su calidad (3,4,5,6,7) y el resto de cualidades ponemos la media de los vinos con esa calidad


##2##Normalización de os datos (para que todas las columnas/cualidades tengan el mismo peso en la agrupación)
##-----------------------------------------------------------------------------"""
df_norm = (df-df.min())/(df.max()-df.min())
print(df_norm.head())

##3A## Clústering jerárquico o aglomerativo con scikit-learn
##-----------------------------------------------------------------------------"""
from sklearn.cluster import AgglomerativeClustering
clus= AgglomerativeClustering(n_clusters=6, linkage="ward").fit(df_norm) #creamos los clusters
# número de clusters=6, método de enlace=ward
#obtenemos las etiquetas:
md_h=pd.Series(clus.labels_) #md es un elemento con las etiquetas de los cluters
plt.hist(md_h)
plt.title("Histograma de los clusters")
plt.xlabel("Cluster")
plt.ylabel("Número de vinos en el cluster")
plt.show()

#podemos buscar los hijos para cada uno de los nodos hoja:
print(clus.children_)
from scipy.cluster.hierarchy import dendrogram, linkage
Z=linkage(df_norm, "ward")
plt.figure(figsize=(25,10))
plt.title("Drendrograma de los vinos")
plt.xlabel("ID del vino")
plt.ylabel("Distancia")
dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
plt.show()

##3A## Clústering k-mean
##-----------------------------------------------------------------------------"""
from sklearn.cluster import KMeans
from sklearn import datasets

model =KMeans(n_clusters=6)
model.fit(df_norm)
print(model.labels_) #◘nos dice a qué cluster corresponde cada uno de los vinos
md_k = pd.Series(model.labels_)

df_norm["clust_h"]= md_h #añadimos al df el identificador del cluster según el método jerárquico
df_norm["clust_k"]= md_k #añadimos al df el identificador del cluster según el método keans

plt.hist(md_k)
model.cluster_centers_ #nos da las múltiples coordenadas de los baricentros de cada cluster
model.inertia_ #suma de los cuadrados internos (suma de las distancias al cuadrado de cada punto al baricentro de su cluster)

##4## Interpretación final
##-----------------------------------------------------------------------------"""
print(df_norm.groupby("clust_k").mean())

