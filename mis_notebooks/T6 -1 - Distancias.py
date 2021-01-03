# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 17:00:16 2020

@author: 3A

ÍNDICE: DISTANCIAS
------------------------------------------------------------------------------
*Distancias
*Enlaces
*Clustering Jerárquico

"""
import pandas as pd
import numpy as np

from scipy.spatial import distance_matrix #el scipy es una librería de python para la ciencia de datos
mainpath="C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master/datasets"
filepath = "movies/movies.csv"
fullpath = mainpath +"/" + filepath

data = pd.read_csv(fullpath, sep=";")

print(data.head())
print(data.shape)
print(data.columns.values)
"""-----------------------------------------------------------------------------
 DISTANCIAS
-----------------------------------------------------------------------------"""
#mirando el dataset veo que tengo 4 columnas (user + 3 pelis), pero realmente solo me interesan
#las pelis, así que en una lista con los nombres de las columnas, expluyo la primera columna
#"user_id"
movies = data.columns.values.tolist()[1:]#☺selecciono de la coluna 1 en adelante. La 0 no la cojo
print(movies)
dd1 = distance_matrix(data[movies], data[movies],p=1) #matriz de distancias minkowski d=1 -> manhattan
dd2 = distance_matrix(data[movies], data[movies],p=2) #matriz de distancias minkowski d=2 -> eucídea
dd10 = distance_matrix(data[movies], data[movies],p=10) #matriz de distancias minkowski d=10
#dd1, dd2 y dd10 son arrays. Para trabajar con ello es mejor apsarlo a data frame
def dm_to_df (dd,col_name): #dd=matriz de distancias; col_name= nombres de las columnas
    return pd.DataFrame(dd,index=col_name,columns=col_name) #index: nombres de las filas; columns: nombres de las columnas
df_dd1=dm_to_df(dd1,data["user_id"])
df_dd2=dm_to_df(dd2,data["user_id"])
df_dd10=dm_to_df(dd10,data["user_id"])
print(df_dd1.head())
print(df_dd2.head())
print(df_dd10.head())
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #para representaciones en 3d
fig= plt.figure() #creamos una figura vacía
ax = fig.add_subplot(111,projection="3d") #creamos los ejes, definimos las 3 dimensiones. lo de 111 es que los 3 ejes van de 0 a 1 
ax.scatter(xs=data["star_wars"],ys=data["lord_of_the_rings"], zs=data["harry_potter"]) #metemos los datos
#lo de scatter es de scatter plot"""

"""-----------------------------------------------------------------------------
ENLACES
-----------------------------------------------------------------------------"""
df=dm_to_df(dd1,data["user_id"]) #data frame de nuestra matriz de distancias
print("hola")
print(df)
#Lo que pretendemos en este ejercicio es la aglutinación manual jerárquica de nuestros
#datos en clusters 
Z = [] #CREO UNA MATRIZ VACÍA
#Busco la distancia mínima (fila1, col 10) y la añado a una columna nueva, la 11
df[11] = df[1]+[10] #creo una nueva COLUMNA cuyos valores sean el resultado de la suma de las columnas 1 y 10, donde se encuentra la distancia mínima entre distancias
df.loc[11]=df.loc[1]+df.loc[10] #creo una FILA cuyos valores sean el resultado de la suma de las columnas 1 y 10, donde se encuentra la distancia mínima entre distancias

#guardo la info del primer cluster en Z
Z.append([1,10,0.7,2]) #id1, id2, distancia entre distancias, n_elementos en el cluster-->11


##ENLACE SIMPLE: la distancia o similitud entre dos clusters viene dada, respectivamente, por
#la mínima distancia (o máxima similitud) entre sus componentes.

#Entonces, sustituyo los valores de la fila y columna 11 por el valor de la distancia min
for i in df.columns.values.tolist(): 
    df.loc[11][i] = min(df.loc[1][i], df.loc[10][i]) #reemplazo los valores de la fila 11 columna i por el min entre df.loc[1][i] y df.loc[10][i]
    df.loc[i][11] = min(df.loc[i][1], df.loc[i][10]) # lo mismo pero con la columna 11
print (df) #miro esto para comprobar que la sustitucion or los mins ha sido correcta

#elimino las filas y columas cuyos valores ya he aglutinado en la fila y columna 11:
df = df.drop([1,10]) #elimino las columnas 1 y 10
df = df.drop([10,1], axis=1) #elimino las filas 1 y 10

#repetimos el proceso de enlace
#   1)busco la distancia más pequeña: (fila, columna)=(2,7) que es 0.8
print(df)
x=2
y=7
n=12
df[n] = df[x]+[y] #creo una nueva COLUMNA cuyos valores sean el resultado de la suma de las columnas 1 y 10, donde se encuentra la distancia mínima entre distancias
df.loc[n]=df.loc[x]+df.loc[y] #creo una FILA cuyos valores sean el resultado de la suma de las columnas 1 y 10, donde se encuentra la distancia mínima entre distancias
Z.append([x,y,df.loc[x][y],2])
for i in df.columns.values.tolist(): 
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y]) 

#elimino las filas y columas cuyos valores ya he aglutinado en la fila y columna n:
df = df.drop([x,y]) 
df = df.drop([y,x], axis=1) 
print (df)
print (Z) # [[1, 10, 0.7, 2], [2, 7, 0.7999999999999994, 2]] -->tenemos 2 clusters con 2 elementos cada uno

#repetimos el proceso de enlace
#   1)busco la distancia más pequeña: (fila, columna)=(5,8) que es 3.2
x=5
y=8
n=13
df[n] = df[x]+[y] #creo una nueva COLUMNA cuyos valores sean el resultado de la suma de las columnas 1 y 10, donde se encuentra la distancia mínima entre distancias
df.loc[n]=df.loc[x]+df.loc[y] #creo una FILA cuyos valores sean el resultado de la suma de las columnas 1 y 10, donde se encuentra la distancia mínima entre distancias
Z.append([x,y,df.loc[x][y],2])
for i in df.columns.values.tolist(): 
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y]) 

#elimino las filas y columas cuyos valores ya he aglutinado en la fila y columna n:
df = df.drop([x,y]) 
df = df.drop([y,x], axis=1) 
print (df)
print (Z) # [[1, 10, 0.7, 2], [2, 7, 0.7999999999999994, 2], [5, 8, 3.2, 2]]-->tenemos 3 clusters con 2 elementos cada uno

#repetimos el proceso de enlace
#   1)busco la distancia más pequeña: (fila, columna)=(13,11) que es 3.9
x=11
y=13
n=14
df[n] = df[x]+[y] #creo una nueva COLUMNA cuyos valores sean el resultado de la suma de las columnas 1 y 10, donde se encuentra la distancia mínima entre distancias
df.loc[n]=df.loc[x]+df.loc[y] #creo una FILA cuyos valores sean el resultado de la suma de las columnas 1 y 10, donde se encuentra la distancia mínima entre distancias
Z.append([x,y,df.loc[x][y],2])
for i in df.columns.values.tolist(): 
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y]) 

#elimino las filas y columas cuyos valores ya he aglutinado en la fila y columna n:
df = df.drop([x,y]) 
df = df.drop([y,x], axis=1) 
print (df)
print (Z) #[[1, 10, 0.7, 2], [2, 7, 0.7999999999999994, 2], [5, 8, 3.2, 2], [11, 13, 3.900000000000001, 2]]-->tenemos 4 clusters con 2 elementos cada uno

#repetimos el proceso de enlace
#   1)busco la distancia más pequeña: que es 4.9. Necesitaré unir varios elementos: (fila, columna)=(14,9) y (14,12)
x=14
y=9
z=12

n=15

df[n] = df[x]+[y] #creo una nueva COLUMNA cuyos valores sean el resultado de la suma de las columnas 1 y 10, donde se encuentra la distancia mínima entre distancias
df.loc[n]=df.loc[x]+df.loc[y] #creo una FILA cuyos valores sean el resultado de la suma de las columnas 1 y 10, donde se encuentra la distancia mínima entre distancias
Z.append([x,y,df.loc[x][y],3])
for i in df.columns.values.tolist(): 
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i], df.loc[z][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y], df.loc[z][i]) 

#elimino las filas y columas cuyos valores ya he aglutinado en la fila y columna n:
df = df.drop([x,y,z]) 
df = df.drop([y,x,z], axis=1) 
print (df)
print (Z) #-->tenemos 4 clusters con 2 elementos cada uno y uno con 3

#repetimos el proceso de enlace
#   1)busco la distancia más pequeña: que es 5.5. Necesitaré unir varios elementos: (fila, columna)=(15,4) y (15,6)
x=15
y=4
z=6

n=16

df[n] = df[x]+[y] #creo una nueva COLUMNA cuyos valores sean el resultado de la suma de las columnas 1 y 10, donde se encuentra la distancia mínima entre distancias
df.loc[n]=df.loc[x]+df.loc[y] #creo una FILA cuyos valores sean el resultado de la suma de las columnas 1 y 10, donde se encuentra la distancia mínima entre distancias
Z.append([x,y,df.loc[x][y],3])
for i in df.columns.values.tolist(): 
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i], df.loc[z][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y], df.loc[z][i]) 

#elimino las filas y columas cuyos valores ya he aglutinado en la fila y columna n:
df = df.drop([x,y,z]) 
df = df.drop([y,x,z], axis=1) 
print (df)
print (Z) #-->tenemos 4 clusters con 2 elementos cada uno y dos con 3

#repetimos el proceso de enlace
#   1)busco la distancia más pequeña: (fila, columna)=(13,11) que es 3.9
x=3
y=16
n=17
df[n] = df[x]+[y] #creo una nueva COLUMNA cuyos valores sean el resultado de la suma de las columnas 1 y 10, donde se encuentra la distancia mínima entre distancias
df.loc[n]=df.loc[x]+df.loc[y] #creo una FILA cuyos valores sean el resultado de la suma de las columnas 1 y 10, donde se encuentra la distancia mínima entre distancias
Z.append([x,y,df.loc[x][y],2])
for i in df.columns.values.tolist(): 
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y]) 

#elimino las filas y columas cuyos valores ya he aglutinado en la fila y columna n:
df = df.drop([x,y]) 
df = df.drop([y,x], axis=1) 
print (df)
print (Z) #[[1, 10, 0.7, 2], [2, 7, 0.7999999999999994, 2], [5, 8, 3.2, 2], [11, 13, 3.900000000000001, 2]]-->tenemos 4 clusters con 2 elementos cada uno

"""-----------------------------------------------------------------------------
CLUSTERING JERÁRQUICO
-----------------------------------------------------------------------------"""
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage # para hacer los enlaces y demás
print(movies) #cabeceras de las pelis que queríamos usar.
data[movies] #nos da las valoracines de los usuarios 0-1 de las pelis

Z = linkage(data[movies], "ward") #me crea los clusters según el enlace ward, el cual minimiza los cuadrados, las desviaciones
print(Z) 

"""
[[ usuario 0.           user 9.           distancia: 0.41231056     nº elementos: 2.  ] #10
 [ user: 1.             user: 6.          distancia: 0.6164414      nº elementos: 2.  ] #11
 [ user: 4.             user: 7.          distancia: 2.16794834     nº elementos: 2.  ] #12
 [ user: 3.             user: 8.          distancia: 3.48281495     nº elementos:2.   ] #13
 [ user: 5.             elemento: 10.     distancia: 5.2943366      nº elementos:3.   ] #14
 [elemento: 13.         elemento: 14.     distancia: 6.59317829     nº elementos:5.   ] #15
 [elemento: 11.         elemento: 12.     distancia: 6.66408283     nº elementos:4.   ] #16
 [ user: 2.             elemento: 15.     distancia: 10.62355873    nº elementos:6.   ] #17
 [elemento: 16.         elemento: 17.     distancia: 12.8156935     nº elementos:10.  ]]#18 """

plt.figure(figsize=(25,10)) #creamos una figura con un tamaño de 25cm anchura*10cm altura
plt.title("Dendrograma jerárgico para el clustering")
plt.xlabel("ID de los usuarios de Netflix")
plt.ylabel("Distancia")
dendrogram(Z,leaf_rotation=90., leaf_font_size=10) #lo de leaf son las etiquetas de los usuarios
plt.show()

plt.figure(figsize=(25,10)) #creamos una figura con un tamaño de 25cm anchura*10cm altura
plt.title("Dendrograma jerárgico para el clustering")
plt.xlabel("ID de los usuarios de Netflix")
plt.ylabel("Distancia")
dendrogram(Z,leaf_rotation=90., leaf_font_size=10, orientation="right") #lo mismo pero girado

#puedo hacer el clustering con otro tipo de enlace, con el average por ejemplo
Z = linkage(data[movies], "average") #me crea los clusters según el enlace ward, el cual minimiza los cuadrados, las desviaciones
print(Z) 

plt.figure(figsize=(25,10)) #creamos una figura con un tamaño de 25cm anchura*10cm altura
plt.title("Dendrograma jerárgico para el clustering")
plt.xlabel("ID de los usuarios de Netflix")
plt.ylabel("Distancia")
dendrogram(Z,leaf_rotation=90., leaf_font_size=10) #lo de leaf son las etiquetas de los usuarios
plt.show()

plt.figure(figsize=(25,10)) #creamos una figura con un tamaño de 25cm anchura*10cm altura
plt.title("Dendrograma jerárgico para el clustering")
plt.xlabel("ID de los usuarios de Netflix")
plt.ylabel("Distancia")

#puedo hacer el clustering con otro tipo de enlace, con el simple por ejemplo
Z = linkage(data[movies], "single") #me crea los clusters según el enlace ward, el cual minimiza los cuadrados, las desviaciones
print(Z) 

plt.figure(figsize=(25,10)) #creamos una figura con un tamaño de 25cm anchura*10cm altura
plt.title("Dendrograma jerárgico para el clustering")
plt.xlabel("ID de los usuarios de Netflix")
plt.ylabel("Distancia")
dendrogram(Z,leaf_rotation=90., leaf_font_size=10) #lo de leaf son las etiquetas de los usuarios
plt.show()

plt.figure(figsize=(25,10)) #creamos una figura con un tamaño de 25cm anchura*10cm altura
plt.title("Dendrograma jerárgico para el clustering")
plt.xlabel("ID de los usuarios de Netflix")
plt.ylabel("Distancia")

"""puedo tambien elegir la metrica usada para calcular las distancias: (por defecto es la euclídea)
    La función distancia puede ser:
    'braycurtis', 'canberra','chebyshev', 'cityblock', 'correlation','cosine', 'dice', 
    'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski'
    'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 
    'sqeuclidean', 'yule'
    """
Z = linkage(data[movies], "single", metric="cityblock") 
print(Z) 

plt.figure(figsize=(25,10)) #creamos una figura con un tamaño de 25cm anchura*10cm altura
plt.title("Dendrograma jerárgico para el clustering")
plt.xlabel("ID de los usuarios de Netflix")
plt.ylabel("Distancia")
dendrogram(Z,leaf_rotation=90., leaf_font_size=10) #lo de leaf son las etiquetas de los usuarios
plt.show()

plt.figure(figsize=(25,10)) #creamos una figura con un tamaño de 25cm anchura*10cm altura
plt.title("Dendrograma jerárgico para el clustering")
plt.xlabel("ID de los usuarios de Netflix")
plt.ylabel("Distancia")
    