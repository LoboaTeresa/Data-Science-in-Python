# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 20:07:37 2020
@author: 3A

ÍNDICE
------------------------------------------------------------------------------
*Concatenar y apendizar data sets
*Concatenar datos de muchos ficheros
*Joins de Datasets usando la funcion merge de pandas.
*Mostrat imágenes y TIPOS DE JOINS
*Trabajando con joins con la funcion merge de pandas: inner, left, right, outer

"""
#-----------------------------------------------------------------------------
## CONCATENAR Y APENDIZAR DATA SETS
#-----------------------------------------------------------------------------
import pandas as pd
import os

mainpath="C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master/datasets"
filepath1 = "wine/winequality-red.csv"
fullpath1 = os.path.join(mainpath, filepath1)

filepath2 = "wine/winequality-white.csv"
fullpath2 = os.path.join(mainpath, filepath1)

red_wine = pd.read_csv(fullpath1, sep = ";")      
print(red_wine.head())  

#investigo las columnas de mis datos:
red_wine_columns = red_wine.columns.values
print(red_wine_columns)

white_wine = pd.read_csv(fullpath2, sep = ";")      
print(white_wine.head())  

#investigo las columnas de mis datos:
white_wine_columns = white_wine.columns.values
print(white_wine_columns)
print(white_wine.shape) #(filas, columnas)

#vamos a concatenar estos dos datasets, uno debajo del otro.
"""
En python tenemos dos tipos de ejer:
    *axis = 0 denota el eje horizontal
    *axis = 1 denota el eje vertical
"""
wine_data = pd.concat([red_wine, white_wine], axis = 0) #concateno el vino blanco debajo del vino tinto
print (wine_data.shape)

#Tambien se pueden concatenar partes concretas de un data set para formar otro nuevo.
#A esta forma de concatenar los datos se le llama  "scrumbelear" los datos
data1 = wine_data.head(10)
data2 = wine_data[300:310]
data3 = wine_data.tail(10)

wine_scramble = pd.concat([data1, data2, data3], axis = 0)

#-----------------------------------------------------------------------------
## DATOS DISTRIBUIDOS
#-----------------------------------------------------------------------------

"""
1)Lo primero es importar el primer fichero
2)Hacemos un bucle para ir recorriendo todos y cada uno de los ficheros. es importante
tener una consistencia en el nombre de los ficheros
3)Dentro del bucle importamos los ficheros uno a uno, cada uno de ellos debe apendizarse
al final del primer fichero que ya habíamos cargado.
4)Repetimos el bicle hasta que no queden ficheros.
"""

filepath3 = "distributed-data/001.csv"
fullpath3 = mainpath +"/" + filepath3 
data3 = pd.read_csv(fullpath3)
print(data3.head())
#Cuando la mayoria de los datos son NaN 
print(data3.shape) #(filas, columnas)

root_filepath = "C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master/datasets/distributed-data/"
final_length = len(data3)
for i in range(2,333): #recordemos que en el range no me entra el último (333)(tenemos 332 archivos que queremos importar)
    if i<10:
        filename = "00" + str(i)
    if 10 <= i < 100: # elif es para si el primer if no se cumple, pasamos a la condicion que pone en elif, por lo que si ponemos elif, 10<i seria innecesario. por eso porngo un if 10<=i<100 o un elif i<100
        filename = "0" + str(i)
    if i >= 100:
        filename = str(i)
    file = root_filepath + filename + ".csv"
    temp_data = pd.read_csv(file)
    final_length +=  len(temp_data)
    data3 = pd.concat([data3,temp_data], axis = 0)
print(data3.tail()) #para comprobar que los ultimos datos corresponden a loas del ID 332
print(data3.head()) 
print(data3.shape)

#también podemos comprobar que se han añadido todas las filas sumando las finals de cada archivo
#y viendo si coinciden con el numero de filas del data set concatenado final.

print(final_length == data3.shape[0]) #si me sale True, es que es verdad

#-----------------------------------------------------------------------------
## JOINS DE DATASETS usando la funcion merge de pandas
#-----------------------------------------------------------------------------
filepath4 = "athletes/Medals.csv"
fullpath4 = mainpath +"/" + filepath4
data_main = pd.read_csv(fullpath4, encoding = "ISO-8859-1" )
#nosotros estamos trabajando en utf-8 y el fichero que estamos importando esta en 
#ISO-8859-1: es una norma de la ISO que define la codificación del alfabeto latino, 
#incluyendo los diacríticos (como letras acentuadas, ñ, ç), y letras especiales
#(como ß, Ø), necesarios para la escritura de varias lenguas originarias 
#de Europa occidental. Además es necesario para usar el símbolo € y otros.
# En nuestro archivo se usan acentos y otros caracteres, pero en este script estamos
#usando el utf-8 y por eso tenemos que hacer este encoding a ISO-8859-1, también 
#conocido como Alfabeto Latino n.º 1.

print(data_main.head())

#Un mismo atleta puede aparecer más de una vez. Cómo podemos saber si este es el caso?
data_main_dp = data_main.drop_duplicates(subset = "Athlete") #me devuelve una lista con los nombres de los atletas sin repetir. Michael phelps solo aparecera esta vez 1 vez
a = data_main["Athlete"].unique().tolist() #me devuelve una lista con los nombres de los atletas sin repetir. Michael phelps solo aparecera esta vez 1 vez
b = data_main.shape[0]

print("atletas únicos: %d  " % (len(a)))
print("shape datos originales %d " %(b))

filepath5 = "athletes/Athelete_Country_Map.csv"
fullpath5 = mainpath + "/" + filepath5
data_country = pd.read_csv(fullpath5, encoding = "ISO-8859-1" )
print(data_country.head())

#es posible que algún atleta haya jugado para mas de un país (porque el país se dividiera
#o cambiara de nombre) ¿cómo corregimos esto?

#Uno de estos atletas es Aleksandar Ciric. Busquémoslo:
print(data_country [data_country["Athlete"] == "Aleksandar Ciric"])

filepath6 = "athletes/Athelete_Sports_Map.csv"
fullpath6 = mainpath + "/" + filepath6
data_sports = pd.read_csv(fullpath6, encoding = "ISO-8859-1" )
print(data_sports.head())

#Habrá algún atleta que haya competido en más de un deporte?
a = data_main["Athlete"].unique().tolist() #me devuelve una lista con los nombres de los atletas sin repetir. Michael phelps solo aparecera esta vez 1 vez
b = data_sports.shape[0]

print("filas data_main: %d  " % (len(a)))
print("filas data_sports %d " %(b))
print(b > len(a)) #si sale true, hay atletas que han competido por más de un deporte

#alguno de estos atletas me los conozco y los voy a mostrar:
print(data_sports [(data_sports["Athlete"] == "Chen Jing") |
                   (data_sports["Athlete"] == "Richard Thompson") |
                   (data_sports["Athlete"] == "Matt Ryan")                
                   ]) 

#Ahora empieza los joins

#vamos a usar la columna athlete de main con la country de country.
#hay ue especificar qué columna queremos que quede a la izquierda y cual a la derecha
#además hay que especificar el nombre de la columna (tanto para la derecha como la de 
#la izquierda) donde quiero hacer el matching. En este caso quiero unificar la columna 
#Athlete de main y la columna athlete de country. En este caso se llaman igual. 
data_main_country = pd.merge (left = data_main, right = data_country,
                              left_on = "Athlete", right_on = "Athlete")
print(data_main_country.head()) #he combinado las dos tablas
print("filas data_main %d " %(len(data_main)))
print("filas data_country %d " %(len(data_country)))
print("filas data_main_country %d " %(len(data_main_country)))

#al hacer la join aumenta el numero de filas porque hay atletas que han jugado en 
#varios paises o en varios años y salen repes, veamos el ejemplo de Aleksandar Ciric
print(data_main_country [data_main_country["Athlete"] == "Aleksandar Ciric"])
#vemos que la info se ha duplicado al incluir el pais.

#para evitar esto vamos a quitar los duplicados de paises.
data_country_dp = data_country.drop_duplicates(subset = "Athlete") #solo una fila por nombre de atleta. Así me elimino el segundo pais.
#compruebo que main sin repes tiene el mismo numero de filas que country sin duplicados:
print(len(data_country_dp)==len(a))
#Y ahora uno main y country sin duplicados:
data_main_country_dp = pd.merge (left = data_main, right = data_country_dp,
                              left_on = "Athlete", right_on = "Athlete")
print(data_main_country.head()) #he combinado las dos tablas
print("filas data_main %d " %(len(data_main)))
print("filas data_country %d " %(len(data_country_dp)))
print("filas data_main_country %d " %(len(data_main_country_dp)))

#podríamos añadirle tambien una columna con los deportes de cada atleta.
data_sports_dp = data_sports.drop_duplicates(subset = "Athlete") #solo una fila por nombre de atleta. Así me elimino el segundo pais.
#compruebo que main sin repes tiene el mismo numero de filas que country sin duplicados:
print(len(data_sports_dp)==len(a))
#Y ahora uno main y country sin duplicados:
data_final = pd.merge (left = data_main_country, right = data_sports_dp,
                              left_on = "Athlete", right_on = "Athlete")
print(data_final.head())

#-----------------------------------------------------------------------------
## INCORPORAR GRÁFICOS e IMÁGENES A UN NOTEBOOK DE PYTHON
#-----------------------------------------------------------------------------
from IPython.display import Image, display #☺lo d IPython.display es una librería para mostrar cosas

"""
TIPOS DE JOINS:
    *Inner Join
        -Devuelve un data frame con las filas que tienen valor tanto en el primer como
        en el segundo data frame que estamos uniendo.
        -El número de finas será igual al número de filas comunes que tengan ambos data
        sets
            ·Data Set A tiene 60 filas
            ·Data Set B tiene 50 filas
            ·Ambos comparten 30 filas
            ·Entnces A inner Join B tendrá 30 filas
        -En términos de teoría de conjuntos, se trata de la intersección de dos conjuntos
"""
display(Image(filename="C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master/notebooks/resources/inner-join.png"))

"""
TIPOS DE JOINS:
    *Left Joins
        -Devuelve un data frame con las filas que tienen valor en el data set de la 
        izquierda, sin importar si tienen correspondencia en el de la derecha o no.
        -Las filas del data frame final que no correspondan a ninguna fila del data 
        frame derecho, tendrán NAs en las columnas del data frame derecho.
        -El número de finas será igual al número de filas del data frame izquierdo
            ·Data Set A tiene 60 filas
            ·Data Set B tiene 50 filas
            ·Entonces A left Join B tendrá 60 filas
        -En términos de teoría de conjuntos, se trata del propio data set de la 
        izquierda quien, además tiene la intersección en su interior.
"""
display(Image(filename="C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master/notebooks/resources/left-join.png"))

"""
TIPOS DE JOINS:
    *Right Joins
        -Devuelve un data frame con las filas que tienen valor en el data set de la 
        derecha, sin importar si tienen correspondencia en el de la izquierda o no.
        -Las filas del data frame final que no correspondan a ninguna fila del data 
        frame izquierdo, tendrán NAs en las columnas del data frame izquierdo.
        -El número de finas será igual al número de filas del data frame derecho
            ·Data Set A tiene 60 filas
            ·Data Set B tiene 50 filas
            ·Entonces A Right Join B tendrá 50 filas
        -En términos de teoría de conjuntos, se trata del propio data set de la 
        derecha quien, además tiene la intersección en su interior.
"""
display(Image(filename="C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master/notebooks/resources/right-join.png"))
"""
TIPOS DE JOINS:
    *Outer Joins
        -Devuelve un data frame con todas las filas de ambos
        -Las filas del data frame final que no correspondan a ninguna fila del data 
        frame izquierdo (o derecho), tendrán NAs en las columnas del data frame izquierdo.
        -El número de finas será igual al máximo número de filas de ambos data frames
            ·Data Set A tiene 60 filas
            ·Data Set B tiene 50 filas
            ·Ambos comparten 30 filas
            ·Entonces A Outer Join B tendrá 60 + 50 - 30 = 80 filas
        -En términos de teoría de conjuntos, se trata dela unión de conjuntos.
"""
display(Image(filename="C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master/notebooks/resources/outer-join.png"))
#-----------------------------------------------------------------------------
## TRABAJANDO CON JOINS
#-----------------------------------------------------------------------------
#Para trabajar con joins voy a eliminar algun dato de los paises de los atletas
from IPython.display import Image, display
import numpy as np

#Hago una seleccion de 6 atletas de forma aleatoria pero sin coger el mismo mas 
#de una vez (.choice( , , replace = False))
out_athletes = np.random.choice(data_main["Athlete"],size = 6, replace = False)

#creo un array con los atletas que no estan en out_atletes y no son michael phelps

data_country_dlt = data_country_dp[(~data_country_dp["Athlete"].isin( out_athletes)) &
                                   (data_country_dp["Athlete"] != "Michael Phelps")]
print(data_country_dlt.head())
print("data country:")
print(len(data_country_dlt))
print(len(data_country_dp))

#hacemos lo mismo con los datos de los deportes y main data
data_sports_dlt = data_sports_dp[(~data_sports_dp["Athlete"].isin( out_athletes)) &
                                   (data_sports_dp["Athlete"] != "Michael Phelps")]
print("data sports:")
print(len(data_sports_dlt))
print(len(data_sports_dp))

data_main_dlt = data_main_dp[(~data_main_dp["Athlete"].isin( out_athletes)) &
                                   (data_main_dp["Athlete"] != "Michael Phelps")]
print("data main:")
print(len(data_main_dlt))
print(len(data_main_dp))


#procedemos ahora a hacer algun ejercicio con joins
#data_main: contiene todos los datos.
#data_country_dlt: le falta la info de 7 atletas

#INNER JOIN CON LA FUNCION MERGE:
merged_inner = pd.merge(left = data_main_dp, right = data_country_dlt,
                            how = "inner", left_on = "Athlete", right_on = "Athlete")
print("INNER- len data_ main_dp: %d , len data_ country_dlt: %d  y len merged_inner: %d"  %(len (data_main_dp) , len(data_country_dlt), len(merged_inner)))

merged_left = pd.merge(left = data_main_dp, right = data_country_dlt,
                            how = "left", left_on = "Athlete", right_on = "Athlete")
print("LEFT - len data_ main: %d , len data_ country_dlt: %d  y len merged_left: %d"  %(len (data_main_dp) , len(data_country_dlt), len(merged_left)))

merged_right = pd.merge(left = data_main_dlt, right = data_country_dp,
                            how = "right", left_on = "Athlete", right_on = "Athlete")
print("RIGHT - len data_ main_dlt: %d , len data_ country_dlt: %d  y len merged_right: %d"  %(len (data_main_dlt) , len(data_country_dp), len(merged_right)))

#Para usar el outer join voy a al data_country_dlt un nuevo atleta, a mi misma que juego por españa
data_country_dlt_teresa = data_country_dlt.append(
    {
     "Athlete": "Teresa Lobo Alonso",
     "Country": "Spain"
     }, ignore_index = True #lo añado a la última fila
)
print(data_country_dlt_teresa.tail())

merged_outer = pd.merge(left = data_main, right = data_country_dlt_teresa,
                            how = "outer", left_on = "Athlete", right_on = "Athlete")
print("OUTER - len data_ main: %d , len data_ country_dlt_teresa: %d  y len merged_outert: %d"  %(len (data_main) , len(data_country_dlt_teresa), len(merged_outer)))

