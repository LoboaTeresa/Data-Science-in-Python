# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 22:20:32 2020
@author: 3A

ÍNDICE
------------------------------------------------------------------------------
*Agregación de datos por categoría
*Agrupación, agregación de datos y apricacion de operaciones
*Filtrado de datos
*Transformación de variables
    -Lambda x: 
*Operaciones diversas muy útiles
    - .head()
    - .tail ()
    - .nth()
    - .sort_values()
*Conjunto de entrenamiento y de testing. División por 3 métodos:
    -Usando la distribución normal
    -Usando la libreriía skilearn + train_test_split
    -Usando la librería skilearn + función shuffle

"""
#-----------------------------------------------------------------------------
## AGREGACIÓN DE DATOS POR CATEGORÍA
#-----------------------------------------------------------------------------

import numpy as np #operaciones matematicas para generar numeros aleatorios
import pandas as pd #cargar datos
import matplotlib.pyplot  as plt #librería para plots
import os
gender = ['Male', 'Female']
income = ['Poor', 'Middle Class', 'Rich']
np.random.seed(1997) 

#Voy a generar dos colecciones de género y 3 de income
n=500 #lo uso para dfinir el tamaño de la muestra
gender_data = []
income_data = []

for i in range(0,n):
    gender_data.append(np.random.choice(gender))
    income_data.append(np.random.choice(income))

print(gender_data[1:10])

height =160 + 30 * np.random.randn(n)
weight = 65 + 25 * np.random.randn(n)
age = 30 + 12 * np.random.randn(n)
income = 1800 + 3500 * np.random.randn(n)

data = pd.DataFrame(
    {
     'Gender': gender_data,
     'Economic Status': income_data,
     'Age': age,
     'Height': height,
     'Weight': weight,
     'Income': income
     }
    )

print(data.head())
#-----------------------------------------------------------------------------
## AGREGACIÓN DE DATOS 
#-----------------------------------------------------------------------------
grouped_gender = data.groupby('Gender') #separo por grupos las dos categorias encerradas por la variable gender
print('hola')
print (grouped_gender.groups) #si no hago esto no veré realmente la separación por grupos.

for names, groups in grouped_gender: #tiene pinta que recorre primero la columna 1 (female) 
    print(names)                     #y sus valores (groups) y pasa a la siguiente columna (masculino) y recorre todos sus valores
    print(groups)

grouped_gender.get_group('Female') #pillo solo el grupo female.

#A veces nos interesa hacer una agrupación por varias categorias, ej: sexo y estatus social
double_group = data.groupby(["Gender", "Economic Status"])
print(len(double_group)) #6 grupos porque hay 2 sexos y 3 clases.

for names, groups in double_group: #tiene pinta que recorre primero la columna 1 (female) 
    print(names)                     #y sus valores (groups) y pasa a la siguiente columna (masculino) y recorre todos sus valores
    print(groups)
    
suma_cat = double_group.sum() #me suma las edades, alruras, pesos... de cada categoria
print(suma_cat)
media_cat = double_group.mean()
print(media_cat)
tamaño_cat = double_group.size() #numero de casos en cada categoria
print(tamaño_cat)
analisis_cat = double_group.describe()
print(analisis_cat)

groupe_income = double_group["Income"]
groupe_income.describe()

agrupacion_cat = double_group.aggregate( #me aplica las siguientes funciones a las correspondientes categoria
    {
         "Income": np.sum, #suma de los incomes de cada cat (fem middle class, fem poor, fem rich, male middle class, male poor y male rich)
         "Age": np.mean,
         "Height": np.std
     }
    )
print(agrupacion_cat)

double_group.aggregate(
    {
         "Age": np.mean,
         "Height": lambda h:(np.mean(h))/np.std(h) #la funcion lambda me sirve para definir una operación mas compleja que quiero aplicar a la categoria.
     }
    )

#si quiero aplicar la misma operación a todas las categorías:
double_group.aggregate([np.sum, np.mean, np.std])
#Lo anterior me da la suma, media y desviación tipica de las distintas categorías ((fem middle class, fem poor, fem rich, male middle class, male poor y male rich))

#También podemos aplicar a todas las categorías una operación más compleja usando la función lambda.
double_group.aggregate([lambda x: np.mean(x)/np.std(x)])
#-----------------------------------------------------------------------------
## FILTRADO DE DATOS
#-----------------------------------------------------------------------------
dat_filtrado = double_group["Age"].filter(lambda x: x.sum()>2400) #esto no me devuelve elementos clasificacos, solo elementos
print(dat_filtrado)

#-----------------------------------------------------------------------------
## TRANSFORMACIÓN DE VARIABLES
#-----------------------------------------------------------------------------
zscore = lambda x: (x - x.mean()/x.std()) #defino la transformación que quiero hacerle a los datos
data_transform = double_group.transform(zscore)
plt.hist(data_transform["Age"])

#También puedo cambiar los valosres Na de mi dataset agrupado por la media de los datos
fill_na_mean = lambda x: x.fillna(x.mean()) #función super importante!!!

#-----------------------------------------------------------------------------
## OPERACIONES DIVERSAS MUY ÚTILES
#-----------------------------------------------------------------------------
print(double_group.head(1)) #nos muestra el primer valor de cada grupo
print(double_group.tail(1)) #nos muestra el último valor de cada categoría
print(double_group.nth(32)) #♥nos muestra el elemento 32 de cada una de las filas/categorías
                            #hay que tener cuidado con que exista el elemento nésimo de una fila 
data_sorted = data.sort_values(["Age", "Income"]) #me ordena los valores de menor a mayor, primero por edad y en caso de empate, por income.
age_grouped = data_sorted.groupby("Gender")
print(age_grouped.head(1)) #me da la mujer y el hombre más jóvenes
print(age_grouped.tail(1)) #me da la mujer y el hombre más viejos

#-----------------------------------------------------------------------------
## CONJUNTO DE ENTRENAMIENTO Y CONJUNTO DE TESTING
#-----------------------------------------------------------------------------      
mainpath="C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master/datasets"
filepath = "customer-churn-model/Customer Churn Model.txt"
fullpath = os.path.join(mainpath, filepath)

data = pd.read_csv(fullpath)        


## DIVIDIR UTILIZANDO LA DISTRIBUCIÓN NORMAL
#-----------------------------------------------------------------------------              
a = np.random.randn(len(data))
plt.show()
plt.hist(a)

check = (a<0.8) #array de true o false en funcion de si a es mayor o menor que 0.8
check1 = check.astype(int) #convierto la array booleana al tipo int
plt.show()
plt.hist(check1)

training = data[check] #consideto los training data aquellos que cumplen la condicion del check
testing = data[~check] #considero los datos que no cumolen la condicion de check como los data para el testing

print(len(data))
print(len(training)) #veré que el 80% de los datos estan en el training 
print(len(testing)) #y el 20% en el testing

## DIVIDIR UTILIZANDO LA LIBRERÍA SKLEARN
#-----------------------------------------------------------------------------   
from sklearn.model_selection import train_test_split
#divido mis datos en training y data con la función train_test_split.
train, test = train_test_split(data, test_size = 0.2) #el 80% irá a training y el 20% para test.
#compruebo la división de mis datos:
print("1: training y 2: test")
print(len(train)) #80% de 3333
print(len(test)) #20% de 3333

## USANDO LA FUNCIÓN DE SHUFFLE
#----------------------------------------------------------------------------- 
#Primero mezclamos las filas de mi dataset aleatoriamente

import sklearn
data = sklearn.utils.shuffle(data) 
cut_id = int(0.75*len(data))
train_data = data[:cut_id]
test_data = data[cut_id + 1 :]

print("1: training y 2: test")
print(len(train_data)) #75% de 3333
print(len(test_data)) #25% de 3333

