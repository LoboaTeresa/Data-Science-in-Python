# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 16:34:27 2020

@author: 3A

ÍNDICE: Árboles de decisión para especies de flores
------------------------------------------------------------------------------
*Creación del árbol de decisión
*Visualización del árbol

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mainpath="C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master"
filepath = "datasets/iris/iris.csv"
fullpath = mainpath +"/" + filepath

data = pd.read_csv(fullpath)
print(data.head())
print(data.shape)

plt.hist(data.Species) #histograma con los datos de la columna "species"

#Para ver las categorias de la columna species (que son las categorias objetivo):
print(data.Species.unique())

#Array con los nombres de las columnas del data set:
colnames = data.columns.values.tolist()
predictors = colnames[:4] #variables predictoras son todas hasta la 4 (no incluida)
target = colnames[4] #variable objetivo es la cuarta

#creamos os conjuntos de testing y training
data["is_train"] = np.random.uniform(0, 1, len(data))<=0.75
plt.hist(data.is_train) #miramos a ver si se ha hecho una buena división

train, test = data[data["is_train"]==True], data[data["is_train"]==False] #definimos train y testing

#Creo el árbol con los datos de training
from sklearn.tree import DecisionTreeClassifier
tree= DecisionTreeClassifier(criterion = "entropy", min_samples_split = 20, random_state=99) 
#si no especifico el min_ samples, por defecto es =2 y puede dar lugar a overfitting
# No he especificado min_samples_leaf, que por defecto es =1, pero quizas podria estar bien =5

tree.fit(train[predictors], train[target])

preds = tree.predict(test[predictors])
print(pd.crosstab(test[target], preds, rownames=["Actual"], colnames=["Predictions"])) #para comparar la prediccion y los valores reales
"""
Predictions  setosa  versicolor  virginica
Actual                                    
setosa           15           0          0 --> todas las setosas correctamente clasificadas
versicolor        0          15          1 --> 15 correctamente clasificadas y una mal
virginica         0           2          7 --> 7 correctamente clasificadas y 2 mal
"""
#-----------------------------------------------------------------------------
## VISUALIZACIÓN DEL ÁRBOL DE DECISIÓN
#-----------------------------------------------------------------------------
#Empezamos creando un fichero .dot desde el modelo que ha creado el árbol de clasificación
#Habrá que exportar datos para luego representarlos.
from sklearn.tree import export_graphviz

filepath2 = "notebooks/resources/iris_dtree.dot" #llamaré al fichero iris_dtree. w es porwue ya existe un fichero con ese nombre y quiero sobreescribirlo
fullpath2 = mainpath +"/" + filepath2
with open(fullpath2, "w") as dotfile:
    export_graphviz(tree,out_file=dotfile, feature_names=predictors)
    dotfile.close() #cerramos el fichero
import os
from graphviz import Source

file = open(fullpath2, "r")
text = file.read()

Source(text) #mirarme cómo cojones visualizo estoyj

#-----------------------------------------------------------------------------
## CROSS VALIDATION PARA LA PODA
#-----------------------------------------------------------------------------
X = data [predictors]
Y = data [target]

tree = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_split=20,random_state=99) #creamos el arbol
tree.fit(X,Y) #metemos los datos en el arbol

from sklearn.cross_validation import KFold #método KFold validation, para árboles de decisión
#☺hacemos la valoración cruzada con este método
cv =KFold(n = X.shape[0], n_folds=10, shuffle=True, random_state=1)
# n=número de elementos a ser clasificados = longitud X
# n_fols = número de subgrupos. Hay 150 flores y por lo tanto 10 grupos parece apropiado
# shuffle= True para que haga un muestreo aleatorio
#random_state= como la seed

from sklearn.cross_validation import cross_val_score #índice de validación cruzada
score = np.mean(cross_val_score(tree, X, Y, scoring ="accuracy", cv = cv, n_jobs=1))
#scoring="accuracy" para ver los errores
#n_jobs=1 por si se quiere hacer trabajos simultaneos

print(score)


