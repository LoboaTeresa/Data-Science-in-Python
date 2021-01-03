# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:38:17 2020

@author: 3A

ÍNDICE: Implementación del método de la máxima verosimilitud para la
        regresión logístico
------------------------------------------------------------------------------
*Regresión logística para predicciones bancarias
*Conversión de las variables categóricas a dummies
*Implementación del modelo en Python con statsmodel.api
*Implementación del modelo en Python con sckitlearn
*Validación del modelo
*Validación cruzada k fold con sklearn.model_selection
*Matrices de confusión y curvas ROC

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mainpath="C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master/datasets"
filepath = "bank/bank.csv"
fullpath = mainpath +"/" + filepath

data = pd.read_csv(fullpath, sep=";")

print(data.head())
print(data.shape)
print(data.columns.values)

#-----------------------------------------------------------------------------
## REGRESIÓN LOGÍSTICA PARA PREDICCIONES BANCARIAS
#-----------------------------------------------------------------------------

data["y"] = (data["y"]=="yes").astype(int) #transformamos los yes y no a 1 y o respectivamente
print(data["education"].unique()) #los valores distintos que toma esa variable
#
#Vamos a reorganizar las categorias de educación: data cleaning
data["education"] = np.where(data["education"]=="basic.4y", "Basic", data["education"])
data["education"] = np.where(data["education"]=="basic.6y", "Basic", data["education"])
data["education"] = np.where(data["education"]=="basic.9y", "Basic", data["education"])

data["education"] = np.where(data["education"]=="high.school", "High School", data["education"])
data["education"] = np.where(data["education"]=="professional.course", "Professional Course", data["education"])
data["education"] = np.where(data["education"]=="university.degree", "University Degree", data["education"])

data["education"] = np.where(data["education"]=="illiterate", "Illiterate", data["education"])
data["education"] = np.where(data["education"]=="unknow", "Unknow", data["education"])

print(data["education"].unique()) #los valores distintos que toma esa variable

print(data["y"].value_counts()) #hay 3668 0s (Noes) Y 451 1s (Síes)
print(data.groupby("y").mean()) #Me da la media de otras categorías (Edad, duracion, campaing...) de los noes y los sies
print(data.groupby("education").mean())

#%matplotlib inline
pd.crosstab(data.education, data.y).plot(kind="bar")
plt.title("Frecuencia de compra en función del nivel de educación")
plt.xlabel("Nivel de educación")
plt.ylabel("Frecuencia de compra")

table = pd.crosstab(data.marital, data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind="bar", stacked = True)
#lo de .div(table.sum(1).astype(float), axis=0) sirve para que las barras tengan todas
#la misma altura y comparemos únicamente las proporciones dentro de una misma cat
#y no entre categorias
plt.title("Diagrama apilado de estado civil contra el nivel de compras")
plt.xlabel("Estado civil")
plt.ylabel("Proporción de clientes")

#%matplotlib inline
pd.crosstab(data.day_of_week, data.y).plot(kind="bar")
plt.title("Frecuencia de compra en función del día de la semana")
plt.xlabel("Día de la semana")
plt.ylabel("Frecuencia de compra")

#si lo anterior lo quisieramos apilado
table = pd.crosstab(data.day_of_week, data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind="bar", stacked = True)
plt.title("Diagrama apilado de estado civil contra el nivel de compras")
plt.xlabel("Día de la semana")
plt.ylabel("Frecuencia de compra")

#%matplotlib inline
table = pd.crosstab(data.month, data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind="bar", stacked = True)
plt.title("Frecuencia de compra en función del mes")
plt.xlabel("Mes del año")
plt.ylabel("Frecuencia de compra")

#%matplotlib inline
data.age.hist()
plt.title("Histograma de la edad")
plt.xlabel("Edad")
plt.ylabel("Clientes")

pd.crosstab(data.age, data.y).plot(kind="bar")
pd.crosstab(data.poutcome, data.y).plot(kind="bar")

#-----------------------------------------------------------------------------
## CONVERSIÓN DE LAS VARIABLES CATEGÓRICAS A DUMMIES
#-----------------------------------------------------------------------------
categories = ["job", "marital", "education", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]

for category in categories:
    cat_list = "cat" + "_" + category
    cat_dummies = pd.get_dummies(data[category], prefix=category)
    data_new = data.join(cat_dummies)
    data = data_new
    
data_vars = data.columns.values.tolist() #nombres de las columnas de los nuevos datos
to_keep = [v for v in data_vars if v not in categories] #me quedo con las categorías que no aparecen en el array "categories"
to_keep = [v for v in to_keep if v not in ["default"]]
bank_data = data[to_keep] #df con solo los datos de las cat que he seleccionado en to_keep
print(bank_data.columns.values)

bank_data_vars = bank_data.columns.values.tolist() #lista con los nombres de las cat del df de interés
#Variable a predecir:
Y =['y'] #variable a predecir. (Existe una columna en bank_data que se llama y, la cualq uiero predecir)
#Variables predictoras
X = [v for v in bank_data_vars if v not in Y] #separo todas las cat que no sean y.

#-----------------------------------------------------------------------------
## SELECCIÓN DE RASGOS PARA EL MODELO
#-----------------------------------------------------------------------------
#Para decidir cuáles de todas las variables son las mas significativas para predecir
#el valor de salida del modelo

n = 12 #ej de número de variables que quiero usar.

from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
rfe = RFE(lr, n) #con RFE elegimos las n variables mas significativas para el modelo logistico lr
rfe = rfe.fit(bank_data[X], bank_data[Y].values.ravel()) #cargo los datos: X=variables predictoras; Y=variable a predecir
                                                        #el .ravel() me convierte y (que está en fila) en una columna
#me sale que se ha superado el límite de iteraciones: al tratarse de un método 
#iterativo aproximado, el log solo te hace saber que no ha convergido porque ha
#superado el número de iteraciones por defecto. O se las subes para ver si 
#converge o simplemente toma nota de que el resultado es aproximado.

print(rfe.support_) #True: entra en el modelo; False: no
print(rfe.ranking_) #Posición en el ranking. 1= la que + colabora
z= zip(bank_data_vars, rfe.support_, rfe.ranking_)
print(list(z))

#creo una lista con las variables definitivas que contribuyen al modelo (True)
#y están en una de las prumeras posiciones del ranking
cols = ["previous", "euribor3m", "job_management", "job_student","job_technician", "month_aug", "month_dec", "month_jul", "day_of_week_wed", "poutcome_nonexistent" ]

X = bank_data[cols]
Y= bank_data["y"]

#-----------------------------------------------------------------------------
## IMPLEMENTACIÓN DEL MODELO EN PYTHON CON STATSMODEL.API
#-----------------------------------------------------------------------------
#Crearemos nuestro modelo con las variables que hemos seleccionadas (en X)
#import statsmodels.api as sm 
#logit_model = sm.Logit(Y,X) #creaos el modelo logístico con Y como variable a predecir y X, las variables predictoras
#result = logit_model.fit() 
#print(result.summary2())
"""
Model: Logit(modelo logístico)
Dependent Variable : y (la variable a predecir)
No. Observations (No. de datos que tenemos)
Df Model: 9 (número de grados de librertad del modelo, 1 menos que el múmero de variables predictoras(10))
AIC y BIC se usan para comparas las eficiencias de distintos modelos.
P>|z|: P valor, indica el nivel de significación de la variable. cuanto + peque mejor (0.3 ya es muy grande)

Nuestro modelo es una puta mierda y es porque la hemso cagado a la hora de elegir las variables y no sé donde.
"""
#-----------------------------------------------------------------------------
## IMPLEMENTACIÓN DEL MODELO EN PYTHON CON SCIKIT-LEARN
#-----------------------------------------------------------------------------
from sklearn import linear_model #el linear model tiene el método de reg logistica
logit_model = linear_model.LogisticRegression()
logit_model.fit(X,Y) #montamos nustro modelo con la libreria scikit-learn en lugar de statsmodels
print(logit_model.score(X,Y)) # R^2 = 0.8953629521728574, nos ajustamos bien (no entiendo or qué sale esto)

#Creo un dataframe con las columnas de X y los coefs del modelo.
#   *No se puede trabajar con Zip, por eso siempre hay que convertirlo en lista
#   *He de trasponer los coefs porque me los da en fila y los quiero en columna
DF=pd.DataFrame(list(zip(X.columns, np.transpose(logit_model.coef_))) ) 
print(DF)

#los coefs indican los cambios en escala logarítmica (cociente de prob) por cada unidad de cambio de la variable.
#ej: el coef de la variable previous es 0.6, esto significa que si la variable previous
#incrementa en una unidad, el logaritmo del cociente de probabilidades se incrementará 
#en 0.6 y por lo tanto la prob de compra incrementará de forma acore.
#coef = log(pi/1-pi). Las que más influyen tendrán un mayor coef.
#-----------------------------------------------------------------------------
## VALIDACIÓN DEL MODELO LOGÍSTICO
#-----------------------------------------------------------------------------
#Normalmente no se hacen los fitting con todos los datos sino que se dividen los
#datos en testing y training para evitar overfitting. Eso es lo que vamos a hacer ahora
from sklearn.model_selection import train_test_split
#Divido mis datos X e Y en testing (30% de los datos) y training. random_state fija la semilla de la división aleatoria.
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=0) 
#creamos un modelo logístico que se ajuste a los datos de entrenamiento
lm = linear_model.LogisticRegression()
lm.fit(X_train, Y_train)



#Recordamos que el objetivo de usar una regresión logística es obtener unas 
#pobabilidades para unos valores. Para obternerlas:
probs = lm.predict_proba(X_test) #1º columna=probs, cómo de seguro estoy.
                                #2º columna=prob del valor de salida.
print(probs)
#Por defecto, si la prob del valor de salida es superior a 0.5, la observación 
#se categoriza como resultado positivo (compra) y si es menor, como negativo(no compra)
prediction = lm.predict(X_test) #array de 0 (no compra, prob<0.5) y 1 (compra, prob>0.5)
#También podemos elegir nosotros el thershold ditinto a 0.5

#en nuetsro caso solo un 10% de clientes cmpra el producto por lo que quizas
#definir el threshold en 0.1 puede ser una buena opción

prob = probs[:,1] #me quedo solo con la segunda columna
prob_df = pd.DataFrame(prob) #creo un dataframe con las probs
threshold = 0.1
#creamos en el df una columna que se llama prediccion que cuando prob_df en la columna 0 sea>threshold --> pomdremos un 1 y sino como 0
prob_df ["prediction"] = np.where(prob_df[0]>threshold,1,0)
print(prob_df.head())

#creo una tabla para ver cuantos clientes se esperan que compren y cuantos no
print(pd.crosstab(prob_df.prediction, columns="count")) #count es para que cuante lod datos de cada caso
print(390/len(prob_df)*100) # %casos positivos

from sklearn import metrics #para hacer comparaciones
metrics.accuracy_score(Y_test, prediction) #me dice el % de datos que coinciden de Y_test y mi predicicon
#ME SALE 0.9, ES DECIR 90%, muy bien
#-----------------------------------------------------------------------------
## k-FOLD CROSS-VALIDATIOM con SKLEARN
#-----------------------------------------------------------------------------
"""
*El conjunto de datos lo dividimos en K particiones 
*De esas K particiones una se utiliza siempre como conjunto de testing y las
otras k-1 todas ellas juntas se usan como conjunto de entrenamiento.
*Lo anterior se repite k veces, cada una de las particiones será usada 1 vez como
conjunto de testing.
*Para cada una de las iteraciones se medirá la eficacia del modelo y tras k iteraciones
tendremos k niveles de eficacia y podremos promediar la eficacia final del modelo

"""
from sklearn.model_selection import cross_val_score
scores = cross_val_score(linear_model.LogisticRegression(), X, Y, scoring = "accuracy", cv = 10)
# el score=accuracy me calcula la eficacia del modelo. Me dirá qué porcentaje se desvia en cada iteración
#cv= número de iteraciones
print(scores)
print(scores.mean()) #esta bien mirar si con cv mayor o menor se reduce o aumenta mucho el esrror (scores)
#-----------------------------------------------------------------------------
## MATRICES DE CONFUSIÓN Y CURVAS ROC
#-----------------------------------------------------------------------------
#Separamos en test y testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)
#creamos el modelo con los datos de training
lm = linear_model.LogisticRegression()
lm.fit(X_train, Y_train)
#Validaos cómo de bueno es el modelo
probs = lm.predict_proba(X_test)
#comparo con la variabel Y real
prob=probs[:,1]
prob_df=pd.DataFrame(prob)
threshold=0.1
prob_df["prediction"]=np.where(prob_df[0]>threshold,1,0)
prob_df["actual"]=list(Y_test) #pongo el list para perder los índices del dataset original
print(prob_df.head())
#para conocer la tasa de falsos positivos y falsos negativos elaboramos una matriz de confusión
confusion_matrix=pd.crosstab(prob_df.prediction,prob_df.actual)
print(confusion_matrix)

"""
actual        0      1
prediction
0            806(TP)    40(TN)
1            308(FN)    82(FP)

CUIDADO PORQUE A MI JO ME PARECE INTUITIVO A QUÉ SE LE LLAMA FN Y FP
"""
TN=confusion_matrix[0][0]
TP=confusion_matrix[1][1]
FP=confusion_matrix[0][1]
FN=confusion_matrix[1][0]

sens=TP/(TP+FN)
esp_1=1-TN/(TN+FP)

#Lo suyo es hacer otra vez todo pero para distintos thresholds
thresholds = [0.04,0.05,0.07,0.10,0.12,0.15,0.18,0.20,0.25,0.3,0.4,0.5]
sensitivities = [1]
especifities_1 = [1]

for t in thresholds:
    prob_df["prediction"]=np.where(prob_df[0]>threshold,1,0)
    prob_df["actual"]=list(Y_test)
    print(prob_df.head())
    confusion_matrix=pd.crosstab(prob_df.prediction,prob_df.actual)
    print(confusion_matrix)
    
    
    TN=confusion_matrix[0][0]
    TP=confusion_matrix[1][1]
    FP=confusion_matrix[0][1]
    FN=confusion_matrix[1][0]
    
    sens=TP/(TP+FN)
    sensitivities.append(sens)
    esp_1=1-TN/(TN+FP)
    especifities_1.append(esp_1)
sensitivities.append(0)    
especifities_1.append(0)  #para que al plotearlo acabe en la diagonal
#pintamos ahora la curva
import matplotlib.pyplot as plt
"""%matplotlib inline
plt.plot(especifities_1,sensitivities, marker="o", linestyle="--", color="r")
x=[i*0.01 for i in range(100)]
y=[i*0.01 for i in range(100)]
plt.plot(x,y) #pinto la diagonal (el peor modelo que existe)
plt.xlabel("1-Especificidad")
plt.ylabel("Sensibilidad")
plt.title("Curva ROC")
#recordemos que mi seleccion de variables era una mierda absoluta
"""
#cuanto mayor sea el área entre la curva y la diagonal, mejor es el modelo predictivo
from sklearn import metrics
from plotnine import ggplot, aes, geom_line, geom_area, ggtitle, xlim, ylim  #si quiero importar todo pongo solo *
espec_1, sensit, _ = metrics.roc_curve(Y_test,prob)
df = pd.DataFrame({
        "x": espec_1,
        "y": sensit        
})

auc = metrics.auc(espec_1, sensit) #área bajo la curva

print(df.head())
print(ggplot(df,aes(x="x", y="y")) + geom_line() + geom_line(linetype="dashed")+xlim(-0.01,1.01)+ylim(-0.01,1.01))
print(ggplot(df,aes(x="x",y="y")) + geom_area(alpha=0.25) + geom_line(aes(y="y")) + ggtitle("Curva ROC y AUC=%s " %str(auc)))
