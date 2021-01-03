# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 18:04:28 2020
@author: 3A

ÍNDICE
------------------------------------------------------------------------------
*Tratamiento de las variables categóricas en un dataset para ajustes lineales
*Eliminar variables dummy redundantes
*Transformación de variables para conseguir una relación no lineal
*El problema de los outliers

"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

mainpath="C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master/datasets"
filepath = "ecom-expense/Ecom Expense.csv"
fullpath = mainpath +"/" + filepath

df = pd.read_csv(fullpath)
print(df.head())
#-----------------------------------------------------------------------------
## TRATAMIENTO DE LAS VARIABLES CATEGÓRICAS 
#-----------------------------------------------------------------------------
#creo dos variables dummy: 1 para el gender y otra para el city tier
dummy_gender = pd.get_dummies(df["Gender"], prefix="Gender")
dummy_city_tier = pd.get_dummies(df["City Tier"], prefix="City")
print(dummy_gender.head())
print(dummy_city_tier.head())
#Las añadimos al df original para usarlas en el modelo

column_names = df.columns.values.tolist()
print(column_names) #El que ha exportado los datos lo ha hecho mal y en algún nombre se le ha colado espacios.
df_new =df[column_names].join(dummy_gender)
column_names = df_new.columns.values.tolist()
print(column_names)
df_new =df_new[column_names].join(dummy_city_tier)
print(df_new.head())

#incluyo estas dummy en el modelo y elimino las variables que no me interesen.
#Vaya, elijo mis variables predictoras
feature_cols=["Monthly Income", "Transaction Time", "Gender_Female", "Gender_Male", "City_Tier 1","City_Tier 2","City_Tier 3", "Record"]

X=df_new[feature_cols] #variables predictoras
Y=df_new["Total Spend"] #Variable predictiva

lm = LinearRegression() #creo el modelo lineal
lm.fit(X,Y)             #con las variables X e Y

print(lm.intercept_) #ordenada en el origen del modelo lineal
print(lm.coef_) #coefs de las fariables
print(list(zip(feature_cols, lm.coef_))) #me lo muestra (nombre_variable, coef_correspondiente)
print(lm.score(X,Y)) #R^2 = 0.9179923586131016, si me hubiera salido peque, puedo probar a ir añadiendo variables 
"""
El modelo puede ser escrito como:
    Total_Spend = -79.4171303013718 + 'Monthly Income'* 0.14753898049205738 
                    + 'Transaction Time'* 0.15494612549589634 + 'Gender_Female'* 
                    - 131.02501325554624 + 'Gender_Male'* 131.02501325554607 + 
                    + 'City_Tier 1'* 76.76432601049513 +'City_Tier 2'* 55.1389743092325 + 
                    +'City_Tier 3'* - 131.9033003197277 +'Record'* 772.2334457445645

*Si es hombre y vive en CT1: 
    Total_Spend = 128.37220896466724 + 'Monthly Income'* 0.14753898049205738 
                    + 'Transaction Time'* 0.15494612549589634 + 
                    - 131.9033003197277 +'Record'* 772.2334457445645
*Si es hombre y vive en CT2: 
    Total_Spend = 106.74685726340445 + 'Monthly Income'* 0.14753898049205738 
                  + 'Transaction Time'* 0.15494612549589634 +'Record'* 772.2334457445645
*Si es hombre y vive en CT3: 
    Total_Spend = -80.29541736555583 + 'Monthly Income'* 0.14753898049205738 
                  + 'Transaction Time'* 0.15494612549589634 +'Record'* 772.2334457445645
*Si es mujer y vive en CT3: 
   etc
"""
# Total_Spend= -79.4171303013718 + 'Monthly Income'* 0.14753898049205738 +
#                    'Transaction Time'* 0.15494612549589634 + 'Gender_Female'* -131.02501325554624
#                   + 'Gender_Male'* 131.02501325554607 + 'City_Tier 1'* 76.76432601049513 +
#                   + 'City_Tier 2'* 55.1389743092325 + 'City_Tier 3'* -131.9033003197277+
#                   +'Record'* 772.2334457445645
df_new["prediction"] = -79.4171303013718 + df_new['Monthly Income']* 0.14753898049205738 + df_new['Transaction Time']* 0.15494612549589634 + df_new['Gender_Female']* -131.02501325554624 + df_new['Gender_Male']* 131.02501325554607 + df_new['City_Tier 1']* 76.76432601049513 + df_new['City_Tier 2']* 55.1389743092325 + df_new['City_Tier 3']* -131.9033003197277 + df_new['Record']* 772.2334457445645
print(df_new["prediction"].head())
#Esta última columna predicción lo podía haber hecho con la función predict del paquete lm de pandas:
df_new_pred = lm.predict(pd.DataFrame(df_new[feature_cols])) #aquí tendría que añadir una columna a df_new y con el método anterior se añade sola
print(df_new_pred)

SSD = np.sum((df_new["prediction"]- df_new["Total Spend"])**2) #%1517733985.3408163
RSE = np.sqrt(SSD/(len(df_new)-len(feature_cols)-1)) #número de variables predictoras -1
sales_mean = np.mean(df_new["Total Spend"]) 
error = RSE/sales_mean #13%

#-----------------------------------------------------------------------------
## ELIMINAR VARIABLES DUMMY REDUNDANTES
#-----------------------------------------------------------------------------
#Si tenemos una variable dummy con dos columnas, solo necesitaremos 1 para 
#incluirla en el modelo. Solo con usar una ya se sabe el valor de la otra por 
#lo que es redundante
 
dummy_gender2 = dummy_gender.iloc[:,1:] #me quedo con tudas las filas pero solo de la segunda columna (cojo todas las columnas a partir de la 1, es decir, me salto la 0, que recoge los datos fem)
print(dummy_gender2.head())

dummy_city_tier2 = dummy_city_tier.iloc[:,1:]  #me quedo con CT2 y CT3
print(dummy_city_tier2.head())

column_names = df.columns.values.tolist()
df_newnew = df[column_names].join(dummy_gender2)
column_names = df_newnew.columns.values.tolist()
print(column_names)
df_newnew =df_newnew[column_names].join(dummy_city_tier2)
print(df_new.head()) #ahora tenemos menos variables dummies: 1 para el gendermale y 2 pata CT2 y CT3

feature_cols = ["Monthly Income", "Transaction Time", "Gender_Male", "City_Tier 2", "City_Tier 3", "Record"]
X=df_newnew[feature_cols] #variables predictoras
Y=df_newnew["Total Spend"] #Variable predictiva
lm = LinearRegression() #creo el modelo lineal
lm.fit(X,Y)             #con las variables X e Y

print(lm.intercept_) #ordenada en el origen del modelo lineal
print(lm.coef_) #coefs de las fariables
print(list(zip(feature_cols, lm.coef_))) #me lo muestra (nombre_variable, coef_correspondiente)
print(lm.score(X,Y)) #El modelo es igual que el anterior, con el mismo score, pero +  simplificado, con menos variables

"""
Coeficientes con todas las variables del modelo:
    *('Monthly Income', 0.14753898049205738), 
    *('Transaction Time', 0.15494612549589634), 
    *('Gender_Female', -131.02501325554624), 
    *('Gender_Male', 131.02501325554607), 
    *('City_Tier 1', 76.76432601049513), 
    *('City_Tier 2', 55.1389743092325), 
    *('City_Tier 3', -131.9033003197277), 
    *('Record', 772.2334457445645)

Coeficientes tras enmascarar las variables dummy pertinentes:
    *('Monthly Income', 0.14753898049205744), 
    *('Transaction Time', 0.1549461254959002), 
    *('Gender_Male', 262.0500265110948), 
    *('City_Tier 2', -21.62535170126276), 
    *('City_Tier 3', -208.66762633022296), 
    *('Record', 772.2334457445636)

Los cambios re reflejan en Gender male y female y en los city teil.
El gender_male ha de contener la info de un hombre viviendo en CT1 y el CT2 Y 3
contiene la unfo de una mujer que viva en CT2 o CT3.
    *Gender Male: 
        ·antes-->131.02; después:262.05 = (131.02-(-131.02))
    *Gender Female: 
        ·antes--> -131.02; después:0 = (-131.02-(-131.02))
    *CT1: 
        ·antes--> 76.76; después: 0 = (76.76 -76.76)
    *CT2: 
        ·antes--> 55.13; después: -21.62 = (55.13-76.76)
    *CT3: antes--> -131.9; después: -208.66 = (-131.9 -76.76)
"""
#-----------------------------------------------------------------------------
## TRANSFORMACIÓN DE  VARIABLES PARA CONSEGUIR UNA RELACIÓN NO LINEAL
#-----------------------------------------------------------------------------
mainpath="C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master/datasets"
filepath = "auto/auto-mpg.csv"
fullpath = mainpath +"/" + filepath

data_auto = pd.read_csv(fullpath)
print(df.head())
print(data_auto.shape) #(filas, columnas)

import matplotlib.pyplot as plt

%matplotlib inline
#primero eliminamos los NaN
data_auto["mpg"]= data_auto["mpg"].dropna()
data_auto["horsepower"]= data_auto["horsepower"].dropna()
plt.plot(data_auto["horsepower"], data_auto["mpg"], "ro")
plt.xlabel("Caballos de potencia")
plt.ylabel("Consumo (millas por galeón)")
plt.title("CV vs mpg")

#No parece lineal sino curvo, pero aun asi intentaremos ajustarlo a una recta 
#Si no resulta un buen ajuste optaremos por uno cuadrático, cúbico...

#MODELO LINEAL: mpg = a + b * horsepower
X = data_auto["horsepower"].fillna(data_auto["horsepower"].mean()) #reemplazo los Nan por  la media para poder representar valores
Y = data_auto["mpg"].fillna(data_auto["mpg"].mean())
X_data = X[:, np.newaxis]
lm = LinearRegression() #Esta función espera que X sea un dataFrame y en nuetsro caso es un array
lm.fit(X_data,Y)

print(type(X)) #array
print(type(X_data)) #array n dimensional (se ha creado una nueva dimensión)


%matplotlib inline
plt.plot(X, Y, "ro")
plt.plot(X, lm.predict(X_data), color = "blue")
plt.xlabel("Caballos de potencia")
plt.ylabel("Consumo (millas por galeón)")
plt.title("CV vs mpg") 

print(lm.score(X_data,Y))  #R^2 adj = 0.57 -->  aceptable pero no muy bueno
SSD = np.sum((Y - lm.predict(X_data))**2)
RSE = np.sqrt(SSD/len(X_data)-1)
y_mean = np.mean(Y)
error =RSE/y_mean
print(SSD, RSE, y_mean, error) #Aceptable pero no muy bueno

#MODELO DE REGRESIÓN CUADRÁTICO: mpg = a + b * horsepower^2

X_data = X**2
X_data = X_data[:, np.newaxis]
lm = LinearRegression() #Esta función espera que X sea un dataFrame y en nuetsro caso es un array
lm.fit(X_data,Y)

print(lm.score(X_data,Y))  #R^2 adj = 0.48 -->  baja
SSD = np.sum((Y - lm.predict(X_data))**2)
RSE = np.sqrt(SSD/len(X_data)-1)
y_mean = np.mean(Y)
error =RSE/y_mean #incrementa a un 23%
print(SSD, RSE, y_mean, error) #No parece que funcione este modelo


#MODELO DE REGRESIÓN LINEAL Y CUADRÁTICO: mpg = a + b * horsepower + C * horsepower^2

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

poly = PolynomialFeatures(degree=2) #para crear un polinomio de grado max = 2
X_data = poly.fit_transform(X[:,np.newaxis]) #para la transformación lineal a partir de los datos originales X
lm =linear_model.LinearRegression() #cambiamos el método para hacer la regresión
lm.fit(X_data,Y)

print(lm.score(X_data, Y)) #aumenta R^2 BIEEEN
print(lm.intercept_)
print(lm.coef_)

# mpg = 55.02619244708036  -0.43404318 * horsepower + 0.00112615 * horsepower^2

def regresion_validation(X_data, Y, Ypred):
    SSD = np.sum((Y - Y_pred)**2)
    RSE = np.sqrt(SSD/len(X_data)-1)
    y_mean = np.mean(Y)
    error =RSE/y_mean 
    print("SSD: "+ str(SSD) + ",   RSE: " + str(RSE) + ",   y_mean: " + str(y_mean) + ",   error: " + str(error*100)+ "%") #No parece que funcione este modelo
 

X_data = X[:, np.newaxis]

for d in range (2,6): #d=2, 3, 4, 5
    poly = PolynomialFeatures(degree=d) #para crear un polinomio de grado max = 2
    X_data = poly.fit_transform(X[:, np.newaxis]) #para la transformación lineal a partir de los datos originales X
    lm =linear_model.LinearRegression() #cambiamos el método para hacer la regresión
    lm.fit(X_data,Y)
    Y_pred = lm.predict(X_data)
    print("Regresión de grado" + str(d))
    print("R2:" + str(lm.score(X_data, Y)))
    regresion_validation(X_data, Y, Y_pred)
#-----------------------------------------------------------------------------
## EL PROBLEMA DE LOS OUTLIERS
#-----------------------------------------------------------------------------
# Datos extremos no representativos que modifican negativamente al modelo



X = data_auto["displacement"].fillna(data_auto["displacement"].mean())
X = X[:,np.newaxis]
Y = data_auto["mpg"].fillna(data_auto["mpg"].mean())

lm=LinearRegression()
lm.fit(X,Y)

print(lm.score(X,Y)) #R^2 = 0.6261049762826918, no está mal
plt.show()
%matplotlib inline
plt.plot(X,Y, "ro")
plt.plot(X, lm.predict(X), color="blue")

#si quirara los ouliers mejoraría R^2?

print(data_auto[(data_auto["displacement"]>250) & (data_auto["mpg"]>35)]) #a partir de la gráfica he localizado el outlier, este
print(data_auto[(data_auto["displacement"]>300) & (data_auto["mpg"]>20)]) #a partir de la gráfica he localizado el outlier, este

#me sobran las filas 395, 258, 305 y 372

data_auto_clean = data_auto.drop([395,258,305, 372])

X = data_auto_clean["displacement"].fillna(data_auto_clean["displacement"].mean())
X = X[:,np.newaxis]
Y = data_auto_clean["mpg"].fillna(data_auto_clean["mpg"].mean())

lm=LinearRegression()
lm.fit(X,Y)

print(lm.score(X,Y)) #R^2 = 0.6466514317531822, mejora un poco
plt.show()
%matplotlib inline
plt.plot(X,Y, "ro")
plt.plot(X, lm.predict(X), color="blue")