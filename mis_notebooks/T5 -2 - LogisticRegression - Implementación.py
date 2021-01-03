# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:45:46 2020

@author: 3A

ÍNDICE: Implementación del método de la máxima verosimilitud para la
        regresión logístico
------------------------------------------------------------------------------
*Definición de la función de entorno
*Cálculo de las probabilidades para cada observación
*Cálculo de la matriz diagonal w
*Obtención de la función logística (Método Newton Raphson)
*Comprobación experimental
"""

#-----------------------------------------------------------------------------
## DEFINICIÓN DE LA FUNCIÓN DE ENTORNO
#-----------------------------------------------------------------------------
from IPython.display import display, Math, Latex
display(Math(r'L(\beta)=sum_{i=1}^n P_i^{y_i}(i-Pi)^{y_i}'))

def likelihood(y,pi): #vector yi y el vector probabilidades
    import numpy as np
    total_sum = 1
    sum_in = list(range(1,len(y)+1))
    for i in range(len(y)):
        sum_in[i] = np.where(y[i]==1,pi[i], 1-pi[i])
        total_sum = total_sum * sum_in[i]
    return total_sum

#-----------------------------------------------------------------------------
## CÁLCULO DE PROBABILIDADES PARA CADA OBSERVACIÓN
#-----------------------------------------------------------------------------
display(Math(r'Pi = P(x_i) = \frac{1}{1+e^{-\sum_{j=0}^k\beta_j\cdot x_ij}}'))

def logitprobs(X, beta):
    import numpy as np
    n_rows = np.shape(X)[0] #número de filas
    n_cols = np.shape(X)[1] #número columnas
    pi = list(range(1, n_rows+1))
    exponente = list(range(1,n_rows+1))
    for i in range(n_rows):
        exponente[i]= 0
        for j in range(n_cols):
            ex =X[i][j] * beta[j]
            exponente[i] = ex + exponente[i]
    with np.errstate(divide = "ignore", invalid = "ignore"):
        pi[i]=1 / (1 + np.exp(-exponente[i]))
    return pi
#-----------------------------------------------------------------------------
## CÁLCULO DE LA MATRIZ DIAGONAR W
#-----------------------------------------------------------------------------
display(Math(r'W = diag(P_i \ cdot (1 - P_i))_{i=1}^n'))

def findW(pi):
    import numpy as np
    n = len(pi)
    W = np.zeros(n*n).reshape(n,n) #np.zeros cre una matriz de ceros del tamaño que yo le especifique.
                                    #Tendremos n*n=n^2 ceros distribuidos en n filas y n columnas (filas, columnas)
    #Como W es una matriz diagonal, ahora solo ustituímos los ceros de la diagonal por su correspondiente valor
    for i in range(n):
        print(i)
        W[i,i] = pi[i]*(1-pi[i])
        W[i,i].astype(float) #para que no haya errores futuros
    return W
#-----------------------------------------------------------------------------
## OBTENCIÓN DE LA SOLUCIÓN DE LA FUNCIÓN LOGÍSTICA (M. NEWTON RAPHSON)
#-----------------------------------------------------------------------------
display(Math(r"beta_{n+1} = beta_n - \frac{f(beta_n)}{f'(beta_n)}")) 
display(Math(r"f(X) = X(Y-P"))  
display(Math(r"f'(beta) = XWX^T")) 

def logistics(X, Y, limit): #vecto de las X, vector de las predicciones que quiero hacer Y y un límite de iteraciones limit
    #El metodo de Newton raphson podría no converger nunca, por lo que me ayudaré de un ñimite de iteraciones()
    import numpy as np
    from numpy import linalg
    n_row=np.shape(X)[0]
    bias = np.ones(n_row).reshape(n_row, 1) #creo un vector todo de unos con el mismo número de filas (n_row) y lo bvamos a poner en columna(n_row fials,1 columna)
    X_new = np.append(X, bias, axis = 1) #añado al vector original la columna bias. Con lo de axis=1 me asegura que lo añade al final de las columnas preexistentes y no como una fila
    n_col = np.shape(X_new)[1] #número de columnas
    beta = np.zeros(n_col).reshape(n_col,1) #vector de ceros de tamaño n_col. Lo convierto en columna
    root_dif =np.array(range(1,n_col+1)).reshape(n_col,1)
    iter_i=10000
    while(iter_i>limit):
        print("Iter: i " + str(iter_i) + ", limit: " + str(limit))
        pi =logitprobs(X_new, beta)
        print("Pi: " + str(pi))
        W = findW(pi)
        print("W: " + str(W))
        #numerador y denom de newon raphson:
        num = (np.transpose(np.matrix(X_new))*np.matrix(Y - np.transpose(pi)).transpose()) #los transpose es para poder multiplicarse
        den = (np.matrix(np.transpose(X_new))*np.matrix(W)*np.matrix(X_new))
        #incremento de las raíces del sistema
        root_dif = np.array(linalg.inv(den)*num) #inversa del denom * el numerador
        beta = beta + root_dif
        print("Beta: " + str(beta))
        iter_i = np.sum(root_dif*root_dif)
        ll = likelihood(Y, pi) #calculo del entorno con las predicciones y probabiidades
    return beta
#-----------------------------------------------------------------------------
## COMPROBACIÓN MÉTODO EXPERIMENTAL
#-----------------------------------------------------------------------------
import numpy as np
X = np.array(range(10)). reshape(10,1)
print(X)
Y = [0,0,0,0,1,0,1,0,1,1] #el clasificador Y ha de ser binario
bias = np.ones(10).reshape(10,1)
X_new = np.append(X, bias, axis = 1)
print(X_new)

#a = logistics(X,Y,0.00001)    #Por alguna razon no me sale bien
#ll =likelihood (Y, logitprobs(X, a))
##Nos sale: Y=0.66220827 * X - 3.69557172
#-----------------------------------------------------------------------------
## PAQUETE STATSMODELS DE PYTHON
#-----------------------------------------------------------------------------


import statsmodels.api as sm 
logit_model = sm.Logit(Y,X_new)
result =logit_model.fit()
print(result.summary2())

"""
#Iteraciones:6
#Índices de acaique AIC: y bayesianoi BIC: 
    #coefs
Podría ser que el coef de X1 fuera O porque el intervalo de confianza [0.025,0.975] incluye al cero
Puede que el odelo log no sea el más apropiado

"""    
