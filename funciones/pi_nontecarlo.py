# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 20:42:36 2020
@author: 3A
-----------------------------------------------------------------------------
 SIMULACION MONTECARLO PARA OBTENER EL VALOR DE PI
-----------------------------------------------------------------------------
1) Genero dos números aleatorios entre 0 y 1 (x e y) en total n veces
2) Calculamos X * X + Y * Y (x e y han de seguir una distrib uniforme)
    *Si el valor de lo anterior es inferior a 1 --> estamos dentro del círculo
    *Si el valor de lo anterior es superior a 1 --> estamos fuera del círculo
3) Calculamos el número total de puntos dentro del círculo y lo dividimos por 
   el número total de intentos para obtener una aprox de la prob de caer dentro 
   del círulo.
4) Usamos dicha prob para aproximar el valor de pi.
5) Repetimos el experimento un númiro suficiente de veces (ej: n_exp), para obtener 
  (n) aproximaciones diferentes de pi.
6) Calculamos el promedio de los n_exp experimentos anteriores para dar un valor 
   final de pi.
"""

def pi_montecarlo(n, n_exp):
    pi_avg = 0
    pi_value_list =[]
  
    for i in range(n_exp):
        value = 0
        x = np.random.uniform(0,1,n).tolist()
        y = np.random.uniform(0,1,n).tolist()
        for j in range(n):
            z=np.sqrt(x[j] * x[j] + y[j] * y[j])
            if z<=1:
                value += 1
        float_value = float(value)
        pi_value = float_value * 4 /n
        pi_value_list.append(pi_value)
        pi_avg += pi_value
    pi = pi_avg/n_exp
    
    print(pi)
    plt.plot(pi_value_list)   
    
    return (pi, fig)