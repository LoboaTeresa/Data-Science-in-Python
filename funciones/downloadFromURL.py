# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 00:40:34 2020
@author: Teresa

Función que abre un archivo de datos de una url, lo decodifica en utf-8 y lo guarda en 
un fichero de tu ordenador en la extensión que CVS, EXCEL y JASON
"""
def downloadFromURL(url, filepath, sep = ",", delim = "\n", encoding = "utf-8",  mainpath = "C:/Users/Usuario/Teresa/CursoMachineLearningDataSciencePython/python-ml-course-master/datasets"):

    #primero importamos la librería y hacemos la conexión con la web de datos
    import pandas as pd
    import os
    import urllib3
    
    
    
    http =urllib3.PoolManager()
    r = http.request("GET", url)
    r.status
    response=r.data
    
    #El objeto response contiene un STRING BINARIO, así que lo convertimos a un string descodificandolo en UTF-8
    str_data = response.decode (encoding)
    
    #Dividimos el string en un array de filas, separándolo por intros
    lines = str_data.split(delim)
    
    #La primera lína contiene la cabecera, así que la extraemos
    col_names = lines[0].split(sep)
    n_cols = len(col_names) #para saber cuantas columnas tengo
    
    #Generamos un diccionario vacío donde irá la info procesada desde la URL externa
    counter = 0
    main_dict ={}
    for col in col_names:
        main_dict [col] = []
        
    #Procesamos fila a fila la info para ir rellenando el diccionario con los datos
    for line in lines:
        #nos saltamos la primera fila
        if (counter > 0):
            #Dividimos cada string por las comas como separador
            values=line.strip().split(",")
            #Añadimos cada valor a su respectiva columna del diccionario
            for i in (range(n_cols)):
                main_dict[col_names[i]].append(values[i])
        counter+= 1 
            
    print ("El dataset tiene %d filas y %d columnas" % (counter,n_cols))
    
    #convertimos el diccionario procesado en Data Frame y comprobamos que los datos son correctos
    df = pd.DataFrame(main_dict)
    
    #Elegimos donde guardarlo (en la carpeta donde tiene más sentido por el contexto del analisis)
    fullpath = os.path.join(mainpath, filepath)
    
    #Lo guardamos en CSV, en JSON o en EXCEL según queramos
    df.to_csv(fullpath+".csv")
    df.to_json(fullpath+".json")
    df.to_excel(fullpath+".xls")
    
    #print("Los datos se han guardado correctaente en " + fullpath + "en las extensiones .csv, .jason y .xls")
    return df


"""También podríamos leer csv de una url de la siguiente manera y sin funcion alguna:
    urldata = "https://raw.githubusercontent.com/joanby/python-ml-course/master/datasets/titanic/titanic3.csv"
    data = pd.read_csv(urldata)
    print(data.head())
"""