
import os.path
import numpy as np
import operator
import matplotlib.pyplot as plt
# archivo-entrada.py

def inputTime():
    seconds = input("Por favor ingrese el tiempo máximo de ejecución en segundos: ")
    if len(seconds) == 0:
        print('Debe ingresar un valor')
        inputTime()
    else:
        if seconds.isdigit():
            return int(seconds)
        else:
            print('Debe ingresar un valor numérico.\n')
            inputTime()

def inputFile():
    name_file = ""
    name_file = input("Por favor ingrese el nombre de uno de los casos de prueba contenidos en 'Benchmark Instances TSPTW': ")
    if len(name_file) == 0:
        print('Debe ingresar el nombre del archivo para iniciar')
        inputFile()
    else:
        if existsFile(name_file):
            return str(name_file)
        else:
            print('El archivo: [' + name_file + '] no existe.\n')
            inputFile()

def getPath():
    return 'SolomonPotvinBengio/'

def existsFile(name_file):
    path = getPath() + name_file
    if os.path.isfile(path):
        return True
    else:
        return False

def leerarchivo(archivo):
    folder = getPath()
    path = folder + archivo
    file = open (path,'r')
    txt = file.readlines()
    file.close()
    return txt

def getnodos(txt):
    nodos = txt[0].replace('\n','')
    return int(nodos)

def getciudades(nodos):
    ciudades = [i for i in range(nodos)]
    return ciudades

def getarcos(ciudades):
    arcos=[(i,j) for i in ciudades for j in ciudades]
    return arcos

def getmatrizdist(nodos, arcos, txt):
    datos = []
    for i in range (1,nodos + 1,1):
        row = txt[i].replace('\n','')
        linea = row.split(' ')
        datos.append(linea)
    
    distancias = {(i,j): float(datos[i][j]) for i,j in arcos  }
    return distancias

def getmatrizventanas(nodos, txt):
    rest = []
    for j in range (nodos + 1,nodos + nodos + 1,1):
        linea = txt[j].split(' ')
        auxarray = []
        for reg in linea:
            reg = reg.replace('\n','')
            if len(reg) > 0:
                auxarray.append(reg)   
        rest.append(auxarray)

    return rest

def getmatrizorderv(ciudades, matrizventanas, type):
    rest_ini = {}
    rest_fin = {}
    result = {}
    i = 0
    for t in ciudades:
        rest_ini[t] = int(matrizventanas[t][0])
        rest_fin[t] = int(matrizventanas[t][1])

    if type == "inicial":
        orderv = {k: v for k, v in sorted(rest_ini.items(), key=lambda item: item[1])}
    
    if type == "final":
        orderv = {k: v for k, v in sorted(rest_fin.items(), key=lambda item: item[1])}

    for x in orderv:
        result[i] = {0:rest_ini[x],1:rest_fin[x]}
        i += 1
    
    return result