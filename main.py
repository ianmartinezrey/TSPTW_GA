import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
import loaddata
import sys
from math import sqrt, log
from multiprocessing import Process

# Diccionario global donde se guardan las distancias de una ciudad a otra.
distancias = {}
ordervini = []
progress = []

class City:
    def __init__(self,name):
        self.name = int(name)
    
    def distance(self, city):
        distance = distancias[self.name,city.name]
        
        return distance

    def __repr__(self):
        return "(" + str(self.name)+ ")"


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    #Se debe penalizar si la ruta no esta dentro de la ventana de tiempo, con esto se asignara mayor
    #valor fitness a la ruta que cumpla con las ventanas
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]

                if(ordervini[i][0] > pathDistance or  pathDistance > ordervini[i][1]):
                    penality = 50
                else:
                    penality = 0

                pathDistance += fromCity.distance(toCity) + penality
                #print(fromCity,'->',toCity,' = ',pathDistance,' wt: ',ordervini[i][0],'-',ordervini[i][1],' | penality: ',penality)
            self.distance = pathDistance
            #print(pathDistance, penality)
            #print(self.distance,'\n\n')
        return self.distance
    
    #Se asigna el mejor fitness a la ruta que tenga la menor distnacia incluyendo las posibles penalizaciones
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

#Creacion de ruta aleatoria
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

#Crear poblacion inicial
def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

#Se crea una lista ordenada con las mejores rutas
def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    sorted_results=sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)
    return sorted_results


#Crear una función de selección que se utilizará para hacer la lista de rutas principales
def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

#Creacion de pool
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

#Funcion crossover para que dos padres creen un hijo
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        

    childP2 = [item for item in parent2 if item not in childP1]
    #print(startGene, endGene)
    #print(parent1)
    #print(parent2)
    #print(childP1)
    #print(childP2)
    child = childP1 + childP2
    #print(child)
    return child

#Funcion para hacer crossover sobre el pool
def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children


#Funcion que muta una ruta
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

#Funcion para mutar toda la poblacion
def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

#Funcion que ejecuta todos los pasos para crear la siguiente generacion
def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

#Funcion para hacer debug --verbose
def printroute( best_route ):
    pasado = None
    acum = 0
    vfirst = None
    vlast = None
    s = "| " + '#' + " | " + 'Desde' + " | " + 'Hasta' + " | " + 'Dist' + " | " + 'Acum' + " | " + 'Twi' + " | " + 'Twf' + " | Penality " + "\n"
    idx = 0
    for j in best_route:
        penality = 0
        if vfirst is None: vfirst = j
        if pasado:
            vlast = j
            acum += pasado.distance(j)
            if(ordervini[idx][0] > acum or  acum > ordervini[idx][1]):
                penality = 50
            else:
                penality = 0
            #print(fromCity,'->',toCity,' = ',pathDistance,' wt: ',ordervini[i][0],'-',ordervini[i][1],' | penality: ',penality)
            s += "| " + str(idx) + " | " + str(pasado) + " | " + str(j) + " | " + str(pasado.distance(j)) + " | " + str(round(acum, 4)) + " | " + str(ordervini[idx][0]) + " | " + str(ordervini[idx][1]) + " | penality: " + str(penality) + "\n"
            #print(s)
            idx += 1
        pasado = j
        
    acum += vlast.distance(vfirst)
    s += "| " + str(idx) + " | " + str(vlast) + " | " + str(vfirst) + " | " + str(vlast.distance(vfirst)) + " | " + str(round(acum, 4)) + " | " + str(ordervini[idx][0]) + " | " + str(ordervini[idx][1]) + " | penality: " + str("0") + "\n"

    return s

#Funcion para crear el algoritmo genetico
def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    global progress
    pop = initialPopulation(popSize, population)
    progress = [1 / rankRoutes(pop)[0][1]]
    print("Distancia Inicial: " + str(progress[0]))

    for i in range(1, generations+1):
        pop = nextGeneration(pop, eliteSize, mutationRate)    
        progress.append(1 / rankRoutes(pop)[0][1])
        #if i%50==0:
        print('Generacion '+str(i),"Distancia: ",progress[i])
    
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    
    return bestRoute

def plotfitnessxGeneration():
    global progress
    plt.plot(progress)
    plt.ylabel('Distancia')
    plt.xlabel('Generacion')
    plt.title('Mejor Fitness vs Generacion')
    plt.tight_layout()
    plt.show()


def execStartProcesss(name_file):
    global distancias
    global ordervini

    # Se cargan los datos del archivo 
    if len(name_file) > 0:
        txt = loaddata.leerarchivo(name_file)
        nodos = loaddata.getnodos(txt)
        ciudades = loaddata.getciudades(nodos)
        arcos = loaddata.getarcos(ciudades)
        distancias = loaddata.getmatrizdist(nodos, arcos, txt)
        matrizventanas = loaddata.getmatrizventanas(nodos,txt)
        ordervini = loaddata.getmatrizorderv(ciudades,matrizventanas,"inicial")
        cityList = []
        for i in range(nodos): 
            cityList.append(City(name = i))

        #Ejecutar el proceso
        best_route=geneticAlgorithm(population=cityList, popSize=30, eliteSize=10, mutationRate=0.01, generations=25)
        print('Mejor Ruta:')
        print(best_route)
        print('Detalle:')
        print(printroute(best_route))
        plotfitnessxGeneration()


def execTimeLimit(func, args, time):
    p = Process(target=func, args=args)
    p.start()
    p.join(time)
    if p.is_alive():
        p.terminate()
        print("Proceso finalizado por tiempo limite de ejecución, por favor vea los resultados en la consola")
        return False
 
    print("Se ha ejecutado correctamente")
    return True

if __name__ == '__main__':
    # Se reciben parametros
    name_file = loaddata.inputFile()
    seconds = loaddata.inputTime()
    execTimeLimit(execStartProcesss,(name_file,),seconds)
    



"""
x=[]
y=[]
for i in best_route:
    x.append(i.name)
    y.append(i.name)

x.append(best_route[0].name)
y.append(best_route[0].name)
plt.plot(x, y, '--o')
plt.xlabel('X')
plt.ylabel('Y')
ax=plt.gca()
plt.title('Final Route Layout')
bbox_props = dict(boxstyle="circle,pad=0.3", fc='C0', ec="black", lw=0.5)
for i in range(1,len(cityList)+1):
    ax.text(cityList[i-1].name, cityList[i-1].name, str(i), ha="center", va="center",size=8,bbox=bbox_props)
plt.tight_layout()
plt.show()
"""

