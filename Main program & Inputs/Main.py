import numpy as np
from matplotlib import pyplot as plt
import math
import networkx as nx

datafile = open('SD-1-IRIS.txt',"r")
file = np.loadtxt("SD-1-IRIS.txt",skiprows = 3,dtype = float)

# Task 1
data1 = file
lines = datafile.readlines()
# row = 14
# column = len(lines)
row = int(lines[0])
column = int(lines[1])



r=np.zeros((row,row))

def printmap(o,string,title):
    plt.imshow(o,cmap= string, interpolation = 'nearest')
    plt.title(title)
    plt.show()

#Function to find Correlation matrix 
def correlation(data):
    for i in range(row):
        for j in range(row):
            sumx=0
            sumy=0
            xy = 0
            sx = 0
            sy = 0
            for k in range(column):
                sumx += data[i][k]
                sumy += data[j][k]
                xy += data[i][k]*data[j][k]
                sx += data[i][k]**2
                sy += data[j][k]**2
            r[i][j] = (column*xy - sumx*sumy)/(math.sqrt((column*sx)-(sumx**2))*math.sqrt((column*sy)-(sumy**2)))
    return r

def visualize():
    data = file.reshape(row,column)
    r  = correlation(data)
    meanarr = r.mean(axis = 0)
    d = np.zeros((row,row))
    for i  in range(row):
        for j in range(row):
            if meanarr[i] < r[i][j]:
                d[i][j] = int(1)
            else:
                d[i][j] = int(0)
    for i  in range(row):
        for j in range(row):
            d[j][i] = d[i][j]
    printmap(d,"Greys","Discretized Matrix")


def colorcoded():
    
    max_value = np.amax(r,axis=0)
    for i in range(row):
        for j in range(row):
            r[i][j] = (255-(r[i][j]/max_value[j]))*255
    
    printmap(r,"Greens","Color Coded")

visualize()
colorcoded()


# Task 2

d=file.reshape(150,4)

sdata = np.array(d)
per = np.zeros((row,row))
np.random.shuffle(sdata)
r1=correlation(sdata)
printmap(r1,"Greys","After Permutation")

# Function to calculate Signatures
def signaturecal(sdata):
    sumall = sdata.sum(axis=1) #Finding Sum 
    meanall = sdata.mean(axis=1) #Finding Mean
    signature = []

    for i in range(len(sumall)):
        signature.append(sumall[i]*meanall[i]) 

    for i in range(row):
        for j in range(1+i,row):
            if(signature[i] > signature[j]):
                temp = np.array(signature[i])
                temp1 = np.array(sdata[i])
                signature[i] = signature[j]
                sdata[i] = sdata[j]
                signature[j] = temp
                sdata[j] = temp1

    sdata1 = correlation(sdata) #Calculating Correlation Matrix
    printmap(sdata1,"Greens","Correlation Matrix after Signature Generation and Arrangement") #Visualization

signaturecal(sdata)


# Task 3

clustersList = []
newadjacency = r1.copy()

minNode = float(input("Enter value for threshhold value:  ")) #Threshold Input

newadjacency[newadjacency < minNode] = 0 

for i in range(newadjacency.shape[0]):
    newadjacency[i,i] = 0


#Function For Clustering 
def getClusters():

    nodeweights = newadjacency.sum(axis=1) 
    indexMaxNode = np.argmax(nodeweights) 

    if np.max(nodeweights) == 0:
        return
    
    clusterCopy = newadjacency[indexMaxNode]
    newCluster = [i for i, nw in enumerate(clusterCopy) if nw != 0]
    clustersList.append(newCluster)

    print(newCluster)
    graph= nx.Graph()
    print("\n\n")

    for i in newCluster: 
        newadjacency[:, i] = 0
        newadjacency[i, :] = 0
        graph.add_edge(indexMaxNode, i)

    print(len(clustersList))

    nx.draw(graph)
    plt.show()  #Visualization of Clusters
    plt.clf() #For clearing the figure


    getClusters()


getClusters()














