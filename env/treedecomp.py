"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.optimize import linprog
from numpy.linalg import solve
import networkx as nx
from networkx.algorithms import approximation as approx
from operator import itemgetter
import random
import matplotlib.pyplot as plt
import itertools
#import pygraphviz as pgv
from itertools import combinations
import time
import copy
import codecs
import csv
import math

"""
Idea: We want to implement a flow-augmenting algorithm that pairs drivers to passengers in a way that minimizes the total
travel time for the drivers.
The input given is a text file with the (x,y) coordinates of every driver and passenger. The program does the following:
    
    1) Computes the manhatten distance between each driver and every customer
    
    2) Creates a bipartite graph (every driver/passenger is a node) with drivers as one disjoint set and passengers as another disjoint set. It
    is a complete (directed) graph, so every driver is connected to every passenger
    
    3) When initializing the graph, set the weight of every edge as the distance between each driver and every passenger. This distance is the flow on the edge. So if there are n drivers 
    and m passengers, each driver has m outgoing edges (one to every passenger). Also, we set the capacity of every edge to 1, so if we choose to pick an
    edge from one driver to a passenger, the capacity of that edge becomes 0 (indicating that we have to send the full flow along the edge).
    
    4) Compute the minimal weighted bipartite matching, ie, choose one outgoing edge from every driver to one passenger such that no two drivers are matched to
    the same passenger and the total weight of the edges picked is minimized (meaning the combined distance travelled of all the drivers is minimized)
"""

class Lyft:
    
    def __init__(self,row,col,n,m, G,storeWeight):
       self.row = row
       self.col = col
       self.n = n
       self.m = m
       self.G = G
       self.storeWeight = storeWeight
        
    """
    method which finds the minimum distance, by iteratively calling BFS.
    With each round of BFS we check to see if we have found n pathways which minimize distance
    If not, adjust the weights and call BFS again
    this function is void
    """
    def findMin(self, s, t):
        numPaths = 0 
        Lyft.subtractInitial(self, s, t, numPaths) #setup initial augmentation
        while numPaths < self.n: #while we have not found a minimum-distance configuration
            for j in range(0,t+1): #reset visited for BFS for each iteration
                self.G.node[j]['visited'] = False 
            results = Lyft.bfs(self,s, t) #return number 
            numPaths = results[0]
            if numPaths == self.n:
                sumWeight = 0
                for key, val in results[1].items():
                    sumWeight += self.storeWeight[key-1][val-(self.n+1)]
                    print("sum so far", sumWeight, key, val)
        print("We found", self.n , "paths, between the following nodes", results[1], "with total distance", sumWeight)
    
    """
    Here, we want to take the smallest edge value out of every driver node and subtract it from every edge out of that node
    We then do the same for all of the edges into every passenger node
    We then call bfs on the resulting graph, and look at how many paths we can make
    Note this function is void
    """
    def subtractInitial(self, s, t, numPaths):
        #find min value for each driver
        for i in range(1, self.n+1):
            minVal = math.inf
            neighbors = nx.neighbors(self.G, i)
            minVal = min(self.G[i][j]['weight'] for j in neighbors)
            for j in neighbors:
                self.G[i][j]['weight'] -= minVal
                #print("weight of edge", i, j, "is", G[i][j]['weight'])
        #find min value for each passenger
        for i in range(self.n+1, t):
            minVal = math.inf
            minVal = min(self.G[j][i]['weight'] for j in range(1, self.n+1))
            for j in range(1,self.n+1):
                self.G[j][i]['weight'] -= minVal
                #print("weight of current passenger is", j , i , G[j][i]['weight'])
    
    """
    Call BFS to find paths with the following conditions:
        1) capacity between two nodes must be 1 (we can take that edge)
        2) Weight must be 0 (since we subtracted the min value from each outgoing edge, we know
        there is at least one edge of weight zero outgoing from every driver)
        3) Must not have been visited yet
    
    BFS will find the total number of paths satisfying these conditions. IF there are 
    less paths found than the number of drivers, we augment the flow and call BFS again
    """        
    def bfs(self, s, t):
        queue =[] #initialize queue
        numPaths = 0
        storeParents = {} #dictionary for u-v paths
        queue.append(s)
        while queue:
            #grab next node from the queue
            start = queue.pop(0)
            for neighbor in nx.neighbors(self.G, start):
                # if we haven't visited both ends of the edge
                if self.G.node[neighbor]['visited'] == False and self.G.node[start]['visited'] == False:
                    #if weight is 0 and the capacity is still 1
                    if self.G[start][neighbor]['weight'] == 0 and self.G[start][neighbor]['capacity'] == 1:
                       # print("we can add flow to node", neighbor, "from", start)
                        #set visited to true
                        self.G.node[neighbor]['visited'] = True
                        #check if we've already included this edge. If we havent, set capacity to 0
                        if start not in storeParents.keys():
                            self.G[start][neighbor]['capacity'] = 0
                            storeParents[start] = neighbor #add to path dictionary
                            numPaths += 1 #increment paths found
                        elif start not in self.row:
                            self.row.append(start) #we have found a zero path but we cant take it
                            #print("ROW weight of", start, neighbor, "is 0")    
                    queue.append(neighbor)
                elif self.G[start][neighbor]['weight'] == 0: #we found a zero path but cant take it
                   # print("COL weight of", start, neighbor, "is 0")
                    if start not in list(storeParents.keys()):
                        self.col.append(neighbor)
                        #print("COL weight of", start, neighbor, "is 0")
                
        #in this section, we augment the flow based on the number of paths found
        for node in storeParents.keys():
            if node not in self.row:
                if storeParents[node] not in self.col:
                    self.row.append(node)
       # print(storeParents,row, col)
        
        minVal = math.inf
        for i in range(1,self.n+1):
            if i not in self.row:
                for j in range(self.n+1, t):
                    if j not in self.col:
                        minVal = min(minVal, self.G[i][j]['weight'])
        #subtract from each uncovered row
        for i in range(1,self.n+1):
            if i not in self.row:
                for j in range(self.n+1, t):
                    self.G[i][j]['weight'] -= minVal
                    #print("SUBTRACT ROW val of ", i, j, "is", G[i][j]['weight'])
        for i in self.col:
            for j in range(1,self.n+1):
                self.G[j][i]['weight'] += minVal
               # print("val of ", j, i, "is", G[j][i]['weight'])
    
        for i, j in storeParents.items():
            self.G[i][j]['capacity'] = 1
        return numPaths, storeParents
    
    """
    next step is to augment the current path if there is not flow... to do this,
    we grab the smallest 
    """


def createGraph():
        print("working?")
        file_object = open(r"lyft.txt", "r")
        database = file_object.readlines()
        #second line of file is number of drivers
        n = int(database[1].strip('\n'))
        #third line of file is number of customers
        m = int(database[2].strip('\n'))
        row = []
        col = []
        #create a 2d array to store weights
        storeWeight = [[0 for i in range(n)] for j in range(m)]
        #create graph
        G = nx.DiGraph()
        for i in range(0,n+m+2):
            G.add_node(i, visited=False)
        for i in range(1,n+1):
            G.add_edge(0,i, weight=1, capacity=1, res_cap=0)
        for i in range(n+1, m+n+1):
            G.add_edge(i, m+n+1, weight=1, capacity=1, res_cap=0)
        #lines 3 to 3+n are the coordinates of the drivers, and lines 3+n to the end are the coordinates of the customers
        #we are creating the distance array such that there exists an edge between every driver/passenger combination
        #we create extra variables for the source terminal edges
        i = 1
        for driver_location in database[3:3+n]:
            j = n+1
            numbers_n = driver_location.split(' ') #extract location coordinates
            d1 = int(numbers_n[0])
            d2 = int(numbers_n[1])
            for customer_location in database[3+n:]:
                numbers_m = customer_location.split(' ')
                c1 = int(numbers_m[0])
                c2 = int(numbers_m[1])
                weight = abs(d1-c1)+ abs(d2-c2) #compute manhatten distance
                G.add_edge(i, j, weight=weight, capacity=1, res_cap=0) #add edge of appropriate weight and capacity 1
                storeWeight[i-1][j-(m+1)] = weight #store weight in array
                #print(i, j, " weight is ", abs(d1-c1)+ abs(d2-c2))
                j+= 1
            i += 1
        #print(storeWeight)
        nx.draw_networkx(G,pos=None, arrows=False, with_labels=True)
        plt.show()    
        print("working")
        lyft = Lyft(row, col, n, m, G, storeWeight)
        print("worked")
        lyft.findMin(0, n+m+1)

createGraph()



