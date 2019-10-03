#!/usr/bin/env python
# coding: utf-8

# # Game Theory Final Project

# In[30]:


import networkx as nx
from operator import itemgetter
import random
import matplotlib.pyplot as plt
import itertools
#import pygraphviz as pgv
from itertools import combinations
import time
import copy
#from IPython.display import Image
#from mat4py import savemat
import csv



# In[31]:


# Global dictionaries required for our third algorithm, the combined-clustered approach.
# Note that the usage of globals was not ideal, but given the time constraint we opted for the easy solution
global rejectingNodeDict
global clusterDict


# In[32]:


rejectingNodeDict = {}
clusterDict = {}


# ### Useful links
# #### Overleaf Project
# - https://www.overleaf.com/16173247hqfsnqsvshqs#/61849223/
# 
# #### NetworkX Source Code
# - https://networkx.github.io/documentation/stable/_modules/networkx/classes/function.html#degree
# - https://networkx.github.io/documentation/stable/_modules/networkx/classes/function.html#set_node_attributes
# 
# #### NetworkX Public API
# - https://networkx.github.io/documentation/stable/reference/classes/generated/networkx.Graph.degree.html
# 
# #### SNAP
# - https://snap.stanford.edu/

# # SETUP

# ## Generate graphs

# In[33]:


def buildRandomGraph(size, probability):
    
    # Wrapper method. Build a random (Erdos) graph of given size and probability of edge connection.
    
    G = nx.gnp_random_graph(size, probability)
    return G

def buildCavemanGraph(clique, size):

    # Wrapper method to build a caveman graph

    G = nx.connected_caveman_graph(clique, size)
    return G

def buildRandomPartitionGraph(sizes, p_in, p_out):

    #create random partition graph

    G = nx.random_partition_graph(sizes, p_in, p_out)
    return G

# #### Graph from the paper

# In[34]:


G_paper = nx.Graph()
G_paper.add_edge(1,2)

G_paper.add_edge(1,3)
G_paper.add_edge(1,4)
G_paper.add_edge(1,5)

G_paper.add_edge(2,3)
G_paper.add_edge(2,4)
G_paper.add_edge(2,5)

G_paper.add_edge(6,7)

G_paper.add_edge(6,3)
G_paper.add_edge(6,4)
G_paper.add_edge(6,5)

G_paper.add_edge(7,3)
G_paper.add_edge(7,4)
G_paper.add_edge(7,5)

G_paper.node[1]['criticality'] = .4
G_paper.node[2]['criticality'] = .4
G_paper.node[3]['criticality'] = .8
G_paper.node[4]['criticality'] = .8
G_paper.node[5]['criticality'] = .8
G_paper.node[6]['criticality'] = .4
G_paper.node[7]['criticality'] = .4


#A = nx.nx_agraph.to_agraph(G_paper)
#A.layout(prog='dot')


# #### Another graph example; naive fails but clustered works.

# In[35]:


G_naiveFail = nx.Graph()

G_naiveFail.add_edge(1,2)
G_naiveFail.add_edge(2,3)
G_naiveFail.add_edge(3,4)
G_naiveFail.add_edge(4,1)

G_naiveFail.add_edge(4,5)
G_naiveFail.add_edge(5,7)

G_naiveFail.add_edge(6,7)
G_naiveFail.add_edge(6,8)
G_naiveFail.add_edge(6,9)
G_naiveFail.add_edge(6,10)

G_naiveFail.add_edge(8,11)
G_naiveFail.add_edge(8,12)

G_naiveFail.add_edge(9,13)
G_naiveFail.add_edge(9,14)

G_naiveFail.add_edge(10,15)
G_naiveFail.add_edge(10,16)

G_naiveFail.node[1]['criticality'] = .4
G_naiveFail.node[2]['criticality'] = .4
G_naiveFail.node[3]['criticality'] = .4
G_naiveFail.node[4]['criticality'] = .4
G_naiveFail.node[5]['criticality'] = .8
G_naiveFail.node[6]['criticality'] = .4
G_naiveFail.node[7]['criticality'] = .4
G_naiveFail.node[8]['criticality'] = .4
G_naiveFail.node[9]['criticality'] = .4
G_naiveFail.node[10]['criticality'] = .4
G_naiveFail.node[11]['criticality'] = .8
G_naiveFail.node[12]['criticality'] = .8
G_naiveFail.node[13]['criticality'] = .8
G_naiveFail.node[14]['criticality'] = .8
G_naiveFail.node[15]['criticality'] = .8
G_naiveFail.node[16]['criticality'] = .8

#A = nx.nx_agraph.to_agraph(G_naiveFail)
#A.layout(prog='dot')


# #### Another example. Naive fails, clustered fails, combined works

# In[36]:


G_third = nx.Graph()

G_third.add_edge(1,2)
G_third.add_edge(2,3)

G_third.add_edge(1,4)
G_third.add_edge(1,5)
G_third.add_edge(1,6)
G_third.add_edge(1,7)

G_third.add_edge(2,4)
G_third.add_edge(2,5)
G_third.add_edge(2,6)
G_third.add_edge(2,7)

G_third.add_edge(3,4)
G_third.add_edge(3,5)
G_third.add_edge(3,6)
G_third.add_edge(3,7)

G_third.add_edge(8,9)
G_third.add_edge(9,10)

G_third.add_edge(8,4)
G_third.add_edge(8,5)
G_third.add_edge(8,6)
G_third.add_edge(8,7)
G_third.add_edge(8,11)
G_third.add_edge(8,12)
G_third.add_edge(8,13)
G_third.add_edge(8,14)

G_third.add_edge(9,4)
G_third.add_edge(9,5)
G_third.add_edge(9,6)
G_third.add_edge(9,7)
G_third.add_edge(9,11)
G_third.add_edge(9,12)
G_third.add_edge(9,13)
G_third.add_edge(9,14)

G_third.add_edge(10,4)
G_third.add_edge(10,5)
G_third.add_edge(10,6)
G_third.add_edge(10,7)
G_third.add_edge(10,11)
G_third.add_edge(10,12)
G_third.add_edge(10,13)
G_third.add_edge(10,14)

G_third.add_edge(15,16)
G_third.add_edge(16,17)

G_third.add_edge(15,11)
G_third.add_edge(15,12)
G_third.add_edge(15,13)
G_third.add_edge(15,14)

G_third.add_edge(16,11)
G_third.add_edge(16,12)
G_third.add_edge(16,13)
G_third.add_edge(16,14)

G_third.add_edge(17,11)
G_third.add_edge(17,12)
G_third.add_edge(17,13)
G_third.add_edge(17,14)

G_third.node[1]['criticality'] = .4
G_third.node[2]['criticality'] = .4
G_third.node[3]['criticality'] = .4

G_third.node[4]['criticality'] = .8
G_third.node[5]['criticality'] = .8
G_third.node[6]['criticality'] = .8
G_third.node[7]['criticality'] = .8

G_third.node[8]['criticality'] = .4
G_third.node[9]['criticality'] = .4
G_third.node[10]['criticality'] = .4

G_third.node[11]['criticality'] = .8
G_third.node[12]['criticality'] = .8
G_third.node[13]['criticality'] = .8
G_third.node[14]['criticality'] = .8

G_third.node[15]['criticality'] = .4
G_third.node[16]['criticality'] = .4
G_third.node[17]['criticality'] = .4


G_DP = nx.Graph()

G_DP.add_edge(1,7,weight=6)
G_DP.add_edge(1,2,weight=8)
G_DP.add_edge(7,8,weight=10)
G_DP.add_edge(7,11,weight=2)
G_DP.add_edge(15,16,weight=5)
G_DP.add_edge(3,4, weight=4)
G_DP.add_edge(2,5, weight=3)
G_DP.add_edge(2,3, weight=7)
G_DP.add_edge(7,15, weight=4)
G_DP.add_edge(5,6, weight=7)
G_DP.add_edge(7,13, weight=5)
G_DP.add_edge(13,14, weight=8)
G_DP.add_edge(11,12, weight=3)
G_DP.add_edge(8,10, weight=4)
G_DP.add_edge(8,9, weight=6)
G_DP.node[1]['weight'] = 10
G_DP.node[2]['weight']=12
G_DP.node[3]['weight']=5
G_DP.node[4]['weight']=8
G_DP.node[5]['weight']=4
G_DP.node[6]['weight']=12
G_DP.node[7]['weight']=12
G_DP.node[8]['weight']=8
G_DP.node[9]['weight']=7
G_DP.node[10]['weight']=10
G_DP.node[11]['weight']=5
G_DP.node[12]['weight']=8
G_DP.node[13]['weight']=6
G_DP.node[14]['weight']=6
G_DP.node[15]['weight']=7
G_DP.node[16]['weight']=9

G_DP2 = nx.Graph()

G_DP2.add_edge(0,2,weight=7)
G_DP2.add_edge(0,1,weight=4)
G_DP2.add_edge(1,4,weight=5)
G_DP2.add_edge(1,3,weight=9)
G_DP2.add_edge(0,5,weight=7)

G_DP2.node[0]['weight'] = 15
G_DP2.node[1]['weight']=3
G_DP2.node[2]['weight']=7
G_DP2.node[3]['weight']=3
G_DP2.node[4]['weight']=14
G_DP2.node[5]['weight']=2


#A = nx.nx_agraph.to_agraph(G_paper)
#A.layout(prog='dot')


G_test_rand = nx.fast_gnp_random_graph(2000,0.4)
def setClusterGraphAttributes(G, n):
    for i in range(0,n):
        rand = random.randint(1,15)
        rand *= 10000
        G.node[i]['weight'] = rand
        for neighbor in nx.neighbors(G, i):
            rand2 = random.randint(0,10)*10000
            G.add_edge(i, neighbor, weight=rand2)

def makeMatrix(G, n):
    f = open("make_matrix.txt", "w+")
    matrix = [[0] * n for _ in range(n)] #store payoff
    weight = nx.get_node_attributes(G_test_DP, name='weight')
    print("weight of nodes is:", weight)
    edge = nx.get_edge_attributes(G_test_DP, 'weight')
    print("weight of edges is:", edge)
    for key, value in edge.items():
        matrix[key[0]][key[1]] = value
        matrix[key[1]][key[0]] = value
    for key, value in weight.items():
        matrix[key][key] = value
    for i in range(n):
        fullStr = ','.join([str(elem) for elem in matrix[i] ])
        print(fullStr)
        f.write("[" + fullStr + "]" + "\n")
    f.close()
    with open('make_matrix.csv', mode='w', newline='') as make_matrix:
        matrix_writer = csv.writer(make_matrix, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(n):
            matrix_writer.writerow(matrix[i])




setClusterGraphAttributes(G_test_rand, 2000)




# ### Set note attributes

# In[37]:


def setAllNodeAttributes(G):
    nodeList = G.nodes()
    for nodeID in nodeList:
        G.node[nodeID]["visited"] = False
        G.node[nodeID]['criticality'] = random.uniform(0.2, 1)
        G.node[nodeID]["cluster"] = -1

def setPartitionNodeAttributes(partition_list, G, criticality):
    nodeList = G.nodes()
    print(nodeList)
    nodeNum = 0
    lower = 0.2
    for nodeID in nodeList:
        G.node[nodeID]["visited"] = False
        G.node[nodeID]["cluster"] = -1

    for num in partition_list:
        print(num)
        print(nodeNum)
        for i in range(0, num):
            G.node[nodeNum]['criticality'] = random.uniform(lower, criticality)
            nodeNum += 1
        if criticality < 1:
            criticality += 0.1
        if lower < 0.9:
            lower += 0.1
        print("current ID is:", i)
        
# In[38]:


def setVisitedFalse(G):    
    nodeList = G.nodes()
    for nodeID in nodeList:
        G.node[nodeID]["visited"] = False


# In[39]:


def initClusters(G):
    nodeList = G.nodes()
    for nodeID in nodeList:
        G.node[nodeID]["cluster"] = -1


# In[40]:


def printNodeAttributes(G):
    nodeList = G.nodes()
    for nodeID in nodeList:
        print (nodeID, " : ", G.node[nodeID])


# # ALGORITHMS

# ### 1. Degree Naive Approach: 
# #### Choose K nodes with positive accepting degree

# In[41]:


def buildNaiveSeedSet(G, threshold, k):
    
    # Looks at every node in the graph - has no concept of "cluster" 
    # We consider a node for seedSet if its accepting degree is positive, where accepting degree
    # is the sum of the those neighbors who accept (i.e., criticality < threshold)
    #
    # The seed set is a list of tuples, identified by {nodeID, degreeAcceptingNeighbors}.
    # We build a list of tuples so that we can sort by degreeAcceptingNeighbors.
    
    nodeList = G.nodes()
    seedSet = []
    
    for nodeID in nodeList:
        if G.node[nodeID]['criticality'] > threshold:
            continue
        
        degreeAcceptingNeighbors = 1
        
        neighborList = nx.neighbors(G, nodeID)
        for neighborID in neighborList:
            if G.node[neighborID]['criticality'] < threshold:
                degreeAcceptingNeighbors += 1
            else:
                degreeAcceptingNeighbors -=1
        
        if degreeAcceptingNeighbors >= 0: #>= because you get 0 PLUS the payoff of yourself
            seedSet.append((nodeID, degreeAcceptingNeighbors))
    
    #Now get the naive seed set up to K
    kSet = getseedSetFromTupleList(seedSet, k)
    return kSet     


# ### 2. Clustered Naive Approach:
# #### Build all clusters of accepting nodes. Select 1 node from cluster that maximizes payoff and go down from there.

# In[42]:

def updateCriticality(G, source, appeal):
    numAccepting = 0
    neighborList = nx.neighbors(G, source)
    numNeighbors = len(neighborList)
    k = 0.02
    for neighbors in neighborList:
        if G.node[neighbors]['criticality'] < appeal:
            numAccepting -= k

        else:
            numAccepting += k
    #print("difference is", numAccepting)
    newCrit = G.node[source]['criticality']+ numAccepting
    if newCrit >= 1:
        newCrit = 1
    if newCrit <= 0:
        newCrit = 0
    G.node[source]['criticality'] = newCrit

def labelClusters(G, source, clusterNumber, appeal, thirdAlgorithm=False):
    
    # Perform BFS to label a cluster. Note how we called setVisitedFalse(G) upon finding an unvisited node.
    # The idea is that clusters are "contained" - every cluster is separated from every other cluster by walls of
    # rejecting nodes. You will never find another cluster, because if you could, it would simply be one cluster.
    # However, we need to set rejecting nodes to visited so that we don't count them for each accepting neighbor.
    # This, however, is not problematic, because we label all nodes from the cluster from a single canonical source node.
    # We will only reach this method from another cluster - whereupon we call setVisitedFalse(), clearing the rejecting
    # nodes, allowing us to again "contain" this new cluster.
    
    if G.node[source]['visited'] == False:
        setVisitedFalse(G)
        G.node[source]['visited'] = True
    else:
        return 0

    queue = []
    G.node[source]['cluster'] = clusterNumber
    queue.append(source)
    if thirdAlgorithm:
        if clusterNumber not in clusterDict:
            clusterDict[clusterNumber] = source

    acceptingInThisCluster = 1 #count yourself, you matter!
    
    while queue:
        start = queue.pop(0)

        for neighbor in nx.neighbors(G, start):
            if G.node[neighbor]['visited'] == False:
                if G.node[neighbor]['criticality'] < appeal:
                    queue.append(neighbor)
                    G.node[neighbor]['cluster'] = clusterNumber
                    G.node[neighbor]["visited"] = True
                    acceptingInThisCluster += 1
                    
                    if clusterNumber not in clusterDict:
                        clusterDict[clusterNumber] = neighbor
                    
                else:
                    G.node[neighbor]["visited"] = True
                    acceptingInThisCluster -= 1
                    if (thirdAlgorithm == True):
                        if neighbor not in rejectingNodeDict:
                            rejectingNodeDict[neighbor] = set()
                            rejectingNodeDict[neighbor].add(clusterNumber)
                        else:
                            rejectingNodeDict[neighbor].add(clusterNumber)
                    
    return acceptingInThisCluster


# In[43]:


def buildClusteredSet(G, threshold, k, thirdAlgorithm=False):

    # From each node in the nodeList, try to label its cluster. This will return 0 for many nodes, as they are labeled
    # from the (arbitrary) canonical node in its cluster.
    # We then select a seed set of up to (not always, depending on the composition of the graph) k source nodes.
    # We select the k source clusters with the highest accepting degree, implemented by sorting a list of tuples.
    
    initClusters(G)
    setVisitedFalse(G)
    nodeList = G.nodes()
    seedSet = []
    clusterCount = 0
    
    #Build the clusters
    for nodeID in nodeList:
        if (G.node[nodeID]['criticality'] < threshold) and (G.node[nodeID]['cluster'] == -1):
            summedNeighbors = labelClusters(G, nodeID, clusterCount, threshold, thirdAlgorithm)
            if summedNeighbors > 0:
                seedSet.append( (nodeID, summedNeighbors))
            clusterCount += 1
    #Choose up-to-k    
    kSet = getseedSetFromTupleList(seedSet, k)
    return kSet     

def returnEntireSeedSet(G, threshold, k, thirdAlgorithm=False):

    # Method that returns all of the clusters in the Seed Set, rather than just the k with the highest cardinality
    initClusters(G)
    setVisitedFalse(G)
    nodeList = G.nodes()
    #print("Node list ", nodeList)
    seedSet = []
    clusterCount = 0
    
    #Build the clusters
    for nodeID in nodeList:
        if (G.node[nodeID]['criticality'] < threshold) and (G.node[nodeID]['cluster'] == -1):
            summedNeighbors = labelClusters(G, nodeID, clusterCount, threshold, thirdAlgorithm)
            #if summedNeighbors > 0:
            seedSet.append( (nodeID, summedNeighbors))
            clusterCount += 1
    seedSet.sort(key=itemgetter(1))
    print(seedSet)
    return seedSet
# #### Misc helper method for algorithms (1) and (2)

# In[44]:


def greedy1(G, seedSet, phi, k):
    #Method that uses Dynamic Programming to find the optimal k nodes, using a greedy approach
    if k > len(seedSet):
        k = len(seedSet)
    payoff = -2
    kSet = []
    kSetTup = []
    # Consider up to k nodes as seeds, using a pseudo-greedy approach
    while k > 0: #number of seeds
        for node in seedSet: #test seed from each cluster
            kSet.append(node[0])
            payoff2 = computePayoff(G, kSet, phi)
            if payoff2 <= payoff: 
                kSet.pop()
            else:
                payoff=payoff2
                if len(kSetTup) > 0:
                    kSetTup.pop()
                kSetTup.append(node)
                kSet.pop()
        if len(kSetTup) > 0:
            kSet.append(kSetTup[-1][0])
            seedSet.remove(kSetTup[-1])
            kSetTup.pop()
        k -= 1
    if payoff > 0:
        return kSet
    else:
        return []

def greedy2(G, seedSet, phi, k):

        #Method that uses Dynamic Programming to find the optimal k nodes, using a greedy approach
        if k > len(seedSet):
            k = len(seedSet)
        kSet = []
        kSetTup = []
        start = 0
        payoff = -5
        i=0
        rows, cols = (k+1,1000)
        arr = [[0 for i in range(cols+1)] for j in range(rows+1)]
        # Consider up to k nodes as seeds, using a pseudo-greedy approach
        for seed in range(1, k+1): #number of seeds
            for node in seedSet: #test seed from each cluster
                kSet.append(node[0])
                payoff2 = start + computePayoff(G, [node[0]], phi)
                start = payoff2
                print(payoff2)
                if payoff2 <= payoff: 
                    #print("last payoff was better")
                    arr[seed][i+1]=arr[seed][i] #keep last payoff
                    i+=1
                    kSet.pop()
                else:
                    print("new payoff is better")
                    arr[seed][i+1]=payoff2
                    payoff=payoff2
                    if len(kSetTup) > 0:
                        kSetTup.pop()
                    kSetTup.append(node)
                    i+=1
                    kSet.pop()
            if len(kSetTup) > 0:
                kSet.append(kSetTup[-1][0])
            print("kSet after this round is:", kSet)
            if len(kSetTup) > 0:
                seedSet.remove(kSetTup[-1])
                kSetTup.pop()
        if payoff > 0:
            return kSet
        else:
            return []

def getseedSetFromTupleList(seedSet, k):
    
    # Given a seed set (a tuple list), choose up to K source nodes, if the source seed set is larger than K.
    # Return a new list, containing those k source nodes with the highest degreeAccepting.
    
    kSet = []
    if len(seedSet) > 0:
        numberOfClusters = k if k < len(seedSet) else len(seedSet)
        seedSet.sort(key=itemgetter(1))
        setLength = len(seedSet)-1
        for i in range(numberOfClusters):
            kSet.append(seedSet[setLength-i][0])
            
    return kSet


# ### 3. Shared-Rejecting Nodes

# #1. Build clusters
# #2. Build dictionary of rejecting nodes and the clusters that touch them.
# #3. Call any non-unique entries in dictionary <br>
#      i.e., if <br>
#               3-> 0, 1 <br>
#               4-> 0, 1 <br>
#               5-> 0, 1 <br>
#     then only search one (e.g., remove 4, 5; keep only 3)
# #4. Compute payoff on each unique permutation. If it increases by exposing the permutation, expose it.

# In[45]:


def cullNonDistinctRejectingNodes():
    
    # Delete any non-unique key, values from the rejectingNode dictionary.
    
    for key1, key2 in combinations(rejectingNodeDict.keys(), r = 2):
        if key1 in rejectingNodeDict and key2 in rejectingNodeDict:
            if rejectingNodeDict[key1] == rejectingNodeDict[key2]:
                del rejectingNodeDict[key2]


# In[46]:


def findSubsets(S,m):
    
    # S: The set for which you want to find subsets
    # m: The number of elements in the subset
    # Credit: 
    
    return set(itertools.combinations(S, m))


# In[77]:


ARBITRARY_NEG_NUMBER = -1000

def payoffOfUniqueSubsets(G, appeal):
    
    # This approach considers all clusters connected to rejecting nodes, running off the intuition that maybe
    # a cluster alone can't overpower the negative boundary nodes its connected to but clusters together could.
    # This approach is inspired by the counterexample given in the paper.
    # For each rejecting node with a unique set of clusters touching it, build the subsets of that set
    # testing how payoff improves.
    # The highest performing subset in exploring all subsets in the rejecting node dictionary is then used
    # as a seed set.
    
    bestSubsetPayoff = ARBITRARY_NEG_NUMBER
    bestSubset = []
    print(rejectingNodeDict)
    print(clusterDict)
    for rejectingNode, clusteredSet in rejectingNodeDict.items():
       
        for i in range(1, len(clusteredSet)+1):
            #Generate all subsets of size i (e.g., size 2)
            allSubsets = findSubsets(clusteredSet, i)
            #Keep track of the best performing seedset.
            bestMsubsetPayoff = ARBITRARY_NEG_NUMBER 
            bestMsubset = []
            #Search all m-sized subsets, e.g., (0,1), (1,2), (0,2), building a seed set from each.
            #Compute the payoff from each of these sets. We want the best performing overall.
            for mSubSet in allSubsets:
                seedSet = []
                for element in mSubSet:
                    seedSet.append(clusterDict[element])
                    payoffThisRun = computePayoff(G, seedSet, appeal)
                if payoffThisRun > bestMsubsetPayoff:
                    bestMsubsetPayoff = payoffThisRun
                    bestMsubset = seedSet
            if bestMsubsetPayoff > bestSubsetPayoff:
                bestSubsetPayoff = bestMsubsetPayoff
                bestSubset = bestMsubset
    
    if bestSubsetPayoff <= 0:
        return []
    return bestSubset


# ## Compute Payoff on Seed Set

# In[48]:


def computePayoff(G, seedSet, phi):
    
    # Given a seed set of source nodes, compute the payoff of exposing each node in that seed set.
    # We perform BFS from each node, where an accepting node returns a payoff of +1 and rejecting node a payoff
    # of -1. 
    
    if len(seedSet) is 0:
        print ("Cannot compute payoff. Seed set is of size 0. Exiting...")
        return
    
    setVisitedFalse(G)
    payoff = 0
    for nodeID in seedSet:
        temp = bfs(G, nodeID, phi)
        payoff+=temp
        
    return payoff


# In[49]:


def bfs(G, source, phi):
    
    # From a source node, perform BFS based on criticality.
    
    payoff = 0
    
    if G.node[source]['visited'] == False:
        G.node[source]['visited'] = True
        payoff = 1 #we already know source is accepting
    else:
        return payoff
    
    queue = []
    queue.append(source)
    
    while queue:
        start = queue.pop(0)
        
        for neighbor in nx.neighbors(G, start):
            if G.node[neighbor]['visited'] == False:
                if G.node[neighbor]['criticality'] < phi:
                    queue.append(neighbor)
                    payoff+=1
                else:
                    payoff-=1
                G.node[neighbor]['visited'] = True
    
    return payoff


# ## Graph Drawing

# In[50]:

def computeNegPayoff(G, nodeNum):
    print("node is:" , nodeNum)
    nodeWeight = G.node[nodeNum]['weight']
    negPayoff = nx.neighbors(G, nodeNum)
    for negNode in negPayoff:
        add = G.get_edge_data(nodeNum, negNode)
        add = add['weight']
        nodeWeight -= add
    print("node weight is:", nodeWeight)
    return nodeWeight


def tryDP(G, i, k):
    #This is different since we are considering each node's weight in the graph to be the number of accepting nodes in a given cluster
    #i = number_of_nodes(G)
    #k = number of seeds
    storePayoff = [[0] * i for _ in range(k)] #store payoff
    storeSeeds = [[[]] * i for _ in range(k)] #store seeds at each stage
    tree = nx.bfs_tree(G, 0)
    #print(nodes)
    for numSeeds in range(0,k): #bottom up DP
        nodes = list(reversed(list((nx.topological_sort(tree))))) #look at nodes in reverse topological order
        for node, j in zip(nodes, range(0,i)): 
            if j == 0 and numSeeds == 0: #first entry
                #breakpoint()
                storeSeeds[numSeeds][j] = [node]
                nodeWeight = computeNegPayoff(G, node)
                print(nodeWeight)
                storePayoff[numSeeds][j] = nodeWeight
                #print("first entry,", storePayoff)

            elif numSeeds == 0: #if there is only one seed to consider, aka first row
                last = storePayoff[numSeeds][j-1]
                nodeWeight = computeNegPayoff(G, node)
                if nodeWeight >= last:
                    storePayoff[numSeeds][j]=nodeWeight
                    storeSeeds[numSeeds][j] = [node]
                else:
                    storePayoff[numSeeds][j]= last
                    table = storeSeeds[numSeeds][j-1]
                    table2 = table[:]
                    storeSeeds[numSeeds][j] = table2
                #print("num seeds 0",storePayoff)
            elif j == 0: #we only consider first node, so its simple
                storePayoff[numSeeds][j] = storePayoff[numSeeds - 1][j]
                storeSeeds[numSeeds][j] = storeSeeds[numSeeds - 1][j][:]
            else: #where DP comes in
                last = storePayoff[numSeeds-1][j-1] #diagonal-up entry
                nextGuess = computeNegPayoff(G, node) + last
                for lastNodes in storeSeeds[numSeeds-1][j-1]: #dont want to double count edges!
                    neighbors = nx.neighbors(G, lastNodes)
                    for neighbor in neighbors:
                        if neighbor == node:
                            add = G.get_edge_data(lastNodes, node) #neighbor of new node is current node
                            add = add['weight']
                            nextGuess += add
                lastEntry = storePayoff[numSeeds][j-1] #left entry
                lastEntryUp = storePayoff[numSeeds-1][j]
                storePayoff[numSeeds][j] = max(lastEntry, lastEntryUp, nextGuess, last)
                if storePayoff[numSeeds][j] == last:
                    nextList = storeSeeds[numSeeds-1][j-1][:]
                    storeSeeds[numSeeds][j] = nextList
                elif storePayoff[numSeeds][j] == lastEntry+1:
                    nextList = storeSeeds[numSeeds][j-1][:]
                    storeSeeds[numSeeds][j] = nextList
                    storePayoff[numSeeds][j] -= 1
                elif storePayoff[numSeeds-1][j] == lastEntryUp+1:
                    nextList = storeSeeds[numSeeds-1][j][:]
                    storeSeeds[numSeeds][j] = nextList
                    storePayoff[numSeeds][j] -= 1
                else:
                    #print("new is better")
                    table = storeSeeds[numSeeds-1][j-1][:]
                    table.append(node)
                    storeSeeds[numSeeds][j] = table
    print(storePayoff)
    print(storeSeeds)
    maxVal = storePayoff[k-1][i-1]
    for j in range(0,k):
        if storePayoff[j][i-1] > maxVal:
            maxVal = storePayoff[j][i-1]
    return (maxVal, storeSeeds[j][i-1])

def tryDP2(G, i, k):
    #This is different since we are considering each node's weight in the graph to be the number of accepting nodes in a given cluster
    #i = number_of_nodes(G)
    #k = number of seeds
    storePayoff = [[0] * i for _ in range(k)] #store payoff
    storeSeeds = [[[]] * i for _ in range(k)] #store seeds at each stage

    for numSeeds in range(0,k): #bottom up DP
        for nodeNum in range(i):
            if nodeNum == 0 and numSeeds == 0: #first entry
                #breakpoint()
                #print(nodeNum)
                storeSeeds[numSeeds][nodeNum] = [nodeNum]
                nodeWeight = computeNegPayoff(G, nodeNum)
                storePayoff[numSeeds][nodeNum] = nodeWeight
                #print(storePayoff)

            elif numSeeds == 0: #if there is only one seed to consider, aka first row
                #breakpoint()
                last = storePayoff[numSeeds][nodeNum-1]
                nodeWeight = computeNegPayoff(G, nodeNum)
                #print("neg payoff is,", negPayoff)
                if nodeWeight > last:
                    #print("current is better")
                    storePayoff[numSeeds][nodeNum]=nodeWeight
                    storeSeeds[numSeeds][nodeNum] = [nodeNum]
                else:
                    storePayoff[numSeeds][nodeNum]= last
                    table = storeSeeds[numSeeds][nodeNum-1]
                    table2 = table[:]
                    storeSeeds[numSeeds][nodeNum] = table2
                #print(storePayoff)
            elif nodeNum == 0: #we only consider first node, so its simple
                #breakpoint()
                storePayoff[numSeeds][nodeNum] = storePayoff[numSeeds - 1][nodeNum]
                storeSeeds[numSeeds][nodeNum] = storeSeeds[numSeeds - 1][nodeNum][:]
                #print(storePayoff)
                #print(storeSeeds)
            else: #where DP comes in
                #breakpoint()
                last = storePayoff[numSeeds-1][nodeNum-1] #diagonal-up entry
                nextGuess = computeNegPayoff(G, nodeNum) + last
                for lastNodes in storeSeeds[numSeeds-1][nodeNum-1]: #dont want to double count edges!
                    neighbors = nx.neighbors(G, lastNodes)
                    print(neighbors)
                    for nodes in neighbors:
                        if nodes == nodeNum:
                            add = G.get_edge_data(nodeNum, lastNodes)
                            add = add['weight']
                            nextGuess += add
                lastEntry = storePayoff[numSeeds][nodeNum-1]
                storePayoff[numSeeds][nodeNum] = max(lastEntry, nextGuess, last)
                if storePayoff[numSeeds][nodeNum] == last:
                    nextList = storeSeeds[numSeeds-1][nodeNum-1][:]
                    storeSeeds[numSeeds][nodeNum] = nextList
                elif storePayoff[numSeeds][nodeNum] == lastEntry:
                    nextList = storeSeeds[numSeeds][nodeNum-1][:]
                    storeSeeds[numSeeds][nodeNum] = nextList
                else:
                    #print("new is better")
                    table = storeSeeds[numSeeds-1][nodeNum-1][:]
                    table.append(nodeNum)
                    storeSeeds[numSeeds][nodeNum] = table
    return (storePayoff[k-1][i-1], storeSeeds[k-1][i-1])
    """
    #base case
    if i==1 or k == 1:
        return 0
    print(storeSeeds[k-1][i-1])

    if storeSeeds[k-1][i-1] != 0:
        return storeSeeds[k-1][i-1]
    
    for nodes in alreadyPicked:
        for node in G.neighbors(nodes):
            if node == currentNode:
                add = G_DP.get_edge_data(node, currentNode)
                add = add['weight']
                withSeed += add
    neighbor = nx.neighbors(G, i)
    print(neighbor)
    for nodes in neighbor:
        seed = G.get_edge_data(nodes, i)
        withSeed -= seed['weight']
    withSeed += currentNode['weight'] + tryDP(G, storeSeeds, alreadyPicked, i-1, k-1)
    without = tryDP(G, storeSeeds, alreadyPicked, i-1, k)
    if withSeed > without:
        alreadyPicked.append(i)
        print('aleady picked are: ', alreadyPicked)
    storeSeeds[k-1][i-1]= max(withSeed, without)
    return storeSeeds[k-1][i-1]
    """



def runDPWithTime(k):
    for i in range(10,300, 50):
        G = nx.random_tree(i)
        nx.draw_networkx(G,pos=None, arrows=False, with_labels=True)
        plt.show()
        n = G.number_of_nodes()
        start_time = time.time()
        setClusterGraphAttributes(G, n)
        weight = nx.get_node_attributes(G, name='weight')
        edge = nx.get_edge_attributes(G, 'weight')
        payoff = tryDP(G, n, k)
        end_time = time.time()
        matrix = makeMatrix(G, n)
        print(matrix)
        print("Payoff: ", payoff, "\n time taken:", end_time - start_time, "num nodes: ", i, "\n", weight, "\n", edge)

#runDPWithTime(4)

start_time = time.time()
#l = topological_sort(G_test_DP)
#print(l)
G_test_DP = nx.random_tree(200)
numNode = G_test_DP.number_of_nodes()
weight1 = nx.get_node_attributes(G_DP2, name='weight')
weight2 = nx.get_edge_attributes(G_DP2, name='weight')
setClusterGraphAttributes(G_test_DP, numNode)
makeMatrix(G_test_DP, numNode)

print(weight1,weight2)
#print(list(nx.bfs_edges(tree, 0)))

#setClusterGraphAttributes(tree, numNode)
test1 = tryDP(G_test_DP, numNode, 30)
end_time = time.time()

print("payoff test DP is: ", test1, end_time - start_time)

nx.draw_networkx(G_test_DP,pos=None, arrows=False, with_labels=True)
plt.show()
"""
#i = G_test_DP.number_of_nodes()
test2 = runDPWithTime(G_test_rand, k)
print("payoff is:", test2)
test3 = runDPWithTime(G_DP2, k)
print("payoff: ", test3)

"""

def drawGraph(G,appeal_threshold):

    colorList = []

    nodeList = G.nodes()
    for nodeID in nodeList:
        if G.node[nodeID]['criticality'] < appeal_threshold:
            colorList.append('g')
        else: 
            colorList.append('r')
    
    nx.draw_networkx(G, pos=None, arrows=False, with_labels=True, node_color=colorList)
    plt.show()


# # EXPERIMENTS

# In[60]:


def runNaiveExperiment(G, appeal, k): 
    # Naive
    setVisitedFalse(G)
    seedSetNaive = buildNaiveSeedSet(G, appeal, k)
    payoffNaive = computePayoff(G, seedSetNaive, appeal)
    print ("NAIVE seed set produced payoff of: ", payoffNaive, ". Seedset: ", seedSetNaive)


# In[61]:


def runClusteredExperiment(G, appeal, k):
     # Now clustered
    setVisitedFalse(G)
    seedSetClustered = buildClusteredSet(G, appeal, k, False)
    print("clustered seed set is", seedSetClustered)
    payoffClustered = computePayoff(G, seedSetClustered, appeal)
    print ("CLUSTERED seed set produced payoff of: ", payoffClustered, ". Seedset: ", seedSetClustered)


# In[62]:


def runCombinedExperiment(G, appeal, k):
    # Rejecting Nodes
    clearVisitedNodesAndDictionaries(G)
    buildClusteredSet(G, appeal, k, thirdAlgorithm=True)
    cullNonDistinctRejectingNodes()
    combinedSeedSet = payoffOfUniqueSubsets(G, appeal)
    print("combined seed set is:", combinedSeedSet)
    combinedPayoff = computePayoff(G, combinedSeedSet, appeal)
    print ("COMBINED seed set produced payoff of: ", combinedPayoff, ". Seedset: ", combinedSeedSet)


# In[63]:


def clearVisitedNodesAndDictionaries(G):
    setVisitedFalse(G)
    rejectingNodeDict.clear()
    clusterDict.clear()

def runTryDP(G, appeal, k):
    clearVisitedNodesAndDictionaries(G)
    seedSetClustered = returnEntireSeedSet(G, appeal, k, thirdAlgorithm=False)
    print("DP Seed set: ", seedSetClustered)
    seedSetDP = greedy1(G, seedSetClustered, appeal, k)
    payoffDP = computePayoff(G, seedSetDP, appeal)
    print ("DP seed set produced payoff of: ", payoffDP, ". Seedset: ", seedSetDP)
# ## Driver

# In[64]:

def driver(G, appeal, k):
    #runNaiveExperiment(G, appeal, k)
    runClusteredExperiment(G, appeal, k)
    runCombinedExperiment(G, appeal, k)
    runTryDP(G, appeal, k)
    drawGraph(G, appeal)

# In[65]:


def runDriver():
    threshold = 0.5
    k = 4
    for i in range(10, 100, 15):
        print("NEXT TEST!!!!!")
        G_random = buildRandomGraph(i,.2)
        setAllNodeAttributes(G_random)
        driver(G_random, threshold, k)

#runDriver()

# In[78]:


threshold = .7
k = 4
#driver(G_naiveFail, threshold, k)
#driver(G_paper, threshold, k)
#driver(G_third, threshold, k)


def updateGraph(G, appeal):
    print("updating graph G")
    nodeList = G.nodes()
    for node in nodeList:
        updateCriticality(G, node, appeal)
    driver(G, appeal, 3)

#updateGraph(G_naiveFail, threshold)
#updateGraph(G_paper, threshold)
#updateGraph(G_third, threshold)
#updateGraph(G_third, threshold)

#test caveman graph with k-cliques
def runCaveman():
    j = 5
    for i in range(10, 30, 10):
        G_caveman = buildCavemanGraph(i, j)
        setAllNodeAttributes(G_caveman)
        driver(G_caveman, threshold, k)
        #updateGraph(G_caveman, threshold)
        j+=5

#runCaveman()

#test random partition graph

partition_list = [10,20,40,30,30, 5, 10]
G_random_partition = buildRandomPartitionGraph(partition_list, 0.15, 0.005)
setPartitionNodeAttributes(partition_list, G_random_partition, 0.5)
driver(G_random_partition, threshold, k)
#updateGraph(G_random_partition, threshold)
#test Davis Southern Women graph

#G_david = nx.davis_southern_women_graph()
#setAllNodeAttributes(G_david)
#driver(G_david, threshold, k)
# In[ ]:





