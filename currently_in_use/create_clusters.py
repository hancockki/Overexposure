"""
This script contains all of the code pertaining to creating the graph data
structure we are running optimization algorithms on. The method
testOriginaltoCluster is the driver for converting a tree graph structure to 
a cluster graph. This method creates a randomly generated tree and then calls
buildClusteredSet to identify the clusters of accepting nodes. A cluster of accepting
nodes is all of the acceptable nodes reachable from a given accepting node. In this way,
a cluster is surrounded by a 'wall' of rejecting nodes. Each cluster becomes a node in the 
cluster graph, and each edge has weight equal to the number of rejecting nodes between two
clusters.

The method createClusterGraph is another way to quickly test
as it does not create a cluster graph but generates a tree that 
we will use as our cluster graph.

In creating our cluster graph, we have two requirements for running recursive DP, knapsack DP,
and the cluster linear program.
    1) no cycles present in the graph
    2) no rejecting node is shared by more than 2 clusters (ie only present in one edge)

If we want to ensure these two properties hold, set the global boolean DEBUG to true. If not, set it to false.
The script driver.py will keep calling testOriginalToCluster until it can create a graph that satisfies these properties.
"""

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
#from IPython.display import Image
#from mat4py import savemat
import csv

from datetime import datetime

global rejectingNodeDict
global clusterDict
global allSubsets
global DEBUG
rejectingNodeDict = {}
sharedRejectingNodes = {}
clusterDict = {}
allSubsets = []
DEBUG = True # change to true if want to print debug statements

# In[32]:

"""
This simple function creates a tree graph using the built-in function nx.random_tree, with the number of Nodes (clusters) as the argument. 
    Weights for each Node are randomly assigned, as well as weights for each edge. The MOST important thing to remember for this function is that
    we are creating a CLUSTER graph, so each Node is actually a cluster of nodes, and the weight of the Node is the number of nodes in the cluster.
    
    The weight of each edge is the number of shared nodes between two clusters.
    This function is used to test the algorithms that work on clustered graphs. Thus, it is important to remember with this function that it is a
    completely arbitrary graph, and the issue of going from the original graph to the clustered graph is not considered at all.
    Returns the cluster graph.
"""
def createClusterGraph(n, maxWeight):
    G = nx.random_tree(n)
    for i in G.nodes():
        rand = random.randint(1,maxWeight)
        G.nodes[i]['weight'] = rand
        for neighbor in nx.neighbors(G, i):
            rand2 = random.randint(0,rand)
            G.add_edge(i, neighbor, weight=rand2)
    print("created cluster graph")
    return G

#clear dictionaries for next graph to test
def clearVisitedNodesAndDictionaries(G):
    setVisitedFalse(G)
    rejectingNodeDict.clear()
    clusterDict.clear()

#Set the attributes for the cluster graph which we generate randomly
#input -- n, the number of clusters
#return -- G, a random tree with n nodes
"""
def createClusterGraph(n):
    G = nx.random_tree(n)
    for i in G.nodes():
        rand = random.randint(1,15)
        G.nodes[i]['weight'] = rand
        for neighbor in nx.neighbors(G, i):
            rand2 = random.randint(0,rand)
            G.add_edge(i, neighbor, weight=rand2)
    return G
"""

#Set node attributes for the original graph, where each node is an individual and edges represent 
#connection in the network
def setAllNodeAttributes(G):
    nodeList = G.nodes()
    for nodeID in nodeList:
        G.nodes[nodeID]["visited"] = False
        G.nodes[nodeID]['criticality'] = random.uniform(0, 1)
        G.nodes[nodeID]["cluster"] = -1

def setVisitedFalse(G):    
    nodeList = G.nodes()
    for nodeID in nodeList:
        G.nodes[nodeID]["visited"] = False

"""
Perform BFS to label a cluster. Note how we called setVisitedFalse(G) upon finding an unvisited node.
The idea is that clusters are "contained" - every cluster is separated from every other cluster by walls of
rejecting nodes. You will never find another cluster, because if you could, it would simply be one cluster.
However, we need to set rejecting nodes to visited so that we don't count them for each accepting neighbor.
This, however, is not problematic, because we label all nodes from the cluster from a single canonical source node.
We will only reach this method from another cluster - whereupon we call setVisitedFalse(), clearing the rejecting
nodes, allowing us to again "contain" this new cluster.
"""
def labelClusters(G, source, clusterNumber, appeal, thirdAlgorithm=False):
    
    if G.nodes[source]['visited'] == False:
        setVisitedFalse(G)
        G.nodes[source]['visited'] = True
    else:
        return 0

    queue = []
    G.nodes[source]['cluster'] = clusterNumber
    queue.append(source)

    #MTI: Added the following lines; without these lines, clusters are incomplete
    clusterDict[clusterNumber] = set()
    clusterDict[clusterNumber].add(source)

    acceptingInThisCluster = 1 #count yourself, you matter!
    rejecting = 0
    while queue:
        start = queue.pop(0)

        for neighbor in nx.neighbors(G, start):
            if G.nodes[neighbor]['visited'] == False: #check if we've added a node to a cluster yet
                if G.nodes[neighbor]['criticality'] < appeal:
                    queue.append(neighbor)
                    G.nodes[neighbor]['cluster'] = clusterNumber
                    G.nodes[neighbor]['visited'] = True
                    acceptingInThisCluster += 1
                    
                    clusterDict[clusterNumber].add(neighbor)
                    
                else:
                    #acceptingInThisCluster -= 1
                    rejecting += 1
                    if clusterNumber not in rejectingNodeDict:
                        rejectingNodeDict[clusterNumber] = set()
                        rejectingNodeDict[clusterNumber].add(neighbor)
                    else:
                        rejectingNodeDict[clusterNumber].add(neighbor)
                    
                    """
                    #the below section accounts for "walls" of rejecting nodes
                    rejecting_queue = []
                    rejecting_queue.append(neighbor)
                    while rejecting_queue:
                        start2 = rejecting_queue.pop(0)
                        for neighbor2 in nx.neighbors(G, start2):
                            if G.nodes[neighbor2]['visited'] == False:
                                if G.nodes[neighbor2]['criticality'] > appeal: #rejecting, so we need to consider this
                                    rejecting_queue.append(neighbor2)
                                    rejectingNodeDict[clusterNumber].add(neighbor2)
                                    G.nodes[neighbor2]['visited'] = True
                """

                    G.nodes[neighbor]['visited'] = True #MTI: Added this line to avoid revisiting this node from other accepting nodes within this cluster.

                    #####       MTI: COMMENTING OUT BFS FROM A REJECTING NODE. 
                    ##### We want to account for shared boundaries of any two clusters only, not the rejected nodes connected to those boundaries.
                    ##### This is because the rejecting nodes do not propagate the mesaage.
                    ##### This may be useful in a modified model later on.
                    '''
                    queueRej = []
                    visited = [neighbor]
                    queueRej.append(neighbor)
                    #we have to look at neighbors of rejecting nodes
                    while queueRej:
                        #while we still have rejecting nodes to consider
                        currentNode = queueRej.pop()
                        neighbors = nx.neighbors(G, currentNode)
                        for node in neighbors:
                            if G.nodes[node]['criticality'] >= appeal and node not in visited:
                                G.nodes[node]['visited'] = True #MTI: Changed from == to =
                                queueRej.append(node)
                                rejectingNodeDict[clusterNumber].add(node)
                                visited.append(node)
                    '''
    if clusterNumber not in rejectingNodeDict:
        print("didnt see any neg nodes from cluster ", clusterNumber)

    return acceptingInThisCluster, rejecting, clusterNumber


# In[43]:

"""
From each node in the nodeList, try to label its cluster. This will return 0 for many nodes, as they are labeled
from the (arbitrary) canonical node in its cluster.
We then select a seed set of up to (not always, depending on the composition of the graph) k source nodes.
"""
def buildClusteredSet(G, threshold, removeCycles, thirdAlgorithm=False):
    nodeList = G.nodes()
    seedSet = []
    clusterCount = 0
    G_cluster = nx.Graph()
    #Build the clusters
    for nodeID in nodeList:
        if (G.nodes[nodeID]['criticality'] < threshold) and (G.nodes[nodeID]['cluster'] == -1):
            summedNeighbors = labelClusters(G, nodeID, clusterCount, threshold, thirdAlgorithm)
            #if summedNeighbors[0] > 0:
            seedSet.append((summedNeighbors[2], summedNeighbors[0], summedNeighbors[1]))
            #print("num rejecting=", summedNeighbors[1])
            makeClusterNode(G_cluster, clusterCount, summedNeighbors[0])
            clusterCount += 1
    #Choose up-to-k
    
    #MTI: Decrement the cluster weight by the number of rejecting nodes that are exclusive to a cluster
    print("rejecting node dictionary:", rejectingNodeDict)
    for clusterNum, rejNodes in rejectingNodeDict.items():
        #if DEBUG: print("rejecting nodes,", clusterNum, rejNodes)
        rejNodes_copy = rejNodes.copy()
        for clusterNum2, rejNodes2 in rejectingNodeDict.items():
            #if DEBUG: print("rejecting nodes 2,", clusterNum2, rejNodes2)
            if clusterNum != clusterNum2:
                rejNodes_copy = rejNodes_copy - rejNodes2
        #if DEBUG: print("Subtracting", len(rejNodes_copy), "cluster", clusterNum )
        G_cluster.nodes[clusterNum]['weight'] -= len(rejNodes_copy)
    if DEBUG:
        sharedRejectingNodes = {}
        for clusterNum, rejNodes in rejectingNodeDict.items():
            for rejNode in rejNodes:
                if rejNode in sharedRejectingNodes:
                    sharedRejectingNodes[rejNode] += 1
                    if sharedRejectingNodes[rejNode] > 2:
                        return False
                else:
                    sharedRejectingNodes[rejNode] = 1
        print("\nMap rejecting nodes to clusters:\n", sharedRejectingNodes)

    make_cluster_edge(G_cluster, G, rejectingNodeDict, removeCycles)    
    return G_cluster

"""
Subtract the number of rejecting nodes connected to a given cluster from its weight.
"""
def computeNegPayoff(G, nodeNum):
    nodeWeight = G.nodes[nodeNum]['weight']
    negPayoff = nx.neighbors(G, nodeNum)
    for negNode in negPayoff:
        add = G.get_edge_data(nodeNum, negNode)
        add = add['weight']
        nodeWeight -= add
    #print("node weight is:", nodeWeight)
    return nodeWeight

"""
Perform bfs from a given source accepting node to identify all the accepting nodes
reachable from the source node. Set visited to true once we have identified a node
as rejecting or accepting.
"""
def bfs(G, node, source):
    # From a source node, perform BFS based on criticality.
    queue = []
    subgraph = []
    queue.append(node)
    subgraph.append(node)
    allSubsets.append(node)

    while queue:
        start = queue.pop(0)
        for neighbor in nx.neighbors(G, start):
            if neighbor in allSubsets:
                continue
            if neighbor in subgraph:
                continue
            elif neighbor != source and neighbor != node:
                queue.append(neighbor)
                allSubsets.append(neighbor)
                subgraph.append(neighbor)
                #print("adding:", neighbor)
    return subgraph

#we defined a new cluster and are adding a node to the cluster graph, whose weight is the number of accepting nodes in that cluster
def makeClusterNode(G, clusterNum, weight):
    G.add_node(clusterNum)
    G.nodes[clusterNum]['weight'] = weight

"""
This method takes the rejecting node dictionary, which maps cluster number to rejecting nodes, and assigns a weight for each pair of clusters
that share rejecting nodes
@params:
    G_orig -> original graph
    G_cluster -> cluster graph whose edge is being labelled
    rejectingNodesDict -> rejecting node dictionary
"""
def make_cluster_edge(G_cluster, G_orig, rejectingNodesDict, removeCycles):
    if DEBUG: print("rej nodes dictioary", rejectingNodeDict)
    for clusterNum, rejNodes in rejectingNodesDict.items():
        for clusterNum2, rejNodes2 in rejectingNodesDict.items():
            if clusterNum >= clusterNum2:
                continue
            else:
                #intersection = [value for value in rejNodes if value in rejNodes2] #compute intersection
                rej_nodes = rejNodes.intersection(rejNodes2)
                intersection = []
                for i in rej_nodes:
                    rej_node = -i
                    intersection.append(rej_node)
                #print("intersection is", intersection)

                #####   MTI: COMMENTING OUT FOR NOW. SEE COMMENT IN labelClusters(.) FUNCTION.
                #we have to confront the situation where there are many rejecting nodes appearing in a 'line' such that we never 
                #reach the nodes in the middle of the line
                '''
                for node1 in rejNodes:
                    for node2 in rejNodes2:
                        if node1 in nx.neighbors(G_orig, node2):
                            intersection.add(node1)
                            intersection.add(node2)
                '''
                weight = len(intersection)
                if len(rej_nodes) > 0:
                    G_cluster.add_edge(clusterNum, clusterNum2, weight=weight, rej_nodes=intersection)
                if removeCycles:
                    try:
                        while len(nx.find_cycle(G_cluster)) > 0:
                            cycle = nx.find_cycle(G_cluster)
                            print("cycle was found in graph, oh no", cycle)
                            rej_nodes_repeat = []
                            for edge in cycle:
                                removed = False
                                rej_nodes = G_cluster.get_edge_data(edge[0], edge[1])['rej_nodes']
                                #print("rejecting nodes in cycle edge are: ", rej_nodes)
                                #print("rejecting node is", data)
                                if len(rej_nodes) == 1:
                                    for node in rej_nodes:
                                        if node in rej_nodes_repeat:
                                            print(rej_nodes)
                                            G_cluster.remove_edge(edge[0], edge[1])
                                            #print("already saw" , node ," so removed edge: ", edge[0], edge[1])
                                            removed = True
                                        else:
                                            rej_nodes_repeat.append(node)
                                if removed:
                                    break

                    except nx.exception.NetworkXNoCycle:
                        #print("no cycle between nodes", clusterNum, clusterNum2,)
                        pass
    components = nx.algorithms.components.connected_components(G_cluster)
    #if DEBUG: print("Connected components: ")
    prev = -1
    for comp in components:
        if DEBUG: print("Component: ", comp)
        if prev == -1:
            prev = list(comp)
            if DEBUG: print("list is", list(comp))
            continue
        else:
            G_cluster.add_edge(prev[0], list(comp)[0], weight=0) #add arbitrary weight

#method to test whether our input graph is correctly forming clusters, then runs dynamic programming
#input -- n, number of nodes in random graph
#           c, criticality
#           k, number of clusters to seed
def testOriginaltoCluster(G, n, c, k, removeCycles):
    setAllNodeAttributes(G)
    showOriginalGraph(G, c)
    saveOriginalGraph(G, c, "currently_in_use/tests/original_graph.txt")
    G_cluster = buildClusteredSet(G, c, removeCycles)
    if G_cluster == False:
        print("DIDNT WORK")
        clearVisitedNodesAndDictionaries(G)
        return False
    '''
    f = open("make_matrix.txt", "a")
    f.write("cluster dictionary:" + str(clusterDict) + "\n")
    f.write("rej node dictionary: " + str(rejectingNodeDict) + "\n")
    f.write("edge data:" + str(G_cluster.edges.data()) + "\n")
    f.write("node data:" + str(G_cluster.nodes.data()) + "\n")
    f.close()
    '''
    return G_cluster

def showOriginalGraph(G, c):
    color_map = []
    for nodeID in G.nodes():
        if G.nodes[nodeID]['criticality'] >= c:
            color_map.append('red')
        else:
            color_map.append('green')
    plt.figure('original network')
    nx.draw_networkx(G, node_color = color_map, pos=nx.spring_layout(G, iterations=1000), arrows=False, with_labels=True)

'''
Saves the original graph and associated criticality in a file called original_graph.txt
This file can be used in conjuntion with create_from_file function in create_graph_from_file
The format used here is described in create_graph_from_file class

@params:
    G -> original graph
    c -> criticality (used for show purposes and creating cluster)
'''
def saveOriginalGraph(G, c, filename):
    with open(filename, 'w') as graph_info:
        timestamp = datetime.timestamp(datetime.now())
        date = datetime.fromtimestamp(timestamp)
        graph_info.write("o\n")
        graph_info.write("crit " + str(c) + "\n")
        graph_info.write("# Timestamp: " + str(date) + "\n")
        graph_info.write("# Nodes: " + str(G.number_of_nodes()) + "\n")
        data = G.edges.data()
        graph_info.write("# Edges: " + str(len(data)))
        weights = G.nodes.data('criticality')
        for node in weights:
            #print(node)
            graph_info.write("\n" + str(node[1]))
        for item in data:
            graph_info.write("\n" + str(item[0]) + " " + str(item[1]))

"""
DEPRECATED-- was used to create the matrix of nodes and edges to become input to linear programming.
However, we are now doing linear programming using python.

"""
def makeMatrix(G, n):
    f = open("make_matrix.txt", "w")
    matrix = [[0] * n for _ in range(n)] #store payoff
    node_weights = nx.get_node_attributes(G, name='weight')
    #print("weight of nodes is:", weight)
    edge_weights = nx.get_edge_attributes(G, 'weight')
    #print("weight of edges is:", edge_weights)
    nodes = []
    edges = [[] for _ in range(n)]
    for key, value in edge_weights.items():
        if value == 0:
            continue
        matrix[key[0]][key[1]] = value
        matrix[key[1]][key[0]] = value
        edges[key[0]].append(value)
        edges[key[1]].append(value)
    for key, value in node_weights.items():
        matrix[key][key] = value
        nodes.append(value)
    for i in range(n):
        fullStr = ','.join([str(elem) for elem in matrix[i] ])
        f.write(fullStr + "\n")
    f.close()
    with open('make_matrix.csv', mode='w', newline='') as make_matrix:
        matrix_writer = csv.writer(make_matrix, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(n):
            matrix_writer.writerow(matrix[i])
    return nodes, edges
