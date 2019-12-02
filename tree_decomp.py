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

global rejectingNodeDict
global clusterDict
global allSubsets
rejectingNodeDict = {}
clusterDict = {}
allSubsets = []

# In[32]:

#Set the attributes for the cluster graph which we generate randomly
#input -- n, the number of nodes
#return -- G, a random tree with n nodes
def createClusterGraph(n):
    G = nx.random_tree(200)
    for i in G.nodes():
        rand = random.randint(1,15)
        G.nodes[i]['weight'] = rand
        for neighbor in nx.neighbors(G, i):
            rand2 = random.randint(0,10)*10000
            G.add_edge(i, neighbor, weight=rand2)
    return G

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

def labelClusters(G, source, clusterNumber, appeal, thirdAlgorithm=False):
    
    # Perform BFS to label a cluster. Note how we called setVisitedFalse(G) upon finding an unvisited node.
    # The idea is that clusters are "contained" - every cluster is separated from every other cluster by walls of
    # rejecting nodes. You will never find another cluster, because if you could, it would simply be one cluster.
    # However, we need to set rejecting nodes to visited so that we don't count them for each accepting neighbor.
    # This, however, is not problematic, because we label all nodes from the cluster from a single canonical source node.
    # We will only reach this method from another cluster - whereupon we call setVisitedFalse(), clearing the rejecting
    # nodes, allowing us to again "contain" this new cluster.
    
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
                    
                    #if clusterNumber not in clusterDict:
                    #    clusterDict[clusterNumber] = set()
                    #    clusterDict[clusterNumber].add(neighbor)
                    #else:
                    clusterDict[clusterNumber].add(neighbor) #MTI: Only this line is sufficient. Commented out the previous 4 lines.
                    
                else:
                    #acceptingInThisCluster -= 1
                    rejecting += 1
                    if clusterNumber not in rejectingNodeDict:
                        rejectingNodeDict[clusterNumber] = set()
                        rejectingNodeDict[clusterNumber].add(neighbor)
                    else:
                        rejectingNodeDict[clusterNumber].add(neighbor)
                    
                    G.nodes[neighbor]['visited'] = True #MTI: Added this line to avoid revisiting this node from other accepting nodes within this cluster.

    return acceptingInThisCluster, rejecting, clusterNumber


# In[43]:

def buildClusteredSet(G, threshold, thirdAlgorithm=False):

    # From each node in the nodeList, try to label its cluster. This will return 0 for many nodes, as they are labeled
    # from the (arbitrary) canonical node in its cluster.
    # We then select a seed set of up to (not always, depending on the composition of the graph) k source nodes.
    # We select the k source clusters with the highest accepting degree, implemented by sorting a list of tuples.
    
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
            make_Cluster_node(G_cluster, clusterCount, summedNeighbors[0])
            clusterCount += 1
    #Choose up-to-k
    
    #MTI: Decrement the cluster weight by the number of rejecting nodes that are exclusive to a cluster
    for clusterNum, rejNodes in rejectingNodeDict.items():
        rejNodes_copy = rejNodes.copy()
        for clusterNum2, rejNodes2 in rejectingNodeDict.items():
            if clusterNum != clusterNum2:
                rejNodes_copy = rejNodes_copy - rejNodes2
        #print("Subtracting", len(rejNodes_copy), "cluster", clusterNum )
        G_cluster.nodes[clusterNum]['weight'] -= len(rejNodes_copy)
    

    make_cluster_edge(G_cluster, G, rejectingNodeDict)    
    return G_cluster


def makeMatrix(G, n):
    f = open("make_matrix.txt", "w+")
    matrix = [[0] * n for _ in range(n)] #store payoff
    weight = nx.get_node_attributes(G, name='weight')
    #print("weight of nodes is:", weight)
    edge = nx.get_edge_attributes(G, 'weight')
    #print("weight of edges is:", edge)
    for key, value in edge.items():
        matrix[key[0]][key[1]] = value
        matrix[key[1]][key[0]] = value
    for key, value in weight.items():
        matrix[key][key] = value
    for i in range(n):
        fullStr = ','.join([str(elem) for elem in matrix[i] ])
        #print(fullStr)
        f.write("[" + fullStr + "]" + "\n")
    f.close()
    with open('make_matrix.csv', mode='w', newline='') as make_matrix:
        matrix_writer = csv.writer(make_matrix, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(n):
            matrix_writer.writerow(matrix[i])

def computeNegPayoff(G, nodeNum):
    #print("node is:" , nodeNum)
    nodeWeight = G.nodes[nodeNum]['weight']
    negPayoff = nx.neighbors(G, nodeNum)
    for negNode in negPayoff:
        add = G.get_edge_data(nodeNum, negNode)
        add = add['weight']
        nodeWeight -= add
    #print("node weight is:", nodeWeight)
    return nodeWeight

def partitions(n, k): #stars and bars
    for c in itertools.combinations(range(n+k-1), k-1):
        yield [b-a-1 for a, b in zip((-1,)+c, c+(n+k-1,))]


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
                #print("already picked", neighbor)
                continue
            elif neighbor != source and neighbor != node:
                queue.append(neighbor)
                allSubsets.append(neighbor)
                subgraph.append(neighbor)
                #print("adding:", neighbor)
    print("subgraph is", subgraph)
    return subgraph

#we defined a new cluster and are adding a node to the cluster graph, whose weight is the number of accepting nodes in that cluster
def make_Cluster_node(G, clusterNum, weight):
    G.add_node(clusterNum)
    G.nodes[clusterNum]['weight'] = weight

#takes the rejecting node dictionary, which maps cluster number to rejecting nodes, and assigns a weight for each pair of clusters
#that share rejecting nodes
#input -- original graph, cluster graph, rejecting node dictionary
def make_cluster_edge(G_cluster, G_orig, rejectingNodesDict):
    for clusterNum, rejNodes in rejectingNodesDict.items():
        for clusterNum2, rejNodes2 in rejectingNodesDict.items():
            if clusterNum >= clusterNum2:
                continue
            else:
                #intersection = [value for value in rejNodes if value in rejNodes2] #compute intersection
                intersection = rejNodes.intersection(rejNodes2)

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
                if weight > 0:  
                    G_cluster.add_edge(clusterNum, clusterNum2, weight=weight)
                    #print("intersection between nodes ", clusterNum, clusterNum2, "is:", intersection, "of weight", weight)
    components = nx.algorithms.components.connected_components(G_cluster)
    print("Connected components: ")
    prev = -1
    for comp in components:
        #print("Component: ", comp)
        if prev == -1:
            prev = list(comp)
            continue
        else:
            G_cluster.add_edge(prev[0], list(comp)[0], weight=0)



#method to test whether our input graph is correctly forming clusters, then runs dynamic programming
#input -- n, number of nodes in random graph
#           c, criticality
#           k, number of clusters to seed
def testOriginaltoCluster(n, c, k):
    G_test = nx.random_tree(n)
    setAllNodeAttributes(G_test)
    G_cluster = buildClusteredSet(G_test, c) #return cluster graph
    #start tree decomposition
    tree_decomp = approx.treewidth_min_degree(G_cluster)
    print("cluster dict:", clusterDict)
    print("rej node dict", rejectingNodeDict)
    clearVisitedNodesAndDictionaries(G_cluster)
    makeMatrix(G_cluster, G_cluster.number_of_nodes())

    color_map = []
    for nodeID in G_test.nodes():
        if G_test.nodes[nodeID]['criticality'] >= c:
            color_map.append('red')
        else:
            color_map.append('green')
    fig2 = plt.figure(1)
    G_tree = tree_decomp[1]
    print("List of nodes, where each node is a bag, and each bag contains a set of nodes in the bag:\n", list(G_tree.nodes()), "\nList of edges, where each edge is listed:\n" , list(G_tree.edges()))
    nx.draw_networkx(G_tree, pos=nx.spring_layout(G_tree, iterations=200), arrows=False, with_labels=True)
    #nx.draw_networkx(G_cluster, node_color = color_map, pos=nx.spring_layout(G_test, iterations=1000), arrows=False, with_labels=True)
    return G_cluster

#clear dictionaries for next graph to test
def clearVisitedNodesAndDictionaries(G):
    setVisitedFalse(G)
    rejectingNodeDict.clear()
    clusterDict.clear()

def main():
    G = testOriginaltoCluster(30, 0.6, 6)
    #testRandomCluster()
    #G = college_Message()

    fig1 = plt.figure(2)
    nx.draw_networkx(G, pos=nx.spring_layout(G, iterations=200), arrows=False, with_labels=True)
    plt.show()

if __name__== "__main__":
  main()

  """
  NEW PLAN: We want to consider tree decomposition, so we basically do tree decomposition and end up with 'bags' of nodes
  BUT there are nodes that are in multiple bags, and as a result we want to avoid the case of double counting edges in the cluster graph
  Thus, we implement the following conditions:
    1) We create a cluster graph where any two clusters do not share more than 3 rejecting nodes
    2) If we DO get a cycle, we know that the rejecting nodes between the clusters in the cycle are different
    3) In this case, we do tree decomposition, create bags, and then compute all the possible allocations of nodes, creating a table for picking
    nodes where we enumerate all possibilities for U
  """