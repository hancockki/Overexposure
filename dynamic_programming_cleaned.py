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
#this is a dictionary where the keys are cluster numbers and the value is the rejecting nodes connected to the cluster
rejectingNodeDict = {}
clusterDict = {}
allSubsets = []

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
    return G

"""
Set node attributes for the original graph, where each node is an individual and edges represent 
connection in the network

Used in creating the cluster graph
"""
def setAllNodeAttributes(G):
    nodeList = G.nodes()
    for nodeID in nodeList:
        G.nodes[nodeID]["visited"] = False
        G.nodes[nodeID]['criticality'] = random.uniform(0, 1)
        G.nodes[nodeID]["cluster"] = -1

"""
Node attribute 'visited' is set to false. Used to reset the graph in each iteration of BFS where we need to check and see if we have
    already visited that node in our BFS.
    Does not return anything.
"""
def setVisitedFalse(G):    
    nodeList = G.nodes()
    for nodeID in nodeList:
        G.nodes[nodeID]["visited"] = False

"""
Perform breadth first search (BFS) to label a cluster. Note how we called setVisitedFalse(G) upon finding an unvisited node.
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
    # while we still have nodes to label
    while queue:
        start = queue.pop(0)

        for neighbor in nx.neighbors(G, start):
            if G.nodes[neighbor]['visited'] == False: #check if we've added a node to a cluster yet
                if G.nodes[neighbor]['criticality'] < appeal: # if we are below the threshold, add to cluster
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
    return acceptingInThisCluster, rejecting, clusterNumber

"""
From each node in the nodeList, try to label its cluster. This will return 0 for many nodes, as they are labeled
from the (arbitrary) canonical node in its cluster.
We then select a seed set of up to (not always, depending on the composition of the graph) k source nodes.
We select the k source clusters with the highest accepting degree, implemented by sorting a list of tuples.
"""
def buildClusteredSet(G, threshold, thirdAlgorithm=False):

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
        print("rejecting nodes,", clusterNum, rejNodes)
        rejNodes_copy = rejNodes.copy()
        for clusterNum2, rejNodes2 in rejectingNodeDict.items():
            if clusterNum != clusterNum2:
                rejNodes_copy = rejNodes_copy - rejNodes2
        print("Subtracting", len(rejNodes_copy), "cluster", clusterNum )
        G_cluster.nodes[clusterNum]['weight'] -= len(rejNodes_copy)

    make_cluster_edge(G_cluster, G, rejectingNodeDict)    
    return G_cluster


def makeMatrix(G, n):
    f = open("make_matrix.txt", "w")
    f.write("\n Next test: \n")
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
        f.write("[" + fullStr + "]" + "\n")
    f.close()
    with open('make_matrix.csv', mode='w', newline='') as make_matrix:
        matrix_writer = csv.writer(make_matrix, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(n):
            matrix_writer.writerow(matrix[i])

"""
THIS IS THE MOST USEFUL FUNCTION IN THE ENTIRE FILE

Here we are running dynamic programming recursively to choose the best k clusters in the graph to seed.

@params:
    G --> the cluster graph we are seeding from
    tree --> the bfs tree of the graph
    k --> the number of clusters we are seeding
    source --> the starting cluster (highest degree)
    storePayoff --> the payoff matrix
    witness --> witness vector, used in recursive calls
"""
def recursive_DP(G, tree, k, source, storePayoff, witness):
    #TRUE is 0 and FALSE is 1 for storePayoff
    #print("source is:", source)
    precomputed_0 = precomputed_1 = False
    if storePayoff[0][source][k]  != None: #indicates we have already computed the payoff for this subtree
        precomputed_0 = True
    if storePayoff[1][source][k] != None: #already computed
        precomputed_1 = True

    if k <= 0: #base case, meaning we have no seeds
        #print("no seeds")
        storePayoff[0][source][k] = float("-inf") #set payoff to negative infinity
        storePayoff[1][source][k] = 0
        return 
    if tree.out_degree(source) == 0: #base case, meaning we are at a leaf node
        #print("at leaf node")
        #if k >= 1:
        storePayoff[0][source][k] = G.nodes[source]['weight']
        storePayoff[1][source][k] = 0
        return 
    
    #CASE 1: LEAVE SOURCE 
    # we have to consider taking and leaving the source node with every recursive call, and compare both results
    if not precomputed_1:
        neighbors_list = []
        for i in list(tree.out_edges(source)):
            neighbors_list.append(i[1])

        #print(neighbors_list, "NEIGHBORS LIST")
        num_children = len(neighbors_list)
        partitions_list = list(partitions(k, num_children)) #seed all k seeds among the child nodes
        maxSum = float("-inf")
        opt_allocation = None
        opt_take_child = None
        #take_child = {(i, j):False for i, j in zip(neighbors_list, partitions_list)} #dictionary to keep track of whether we've taken the children
       # print("LEAVE SOURCE")
        for p in partitions_list: #loop through partitions of seeds
            take_child = {} #reset take_child
            if p == [2,1,0] and source == 1:
                print("debugging") #IGNORE, used for debugging
           # print(p)
            sum_so_far = 0
            allocation = {}
            for i in range(0, num_children):
               # print("p is", p[i])
                allocation[neighbors_list[i]] = p[i] # set our allocation of seeds to current partition
                recursive_DP(G, tree, p[i], neighbors_list[i], storePayoff, witness) #recurse on current allocation
                edge_weight = G.get_edge_data(source, neighbors_list[i]) # get the edge weight

                #IMPORTANT!!!!!!! If the payoff for taking the child minus the weight of the negative edge is GREATER than the payoff for leaving the child, take it
                if storePayoff[0][neighbors_list[i]][p[i]] - edge_weight['weight'] >= storePayoff[1][neighbors_list[i]][p[i]]:
                   # print("take child:", neighbors_list[i])
                    sum_so_far += storePayoff[0][neighbors_list[i]][p[i]] - edge_weight['weight']
                    take_child[neighbors_list[i]] = True
                else:
                    sum_so_far += storePayoff[1][neighbors_list[i]][p[i]]
                    take_child[neighbors_list[i]] = False
            # if this partition is better than maxSum, take it!
            if sum_so_far > maxSum:
                maxSum = sum_so_far
                opt_allocation = allocation
                opt_take_child = take_child
        if source == 1:
            print("debugging")
        
        #populate the table for leaving source
        storePayoff[1][source][k] = maxSum
        witness[1][source][k] = (opt_take_child, opt_allocation)

    #CASE 2: TAKE SOURCE
    # in this case, we are taking the source node and recurse accordingly
    if not precomputed_0:
        partitions_list = partitions(k-1, num_children)
        maxSum = float("-inf")
        opt_allocation = None
        take_child = {}
       # take_child = {i:False for i in neighbors_list}
        #print("TAKE SOURCE")
        for p in partitions_list:
            sum_so_far = 0
            allocation = {}
            for i in range(0, num_children): #loop through children and partitions
                allocation[neighbors_list[i]] = p[i]
                recursive_DP(G, tree, p[i], neighbors_list[i], storePayoff, witness)
                edge_data = G.get_edge_data(neighbors_list[i], source)
                # print("current partition:", p[i], " \n take child payoff:", storePayoff[0][neighbors_list[i]][p[i]-1])

                #IMPORTANT: we need to check if taking the child is better than leaving. We've already subtracted the weight of the negative edge
                if storePayoff[0][neighbors_list[i]][p[i]] >= storePayoff[1][neighbors_list[i]][p[i]]:
                    #print("take child, root:", neighbors_list[i])
                    sum_so_far += storePayoff[0][neighbors_list[i]][p[i]] - edge_data['weight']
                    take_child[neighbors_list[i]] = True
                else:
                    # leave child
                    sum_so_far += storePayoff[1][neighbors_list[i]][p[i]] - edge_data['weight']
                    take_child[neighbors_list[i]] = False
            # again, if the current partition is better than any seen so far, keep it
            if sum_so_far > maxSum:
                maxSum = sum_so_far
                opt_allocation = allocation
        #populate the table for taking source
        storePayoff[0][source][k] = maxSum + G.nodes[source]['weight'] 
        witness[0][source][k] = (take_child, opt_allocation)

"""
Stars and bars problem

"""
def partitions(n, k): #stars and bars, k subtrees and n seeds to allocate among them
    for c in itertools.combinations(range(n+k-1), k-1):
        yield [b-a-1 for a, b in zip((-1,)+c, c+(n+k-1,))]

#we defined a new cluster and are adding a node to the cluster graph, whose weight is the number of accepting nodes in that cluster
def make_Cluster_node(G, clusterNum, weight):
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
def make_cluster_edge(G_cluster, G_orig, rejectingNodesDict):
    print(rejectingNodeDict)
    # 
    for clusterNum, rejNodes in rejectingNodesDict.items():
        for clusterNum2, rejNodes2 in rejectingNodesDict.items():
            if clusterNum >= clusterNum2:
                continue
            else:
                #intersection = [value for value in rejNodes if value in rejNodes2] #compute intersection
                intersection = rejNodes.intersection(rejNodes2)
                print("intersection is", intersection)

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
                    print("intersection between nodes ", clusterNum, clusterNum2, "is:", intersection, "of weight", weight)
    components = nx.algorithms.components.connected_components(G_cluster)
    print("Connected components: ")
    prev = -1
    for comp in components:
        print("Component: ", comp)
        if prev == -1:
            prev = list(comp)
            print("list is", list(comp))
            continue
        else:
            G_cluster.add_edge(prev[0], list(comp)[0], weight=0) #add arbitrary weight

"""
Here, we read in the file from SNAP, and read it line by line. Each line is composed of the edges (u,v) as well as the 
time stamp for creating the graph
returns a graph of college students connected by edges, with no attributes
"""
def college_Message():
    fh=open("CollegeMsg.txt", encoding='utf=8') # use utf-8 encoding
    G=fh.readlines()
    G_College_Msg = nx.Graph()
    for i in G: #iterate through (i,j) pairs, adding edges to graph
        i = i.split()
        #print(i, i[0])
        G_College_Msg.add_node(int(i[0]))
        G_College_Msg.add_node(int(i[1]))
        G_College_Msg.add_edge(int(i[0]), int(i[1]))
    setAllNodeAttributes(G_College_Msg)
    print("sucess")
    G = buildClusteredSet(G_College_Msg, 0.7)

    clearVisitedNodesAndDictionaries(G_College_Msg)
    return G

"""
Here, we are testing how our algorithm creates a cluster graph based on the original graph. We start with a tree
and build a clustered graph based on that tree.

@params:
    n --> number of nodes in original graph
    threshold --> our threshold for accepting

@returns:
    G_cluster -> the cluster graph created

"""
def testOriginaltoCluster(n, threshold):
    G_test = nx.random_tree(n)
    setAllNodeAttributes(G_test)
    G_cluster = buildClusteredSet(G_test, threshold)

    f = open("make_matrix_info.txt", "w+")
    f.write("cluster dictionary:" + str(clusterDict) + "\n")
    f.write("rej node dictionary: " + str(rejectingNodeDict) + "\n")
    f.write("edge data:" + str(G_cluster.edges.data()) + "\n")
    f.write("node data:" + str(G_cluster.nodes.data()) + "\n")
    f.close()
    clearVisitedNodesAndDictionaries(G_cluster)
    makeMatrix(G_cluster, G_cluster.number_of_nodes())

    color_map = []
    for nodeID in G_test.nodes():
        if G_test.nodes[nodeID]['criticality'] >= threshold:
            color_map.append('red')
        else:
            color_map.append('green')
    # graph original tree
    plt.figure(1)
    nx.draw_networkx(G_test, node_color = color_map, pos=nx.spring_layout(G_test), arrows=False, with_labels=True)
    return G_cluster

"""
Driver function for running dynamic programming

@params:
    G --> graph for which we are choosing seeds
    k --> number of seeds to pick
"""
def runRecursiveDP(G, k):
    makeMatrix(G, G.number_of_nodes())

    storePayoff = [ [ [None] * (k+1) for _ in range(G.number_of_nodes())] for _ in range(2)]
    witness = [ [ [None] * (k+1) for _ in range(G.number_of_nodes())] for _ in range(2)]
    #tree = nx.bfs_tree(G, 0)
    nodes_tup = sorted(G.degree, key=lambda x: x[1], reverse=True) #sort by highest degree node
    print("root is", nodes_tup[0][0]) #take top degree node as root
    root = nodes_tup[0][0]
    tree = nx.bfs_tree(G, root)
    recursive_DP(G, tree, k, root, storePayoff, witness)
    print("best payoff root", storePayoff[0][root][k])
    print("best payoff no root",storePayoff[1][root][k])

    #print("payoff test DP is: ", test1)
    #print("payoff subtree DP is:", maxval, "with seeds: ", seeds)
    clearVisitedNodesAndDictionaries(G)



#clear dictionaries for next graph to test
def clearVisitedNodesAndDictionaries(G):
    setVisitedFalse(G)
    rejectingNodeDict.clear()
    clusterDict.clear()

#main function, used for calling things
def main():
    #G = testOriginaltoCluster(50, 0.7)
   # G = college_Message()
    G = createClusterGraph(15, 20)
    runRecursiveDP(G, 5)
    pos = nx.spring_layout(G)
    node_labels = nx.get_node_attributes(G,'weight')

    plt.figure(2)
    nx.draw(G, pos)
    nx.draw_networkx_labels(G, pos=pos, labels=node_labels)
    #edge_labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G, pos=pos)
    plt.savefig('this.png')
    plt.show()

if __name__== "__main__":
  main()