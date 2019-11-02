import networkx as nx
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


# In[32]:


rejectingNodeDict = {}
clusterDict = {}

fh=open("CollegeMsg.txt", encoding='utf=8') # use utf-8 encoding

G=fh.readlines()
#print(G)
G_College_Msg = nx.Graph()
for i in G: #iterate through (i,j) pairs, adding edges to graph
    i = i.split()
    #print(i, i[0])
    G_College_Msg.add_node(int(i[0]))
    G_College_Msg.add_node(int(i[1]))
    G_College_Msg.add_edge(int(i[0]), int(i[1]))
    
n = G_College_Msg.number_of_nodes()

G_DP = nx.Graph()

G_DP.add_edge(8,16,weight=8)
G_DP.add_edge(7,15,weight=10)
G_DP.add_edge(7,14,weight=2)
G_DP.add_edge(6,13,weight=5)
G_DP.add_edge(6,12, weight=4)
G_DP.add_edge(5,11, weight=3)
G_DP.add_edge(4,10, weight=7)
G_DP.add_edge(3,9, weight=4)
G_DP.add_edge(3,8, weight=7)
G_DP.add_edge(2,7, weight=5)
G_DP.add_edge(2,6, weight=8)
G_DP.add_edge(1,5, weight=3)
G_DP.add_edge(1,4, weight=4)
G_DP.add_edge(0,3, weight=6)
G_DP.add_edge(0,2, weight=9)
G_DP.add_edge(0,1, weight=9)
G_DP.node[0]['weight'] = 8
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

G_cluster = nx.Graph()
def setClusterGraphAttributes(G, n):
    for i in G.nodes():
        rand = random.randint(1,15)
        rand *= 10000
        G.node[i]['weight'] = rand
        for neighbor in nx.neighbors(G, i):
            rand2 = random.randint(0,10)*10000
            G.add_edge(i, neighbor, weight=rand2)

def setAllNodeAttributes(G):
    nodeList = G.nodes()
    for nodeID in nodeList:
        G.node[nodeID]["visited"] = False
        G.node[nodeID]['criticality'] = random.uniform(0, 1)
        G.node[nodeID]["cluster"] = -1

def clearVisitedNodesAndDictionaries(G):
    setVisitedFalse(G)
    rejectingNodeDict.clear()
    clusterDict.clear()

def setVisitedFalse(G):    
    nodeList = G.nodes()
    for nodeID in nodeList:
        G.node[nodeID]["visited"] = False

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

    acceptingInThisCluster = 1 #count yourself, you matter!
    rejecting = 0
    while queue:
        start = queue.pop(0)

        for neighbor in nx.neighbors(G, start):
            if G.node[neighbor]['visited'] == False: #check if we've added a node to a cluster yet
                if G.node[neighbor]['criticality'] < appeal:
                    queue.append(neighbor)
                    G.node[neighbor]['cluster'] = clusterNumber
                    G.node[neighbor]["visited"] = True
                    acceptingInThisCluster += 1
                    
                    if clusterNumber not in clusterDict:
                        clusterDict[clusterNumber] = set()
                        clusterDict[clusterNumber].add(neighbor)
                    else:
                        clusterDict[clusterNumber].add(neighbor)
                    
                else:
                    #acceptingInThisCluster -= 1
                    rejecting += 1
                    if clusterNumber not in rejectingNodeDict:
                        rejectingNodeDict[clusterNumber] = set()
                        rejectingNodeDict[clusterNumber].add(neighbor)
                    else:
                        rejectingNodeDict[clusterNumber].add(neighbor)
                    queueRej = []
                    visited = [neighbor]
                    queueRej.append(neighbor)
                    #we have to look at neighbors of rejecting nodes
                    while queueRej:
                        #while we still have rejecting nodes to consider
                        currentNode = queueRej.pop()
                        neighbors = nx.neighbors(G, currentNode)
                        for node in neighbors:
                            if G.node[node]['criticality'] >= appeal and node not in visited:
                                G.node[node]['visited'] == True
                                queueRej.append(node)
                                rejectingNodeDict[clusterNumber].add(node)
                                visited.append(node)
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
    
    #Build the clusters
    for nodeID in nodeList:
        if (G.node[nodeID]['criticality'] < threshold) and (G.node[nodeID]['cluster'] == -1):
            summedNeighbors = labelClusters(G, nodeID, clusterCount, threshold, thirdAlgorithm)
            #if summedNeighbors[0] > 0:
            seedSet.append((summedNeighbors[2], summedNeighbors[0], summedNeighbors[1]))
            #print("num rejecting=", summedNeighbors[1])
            make_Cluster_node(G_cluster, clusterCount, summedNeighbors[0])
            clusterCount += 1
    #Choose up-to-k
    make_cluster_edge(G_cluster, G_test_DP, rejectingNodeDict)    
    return seedSet 


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
    nodeWeight = G.node[nodeNum]['weight']
    negPayoff = nx.neighbors(G, nodeNum)
    for negNode in negPayoff:
        add = G.get_edge_data(nodeNum, negNode)
        add = add['weight']
        nodeWeight -= add
    #print("node weight is:", nodeWeight)
    return nodeWeight


def tryDP(G, i, k):
    #This is different since we are considering each node's weight in the graph to be the number of accepting nodes in a given cluster
    #i = number_of_nodes(G)
    #k = number of seeds
    storePayoff = [[0] * i for _ in range(k)] #store payoff
    storeSeeds = [[[]] * i for _ in range(k)] #store seeds at each stage
    tree = nx.bfs_tree(G, 1)
    #print(nodes)
    for numSeeds in range(0,k): #bottom up DP
        nodes = list(reversed(list((nx.topological_sort(tree))))) #look at nodes in reverse topological order
        for node, j in zip(nodes, range(0,i)): 
            if j == 0 and numSeeds == 0: #first entry
                #breakpoint()
                storeSeeds[numSeeds][j] = [node]
                nodeWeight = computeNegPayoff(G, node)
                #print(nodeWeight)
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


def computePayoff(G, i, k):
    #This is different since we are considering each node's weight in the graph to be the number of accepting nodes in a given cluster
    #i = number_of_nodes(G)
    #k = number of seeds
    storePayoff = [[0] * i for _ in range(k)] #store payoff
    storeSeeds = [[[]] * i for _ in range(k)] #store seeds at each stage
    for numSeeds in range(0,k): #bottom up DP
        for node, j in zip(G, range(0,i)): 
            if j == 0 and numSeeds == 0: #first entry
                #breakpoint()
                storeSeeds[numSeeds][j] = [node]
                nodeWeight = computeNegPayoff(G, node)
                #print(nodeWeight)
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
    #print(storePayoff)
    #print(storeSeeds)
    maxVal = storePayoff[k-1][i-1]
    for j in range(0,k):
        if storePayoff[j][i-1] > maxVal:
            maxVal = storePayoff[j][i-1]
    return (maxVal, storeSeeds[j][i-1])

def DP_Improved(G, i, k):
    #storePayoff = [[0] * i for _ in range(k)] #store payoff
    #storeSeeds = [[[]] * i for _ in range(k)] #store seeds at each stage
    tree = nx.bfs_tree(G, 1)
    nodes = list((nx.topological_sort(tree))) #look at nodes in reverse topological order
    print(nodes)
    print("root is", nodes[0])
    neighbors = nx.neighbors(G, nodes[0])
    print(neighbors)
    subgraph_list = [] #store subgraph
    for node in neighbors: #we want to compute payoff for the subtree 
        #print(node)
        subgraph = bfs(G, node, nodes[0])
        #print("subgraph:", subgraph)
        subgraph_list.append(subgraph)
    num_children = len(subgraph_list)
    #print("num subgraphs:", num_children)
    part = partitions(k, num_children)
    amount = 0
    storePayoffs = []
    for p in part: #get each partition of the seeds
        #print(p)
        for i, subgraph in zip(range(0,num_children), subgraph_list): #pick the correct number of seeds for each subgraph
            G_sub = G.subgraph(subgraph)
            j = G_sub.number_of_nodes()
            #print("nodes in subgraph:", j)
            if p[i] != 0:
                amountCur = computePayoff(G_sub, j, p[i])
                amount += amountCur[0]
       # print('payoff is:', amountCur)
        storePayoffs.append(amount)
        amount = 0
    maxval = max(storePayoffs) #get largest subset value
    print("max val is:", maxval)
    return maxval


def partitions(n, k): #stars and bars
    for c in itertools.combinations(range(n+k-1), k-1):
        yield [b-a-1 for a, b in zip((-1,)+c, c+(n+k-1,))]


def bfs(G, node, source):
    
    # From a source node, perform BFS based on criticality.
    queue = []
    subgraph = []
    queue.append(node)
    subgraph.append(node)

    while queue:
        start = queue.pop(0)
        for neighbor in nx.neighbors(G, start):
            if neighbor in subgraph:
                #print("already picked", neighbor)
                continue
            elif neighbor != source:
                queue.append(neighbor)
                subgraph.append(neighbor)
                #print("adding:", neighbor)
    return subgraph

#we defined a new cluster and are adding a node to the cluster graph, whose weight is the number of accepting nodes in that cluster
def make_Cluster_node(G, clusterNum, weight):
    G.add_node(clusterNum)
    G.node[clusterNum]['weight'] = weight

def make_cluster_edge(G_cluster, G_orig, rejectingNodesDict):
    for clusterNum, rejNodes in rejectingNodesDict.items():
        for clusterNum2, rejNodes2 in rejectingNodesDict.items():
            if clusterNum >= clusterNum2:
                continue
            else:
                #intersection = [value for value in rejNodes if value in rejNodes2] #compute intersection
                intersection = rejNodes.intersection(rejNodes2)
                #we have to confront the situation where there are many rejecting nodes appearing in a 'line' such that we never 
                #reach the nodes in the middle of the line
                for node1 in rejNodes:
                    for node2 in rejNodes2:
                        if node1 in nx.neighbors(G_orig, node2):
                            intersection.add(node1)
                            intersection.add(node2)
                weight = len(intersection)
                if weight > 0:
                    G_cluster.add_edge(clusterNum, clusterNum2, weight=weight)
                    print("intersection between nodes ", clusterNum, clusterNum2, "is:", intersection, "of weight", weight)

start_time = time.time()

G_test_DP = nx.random_tree(200)
numNode = G_test_DP.number_of_nodes()

setClusterGraphAttributes(G_test_DP, numNode)
setAllNodeAttributes(G_test_DP)
seeds = buildClusteredSet(G_test_DP, 0.7)
print("success:", seeds)
print("cluster dict:", clusterDict)
print("rej node dict", rejectingNodeDict)
weight1 = nx.get_node_attributes(G_cluster, name='weight')
weight2 = nx.get_edge_attributes(G_cluster, name='weight')
print(weight1, weight2)
makeMatrix(G_test_DP, numNode)

print(weight1,weight2)

test1 = tryDP(G_cluster, G_cluster.number_of_nodes(), 6)
end_time = time.time()
maxval = DP_Improved(G_cluster, numNode, 6)
print("payoff test DP is: ", test1, end_time - start_time)
print("payoff subtree DP is:", maxval)

nx.draw_networkx(G_cluster, pos=None, arrows=False, with_labels=True)
plt.show()
nx.draw_networkx(G_test_DP, pos=None, arrows=False, with_labels=True)
plt.show()