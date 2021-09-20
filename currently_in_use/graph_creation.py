# wq:these comments will need a massive update once code is done

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

import view

import networkx as nx
from networkx.algorithms import tree
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
import sys

from datetime import datetime

global rejectingNodeDict
global DEBUG
rejectingNodeDict = {}
DEBUG = False

MAX_ATTEMPTS_SASISFY_ASS_1 = 100

#method to test whether our input graph is correctly forming clusters, then runs dynamic programming
#input -- n, number of nodes in random graph
#           c, criticality
def generate_test_graphs(O, threshold):
    count = 0
    do_while = True
    while do_while:
        count += 1
        C = create_cluster_graph(O, threshold)
        # C will be false if there are less than two nodes in the cluster graph. In this case attempt a new assignment of variables
        if C == False:
            reset_original_graph_data(O)
        else:
            do_while = False
        if count > MAX_ATTEMPTS_SASISFY_ASS_1:
            print("Parameters cannot feasibly satisfy parameters. Too few clusters are being generated (small number of nodes or very low criticality)")
            sys.exit()
    
    B = create_bipartite_from_cluster(C)
    # unmodified_B is used for general algorithms and calculating the payoffs of seeds picked from other algorithms (because there is no loss of info from the original graph for this bipartie)
    unmodified_B = B.copy()

    # cluster graph satisfy assumption 1. used for assumption one algorithms
    statisfy_assumption_one(B)
    C_sat_assume_one = create_cluster_from_bipartite(B)

    # remove cycles and satisfy assumption 1 (never need to node sat ass 1 if removing cycles) used for tree algorithms
    C_remove_cyc = C_sat_assume_one.copy()
    remove_cycles(C_remove_cyc)
    rejectingNodeDict.clear()
    B.clear()
    return C_remove_cyc, C_sat_assume_one, unmodified_B, count

"""
From each node in the nodeList, try to label its cluster. This will return 0 for many nodes, as they are labeled
from the (arbitrary) canonical node in its cluster.
We then select a seed set of up to (not always, depending on the composition of the graph) k source nodes.
"""
def create_cluster_graph(O, threshold):
    nodeList = O.nodes()
    clusterCount = 0
    G_cluster = nx.Graph()
    clearVisitedNodesAndDictionaries(O)
    # build the clusters
    for node_ID in nodeList:
        if (O.nodes[node_ID]['criticality'] < threshold) and (O.nodes[node_ID]['cluster'] == -1):
            accepting_in_cluster, rejecting_in_cluster = label_cluster(O, node_ID, clusterCount, threshold)
            make_cluster_node(G_cluster, clusterCount, accepting_in_cluster)
            clusterCount += 1
    # cannot have a cluster graph with no edges... ? (wq:try to understand this later)
    if len(G_cluster.nodes()) < 2:
        print("Less than 3 nodes in cluster graph")
        return False
    
    # decrement the cluster weight by the number of rejecting nodes that are exclusive to a cluster
    # wq:compare output of cluster graph from NIC code to this, to see if the same and more efficient?
    for clusterNum, rejNodes in rejectingNodeDict.items():
        #if DEBUG: print("rejecting nodes,", clusterNum, rejNodes)
        rejNodes_copy = rejNodes.copy()
        for clusterNum2, rejNodes2 in rejectingNodeDict.items():
            #if DEBUG: print("rejecting nodes 2,", clusterNum2, rejNodes2)
            if clusterNum != clusterNum2:
                rejNodes_copy = rejNodes_copy - rejNodes2
        #if DEBUG: print("Subtracting", len(rejNodes_copy), "cluster", clusterNum )
        G_cluster.nodes[clusterNum]['weight'] -= len(rejNodes_copy)
    make_cluster_edge(G_cluster, O)
    return G_cluster

"""
Perform BFS to label a cluster. Note how we called setVisitedFalse(O) upon finding an unvisited node.
The idea is that clusters are "contained" - every cluster is separated from every other cluster by walls of
rejecting nodes. You will never find another cluster, because if you could, it would simply be one cluster.
However, we need to set rejecting nodes to visited so that we don't count them for each accepting neighbor.
This, however, is not problematic, because we label all nodes from the cluster from a single canonical source node.
We will only reach this method from another cluster - whereupon we call setVisitedFalse(), clearing the rejecting
nodes, allowing us to again "contain" this new cluster.
"""
def label_cluster(O, source, clusterNumber, appeal):
    # wq:don't think this needs to be here. If 'cluster' = -1, this shouldn't be visited. Consider deleting after tests
    if O.nodes[source]['visited'] == False:
        setVisitedFalse(O)
        O.nodes[source]['visited'] = True
    else:
        return 0

    queue = []
    O.nodes[source]['cluster'] = clusterNumber
    queue.append(source)

    accepting_in_cluster = 1 # count yourself, you matter!
    rejecting_in_cluster = 0
    
    # BFS!
    while queue:
        start = queue.pop(0)

        for neighbor in nx.neighbors(O, start):
            if O.nodes[neighbor]['visited'] == False: # check if we've added a node to a cluster yet
                if O.nodes[neighbor]['criticality'] < appeal:
                    queue.append(neighbor)
                    O.nodes[neighbor]['cluster'] = clusterNumber
                    O.nodes[neighbor]['visited'] = True
                    accepting_in_cluster += 1
                    
                else:
                    rejecting_in_cluster += 1
                    # record rejecting nodes of this cluster
                    if clusterNumber not in rejectingNodeDict:
                        rejectingNodeDict[clusterNumber] = set()
                        rejectingNodeDict[clusterNumber].add(neighbor)
                    else:
                        rejectingNodeDict[clusterNumber].add(neighbor)
                    #MTI: Added this line to avoid revisiting this node from other accepting nodes within this cluster.
                    O.nodes[neighbor]['visited'] = True

    if clusterNumber not in rejectingNodeDict:
        if DEBUG: print("didnt see any neg nodes from cluster ", clusterNumber)

    return accepting_in_cluster, rejecting_in_cluster

#we defined a new cluster and are adding a node to the cluster graph, whose weight is the number of accepting nodes in that cluster
def make_cluster_node(C, clusterNum, weight):
    #print("---------MAKING NODE-----------", clusterNum)
    C.add_node(clusterNum)
    C.nodes[clusterNum]['weight'] = weight

"""
This method takes the rejecting node dictionary, which maps cluster number to rejecting nodes, and assigns a weight for each pair of clusters
that share rejecting nodes
@params:
    G_orig -> original graph
    G_cluster -> cluster graph whose edge is being labelled
"""
def make_cluster_edge(G_cluster, G_orig):
    if DEBUG: print("rej nodes dictioary", rejectingNodeDict)
    for clusterNum, rejNodes in rejectingNodeDict.items():
        for clusterNum2, rejNodes2 in rejectingNodeDict.items():
            if clusterNum >= clusterNum2:
                continue
            else:
                shared_rej_nodes = rejNodes.intersection(rejNodes2)
                intersection = []
                for i in shared_rej_nodes:
                    rej_node = -i
                    intersection.append(rej_node)
                weight = len(intersection)
                # create and wedge, with weight being the number of shared rejecting nodes between two clusters, as well
                # as the ID of the rejecting nodes
                if len(shared_rej_nodes) > 0:
                    G_cluster.add_edge(clusterNum, clusterNum2, weight=weight, rej_nodes=intersection)
    
    # add edges with no weight if cluster graph not connected
    components = nx.algorithms.components.connected_components(G_cluster)
    if DEBUG: print("Connected components: ", components)
    prev = -1
    for comp in components:
        if DEBUG: print("Component: ", comp)
        if prev == -1:
            prev = list(comp)
            if DEBUG: print("list is", list(comp))
            continue
        else:
            # connect disconnected componeents
            G_cluster.add_edge(prev[0], list(comp)[0], weight=0) #add arbitrary weight





"""
This code does work (mostly), however since creation of clusters requires BFS and reject nodes cannot be marked visited
(bc rejects can belong to multiple clusters) doing cluster -> bipartite -> (maybe sat assum 1, bipartite -> C) could be smarter
workflow
"""
def create_bipartite_from_original(O, threshold):
    nodeList = O.nodes()
    clusterCount = 0
    B = nx.DiGraph()
    clearVisitedNodesAndDictionaries(O)
    # build the clusters and add to bipartite graph
    for node_ID in nodeList:
        if (O.nodes[node_ID]['criticality'] < threshold) and (O.nodes[node_ID]['cluster'] == -1):
            add_cluster_and_corresponding_rejects(O, node_ID, clusterCount, threshold, B)
            clusterCount += 1
    # cannot have a cluster graph with no edges... ? (wq:try to understand this later)
    if len(B.nodes()) < 2:
        print("Less than 3 nodes in cluster graph")
        return False
    # consume reject into cluster (decrement weight by one) if a reject is exclusive to one cluster
    nodes_to_remove = set()
    for node in B.nodes():
        if B.nodes[node]['bipartite'] == 1 and B.degree[node] == 1:
            # get the cluster the reject is connected to
            connected_cluster = B.neighbors(node)
            # neighbors returns iterable, so iterate over the ONE variable in iterable
            for ID in connected_cluster:
                B.nodes[ID]['weight'] = B.nodes[ID]['weight'] - 1
                nodes_to_remove.add(node)
    B.remove_nodes_from(nodes_to_remove)
    return B


"""
Helper for original to bipartite. Similar to lable clusters
"""
def add_cluster_and_corresponding_rejects(O, source, clusterNum, appeal, B):
    B.add_node(clusterNum)
    B.nodes[clusterNum]['weight'] = 'not assigned'
    B.nodes[clusterNum]['bipartite'] = 0

    queue = []
    O.nodes[source]['cluster'] = clusterNum
    queue.append(source)
    accepting_in_cluster = 1 # count yourself, you matter!
    
    # BFS!
    while queue:
        start = queue.pop(0)
        for neighbor in nx.neighbors(O, start):
            if O.nodes[neighbor]['visited'] == False: # check if we've added a node to a cluster yet
                if O.nodes[neighbor]['criticality'] < appeal:
                    # assign node to a cluster, increment weight count and add to queue for BFS
                    queue.append(neighbor)
                    O.nodes[neighbor]['cluster'] = clusterNum
                    accepting_in_cluster += 1
                    O.nodes[neighbor]['visited'] = True # note that we only mark visited if accepting node, because we want to be able to visit rejects again!
                else:
                    # connect reject w/ edge to cluster
                    rej_label = "r-" + str(neighbor)
                    if not B.has_node(rej_label):
                        B.add_node(rej_label)
                        B.nodes[rej_label]['bipartite'] = 1
                    B.add_edge(rej_label, clusterNum)
    # assign weight as number of accepting in the cluster
    B.nodes[clusterNum]['weight'] = accepting_in_cluster

'''
Create a cluster graph from bipartite graph, useful when making graph satisfy assumption 1
'''
def create_cluster_from_bipartite(B):
    C = nx.Graph()
    rejects = []
    for node in B.nodes():
        # add clusters from bipartite graph directly into cluster graph
        if B.nodes[node]['bipartite'] == 0:
            C.add_node(node)
            C.nodes[node]['weight'] = B.nodes[node]['weight']
        # record list of reject nodes in bipartite graph
        else:
            rejects.append(node)

    for reject in rejects:
        # add edges between clusters based on shared rejecting nodes in bipartite graph
        for neighbor_1 in B.neighbors(reject):
            for neighbor_2 in B.neighbors(reject):
                # if neighbor_1 larger than neighbor_2, this edge combination has been considered
                if neighbor_1 >= neighbor_2:
                    continue
                # if the edge between these two does not exist, create it
                elif not C.has_edge(neighbor_1, neighbor_2):
                    C.add_edge(neighbor_1, neighbor_2, weight=1, rej_nodes=[reject[1:]])
                # the edge between these nodes does exist, therefore update weight of edge and shared reject nodes
                else:
                    C[neighbor_1][neighbor_2]['weight'] = C[neighbor_1][neighbor_2]['weight'] + 1
                    rejects = C[neighbor_1][neighbor_2]['rej_nodes']
                    rejects.append(reject[1:])
                    C[neighbor_1][neighbor_2]['rej_nodes'] = rejects

    # add edges with no weight if cluster graph not connected
    connected_sections = nx.algorithms.components.connected_components(C)
    prev = -1
    for component in connected_sections:
        if prev == -1:
            prev = list(component)
            continue
        else:
            # connect disconnected componeents
            C.add_edge(prev[0], list(component)[0], weight=0) #add arbitrary weight
    return C

'''
Remove edges in bipartite graph until no cluster has more than two shared rejecting nodes

@params
B -> Bipartite graph
'''
def statisfy_assumption_one(B):
    for node in B.nodes():
        if B.nodes[node]['bipartite'] == 1:
            number_clusters_sharing_reject = B.degree[node]
            if number_clusters_sharing_reject > 2:
                shared_clusters = [(node, cluster) for cluster in B.neighbors(node)]
                # remove all edges from reject but two
                B.remove_edges_from(shared_clusters[:-2])

'''
Check if a cluster graph has more than two shared rejecting nodes.
This is assumption 1, and is required for all tree algorithms and algorithms that requre assumption 1

@params
C -> Cluster graph
@returns
Boolean -> True if has more than two, False otherwise
'''
def has_more_two_shared_rejects(C):
    sharedRejectingNodes = {}
    for clusterNum, rejNodes in rejectingNodeDict.items():
        for rejNode in rejNodes:
            if rejNode in sharedRejectingNodes:
                sharedRejectingNodes[rejNode] += 1
                if sharedRejectingNodes[rejNode] > 2:
                    if DEBUG: print(sharedRejectingNodes[rejNode], rejNode, sharedRejectingNodes)
                    if DEBUG: print("RETURNING FALSE")
                    return True
            else:
                sharedRejectingNodes[rejNode] = 1
        if DEBUG: print("\nMap rejecting nodes to clusters:\n", sharedRejectingNodes)
    return False

'''
Remove cycles from a cluster graph via max spanning tree

@params
C -> Cluster graph
'''
def remove_cycles(C):
    if DEBUG: print("IN REMOVE CYCLES")
    try:
        nx.find_cycle(C)
        mst = tree.maximum_spanning_edges(C, algorithm="kruskal", data=False)
        edgelist = list(mst)
        sorted(sorted(e) for e in edgelist)
        if DEBUG: print("EDGE LIST-------->", edgelist)
        edges = C.edges()
        if DEBUG: print("ORIGINAL EDGES:", edges)
        for e in edges:
            if e not in edgelist:
                C.remove_edge(e[0],e[1])
        if DEBUG: print("NEW EDGES: ", C.edges())

    except nx.exception.NetworkXNoCycle:
        pass

    # throw error of cycles are not removed
    try:
        nx.find_cycle(C)
        print("ERROR: Cycles not removed")
        sys.exit()
    except nx.exception.NetworkXNoCycle:
        pass


#clear dictionaries for next graph to test
def clearVisitedNodesAndDictionaries(O):
    setVisitedFalse(O)
    rejectingNodeDict.clear()


# Set node attributes for the original graph, where each node is an individual and edges represent 
# connection in the network
def reset_original_graph_data(O):
    for node_ID in O.nodes():
        O.nodes[node_ID]["visited"] = False
        O.nodes[node_ID]['criticality'] = random.uniform(0, 1)
        O.nodes[node_ID]["cluster"] = -1

def setVisitedFalse(O):    
    nodeList = O.nodes()
    for node_ID in nodeList:
        O.nodes[node_ID]["visited"] = False

"""
Convert a cluster graph to a bipartite graph.
The bipartite graph is structured so each cluster (node)
in the cluster graph can be visualized as a node on the RHS
of the bipartite graph and each rejecting node encompassed in the edges
of the cluster graph is a node on the LHS of the bipartite graph.
The graph is directed, so we have an edge from each rejecting node 
to the cluster(s) it is connected to.
"""
def create_bipartite_from_cluster(C):
    B = nx.DiGraph()
    #look at every edge in the cluster graph and create nodes in bipartite
    for edge in C.edges.data():
        #check if either endpoint of the edge has been added to the bipartite
        if not B.has_node(edge[0]):
            B.add_node(edge[0])
            B.nodes[edge[0]]['weight'] = C.nodes[edge[0]]['weight']
            B.nodes[edge[0]]['bipartite'] = 0
        if not B.has_node(edge[1]):
            B.add_node(edge[1])
            B.nodes[edge[1]]['weight'] = C.nodes[edge[1]]['weight']
            B.nodes[edge[1]]['bipartite'] = 0
        #not all edges have a rej_nodes attribute. If it does, create a node in the bipartite
        #for that rej node (if we havent already) and an edge from that rej node to the clusters
        #it is attached to.
        try:
            for rej_node in edge[2]['rej_nodes']: #look at all rejecting nodes
                rej_label = "r" + str(rej_node) # create a different lable from the cluster nodes so that they can never be confused for eachother
                if not B.has_node(rej_label):
                    B.add_node(rej_label)
                    # don't need weights here!
                    # B.nodes[rej_label]['weight'] = rej_node
                    B.nodes[rej_label]['bipartite'] = 1
                if not B.has_edge(rej_label, edge[0]):
                    B.add_edge(rej_label, edge[0])
                if not B.has_edge(rej_label, edge[1]):
                    B.add_edge(rej_label, edge[1])            
        except KeyError:
            continue
    return B