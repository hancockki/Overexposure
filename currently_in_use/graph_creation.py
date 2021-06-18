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

#method to test whether our input graph is correctly forming clusters, then runs dynamic programming
#input -- n, number of nodes in random graph
#           c, criticality
def generate_test_graphs(O, threshold, do_remove_cycles, do_assumption_1):
    do_while = True
    while do_while:
        C = create_cluster_graph(O, threshold)
        if do_remove_cycles:
            remove_cycles(C)
            do_while = False
        # end the while loop after one run if assumption 1 is not enforced
        if do_assumption_1:
            # if the cluster graph has more than two shared rejecting nodes, violating assumption 1
            # generate a new assignment of node criticalites to original graph
            # (this will lead to the formation of a new cluster graph)
            if has_more_two_shared_rejects(C):
                reset_original_graph_data(O)
            else:
                do_while = False
        else:
            do_while = False
    B = create_bipartite_from_cluster(C)
    rejectingNodeDict.clear()
    return C, B

"""
From each node in the nodeList, try to label its cluster. This will return 0 for many nodes, as they are labeled
from the (arbitrary) canonical node in its cluster.
We then select a seed set of up to (not always, depending on the composition of the graph) k source nodes.
"""
def create_cluster_graph(O, threshold):
    nodeList = O.nodes()
    clusterCount = 0
    G_cluster = nx.Graph()
    # build the clusters
    for node_ID in nodeList:
        if (O.nodes[node_ID]['criticality'] < threshold) and (O.nodes[node_ID]['cluster'] == -1):
            accepting_in_cluster, rejecting_in_cluster = label_cluster(O, node_ID, clusterCount, threshold)
            make_cluster_node(G_cluster, clusterCount, accepting_in_cluster)
            clusterCount += 1
    # cannot have a cluster graph with no edges... ? (wq:try to understand this later)
    if len(G_cluster.nodes()) < 2:
        print("NOT GONNA WORK")
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
            G_cluster.add_edge(prev[0], list(comp)[0], weight=0) #add arbitrary weight

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


# #clear dictionaries for next graph to test
# def clearVisitedNodesAndDictionaries(O):
#     setVisitedFalse(O)
#     rejectingNodeDict.clear()


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
        if not B.has_node(edge[1]):
            B.add_node(edge[1])
            B.nodes[edge[1]]['weight'] = C.nodes[edge[1]]['weight']
        #not all edges have a rej_nodes attribute. If it does, create a node in the bipartite
        #for that rej node (if we havent already) and an edge from that rej node to the clusters
        #it is attached to.
        try:
            for rej_node in edge[2]['rej_nodes']: #look at all rejecting nodes
                rej_label = "r" + str(rej_node) # create a different lable from the cluster nodes so that they can never be confused for eachother
                if not B.has_node(rej_label):
                    B.add_node(rej_label)
                    B.nodes[rej_label]['weight'] = rej_node
                if not B.has_edge(rej_label, edge[0]):
                    B.add_edge(rej_label, edge[0])
                if not B.has_edge(rej_label, edge[1]):
                    B.add_edge(rej_label, edge[1])            
        except KeyError:
            continue
    return B