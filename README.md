# Overview of files

Here I will explain what each function does in some more detail, for the dynamic_programming_cleaned.py file. This file is where the most recent work is. 
Just for a summary, really the most important function right now is recursiveDP...its the most recent algorithm

## How do I understand the file?
This file has two main goals related to this project
1. Given a graph in which each node is either an accepting or rejecting node, create a **clustered** graph based on this, in which each cluster contains all the accepting nodes reachable from a given accepting node. Think of it as having a wall of rejecting nodes around it. The file takes a graph, labels the clusters, and produces a new graph in which each node is a CLUSTER of nodes from the original graph (weight of which is number of accepting nodes in the cluser). Each edge has a weight equalling the number of rejecting nodes between each cluster.
2. The second step of the file is choosing the optimal seed set based on the cluster graph. the logic here is that if we seed one individual in a given cluster, the entire cluster will end up accepting since they are connected.

### Note on 1.
The first step is incomplete, as making the cluster graph only works when your input is a tree. this is still being debugged

## Overview of methods in the main file
### createClusterGraph -- 
    This simple function creates a tree graph using the built-in function nx.random_tree, with the number of Nodes (clusters) as the argument. 
    Weights for each Node are randomly assigned, as well as weights for each edge. The MOST important thing to remember for this function is that
    we are creating a CLUSTER graph, so each Node is actually a cluster of nodes, and the weight of the Node is the number of nodes in the cluster.
    
    The weight of each edge is the number of shared nodes between two clusters.
    This function is used to test the algorithms that work on clustered graphs. Thus, it is important to remember with this function that it is a
    completely arbitrary graph, and the issue of going from the original graph to the clustered graph is not considered at all.
    Returns the cluster graph.

### setAllNodeAttributes --
    Basic function. Creates random weights for each node in a generated graph, and a value of -1 for the cluster indicates that the node has not 
    yet been assigned a cluster.
    Does not return anything.

### setVisitedFalse --
    Node attribute 'visited' is set to false. Used to reset the graph in each iteration of BFS where we need to check and see if we have
    already visited that node in our BFS.
    Does not return anything.

### labelClusters --
    Perform BFS to label a cluster. Note how we called setVisitedFalse(G) upon finding an unvisited node.
    The idea is that clusters are "contained" - every cluster is separated from every other cluster by walls of
    rejecting nodes. You will never find another cluster, because if you could, it would simply be one cluster.
    However, we need to set rejecting nodes to visited so that we don't count them for each accepting neighbor.
    This, however, is not problematic, because we label all nodes from the cluster from a single canonical source node.
    We will only reach this method from another cluster - whereupon we call setVisitedFalse(), clearing the rejecting
    nodes, allowing us to again "contain" this new cluster.

    This function returns:
         acceptingInThisCluster, the number of accepting nodes in the newly created cluster
         rejecting, the number of rejecting nodes in the cluster
         clusterNumber, the number of the cluster created

### buildClusteredSet --
    This function basically is the driver for labelClusters, adding new clusters until every node in the graph is in some cluster
    (even if the cluster contains a single node)
    Then, once each cluster has been created, a NEW cluster graph is made, using the funtions make_Cluster_node and make_Cluster_edge.
    The difficulties with this are:
        A) identifying rejecting nodes that are not shared with other clusters (in which case this must be subtracted from the cluster weight, 
        since the rejecting nodes won't be encapsulated in the edge)
        B) determining the number of shared rejecting nodes between two clusters
        C) the case where there is more than one rejecting node between two clusters (in which case two clusters are connected but do not have
        any 'shared' rejecting nodes)
        D) We might encounter "triangles," where cluster A has shared rejecting nodes with cluster B, cluster B has shared rejecting nodes with
        cluster C, and cluster C has shared rejecting nodes with cluster A.

    Returns the cluster graph

### makeMatrix --
    Simply construct a matrix to be used in linear programming, where the main diagonal of the matrix is the cluster, and the edges are the other 
    values. Used in a separate mathematica program.

### computeNegPayoff --
    Given a node, adds the weights of every outgoing edge. This corresponds to the number of rejecting nodes connected to the cluster, aka the negative
    payoff.
    Subtracts this value from the node weight and returns that value.

### DP --
    Picks k clusters to seed using table-based dynamic programming.
    We do a topological sort of the nodes in the graph and start our search from the bottom of this topological sort (leaves of the tree)
    Thus, we are building up to the k best clusters, and seeding one individual in that cluster.
    The main struggle here that still needs to be worked out is how we are ordering DP, since the order we consider clusters can greatly
    impact the outcome. Returns the payoff of seeding those clusters and the cluster numbers. This DP algorithm does not consider subtrees or different
    orderings of picking clusters, unlike the below algorithms.
    returns the payoff and chosen clusters.

### subsetDP --
    Basically does the same as the above algorithm but only on a SUBSET of nodes (ie, starting DP from somewhere partially up the tree)

### recursiveDP --

    THIS IS THE CURRENTLY USED algorithm
    This algorithm is the most recent improvement since it actually uses recursion to solve the problem.
    The key inequalities in the algorithm are outlined in the Overleaf pseudocode.
    Does not return anything, rather builds a memoization table with all the payoffs
    
### DP_Improved --

    Improve dynamic programming by trying to figure out which is the best ordering of seeding. We look at all the neighbors of the root node and
    identify subtrees using BFS from each of the children of the root node. thus the number of subtrees is the same as the number of children of 
    the root node. We could improve this by trying to recursively test subtrees rather than having it be fixed.
    We partition the k seeds among the n children in every possible way, and compute the payoff by seeding the subtrees with every combination and 
    choosing the highest payoff. NOTE that this is SLOW if we have a lot of children and/or a lot of seeds.

    Basically, this helps us avoid the issue of "spreading" the seeds too thin--maybe with regular DP seeding a lot of nearby clusters is best, but
    because we look at all the leaves first, we may not ever see that cluster of clusters pop up.

    Returns maxVal, the max payoff from all the partitions.

bfs --
    Does simple bfs, used in DP_improved to find subtrees.

### make_Cluster_node --
    Simple. Creates a cluster Node with the weight being the number of accepting nodes in the cluster minus the number of rejecting nodes that are not
    shared with another cluster.

### make_cluster_edge --
    More complex. Creates an edge between two clusters, BUT we have to correctly compute the number of shared rejecting nodes between two clusters. Also there's
    a big issue where if cluster A shares a node with cluster B and cluster C (the same node), then we are creating an edge A-B and an edge A-C, so the
    rejecting node is being considered TWICE when it should only be considered ONCE. We need to think more about how to compute this edge weight.

### college_Message --
    Sort of useless at this point. Uses a built in network from networkx as a test for the DP algorithms

### testOriginaltoCluster --
    Tests how well we are converting the original graph to a cluster graph.


## CURRENT PROBLEM:
How do we properly construct the cluster graph, when A) clusters may be separated by more than one node, and B) multiple clusters share the same rejecting nodes
We also previously discussed C) the issue of triangularization

I propose trying to implement a model where instead of the edges having numerical weights, the edges have vectors which are represented as a list of rejecting nodes.
Then, when we seed a given cluster, we "check" off all those rejecting nodes, so when we choose a cluster with overlapping rejecting nodes,
we know which ones to not count. 

However, this bears another problem: we have to completely change our dynamic programming algorithm, which is based on numerical edge weights. How can we alter DP to account for this?
    A) Try all possible combinations, perhaps some sort of backtracking algorithm. 
    B) We have to consider only immediate neighbors in numRejecting, as if there is more than one rejecting node separating two clusters, the rejecting nodes in the "middle" 
    will never be reached.

My question now is:
    What should we focus on? Restricting the problem and getting a working optimization algorithm, or trying to generalize the problem?


Tree Case (original graph):
    You start with a tree. Only scenario that leads to a complete cluster graph is when a lot of branches share a single rejecting node. Can we delete zeo weight
    edge without any impact on optimal solution? Try to prove by contradiction for the tree case
    The cluster graph is not necessarily a tree


Tree Case (cluster graph):
    Prove the complexity is NP-hard (analyze running time)
    Prove that it gives an optimal solution
    Maybe the bug is actually in the linear program
    Show some smaller examples
    Save them in a file --> append the code to save files uniquely












