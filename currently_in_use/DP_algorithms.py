import networkx as nx
import itertools
import create_clusters as cc
import brute_force as bf

"""
Driver function for running dynamic programming

@params:
    G --> graph for which we are choosing seeds
    k --> number of seeds to pick
"""
def runRecursiveDP(G, k):
    #makeMatrix(G, G.number_of_nodes())

    storePayoff = [ [ [None] * (k+1) for _ in range(G.number_of_nodes())] for _ in range(2)]

    witness = [ [ [None] * (k+1) for _ in range(G.number_of_nodes())] for _ in range(2)]
    nodes_tup = sorted(G.degree, key=lambda x: x[1], reverse=True) #sort by highest degree node
    #print("root is", nodes_tup[0][0]) #take top degree node as root
    root = nodes_tup[0][0]
    tree = nx.bfs_tree(G, root)
    recursiveDP(G, tree, k, root, storePayoff, witness)
    cc.clearVisitedNodesAndDictionaries(G)
    return storePayoff[0][root][k], storePayoff[1][root][k]

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
def recursiveDP(G, tree, k, source, storePayoff, witness):
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
                continue
                #print("debugging") #IGNORE, used for debugging
           # print(p)
            sum_so_far = 0
            allocation = {}
            for i in range(0, num_children):
               # print("p is", p[i])
                allocation[neighbors_list[i]] = p[i] # set our allocation of seeds to current partition
                recursiveDP(G, tree, p[i], neighbors_list[i], storePayoff, witness) #recurse on current allocation
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
        #if source == 1:
            #print("debugging")
        
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
                recursiveDP(G, tree, p[i], neighbors_list[i], storePayoff, witness)
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
THIS IS THE MOST USEFUL FUNCTION IN THE ENTIRE FILE

Here we are running dynamic programming recursively to choose the best k clusters in the graph to seed.

@params:
    G --> the cluster graph we are seeding from
    tree --> the tree decomposition graph we are recursing on
    k --> the number of clusters we are seeding
    source --> the starting bag in the tree decomposition (highest degree)
    storePayoff --> the payoff matrix
    rejecting --> used to keep track of which rejecting nodes we have accounted for already

try doing all subsets of nodes in the bag --> 
"""
def tree_decomp_DP(G, tree, k, source, storePayoffCluster, storePayoffTree, rejecting):
    #TRUE is 0 and FALSE is 1 for storePayoff
    #print("source is:", source)
    precomputed_0 = precomputed_1 = False
    #print("root is",source)
    if k <= 0: #base case, meaning we have no seeds
        #print("no seeds")
        return 
    if tree.out_degree(source) == 0: #base case, meaning we are at a leaf node
       # print("at leaf node")
        for node in source:
            storePayoffCluster[0][node][k] = G.nodes[node]['weight']
            storePayoffCluster[1][node][k] = 0
        #if k >= 1:
        return 

    neighbors_list = [] # get all the children of the current bag
    for i in list(tree.out_edges(source)):
        neighbors_list.append(i[1])

    #print(neighbors_list, "NEIGHBORS LIST")
    num_children = len(neighbors_list) #used to partition seeds
    partitions_list = list(partitions(k, num_children)) #seed all k seeds among the child nodes
    maxSum = float("-inf")
    opt_allocation = None
    opt_take_child = None
    for p in partitions_list: #loop through partitions of seeds
        rejecting = []
        print(p)
        sum_so_far = 0
        allocation = {}
        j = 0
        for child in neighbors_list: #allocate seeds among the children
            intersection = set() #get the intersection between the child bag and parent bag
            #print("child is:", child)
            for node in child:
                if node not in source:
                    intersection.add(node)
            #print("intersection between the nodes:", intersection)
            tree_decomp_DP(G, tree, p[k], child, storePayoffCluster, storePayoffTree, rejecting) #recurse on current allocation
            for node in intersection: #at this stage, we have to check if we have already computed the payoff for this node, in which case, we don't want to count it again
                if storePayoffCluster[0][node][p[k]] != None: # if we have already computed the payoff for taking this child
                    continue
                else:
                    edge_weight = 0
                    neighbors = list(G.neighbors(node)) # get the edge weights, don't want to double count
                    for neighbor in neighbors: # we loop through all the neighbors and look to see if we have already accounted for this rej node
                        print(G.get_edge_data(node, neighbor))
                        for rejecting_nodes in G.get_edge_data(node, neighbor)['data']:
                            if rejecting_nodes in rejecting:
                                continue
                            else:
                                rejecting.append(rejecting_nodes)
                                edge_weight += G.get_edge_data(node, neighbor)['weight']
                #print("edge weight is:", edge_weight)
                #print("node:", node, "num seeds:", p[k])
                if p[k] == 0:
                    continue
                #IMPORTANT!!!!!!! If the payoff for taking the child minus the weight of the negative edge is GREATER than the payoff for leaving the child, take it
                if storePayoffCluster[0][node][p[j]] - edge_weight >= storePayoffCluster[1][node][p[j]]:
                #print("take child:", neighbors_list[i])
                    sum_so_far += storePayoffCluster[0][node][p[j]] - edge_weight
                else:
                    sum_so_far += storePayoffCluster[1][node][p[j]]
            # if this partition is better than maxSum, take it!
            if sum_so_far > maxSum:
                #print(sum_so_far, "MAX SUM")
                maxSum = sum_so_far
                opt_allocation = allocation
        if source == 1:
            continue
            #print("debugging")
        
        #populate the table for leaving source
        storePayoffTree[j][k] = maxSum
        j+=1

"""
Stars and bars problem

"""
def partitions(n, k): #stars and bars, k subtrees and n seeds to allocate among them
    for c in itertools.combinations(range(n+k-1), k-1):
        yield [b-a-1 for a, b in zip((-1,)+c, c+(n+k-1,))]