import networkx as nx
import itertools

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
    # wq: commented this out because I don't think it's necessary. Revisit to be sure!
    # cc.clearVisitedNodesAndDictionaries(G)
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
        #partitions_list = list(partitions(k, num_children)) #seed all k seeds among the child nodes
        partitions_list = [[int(k/num_children) for i in range(num_children)]]
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
Stars and bars problem

"""
def partitions(n, k): #stars and bars, k subtrees and n seeds to allocate among them
    for c in itertools.combinations(range(n+k-1), k-1):
        yield [b-a-1 for a, b in zip((-1,)+c, c+(n+k-1,))]


def greedyDP(G, i, k): #doesn't consider subtrees
    #This is different since we are considering each node's weight in the graph to be the number of accepting nodes in a given cluster
    #i = number_of_nodes(G)
    #k = number of seeds
    storePayoff = [['a'] * i for _ in range(k)] #store payoff
    storeSeeds = [[[]] * i for _ in range(k)] #store seeds at each stage
    tree = nx.bfs_tree(G, 1)
    for numSeeds in range(0,k): #bottom up DP
        nodes = list(reversed(list((nx.topological_sort(tree))))) #look at nodes in reverse topological order
        for j in range(0,i): 
            if j == 0 and numSeeds == 0: #first entry
                #breakpoint()
                storeSeeds[numSeeds][j] = [nodes[j]]
                nodeWeight = computeNegPayoff(G, nodes[j])
                storePayoff[numSeeds][j] = nodeWeight
                #print("first entry,", storePayoff)

            elif numSeeds == 0: #if there is only one seed to consider, aka first row
                last = storePayoff[numSeeds][j-1]
                nodeWeight = computeNegPayoff(G, nodes[j])
                if nodeWeight > last:
                    storePayoff[numSeeds][j]=nodeWeight
                    storeSeeds[numSeeds][j] = [nodes[j]]
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
                nextGuess = computeNegPayoff(G, nodes[j]) + last
                for lastNodes in storeSeeds[numSeeds-1][j-1]: #dont want to double count edges!
                    neighbors = nx.neighbors(G, lastNodes)
                    for neighbor in neighbors:
                        if neighbor == nodes[j]:
                            add = G.get_edge_data(lastNodes, nodes[j]) #neighbor of new node is current node
                            add = add['weight']
                            nextGuess += add
                lastEntry = storePayoff[numSeeds][j-1] #left entry
                lastEntryUp = storePayoff[numSeeds-1][j]
                tup = [(storeSeeds[numSeeds][j-1], lastEntry), (storeSeeds[numSeeds-1][j], lastEntryUp), (storeSeeds[numSeeds-1][j-1], nextGuess)] # , (storeSeeds[numSeeds-1][j-1], last)
                tup.sort(key = lambda x: x[1])
                nextList = tup[-1][0][:]
                storeSeeds[numSeeds][j] = nextList
                storePayoff[numSeeds][j] = tup[-1][1]
                if tup[-1][0] == storeSeeds[numSeeds-1][j-1]:
                    storeSeeds[numSeeds][j].append(nodes[j])
    maxVal = storePayoff[k-1][i-1]
    for j in range(0,k):
        if storePayoff[j][i-1] > maxVal:
            maxVal = storePayoff[j][i-1]
    return (maxVal, storeSeeds[j][i-1])

# def greedyDP(G, i, k): #doesn't consider subtrees
#     #This is different since we are considering each node's weight in the graph to be the number of accepting nodes in a given cluster
#     #i = number_of_nodes(G)
#     #k = number of seeds
#     storePayoff = [['a'] * i for _ in range(k)] #store payoff
#     storeSeeds = [[[]] * i for _ in range(k)] #store seeds at each stage
#     tree = nx.bfs_tree(G, 1)
#     for numSeeds in range(0,k): #bottom up DP
#         nodes = list(reversed(list((nx.topological_sort(tree))))) #look at nodes in reverse topological order
#         for j in range(0,i): 
#             if j == 0 and numSeeds == 0: #first entry
#                 negative_edges = {}
#                 negative_weight = store_edges(negative_edges, nodes[j], G)
#                 # [neg, pos, edges, seed set]
#                 storeSeeds[numSeeds][j] = [negative_weight, G.nodes[nodes[j]]['weight'], negative_edges, nodes[j]]
#                 storePayoff[numSeeds][j] = G.nodes[nodes[j]]['weight'] - negative_weight
#                 #print("first entry,", storePayoff)

#             elif numSeeds == 0: #if there is only one seed to consider, aka first row
#                 last = storePayoff[numSeeds][j-1]
#                 negative_edges = {}
#                 negative_weight = store_edges(negative_edges, nodes[j], G)
#                 nodeWeight = G.nodes[nodes[j]]['weight'] - negative_weight
#                 if nodeWeight > last:
#                     storePayoff[numSeeds][j]=nodeWeight
#                     storeSeeds[numSeeds][j] = [negative_weight, G.nodes[nodes[j]]['weight'], negative_edges, nodes[j]]
#                 else:
#                     storePayoff[numSeeds][j]= last
#                     table = storeSeeds[numSeeds][j-1]
#                     copy = table[:]
#                     storeSeeds[numSeeds][j] = copy
#                 #print("num seeds 0",storePayoff)
#             elif j == 0: #we only consider first node, so its simple
#                 storePayoff[numSeeds][j] = storePayoff[numSeeds - 1][j]
#                 storeSeeds[numSeeds][j] = storeSeeds[numSeeds - 1][j][:]
#             else: #where DP comes in
#                 last = storeSeeds[numSeeds-1][j-1] #diagonal-up entry
#                 next_pos_weights = last[1]
#                 next_neg_weights = last[2]
#                 next_edges = last[3]
#                 next_seeds = last[4]
#                 nextGuess = computeNegPayoff(G, nodes[j]) + last
#                 for lastNodes in storeSeeds[numSeeds-1][j-1]: #dont want to double count edges!
#                     neighbors = nx.neighbors(G, lastNodes)
#                     for neighbor in neighbors:
#                         if neighbor == nodes[j]:
#                             add = G.get_edge_data(lastNodes, nodes[j]) #neighbor of new node is current node
#                             add = add['weight']
#                             nextGuess += add
#                 lastEntry = storePayoff[numSeeds][j-1] #left entry
#                 lastEntryUp = storePayoff[numSeeds-1][j]
#                 tup = [(storeSeeds[numSeeds][j-1], lastEntry), (storeSeeds[numSeeds-1][j], lastEntryUp), (storeSeeds[numSeeds-1][j-1], nextGuess)]
#                 tup.sort(key = lambda x: x[1])
#                 nextList = tup[-1][0][:]
#                 storeSeeds[numSeeds][j] = nextList
#                 storePayoff[numSeeds][j] = tup[-1][1]
#                 if tup[-1][0] == storeSeeds[numSeeds-1][j-1]:
#                     storeSeeds[numSeeds][j].append(nodes[j])
#     maxVal = storePayoff[k-1][i-1]
#     for j in range(0,k):
#         if storePayoff[j][i-1] > maxVal:
#             maxVal = storePayoff[j][i-1]
#     return (maxVal, storeSeeds[j][i-1])

# def store_edges(negative_edge_set, node, G):
#     neighbors = nx.neighbors(G, node)
#     negative_weights = 0
#     for neighbor in neighbors:
#         edge = ""
#         if node < neighbor:
#             edge = str(node) + "," + str(neighbor)
#         else:
#             edge = str(neighbor) + "," + str(node)
#         # if this edge has not been considered, add to negative calculations
#         if not negative_edge_set.contains(edge):
#             incedent_edge_weight = G.get_edge_data(node, neighbor)['weight']
#             negative_weights += incedent_edge_wight
#             negative_edge_set.add(edge)
#     #print("node weight is:", nodeWeight)
#     return negative_weights

def knapsack(G, i, k): #doesn't consider subtrees
    #This is different since we are considering each node's weight in the graph to be the number of accepting nodes in a given cluster
    #i = number_of_nodes(G)
    #k = number of seeds
    storePayoff = [['x'] * (k + 1) for _ in range(i)] #store payoff
    storeSeeds = [[[]] * (k + 1) for _ in range(i)] #store seeds at each stage
    tree = nx.bfs_tree(G, 1)
    for j in range(0,i): #bottom up DP
        nodes = list(reversed(list((nx.topological_sort(tree))))) #look at nodes in reverse topological order
        for numSeeds in range(0,k + 1): 
            if numSeeds == 0: #when k = 0, no payoff or selection
                storePayoff[j][numSeeds] = 0
                storeSeeds[j][numSeeds] = []
            elif j == 0: #we only consider first node, so its simple
                nodeWeight = computeNegPayoff(G, nodes[j])
                storePayoff[j][numSeeds] = nodeWeight
                storeSeeds[j][numSeeds] = [nodes[j]]
            else: #where DP comes in
                last = storePayoff[j-1][numSeeds-1] #diagonal-up entry
                take_node_payoff = computeNegPayoff(G, nodes[j]) + last
                for lastNodes in storeSeeds[j-1][numSeeds-1]: #dont want to double count edges!
                    neighbors = nx.neighbors(G, lastNodes)
                    for neighbor in neighbors:
                        if neighbor == nodes[j]:
                            add = G.get_edge_data(lastNodes, nodes[j]) #neighbor of new node is current node
                            add = add['weight']
                            take_node_payoff += add
                no_take_payoff = storePayoff[j-1][numSeeds]
                if take_node_payoff >= no_take_payoff:
                    storePayoff[j][numSeeds] = take_node_payoff
                    storeSeeds[j][numSeeds] = storeSeeds[j-1][numSeeds-1][:]
                    storeSeeds[j][numSeeds].append(nodes[j])
                else:
                    storePayoff[j][numSeeds] = no_take_payoff
                    storeSeeds[j][numSeeds] = storeSeeds[j-1][numSeeds][:]
    # f = open("make_matrix.txt", "a")
    # f.write("\n  regular DP payoff: " + str(storePayoff))
    # f.write("\n with seeds: " + str(storeSeeds))
    maxVal = storePayoff[k-1][i-1]
    for j in range(0,k):
        if storePayoff[j][i-1] > maxVal:
            maxVal = storePayoff[j][i-1]
    return (maxVal, storeSeeds[j][i-1])

def computeNegPayoff(G, nodeNum):
    nodeWeight = G.nodes[nodeNum]['weight']
    negPayoff = nx.neighbors(G, nodeNum)
    for negNode in negPayoff:
        add = G.get_edge_data(nodeNum, negNode)
        add = add['weight']
        nodeWeight -= add
    #print("node weight is:", nodeWeight)
    return nodeWeight