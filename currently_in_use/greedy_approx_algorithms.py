import networkx as nx

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
Compute the payoff for the greedy algorithm.
Must have assumption 1
"""
def computePayoffGreedy(G, k_highest):
    payoff = 0
    nodes_counted = []
    for payoff_tuple in k_highest:
        cur_node = payoff_tuple[1]
        node_payoff = computeNegPayoff(G, cur_node)
        for node in nodes_counted: #dont want to double count edges!
            neighbors = nx.neighbors(G, node)
            for neighbor in neighbors:
                if neighbor == cur_node:
                    add = G.get_edge_data(node, cur_node) #neighbor of new node is current node
                    add = add['weight']
                    node_payoff += add
                    #print("didn't double count!")
        nodes_counted.append(cur_node)
        payoff += node_payoff
    return payoff

"""
Loop through all nodes in the graph, pick the k highest clusters to be our seed set.
Most basic greedy approach to this problem.
"""
def kHighestClusters(G, k):
    num_nodes = G.number_of_nodes()
    weights_list = []
    for i in range(num_nodes):
        weight = G.nodes[i]['weight']
        weights_list.append((weight, i))
    weights_list = sorted(weights_list, reverse=True)
    #print(weights_list)
    payoff = computePayoffGreedy(G, weights_list[0:k])
    return payoff, weights_list[0:k]

def greedyDP(G, i, k): #doesn't consider subtrees
    #This is different since we are considering each node's weight in the graph to be the number of accepting nodes in a given cluster
    #i = number_of_nodes(G)
    #k = number of seeds
    storePayoff = [[0] * i for _ in range(k)] #store payoff
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
                tup = [(storeSeeds[numSeeds][j-1], lastEntry), (storeSeeds[numSeeds-1][j], lastEntryUp), (storeSeeds[numSeeds-1][j-1], nextGuess), (storeSeeds[numSeeds-1][j-1], last)]
                tup.sort(key = lambda x: x[1])
                nextList = tup[-1][0][:]
                storeSeeds[numSeeds][j] = nextList
                storePayoff[numSeeds][j] = tup[-1][1]
                if tup[-1][0] == storeSeeds[numSeeds-1][j-1]:
                    storeSeeds[numSeeds][j].append(nodes[j])
    f = open("make_matrix.txt", "a")
    f.write("\n  regular DP payoff: " + str(storePayoff))
    f.write("\n with seeds: " + str(storeSeeds))
    maxVal = storePayoff[k-1][i-1]
    for j in range(0,k):
        if storePayoff[j][i-1] > maxVal:
            maxVal = storePayoff[j][i-1]
    return (maxVal, storeSeeds[j][i-1])
