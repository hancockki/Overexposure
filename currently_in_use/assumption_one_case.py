from pulp import *
import networkx as nx

"""
Loop through all nodes in the graph, pick the k highest clusters to be our seed set.
Most basic greedy approach to this problem.
"""
def kHighestClusters(G, k, debug=False):
    num_nodes = G.number_of_nodes()
    weights_list = []
    for i in range(num_nodes):
        weight = G.nodes[i]['weight']
        weights_list.append((weight, i))
    if debug: print(weights_list)
    weights_list = sorted(weights_list, reverse=True)
    if debug: print(weights_list)
    #print(weights_list)
    payoff = computePayoffGreedy(G, weights_list[0:k])
    seed_set = []
    for weight_node_tuple in weights_list[0:k]:
        seed_set.append(weight_node_tuple[1])
    return payoff, seed_set
       
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

    
def computeNegPayoff(G, nodeNum):
    nodeWeight = G.nodes[nodeNum]['weight']
    negPayoff = nx.neighbors(G, nodeNum)
    for negNode in negPayoff:
        add = G.get_edge_data(nodeNum, negNode)
        add = add['weight']
        nodeWeight -= add
    #print("node weight is:", nodeWeight)
    return nodeWeight


def lp_setup(G, k, debug):
    """
    Set up our LP based on the cluster data. We want each cluster and rejecting node to be a variable, 
    with the constraint that if you pick a cluster you must pick the rejecting nodes it is connected to.
    """
    
    prob = LpProblem("Overexposure", LpMaximize)       

    #node_weights, edge_weights = cc.makeMatrix(G, G.number_of_nodes())
    #print("Node weights:\n", node_weights, "\n", "Edge Weights:\n", edge_weights)

    edges = G.edges()
    nodes = G.nodes()
    node_weights = nx.get_node_attributes(G, name='weight')
    edge_weights = nx.get_edge_attributes(G, 'weight')

    if debug: print("Nodes: ", nodes)

    node_vars = LpVariable.dicts("Nodes", nodes, lowBound=0, upBound=1, cat=LpInteger)
    edge_vars = LpVariable.dicts("Edges", edges, lowBound=0, upBound=1, cat=LpInteger)

    if debug: print(edge_vars, node_vars)

    #define our objective
    prob += lpSum(node_vars[i]*node_weights[i] for i in node_vars) - (lpSum(edge_weights[j]*edge_vars[j] for j in edge_vars))

    #define our constraints
    prob += lpSum(node_vars[i] for i in node_vars) <= k #budget
    '''
    define constraints based on what edges are connected to each node, don't want to double count
    '''
    for e in edge_vars:
        for i in e:
            prob += edge_vars[e] >= node_vars[i]

    # Solve the LP
    status = prob.solve(PULP_CBC_CMD(msg=0))
    if debug: print("Status:", status)

    #Print solution
    seed_set = []
    for var in prob.variables():
        if value(var) == 1 and var.name[0] == 'N':
            if debug: print(var, "=", value(var))
            seed_set.append(int(var.name[6:]))
    if debug: print("OPT LP=", value(prob.objective))
    return value(prob.objective), seed_set