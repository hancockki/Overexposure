from pulp import *
import create_clusters as cc
import networkx as nx

def lp_setup(G, k):
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

    print("Nodes: ", nodes)

    node_vars = LpVariable.dicts("Nodes", nodes, lowBound=0, upBound=1, cat=LpInteger)
    edge_vars = LpVariable.dicts("Edges", edges, lowBound=0, upBound=1, cat=LpInteger)

    print(edge_vars, node_vars)

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
    print("Status:", status)

    #Print solution
    for var in prob.variables():
        print(var, "=", value(var))
    print("OPT =", value(prob.objective))
"""
We want to maximize the sum of nodes minus the sum of edges for each cluster. We use a bipartite graph and ensure that
we do not double count each edge by ensuring that once we subtract an edge once, we cannot subtract it again.

Thus, we have:

max (Sum(x_i w_i) - Sum(j_e w_e)) such that Sum(x_i <= k)

could we also make variables for the edges, and have a constraint where the sum of edges incoming to each rejecting
node is 1, meaning we cannot count the node more than once? 

constraints:
    if we take a cluster we must take each rejecting node it is connected to
    we cannot take the same rejecting node more than once

ASSUME critical node can be shared by at MOST two clusters
Implement original linear program

    with open("currently_in_use/make_matrix.txt") as constraints:
        lines = constraints.readlines()
        adjacency_matrix = []
        temp = []
        for line in lines:
            numbers = line.strip().split(',')
            for number in numbers:
                temp.append(int(number))
            adjacency_matrix.append(temp)
            temp = []
    constraints.close()
    for row in constraints:
        for column in constraints:
            if row==column:
                node_weights.append(row)
            else:
                edge_weights.append(row)
"""      