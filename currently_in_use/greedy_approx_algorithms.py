import networkx as nx
import DP_algorithms as dp

"""
Compute the payoff for the greedy algorithm
"""
def computePayoffGreedy(G, k_highest):
    payoff = 0
    nodes_counted = []
    for payoff_tuple in k_highest:
        cur_node = payoff_tuple[1]
        node_payoff = dp.computeNegPayoff(G, cur_node)
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
    return weights_list[0:k], payoff
