import networkx as nx
import itertools
from itertools import combinations
DEBUG = False # DEBUG --> do (or not) DEBUG print statments (will delete these later bc makes code look messy, but left for now)
'''
Brute force algorithm used to check if tree decomposition is working properly

@params:
    G --> the cluster graph we are seeding from
    k --> the number of clusters we are seeding
@returns:
    best_payoff --> payoff of optimal k seed set
'''
def computePayoff(G, k):
    combinations = []
    for i in range(k):
        i += 1
        combinations.extend(list(itertools.combinations(G.nodes(), i))) # all possible combinations n choose k
    if DEBUG: print("posible combinations:",combinations)
    # print('G.nodes.data():',G.nodes.data())
    # print('G.edges.data():',G.edges.data())
    best_payoff = 0 # record of best payoff   
    best_payoff_selection = -1 # record of which clusters produce best payoff
    for combo in combinations:
        used_negative_edges = set() # set used to prevent double counting an edge
        used_reject_nodes = set() # set used to prevent reject nodes from being double counted
        payoff = 0
        for node in combo:
            if DEBUG: print('in node',node, 'val', G.nodes[node]['weight'])
            payoff += G.nodes[node]['weight']
            if DEBUG: print('\tupdated payoff',payoff)
            for neighbor in G.neighbors(node): # subtracting edges from payoff (no repeats)
                if node > neighbor:
                    edge = (neighbor, node)
                else:
                    edge = (node, neighbor)
                if DEBUG: print('\tedge',edge, 'weight', G.edges[edge]['weight'])
                if edge not in used_negative_edges:
                    used_negative_edges.add(edge)
                    try:
                        reject_node_set = G.edges[edge]['rej_nodes']
                        for reject_node in reject_node_set:
                            if reject_node in used_reject_nodes: # don't subtract from payoff if rejecting node already been counted
                                payoff += 1
                            else:
                                used_reject_nodes.add(reject_node)
                    except:
                        pass
                    payoff = payoff - G.edges[edge]['weight']
                    if DEBUG: print('\tupdated payoff',payoff)
        if (payoff > best_payoff):
            best_payoff = payoff
            best_payoff_selection = combo
        if DEBUG: print('selected nodes',combo,'negative edges',used_negative_edges,'total payoff',payoff)
    return best_payoff_selection,best_payoff