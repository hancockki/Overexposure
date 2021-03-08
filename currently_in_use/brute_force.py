import networkx as nx
import itertools
from itertools import combinations
'''
Brute force algorithm used to check if tree decomposition is working properly

@params:
    G --> the cluster graph we are seeding from
    k --> the number of clusters we are seeding
    debug --> do (or not) debug print statments (will delete these later bc makes code look messy, but left for now)
@returns:
    best_payoff --> payoff of optimal k seed set
'''
def computePayoff(G, k, debug):
    combinations = []
    for i in range(k):
        i += 1
        combinations.extend(list(itertools.combinations(G.nodes(), i))) # all possible combinations n choose k
    if debug: print("posible combinations:",combinations)
    # print('G.nodes.data():',G.nodes.data())
    # print('G.edges.data():',G.edges.data())
    best_payoff = 0 # record of best payoff   
    best_payoff_selection = -1 # record of which clusters produce best payoff
    for combo in combinations:
        set_negative_edges = set() # set used to prevent double counting an edge
        set_reject_nodes = set() # set used to prevent reject nodes from being double counted
        payoff = 0
        for node in combo:
            if debug: print('in node',node, 'val', G.nodes[node]['weight'])
            payoff += G.nodes[node]['weight']
            if debug: print('\tupdated payoff',payoff)
            for neighbor in G.neighbors(node): # subtracting edges from payoff (no repeats)
                if node > neighbor:
                    edge = (neighbor, node)
                else:
                    edge = (node, neighbor)
                if debug: print('\tedge',edge, 'weight', G.edges[edge]['weight'])
                if edge not in set_negative_edges:
                    set_negative_edges.add(edge)
                    try:
                        reject_node_set = G.edges[edge]['data']
                        reject_node = reject_node_set.pop()
                        reject_node_set.add(reject_node)
                        if reject_node in set_reject_nodes: # don't subtract from payoff if rejecting node already been counted
                            payoff += 1
                        else:
                            set_reject_nodes.add(reject_node)
                    except:
                        pass
                    payoff = payoff - G.edges[edge]['weight']
                    if debug: print('\tupdated payoff',payoff)
        if (payoff > best_payoff):
            best_payoff = payoff
            best_payoff_selection = combo
        if debug: print('selected nodes',combo,'negative edges',set_negative_edges,'total payoff',payoff)
    print('clusters selected for best payoff:',best_payoff_selection)
    return best_payoff_selection,best_payoff