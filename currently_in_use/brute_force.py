import networkx as nx
import itertools
from itertools import combinations

# import create_graph_from_file as cff
# import make_bipartite_graph as mbg
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
    # get all nodes that are clusters (don't want combinations that include reject node)
    cluster_nodes = []
    for node in G.nodes():
        if node >= 0:
            cluster_nodes.append(node)
    for i in range(k):
        i += 1
        combinations.extend(list(itertools.combinations(cluster_nodes, i))) # all possible combinations n choose k
    if DEBUG: print("posible combinations:",combinations)

    best_payoff = -1 # record of best payoff   
    best_payoff_selection = -1 # record of which clusters produce best payoff
    for combo in combinations:
        payoff = 0
        # keep track of edges that exist at a given iteration
        edges = list(G.edges())
        # each cluter node in a combo. Add to payoff and "remove" negative reject nodes that that cluster is connected to
        for node in combo:
            if DEBUG: print('in node',node, 'val', G.nodes[node]['weight'])
            payoff += G.nodes[node]['weight']
            num_neg = 0
            # the reject edges that one cluster is connected to (Based off of bipartite graph that is not being modified)
            reject_edges = G.in_edges(node)
            for reject_edge in reject_edges:
                # iterating in reverse order so edges are not skipped
                # ex if you have a list [0,1,2] and remove 1, you will never look at 2 because index 1 has already been iterated over
                for edge in reversed(edges):
                    reject_used = False
                    # remove all edges that contain a rejecting node that the cluster touches, so that they cannot be double counted 
                    if edge[0] == reject_edge[0]:
                        if DEBUG: print('remove edge',edge)
                        edges.remove(edge)
                        reject_used = True
                # subtract from payoff if the rejecting edge was in the list of available edges (see if it was used)
                if reject_used: num_neg += 1

            payoff -= num_neg

        if (payoff > best_payoff):
            best_payoff = payoff
            best_payoff_selection = combo
        if DEBUG: print('selected nodes',combo,'num reject nodes',num_neg,'total payoff',payoff)
    return best_payoff_selection,best_payoff

    # if DEBUG: print('\tupdated payoff',payoff)
    #         for neighbor in G.neighbors(node): # subtracting edges from payoff (no repeats)
    #             if node > neighbor:
    #                 edge = (neighbor, node)
    #             else:
    #                 edge = (node, neighbor)
    #             if DEBUG: print('\tedge',edge, 'weight', G.edges[edge]['weight'])
    #             if edge not in used_negative_edges:
    #                 used_negative_edges.add(edge)
    #                 try:
    #                     reject_node_set = G.edges[edge]['rej_nodes']
    #                     for reject_node in reject_node_set:
    #                         if reject_node in used_reject_nodes: # don't subtract from payoff if rejecting node already been counted
    #                             payoff += 1
    #                         else:
    #                             used_reject_nodes.add(reject_node)
    #                 except:
    #                     pass
    #                 payoff = payoff - G.edges[edge]['weight']
    #                 if DEBUG: print('\tupdated payoff',payoff)


# def main():
#     G = cff.create_from_file("currently_in_use/tests/cluster_graph_details.txt")
#     bipartite = mbg.graph_to_bipartite(G)
#     print(computePayoff(bipartite,3))
# main()