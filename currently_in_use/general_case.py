import networkx as nx
import math
import random
from pulp import *
import itertools
from itertools import combinations

CONST_SUFFIX = "_constraint"
DEBUG = False # DEBUG --> do (or not) DEBUG print statments (will delete these later bc makes code look messy, but left for now)

""" 
Called from witihin driver to solve the linear program.
Create variables for each cluster and rejecting node and then
map the weights of each edge onto it.
"""
def solve_blp(G,k, debug):
    x_keys = []
    y_keys = []
    # weight dict for clusters
    weight_dict = nx.get_node_attributes(G,'weight')
    
    # # wq: this adds more x_keys than nessesary!
    # for edge in G.edges():
    #     if edge[0] not in y_keys:
    #         y_keys.append(edge[0])
    #     if edge[1] not in x_keys:
    #         x_keys.append(edge[1])
    # for node in G.nodes():
    #     if node not in x_keys:
    #         x_keys.append(node)

    # cluster nodes x keys, reject y keys
    for node in G.nodes():
        if G.nodes[node]['bipartite'] == 0:
            x_keys.append(node)
        else:
            y_keys.append(node)

    #add x and y variables
    x = LpVariable.dicts("x", x_keys, lowBound=0, cat="Integer")
    y = LpVariable.dicts("y", y_keys, lowBound=0, cat="Integer")
    # print("X_keys: ", x_keys, x, "\nY_keys: ", y_keys, y)

    # create lp with LpProblem
    lp = LpProblem("Bipartite_ILP", LpMaximize)

    # add in objective function w/ lpSum
    lp += lpSum(x[i] * weight_dict[i] for i in x_keys) - lpSum(y[i] for i in y_keys)
    
    # create constraints
    lp += lpSum(x[x_key] for x_key in x_keys) <= k, "max_cluster_select_of_" + str(k)

    for x_key in x_keys:
        lp += x[x_key] <= 1, "cluster_" + str(x_key) + CONST_SUFFIX

    for y_key in y_keys:
        for node in G.neighbors(y_key):
            #print("Neighbor of ", y_key, " is ", node)
            lp += y[y_key] >= x[node], "reject_" + str(y_key) + "_to_node_" + str(node) + CONST_SUFFIX

    # solve lp
    status = lp.solve(PULP_CBC_CMD(msg=0)) # PULP_CBC_CMD(msg=0)
    
    if debug: print("Status:",status)
    seed_set = []
    for var in lp.variables():
        if value(var) == 1 and var.name[0] == 'x':
            if debug: print(var, "=", value(var))
            seed_set.append(int(var.name[2:]))
    if debug: print("OPT Bipartite=",value(lp.objective))
    return value(lp.objective), seed_set

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def simple_greedy_selection(B, k):
    # create list of weights of clusters with their corresponding node_ID
    weight_node_list = []
    for node in B.nodes():
        if B.nodes[node]['bipartite'] == 0:
            weight = B.nodes[node]['weight']
            weight_node_list.append((weight,node))
    # sort the list of weight,node pairs from highest to lowest weights
    weight_node_list = sorted(weight_node_list, reverse=True)
    payoff = 0
    rej_nodes = set()
    seed_set = []
    # calculate payoff of selecting the largest weight nodes (with no consideration for the rejects associated with them)
    for weight, node in weight_node_list[0:k]:
        payoff += weight
        seed_set.append(node)
        for rejects in B.in_edges(node):
            rej_nodes.add(rejects[0])
    return payoff - len(rej_nodes), seed_set

def random_selection(B, k):
    clusters = []
    # get all the cluster nodes
    for node in B.nodes():
        if B.nodes[node]['bipartite'] == 0:
            clusters.append(node)

    seed_set = []
    rej_nodes = set()
    payoff = 0
    # pick k random clusters
    for i in range(k):
        try:
            cluster_location = random.randint(0,len(clusters) - 1)
            selected_cluster = clusters.pop(cluster_location)
            seed_set.append(selected_cluster)
            # calculate payoff
            payoff += B.nodes[selected_cluster]['weight']
            for rejects in B.in_edges(selected_cluster):
                rej_nodes.add(rejects[0])
        except ValueError:
            continue
    return payoff - len(rej_nodes), seed_set

"""
Faster greedy implementation!!!
"""
def greedy_selection_graph_implementation(B, k):
    total_payoff = 0
    seed_set = []
    # gets all cluster nodes
    clusters = [node for node in B.nodes() if B.nodes[node]['bipartite'] == 0]
    # create a copy of the graph because cannot modify a graph while iterating through it
    record_graph = B.copy()
    for i in range(k):
        # set max_weight as negative infinity so not afraid to take a cluster that in the long run could be better
        # when this is set to 0, algorithm more likley to select less or no seeds
        max_weight = -float("inf")
        max_weight_node = None
        # iterate over clusters to find the best payoff (when considering rejecting nodes)
        for cluster in clusters:
            cluster_weight = record_graph.nodes[cluster]['weight']
            # skip cluster if already been selected (bc weight of selected clusters are set to 0)
            if cluster_weight < 0:
                continue
            #compute node payoff by weight - rejects
            cluster_payoff = cluster_weight - len(record_graph.in_edges(cluster))
            if cluster_payoff > max_weight:
                #print("In degree: ", num_neg)
                max_weight = cluster_payoff
                max_weight_node = cluster
        if max_weight_node is not None:
            seed_set.append(max_weight_node)
            # set payoff of selected cluster so not select again
            record_graph.nodes[max_weight_node]['weight'] = -float("inf")
            # get the rejecting nodes that are connected to the selected cluster
            rej_nodes = [edge[0] for edge in record_graph.in_edges(max_weight_node)]
            # remove these rejecting nodes from graph (which means they won't count in payoff again)
            record_graph.remove_nodes_from(rej_nodes)
            total_payoff += max_weight
    return total_payoff, seed_set

# """
# Greedy select the k highest nodes, deleting the edges between the
# cluster picked and rejecting nodes it is connected to after picking
# (to avoid double counting)
# """
def greedy_selection(G, k, debug):
    max_weight = -float("inf")
    max_weight_node = None
    total_payoff = 0
    seed_set = []
    if debug: print("Weight dictionary:\n", nx.get_node_attributes(G,'weight'))
    if debug: print("Edges: \n", G.edges())
    weight_dict = dict(nx.get_node_attributes(G,'weight'))
    edges = list(G.edges())
    rej_nodes = set()
    for _ in range(k):
        for node, payoff in weight_dict.items():
            #find max weight cluster
            if payoff < 0:
                continue
            #compute intersection of in edges from unmodified graph and
            #the edges we have not yet removed from the modified graph
            num_neg = len(intersection(edges, G.in_edges(node)))
            if payoff - num_neg > max_weight:
                #print("In degree: ", num_neg)
                max_weight = payoff - num_neg
                max_weight_node = node
        if max_weight_node is not None:
            seed_set.append(max_weight_node)
            weight_dict[max_weight_node] = -float("inf")
            #add rej nodes connected to highest payoff cluster to list
            #print("Max weight node: ", max_weight_node)
            rej_nodes = [x[0] for x in G.in_edges(max_weight_node)]
            #print("Rejecting nodes: ", rej_nodes)
            #iterate through edges, remove any edge that contains rej nodes connected to picked cluster
            i = 0
            while i < len(edges):
                if edges[i][0] in rej_nodes:
                    edges.pop(i)
                else:
                    i += 1
            #print("Num edges: ", num_neg, " Max weight node: ", max_weight_node)
            #G.remove_node(max_weight_node)
            total_payoff += max_weight
        max_weight = -float("inf")
        rej_nodes = set()
        max_weight_node = None
    if debug: print("Payoff greedy bipartite: ", total_payoff)
    return total_payoff, seed_set

""" 
TODO: use deepcopy to copy G at the beginning of the method, and then
delete edges within the copy of G rather than copying the list of edges
"""
def forward_thinking_greedy_graph_implementation(G, k):
    total_payoff = 0
    seed_set = []
    weight_total = 0
    # get all clusters, sorted by weight (bc this improves algorithms)
    # this is a list of only clusters bc, by construction, only clusters have weight attribute
    clusters = []
    for elem in sorted(nx.get_node_attributes(G,'weight').items(), reverse=True, key=lambda x: x[1]):
        clusters.append(elem[0]) 
    # create a copy of the graph because cannot modify a graph while iterating through it
    record_graph = G.copy()

    for i in range(k): #number of seeds = # iterations
        max_weight = -float("inf")
        max_weight_node = None
        for node1_index in range(len(clusters)):
            node1 = clusters[node1_index]
            value1 = record_graph.nodes[node1]['weight']
            if value1 < 0:
                continue
            # get the negative nodes associated with node1
            neg_nodes1 = set()
            for edge in record_graph.in_edges(node1):
                neg_nodes1.add(edge[0])
            #get weight of the first node we are picking
            weight = value1
            for node2_index in range(node1_index + 1, len(clusters)):
                node2 = clusters[node2_index]
                value2 = record_graph.nodes[node2]['weight']
                if value2 < 0:
                    continue
                neg_nodes2 = set()
                for edge in record_graph.in_edges(node2):
                    neg_nodes2.add(edge[0])
                weight2 = value2
                weight_total = weight + weight2 - len(neg_nodes1.union(neg_nodes2))
                if weight_total > max_weight:
                    max_weight = weight_total
                    max_weight_node = node1
        if max_weight_node is not None: #we picked a node
            rej_nodes = [edge[0] for edge in record_graph.in_edges(max_weight_node)]
            total_payoff += record_graph.nodes[max_weight_node]['weight'] - len(rej_nodes)
            seed_set.append(max_weight_node)
            record_graph.nodes[max_weight_node]['weight'] = -float("inf")
            # remove these rejecting nodes from graph (which means they won't count in payoff again)
            record_graph.remove_nodes_from(rej_nodes)
    return total_payoff, seed_set

# """ 
# TODO: use deepcopy to copy G at the beginning of the method, and then
# delete edges within the copy of G rather than copying the list of edges
# """
def forward_thinking_greedy(G, k, debug):
    max_weight_node = None
    payoff = 0
    seed_set = []
    max_weight = -float("inf")
    weight_total = 0
    weight_dict = dict(nx.get_node_attributes(G,'weight'))
    #print(weight_dict)
    edges = list(G.edges())
    weight_dict_sorted = {}
    if debug: print("Weight dictionary:\n", nx.get_node_attributes(G,'weight'))
    for elem in sorted(weight_dict.items(), reverse=True, key=lambda x: x[1]):
        weight_dict_sorted[elem[0]] = elem[1] 
    if debug: print("Weight dictionary:\n", weight_dict_sorted)
    if debug: print("Edges: \n", G.edges())
    rej_nodes = []
    neg_nodes2 = set()
    for _ in range(k): #number of seeds = # iterations
        max_neg_nodes = set()
        for node1, value1 in weight_dict_sorted.items():
            neg_nodes1 = set()
            if value1 < 0:
                continue
            for edge in edges:
                if edge[1] == node1:
                    neg_nodes1.add(edge[0])
            #get weight of the first node we are picking
            weight = value1
            for node2, value2 in weight_dict_sorted.items():
                if value2 < 0:
                    continue
                elif node2 == node1: #case where the nodes are the same
                    continue #weight_total = weight - len(neg_nodes1)
                else:
                    for edge in edges:
                        if edge[1] == node2:
                            #print("Neg node: ", nodes[0], " Cluster: ", nodes[1], " Cluster weight: ", value2)
                            neg_nodes2.add(edge[0])
                    weight2 = value2
                    weight_total = weight + weight2 - len(neg_nodes1.union(neg_nodes2))
                if weight_total > max_weight:
                    if debug: print("Neg Nodes: ", neg_nodes1, neg_nodes2, " Total: ", weight_total)
                    max_weight = weight_total
                    max_weight_node = node1
                    max_neg_nodes = neg_nodes1.copy()
                neg_nodes2 = set()
                weight_total = 0
            neg_nodes1 = set() #reinitialize for first for loop
        if max_weight_node is not None: #we picked a node
            payoff += weight_dict_sorted[max_weight_node] - len(max_neg_nodes)
            seed_set.append(max_weight_node)
            if debug: print("Adding ", weight_dict[max_weight_node], " to total payoff")
            j = 0
            """
            while j < len(edges):
                if edges[j][1] == max_weight_node:
                    rej_nodes.append(edges[j][0])
                    print("rejecting node: ", edges[j])
                    payoff -= 1
                    edges.pop(j)
                    print(edges)
                else:
                    j += 1
            """
            #remove edges that are connected to the cluster we are picking            
            j = 0
            while j < len(edges):
                if edges[j][0] in max_neg_nodes:
                    #print("rejecting node: ", edges[j])
                    edges.pop(j)
                else:
                    j += 1
        #print("payoff: ", payoff, " Node: ", max_weight_node)
        weight_dict_sorted[max_weight_node] = -float("inf")
        max_weight = -float("inf")
        if debug: print("Max weight node: ", max_weight_node)
        max_weight_node = None
    if debug: print("Payoff forward thinking bipartite: ", payoff)
    return payoff, seed_set


'''
Brute force algorithm used to check if tree decomposition is working properly
@params:
    G --> the cluster graph we are seeding from
    k --> the number of clusters we are seeding
@returns:
    best_payoff --> payoff of optimal k seed set
'''
def brute_force(G, k):
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


