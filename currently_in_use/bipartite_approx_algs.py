import networkx as nx
import math

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

"""
Greedy select the k highest nodes, deleting the edges between the
cluster picked and rejecting nodes it is connected to after picking
(to avoid double counting)
"""
def greedy_selection(G, k):
    max_weight = 0
    max_weight_node = None
    total_payoff = 0
    print("Weight dictionary:\n", nx.get_node_attributes(G,'weight'))
    print("Edges: \n", G.edges())
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
        if max_weight_node is not None:
            total_payoff += max_weight
        max_weight = 0
        rej_nodes = set()
        max_weight_node = None
    print("Payoff greedy bipartite: ", total_payoff)
    return total_payoff

""" 
TODO: use deepcopy to copy G at the beginning of the method, and then
delete edges within the copy of G rather than copying the list of edges
"""
def forward_thinking_greedy(G,k):
    max_weight_node = None
    payoff = 0
    max_weight = 0
    weight_total = 0
    weight_dict = dict(nx.get_node_attributes(G,'weight'))
    #print(weight_dict)
    edges = list(G.edges())
    weight_dict_sorted = {}
    print("Weight dictionary:\n", nx.get_node_attributes(G,'weight'))
    for elem in sorted(weight_dict.items(), reverse=True, key=lambda x: x[1]):
        weight_dict_sorted[elem[0]] = elem[1] 
    print("Weight dictionary:\n", weight_dict_sorted)
    print("Edges: \n", G.edges())
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
                    weight_total = weight - len(neg_nodes1)
                else:
                    for edge in edges:
                        if edge[1] == node2:
                            #print("Neg node: ", nodes[0], " Cluster: ", nodes[1], " Cluster weight: ", value2)
                            neg_nodes2.add(edge[0])
                    weight2 = value2
                    weight_total = weight + weight2 - len(neg_nodes1.union(neg_nodes2))
                if weight_total > max_weight:
                    print("Neg Nodes: ", neg_nodes1, neg_nodes2, " Total: ", weight_total)
                    max_weight = weight_total
                    max_weight_node = node1
                    max_neg_nodes = neg_nodes1.copy()
                neg_nodes2 = set()
                weight_total = 0
            neg_nodes1 = set() #reinitialize for first for loop
        if max_weight_node is not None: #we picked a node
            payoff += weight_dict_sorted[max_weight_node] - len(max_neg_nodes)
            print("Adding ", weight_dict[max_weight_node], " to total payoff")
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
        max_weight = 0
        print("Max weight node: ", max_weight_node)
        max_weight_node = None
    print("Payoff forward thinking bipartite: ", payoff)
    return payoff





