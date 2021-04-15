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
    max_weight = -float("inf")
    max_weight_node = None
    total_payoff = 0
    print(nx.get_node_attributes(G,'weight'))
    print(G.edges())
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
                print("In degree: ", num_neg)
                max_weight = payoff - num_neg
                max_weight_node = node
        weight_dict[max_weight_node] = -float("inf")
        #add rej nodes connected to highest payoff cluster to list
        rej_nodes = [x[0] for x in G.in_edges(max_weight_node)]
        print("Rejecting nodes: ", rej_nodes)
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
    print("Payoff greedy bipartite: ", total_payoff)
    return total_payoff

def forward_thinking_greedy(G,k):
    max_weight_node = None
    payoff = 0
    max_weight = 0
    weight_total = 0
    weight_dict = dict(nx.get_node_attributes(G,'weight'))
    #print(weight_dict)
    edges = list(G.edges())
    rej_nodes = []
    neg_nodes1 = set()
    neg_nodes2 = set()
    for i in range(k):
        for node1, value1 in weight_dict.items():
            if value1 < 0:
                continue
            for nodes in edges:
                if nodes[1] == node1:
                    neg_nodes1.add(nodes[0])
            weight = value1
            for node2, value2 in weight_dict.items():
                if value2 < 0:
                    continue
                elif node2 == node1:
                    continue
                for nodes in edges:
                    if nodes[1] == node2:
                        #print("Neg node: ", nodes[0], " Cluster: ", nodes[1], " Cluster weight: ", value2)
                        neg_nodes2.add(nodes[0])
                weight2 = value2
                weight_total = weight + weight2 - len(neg_nodes1) - len(neg_nodes2)
                #print("Nodes: ", node1, node2)
                #print("Neg Nodes: ", neg_nodes1, neg_nodes2, "lengths: ", len(neg_nodes2), len(neg_nodes1), "Weights:", weight, weight2, " Total: ", weight_total)
                if weight_total > max_weight:
                    print("New max weight node: ", node1)
                    max_weight = weight_total
                    max_weight_node = node1
                #print("Neg nodes: ", neg_nodes2)
                neg_nodes2 = set()
                weight_total = 0
            #print("Neg nodes outer: ", neg_nodes1)
            neg_nodes1 = set()
        if max_weight_node is not None:
            payoff += weight_dict[max_weight_node]
        for nodes in edges:
            if nodes[1] == max_weight_node:
                payoff -= 1
                edges.remove(nodes)
                rej_nodes.append(nodes[0])
        #remove edges that are connected to the cluster we are picking
        for nodes in reversed(edges):
            if nodes[0] in rej_nodes:
                edges.remove(nodes)
        #print("payoff: ", payoff, " Node: ", max_weight_node)
        weight_dict[max_weight_node] = -1000
        max_weight = 0
        max_weight_node = None
        print("Max weight node: ", max_weight_node)
    print("Payoff forward thinking bipartite: ", payoff)
    return payoff





