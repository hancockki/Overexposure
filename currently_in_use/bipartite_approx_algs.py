import networkx as nx

"""
Greedy select the k highest nodes, deleting the edges between the
cluster picked and rejecting nodes it is connected to after picking
(to avoid double counting)
"""
def greedy_selection(G, k):
    max_weight = 0
    max_weight_node = None
    payoff = 0
    print(nx.get_node_attributes(G,'weight'))
    print(G.edges())
    weight_dict = dict(nx.get_node_attributes(G,'weight'))
    edges = list(G.edges())
    for i in range(k):
        for key, value in weight_dict.items():
            if value > max_weight:
                max_weight = value
                max_weight_node = key
        weight_dict[max_weight_node] = -1000
        num_neg = 0
        for nodes in edges:
            if nodes[1] == max_weight_node:
                print(nodes)
                num_neg += 1
                edges.remove(nodes)
        print("Num edges: ", num_neg, " Max weight node: ", max_weight_node)
        #G.remove_node(max_weight_node)
        payoff += max_weight - num_neg
        max_weight = 0
        max_weight_node = None
    print("Payoff greedy bipartite: ", payoff)
    return payoff

def forward_thinking_greedy(G,k):
    payoff = 0
    max_weight_node = None
    max_weight = 0
    weight_total = 0
    weight_dict = dict(nx.get_node_attributes(G,'weight'))
    print(weight_dict)
    edges = list(G.edges())
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
                #print("Neg Nodes: ", neg_nodes1, neg_nodes2, "lengths: ", len(neg_nodes2), len(neg_nodes1), "Weights:", weight, weight2, " Total: ", weight_total)
                if weight_total > max_weight:
                    #print("New max weight node: ", node1)
                    max_weight = weight_total
                    max_weight_node = node2
                neg_nodes1 = set()
                weight_total = 0
            neg_nodes2 = set()
        print("weight total: ", weight_total, " Node: ", max_weight_node)
        payoff += max_weight
        for nodes in edges:
            if nodes[1] == max_weight_node:
                edges.remove(nodes)
        #G.remove_node(max_weight_node)
        weight_dict[max_weight_node] = -1000
        max_weight = 0
        max_weight_node = None
        print("Max weight node: ", max_weight_node)
    print("Payoff forward thinking bipartite: ", payoff)
    return payoff





