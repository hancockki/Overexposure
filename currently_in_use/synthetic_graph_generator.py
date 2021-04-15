import networkx as nx
import matplotlib.pyplot as plt
import time
import create_clusters as cc
import driver as d # TEMPORARY

def create_test_graphs(criticality, num_nodes, num_links, prob_rewrite_edge, file_suffix):
    file_prefix = "testing_files/"
    G_ba = create_ba(num_nodes, num_links)
    G_er = create_er(num_nodes, (2 * num_links) / num_nodes)
    avg_num_edges = (G_ba.number_of_edges() + G_er.number_of_edges()) / 2
    # print(avg_num_edges)
    # print(round((2 * avg_num_edges) / num_nodes))
    G_ws = create_ws(num_nodes, round((2 * avg_num_edges) / num_nodes), prob_rewrite_edge)

    all_graphs = [G_ba, G_er, G_ws]
    all_types = ["ba","er","ws"]
    total_edges = 0
    for (G_type,G) in zip(all_types, all_graphs):
        print(G, "e",G.number_of_edges(), "n",G.number_of_nodes())
        cc.setAllNodeAttributes(G)
        cc.saveOriginalGraph(G, criticality, file_prefix + G_type + str(file_suffix) + ".txt")
        total_edges += G.number_of_edges()
        cc.showOriginalGraph(G,criticality)
        plt.show()
        G_cluster = cc.buildClusteredSet(G,criticality)
        d.printGraph(G_cluster)
    return total_edges, all_graphs

def create_ba(n,m):
    return nx.barabasi_albert_graph(n,m)

def create_er(n,p):
    return nx.erdos_renyi_graph(n,p)

def create_ws(n,k,p):
    return nx.watts_strogatz_graph(n,k,p)

def main():
    num_graphs = 1
    c = 0.5
    n = 10
    m = 2
    p = 0.2
    all_edges = 0
    all_graphs = []
    for i in range(num_graphs):
        edges, graphs = create_test_graphs(c,n,m,p,i)
        all_edges += edges
        all_graphs.extend(graphs)
    print("overall edge average", str(all_edges / (3 * num_graphs)))
    # to create clusters use cc.buildClusteredSet(G, c)
    # for G in all_graphs:
    #     G_cluster = cc.buildClusteredSet(G,c)
    #     d.printGraph(G_cluster)
main()
