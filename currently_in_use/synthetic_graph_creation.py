import networkx as nx
import matplotlib.pyplot as plt
import time
import create_clusters as cc

'''
Not really sure what this will be used for, but the math here makes the graphs comparable
'''
def create_test_graphs(criticality, num_nodes, num_links, prob_rewrite_edge, file_suffix):
    file_prefix = "testing_files/"
    BA = create_ba(num_nodes, num_links)
    ER = create_er(num_nodes, (2 * num_links) / num_nodes)
    avg_num_edges = (BA.number_of_edges() + ER.number_of_edges()) / 2
    # print(avg_num_edges)
    # print(round((2 * avg_num_edges) / num_nodes))
    WS = create_ws(num_nodes, round((2 * avg_num_edges) / num_nodes), prob_rewrite_edge)

    original_graphs = [['ba', BA], ['er', ER], ['ws', WS]]
    cluster_graphs = []
    total_edges = 0
    for G in original_graphs:
        G_cluster_no_cycle = False
        G_cluster_cycle = False
        print("Graph Type:", G[0], "Num edges: ", G[1].number_of_edges(), "n",G[1].number_of_nodes())
        cc.setAllNodeAttributes(G[1])
        #cc.saveOriginalGraph(G, criticality, file_prefix + G_type + str(file_suffix) + ".txt")
        total_edges += G[1].number_of_edges()
        #cc.showOriginalGraph(G,criticality)
        #plt.show()
        while G_cluster_cycle == False:
            G_cluster_cycle = cc.testOriginaltoCluster(G[1], num_nodes, criticality, False, False)
        while G_cluster_no_cycle == False:
            G_cluster_no_cycle = cc.removeClusterCycles(G_cluster_cycle)
        
        cluster_graphs.append(G_cluster_cycle)
        cluster_graphs.append(G_cluster_no_cycle)
    print(original_graphs, cluster_graphs)
    return total_edges, original_graphs, cluster_graphs

def create_ba(n,m):
    return nx.barabasi_albert_graph(n,m)

def create_er(n,p):
    return nx.erdos_renyi_graph(n,p)

def create_ws(n,k,p):
    return nx.watts_strogatz_graph(n,k,p)

def make_graphs(c, n):
    num_graphs = 1
    m = 2
    p = 0.2
    all_edges = 0
    all_graphs = []
    for i in range(num_graphs):
        edges, original_graphs, cluster_graphs = create_test_graphs(c,n,m,p,i)
        all_edges += edges
        all_graphs.extend(original_graphs)
    print("overall edge average", str(all_edges / (3 * num_graphs)))
    return cluster_graphs, original_graphs
    # to create clusters use cc.buildClusteredSet(G, c)
    # for G in all_graphs:
    #     G_cluster = cc.buildClusteredSet(G,c)
    #     d.printGraph(G_cluster)