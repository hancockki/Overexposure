import networkx as nx

#G is the cluster graph
def graph_to_bipartite(G):
    bipartite_graph = nx.DiGraph()
    for edge in G.edges.data():
        #get all rej nodes in that edge
        print("Edge: ", edge)
        if not bipartite_graph.has_node(edge[0]):
            bipartite_graph.add_node(edge[0])
            bipartite_graph.nodes[edge[0]]['weight'] = G.nodes[edge[0]]['weight']
        if not bipartite_graph.has_node(edge[1]):
            bipartite_graph.add_node(edge[1])
            bipartite_graph.nodes[edge[1]]['weight'] = G.nodes[edge[1]]['weight']
        try:
            for rej_node in edge[2]['data']: #look at all rejecting nodes
                if not bipartite_graph.has_node(rej_node):
                    bipartite_graph.add_node(rej_node)
                if not bipartite_graph.has_edge(rej_node, edge[0]):
                    bipartite_graph.add_edge(rej_node, edge[0])
                if not bipartite_graph.has_edge(rej_node, edge[1]):
                    bipartite_graph.add_edge(rej_node, edge[1])            
        except KeyError:
            continue
    return bipartite_graph
            




