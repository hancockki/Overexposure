import networkx as nx

"""
Convert a cluster graph to a bipartite graph.
The bipartite graph is structured so each cluster (node)
in the cluster graph can be visualized as a node on the RHS
of the bipartite graph and each rejecting node encompassed in the edges
of the cluster graph is a node on the LHS of the bipartite graph.
The graph is directed, so we have an edge from each rejecting node 
to the cluster(s) it is connected to.
"""
def graph_to_bipartite(G):
    bipartite_graph = nx.DiGraph()
    #look at every edge in the cluster graph and create nodes in bipartite
    for edge in G.edges.data():
        #check if either endpoint of the edge has been added to the bipartite
        if not bipartite_graph.has_node(edge[0]):
            bipartite_graph.add_node(edge[0])
            bipartite_graph.nodes[edge[0]]['weight'] = G.nodes[edge[0]]['weight']
        if not bipartite_graph.has_node(edge[1]):
            bipartite_graph.add_node(edge[1])
            bipartite_graph.nodes[edge[1]]['weight'] = G.nodes[edge[1]]['weight']
        #not all edges have a rej_nodes attribute. If it does, create a node in the bipartite
        #for that rej node (if we havent already) and an edge from that rej node to the clusters
        #it is attached to.
        try:
            for rej_node in edge[2]['rej_nodes']: #look at all rejecting nodes
                rej_label = "r" + str(rej_node) # create a different lable from the cluster nodes so that they can never be confused for eachother
                if not bipartite_graph.has_node(rej_label):
                    bipartite_graph.add_node(rej_label)
                    bipartite_graph.nodes[rej_label]['weight'] = rej_node
                if not bipartite_graph.has_edge(rej_label, edge[0]):
                    bipartite_graph.add_edge(rej_label, edge[0])
                if not bipartite_graph.has_edge(rej_label, edge[1]):
                    bipartite_graph.add_edge(rej_label, edge[1])            
        except KeyError:
            continue
    return bipartite_graph
            




