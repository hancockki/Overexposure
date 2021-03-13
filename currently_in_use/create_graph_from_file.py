import networkx as nx
import matplotlib.pyplot as plt
import re

import driver as d

def create_cluster_graph(filename):
    file = open(filename,"r")
    if file.mode == 'r':
        contents = file.read()
    filename = filename.replace('.txt','')
    lines = re.split("\n", contents)

    G = nx.Graph()

    for i, line in enumerate(lines, start=-3):
        words = re.split(" ", line)
        if words[0] != "c":
            if len(words) == 1:
                G.add_node(i)
                G.nodes[i]['weight'] = int(words[0])
            else:
                create_edge(G, words)
    file.close()
    return G

def create_edge(G, words):
    edge = "not assigned"
    for i,item in enumerate(words):
        if i == 1:
            G.add_edge(int(words[i-1]),int(words[i]))
            edge = (int(words[i-1]),int(words[i]))
        elif i == 2:
            G.edges[edge]['weight'] = int(words[i])
        elif i == 3:
            G.edges[edge]['data'] = {int(words[i])}
        elif i > 3:
            G.edges[edge]['data'].add(int(words[i]))