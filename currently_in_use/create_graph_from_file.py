import networkx as nx
import matplotlib.pyplot as plt
import re

'''
The goal of this class is to create a graph from a file. It is written to accomodate
both original graphs and cluster graphs, each with thier own slighly different format

Cluster graph format:
    c
    # Timestamp: {timestamp in date format}
    # Nodes: {number of n nodes in graph}
    # Edges: {number of e edges in graph}
    {weight of node 0}
    {weight of node 1}
            .
            . n times
            .
    {weight of node n-1}
    {node in edge} {other node in edge} {weight of edge} {OPTIONAL: reject node} {OPTIONAL: second reject node} ...
            .
            . e times
            .
    {node in edge} {other node in edge} {weight of edge} {OPTIONAL: reject node} {OPTIONAL: second reject node} ...

    Example file:
    c
    # Timestamp: 2021-03-13 23:18:44.662269
    # Nodes: 3
    # Edges: 2
    3
    4
    2
    3 4 1 1
    2 3 0

    format in english (will probabily delete)
    the first line will be 'c' to identify it as a cluster graph file
    the second line will be a comment with a time stamp of when the graph was generated
    the third line is a comment of the number of n nodes in the graph
    the fourth line is a comment of the number of e edges in the graph
    the next n lines are the weights of the node. The first of these lines represents node 0 etc
    the next e lines are the edges.
        The first two numbers identify the nodes in the edge
        The next number is the weight of the edge (which is often either 0 or 1)
        Any following numbers in an edge line are a record of the reject nodes that connect two cluster (these are OPTIONAL)
   

Original graph format:
    o
    crit {criticality of graph}
    # Timestamp: {timestamp in date format}
    # Nodes: {number of n nodes in graph}
    # Edges: {number of e edges in graph}
    {criticality of node 0}
    {criticality of node 1}
            .
            . n times
            .
    {criticality of node n-1}
    {node in edge} {other node in edge}
            .
            . e times
            .
    {node in edge} {other node in edge}

    Example file:
    o
    # Timestamp: 2021-03-13 23:18:44.662269
    # Nodes: 4
    # Edges: 3
    .5
    .25
    .3
    .9
    0 1
    1 2
    1 3

    format in english (will probabily delete)
    the first line will be 'o' to indicate it is an original graph file
    the second line is a record of the criticality of the graph
    the third line will be a comment with a time stamp of when the graph was generated
    the fourth line is a comment of the number of n nodes in the graph
    the fifth line is a comment of the number of e edges in the graph
    the next n lines are criticality of each node. The first of these lines represents node 0 etc
    the next e lines are edges, where two numbers identify the nodes in the edge
        there are NO edge attributes in the original graph
'''

COMMENT = '#' # acts as comment character. will be ignored if read in file
CLUSTER = 'c' # identify file as cluster graph format
ORIGINAL = 'o' # identify file as originial graph format
# use "currently_in_use/" if in Overexposue folder, "" if in currently_in_use already (personal war im fighting with the vs code debugger)
FILE_DIRECTORY_PREFIX = "currently_in_use/"#currently_in_use/

global G
G = nx.Graph()

'''
Takes a file and has different returns based on input
     the file is an original, returns a the criticality and original graph
    If the file is a cluster, returns a cluster graph

@params
    filename --> file input
@returns
    IF ORIGINAL c --> the criticality of the original graph
    G --> a graph
'''
def create_from_file(filename):
    file = open(FILE_DIRECTORY_PREFIX + filename,"r")
    if file.mode == 'r':
        contents = file.read()
    lines = re.split("\n", contents)
    file.close()

    if lines[0] == CLUSTER:
        return create_cluster(lines[1:])
    elif lines[0] == ORIGINAL:
        k = int(re.split(" ", lines[1])[1])
        crit = float(re.split(" ", lines[2])[1])
        graph_type = re.split(" ", lines[3])[1]
        ID = re.split(" ", lines[4])[1]
        remove_cycles = bool(int(re.split(" ", lines[5])[1]))
        assumption_1 = bool(int(re.split(" ", lines[6])[1]))
        return k, crit, graph_type, ID, remove_cycles, assumption_1, create_original(lines[7:])

'''
Creates an original graph based on content of file. Will ignore comments marked with a '#'
at the begining of a line
@params:
    lines -->    the lines in the file, excluding the first two because these indicate
                that just indicated it was and original and its criticality
@returns:
    G -->        an orginal graph!
'''
def create_original(lines):
    end_preamble = 0 # used to offset comments
    for i, line in enumerate(lines):
        nums = re.split(" ", line)
        if nums[0] != COMMENT: # ignore comments
            if len(nums) == 1:
                # create a node!
                G.add_node(i-end_preamble)
                G.nodes[i-end_preamble]['criticality'] = float(nums[0])
                G.nodes[i-end_preamble]['visited'] = False
                G.nodes[i-end_preamble]['cluster'] = -1
            else:
                create_edge(nums)
        else:
            end_preamble += 1
    return G

'''
Creates a cluster graph based on content of file. Will ignore comments marked with a '#'
at the begining of a line
@params:
    lines -->   the lines in the file, excluding the first line because that just indicated it was a cluster graph
@returns:
    G -->       an orginal graph!
'''
def create_cluster(lines):
    end_preamble = 0 # used to offset comments
    for i, line in enumerate(lines):
        nums = re.split(" ", line)
        if nums[0] != COMMENT: # ignore comments
            if len(nums) == 1:
                # create a node!
                G.add_node(i-end_preamble)
                G.nodes[i-end_preamble]['weight'] = int(nums[0])
            else:
                create_edge(nums)
        else:
            end_preamble += 1
    return G

'''
Takes attributes and creates an edge in graph G. This is used in both create cluster 
and create original

@params:
    attrib_info --> the info for edge, including possible attributes
'''
def create_edge(attrib_info):
    edge = "not assigned"
    for i,item in enumerate(attrib_info):
        if i == 1: # create the edge between two nodes
            G.add_edge(int(attrib_info[i-1]),int(attrib_info[i]))
            edge = (int(attrib_info[i-1]),int(attrib_info[i]))
        elif i == 2: # add weight attribute if data exists
            G.edges[edge]['weight'] = int(attrib_info[i])
        elif i == 3: # create a set with reject node (if available)
            temp = set()
            temp.add(int(attrib_info[i]))
            G.edges[edge]['rej_nodes'] = temp
        elif i > 3: # add to reject set if more than one reject is present in an edge
            G.edges[edge]['rej_nodes'].add(int(attrib_info[i]))

# def main():
#     crit, T = create_from_file(FILE_DIRECTORY_PREFIX + "testing_files/er0.txt")
#     c = 0.5
#     color_map = []
#     for nodeID in T.nodes():
#         if T.nodes[nodeID]['criticality'] >= c:
#             color_map.append('red')
#         else:
#             color_map.append('green')
#     plt.figure('original network')
#     nx.draw_networkx(T, node_color = color_map, pos=nx.spring_layout(T, iterations=1000), arrows=False, with_labels=True)
#     plt.show()
# main()