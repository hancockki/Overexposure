import networkx as nx
import matplotlib.pyplot as plt
import re

import driver as d
COMMENT = '#'
CLUSTER = 'c'
ORIGINAL = 'o'
# use "currently_in_use/" if in Overexposue folder, "" if in currently_in_use already (personal war im fighting with the vs code debugger)
FILE_DIRECTORY_PREFIX = "currently_in_use/"

global G
G = nx.Graph()

def create_from_file(filename):
    file = open(filename,"r")
    if file.mode == 'r':
        contents = file.read()
    filename = filename.replace('.txt','')
    lines = re.split("\n", contents)

    if lines[0] == CLUSTER:
        create_cluster(lines[1:])
    elif lines[0] == ORIGINAL:
        create_original(lines[1:])
    file.close()
    return G

def create_original(lines):
    end_preamble = 0
    for i, line in enumerate(lines):
        nums = re.split(" ", line)
        if nums[0] != COMMENT:
            if len(nums) == 1:
                G.add_node(i)
                G.nodes[i]['criticality'] = float(nums[0])
            else:
                create_edge(nums)
        else:
            end_preamble += 1

def create_cluster(lines):
    end_preamble = 0
    for i, line in enumerate(lines):
        nums = re.split(" ", line)
        if nums[0] != COMMENT:
            if len(nums) == 1:
                G.add_node(i-end_preamble)
                G.nodes[i-end_preamble]['weight'] = int(nums[0])
            else:
                create_edge(nums)
        else:
            end_preamble += 1

def create_edge(attrib_info):
    edge = "not assigned"
    for i,item in enumerate(attrib_info):
        if i == 1:
            G.add_edge(int(attrib_info[i-1]),int(attrib_info[i]))
            edge = (int(attrib_info[i-1]),int(attrib_info[i]))
        elif i == 2:
            G.edges[edge]['weight'] = int(attrib_info[i])
        elif i == 3:
            G.edges[edge]['data'] = {int(attrib_info[i])}
        elif i > 3:
            G.edges[edge]['data'].add(int(attrib_info[i]))
'''
def main():
    T = create_from_file(FILE_DIRECTORY_PREFIX + "original_graph.txt")
    c = 0.5
    color_map = []
    for nodeID in T.nodes():
        if T.nodes[nodeID]['criticality'] >= c:
            color_map.append('red')
        else:
            color_map.append('green')
    plt.figure('original network')
    nx.draw_networkx(T, node_color = color_map, pos=nx.spring_layout(T, iterations=1000), arrows=False, with_labels=True)
    plt.show()
main()
'''