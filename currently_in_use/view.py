import networkx as nx
import matplotlib.pyplot as plt
import openpyxl

FILE_DIRECTORY_PREFIX = ""#"currently_in_use/"
ORIGINAL_FILE_LOCATION = "test_files/original/"
CLUSTER_FILE_LOCATION = "test_files/cluster/"
BIPARTITE_FILE_LOCATION = "test_files/bipartite/"

def plot_original(O, c):
    color_map = []
    for node_ID in O.nodes():
        if O.nodes[node_ID]['criticality'] >= c:
            color_map.append('red')
        else:
            color_map.append('green')
    plt.figure('original network')
    nx.draw_networkx(O, node_color = color_map, pos=nx.spring_layout(O, iterations=1000), arrows=False, with_labels=True)

""" display cluster graph """
def plot_cluster(C, name):
    plt.figure(name)
    pos = nx.spring_layout(C)
    nx.draw(C, pos)

    node_labels = nx.get_node_attributes(C,'weight')
    # do (id, weight) pair for lable instead of just weight
    for key,val in node_labels.items():
        node_labels[key] = (key,val)
    nx.draw_networkx_labels(C, pos=pos, labels=node_labels)
    data_info = nx.get_edge_attributes(C,'rej_nodes') # edge lables rejecting node
    weight_info = nx.get_edge_attributes(C,'weight') # edge lables rejecting node
    edge_labels = {}
    for key in weight_info.keys():
        if key in data_info.keys():
            edge_labels[key] = (data_info[key],weight_info[key])
        else:
            edge_labels[key] = ("na",weight_info[key])
    nx.draw_networkx_edge_labels(C, pos=pos, edge_labels=edge_labels)
    # # uncomment to save figure
    # plt.savefig("saved-graphs/" + name + ".png")

""" Print bipartite graph using network x. Saved to file"""
def plot_bipartite(bipartite, name):
    print("printing bipartite graph")
    plt.figure(name+ "-bipartite")
    pos = nx.spring_layout(bipartite)
    nx.draw(bipartite, pos)

    node_labels = nx.get_node_attributes(bipartite,'weight')
    print(node_labels)
    # do (id, weight) pair for lable instead of just weight
    for key,val in node_labels.items():
        node_labels[key] = (key,val)
    nx.draw_networkx_labels(bipartite, pos=pos, labels=node_labels)
    # # uncomment to save figure
    # plt.savefig("saved-graphs/"+ name + "-bipartite.png")

'''
Saves the original graph and associated criticality in a file called original_graph.txt
This file can be used in conjuntion with create_from_file function in create_graph_from_file
The format used here is described in create_graph_from_file class

@params:
    O -> original graph
    c -> criticality (used for show purposes and creating cluster)
'''
def save_original(O, c, k, graph_type, ID, remove_cycles, assumption_1):
    filename = FILE_DIRECTORY_PREFIX + ORIGINAL_FILE_LOCATION + ID + ".txt"
    with open(filename, 'w') as graph_info:
        graph_info.write("o\n")
        graph_info.write("num_seeded: " + str(k) + "\n")        
        graph_info.write("crit: " + str(c) + "\n")
        graph_info.write("type: " + str(graph_type) + "\n")
        graph_info.write("ID: " + str(ID) + "\n")
        graph_info.write("removed_cycles: " + str(remove_cycles) + "\n")
        graph_info.write("satisfied_assumption_1: " + str(assumption_1) + "\n")
        graph_info.write("# Nodes: " + str(O.number_of_nodes()) + "\n")
        data = O.edges.data()
        graph_info.write("# Edges: " + str(len(data)))
        weights = O.nodes.data('criticality')
        for node in weights:
            #print(node)
            graph_info.write("\n" + str(node[1]))
        for item in data:
            graph_info.write("\n" + str(item[0]) + " " + str(item[1]))

""" Store more specific info about each graph, to be used for testing if the 
results output in the excel sheet are inaccurate / do not make sense """
def save_cluster(C, k, c, ID, remove_cycles, assumption_1):
    print('\nNext Test:\n')
    filename = FILE_DIRECTORY_PREFIX + CLUSTER_FILE_LOCATION + ID + ".txt"
    with open(filename, 'w') as graph_info:
        graph_info.write('c\n')
        graph_info.write("# crit: " + str(c) + "\n")
        graph_info.write("# removed_cycles: " + str(remove_cycles) + "\n")
        graph_info.write("# satisfied_assumption_1: " + str(assumption_1) + "\n")
        graph_info.write("# ID: " + str(ID) + "\n")
        graph_info.write("# Nodes: " + str(C.number_of_nodes()) + "\n")
        data = C.edges.data()
        graph_info.write("# Edges: " + str(len(data)))
        weights = C.nodes.data('weight')
        for node in weights:
            graph_info.write("\n" + str(node[1]))
        for item in data:
            graph_info.write("\n" + str(item[0]) + " " + str(item[1]) + " " + str(item[2]['weight']))
            try:
                data = item[2]['rej_nodes']
                for reject in data:
                    graph_info.write(" " + str(reject))
            except:
                pass

def generate_ID():
    filename = FILE_DIRECTORY_PREFIX + "ID_tracker.txt"
    file = open(filename, 'r')
    if file.mode == 'r':
        ID = file.read()

    with open(filename, 'w') as write_to_file:
        new_ID = int(ID) + 1
        write_to_file.write(str(new_ID))
    return ID

""" Write results to an excel sheet stored in the currently_in_use/tests folder """
def write_results_to_excel(preamble, payoffs, runtimes):
    print(preamble)
    print(payoffs)
    print(runtimes)
    wb = openpyxl.load_workbook(FILE_DIRECTORY_PREFIX + 'Experimental_Results.xlsx')
    print("WRITING RESULTS")
    sheets = wb.sheetnames
    payoff_sheet = wb[sheets[0]]
    runtime_sheet = wb[sheets[1]]
    payoff_sheet.append(preamble + payoffs)
    runtime_sheet.append(preamble + runtimes)
    wb.save(FILE_DIRECTORY_PREFIX + 'Experimental_Results.xlsx')