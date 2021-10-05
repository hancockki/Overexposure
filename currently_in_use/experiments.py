import driver
import view
import sys
import graph_creation
import create_graph_from_file as cff
import networkx as nx
FL_PREFIX = "test_files/"
FILE_DIRECTORY_PREFIX = "currently_in_use/"

def is_ID_reset():
    filename = FILE_DIRECTORY_PREFIX + "ID_tracker.txt"
    file = open(filename, 'r')
    # get current ID that is recorded in file
    if file.mode == 'r':
        ID = file.read()

    return ID == "1"

def generate_and_save_graphs(node_sizes, runs, m=2,p=0.2):
    if not is_ID_reset():
        do_stop = input("The ID counter is not reset, is this intentional? Data may be overwritten if you continue. Enter [Y] or [N]: ")
        if do_stop == "Y":
            sys.exit()
    for num_nodes in node_sizes:
        for i in range(runs):
            original_graphs, original_types = driver.generate_original_graphs(num_nodes, m, p)
            for O, graph_type in zip(original_graphs, original_types):
                ID = view.generate_ID()
                location = view.save_original(O, "n/a", "n/a", graph_type, ID, "n/a", "n/a")

def test_dif_combos(node_sizes, runs, k_vals, appeals):
    graph_types = ["BA","ER","WS"]
    offset = [1,2,3]

    for graph, ID_off in zip(graph_types, offset):
        for appeal in appeals:
            for num_nodes_offset, num_nodes in enumerate(node_sizes):
                for k in k_vals:
                    for i in range(runs):
                        filename = graph + "/" + str(num_nodes) + "/" + str((num_nodes_offset * runs * 3) + (i * 3) + ID_off)
                        driver.test_file(filename, k, appeal)

def get_cluster_data():
    filename = graph + "/" + str(num_nodes) + "/" + str((num_nodes_offset * runs * 3) + (i * 3) + ID_off)
    file_k, file_appeal, graph_type, ID, file_remove_cycles, file_assumption_1, O = cff.create_from_file(location)
    C = graph_creation.create_cluster_graph(O, threshold)
    B = create_bipartite_from_cluster(C)

    print("File: " + filename)
    cluster_occurance = dict()
    for i in range(runs):
        for node in B.nodes():
            if B.nodes[node]['bipartite'] == 0:
                weight = B.nodes[node]['weight']
                if weight in cluster_occurance.keys():
                    cluster_occurance[weight] = cluster_occurance[weight] + 1
                else:
                    cluster_occurance[weight] = 1
    print(cluster_occurance)
    B.clear()
    C.clear()

# def get_all_cluster_data():
#     node_sizes = [5000]
#     runs = 25
#     graph_types = ["BA","ER","WS"]
#     offset = [1,2,3]

#     cluster_dicts_by_graph = []
#     for i in offset:
#         cluster_dicts_by_graph.append(dict())
#     for graph, ID_off in zip(graph_types, offset):
#         for num_nodes_offset, num_nodes in enumerate(node_sizes):
                

def main():
    node_sizes = [500]#, 1000, 2000, 5000] # ["500", "1000", "2000", "5000"] | ["150", "500", "1000", "2000"]
    k_vals = ["10"]#,"20","50","100"] # ["5","10","20","50"]
    appeals = ["0.5"]#["0.25","0.5", "0.75"] # ["0.5", "0.5", "0.5", "0.5"]
    runs = 25

    m = 2
    p = 0.2

    # if not is_ID_reset():
    #     do_gen = input("Do you want to generate new files? Enter [Y] or [N]: ")
    #     if do_gen == "Y":
    #         generate_and_save_graphs(node_sizes, runs, m=2,p=0.2)
    test_dif_combos(node_sizes, runs, k_vals, appeals)

main()
# get_cluster_data()
# driver.test_file("1", "5", "0.5")

# # the functions below were lazy ways to generate and test programs, relying on generation while testing
# def generate_all_possible_combos():
#     node_sizes = ["500", "1000", "2000", "5000"] # ["150", "500", "1000", "2000"]
#     possible_k = ["10","20","50","100"] # ["5","10","20","50"]
#     possible_criticalities = ["0.5", "0.75"] # ["0.5", "0.5", "0.5", "0.5"]

#     do_remove_cycles = "False"
#     do_assumption_1 = "False"

#     runs = 25

#     for criticality in possible_criticalities:
#         for num_nodes in node_sizes:
#             for k in possible_k:
#                 for i in range(runs):
#                     print("RUN " + str(i) + ": " + str(num_nodes) + " " + str(k) + " " + str(criticality))
#                     driver.test_new_file(num_nodes, k, criticality, do_remove_cycles, do_assumption_1)

# def test():
#     graph_types = ["BA"]
#     offset = [1]
#     node_size = ["150"]
#     do_remove_cycles = "False"
#     do_assumption_1 = "False"

#     for graph, ID_off in zip(graph_types, offset):
#         for size_offset, size in enumerate(node_size):
#             size_offset = 0
#             for i in range(50):
#                 filename = graph + "/" + size + "/" + str((size_offset * 150) + (i * 3) + ID_off)
#                 # if filename[:3] != "BA/":
#                 driver.retest_old_file(filename, do_remove_cycles, do_assumption_1)

# def practice():
#     num_nodes = "500"
#     k = "10"
#     criticality = "0.5"
#     do_remove_cycles = "False"
#     do_assumption_1 = "False"

#     for i in range(6):
#         print("RUN " + str(i) + ": " + str(num_nodes) + " " + str(k) + " " + str(criticality))
#         driver.test_new_file(num_nodes, k, criticality, do_remove_cycles, do_assumption_1)
# def retest_all_ER_WS():
#     graph_types = ["ER","WS"]
#     offset = [2,3]
#     node_size = ["150", "500", "1000", "2000"]
#     do_remove_cycles = "False"
#     do_assumption_1 = "False"

#     for graph, ID_off in zip(graph_types, offset):
#         for size_offset, size in enumerate(node_size):
#             for i in range(50):
#                 filename = graph + "/" + size + "/" + str((size_offset * 150) + (i * 3) + ID_off)
#                 # if filename[:3] != "BA/":
#                 driver.retest_old_file(filename, do_remove_cycles, do_assumption_1)

# def retest_all_files():
#     graph_types = ["BA","ER","WS"]
#     offset = [1,2,3]
#     node_size = ["150", "500", "1000", "2000"]
#     do_remove_cycles = "False"
#     do_assumption_1 = "False"

#     for graph, ID_off in zip(graph_types, offset):
#         for size_offset, size in enumerate(node_size):
#             for i in range(50):
#                 filename = graph + "/" + size + "/" + str((size_offset * 150) + (i * 3) + ID_off)
#                 # if filename[:3] != "BA/":
#                 driver.retest_old_file(filename, do_remove_cycles, do_assumption_1)

# def retest_BA():
#     graph_types = ["BA"]
#     offset = [1]
#     node_size = ["150", "500", "1000", "2000"]
#     do_remove_cycles = "False"
#     do_assumption_1 = "False"

#     for graph, ID_off in zip(graph_types, offset):
#         for size_offset, size in enumerate(node_size):
#             for i in range(50):
#                 filename = graph + "/" + size + "/" + str((size_offset * 150) + (i * 3) + ID_off)
#                 # if filename[:3] != "BA/":
#                 driver.retest_old_file(filename, do_remove_cycles, do_assumption_1)

# # practice()