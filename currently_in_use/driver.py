# wq:NAME IN PROGRASS!!!!!!!!!!!!
"""
wq:have not updated yet
Driver for entire program.
How to run the program:
The program takes four commad line arguments. 
The first is the program name, the second is the number of 
nodes in the original graph, the third is the number of seeds
we are choosing for our seed set, and the fourth is the criticality
parameter. For example, typing the following in the terminal would 
output an optimal seed set of size 5 for an original graph with 50 nodes
and a criticality threshold of 0.7 (any node whose criticality is above
0.7 will be an accepting node)

python3 driver.py 50 5 0.7

Note that the above line requires that you are inside the currently_in_use folder

The main method which reads in these arguments passes them onto runTests(),
which acts as a driver for the following algorithms:

1) greedy dynamic programming
2) greedy payoff_knapsack
3) recursive DP (optimal for trees where <=2 clusters share any rej node)
4) Brute force (commented out by default because it takes so long. User
can choose to uncomment it and run)
5) Linear Program (optimal only when <= 2 clusters share any rej node)
6) Bipartite linear program (optimal on any graph)

All of the results are printed to the terminal and then
written to an excel sheet which is within the folder currently_in_use/tests

We also print the original, cluster, and bipartite graph to be used for comparison.
Note that the first thing we do in runTests is try to create 
a graph that satisfies our requirements, namely:
    1) No rejecting node is shared by more than 2 clusters
    2) No cycles in the cluster graph

We continuously try to make the cluster graph until these
properties are satisfied.
TODO: find a more efficient way to satisfy these properties. For 
original graphs with >100 nodes, it takes a long time to create
a cluster graph which satisfies the properties.
"""
import graph_creation
import view
import general_case
import tree_case
import assumption_one_case
# import DP_algorithms as dp
# import greedy_approx_algorithms as greedy
# import brute_force as bf
# import bipartite_linear_program as blp
# import bipartite_approx_algs as baa
# import cluster_linear_program as clp
import create_graph_from_file as cff
import networkx as nx
import matplotlib.pyplot as plt
import sys
import openpyxl
import random
import timeit
from datetime import datetime
import networkx as nx
import networkx.algorithms.isomorphism as iso

# use "currently_in_use/" if in Overexposue folder, "" if in currently_in_use already (personal war im fighting with the vs code debugger)
FILE_DIRECTORY_PREFIX = ""#"currently_in_use/"
ORIGINAL_FILE_LOCATION = "test_files/original/"

#TODO: run more rigorous tests
#TODO: fix logging output to take max of root and no root
#TODO: work on paper, write analysis on run times/approximation factor
#TODO: tree decomposition algorithm

"""
wq:have not updated yet
Experiment Design:
    - previous experiments in previous papers
    - what kinds of networks they are building, how many nodes they are taking into account
    - SNAP graphs?
    - we are limited to cluster graphs
    - Erdos-Renyi
    - Barabasi-Alber
    - Watts-Strogatz
    - real world networks like college messaging

Come up with an experiment design based on papers that we've read, then see how they are doing their experiments section.
    What are the common things they are doing? What kinds of random graph structures?
    Outline what the things they are showing are
    Write some brief notes and then design the experiment and then go with that
    Present those things next week
    We want to save the graph and run tests based on that
"""

'''
Not really sure what this will be used for, but the math here makes the graphs comparable
'''

#main function, used for calling things
def test_new_file(num_nodes, k, criticality, do_remove_cycles, do_assumption_1):
    num_nodes = int(num_nodes)
    k = int(k)
    criticality = float(criticality)
    do_remove_cycles = string_to_boolean(do_remove_cycles)
    do_assumption_1 = string_to_boolean(do_assumption_1)
    m = 2
    p = 0.2
    print("generating original graphs")
    # create three types of graphs (ba, er, ws) or roughly the same size
    original_graphs, original_types = generate_original_graphs(num_nodes, m, p)
    
    # for each graph, create cluster and bipartite graphs then save results, graphs, and plot them
    for O, graph_type in zip(original_graphs, original_types):
        C, B, loops_through_while = graph_creation.generate_test_graphs(O, criticality, do_remove_cycles, do_assumption_1)
        payoffs, runtimes, opt_seeds = run_tests_on_graph(C, B, k, do_remove_cycles, do_assumption_1)
        
        ID = view.generate_ID()
        # save to excel. Also provides unique ID for each row to a test can be ran again
        view.write_results_to_excel([num_nodes, k, criticality, graph_type, int(ID)], payoffs, runtimes, opt_seeds)
        
        # plot the different graphs
        view.plot_original(O, criticality)
        view.plot_cluster(C, graph_type + " cluster graph")
        view.plot_bipartite(B, graph_type + " bipartite graph")
       
        # save the different graphs by ID
        view.save_original(O, criticality, k, graph_type, ID, do_remove_cycles, do_assumption_1)
        view.save_cluster(C, k, criticality, ID, do_remove_cycles, do_assumption_1)
        # make sure these are clear so original graphs do not accidentally interfere with each other
        C.clear()
        B.clear()

def retest_old_file(original_graph_filename):
    # get all information used to make ID in excel sheet
    if original_graph_filename[-4:] != ".txt":
       original_graph_filename = original_graph_filename + ".txt"
    k, criticality, graph_type, ID, do_remove_cycles, do_assumption_1, O = cff.create_from_file(ORIGINAL_FILE_LOCATION + original_graph_filename)
    num_nodes = O.number_of_nodes()
    print("retesting " + original_graph_filename)
    
    # create cluster and bipartite based on information from file
    C, B, loops_through_while = graph_creation.generate_test_graphs(O, criticality, do_remove_cycles, do_assumption_1)
    if loops_through_while != 1:
        print("Previous tests of this graph may not have satisfied assumption 1")
        sys.exit()
    payoffs, runtimes, opt_seeds= run_tests_on_graph(C, B, k, do_remove_cycles, do_assumption_1)

    # save to excel. Also provides unique ID for each row to a test can be ran again
    view.write_results_to_excel([num_nodes, k, criticality, graph_type, int(ID)], payoffs, runtimes, opt_seeds)
    
    # plot the different graphs
    view.plot_original(O, criticality)
    view.plot_cluster(C, graph_type + " cluster graph")
    view.plot_bipartite(B, graph_type + " bipartite graph")
    print(C.edges.data())

    
    # # DONT SAVE FILE HERE BECAUSE DO NOT WANT TO OVERWRITE!
    # # wq: should a method be made to compare two graphs and see if they are (relativley) the same?
    # view.save_original(O, criticality, k, ID, do_remove_cycles, do_assumption_1)
    # view.save_cluster(C, k, criticality, ID, do_remove_cycles, do_assumption_1)        

'''
Generate three types of original graphs (Barabasi-Albert, Erdos-Renyi, and Watts-Strogatz) of the same size
@params:
    num_nodes           --> number of nodes in graph
    num_links           --> links used for BA and ER
    prob_rewrite_edge   --> used for WS
@returns
    list of three graphs (as networkx graphs!)
    list of three graph types (as strings)
'''
def generate_original_graphs(num_nodes, num_links, prob_rewrite_edge):
    # the math here is used to create three graphs with the same number of nodes and
    # roughly the same number of edges, so the three are comparable
    BA = nx.barabasi_albert_graph(num_nodes, num_links)
    ER = nx.erdos_renyi_graph(num_nodes, (2 * num_links) / num_nodes)
    avg_num_edges = (BA.number_of_edges() + ER.number_of_edges()) / 2
    WS = nx.watts_strogatz_graph(num_nodes, round((2 * avg_num_edges) / num_nodes), prob_rewrite_edge)

    original_graphs = [BA, ER, WS]
    # assign the basic attributes of each node (criticality is random!)
    for O in original_graphs:
        for node_ID in O.nodes():
            O.nodes[node_ID]["visited"] = False
            O.nodes[node_ID]['criticality'] = random.uniform(0, 1)
            O.nodes[node_ID]["cluster"] = -1
    return [BA, ER, WS], ["ba", "er", "ws"]

"""
Run algorithms on cluster graph, and its corresponding bipartite graph, based on class of algorithm.
Uses booleans remove_cycles and assumption_1 to id class of algorithm
@params:
    C               --> cluster graph (can be tree, assumption 1, or unmodified)
    B               --> bipartite graph created from this cluster graph
    k               --> number of nodes to seed
    remove_cycles   --> if true, test knapsack, greedy, recurisve DP
    assumption_1    --> if true, test cluster linear program
    NOTE: bipartite algorithms will run for all, regardless of boolean variables
@returns:
    list of payoffs
        Indecies are based off of excel speadsheet "Experimental_Results.xlsx" with each algorithm result
        contained in one index. If the algorithm was not run, contains '-'
    list of runtimes
        Indecies are based off of excel speadsheet "Experimental_Results.xlsx" with each algorithm result
        contained in one index. If the algorithm was not run, contains '-'
"""
def run_tests_on_graph(C, B, k, remove_cycles, assumption_1):
    # set default value of array (if an algorithm is not run)
    runtimes = []
    payoffs = []
    for i in range(7):
        runtimes.append('-')
        payoffs.append('-')
    
    # algorithms that work with trees
    if remove_cycles: print("Remove Cycs: " + str(remove_cycles))
    if assumption_1: print("Ass 1: " + str(assumption_1))
    if remove_cycles:
        # compute payoff using knapsack approach
        start = timeit.default_timer()
        payoff_knapsack, seedset = tree_case.greedyDP(C, C.number_of_nodes(), k)
        stop = timeit.default_timer()
        runtimes[0] = stop - start
        payoffs[0] = payoff_knapsack
        print("\nGreedy DP Payoff: ", payoff_knapsack)

        # compute payoff for recursive DP
        start = timeit.default_timer()
        payoff_root, payoff_no_root = tree_case.runRecursiveDP(C, k)
        payoff_recursive_dp = max(payoff_root, payoff_no_root)
        stop = timeit.default_timer()
        runtimes[2] = stop - start
        payoffs[2] = payoff_recursive_dp
        print("Recursive DP payoff: ", payoff_recursive_dp)

        # # if there are no cycles, assumption 1 holds true to they can be run
        # # UPON TESTING, THIS IS FALSE!!!!!!!!!!!
        # assumption_1 = True

    # algorithm for assumption 1 (nodes cannot share more than two rejecting nodes, which means that there CAN be cycles)
    if assumption_1:
        # compute payoff for most basic greedy algorithm
        start = timeit.default_timer()
        payoff_greedy, greedy_seedset = assumption_one_case.kHighestClusters(C, k)
        stop = timeit.default_timer()
        runtimes[1] = stop - start
        payoffs[1] = payoff_greedy
        print("Greedy Approach Seeds Chosen:", greedy_seedset, " with payoff: ", payoff_greedy)

        # run linear program on cluster graph
        start = timeit.default_timer()
        payoff_clp = assumption_one_case.lp_setup(C, k)
        stop = timeit.default_timer()
        runtimes[3] = stop - start
        payoffs[3] = payoff_clp

    # all general test cases (no restrictions)

    # run bipartite linear program
    start = timeit.default_timer()
    payoff_blp, blp_seeds = general_case.solve_blp(B, k)
    stop = timeit.default_timer()
    runtimes[4] = stop - start
    payoffs[4] = payoff_blp

    # run bipartite greedy algorithm
    start = timeit.default_timer()
    payoff_greedy = general_case.greedy_selection(B, k)
    stop = timeit.default_timer()
    runtimes[5] = stop - start
    payoffs[5] = payoff_greedy

    # run bipartite forward thinking algorithm
    start = timeit.default_timer()
    payoff_forward_thinking = general_case.forward_thinking_greedy(B, k)
    stop = timeit.default_timer()
    runtimes[6] = stop - start
    payoffs[6] = payoff_forward_thinking

    return payoffs, runtimes, blp_seeds

def string_to_boolean(input):
    if input == "false" or input == "False" or input == "0":
        return 0
    elif input == "true" or input == "True" or input == "1":
        return 1
    else:
        print('ERROR: Invalid boolean input (try using "True", "true", or "1" [same for false variations])')
        sys.exit()

# Uncomment to run using command line!
if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) == 2:
        retest_old_file(sys.argv[1])
    elif len(sys.argv) == 6 or len(sys.argv) == 7:
        # test_if_saved_graphs_same(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        test_new_file(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        if len(sys.argv) == 7:
            if string_to_boolean(sys.argv[6]):
                plt.show()
    else:
        print('ERROR: Invalid input')
        sys.exit()

# test_if_saved_graphs_same("100","5","0.5","True","True")
# test_new_file("100","5","0.5","True","True")
# test_new_file("40","2","1","True","True")
# retest_old_file("80.txt")
# plt.show()

# '''
# Written for debugging purposes. Testing if generated graphs and their saved counterparts generate the same graph (isomorphic). Created
# because tests of generation vs repeated tests were not matching in values.
# '''
# def test_if_saved_graphs_same(num_nodes, k, criticality, do_remove_cycles, do_assumption_1):
#     num_nodes = int(num_nodes)
#     k = int(k)
#     criticality = float(criticality)
#     do_remove_cycles = string_to_boolean(do_remove_cycles)
#     do_assumption_1 = string_to_boolean(do_assumption_1)
#     m = 2
#     p = 0.2
#     print("generating original graphs")
#     # create three types of graphs (ba, er, ws) or roughly the same size
#     original_graphs, original_types = generate_original_graphs(num_nodes, m, p)
#     count = 0
#     for O, graph_type in zip(original_graphs, original_types):
#         ID = "test-" + str(count)
#         print(ID)
#         C, B = graph_creation.generate_test_graphs(O, criticality, do_remove_cycles, do_assumption_1)
#         view.save_original(O, criticality, k, graph_type, ID, do_remove_cycles, do_assumption_1)
#         count += 1
#         k, criticality, graph_type, new_ID, do_remove_cycles, do_assumption_1, saved_O = cff.create_from_file(ORIGINAL_FILE_LOCATION + ID + ".txt")
#         saved_C, saved_B = graph_creation.generate_test_graphs(saved_O, criticality, do_remove_cycles, do_assumption_1)
#         nm = iso.categorical_node_match("criticality",0)
#         is_O_copy = nx.is_isomorphic(O,saved_O, node_match=nm)

#         # # never got this bit to work
#         # nm = iso.categorical_node_match("weight",0)
#         # em = iso.numerical_multiedge_match(["weight", "rej_nodes"], [0, None])
#         # is_C_copy = nx.is_isomorphic(C,saved_C, node_match=nm, edge_match=em)
#         print(ID + " originial same? " + str(is_O_copy))
#         print(ID + " cluster same? " + str(is_C_copy))
#         C.clear()
#         B.clear()
#         saved_O.clear()