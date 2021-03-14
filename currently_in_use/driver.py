import create_clusters as cc
import DP_algorithms as dp
import greedy_approx_algorithms as greedy
import brute_force as bf
import bipartite_linear_program as blp
import create_graph_from_file as cff
import networkx as nx
import matplotlib.pyplot as plt
import sys
import linear_program as lp
from datetime import datetime

# use "currently_in_use/" if in Overexposue folder, "" if in currently_in_use already (personal war im fighting with the vs code debugger)
FILE_DIRECTORY_PREFIX = "currently_in_use/"

#TODO: allow user to type in how many nodes they want in the graph
#TODO: timestamp each graph with when you ran it
#TODO: tree decomposition algorithm
#TODO: put graphs in a format that our files can read (Laura)

"""
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

def runTests():
    #create cluster graph
    k = 3
    num_nodes = 10
    criticality = 0.7
    max_weight = 5
    #G = cc.testOriginaltoCluster(num_nodes, criticality, k)
    G = cc.createClusterGraph(num_nodes, max_weight)
    #c, G = cff.create_from_file(FILE_DIRECTORY_PREFIX + "original_graph.txt")
    #cc.showOriginalGraph(G,c)
    #plt.show()
    
    #compute payoff for greedy DP
    max_val_greedyDP = dp.greedyDP(G, G.number_of_nodes(), k)
    with open(FILE_DIRECTORY_PREFIX + "results_details.txt", "a") as results_details:
        store_info(G,k)
        print("\nGreedy DP Payoff: ", max_val_greedyDP)
        
        #compute payoff for most basic greedy algorithm
        greedy_seedset, payoff = greedy.kHighestClusters(G, k)
        print("Greedy Approach Seeds Chosen:", greedy_seedset, " with payoff: ", payoff)

        #compute payoff for recursive DP
        payoff_root, payoff_no_root = dp.runRecursiveDP(G, k)
        print("Recursive DP payoff: \n Root: ", payoff_root, "\n No Root: ", payoff_no_root)

        #compute payoff using brute force algorithm
        best_payoff_selection,best_payoff = bf.computePayoff(G, k)
        print("Brute Force payoff: ", best_payoff_selection, best_payoff)

        #run linear program
        lp.lp_setup(G, k)

        blp.solve_lp(G, k)

        
    results_details.close()
    with open(FILE_DIRECTORY_PREFIX + "compare_results.txt", 'a') as results:
        results.write('\n'+ str(max_val_greedyDP[0]) + '\t\t\t' + str(payoff) + '\t\t\t' + str(payoff_root) + ' ' + str(payoff_no_root))
    results.close()
    printGraph(G)

""" display graph """
def printGraph(G):
    print("printing graph")
    plt.figure("normal cluster graph")
    pos = nx.spring_layout(G)
    nx.draw(G, pos)

    node_labels = nx.get_node_attributes(G,'weight')
    # do (id, weight) pair for lable instead of just weight
    for key,val in node_labels.items():
        node_labels[key] = (key,val)
    nx.draw_networkx_labels(G, pos=pos, labels=node_labels)
    data_info = nx.get_edge_attributes(G,'data') # edge lables rejecting node
    weight_info = nx.get_edge_attributes(G,'weight') # edge lables rejecting node
    edge_labels = {}
    for key in weight_info.keys():
        if key in data_info.keys():
            edge_labels[key] = (data_info[key],weight_info[key])
        else:
            edge_labels[key] = ("na",weight_info[key])
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)

    #edge_labels = nx.get_edge_attributes(G_DP,'weight')
    # nx.draw_networkx_edge_labels(G, pos)
    plt.savefig(FILE_DIRECTORY_PREFIX + "this.png")
    plt.show()

def store_info(G,k):
    print('\nNext Test:\n')
    with open(FILE_DIRECTORY_PREFIX + "cluster_graph_details.txt", 'w') as graph_info:
        timestamp = datetime.timestamp(datetime.now())
        date = datetime.fromtimestamp(timestamp)
        graph_info.write("c\n")
        graph_info.write("# Timestamp: " + str(date) + "\n")
        graph_info.write("# Nodes: " + str(G.number_of_nodes()) + "\n")
        data = G.edges.data()
        graph_info.write("# Edges: " + str(len(data)))
        weights = G.nodes.data('weight')
        for node in weights:
            #print(node)
            graph_info.write("\n" + str(node[1]))
        for item in data:
            graph_info.write("\n" + str(item[0]) + " " + str(item[1]) + " " + str(item[2]['weight']))
            try:
                data = item[2]['data']
                for reject in data:
                    graph_info.write(" " + str(reject))
            except:
                pass
            #print(item)
    #cc.makeMatrix(G,k)
#main function, used for calling things
def main():
    runTests()

if __name__== "__main__":
  main()