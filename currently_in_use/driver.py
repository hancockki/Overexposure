import create_clusters as cc
import DP_algorithms as dp
import greedy_approx_algorithms as greedy
import brute_force as bf
import networkx as nx
import matplotlib.pyplot as plt
import sys
import linear_program as lp
import time

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
    #create cluster greaph
    #G = cc.testOriginaltoCluster(30, 0.7, 3)
    num_seeds = 10
    G = cc.createClusterGraph(50, 5)
    #compute payoff for greedy DP
    max_val_greedyDP = dp.greedyDP(G, G.number_of_nodes(), num_seeds)
    with open("currently_in_use/results_details.txt", "a") as results_details:
        store_info(G,20)
        print("\nGreedy DP Payoff: ", max_val_greedyDP)
        
        #compute payoff for most basic greedy algorithm
        greedy_seedset, payoff = greedy.kHighestClusters(G, num_seeds)
        print("Greedy Approach Seeds Chosen:", greedy_seedset, " with payoff: ", payoff)

        #compute payoff for recursive DP
        payoff_root, payoff_no_root = dp.runRecursiveDP(G, num_seeds)
        print("Recursive DP payoff: \n Root: ", payoff_root, "\n No Root: ", payoff_no_root)

        #compute payoff using brute force algorithm
        best_payoff_selection,best_payoff = bf.computePayoff(G, num_seeds, False)
        print("Brute Force payoff: ", best_payoff_selection, best_payoff)

        #run linear program
        lp.lp_setup(G, num_seeds)
    results_details.close()
    with open('compare_results.txt', 'a') as results:
        results.write('\n'+ str(max_val_greedyDP[0]) + '\t\t\t' + str(payoff) + '\t\t\t' + str(payoff_root) + ' ' + str(payoff_no_root))
    results.close()
    printGraph(G)

""" display graph """
def printGraph(G):
    print("printing graph")
    plt.figure(2)
    pos = nx.spring_layout(G)
    nx.draw(G, pos)

    node_labels = nx.get_node_attributes(G,'weight')
    nx.draw_networkx_labels(G, pos=pos, labels=node_labels)
    edge_labels = nx.get_edge_attributes(G,'data') # edge lables rejecting node
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)

    #edge_labels = nx.get_edge_attributes(G_DP,'weight')
    # nx.draw_networkx_edge_labels(G, pos)
    plt.savefig('this.png')
    plt.show()

def store_info(G,k):
    print('\nNext Test:\n')
    with open("currently_in_use/cluster_graph_details.txt", 'w') as graph_info:
        graph_info.write("Timestamp: " + str(time.time()) + "\n")
        graph_info.write("Nodes: " + str(G.number_of_nodes()) + "\n")
        data = G.edges.data()
        graph_info.write("Edges: " + str(len(data)))
        weights = G.nodes.data('weight')
        for node in weights:
            #print(node)
            graph_info.write("\n" + str(node[0]) + " " + str(node[1]))
        for item in data:
            graph_info.write("\n" + str(item[0]) + " " + str(item[1]) + " " + str(item[2]['weight']))
            #print(item)
    #cc.makeMatrix(G,k)
#main function, used for calling things
def main():
    runTests()

if __name__== "__main__":
  main()