import create_clusters as cc
import DP_algorithms as dp
import greedy_approx_algorithms as greedy
import networkx as nx
import matplotlib.pyplot as plt
import sys

def runTests():
    #create cluster greaph
    #G = cc.testOriginaltoCluster(20, 0.7, 3)
    G = cc.createClusterGraph(20, 10)
    #compute payoff for greedy DP
    max_val_greedyDP = dp.greedyDP(G, G.number_of_nodes(), 3)
    sys.stdout = open("currently_in_use/results.txt", "w")
    #print info about the graph
    print_info(G)
    print("\nGreedy DP Payoff: ", max_val_greedyDP)

    #compute payoff for most basic greedy algorithm
    greedy_seedset, payoff = greedy.kHighestClusters(G, 3)
    print("Greedy Approach Seeds Chosen:", greedy_seedset, " with payoff: ", payoff)

    #compute payoff for recursive DP
    payoff_root, payoff_no_root = dp.runRecursiveDP(G, 3)
    print("Recursive DP payoff: \n Root: ", payoff_root, "\n No Root: ", payoff_no_root)
    sys.stdout.close()
    #print graph
    plt.figure(2)
    pos = nx.spring_layout(G)
    nx.draw(G, pos)

    node_labels = nx.get_node_attributes(G,'weight')
    nx.draw_networkx_labels(G, pos, labels = node_labels)

    #edge_labels = nx.get_edge_attributes(G_DP,'weight')
    nx.draw_networkx_edge_labels(G, pos)
    plt.savefig('this.png')
    plt.show()

def print_info(G):
    data = G.edges.data()
    for item in data:
       print("Edge between", item[0], item[1], " Weight: ", item[2]['weight'])

#main function, used for calling things
def main():
    runTests()

if __name__== "__main__":
  main()