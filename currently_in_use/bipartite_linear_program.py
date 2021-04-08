'''
Run linear programming with the input graph as a bipartite graph. To do this, we take the cluster graph and transform it 
into a bipartite graph. We do this in the script make_bipartite_graph (look for more details).
We then call solve_lp with the bipartite graph as input, where each rejecting node is a variable in the lp
and each accepting cluster is a variable. Note that this is different from the regular linear program, where
the variables are all the accepting clusters and all the edges in the graph (rather than here where each rejecting
node in the cluster graph is a node in the bipartite graph). We then know if we take a cluster we must take all the 
rejecting nodes which point to that cluster.

LP Formulation:
    Constants: k, the number of clusters can take

    Coeficients:
    w[i], which is the weight of cluster i (coef bc not being solved for)

    Variables:
    x[i] \in {all clusters}, which indicates whether or not cluster i taken
    y[r] \in {all reject nodes}, which indicates reject nodes taken

    Maximize Sum_i (x[i]w[i]) - Sum_r (y[r])
    Subject to
        Sum_i x[i] <= k (can't take more than k clusters)
        x[i] <= 1  for each cluster i (means can only take cluster i once. Bc of formulation x_i \in {0,1})
        x[i] <= y[r] for all "edges" (r,i) in the bipartite graph (I say "edges" because this can be literal or conceptual)
        -y[r] <= 0 means can't allow for negative taking of edges (more easliy understood as y[r] >= 0) 
'''

from pulp import *
import create_clusters as cc
import networkx as nx
import matplotlib.pyplot as plt
import brute_force as bf
import make_bipartite_graph as mbg

CONST_SUFFIX = "_constraint"

""" 
Called from witihin driver to solve the linear program.
Create variables for each cluster and rejecting node and then
map the weights of each edge onto it.
"""
def solve_lp(G,k):
    x_keys = []
    y_keys = []
    weight_dict = nx.get_node_attributes(G,'weight')
    print("Weight dictionary: ", weight_dict)
    print("\nNodes in bipartite \n", G.nodes(), "\nOut edges: \n", G.out_edges())
    for edge in G.edges():
        if edge[0] not in y_keys:
            y_keys.append(edge[0])
        if edge[1] not in x_keys:
            x_keys.append(edge[1])
    for node in G.nodes():
        if node not in x_keys:
            x_keys.append(node)

    #add x and y variables
    x = LpVariable.dicts("x", x_keys, lowBound=0, cat="Integer")
    y = LpVariable.dicts("y", y_keys, lowBound=0, cat="Integer")
    print("X_keys: ", x_keys, x, "\nY_keys: ", y_keys, y)

    # create lp with LpProblem
    lp = LpProblem("Bipartite_ILP", LpMaximize)

    # add in objective function w/ lpSum
    lp += lpSum(x[i] * weight_dict[i] for i in x_keys) - lpSum(y[i] for i in y_keys)
    
    # create constraints
    lp += lpSum(x[x_key] for x_key in x_keys) <= k, "max_cluster_select_of_" + str(k)

    for x_key in x_keys:
        lp += x[x_key] <= 1, "cluster_" + str(x_key) + CONST_SUFFIX

    for y_key in y_keys:
        for node in G.neighbors(y_key):
            #print("Neighbor of ", y_key, " is ", node)
            lp += y[y_key] >= x[node], "reject_" + str(y_key) + "_to_node_" + str(node) + CONST_SUFFIX

    # solve lp
    status = lp.solve(PULP_CBC_CMD(msg=0)) # PULP_CBC_CMD(msg=0)
    print("Status:",status)
    for var in lp.variables():
        if value(var) == 1:
            print(var,"=",value(var))
    print("OPT Bipartite=",value(lp.objective))
    return value(lp.objective)

'''
def main():
    n = 20
    c = 0.7
    k = 5
    #G = get_input_graph(n,c,k)
    #print(G.nodes)
    #print(G.edges)
    #solve_lp(G,k)
    print("Brute force:", bf.computePayoff(G,k))

    #print graph
    plt.figure('normal cluster graph')
    pos = nx.spring_layout(G)
    nx.draw(G, pos)

    node_labels = nx.get_node_attributes(G,'weight')
    for key,val in node_labels.items():
        node_labels[key] = (key,val)
    nx.draw_networkx_labels(G, pos=pos, labels=node_labels) # node lables are (id, weight) pair
    edge_labels = nx.get_edge_attributes(G,'data') # edge lables rejecting node
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)

    #edge_labels = nx.get_edge_attributes(G_DP,'weight')
    # nx.draw_networkx_edge_labels(G, pos)
    plt.savefig('this.png')
    plt.show()

    for edge in G.edges:
        try:
            # add reject node number to set (need to pop and replace so don't loose in other locations bc memory location the same)
            reject_node_set = G.edges[edge]['data']
            reject_node = reject_node_set.pop()
            reject_node_set.add(reject_node)
            
            temp = -1
            if reject_node in r_to_n_record:
                temp = r_to_n_record[reject_node]
            else:
                temp = set()
            temp.add(edge[0])
            temp.add(edge[1])
            r_to_n_record[reject_node] = temp
        except:
            pass
'''