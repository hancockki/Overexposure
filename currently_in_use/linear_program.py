import pulp
import dynamic_programming_cleaned as dp

def get_input_graph():
    """
    Make a cluster graph to run the lp on
    """
    G_cluster, G_cluster2, tree_decomp = dp.testOriginaltoCluster(12,0.5)
    dp.print_info(G_cluster2)
    return G_cluster2

def lp_setup(G):
    """
    Set up our LP based on the cluster data. We want each cluster and rejecting node to be a variable, 
    with the constraint that if you pick a cluster you must pick the rejecting nodes it is connected to.
    """
    maximize_payoff = pulp.LpProblem("Overexposure_Maximization", pulp.LpMaximize)
    cluster_variables = []
    edge_variables = []
    # loop through edge data to make our constraints for the lp
    for edge_data in G.edges.data():
        rej_node = edge_data[2]['data'][0]
        if edge_data[0] not in cluster_variables:
            continue

get_input_graph()
# define our LP problem

"""
We want to maximize the sum of nodes minus the sum of edges for each cluster. We use a bipartite graph and ensure that
we do not double count each edge by ensuring that once we subtract an edge once, we cannot subtract it again.

Thus, we have:

max (Sum(x_i w_i) - Sum(j_e w_e)) such that Sum(x_i <= k)

could we also make variables for the edges, and have a constraint where the sum of edges incoming to each rejecting
node is 1, meaning we cannot count the node more than once? 

constraints:
    if we take a cluster we must take each rejecting node it is connected to
    we cannot take the same rejecting node more than once
"""