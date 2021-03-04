import pulp
import create_clusters as cc

def lp_setup(G):
    """
    Set up our LP based on the cluster data. We want each cluster and rejecting node to be a variable, 
    with the constraint that if you pick a cluster you must pick the rejecting nodes it is connected to.
    """
    NODES = EDGES = range(G.number_of_nodes())
    maximize_payoff = pulp.LpProblem("Overexposure_Maximization", pulp.LpMaximize)
    choices = pulp.LpVariable.dicts("Choice", (NODES), cat="Integer")
    cluster_variables = []
    edge_variables = []
    # loop through edge data to make our constraints for the lp
    for edge_data in G.edges.data():
        rej_node = edge_data[2]['weight']
        if edge_data[0] not in cluster_variables:
            continue

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