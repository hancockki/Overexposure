import pulp
import dynamic_programming_cleaned

# define our LP problem
maximize_payoff = pulp.LpProblem("Overexposure Maximization", pulp.LpMaximize)

"""
We want to maximize the sum of nodes minus the sum of edges for each cluster. We use a bipartite graph and ensure that
we do not double count each edge by ensuring that once we subtract an edge once, we cannot subtract it again.

Thus, we have:

max (Sum(x_i w_i) - Sum(j_e w_e)) such that Sum(x_i <= k)

"""