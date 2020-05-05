import networkx as nx
from operator import itemgetter
import random
import matplotlib.pyplot as plt
import itertools
import copy
from pathlib import Path
#import pygraphviz as pgv
from itertools import combinations
#from IPython.display import Image

# ### Useful links
# #### Overleaf Project
# - https://www.overleaf.com/16173247hqfsnqsvshqs#/61849223/
# 
# #### NetworkX Source Code
# - https://networkx.github.io/documentation/stable/_modules/networkx/classes/function.html#degree
# - https://networkx.github.io/documentation/stable/_modules/networkx/classes/function.html#set_node_attributes
# 
# #### NetworkX Public API
# - https://networkx.github.io/documentation/stable/reference/classes/generated/networkx.Graph.degree.html
# 
# #### SNAP
# - https://snap.stanford.edu/

# # SETUP

# ## Generate graphs

# In[33]:


def buildRandomGraph(size, probability):
    
    # Wrapper method. Build a random (Erdos) graph of given size and probability of edge connection.
    
    G = nx.gnp_random_graph(size, probability)
    return G

def buildCavemanGraph(clique, size):

    # Wrapper method to build a caveman graph

    G = nx.connected_caveman_graph(clique, size)
    return G

def buildRandomPartitionGraph(sizes, p_in, p_out):

    #create random partition graph

    G = nx.random_partition_graph(sizes, p_in, p_out)
    return G

G_third = nx.Graph()

G_third.add_edge(1,2)
G_third.add_edge(2,3)

G_third.add_edge(1,4)
G_third.add_edge(1,5)
G_third.add_edge(1,6)
G_third.add_edge(1,7)

G_third.add_edge(2,4)
G_third.add_edge(2,5)
G_third.add_edge(2,6)
G_third.add_edge(2,7)

G_third.add_edge(3,4)
G_third.add_edge(3,5)
G_third.add_edge(3,6)
G_third.add_edge(3,7)

G_third.add_edge(8,9)
G_third.add_edge(9,10)

G_third.add_edge(8,4)
G_third.add_edge(8,5)
G_third.add_edge(8,6)
G_third.add_edge(8,7)
G_third.add_edge(8,11)
G_third.add_edge(8,12)
G_third.add_edge(8,13)
G_third.add_edge(8,14)

G_third.add_edge(9,4)
G_third.add_edge(9,5)
G_third.add_edge(9,6)
G_third.add_edge(9,7)
G_third.add_edge(9,11)
G_third.add_edge(9,12)
G_third.add_edge(9,13)
G_third.add_edge(9,14)

G_third.add_edge(10,4)
G_third.add_edge(10,5)
G_third.add_edge(10,6)
G_third.add_edge(10,7)
G_third.add_edge(10,11)
G_third.add_edge(10,12)
G_third.add_edge(10,13)
G_third.add_edge(10,14)

G_third.add_edge(15,16)
G_third.add_edge(16,17)

G_third.add_edge(15,11)
G_third.add_edge(15,12)
G_third.add_edge(15,13)
G_third.add_edge(15,14)

G_third.add_edge(16,11)
G_third.add_edge(16,12)
G_third.add_edge(16,13)
G_third.add_edge(16,14)

G_third.add_edge(17,11)
G_third.add_edge(17,12)
G_third.add_edge(17,13)
G_third.add_edge(17,14)

G_third.node[1]['criticality'] = [0.2,0.4,0.6]
G_third.node[2]['criticality'] = [0.2,0.4,0.6]
G_third.node[3]['criticality'] = [0.2,0.4,0.6]

G_third.node[4]['criticality'] = [0.6,0.8,1]
G_third.node[5]['criticality'] = [0.6,0.8,1]
G_third.node[6]['criticality'] = [0.6,0.8,1]
G_third.node[7]['criticality'] = [0.6,0.8,1]

G_third.node[8]['criticality'] = [0.2,0.4,0.6]
G_third.node[9]['criticality'] = [0.2,0.4,0.6]
G_third.node[10]['criticality'] = [0.2,0.4,0.6]

G_third.node[11]['criticality'] = [0.6,0.8,1]
G_third.node[12]['criticality'] = [0.6,0.8,1]
G_third.node[13]['criticality'] = [0.6,0.8,1]
G_third.node[14]['criticality'] = [0.6,0.8,1]

G_third.node[15]['criticality'] = [0.2,0.4,0.6]
G_third.node[16]['criticality'] = [0.2,0.4,0.6]
G_third.node[17]['criticality'] = [0.2,0.4,0.6]

#A = nx.nx_agraph.to_agraph(G_paper)
#A.layout(prog='dot')


# ### Set note attributes

# In[37]:
"""
def setPartitionNodeAttributes(partition_list, G):
    nodeList = G.nodes()
    print(nodeList)
    nodeNum = 0
    for nodeID in nodeList:
        G.node[nodeID]["visited"] = False
        G.node[nodeID]["cluster"] = -1
    add = 0
    for num in partition_list:
        print(num)
        print(nodeNum)
        
        for i in range(0, num):
            parameters = [random.uniform(0,0.2+add), random.uniform(0,0.4+add), random.uniform(0,0.6+add)]
            parameters.sort()
            G.node[nodeID]['criticality'] = parameters[1] if parameters[1] < 1 else 1
            G.node[nodeID]['lower bound'] = parameters[0] if parameters[0] < 1 else 1
            G.node[nodeID]['upper bound'] = parameters[2] if parameters[2] < 1 else 1
            nodeNum += 1
        if add < 0.5:
            add += 0.05
        print("current ID is:", i)

"""
# # ALGORITHMS

# ### 1. Degree Naive Approach: 
# #### Choose K nodes with positive accepting degree

# ### 2. Clustered Naive Approach:
# #### Build all clusters of accepting nodes. Select 1 node from cluster that maximizes payoff and go down from there.

# In[42]:

"""
def updateCriticality(G, source, appeal):
    numAccepting = 0
    neighborList = nx.neighbors(G, source)
    numNeighbors = len(neighborList)
    k = 0.02
    for neighbors in neighborList:
        if G.node[neighbors]['criticality'] < appeal:
            numAccepting -= k

        else:
            numAccepting += k
    #print("difference is", numAccepting)
    newCrit = G.node[source]['criticality']+ numAccepting
    if newCrit >= 1:
        newCrit = 1
    if newCrit <= 0:
        newCrit = 0
    G.node[source]['criticality'] = newCrit

"""
# ### 3. Shared-Rejecting Nodes

# #1. Build clusters
# #2. Build dictionary of rejecting nodes and the clusters that touch them.
# #3. Call any non-unique entries in dictionary <br>
#      i.e., if <br>
#               3-> 0, 1 <br>
#               4-> 0, 1 <br>
#               5-> 0, 1 <br>
#     then only search one (e.g., remove 4, 5; keep only 3)
# #4. Compute payoff on each unique permutation. If it increases by exposing the permutation, expose it.

# In[45]:

"""
def cullNonDistinctRejectingNodes():
    
    # Delete any non-unique key, values from the rejectingNode dictionary.
    
    for key1, key2 in combinations(rejectingNodeDict.keys(), r = 2):
        if key1 in rejectingNodeDict and key2 in rejectingNodeDict:
            if rejectingNodeDict[key1] == rejectingNodeDict[key2]:
                del rejectingNodeDict[key2]


# In[46]:


def findSubsets(S,m):
    
    # S: The set for which you want to find subsets
    # m: The number of elements in the subset
    # Credit: 
    
    return set(itertools.combinations(S, m))


# In[77]:


ARBITRARY_NEG_NUMBER = -1000

def payoffOfUniqueSubsets(G, appeal):
    
    # This approach considers all clusters connected to rejecting nodes, running off the intuition that maybe
    # a cluster alone can't overpower the negative boundary nodes its connected to but clusters together could.
    # This approach is inspired by the counterexample given in the paper.
    # For each rejecting node with a unique set of clusters touching it, build the subsets of that set
    # testing how payoff improves.
    # The highest performing subset in exploring all subsets in the rejecting node dictionary is then used
    # as a seed set.
    
    bestSubsetPayoff = ARBITRARY_NEG_NUMBER
    bestSubset = []
        
    for rejectingNode, clusteredSet in rejectingNodeDict.items():
       
        for i in range(1, len(clusteredSet)+1):
            #Generate all subsets of size i (e.g., size 2)
            allSubsets = findSubsets(clusteredSet, i)
            #Keep track of the best performing seedset.
            bestMsubsetPayoff = ARBITRARY_NEG_NUMBER 
            bestMsubset = []
            #Search all m-sized subsets, e.g., (0,1), (1,2), (0,2), building a seed set from each.
            #Compute the payoff from each of these sets. We want the best performing overall.
            for mSubSet in allSubsets:
                seedSet = []
                for element in mSubSet:
                    seedSet.append(clusterDict[element])
                    payoffThisRun = computePayoff(G, seedSet, appeal)
                if payoffThisRun > bestMsubsetPayoff:
                    bestMsubsetPayoff = payoffThisRun
                    bestMsubset = seedSet
            if bestMsubsetPayoff > bestSubsetPayoff:
                bestSubsetPayoff = bestMsubsetPayoff
                bestSubset = bestMsubset
    
    if bestSubsetPayoff <= 0:
        return []
    return bestSubset

def runCombinedExperiment(G, appeal, k):
    # Rejecting Nodes
    clearVisitedNodesAndDictionaries(G)
    buildClusteredSet(G, appeal, k, thirdAlgorithm=True)
    cullNonDistinctRejectingNodes()
    combinedSeedSet = payoffOfUniqueSubsets(G, appeal)
    print("combined seed set is:", combinedSeedSet)
    combinedPayoff = computePayoff(G, combinedSeedSet, appeal)
    print ("COMBINED seed set produced payoff of: ", combinedPayoff, ". Seedset: ", combinedSeedSet)

def updateGraph(G, appeal):
    print("updating graph G")
    nodeList = G.nodes()
    for node in nodeList:
        updateCriticality(G, node, appeal)
    driver(G, appeal, 3)


partition_list = [10,20,40,30,30]
G_random_partition = buildRandomPartitionGraph(partition_list, 0.25, 0.01)
setPartitionNodeAttributes(partition_list, G_random_partition)
driver(G_random_partition, threshold, k)

"""


class TestGraphs(object):
    #constructor
    def __init__(self, G, k, appeal):
        self.G = G
        self.k = k
        self.appeal = appeal

    def setAllNodeAttributes(self):
        nodeList = self.G.nodes()
        for nodeID in nodeList:
            parameters = []
            parameters = [random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)]
            parameters.sort()
            #print(parameters)
            self.G.node[nodeID]["visited"] = False
            self.G.node[nodeID]['criticality'] = parameters[1]
            self.G.node[nodeID]['lower bound'] = parameters[0]
            self.G.node[nodeID]['upper bound'] = parameters[2]
            self.G.node[nodeID]["cluster"] = -1

    def setVisitedFalse(self):    
        nodeList = self.G.nodes()
        for nodeID in nodeList:
            self.G.node[nodeID]["visited"] = False

    def initClusters(self):
        nodeList = self.G.nodes()
        for nodeID in nodeList:
            self.G.node[nodeID]["cluster"] = -1

    def printNodeAttributes(self, G):
        nodeList = G.nodes()
        for nodeID in nodeList:
            print (nodeID, " : ", G.node[nodeID])

    def drawGraph(self):

        colorList = []

        nodeList = self.G.nodes()
        for nodeID in nodeList:
            if self.G.node[nodeID]['criticality'] < self.appeal:
                colorList.append('g')
            else: 
                colorList.append('r')
        
        nx.draw_networkx(self.G, pos=None, arrows=False, with_labels=True, node_color=colorList)
        plt.show()

    def computePayoff(self, seedSet, WOM=False):
    
        # Given a seed set of source nodes, compute the payoff of exposing each node in that seed set.
        # We perform BFS from each node, where an accepting node returns a payoff of +1 and rejecting node a payoff
        # of -1. 
        
        if len(seedSet) is 0:
            print ("Cannot compute payoff. Seed set is of size 0. Exiting...")
            return
        
        TestGraphs.setVisitedFalse(self)
        payoff = 0
        for nodeID in seedSet:
            if WOM:
                temp = TestGraphs.bfs(self, nodeID, WOM=True)
            else:
                temp = TestGraphs.bfs(self, nodeID, WOM=False)
            payoff+=temp
            
        return payoff

    def bfs(self, source, WOM=False):
        
        # From a source node, perform BFS based on criticality.
        
        payoff = 0
        
        if self.G.node[source]['visited'] == False:
            self.G.node[source]['visited'] = True
            payoff = 1 #we already know source is accepting
        else:
            return payoff
        
        queue = []
        if self.G.node[source]['upper bound'] < self.appeal:
            queue.append(source) 
        while queue:
            start = queue.pop(0)
            length = len(nx.neighbors(self.G, start))
            for neighbor in nx.neighbors(self.G, start):
                if WOM:
                    TestGraphs.posWOM(self, self.G, neighbor, length/(length*length))
                if self.G.node[neighbor]['visited'] == False:
                    if self.G.node[neighbor]['upper bound'] < self.appeal: #spread positive WOM
                        queue.append(neighbor)
                        payoff+=1

                    elif self.G.node[neighbor]['criticality'] < self.appeal <= self.G.node[neighbor]['upper bound']: #neutral
                        payoff += 1

                    elif self.G.node[neighbor]['lower bound'] < self.appeal <= self.G.node[neighbor]['criticality']: #did not like
                        payoff -= 1
                        if WOM:
                            TestGraphs.negWOM(self, self.G, neighbor)
                    elif self.appeal < self.G.node[neighbor]['lower bound']: #did not purchase product
                        payoff += 0
                        if WOM:
                            TestGraphs.negWOM(self, self.G, neighbor)
                    self.G.node[neighbor]['visited'] = True
        
        return payoff

    def posWOM(self, G, source, amount):
        #print("spreading posWOM amount", amount)
        G.node[source]['criticality'] -= amount if G.node[source]['criticality'] - amount > 0 else 0
        G.node[source]['lower bound'] -= amount if G.node[source]['lower bound'] - amount > 0 else 0
        G.node[source]['upper bound'] -= amount if G.node[source]['upper bound'] - amount > 0 else 0

    def negWOM(self, G, source):
        #print("spreading neg posWOM")
        spreadNegWOM = nx.neighbors(G, source)
        numNeighbors = len(spreadNegWOM)
        amount = numNeighbors / (numNeighbors*numNeighbors)
        #print("amount is : ", amount)
        for neighbor in spreadNegWOM:
            G.node[neighbor]['upper bound'] += amount if G.node[neighbor]['upper bound'] + amount < 1 else 1
            G.node[neighbor]['criticality'] += amount if G.node[neighbor]['criticality'] + amount < 1 else 1
            G.node[neighbor]['lower bound'] += amount if G.node[neighbor]['lower bound'] + amount < 1 else 1
            #print(G.node[neighbor]['criticality'])
class SeedingStrategies(TestGraphs):
    #constructor from parent class
    def __init__(self, G, k, appeal):
        TestGraphs.__init__(self, G, k, appeal)
    """
    def clearVisitedNodesAndDictionaries(G):
        setVisitedFalse(G)
        rejectingNodeDict.clear()
        clusterDict.clear()
"""

    def runNaiveExperiment(self, G, k, appeal): 
        # Naive
        TestGraphs.setVisitedFalse(self)
        seedSetNaive = SeedingStrategies.buildNaiveSeedSet(self, G, appeal, k)
        payoffNaive = TestGraphs.computePayoff(self, seedSetNaive, WOM=True)
        print ("NAIVE seed set produced payoff of: ", payoffNaive, ". Seedset: ", seedSetNaive)

    def runClusteredExperiment(self, G):
        # Now clustered
        TestGraphs.setVisitedFalse(self)
        seedSetClustered = SeedingStrategies.buildClusteredSet(self, self.G, self.appeal, self.k, False)
        print("clustered seed set is", seedSetClustered)
        payoffClustered = TestGraphs.computePayoff(self, seedSetClustered, WOM=True)
        print ("CLUSTERED seed set produced payoff of: ", payoffClustered, ". Seedset: ", seedSetClustered)

    def runDP(self, G, appeal, k):
        #clearVisitedNodesAndDictionaries(G)
        seedSet = SeedingStrategies.returnEntireSeedSet(self, self.G, appeal, self.k, thirdAlgorithm=False)
        print("DP Seed set: ", seedSet)
        seedSetDP = SeedingStrategies.tryDP(self, self.G, seedSet, self.appeal, self.k)
        payoffDP = TestGraphs.computePayoff(self, seedSetDP, WOM=True)
        print ("DP seed set produced payoff of: ", payoffDP, ". Seedset: ", seedSetDP)

    def labelClusters(self, G, source, clusterNumber, appeal, thirdAlgorithm=False):
    
        # Perform BFS to label a cluster. Note how we called setVisitedFalse(G) upon finding an unvisited node.
        # The idea is that clusters are "contained" - every cluster is separated from every other cluster by walls of
        # rejecting nodes. You will never find another cluster, because if you could, it would simply be one cluster.
        # However, we need to set rejecting nodes to visited so that we don't count them for each accepting neighbor.
        # This, however, is not problematic, because we label all nodes from the cluster from a single canonical source node.
        # We will only reach this method from another cluster - whereupon we call setVisitedFalse(), clearing the rejecting
        # nodes, allowing us to again "contain" this new cluster.
        
        if G.node[source]['visited'] == False:
            TestGraphs.setVisitedFalse(self)
            G.node[source]['visited'] = True
        else:
            return 0

        queue = []
        G.node[source]['cluster'] = clusterNumber
        if G.node[source]['upper bound'] < appeal:
            queue.append(source)
        #if thirdAlgorithm:
           # if clusterNumber not in clusterDict:
             #   clusterDict[clusterNumber] = source

        acceptingInThisCluster = 1 #count yourself, you matter!
        
        while queue:
            start = queue.pop(0)
            for neighbor in nx.neighbors(G, start):
                if G.node[neighbor]['visited'] == False:
                    if G.node[neighbor]['upper bound'] < appeal:
                        queue.append(neighbor)
                        G.node[neighbor]['cluster'] = clusterNumber
                        G.node[neighbor]["visited"] = True
                        acceptingInThisCluster += 1
                        
                        #if clusterNumber not in clusterDict:
                           # clusterDict[clusterNumber] = neighbor
                    elif G.node[neighbor]['criticality'] < appeal:
                        G.node[neighbor]['cluster'] = clusterNumber
                        G.node[neighbor]["visited"] = True
                    else:
                        G.node[neighbor]["visited"] = True
                        acceptingInThisCluster -= 1
                        #if (thirdAlgorithm == True):
                           # if neighbor not in rejectingNodeDict:
                            #    rejectingNodeDict[neighbor] = set()
                            #    rejectingNodeDict[neighbor].add(clusterNumber)
                            #else:
                            #    rejectingNodeDict[neighbor].add(clusterNumber)
        return acceptingInThisCluster

    def buildClusteredSet(self, G, threshold, k, thirdAlgorithm=False):

        # From each node in the nodeList, try to label its cluster. This will return 0 for many nodes, as they are labeled
        # from the (arbitrary) canonical node in its cluster.
        # We then select a seed set of up to (not always, depending on the composition of the graph) k source nodes.
        # We select the k source clusters with the highest accepting degree, implemented by sorting a list of tuples.
        
        TestGraphs.initClusters(self)
        TestGraphs.setVisitedFalse(self)
        nodeList = G.nodes()
        seedSet = []
        clusterCount = 0
        
        #Build the clusters
        for nodeID in nodeList:
            if (G.node[nodeID]['criticality'] < threshold) and (G.node[nodeID]['cluster'] == -1):
                summedNeighbors = SeedingStrategies.labelClusters(self, G, nodeID, clusterCount, threshold, thirdAlgorithm)
                if summedNeighbors > 0:
                    seedSet.append( (nodeID, summedNeighbors))
                clusterCount += 1
        #Choose up-to-k    
        kSet = SeedingStrategies.getseedSetFromTupleList(self, seedSet, k)
        return kSet     

    def returnEntireSeedSet(self, G, threshold, k, thirdAlgorithm=False):

        # Method that returns all of the clusters in the Seed Set, rather than just the k with the highest cardinality
        TestGraphs.initClusters(self)
        TestGraphs.setVisitedFalse(self)
        nodeList = G.nodes()
        #print("Node list ", nodeList)
        seedSet = []
        clusterCount = 0
        
        #Build the clusters
        for nodeID in nodeList:
            if (G.node[nodeID]['criticality'] < threshold) and (G.node[nodeID]['cluster'] == -1):
                summedNeighbors = SeedingStrategies.labelClusters(self, G, nodeID, clusterCount, threshold, thirdAlgorithm)
                #if summedNeighbors > 0:
                seedSet.append( (nodeID, summedNeighbors))
                clusterCount += 1
        #Choose up-to-k 
        print(seedSet)   
        return seedSet

    def tryDP(self, G, seedSet, phi, k):

        #Method that uses Dynamic Programming to find the optimal k nodes, using a greedy approach
        if k > len(seedSet):
            k = len(seedSet)
        kSet = []
        kSetTup = []
        payoff = -6
        i=0
        #rows, cols = (k+1,1000)
        #arr = [[0 for i in range(cols+1)] for j in range(rows+1)]
        # Consider up to k nodes as seeds, using a pseudo-greedy approach
        for seed in range(1, k+1): #number of seeds
            for node in seedSet: #test seed from each cluster
                kSet.append(node[0])
                print(kSet)
                payoff2 = TestGraphs.computePayoff(self, kSet)
                if payoff2 <= payoff: 
                    #print("last payoff was better")
                    #arr[seed][i+1]=arr[seed][i] #keep last payoff
                    i+=1
                    kSet.pop()
                else:
                    #print("new payoff is better")
                    #arr[seed][i+1]=payoff2
                    payoff=payoff2
                    if len(kSetTup) > 0:
                        kSetTup.pop()
                    kSetTup.append(node)
                    i+=1
                    kSet.pop()
            #kSet2.append(kSetTup[-1][0])
            if len(kSetTup) > 0:
                kSet.append(kSetTup[-1][0])
            print("kSet after this round is:", kSet)
            #print("kSet tup is:", kSetTup)
            if len(kSetTup) > 0:
                seedSet.remove(kSetTup[-1])
                kSetTup.pop()
            #print("seed set is:", seedSet)
        #print("payoff for DP in alg is:", payoff)
        #print("kset is:", kSet)
        if payoff > 0:
            return kSet
        else:
            return []

    def getseedSetFromTupleList(self, seedSet, k):
        
        # Given a seed set (a tuple list), choose up to K source nodes, if the source seed set is larger than K.
        # Return a new list, containing those k source nodes with the highest degreeAccepting.
        
        kSet = []
        if len(seedSet) > 0:
            numberOfClusters = k if k < len(seedSet) else len(seedSet)
            seedSet.sort(key=itemgetter(1))
            setLength = len(seedSet)-1
            for i in range(numberOfClusters):
                kSet.append(seedSet[setLength-i][0])
                
        return kSet

    def buildNaiveSeedSet(self, G, threshold, k):
        
        # Looks at every node in the graph - has no concept of "cluster" 
        # We consider a node for seedSet if its accepting degree is positive, where accepting degree
        # is the sum of the those neighbors who accept (i.e., criticality < threshold)
        #
        # The seed set is a list of tuples, identified by {nodeID, degreeAcceptingNeighbors}.
        # We build a list of tuples so that we can sort by degreeAcceptingNeighbors.
        
        nodeList = G.nodes()
        seedSet = []
        
        for nodeID in nodeList:
            if G.node[nodeID]['criticality'] > threshold:
                continue
            
            degreeAcceptingNeighbors = 1
            
            neighborList = nx.neighbors(G, nodeID)
            for neighborID in neighborList:
                if G.node[neighborID]['criticality'] < threshold:
                    degreeAcceptingNeighbors += 1
                else:
                    degreeAcceptingNeighbors -=1
            
            if degreeAcceptingNeighbors >= 0: #>= because you get 0 PLUS the payoff of yourself
                seedSet.append((nodeID, degreeAcceptingNeighbors))
        
        #Now get the naive seed set up to K
        kSet = SeedingStrategies.getseedSetFromTupleList(self, seedSet, k)
        return kSet     

def runCaveman():
    j = 3
    for i in range(5, 15, 2):
        G_caveman = buildCavemanGraph(i, j)
        graph = SeedingStrategies(G_caveman, 3, 0.6)
        graph.setAllNodeAttributes()
        G_DP = copy.deepcopy(G_caveman)
        graph2 = SeedingStrategies(G_DP, 3, 0.6)
        #graph2.setAllNodeAttributes()
        #graph.printNodeAttributes(G_caveman)
        graph.runClusteredExperiment(G_caveman)
        #graph2.printNodeAttributes(G_DP)
        graph2.runDP(G_caveman, 0.6, 3)
        graph.drawGraph()
        #updateGraph(G_caveman, threshold)
        j+=2

#runCaveman()
file = Path("C:/Users/Owner/Desktop/Overexposure Summer Research/facebook_combined.txt", encoding='utf-8')
print(file)
file = str(file)
G = nx.read_edgelist(file, create_using=nx.Graph())


"""
def driver(self):
    #runNaiveExperiment(G, appeal, k)
    SeedingStrategies.runClusteredExperiment(self)
    #SeedingStrategies.runCombinedExperiment(self)
    SeedingStrategies.runDP(self, self.G, self.appeal, self.k)
    TestGraphs.drawGraph(self)

        """
        