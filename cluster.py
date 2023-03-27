import numpy as np
import networkx as nx
from networkx.algorithms import community

# print(distance)
can={'12','34','24'}
cans=list(can)
# print(cans)
G=nx.Graph()
nodes=[]
for i in range(len(cans)):
    # print(cans[i])
    s=cans[i]
    s2=list(s)
    # print(s2)
    pairs=[]
    for j in range(0,len(s2)):
        # print(s2[j])
        n=s2[j]
        nint= int(n)
        nint=nint-1
        pairs.append(nint)
    r=pairs[0]
    c=pairs[1]
    for ele in nodes:
        if r not in nodes:
            nodes.append(r)
            G.add_node(r)
        if c not in nodes:
             nodes.append(c)
             G.add_node(c)


    G.add_edge(r,c)

print(G)

# G=nx.Graph()
# G.add_nodes_from([1,4])
# G.add_edge(1,2)
# G.add_edge(2,3)
# G.add_edge(3,4)
# print(G.number_of_edges())
# print(community.modularity(G))
# nx.flow_hierarchy(G,weigh=None)
# Authors: Gael Varoquaux, Nelle Varoquaux
# License: BSD 3 clause

# import time
# import matplotlib.pyplot as plt
# import numpy as np
#
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.neighbors import kneighbors_graph
#
# # Generate sample data
# n_samples = 1500
# np.random.seed(0)
# t = 1.5 * np.pi * (1 + 3 * np.random.rand(1, n_samples))
# x = t * np.cos(t)
# y = t * np.sin(t)
#
#
# X = np.concatenate((x, y))
# X += .7 * np.random.randn(2, n_samples)
# X = X.T

# Create a graph capturing local connectivity. Larger number of neighbors
# will give more homogeneous clusters to the cost of computation
# time. A very large number of neighbors gives more evenly distributed
# cluster sizes, but may not impose the local manifold structure of
# the data