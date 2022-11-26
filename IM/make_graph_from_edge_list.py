import networkx as nx
from functions import relabel_graph
import numpy as np

np.random.seed(1)
graph_name = "wiki"

f = open(f"{graph_name}/edges.txt", mode="r")
lines = f.readlines()
edges = []
for line in lines:
    line = line.split()
    edges.append([int(line[0]), int(line[1])])

graph = nx.DiGraph()
graph.add_edges_from(edges)

graph, trans = relabel_graph(graph, True)

e_weights = np.random.choice([0.1, 0.01, 0.001], size=len(edges))
for edge, weight in zip(graph.edges(), e_weights):
    u, v = edge
    graph[u][v]['weight'] = weight

nx.write_gpickle(graph, f"{graph_name}/main")
