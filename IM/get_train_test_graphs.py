import networkx as nx
from functions import relabel_graph
import numpy as np
import random
import os
import sys
import getopt

np.random.seed(1)
random.seed(1)

graph_name = "wiki"
EDGES = False
args = sys.argv[1:]
opts, args = getopt.getopt(args, "g:e:")
for opt, arg in opts:
    if opt in ["-g"]:
        graph_name = arg
    elif opt in ["-e"]:
        EDGES = bool(int(arg))

if not os.path.isdir(f"{graph_name}_train/"):
    os.mkdir(f"{graph_name}_train/")

if not os.path.isdir(f"{graph_name}_test/"):
    os.mkdir(f"{graph_name}_test/")

n = 0.2  # proportion to be used as train graph

graph = nx.read_gpickle(f"{graph_name}/main")
graph = relabel_graph(graph)

if EDGES:

    edges = np.array(graph.edges())
    indices = [i for i in range(len(edges))]
    random.shuffle(indices)
    train_ids = edges[indices[:int(n * graph.number_of_edges())]]
    test_ids = edges[indices[int(n * graph.number_of_edges()):]]

    train_graph = nx.DiGraph()
    train_graph.add_edges_from(train_ids)
    test_graph = nx.DiGraph()
    test_graph.add_edges_from(test_ids)

    for edge in train_graph.edges():
        u, v = edge
        train_graph[u][v]['weight'] = graph[u][v]['weight']

    for edge in test_graph.edges():
        u, v = edge
        test_graph[u][v]['weight'] = graph[u][v]['weight']

    train_graph = relabel_graph(train_graph)
    test_graph = relabel_graph(test_graph)
    nx.write_gpickle(train_graph, f"{graph_name}_train/main")
    nx.write_gpickle(test_graph, f"{graph_name}_test/main")

else:
    n = int(graph.number_of_nodes() * n)
    nodes = list(graph.nodes())
    random.shuffle(nodes)
    train_nodes = nodes[:n]
    test_nodes = nodes[n:]

    train_subgraph = nx.DiGraph(nx.subgraph(graph, train_nodes))
    for node in list(train_subgraph.nodes()):
        if train_subgraph.degree(node) == 0:
            train_subgraph.remove_node(node)

    test_subgraph = nx.DiGraph()
    edges_to_add = []
    for u, v in graph.edges():
        if (u, v) not in train_subgraph.edges():
            edges_to_add.append((u, v, graph[u][v]['weight']))
    test_subgraph.add_weighted_edges_from(edges_to_add)

    train_subgraph = relabel_graph(train_subgraph)
    test_subgraph = relabel_graph(test_subgraph)

    print(f"train graph is {train_subgraph.number_of_edges() / test_subgraph.number_of_edges() * 100}% of the test graph")

    nx.write_gpickle(train_subgraph, f"{graph_name}_train/main")
    nx.write_gpickle(test_subgraph, f"{graph_name}_test/main")
