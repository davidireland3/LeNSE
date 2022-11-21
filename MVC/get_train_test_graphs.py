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
    if opt in ['-g']:
        graph_name = arg
    elif opt in ['-e']:
        EDGES = bool(int(arg))


if not os.path.isdir(f"{graph_name}_train/"):
    os.mkdir(f"{graph_name}_train/")

if not os.path.isdir(f"{graph_name}_test/"):
    os.mkdir(f"{graph_name}_test/")

n = 0.2  # proportion to be used as train graph

graph = nx.read_gpickle(f"{graph_name}/main")
# graph = relabel_graph(graph)

if EDGES:

    edges = np.array(graph.edges())
    indices = [i for i in range(len(edges))]
    random.shuffle(indices)
    train_ids = edges[indices[:int(n * graph.number_of_edges())]]
    test_ids = edges[indices[int(n * graph.number_of_edges()):]]

    train_graph = nx.Graph()
    train_graph.add_edges_from(train_ids)
    for node in list(train_graph.nodes()):
        if train_graph.degree(node) == 0:
            print(node)
            train_graph.remove_node(node)

    test_graph = nx.Graph()
    test_graph.add_edges_from(test_ids)
    for node in list(test_graph.nodes()):
        if test_graph.degree(node) == 0:
            print(node)
            test_graph.remove_node(node)

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

    train_subgraph = nx.Graph(nx.subgraph(graph, train_nodes))
    for node in list(train_subgraph.nodes()):
        if train_subgraph.degree(node) == 0:
            train_subgraph.remove_node(node)

    test_subgraph = nx.Graph()
    edges_to_add = []
    for u, v in graph.edges():
        if (u, v) not in train_subgraph.edges():
            edges_to_add.append((u, v))
    test_subgraph.add_edges_from(edges_to_add)

    train_subgraph = relabel_graph(train_subgraph)
    test_subgraph = relabel_graph(test_subgraph)
    nx.write_gpickle(train_subgraph, f"{graph_name}_train/main")
    nx.write_gpickle(test_subgraph, f"{graph_name}_test/main")

