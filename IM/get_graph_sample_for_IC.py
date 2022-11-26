import numpy as np
import networkx as nx
import multiprocessing as mp
import sys
import getopt
import os

graph = "wiki_test"
Type = "main"
num_samples = 1000
chunksize = 80
args = sys.argv[1:]
opts, args = getopt.getopt(args, "g:t:n:c:")
for opt, arg in opts:
    if opt in ['-g']:
        graph = arg
    elif opt in ["-t"]:
        Type = arg
    elif opt in ["-n"]:
        num_samples = int(arg)
    elif opt in ['c']:
        chunksize = int(arg)

if not os.path.isdir(f"{graph}/IC_main_subgraphs"):
    os.mkdir(f"{graph}/IC_main_subgraphs")


def make_subgraph(count, edges_, p_, graph_):
    outcomes = np.random.binomial(1, p=1 - p_)
    g2 = nx.DiGraph()
    edges_to_keep = list(map(tuple, edges_[outcomes == 0]))
    g2.add_edges_from(edges_to_keep)
    nx.write_gpickle(g2, f"{graph_}/IC_{'main'}_subgraphs/subgraph_{count}")


if __name__ == '__main__':
    np.random.seed(1)
    counts = [i for i in range(1, num_samples + 1)]

    Graph = nx.read_gpickle(f"{graph}/{Type}")

    p = [Graph[u][v]['weight'] for u, v in Graph.edges()]
    p = np.array(p)
    p = [p for i in range(len(counts))]
    edges = np.array(Graph.edges())
    edges = [edges for _ in range(len(counts))]
    graph = [graph for _ in range(len(counts))]
    del Graph

    with mp.Pool() as pool:
        pool.starmap(make_subgraph, zip(counts, edges, p, graph), chunksize=chunksize)
