import networkx as nx
import random
import numpy as np
from functions import cut_value, max_cut_heuristic, make_graph_features_for_encoder
import time
import pickle
import sys
import getopt
import os


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    BUDGET = 100
    graph_name = "wiki_test"
    args = sys.argv[1:]
    opts, args = getopt.getopt(args, "g:b:")
    for opt, arg in opts:
        if opt in ['-g']:
            graph_name = arg
        elif opt in ['-b']:
            BUDGET = int(arg)
    print(graph_name)

    graph = nx.read_gpickle(f"{graph_name}/main")
    start = time.time()
    good_seeds = max_cut_heuristic(graph, BUDGET)
    best_score = cut_value(graph, good_seeds)
    end = time.time()
    print(f"It took {(end - start) / 60:.3f} minutes\n")

    graph_features = make_graph_features_for_encoder(graph, graph_name)

    if not os.path.isdir(f"{graph_name}/budget_{BUDGET}/"):
        os.mkdir(f"{graph_name}/budget_{BUDGET}/")

    with open(f"{graph_name}/budget_{BUDGET}/score_and_seeds", mode="wb") as f:
        pickle.dump((good_seeds, best_score), f)

    with open(f"{graph_name}/budget_{BUDGET}/time_taken_to_get_seeds", mode='w') as f:
        f.write(f"It took {(end - start) / 60} minutes to get a solution.")
