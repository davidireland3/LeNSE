from IMM import imm
import networkx as nx
import random
import numpy as np
from functions import expected_spread, make_graph_features_for_encoder
import multiprocessing as mp
import time
import pickle
import sys
import getopt
import os


if __name__ == '__main__':
    pool = mp.Pool()
    random.seed(1)
    np.random.seed(1)
    BUDGET = 50
    graph_name = "wiki_train"
    args = sys.argv[1:]
    opts, args = getopt.getopt(args, "g:b:")
    for opt, arg in opts:
        if opt in ['-g']:
            graph_name = arg
        elif opt in ['-b']:
            BUDGET = int(arg)
    print(graph_name)

    graph = nx.read_gpickle(f"{graph_name}/main")
    counts = {}
    all_seeds = set()
    scores = []
    for i in range(1):
        print(i + 1)
        start = time.time()
        seeds, _ = imm(graph, BUDGET)
        for seed in seeds:
            if seed in counts:
                counts[seed] += 1
            else:
                counts[seed] = 1
        scores.append(expected_spread(seeds, graph=graph_name, pool=pool))
        end = time.time()
        print(f"It took {(end - start) / 60:.3f} minutes\n")
        all_seeds |= seeds
    good_seeds = all_seeds.copy()
    best_score = max(scores)

    graph_features = make_graph_features_for_encoder(graph, graph_name)

    if not os.path.isdir(f"{graph_name}/budget_{BUDGET}/"):
        os.mkdir(f"{graph_name}/budget_{BUDGET}/")

    with open(f"{graph_name}/budget_{BUDGET}/score_and_seeds", mode="wb") as f:
        pickle.dump((good_seeds, best_score, counts), f)

    with open(f"{graph_name}/budget_{BUDGET}/time_taken_to_get_seeds", mode='w') as f:
        f.write(f"It took {(end - start) / 60} minutes to get a solution.")
