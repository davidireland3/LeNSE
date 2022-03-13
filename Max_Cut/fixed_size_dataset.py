import networkx as nx
import random
import numpy as np
from functions import make_graph_features_for_encoder, get_fixed_size_subgraphs, cut_value, max_cut_heuristic
import time
import pickle
import sys
import getopt
import glob
import os


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    NUM_SAMPLES = 1
    NUM_CHECKPOINTS = 1
    BUDGET = 100
    FIXED_SIZE = 10000
    graph_name = "amazon_train"
    args = sys.argv[1:]
    opts, args = getopt.getopt(args, "g:n:b:c:f:")
    for opt, arg in opts:
        if opt in ['-g']:
            graph_name = arg
        elif opt in ['-b']:
            BUDGET = int(arg)
        elif opt in ['-n']:
            NUM_SAMPLES = int(arg)
        elif opt in ['-c']:
            NUM_CHECKPOINTS = int(arg)
        elif opt in ['-f']:
            FIXED_SIZE = int(arg)
    print(graph_name)

    graph = nx.read_gpickle(f"{graph_name}/main")
    try:
        with open(f"{graph_name}/budget_{BUDGET}/score_and_seeds", mode="rb") as f:
            good_seeds, best_score = pickle.load(f)

    except FileNotFoundError:
        scores = []
        start = time.time()
        good_seeds = max_cut_heuristic(graph, BUDGET)
        best_score = cut_value(graph, good_seeds)
        end = time.time()
        print(f"It took {(end - start) / 60:.3f} minutes\n")
        if not os.path.isdir(f"{graph_name}/budget_{BUDGET}/"):
            os.mkdir(f"{graph_name}/budget_{BUDGET}/")
        with open(f"{graph_name}/budget_{BUDGET}/score_and_seeds", mode="wb") as f:
            pickle.dump((good_seeds, best_score), f)

    graph_features = make_graph_features_for_encoder(graph, graph_name)
    N_PER_LOOP = NUM_SAMPLES // NUM_CHECKPOINTS
    count = 0
    for i in range(NUM_CHECKPOINTS):
        count += 1
        print("checkpoint", count)
        subgraphs = get_fixed_size_subgraphs(graph, good_seeds, N_PER_LOOP, BUDGET, FIXED_SIZE, best_score, graph_features, 1)
        with open(f"{graph_name}/budget_{BUDGET}/data_{count}", mode="wb") as f:
            pickle.dump(subgraphs, f)
        del subgraphs

    for i in range(NUM_CHECKPOINTS):
        count += 1
        print("checkpoint", count)
        subgraphs = get_fixed_size_subgraphs(graph, good_seeds, N_PER_LOOP, BUDGET, FIXED_SIZE, best_score, graph_features, 2)
        with open(f"{graph_name}/budget_{BUDGET}/data_{count}", mode="wb") as f:
            pickle.dump(subgraphs, f)
        del subgraphs

    for i in range(NUM_CHECKPOINTS):
        count += 1
        print("checkpoint", count)
        subgraphs = get_fixed_size_subgraphs(graph, good_seeds, N_PER_LOOP, BUDGET, FIXED_SIZE, best_score, graph_features, 3)
        with open(f"{graph_name}/budget_{BUDGET}/data_{count}", mode="wb") as f:
            pickle.dump(subgraphs, f)
        del subgraphs

    for i in range(NUM_CHECKPOINTS):
        count += 1
        print("checkpoint", count)
        subgraphs = get_fixed_size_subgraphs(graph, good_seeds, N_PER_LOOP, BUDGET, FIXED_SIZE, best_score, graph_features, 4)
        with open(f"{graph_name}/budget_{BUDGET}/data_{count}", mode="wb") as f:
            pickle.dump(subgraphs, f)
        del subgraphs

    subgraphs = []
    for fname in glob.glob(f"{graph_name}/budget_{BUDGET}/data_*"):
        with open(fname, mode="rb") as f:
            hold = pickle.load(f)
            subgraphs += hold

    with open(f"{graph_name}/budget_{BUDGET}/graph_data", mode="wb") as f:
        pickle.dump(subgraphs, f)

    for fname in glob.glob(f"{graph_name}/budget_{BUDGET}/data_*"):
        os.remove(fname)
