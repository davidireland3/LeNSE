from IMM import imm
import networkx as nx
import random
import numpy as np
from functions import make_graph_features_for_encoder, expected_spread, close_pool, get_fixed_size_subgraphs
import multiprocessing as mp
import time
import pickle
import sys
import getopt
import glob
import os


if __name__ == '__main__':
    pool = mp.Pool()
    random.seed(1)
    np.random.seed(1)
    NUM_SAMPLES = 5
    NUM_CHECKPOINTS = 1
    BUDGET = 50
    FIXED_SIZE = 50
    graph_name = "wiki_train"
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
            good_seeds, best_score, counts = pickle.load(f)

    except FileNotFoundError:
        counts = {}
        all_seeds = set()
        scores = []
        for i in range(10):
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
        if not os.path.isdir(f"{graph_name}/budget_{BUDGET}/"):
            os.mkdir(f"{graph_name}/budget_{BUDGET}/")
        with open(f"{graph_name}/budget_{BUDGET}/score_and_seeds", mode="wb") as f:
            pickle.dump((good_seeds, best_score, counts), f)

    graph_features = make_graph_features_for_encoder(graph, graph_name)
    N_PER_LOOP = NUM_SAMPLES // NUM_CHECKPOINTS
    count = 0
    # for i in range(NUM_CHECKPOINTS):
    #     count += 1
    #     print("checkpoint", count)
    #     subgraphs = get_fixed_size_subgraphs(graph, good_seeds, N_PER_LOOP, counts, BUDGET, FIXED_SIZE, pool, graph_name, best_score, graph_features, 1)
    #     with open(f"{graph_name}/budget_{BUDGET}/data_{count}", mode="wb") as f:
    #         pickle.dump(subgraphs, f)
    #     del subgraphs
    #
    # for i in range(NUM_CHECKPOINTS):
    #     count += 1
    #     print("checkpoint", count)
    #     subgraphs = get_fixed_size_subgraphs(graph, good_seeds, N_PER_LOOP, counts, BUDGET, FIXED_SIZE, pool, graph_name, best_score, graph_features, 2)
    #     with open(f"{graph_name}/budget_{BUDGET}/data_{count}", mode="wb") as f:
    #         pickle.dump(subgraphs, f)
    #     del subgraphs

    for i in range(NUM_CHECKPOINTS):
        count += 1
        print("checkpoint", count)
        subgraphs = get_fixed_size_subgraphs(graph, good_seeds, N_PER_LOOP, counts, BUDGET, FIXED_SIZE, pool, graph_name, best_score, graph_features, 3)
        with open(f"{graph_name}/budget_{BUDGET}/data_{count}", mode="wb") as f:
            pickle.dump(subgraphs, f)
        del subgraphs

    # for i in range(NUM_CHECKPOINTS):
    #     count += 1
    #     print("checkpoint", count)
    #     subgraphs = get_fixed_size_subgraphs(graph, good_seeds, N_PER_LOOP, counts, BUDGET, FIXED_SIZE, pool, graph_name, best_score, graph_features, 4)
    #     with open(f"{graph_name}/budget_{BUDGET}/data_{count}", mode="wb") as f:
    #         pickle.dump(subgraphs, f)
    #     del subgraphs

    subgraphs = []
    for fname in glob.glob(f"{graph_name}/budget_{BUDGET}/data_*"):
        with open(fname, mode="rb") as f:
            hold = pickle.load(f)
            subgraphs += hold

    with open(f"{graph_name}/budget_{BUDGET}/graph_data", mode="wb") as f:
        pickle.dump(subgraphs, f)

    for fname in glob.glob(f"{graph_name}/budget_{BUDGET}/data_*"):
        os.remove(fname)

    close_pool(pool)
