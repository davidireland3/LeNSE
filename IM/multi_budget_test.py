from environment import TestEnvMultiBudget
import torch
from functions import close_pool
import networkx as nx
import pickle
import random
import sys
import getopt
import numpy as np
import os

if __name__ == "__main__":
    torch.random.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    train_graph_name = "wiki_train"
    test_graph_name = "wiki_test"
    num_eps = 1
    soln_budget = 100
    subgraph_size = 300
    encoder_name = "encoder"
    action_limit = 5000
    cuda = False
    budgets = [1, 10, 25, 50, 75]
    args = sys.argv[1:]
    opts, args = getopt.getopt(args, "g:n:b:a:t:f:C:E:B:")
    for opt, arg in opts:
        if opt in ['-g']:
            train_graph_name = arg
        elif opt in ["-n"]:
            num_eps = int(arg)
        elif opt in ["-b"]:
            soln_budget = int(arg)
        elif opt in ["-a"]:
            action_limit = int(arg)
        elif opt in ["-t"]:
            test_graph_name = arg
        elif opt in ["-f"]:
            subgraph_size = int(arg)
        elif opt in ["-C"]:
            cuda = bool(int(arg))
        elif opt in ["-E"]:
            encoder_name = arg
        elif opt in ["-B"]:
            budgets = arg.split(".")
            budgets = [int(b) for b in budgets]
            assert len(budgets) > 0, "budget length is 0!"

    if not os.path.isdir(f"{test_graph_name}/budget_{soln_budget}"):
        os.mkdir(f"{test_graph_name}/budget_{soln_budget}")

    if not os.path.isdir(f"{test_graph_name}/budget_{soln_budget}/{encoder_name}"):
        os.mkdir(f"{test_graph_name}/budget_{soln_budget}/{encoder_name}")

    encoder = torch.load(f"{train_graph_name}/budget_{soln_budget}/{encoder_name}/{encoder_name}", map_location=torch.device("cpu"))
    graph = nx.read_gpickle(f"{test_graph_name}/main")
    best_embeddings = None
    encoder.to("cpu")
    with open(f"{train_graph_name}/budget_{soln_budget}/{encoder_name}/trained_dqn", mode="rb") as f:
        dqn = pickle.load(f)
    dqn.epsilon = 0.01
    env = TestEnvMultiBudget(graph, soln_budget, subgraph_size, encoder, test_graph_name, action_limit=action_limit, beta=1, cuda=cuda, budgets=budgets)
    # env = BigGraphMultiBudget(graph, soln_budget, subgraph_size, encoder, test_graph_name, action_limit=action_limit, cuda=cuda, budgets=budgets)
    num_nodes = []
    num_edges = []
    for episode in range(num_eps):
        count = 0
        print(f"starting episode {episode+1}")
        state = env.reset()
        done = False
        while not done:
            count += 1
            if count % 100 == 0:
                print(count)
            action, state_for_buffer = dqn.act(state)
            next_state, reward, done = env.step(action)
            state = next_state
            if done:
                num_nodes.append(env.state.number_of_nodes())
                num_edges.append(env.state.number_of_edges())

        for budget in budgets:
            print(f"Budget {budget}: ratio of {env.ratios[budget][-1]:.4f}")
        print("\n")

    lines = []
    lines.append(f"Graph: {test_graph_name}. Budget: {soln_budget}. Fixed graph size: {subgraph_size}.")
    subgraph_lines = []
    subgraph_lines.append(f"Graph: {test_graph_name}. Budget: {soln_budget}. Fixed graph size: {subgraph_size}.")
    if num_eps > 1:
        N = graph.number_of_nodes()
        E = graph.number_of_edges()
        mean_n = np.mean(num_nodes)
        mean_e = np.mean(num_edges)
        stderror_n = np.std(num_nodes) / np.sqrt(num_eps)
        stderror_e = np.std(num_edges) / np.sqrt(num_eps)
        print(f"Graph: {test_graph_name}. Budget: {soln_budget}. Fixed graph size: {subgraph_size}. Number of nodes: {N}. Number of edges: {E}")
        print(f"Average number of nodes in final in subgraphs: {mean_n} ({stderror_n}). This is a reduction of {(1 - mean_n / N) * 100}%")
        print(f"Average number of edges in final in subgraphs: {mean_e} ({stderror_e}). This is a reduction of {(1 - mean_e / E) * 100}%\n")
        subgraph_lines.append(f"Average number of nodes in final in subgraphs: {mean_n} ({stderror_n}). This is a reduction of {(1 - mean_n / N) * 100}%")
        subgraph_lines.append(f"Average number of edges in final in subgraphs: {mean_e} ({stderror_e}). This is a reduction of {(1 - mean_e / E) * 100}%\n")
        with open(f"{test_graph_name}/budget_{soln_budget}/{encoder_name}/multi_budget_subgraph_results.txt", mode="w") as f:
            for line in subgraph_lines:
                f.write(line)
                f.write("\n")

    for budget in env.budgets:

        mean_r = np.mean(env.ratios[budget])
        mean_time = np.mean(env.times[budget])
        if num_eps > 1:
            stderror_r = np.std(env.ratios[budget]) / np.sqrt(num_eps)
            stderror_time = np.std(env.times[budget]) / np.sqrt(num_eps)
            lines.append(f"Budget {budget}: average ratio was {mean_r:.4f}, std error was {stderror_r}\n")
            lines.append(f"Budget {budget}: average time was {mean_time:.4f}, std error was {stderror_time}\n")
        else:
            lines.append(f"Budget {budget}: final ratio was {mean_r:.4f}\n")
            lines.append(f"Budget {budget}: final time was {mean_time:.4f}")

    with open(f"{test_graph_name}/budget_{soln_budget}/{encoder_name}/multi_budget_test_results.txt", mode="w") as f:
        for line in lines:
            f.write(line)
            f.write("\n")

    close_pool(env.pool)
    env.pool = None
