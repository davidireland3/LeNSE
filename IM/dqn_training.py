from environment import GraphModification
import torch
import networkx as nx
from functions import close_pool, get_best_embeddings, moving_average
from rl_algs import DQN
import matplotlib.pyplot as plt
import pickle
import random
import sys
import getopt
import numpy as np
import copy


def distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


if __name__ == "__main__":
    torch.random.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    graph_name = "twitter_train"
    encoder_name = "encoder"
    num_eps = 20
    chunksize = 28
    soln_budget = 100
    subgraph_size = 750
    selection_budget = 7500
    gnn_input = 50
    max_memory = 20000
    embedding_size = 10
    ff_size = 128
    decay_rate = 0.999975
    beta = 0.1
    cuda = True
    args = sys.argv[1:]
    opts, args = getopt.getopt(args, "g:n:c:e:b:s:d:f:m:C:h:B:E:D:")
    for opt, arg in opts:
        if opt in ['-g']:
            graph_name = arg
        elif opt in ["-n"]:
            num_eps = int(arg)
        elif opt in ["-c"]:
            chunksize = int(arg)
        elif opt in ["-e"]:
            embedding_size = int(arg)
        elif opt in ["-b"]:
            soln_budget = int(arg)
        elif opt in ["-s"]:
            selection_budget = int(arg)
        elif opt in ["-d"]:
            gnn_input = int(arg)
        elif opt in ["-f"]:
            subgraph_size = int(arg)
        elif opt in ["-m"]:
            max_memory = int(arg)
        elif opt in ["-C"]:
            cuda = bool(int(arg))
        elif opt in ["-h"]:
            ff_size = int(arg)
        elif opt in ["-B"]:
            beta = float(arg)
        elif opt in ["-E"]:
            encoder_name = arg
        elif opt in ['-D']:
            decay_rate = float(arg)

    encoder = torch.load(f"{graph_name}/budget_{soln_budget}/{encoder_name}/{encoder_name}", map_location=torch.device("cpu"))
    graph = nx.read_gpickle(f"{graph_name}/main")
    best_embeddings = get_best_embeddings(encoder, f"{graph_name}/budget_{soln_budget}/graph_data")
    encoder.to("cpu")
    dqn = DQN(gnn_input=gnn_input, batch_size=128, decay_rate=decay_rate, ff_hidden=ff_size, state_dim=embedding_size, gamma=0.95, max_memory=max_memory, cuda=cuda)
    env = GraphModification(graph, soln_budget, subgraph_size, encoder, best_embeddings, graph_name, chunksize=chunksize, action_limit=selection_budget, beta=beta, cuda=cuda)
    best_embedding = env.best_embedding_cpu.numpy()

    distances = []
    ratios = []
    rewards = []
    for episode in range(num_eps):
        print(f"starting episode {episode+1}")
        state = env.reset()
        done = False
        count = 0
        while not done:
            count += 1
            if count % 100 == 0:
                print(count)
            action, state_for_buffer = dqn.act(state)
            next_state, reward, done = env.step(action)
            dqn.remember(state_for_buffer, reward, next_state[0], done)
            if count % 20 == 0:
                dqn.experience_replay()
            state = next_state

        if dqn.epsilon > dqn.epsilon_min:
            print(f"Exploration rate currently at {dqn.epsilon:.3f}")
        final_dist = distance(env.subgraph_embedding, best_embedding)
        distances.append(-final_dist)
        print(f"final distance of {final_dist}")
        print(f"Ratio of {env.ratios[-1]:.3f}, sum of rewards {sum(env.episode_rewards)}\n")
        ratios.append(env.ratios[-1])
        plt.plot(env.episode_rewards)
        plt.savefig("ep_rewards.pdf")
        plt.clf()

        if (episode + 1) % 5 == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.plot(env.ratios)
            ax1.plot(moving_average(env.ratios, 50))
            ax1.hlines(0.95, 0, len(env.ratios) - 1, colors="red")
            ax2.plot(distances)
            plt.savefig(f"{graph_name}/budget_{soln_budget}/{encoder_name}/dqn_training.pdf")
            plt.clf()

            with open(f"{graph_name}/budget_{soln_budget}/{encoder_name}/trained_dqn", mode="wb") as f:
                dqn_copy = copy.deepcopy(dqn)
                dqn_copy.memory = ["hold"]
                dqn_copy.batch_size = 0
                pickle.dump(dqn_copy, f)
                del dqn_copy

    close_pool(env.pool)
    env.pool = None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(env.ratios)
    ax1.plot(moving_average(env.ratios, 50))
    ax1.hlines(0.95, 0, len(env.ratios) - 1, colors="red")
    ax2.plot(distances)
    plt.savefig(f"{graph_name}/budget_{soln_budget}/{encoder_name}/dqn_training.pdf")
    plt.clf()

    dqn.memory = ["hold"]
    dqn.batch_size = 0
    with open(f"{graph_name}/budget_{soln_budget}/{encoder_name}/trained_dqn", mode="wb") as f:
        pickle.dump(dqn, f)
