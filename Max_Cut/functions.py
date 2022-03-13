import numpy as np
import networkx as nx
import random
import torch
import pickle
from torch_geometric.data import DataLoader, Data
from torch_geometric.transforms import ToUndirected
import time
from networkx.algorithms.centrality import eigenvector_centrality
import gurobipy as gp
from gurobipy import GRB


class ProblemError(Exception):

    def __init__(self):
        self.message = "Input an invalid problem"
        super().__init__(self.message)


def cut_value(graph, seeds):
    value = 0
    for node in seeds:
        for neighbour in graph.neighbors(node):
            if neighbour not in seeds:
                value += 1
    return value


def max_cut_heuristic(graph, budget):
    N = graph.number_of_nodes()
    model = gp.Model("mip")
    model.Params.LogToConsole = 0
    model.setParam(GRB.Param.MIPGap, 0.05)
    model.setParam('TimeLimit', 48 * 3600)
    nodes = [model.addVar(name=f"{i}", vtype=GRB.BINARY) for i in range(N)]  # add a binary decision variable for every node
    edges = {}
    for i, j in graph.edges():
        edges[(i, j)] = model.addVar(name=f"{(i, j)}", vtype=GRB.BINARY)  # add a binary decision variable for every edge

    model.addConstr(sum(nodes) == budget)
    for i, j in graph.edges():
        model.addConstr(nodes[i] + nodes[j] - 1 <= edges[(i, j)])
    model.setObjective(gp.quicksum([nodes[i] + nodes[j] - 2 * edges[(i, j)] for i, j in graph.edges()]), GRB.MAXIMIZE)
    model.optimize()
    sol = [i for i in range(N) if nodes[i].x == 1]
    return sol


def get_edge_list(graph):
    edges = graph.edges()
    source = []
    root = []
    [(source.append(u), root.append(v)) for u, v in edges]
    return [source, root]


def make_graph_features_for_encoder(graph, graph_name):
    try:
        with open(f"{graph_name}/graph_features", mode="rb") as f:
            features = pickle.load(f)
    except FileNotFoundError:
        features = {}
        out_degrees = [len(list(graph.neighbors(node))) for node in range(graph.number_of_nodes())]
        out_degree_max = np.max(out_degrees)
        out_degree_min = np.min(out_degrees)

        ev_values = eigenvector_centrality(graph, max_iter=1000)

        for node in range(graph.number_of_nodes()):
            features[node] = [(out_degrees[node] - out_degree_min) / (out_degree_max - out_degree_min), ev_values[node]]
        with open(f"{graph_name}/graph_features", mode="wb") as f:
            pickle.dump(features, f)
    return features


def make_subgraph(graph, nodes):
    assert type(graph) == nx.Graph
    subgraph = nx.Graph()
    edges_to_add = []
    for node in nodes:
        edges_to_add += [(u, v) for u, v in list(graph.edges(node))]
    subgraph.add_edges_from(edges_to_add)
    return subgraph


def get_test_accuracy(predictions, targets):
    predictions = torch.argmax(predictions, dim=1)
    score = torch.sum(predictions == targets)
    return float(score / predictions.shape[0] * 100)


def get_best_embeddings(encoder, filepath):
    with open(filepath, mode='rb') as f:
        graph_data = pickle.load(f)
    device = "cpu" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device)

    graphs_ = [g.to(device) for g in graph_data if g.spread_label == 1]
    loader = DataLoader(graphs_, batch_size=len(graphs_))
    with torch.no_grad():
        embd = encoder.forward(next(iter(loader)))
    best_embedding = embd.cpu()
    return best_embedding


def moving_average(x, w):
    means = [np.mean(x[i:max(0, i - w):-1]) if i != 0 else x[0] for i in range(len(x))]
    return means


def get_good_subgraphs(features, labels):
    good_embeddings = [feature.reshape((1, features.shape[1])) for feature, label in zip(features, labels) if int(label) == 1]
    return good_embeddings


def get_label(score):
    if score >= 0.95:
        label = 1
    elif score >= 0.8:
        label = 2
    elif score >= 0.6:
        label = 3
    elif score >= 0.3:
        label = 4
    else:
        label = 5
    return label


def get_fixed_size_subgraphs(graph, good_seeds, num_samples, BUDGET, size, best_score, graph_features, target_label=None):
    make_undirected = ToUndirected()
    subgraphs = []
    while len(subgraphs) < num_samples:
        start = time.time()
        all_good_seeds = good_seeds

        if target_label == 1:

            seeds = np.random.choice(list(good_seeds), size=BUDGET, replace=False).tolist()
            seeds += random.sample([n for n in graph.nodes() if n not in seeds], size - len(seeds))

        elif target_label == 2:

            r = np.random.uniform(0.8, 0.9)
            seeds = np.random.choice(list(good_seeds), size=int(BUDGET * r), replace=False).tolist()
            seeds += random.sample([n for n in graph.nodes() if n not in (list(good_seeds) + seeds)], size - len(seeds))

        elif target_label == 3:
            seeds = np.random.choice(list(good_seeds), size=int(BUDGET * 0.6), replace=False).tolist()
            seeds += random.sample([n for n in graph.nodes() if n not in (list(good_seeds) + seeds)], size - len(seeds))

        elif target_label == 4:
            seeds = np.random.choice(list(good_seeds), size=int(BUDGET * 0.4), replace=False).tolist()
            seeds += random.sample([n for n in graph.nodes() if n not in (list(good_seeds) + seeds)], size - len(seeds))

        else:
            seeds = random.sample(graph.nodes(), size)

        subgraph = make_subgraph(graph, seeds)
        subgraph_, transformation = relabel_graph(subgraph, True)
        seeds = max_cut_heuristic(subgraph_, BUDGET)
        seeds = [transformation[seed] if seed in transformation else seed for seed in seeds]
        score_ = cut_value(graph, seeds)
        label = get_label(score_ / best_score)
        print(label)
        if label != target_label:
            continue

        g = Data(edge_index=torch.LongTensor(get_edge_list(subgraph_)), num_nodes=subgraph_.number_of_nodes())
        make_undirected(g)
        g.y = torch.LongTensor([label])
        features = [graph_features[transformation[node]] if node in transformation else graph_features[node] for node in range(subgraph_.number_of_nodes())]
        g.x = torch.FloatTensor(features)
        g.num_seeds = np.sum([seed in all_good_seeds for seed in seeds])
        g.spread_label = int(score_ / best_score >= 0.95)
        g.score = score_
        subgraphs.append(g)
        end = time.time()
        print(f"It took {((end - start) / 60):.3f} minutes\n")
    return subgraphs


def relabel_edgelist(roots, dests, unique):
    N = len(unique)
    desired_labels = set([i for i in range(N)])
    already_labeled = set([int(node) for node in unique if node < N])
    desired_labels = desired_labels - already_labeled
    transformation = {}
    reverse_transformation = {}
    for node in unique:
        if node >= N:
            transformation[node] = desired_labels.pop()
            reverse_transformation[transformation[node]] = node

    new_roots = [transformation[r] if r in transformation else r for r in roots]
    new_dests = [transformation[r] if r in transformation else r for r in dests]
    edge_list = [new_roots, new_dests]
    return edge_list, transformation, reverse_transformation, N


def relabel_graph(graph: nx.Graph, return_reverse_transformation_dic=False, return_forward_transformation_dic=False):
    """
    forward transformation has keys being original nodes and values being new nodes
    reverse transformation has keys being new nodes and values being old nodes
    """
    nodes = graph.nodes()
    n = graph.number_of_nodes()
    desired_labels = set([i for i in range(n)])
    already_labeled = set([int(node) for node in nodes if node < n])
    desired_labels = desired_labels - already_labeled
    transformation = {}
    reverse_transformation = {}
    for node in nodes:
        if node >= graph.number_of_nodes():
            transformation[node] = desired_labels.pop()
            reverse_transformation[transformation[node]] = node

    if return_reverse_transformation_dic and return_forward_transformation_dic:
        return nx.relabel_nodes(graph, transformation), transformation, reverse_transformation

    elif return_forward_transformation_dic:
        return nx.relabel_nodes(graph, transformation), transformation

    elif return_reverse_transformation_dic:
        return nx.relabel_nodes(graph, transformation), reverse_transformation

    else:
        return nx.relabel_nodes(graph, transformation)
