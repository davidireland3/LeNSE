import numpy as np
import networkx as nx
import glob
import random
import torch
import pickle
from torch_geometric.data import DataLoader, Data
from scipy.special import softmax
import time
from IMM import imm
from networkx.algorithms.centrality import eigenvector_centrality


class ProblemError(Exception):

    def __init__(self):
        self.message = "Input an invalid problem"
        super().__init__(self.message)


def spread(fname, seed_nodes):
    node_to_add = 999999999999999
    if len(seed_nodes) == 0:
        return 0
    subgraph = nx.read_gpickle(fname)
    for node in seed_nodes:
        subgraph.add_edge(node_to_add, node, weight=1)
    activated_nodes = nx.descendants(subgraph, node_to_add)
    return len(activated_nodes)


def expected_spread(seed_nodes, graph="facebook", Type="main", model="IC", chunksize=50,
                    spread_dic=None, pool=None):
    if pool is None:
        print("haven't given a pool for MP")

    if model == "IC":
        fnames = glob.glob(f"{graph}/IC_{Type}_subgraphs/subgraph_*")
    else:
        print("entered an incorrect model!")
        return

    seed_nodes = [seed_nodes for _ in range(len(fnames))]

    if spread_dic is not None and tuple(sorted(seed_nodes[0])) in spread_dic:
        return spread_dic[tuple(sorted(seed_nodes[0]))], spread_dic

    spreads = pool.starmap(spread, zip(fnames, seed_nodes), chunksize)
    if spread_dic:
        spread_dic[tuple(sorted(seed_nodes[0]))] = np.mean(spreads)
        return np.mean(spreads), spread_dic
    else:
        return np.mean(spreads)


def close_pool(pool):
    pool.close()
    pool.join()


def get_edge_list(graph):
    edges = graph.edges()
    source = []
    root = []
    [(source.append(u), root.append(v)) for u, v in edges]
    return [source, root]


def make_graph_features_for_encoder(graph, graph_name):
    if type(graph) == nx.Graph:
        print("not a di graph")
        return

    elif type(graph) == nx.DiGraph:
        try:
            with open(f"{graph_name}/graph_features", mode="rb") as f:
                features = pickle.load(f)
        except FileNotFoundError:
            features = {}
            out_degrees = [len(graph.out_edges(node)) for node in range(graph.number_of_nodes())]
            out_degree_max = np.max(out_degrees)
            out_degree_min = np.min(out_degrees)
            out_e_weights = [sum([graph[node][neighbour]["weight"] for _, neighbour in graph.out_edges(node)]) for node in
                             range(graph.number_of_nodes())]
            out_max_e_weight = max(out_e_weights)
            out_min_e_weight = min(out_e_weights)

            ev_values = eigenvector_centrality(graph, max_iter=1000)

            for node in range(graph.number_of_nodes()):
                features[node] = [(out_degrees[node] - out_degree_min) / (out_degree_max - out_degree_min), (
                        out_e_weights[node] - out_min_e_weight) / (out_max_e_weight - out_min_e_weight), ev_values[node]]
            with open(f"{graph_name}/graph_features", mode="wb") as f:
                pickle.dump(features, f)
        return features


def make_subgraph(graph, nodes):
    assert type(graph) == nx.DiGraph
    subgraph = nx.DiGraph()
    edges_to_add = []
    for node in nodes:
        edges_to_add += [(u, v, w) for u, v, w in list(graph.out_edges(node, data=True)) + list(graph.in_edges(node, data=True))]
    subgraph.add_weighted_edges_from(edges_to_add)
    return subgraph


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
    else:
        label = 4
    return label


def get_fixed_size_subgraphs(graph, good_seeds, num_samples, counts, BUDGET, size, pool, graph_name, best_score,
                             graph_features, target_label=None):
    subgraphs = []
    probs = np.array([counts[seed] for seed in good_seeds])
    probs = softmax(probs)
    while len(subgraphs) < num_samples:
        start = time.time()
        all_good_seeds = good_seeds

        """
        This is where values may need to be adjusted to ensure we get a subgraph in the right range for the Fixed Size Dataset file.
        """
        if target_label == 1:

            seeds = np.random.choice(list(good_seeds), size=BUDGET, replace=False, p=probs).tolist()
            seeds += random.sample([n for n in graph.nodes() if n not in seeds], size - len(seeds))

        elif target_label == 2:

            r = np.random.uniform(0.7, 0.8)
            seeds = np.random.choice(list(good_seeds), size=int(BUDGET * r), replace=False, p=probs).tolist()
            seeds += random.sample([n for n in graph.nodes() if n not in (list(good_seeds) + seeds)], size - len(seeds))

        elif target_label == 3:
            seeds = np.random.choice(list(good_seeds), size=int(BUDGET * 0.25), replace=False, p=probs).tolist()
            seeds += random.sample([n for n in graph.nodes() if n not in (list(good_seeds) + seeds)], size - len(seeds))

        elif target_label == 4:
            seeds = np.random.choice(list(good_seeds), size=int(BUDGET * 0.0), replace=False, p=probs).tolist()
            seeds += random.sample([n for n in graph.nodes() if n not in (list(good_seeds) + seeds)], size - len(seeds))

        else:
            seeds = random.sample(graph.nodes(), size)

        subgraph = make_subgraph(graph, seeds)
        subgraph_, transformation = relabel_graph(subgraph, True)
        seeds, _ = imm(subgraph_, BUDGET)
        seeds = [transformation[seed] if seed in transformation else seed for seed in seeds]
        score_ = expected_spread(seeds, graph=graph_name, pool=pool)
        label = get_label(score_ / best_score)
        print(label)
        if label != target_label:
            continue

        g = Data(edge_index=torch.LongTensor(get_edge_list(subgraph_)), num_nodes=subgraph_.number_of_nodes())
        g.y = torch.LongTensor([label])
        features = [graph_features[transformation[node]] if node in transformation else graph_features[node] for node in range(subgraph_.number_of_nodes())]
        g.x = torch.FloatTensor(features)
        g.num_seeds = np.sum([seed in all_good_seeds for seed in seeds])
        g.graph_name = graph_name
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
