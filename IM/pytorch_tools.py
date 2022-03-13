import torch
import torch.nn as nn
# import dgl
# import dgl.function as fn
from torch_cluster import random_walk
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                        best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                        best * min_delta / 100)


def unsupervised_loss_geom(embeddings, graph_obj, num_nodes, rw_length, num_neg_examples):
    # first we will get the positive embeddings
    # embedding_size = embeddings.shape[1]
    f = nn.LogSigmoid()
    cos = nn.CosineSimilarity()
    start = torch.LongTensor([i for i in range(num_nodes)]).to(device)
    rw_ids = random_walk(graph_obj.edge_index[0], graph_obj.edge_index[1], start, rw_length)[:, -1]
    # loss = -f(torch.sum(embeddings * embeddings[rw_ids], dim=1))
    pos = cos(embeddings, embeddings[rw_ids])

    # now get the negatives
    neg = 0
    for _ in range(num_neg_examples):
        neg_ids = torch.LongTensor(np.random.randint(0, num_nodes, size=num_nodes)).to(device)
        # loss -= f(torch.sum(-embeddings * embeddings[neg_ids], dim=1))
        neg += cos(embeddings, embeddings[neg_ids])

    loss = -pos + neg

    return loss.mean()
