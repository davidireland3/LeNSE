from invgraph import Graph
import random
import multiprocessing as mp
import time
import math
import networkx as nx

NUM_PROCESSORS = mp.cpu_count()


class Worker(mp.Process):
    def __init__(self, inQ, outQ, node_num, model, graph_):
        super(Worker, self).__init__(target=self.start)
        self.inQ = inQ
        self.outQ = outQ
        self.R = []
        self.count = 0
        self.node_num = node_num
        self.model = model
        self.graph = graph_

    def run(self):

        while True:
            theta = self.inQ.get()
            while self.count < theta:
                v = random.randint(1, self.node_num-1)
                rr = generate_rr(v, self.model, self.graph)
                self.R.append(rr)
                self.count += 1
            self.count = 0
            self.outQ.put(self.R)
            self.R = []


def create_worker(num, worker, node_num, model, graph_):
    for i in range(num):
        worker.append(Worker(mp.Queue(), mp.Queue(), node_num, model, graph_))
        worker[i].start()


def finish_worker(worker):
    for w in worker:
        w.terminate()


def sampling(epsoid, l, node_num, seed_size, worker, graph_, model):
    R = []
    LB = 1
    n = node_num
    k = seed_size
    epsoid_p = epsoid * math.sqrt(2)
    worker_num = NUM_PROCESSORS
    create_worker(worker_num, worker, node_num, model, graph_)
    for i in range(1, int(math.log2(n - 1)) + 1):
        s = time.time()
        x = n / (math.pow(2, i))
        lambda_p = ((2 + 2 * epsoid_p / 3) * (logcnk(n, k) + l * math.log(n) + math.log(math.log2(n))) * n) / pow(
            epsoid_p, 2)
        theta = lambda_p / x
        for ii in range(worker_num):
            worker[ii].inQ.put((theta - len(R)) / worker_num)
        for w in worker:
            R_list = w.outQ.get()
            R += R_list
        # finish_worker()
        # worker = []
        end = time.time()
        # print('time to find rr', end - s)
        start = time.time()
        Si, f, my_variable = node_selection(R, k, node_num)
        end = time.time()
        # print('node selection time', time.time() - start)
        # f = F(R,Si)
        if n * f >= (1 + epsoid_p) * x:
            LB = n * f / (1 + epsoid_p)
            break
    # finish_worker()
    alpha = math.sqrt(l * math.log(n) + math.log(2))
    beta = math.sqrt((1 - 1 / math.e) * (logcnk(n, k) + l * math.log(n) + math.log(2)))
    lambda_aster = 2 * n * pow(((1 - 1 / math.e) * alpha + beta), 2) * pow(epsoid, -2)
    theta = lambda_aster / LB
    length_r = len(R)
    diff = theta - length_r
    _start = time.time()
    if diff > 0:
        for ii in range(worker_num):
            worker[ii].inQ.put(diff / worker_num)
        for w in worker:
            R_list = w.outQ.get()
            R += R_list
    '''

    while length_r <= theta:
        v = random.randint(1, n)
        rr = generate_rr(v)
        R.append(rr)
        length_r += 1
    '''
    _end = time.time()
    finish_worker(worker)
    return R


def generate_rr(v, model, graph):
    if model == 'IC':
        return generate_rr_ic(v, graph)
    elif model == 'LT':
        return generate_rr_lt(v, graph)


def node_selection(R, k, node_num):
    Sk = set()
    list1 = []
    rr_degree = [0 for ii in range(node_num)]
    node_rr_set = dict()
    # node_rr_set_copy = dict()
    matched_count = 0
    for j in range(0, len(R)):
        rr = R[j]
        for rr_node in rr:
            rr_degree[rr_node] += 1
            if rr_node not in node_rr_set:
                node_rr_set[rr_node] = list()
                # node_rr_set_copy[rr_node] = list()
            node_rr_set[rr_node].append(j)
            # node_rr_set_copy[rr_node].append(j)
    for i in range(k):
        max_point = rr_degree.index(max(rr_degree))
        Sk.add(max_point)
        list1.append(max_point)
        matched_count += len(node_rr_set[max_point])
        index_set = []
        for node_rr in node_rr_set[max_point]:
            index_set.append(node_rr)
        for jj in index_set:
            rr = R[jj]
            for rr_node in rr:
                rr_degree[rr_node] -= 1
                node_rr_set[rr_node].remove(jj)
    return Sk, matched_count / len(R), list1


def generate_rr_ic(node, graph):
    activity_set = list()
    activity_set.append(node)
    activity_nodes = list()
    activity_nodes.append(node)
    while activity_set:
        new_activity_set = list()
        for seed in activity_set:
            for node, weight in graph.get_neighbors(seed):
                if node not in activity_nodes:
                    if random.random() < weight:
                        activity_nodes.append(node)
                        new_activity_set.append(node)
        activity_set = new_activity_set
    return activity_nodes


def generate_rr_lt(node, graph):
    # calculate reverse reachable set using LT model
    # activity_set = list()
    activity_nodes = list()
    # activity_set.append(node)
    activity_nodes.append(node)
    activity_set = node

    while activity_set != -1:
        new_activity_set = -1

        neighbors = graph.get_neighbors(activity_set)
        if len(neighbors) == 0:
            break
        candidate = random.sample(neighbors, 1)[0][0]
        if candidate not in activity_nodes:
            activity_nodes.append(candidate)
            # new_activity_set.append(candidate)
            new_activity_set = candidate
        activity_set = new_activity_set
    return activity_nodes


def imm(graph, seed_size, model="IC", epsoid=0.5, l=1):
    """
    graph must be a file path to a .txt file of edge lists where the first line has the number of nodes in the first
    column, or it must be a networkx graph object with edge weights under the key 'weight'.
    """
    graph_, node_num = get_graph(graph)
    worker = []
    n = node_num
    k = seed_size
    l = l * (1 + math.log(2) / math.log(n))
    R = sampling(epsoid, l, node_num, seed_size, worker, graph_, model)
    Sk, z, x = node_selection(R, k, node_num)
    return Sk, x


def logcnk(n, k):
    res = 0
    for i in range(n - k + 1, n + 1):
        res += math.log(i)
    for i in range(1, k + 1):
        res -= math.log(i)
    return res


def get_graph(network):
    """
    Takes in either filepath to graph object or a networkx object
    """
    if type(network) == str:
        graph_ = Graph()
        data_lines = open(network, 'r').readlines()
        node_num = int(float(data_lines[0].split()[0]))
        # edge_num = int(float(data_lines[0].split()[1]))

        for data_line in data_lines[1:]:
            start, end, weight = data_line.split()
            graph_.add_edge(int(float(start)), int(float(end)), float(weight))
            # pGraph.add_edge(int(float(start)), int(float(end)), float(weight))
        return graph_, node_num

    elif type(network) == nx.Graph or type(network) == nx.DiGraph:
        graph_ = Graph()
        node_num = network.number_of_nodes()

        for u, v in network.edges():
            weight = network[u][v]['weight']
            if type(weight) == dict:
                weight = weight['weight']
            graph_.add_edge(int(u), int(v), weight)
            if type(network) == nx.Graph:
                graph_.add_edge(int(v), int(u), network[v][u]['weight'])

        return graph_, node_num
