from torch_geometric.loader import DataLoader
import torch
import pickle
import sys
import getopt

graph_name = "wiki_train"
encoder_graph_name = "wiki_train"
budget = 100
args = sys.argv[1:]
opts, args = getopt.getopt(args, "g:e:b:")
for opt, arg in opts:
    if opt in ['-g']:
        graph_name = arg
    elif opt in ['-e']:
        encoder_graph_name = arg
    elif opt in ['-b']:
        budget = int(arg)

load_name = "graph_data"
save_name = "train_data"

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = torch.load(f"{encoder_graph_name}/budget_{budget}/encoder/encoder")
encoder = encoder.to(device)

with open(f"{graph_name}/budget_{budget}/{load_name}", mode="rb") as f:
    data = pickle.load(f)

data = [d.to(device) for d in data]
for d in data:
    d.y = d.y - 1
loader = DataLoader(data, batch_size=128)


with torch.no_grad():
    features = []
    labels = []
    for batch in loader:
        embeddings = encoder(batch)
        features += embeddings.tolist()
        labels += batch.y.tolist()
        # labels += batch.spread_label
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    with open(f"{graph_name}/budget_{budget}/encoder/{save_name}", mode="wb") as f:
        pickle.dump((features.to("cpu"), labels.to("cpu")), f)
