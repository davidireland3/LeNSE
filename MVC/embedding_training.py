from torch_geometric.loader import DataLoader
import torch
import numpy as np
import pickle
import random
from networks import KPooling, GNN
from pytorch_metric_learning import losses, miners
from pytorch_tools import EarlyStopping
import sys
import getopt

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)

graph = "wiki_train"
pooling = True
ratio = 0.8
embedding_size = 30
temperature = 0.1
output_size = 10
budget = 100
batch_size = 64
learning_rate = 1e-3
metric = "distance"
args = sys.argv[1:]
opts, args = getopt.getopt(args, "g:e:r:o:t:m:b:l:p:")
for opt, arg in opts:
    if opt in ['-g']:
        graph = arg
    elif opt in ["-e"]:
        embedding_size = int(arg)
    elif opt in ["-r"]:
        ratio = float(arg)
    elif opt in ["-o"]:
        output_size = int(arg)
    elif opt in ["-t"]:
        temperature = float(arg)
    elif opt in ["-m"]:
        metric = arg
    elif opt in ["-b"]:
        budget = int(arg)
    elif opt in ["-l"]:
        learning_rate = float(arg)
    elif opt in ["-p"]:
        pooling = bool(int(arg))

device = "cuda" if torch.cuda.is_available() else "cpu"
train_ratio = 0.8

with open(f"{graph}/budget_{budget}/graph_data", mode="rb") as f:
    data = pickle.load(f)
random.shuffle(data)
data = data[:2500]
data = [d.to(device) for d in data]
n = int(len(data) * train_ratio)


train_data = data[:n]
val_data = data[n:]
del data

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

if pooling:
    encoder = KPooling(ratio, 2, embedding_size, output_size).to(device)
else:
    encoder = GNN(2, embedding_size, output_size).to(device)
optimiser = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
loss_fn = losses.NTXentLoss(temperature)
miner = miners.MultiSimilarityMiner()
es = EarlyStopping(patience=10, percentage=False)

losses = []
val_losses = []
for epoch in range(1000):
    epoch_train_loss = []
    epoch_val_loss = []
    for count, batch in enumerate(train_loader):
        optimiser.zero_grad()
        inputs = encoder.forward(batch)
        hard_pairs = miner(inputs, batch.y)
        loss = loss_fn(inputs, batch.y, hard_pairs)
        epoch_train_loss.append(loss.item())
        loss.backward()
        optimiser.step()

    for batch in val_loader:
        with torch.no_grad():
            inputs = encoder.forward(batch)
            loss = loss_fn(inputs, batch.y)
            epoch_val_loss.append(loss.item())

    losses.append(np.mean(epoch_train_loss))
    val_losses.append(np.mean(epoch_val_loss))
    print(f"Epoch {epoch+1}:\n"
          f"Train loss -- {losses[-1]:.3f}\n"
          f"Val loss -- {val_losses[-1]:.3f}\n")

    if es.step(torch.FloatTensor([val_losses[-1]])) and epoch > 20:
        break

torch.save(encoder, f"{graph}/budget_{budget}/encoder/encoder")
