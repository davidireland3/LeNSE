from torch_geometric.data import DataListLoader
from torch_geometric.nn import DataParallel
import torch
import numpy as np
import pickle
import random
from networks import KPooling
from pytorch_metric_learning import losses, miners
from collections import OrderedDict
from pytorch_tools import EarlyStopping
import sys
import getopt

random.seed(1)
torch.manual_seed(1)
np.random.seed(1)

graph = "youtube_train"
ratio = 0.8
embedding_size = 40
temperature = 0.1
output_size = 10
budget = 100
batch_size = 128
learning_rate = 0.01
metric = "distance"
args = sys.argv[1:]
opts, args = getopt.getopt(args, "g:e:r:o:t:m:b:l:")
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_ratio = 0.8

with open(f"{graph}/budget_{budget}/graph_data", mode="rb") as f:
    data = pickle.load(f)
random.shuffle(data)
data = data[:1500]

n = int(len(data) * train_ratio)
train_data = data[:n]
val_data = data[n:]
del data

train_loader = DataListLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataListLoader(val_data, batch_size=batch_size, shuffle=True)

encoder = KPooling(ratio, 2, embedding_size, output_size)
print(f"Let's use {torch.cuda.device_count()} GPUs!")
encoder = DataParallel(encoder)
encoder = encoder.to(device)
optimiser = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
loss_fn = losses.NTXentLoss(temperature)
miner = miners.MultiSimilarityMiner()
es = EarlyStopping(patience=20, percentage=False)

losses = []
val_losses = []
for epoch in range(1000):
    epoch_train_loss = []
    epoch_val_loss = []
    for batch in train_loader:
        optimiser.zero_grad()
        inputs = encoder(batch)
        print(f'Outside model - num graphs: {inputs.size(0)}')
        y = torch.cat([data.y for data in batch]).to(inputs.device)
        hard_pairs = miner(inputs, y)
        loss = loss_fn(inputs, y, hard_pairs)
        epoch_train_loss.append(loss.item())
        loss.backward()
        optimiser.step()

    for batch in val_loader:
        with torch.no_grad():
            inputs = encoder(batch)
            print(f'Outside model - num graphs: {inputs.size(0)}')
            y = torch.cat([data.y for data in batch]).to(inputs.device)
            loss = loss_fn(inputs, y)
            epoch_val_loss.append(loss.item())

    losses.append(np.mean(epoch_train_loss))
    val_losses.append(np.mean(epoch_val_loss))
    print(f"Epoch {epoch+1}:\n"
          f"Train loss -- {losses[-1]:.3f}\n"
          f"Val loss -- {val_losses[-1]:.3f}\n")

    if es.step(torch.FloatTensor([val_losses[-1]])) and epoch > 20:
        break

state_dict = encoder.state_dict().copy()
encoder = KPooling(ratio, 2, embedding_size, output_size)

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]
    new_state_dict[name] = v
encoder.load_state_dict(new_state_dict)
torch.save(encoder, f"{graph}/budget_{budget}/encoder/encoder")
