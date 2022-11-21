import pickle
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from networks import CustomDataset
from torch.utils.data import DataLoader
from networks import Autoencoder
import random
import numpy as np
import sys
import getopt

graph_name = "wiki_train"
input_size = 10
num_classes = 3
num_layers = 1
budget = 100
args = sys.argv[1:]
opts, args = getopt.getopt(args, "g:i:b:n:l:")
for opt, arg in opts:
    if opt in ['-g']:
        graph_name = arg
    elif opt in ['-i']:
        input_size = int(arg)
    elif opt in ['-b']:
        budget = int(arg)
    elif opt in ['-n']:
        num_classes = int(arg)
    elif opt in ['-l']:
        num_layers = int(arg)

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)

with open(f"{graph_name}/budget_{budget}/encoder/train_data", mode="rb") as f:
    data, labels = pickle.load(f)

dataset = CustomDataset(data, labels)
loader = DataLoader(dataset, batch_size=128, shuffle=True)
loss_fn = nn.MSELoss()

net = Autoencoder(input_size, num_layers)
optimiser = torch.optim.Adam(net.parameters())

for epoch in range(200):
    for x, y in loader:
        optimiser.zero_grad()
        loss = loss_fn(net(x), x)
        loss.backward()
        optimiser.step()

with torch.no_grad():
    output = net(data, True)
    output = output.numpy()
    labels = labels.numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

colours = ["red", "blue", "black", "yellow", "green"]
for class_, colour in zip(range(num_classes), colours):
    ax1.scatter(output[labels == class_, 0], output[labels == class_, 1], color=colour, label=f"class {class_ + 1}")
ax1.legend()
ax1.title.set_text("embedding space")

good_labels = [i for i in range(data.shape[0]) if int(labels[i]) == 0]
good_point = torch.mean(data[good_labels], dim=0)
similarities = torch.norm((good_point.reshape((1, input_size)) - data), p=2, dim=1).numpy()

im1 = ax2.scatter(output[:, 0], output[:, 1], c=similarities)
fig.colorbar(im1, ax=ax2)
ax2.title.set_text("L2 distance")
plt.savefig(f"{graph_name}/budget_{budget}/encoder/embeddings.pdf")
plt.show()
