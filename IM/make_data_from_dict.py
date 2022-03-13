import pickle
from torch_geometric.data import Data
import sys
import getopt

graph_name = "twitter_undir"
budget = 100

args = sys.argv[1:]
opts, args = getopt.getopt(args, "g:b:")
for opt, arg in opts:
    if opt in ['-g']:
        graph_name = arg
    elif opt in ["-b"]:
        budget = int(arg)

with open("hold", mode="rb") as f:
    data = pickle.load(f)

data = [Data.from_dict(d) for d in data]

with open(f"{graph_name}/budget_{budget}/graph_data", mode="wb") as f:
    pickle.dump(data, f)
