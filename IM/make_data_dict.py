import pickle
import sys
import getopt

graph_name = "youtube_train"
budget = 100

args = sys.argv[1:]
opts, args = getopt.getopt(args, "g:b:")
for opt, arg in opts:
    if opt in ['-g']:
        graph_name = arg
    elif opt in ["-b"]:
        budget = int(arg)

with open(f"{graph_name}/budget_{budget}/graph_data", mode="rb") as f:
    data = pickle.load(f)

data = [d.to_dict() for d in data]

with open("hold", mode="wb") as f:
    pickle.dump(data, f)
