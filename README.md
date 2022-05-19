# LeNSE

The repo for LeNSE is broken into subdirectories for each of the different problems. They have almost identical structure, with the main difference being which heuristic to use. There is also slightly more to do for IM, as we need to pre-calculate (sub)graphs based on coin flips to evaluate the score of a node -- details on this can be found in [Kempe et al.]([url](http://www.theoryofcomputing.org/articles/v011a004/v011a004.pdf))

## Obtaining the data
First, we need to identify which graph we want to use. In the paper we used graphs from the [SnapStanford database]([url](http://snap.stanford.edu/)). It is expected that the data is stored in a similar way to how it is presented in the SnapStanford database, i.e. a .txt file with two columns indicating edges, WLOG we assume the first column is the root node and the second column is the receiver node. Once we have the .txt file, we can use the file `get_train_test_graphs.py` to make this into a `networkx` object. 
