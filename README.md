# LeNSE

This is the repo for the paper "LeNSE: Learning To Navigate Subgraph Embeddings for Large-Scale Combinatorial Optimisation", to appear at ICML 2022. The paper can be found [here](https://arxiv.org/abs/2205.10106). In this work we look to scale up existing CO solvers by pruning the original problem. This is done by learning to navigate subgraph embeddings. Using a small training graph, we generate a dataset of subgraphs with different rankings (including optimal subgraphs). Then, using this dataset, we train a Graph Neural Network to learn a discriminative subgraph embedding via the InfoNCE loss. This embedding is then used to guide the RL process to modify any random subgraph into an optimal subgraph. A heuristic is then deployed on the optimal subgraph to find a solution to the original problem.

The repo for LeNSE is broken into subdirectories for each of the different problems. They have almost identical structure, with the main difference being which heuristic to use. There is also slightly more to do for IM, as we need to pre-calculate (sub)graphs based on coin flips to evaluate the score of a node -- details on this can be found in [Kempe et al.]([url](http://www.theoryofcomputing.org/articles/v011a004/v011a004.pdf))

## Obtaining the data
First, we need to identify which graph we want to use. In the paper we used graphs from the [SnapStanford database]([url](http://snap.stanford.edu/)). It is expected that the data is stored in a similar way to how it is presented in the SnapStanford database, i.e. a .txt file with two columns indicating edges, WLOG we assume the first column is the root node and the second column is the receiver node. Once we have the .txt file, we first need to process it into a `networkx` graph. To do this we use the `make_graph_from_edge_list.py` which will read in the edge list from the txt file, create a networkx object and pickle save it. We can then use the file `get_train_test_graphs.py` to separate this into two `networkx` objects -- one will be the train graph, and the other will be the held out test graph which contains all edges from the original graph not in the train graph. The new graphs will added to directories named `'graph_name'_{test/train}`. For instance, if we use the Facebook graph as an example then the train graph would be stored in a directory called `facebook_train`. For IM, we also need to run `get_graph_sample_for_IC.py` for the main, train and test graphs. We need to do this so that we can efficiently calculate the spread of nodes in these graphs. 

## Running the heuristics and obtaining the dataset
To obtain the seed nodes for the train/test graphs we need to run the file `get_scores_seeds.py` with the graph name given as an argument along with the budget. This file will run the heuristics and obtain the _budget_ seed nodes, along with the score, and set up a subdirectory in the graph_name directory for the given budget. It stores these in this subdirectory -- this is so that we can separate solutions for different budgets etc.

Once we have the scores and seeds, we now create the dataset used to pre-train the embedding that guides LeNSE. This is done in the `fixed_size_dataset.py` file. It takes arguments that specify the budget, number of samples per class, and and fixed size, which specifies the number of root nodes we select ($X$ in the paper) to induce a subgraph from. The subgraphs are made using the function `get_fixed_size_subgraphs` in the `functions.py` file. As mentioned in the paper, we ensure a balanced dataset. We achieve this by ensuring that some fraction of the previously found solution nodes are present in $X$ when generating the subgraphs. The required ratio for each class may vary slightly for each dataset, though this tuning should take minutes; the best way is to first generate a small dataset for your graph with say ~5 subgraphs per class and ensure the ratios are correct. Once we have run `fixed_size_data.py` it will save a dataset of PytorchGeometric data objects ready to be passed into the encoder. 

## Training and visualising the embedding
Once we have our dataset, we can use `embedding_training.py` to train our GNN encoder. We can specify hyper-parameters in the file. Once it has trained, you can use the `prepare_data.py` file to save a Pytorch tensor object which contains all the embeddings of our subgraphs from the encoder. Then, to visualise this, we can use the `autoencoder.py` file to (quickly) train an autoencoder and see the resulting plots in the latent space. 

## RL training
Once the encoder has been properly trained, we can then use the `guided_exploration.py` file to do the RL training. Note that the arugment `encoder_name` should be set to 'encoder'. This was an argument used for the ablations in the appendix when comparing to an Ordinal and Cross-Entropy loss. 
