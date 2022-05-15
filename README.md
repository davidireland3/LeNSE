# LeNSE

The repo is broken into a directory for each problem. Each repo follows an almost identical structure, the main difference between files is which heuristics to use in the environment, and the fact that the GNN in the IM problem has 3 input features as opposed to 2. 

To run LeNSE, follow the basic structure:
- obtain edge list of graph (e.g. from snap dataset) and save into a file of the graphs name as a .txt file. Use make_graph_from_edge_list.py to make this into a networkx object. Then use get_train_test_graphs.py to get a train/test split, where you can specify the size of the split you want to use. Note that there is a flag `EDGES` which denotes how to obtain the split. If this is set to True then the pecified proportion of edges will be taken for the train graph.

- use get_scores_seeds.py to obtain the solution nodes and score for the graph (have to specify which graph you want to use, e.g. graph name and train/test).

- use fixed_size_dataset.py to obtain the dataset used to train the encoder. the default is set to generate 4 classes, but code could easily be modified to change this. to obtain the classes, first I would recommend trying to obtain a small number per class first to make sure the right number of seeds are being sampled. the reason for this is that in the function.py file, around line 190 there is a function called 'get_fixed_size_subgraphs' which is the function that generates the subgraphs. to get a subgraph of class 1 for instance, we typically ensure all the nodes are in the subgraph. however, if we want a subgraph for class 2 (i.e. one with a score in the range 0.8 - 0.95) then a portion of these need to be sampled. This can be seen from the lines in the block of elif statements where we have seeds=np.random.choice(list(good_seeds), size=int(BUDGET * r), replace=False).tolist() -- here we are sampling budget * r (r is the proportion) of the good nodes to ensure they're in our graph. this is the easiest way to ensure we get a subgraph with score in the desired range; there is not set rule how this should work as it can vary from graph to graph but it should take no more than ~2 minutes to find the right values. 

- use embedding_training.py to train the encoder, followed by prepare_data.py and autoencoder.py to save the embeddings and then train an autoencoder and visualise the plots in the lower dimension.

- guided_exploration.py runs the RL training once the encoder has been trained

- multi_budget_test.py or dqn_test.py will test LeNSE on the test graph (if only interested in the budget trained on then use dqn_test.py as it will be quicker, only having to run the heuristic on the subgraph once)


NOTE: after train/test graphs have been obtained for IM, you will additionally need to run the file get_graph_sample_for_IC.py as these will be needed to compute the spread of the node. see the original Kempe et al. 2003 paper (Maximizing the spread of influence through a social network) for details on how these are used to calculate the spread of a node. 

For any questions, please contact david.ireland@warwick.ac.uk.
