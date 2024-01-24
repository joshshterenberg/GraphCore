# GraphCore (Working Title)

### Josh's Version

Graph-based approach to track reconstruction in cores of jets.

## Workflow

0. Setup python3 virtual environment with `. env.sh`. Ensure that data is preprocessed correctly by ntuplizer into .root file (example and folder are included).

1. `python mlp.py`: trains a neural network to cluster with a metric learning approach. Generates trained model as file.

2. `python gnn.py`: pushes data though trained MLP, creates a graph in feature space with KNN with max edge length, and trains a GNN. Generates trained model as file.

3. `python test.py`: pushes through trained MLP, creates KNN graph, pushes through trained GNN, and clusters with DBSCAN. Outputs kinematics of recovered tracks as file. 

4. `python post.py`: converts output of test.py to input of next step in CMSSW.



Thanks to Nick for initializing steps 0 and 1!




