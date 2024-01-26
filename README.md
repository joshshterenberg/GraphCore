<div align="center">
  
# :octopus: OCToPi: Object Condensation Tracking on Pixels :octopus:
![](readme_assets/banner.jpg) 

A graph-neural-network-based approach to track reconstruction in the near-axis cores of jets, for the CMSSW.
#### Joshua Shterenberg & Nick Haubrich
</div>

## Goal 
Improve resolution of charged particle trajectory reconstruction, decrease computation time. For this project: Evaluate GNN model in CMSSW, do a head-to-head comparison with DeepCore.

## Introduction
LHC tracks mix, which produce merged clusters of charge in tracking layers. Difficult to optimize splitting these clusters. As angle between track and jet decreases, efficiency decreases. To improve reconstruction, DeepCore: a CNN that is basically doing it all on its own, helps gain efficiency with high momentum. DeepCore has drawbacks: not clear if learning correctly, doesn’t work well for endcaps, inefficient predictions, constrains data representations. We want a GNN instead to help mitigate these drawbacks.

## I/O Details & Assumptions
* I will start by using some sample input data, which will be an idealized set of data I’ll use from CMS in testing. Input data will consist of 4 2D grids (each with unique dims) of pixels, jet information, and truth information. Each pixel will have it’s own 2D+1 coordinate and charge value. Jet info will be the recorded momentum and rapidity from ECAL. Truth information will be an exact XY hit in each layer per track (4N total). 
* Output will be the predicted number of tracks that cross a given pixel for each pixel in 2D+1 space, and the kinematic information of the calculated tracks (to be used for seed).
* All particles are charged, ultra-relativistic, non-radiative, and always leave a trace. I.e., tracks are helixes, coaxial to beam line, with momentum comparable to energy detected in ECAL. All lit pixels correspond to tracks.
* Tracks do not have to cross all 4 layers.
* Tracks are randomly distributed around the known jet axis. Gives some approximation for how many tracks we’re dealing with.

***

## Version 1.0 Workflow

0. `. env.sh`: set up venv. Ensure that data is preprocessed correctly by ntuplizer into .root file (example and folder are included).
1. `python mlp.py`: trains a neural network to cluster with a metric learning approach. Generates trained model as file.
2. `python gnn.py`: pushes data though trained MLP, creates a graph in feature space with KNN with max edge length, and trains a GNN. Generates trained model as file.
3. `python test.py`: pushes through trained MLP, creates KNN graph, pushes through trained GNN, and clusters with DBSCAN. Outputs kinematics of recovered tracks as file. 
4. `python post.py`: converts output of test.py to input of next step in CMSSW.




