<div align="center">
  
## :octopus: OCToPi: Object Condensation Tracking on Pixels :octopus:
![](readme_assets/banner.jpg) 

A graph-neural-network-based approach to track reconstruction in the near-axis cores of jets, for the CMSSW.
#### Joshua Shterenberg
</div>

## Goal 
Improve resolution of charged particle trajectory reconstruction, decrease computation time.

## Introduction
LHC tracks mix, which produce merged clusters of charge in tracking layers. IT's difficult to optimize splitting these clusters. As the angle between a track and jet decreases, efficiency decreases. To improve reconstruction, enter DeepCore: a CNN that is basically doing it all on its own, and helps gain efficiency with high momentum. DeepCore has its own drawbacks: not clear if ti's learning correctly, doesn’t work well for endcaps, inefficient predictions, and constrains data representations. We want a GNN instead to help mitigate these drawbacks.

## I/O Details & Assumptions
* I will start by using some sample input data, which will be an idealized set of data I’ll use from CMS in testing. Input data will consist of 4 2D grids (each with unique dims) of pixels, jet information, and truth information. Each pixel will have it’s own 2D+1 coordinate and charge value. Jet info will be the recorded momentum and rapidity from ECAL. Truth information will be an exact XY hit in each layer per track (4N total). 
* Output will be the predicted number of tracks that cross a given pixel for each pixel in 2D+1 space, and the kinematic information of the calculated tracks (to be used for seed).
* All particles are charged, ultra-relativistic, non-radiative, and always leave a trace. I.e., tracks are helixes, coaxial to beam line, with momentum comparable to energy detected in ECAL. All lit pixels correspond to tracks.
* Tracks do not have to cross all 4 layers.
* Tracks are randomly distributed around the known jet axis. Gives some approximation for how many tracks we’re dealing with.

## Current Implementation
* Can now be run on real CMSSW data, processed through the onboard nTuplizer, modified from DeepCore.
* Taking in 8d data and returns track ID predictions.
* Testing is done on high-pt QCD environments in the cores of gets (delta R < 0.1). 
* All particle, track, and jet assumptions still hold.

***

## Workflow

0. `. env.sh`: sets up venv. Ensure that data is preprocessed correctly by nTuplizer into .root file (example and folder are included).
1. `python3 mlp.py`: trains a fully connected neural network to cluster with a metric learning approach. Generates trained model as file.
2. `python3 gnn.py`: pushes data though trained MLP, creates a graph in feature space with KNN with max edge length, and trains a GNN. Generates trained model as file.
3. `python3 test_eps.py`: pushes through trained MLP, creates KNN graph, pushes through trained GNN, and clusters with DBSCAN. Outputs relevant performance metrics as a function of DBSCAN epsilon parameter to easily choose a maximum. 
4. `python3 test_pt.py`: similarly evaluates performance metrics w/o averaging, per particle pt. Requires pt data to exist at nTuplizer level, which depends on the run/event.

***
<div align="center">
<strong>Huge thanks to Nick Haubrich for vital contributions to the project in the form of technical support, programming expertise, and theoretical insights!</strong>
  
***
:octopus: :octopus: :octopus: :octopus: :octopus:

