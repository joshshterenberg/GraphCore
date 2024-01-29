import uproot
import os
import matplotlib.pyplot as plt
import sys
import pdb
import numpy as np
from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import distances, losses, miners, reducers, testers
import torch_geometric.nn as geonn

from mlp import Net
from gnn import GCN

from collections import Counter

from sklearn.cluster import DBSCAN, KMeans, SpectralClustering

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    directory_path = 'QCDJan26/'

    test_filename = input("Filename: ")
    i = int(input("Type track number to test: "))

    if os.path.isfile(os.path.join(directory_path, test_filename)):
        with uproot.open(os.path.join(directory_path, test_filename)) as f:
    
            
            tree = f['ntuples/tree']
            coords = tree.arrays()
            n=1
        
            #load mlp
            mva = torch.load('models/trained_mlp.pth')
            mva.eval()
        
            #load gnn
            model = torch.load('models/trained_gnn.pth')
            model.eval()
        
            #testing
            xvals = coords['pixelX'][i].to_numpy()
            yvals = coords['pixelY'][i].to_numpy()
            zvals = coords['pixelZ'][i].to_numpy()
            etavals = coords['pixelEta'][i].to_numpy()
            phivals= coords['pixelPhi'][i].to_numpy()
            charges = coords['pixelCharge'][i].to_numpy()
            simIDs = coords["pixelSimTrackID"][i].to_numpy()
        
            jetPt = coords['caloJetPt'][i]
            jetEta = coords['caloJetEta'][i]
            jetPhi = coords['caloJetPhi'][i]
        
            uniqueIDs = set(simIDs)
            nUniqueIDs = len(uniqueIDs)
        
            X = torch.from_numpy(np.vstack([xvals,yvals,zvals,etavals,phivals,charges]).T)
            X = X.to(torch.float32)
            Y = torch.from_numpy(simIDs)
            Y = Y.to(torch.float32)
        
            #push data through model
            with torch.no_grad():
                latent = mva(X)
        
            #implement knn graph with max radius (arbitrarily 0.01 for now)
            edge_index = geonn.knn_graph(latent, k=4)
            #dists = (latent[edge_index[0]] - latent[edge_index[1]]).norm(dim=-1)
            #edge_index = edge_index[:, dists < 0.01]
        
            #push data through gnn model
            with torch.no_grad():
                latent_2 = model(latent, edge_index)
        
            #clusterize data with DBSCAN (arbitrary hyperparams so far)
            preclust = latent_2.numpy()
            clusterizer = DBSCAN(eps=0.01, min_samples=3) # to match knn graph
            clusterizer.fit(preclust)
        
        
        
            #get metadata of predicted track and compare
            u_labels = np.unique(clusterizer.labels_)
            n_particles = np.unique(Y)
            print("Metric 0: Number of clusters vs number of true particles.")
            print(f"Particles: {n_particles}. Clusters: {u_labels}")
        
        
            for l in u_labels:
                if l == -1: continue
        
                c_i = preclust[clusterizer.labels_ == l]
                avg_i = []
                for j in range(len(preclust[0])):
                    avg_i.append(np.mean(c_i[:,j]))
                print(f"Cluster {l} Average: {avg_i}")
               
        
            print()
        
            # metric definitions:
            print("Metric 1: % of various match efficiencies.")
            for l in u_labels:
                if l==-1: continue
        
                extraneous_i = 0
                total_sims = []
                for i in range(len(preclust)):
                    if clusterizer.labels_[i] == l:
                        total_sims.append(Y[i])
                print(f"Cluster {l}:", end=" ")
                all = len(total_sims)
                counts = Counter(total_sims)
                mce = counts.most_common(1)[0][0]
                total_sims = [item for item in total_sims if item != mce]
                extraneous_i = len(total_sims)
                match_eff = 1 - (extraneous_i/all)
                print(f"Ratio: {extraneous_i}/{all}. Match efficiency: {match_eff}.", end=" ")
                if match_eff > 0.75: print("(LHC Match)")
                if match_eff == 1: print("(Perfect Match)")
                print()
        
        
if __name__=="__main__":
    main()
        
        
        
        
        
        
        
