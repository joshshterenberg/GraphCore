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

from mlp import Net, OctopiDataset
#from gnn import GCN

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR

from collections import Counter

import multiprocessing
import time
import glob

from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
#from cuml.cluster import DBSCAN

#####
lsize = sys.argv[1]
print(f"LSize: {lsize}")
lsize = int(lsize)
#####




def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    featureBranches = ["pixelU","pixelV","pixelEta","pixelPhi","pixelR","pixelZ","pixelCharge","pixelTrackerLayer"]

    
    testDS = OctopiDataset(glob.glob("/eos/user/n/nihaubri/OctopiNtuples/QCDJan31/test/*"), featureBranches=featureBranches,labelBranch="pixelSimTrackID",batchsize=1)

    print("test dataset has {} jets. Running {} batches".format(len(testDS)*testDS.batchsize,len(testDS)))
    #directory_path = 'QCDJan26/'

    #load models
    mva = torch.load(f"models/trained_mlp_{lsize}.pth")
    mva.to(device).eval()
    #model = torch.load('models/trained_gnn.pth')
    #model.eval()



    #define vars for metric calc, using pseudo-exponential dist below 0.4
    #EPS_arr = [0.001, 0.002, 0.005, 0.011, 0.022, 0.046, 0.095, 0.194, 0.290, 0.400]
    EPS_arr = [0.001, 0.011, 0.021, 0.031, 0.041, 0.051, 0.061, 0.071, 0.081, 0.091, 1.001]
    #EPS_arr = [0.001] #change
    LHC_arr = []
    perfect_arr = []
    for EPS in EPS_arr:
    
        cluster_ratios = []
        LHC_matches = []
        perfect_matches = []
        LHC_percents = []
        perfect_percents = []

        #print("\n")
        #print("Metric 0: Trend of number (%) of recognized clusters in sample (saved to cluster_ratios.png).")
        #print("Metric 1: Total average of match efficiencies greater than 75% (LHC matches).")
        #print("\t get m.e. per cluster, how many m.e.>0.75 is X%, average.")
        #print("Metric 2: Same as metric 1, but for match efficiencies exactly 100% (perfect matches).")
        #print("\nCalculating metrics...")

        
        
        for i,(X,Y,sizeList) in enumerate(testDS):
            if i>len(testDS):
                i=0
                break

            print(f"EPS {EPS}. Calculating {i} of {len(testDS)}.")

            X=X.to(device)

            pred = mva(X) #mlp, not using gnn for now
        
            #here must move data to CPU for numpy
            pred = pred.cpu()
            if pred.shape[0] <= 0: 
                print("Empty datapoint, can't apply DBSCAN")
                continue

            #clusterize data with DBSCAN (arbitrary hyperparams so far)
            pred = pred.detach().numpy()
            clusterizer = DBSCAN(eps=EPS, min_samples=3) # to match knn graph
            clusterizer.fit(pred)


            #get metadata of predicted track and compare
            u_labels = np.unique(clusterizer.labels_)
            n_particles = np.unique(Y)
            cluster_ratios.append(len(u_labels)/len(n_particles))
            
            
            # metric definitions:
            LHC_match_efficiencies = []
            perfect_match_efficiencies = []
            for l in u_labels:
                if l==-1: continue
            
                extraneous_i = 0
                total_sims = []
                for i in range(len(pred)):
                    if clusterizer.labels_[i] == l:
                        total_sims.append(Y[i])
                #print(f"Cluster {l}:", end=" ")
                all = len(total_sims)
                counts = Counter(total_sims)
                mce = counts.most_common(1)[0][0]
                total_sims = [item for item in total_sims if item != mce]
                extraneous_i = len(total_sims)
                match_eff = 1 - (extraneous_i/all)
                #print(f"Ratio: {extraneous_i}/{all}. Match efficiency: {match_eff}.", end=" ")
                #if match_eff > 0.75: print("(LHC Match)")
                #if match_eff == 1: print("(Perfect Match)")
                #print()
                if match_eff == 1: perfect_match_efficiencies.append(match_eff)
                if match_eff >= 0.75: LHC_match_efficiencies.append(match_eff)
            LHC_percent = len(LHC_match_efficiencies)/len(u_labels)
            perfect_percent = len(perfect_match_efficiencies)/len(u_labels)
        
            LHC_percents.append(LHC_percent)
            perfect_percents.append(perfect_percent)
            #if LHC_percent >= 0.9: LHC_matches.append(1)
            #else: LHC_matches.append(0)
            #if perfect_percent >= 0.9: perfect_matches.append(1)
            #else: perfect_matches.append(0)


        LHC_arr.append(np.mean(LHC_percents))
        perfect_arr.append(np.mean(perfect_percents))
        
        

    print("Done, generating plots")
    plt.plot(EPS_arr, LHC_arr, label='LHC')
    plt.plot(EPS_arr, perfect_arr, label='perfect')
    plt.xlabel("EPS")
    plt.ylabel("Percent")
    plt.title("Performance vs. DBSCAN EPS")
    plt.legend()
    plt.savefig(f"eps_graph_{lsize}.png")
    plt.close()
    


    '''

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
        
    '''

if __name__=="__main__":
    main()
        
        
        
        
        
        
        
