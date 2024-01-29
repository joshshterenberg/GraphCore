import uproot
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
    with uproot.open("GraphCoreNtuples.root") as f:
        tree = f['ntuples/tree']
        coords = tree.arrays()
        n=1

        #load mlp
        mva = torch.load('models/trained_mlp.pth')
        mva.eval()

        #load gnn
        model = torch.load('models/trained_gnn.pth')
        model.eval()


        tt = input("Plot one track or all? (o/a)")
        if tt=="o":
            #testing
            i = int(input("Type track number to plot: "))
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
            edge_index = geonn.knn_graph(latent, k=8)
            dists = (latent[edge_index[0]] - latent[edge_index[1]]).norm(dim=-1)
            edge_index = edge_index[:, dists < 0.01]

            #push data through gnn model
            with torch.no_grad():
                latent_2 = model(latent, edge_index)

            
            colors = np.zeros(len(simIDs))
            for i,uniqueID in enumerate(uniqueIDs):
                colors[simIDs==uniqueID]=i/(nUniqueIDs+1.0)
            

            fig=plt.figure(constrained_layout=True)
            fig.suptitle("Pixels in Jet by SimTrack",fontsize=14,weight='bold')
            #fig.tight_layout()

            ax=fig.add_subplot(311,projection='3d')
            ax.scatter(xvals,yvals,zvals,c=colors,cmap="hsv")
            ax.set_title("Jet")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

            ax2=fig.add_subplot(312,projection='3d')
            ax2.scatter(latent[:,0].detach().numpy(),latent[:,1].detach().numpy(),latent[:,2].detach().numpy(),c=colors,cmap="hsv")
            ax2.set_title("Post-MLP Representation")
            ax2.set_xlabel("v1")
            ax2.set_ylabel("v2")
            ax2.set_zlabel("v3")
            

            ax3=fig.add_subplot(313,projection='3d')
            ax3.scatter(latent_2[:,0].detach().numpy(),latent_2[:,1].detach().numpy(),latent_2[:,2].detach().numpy(),c=colors,cmap="hsv")
            ax3.set_title("Post-GNN Representation")
            ax3.set_xlabel("v1")
            ax3.set_ylabel("v2")
            ax3.set_zlabel("v3")

            plt.show()

        elif tt=="a":
            parts = []
            parts2 = []

            for i in range(coords['caloJetPt'].to_numpy().shape[0]):
                if i%2==1: #odds for testing
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
                    if -1 in u_labels: parts2.append((len(u_labels)-1)/len(n_particles))
                    else: parts2.append(len(u_labels)/len(n_particles))
   
                    # metric definitions:
                    for l in u_labels:
                        if l==-1: continue

                        extraneous_i = 0
                        total_sims = []
                        for i in range(len(preclust)):
                            if clusterizer.labels_[i] == l:
                                total_sims.append(Y[i])
                        all = len(total_sims)
                        counts = Counter(total_sims)
                        mce = counts.most_common(1)[0][0]
                        total_sims = [item for item in total_sims if item != mce]
                        extraneous_i = len(total_sims)
                        match_eff = 1 - (extraneous_i/all)
                        parts.append(match_eff)
            #plot
            ids = list(range(len(parts)))
            ids2 = list(range(len(parts2)))

            fig=plt.figure(constrained_layout=True)
            fig.suptitle("Match efficiencies and % of recognized particles")

            ax=fig.add_subplot(121)
            ax.scatter(ids, parts, cmap="hsv")
            ax.set_ylabel("efficiency %")
            ax.set_xlabel("cluster")
            ax.set_title("Match efficiencies (100% = no outliers)")

            ax2=fig.add_subplot(122)
            ax2.scatter(ids2, parts2, cmap="hsv")
            ax2.set_ylabel("% of recognized particles")
            ax2.set_xlabel("jet")
            ax2.set_title("% of recognized particles per jet")


            plt.show()
    
    

if __name__=="__main__":
    main()
