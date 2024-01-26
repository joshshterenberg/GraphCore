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

        #testing
        i = int(input("Type track number to test on: "))
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

        #clusterize data with DBSCAN (arbitrary hyperparams so far)
        preclust = latent_2.numpy()
        clusterizer = KMeans(eps=0.01)
        clusterizer.fit(preclust)

        #get metadata of predicted track and compare
        u_labels = np.unique(clusterizer.labels_)
        for l in u_labels:
            c_i = preclust[:, clusterizer.labels_ == l]
            avg_i = np.mean(c_i, axis=1)
            print(f"Cluster {l} Average: {avg_i}")











if __name__=="__main__":
    main()







