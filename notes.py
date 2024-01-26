
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

        ###testing
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

        #try to use KMeans clustering to move all 4 datapoints for each x together before passing through
        #that way data will be more clustered initially, have to worry less about pairwise ratio
        #can use this because we know there's 4 clusters already, ezpz
        kmeans = KMeans(n_clusters=4).fit(X[:,:3])
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        u_lables = np.unique(labels)
        #subtract the cluster center of whatever cluster you belong to 
        xmod = []
        ymod = []
        zmod = []
        for l in range(len(u_lables)):
            xmod.append(xvals[labels == l] - centers[l][0])
            ymod.append(yvals[labels == l] - centers[l][1])
            zmod.append(zvals[labels == l] - centers[l][2])

        xmod = np.concatenate(xmod)
        ymod = np.concatenate(ymod)
        zmod = np.concatenate(zmod)

        Xmod = torch.from_numpy(np.vstack([xmod, ymod, zmod]).T)
        Xmod = Xmod.to(torch.float32)

        colors = np.zeros(len(simIDs))
        for i,uniqueID in enumerate(uniqueIDs):
            colors[simIDs==uniqueID]=i/(nUniqueIDs+1.0)
        

        fig=plt.figure(constrained_layout=True)
        fig.suptitle("Pixels in Jet by SimTrack",fontsize=14,weight='bold')
        #fig.tight_layout()
        '''
        ax=fig.add_subplot(211,projection='3d')
        ax.scatter(xvals,yvals,zvals,c=colors,cmap="hsv")
        ax.set_title("Jet")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        '''
        ax2=fig.add_subplot(111,projection='3d')
        ax2.scatter(xmod,ymod,zmod,c=colors,cmap="hsv")
        ax2.set_title("Post-MLP Representation")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")

        plt.show()

if __name__=="__main__":
    main()


