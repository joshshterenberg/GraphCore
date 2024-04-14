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
from torch_geometric.typing import torch_cluster
from sklearn.neighbors import NearestNeighbors

from mlp import Net, OctopiDataset

from sklearn.cluster import DBSCAN, KMeans, SpectralClustering

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
import torch.onnx
from skl2onnx import to_onnx

import time
import glob

#####
if len(sys.argv) > 1:
    lsize = sys.argv[1]
else:
    lsize = 10
print(f"LSize: {lsize}")
lsize = float(lsize)
#####



def read_root(filename,treename,branches):
    #read single root file to memory
    with uproot.open(filename) as f:
        tree = f[treename]
        output = tree.arrays(branches)
        return output


@torch.jit.script
def PairwiseHingeLoss(pred,y,a = torch.tensor(1.0)):
    dists = torch.pdist(pred).flatten()
    ys = torch.pdist(y.to(torch.float).unsqueeze(0).T,0.0).flatten() #0-norm: 0 if same, 1 if different


    hinge_part = torch.max(torch.tensor(0.),a*(1. - dists)) #TODO: margin ('1'-x) is arbitrary and dependent on scale factor of patent space. Needs optimization.
    return torch.mean(torch.where(ys==0.0, dists, hinge_part))


class GNN(nn.Module):
    def __init__(self,d):         
        super().__init__()         
        #torch.manual_seed(1234)         
        self.d = d
        self.conv1 = geonn.GCNConv(self.d, 15) #was GCNConv, going for simplicity  
        self.conv2 = geonn.GCNConv(15,15)
        self.conv3 = geonn.GCNConv(15,15)
        #self.conv4 = geonn.GCNConv(15,15)
        #self.conv5 = geonn.GCNConv(15,15)
        self.classifier = nn.Linear(15, self.d)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)         
        h = h.relu() #pooling here?
        h = self.conv2(h, edge_index)         
        h = h.relu()         
        h = self.conv3(h, edge_index)         
        h = h.relu()
        #h = self.conv4(h, edge_index)         
        #h = h.relu()
        #h = self.conv5(h, edge_index)         
        #h = h.relu()
        out = self.classifier(h) #what does this do
        return out#, h


def main():


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if (device == 'cpu'):
        quit() #needs to run on gpu
    
    featureBranches = ["pixelU","pixelV","pixelEta","pixelPhi","pixelR","pixelZ","pixelCharge","pixelTrackerLayer"]
    labelBranches = ["pixelSimTrackID", "pixelSimTrackPt"]
    trainDS = OctopiDataset(glob.glob("/eos/user/n/nihaubri/OctopiNtuples/QCDMar7/OctopiNtuples_1.root"),featureBranches=featureBranches,labelBranches=labelBranches,batchsize=20) #batches of 50 jets with ~100 pixels each ###mod less data for train
    print("training dataset has {} jets. Running {} batches".format(len(trainDS)*trainDS.batchsize,len(trainDS)))

    valDS = OctopiDataset(glob.glob("/eos/user/n/nihaubri/OctopiNtuples/QCDMar7/OctopiNtuples_59.root"),featureBranches=featureBranches,labelBranches=labelBranches,batchsize=500) #GPU can handle it (5GB VRAM usage), so seems fine for validation


    #directory_path = 'QCDJan26/'

    #load models
    mva = torch.load(f"models/trained_mlp_{lsize}.pth")
    mva.to(device).eval()
    model = GNN(d=len(featureBranches)).to(device)

    opt = torch.optim.Adam(mva.parameters(),lr=0.001)

    scaler = GradScaler()
    scheduler = StepLR(opt, step_size=10, gamma=0.5)

    epochLosses = []
    epochValLosses = []
   

    for epoch in range(20): #was 20
        epochLoss = torch.zeros(1,device=device).detach()
        epochValLoss = torch.zeros(1,device=device).detach()

        model.train()
        print("EPOCH {}".format(epoch)) 
        
        epochStart = time.time()
        
        for i,(X,Y,sizeList) in enumerate(trainDS):
            if i>len(trainDS):
                i=0
                break
            
            X=X.to(device)
            Y=Y[0].to(device)
            opt.zero_grad()

            #mixed precision with torch.cuda.amp
            #added with function, modified backward and step.
            with autocast():

                l1 = mva(X)#.to('cpu').detach() #mlp
                edge_index = geonn.knn_graph(l1, k=5) #trying something new
                #neigh = NearestNeighbors(n_neighbors = 5)
                #neigh.fit(l1)
                #edge_matrix = neigh.kneighbors_graph(l1).toarray()
                #edge_index = []
                #for i in range(len(l1)):
                #    for j in range(i):
                #        if edge_matrix[i][j] != 0:
                #            edge_index.append([i, j])
                #edge_index = [[row[i] for row in edge_index] for i in range(len(edge_index[0]))]

                if i == 0: # needed for ONNX
                    dummy_input_knn = l1
                    dummy_input_gnn = (l1, edge_index)

                pred = model(l1.to(device), edge_index) #gnn

                predsplit = torch.tensor_split(pred,tuple(sizeList),dim=0)
                ysplit = torch.tensor_split(Y,tuple(sizeList),dim=0)
                batchLoss = torch.zeros(1,device=device)
                for j,(jetPred,jetY) in enumerate(zip(predsplit,ysplit)): #vectorize this somehow?
                    if jetY.shape[0]==1: #needed for jan26 ntuples but not later
                        continue
                    batchLoss+=PairwiseHingeLoss(jetPred,jetY, torch.tensor(lsize))
                
                if i%50==epoch:
                    print("batch {} loss: {:.5f}".format(i,float(batchLoss.detach())))
                epochLoss+=batchLoss.detach()


            batchLoss.backward()
            opt.step()
            #scaler.scale(batchLoss).backward() ###put this back for the real thing
            #scaler.step(opt)
            #scaler.update()


        epochLosses.append(float(epochLoss)/float(trainDS.count))
        scheduler.step()
        
        mva.eval()
        for i,(X,Y,sizeList) in enumerate(valDS):
            if i>len(valDS):
                i=0
                break

            X=X.to(device)
            Y=Y[0].to(device)

            l1 = mva(X)#.to('cpu').detach() #mlp
            edge_index = geonn.knn_graph(l1, k=5) #trying something new
            #neigh = NearestNeighbors(n_neighbors = 5)
            #neigh.fit(l1)
            #edge_matrix = neigh.kneighbors_graph(l1).toarray()
            #edge_index = []
            #for i in range(len(l1)):
            #    for j in range(i):
            #        if edge_matrix[i][j] != 0:
            #            edge_index.append([i,j])
            
            pred = model(l1.to(device), edge_index.to(device)) #gnn

            predsplit = torch.tensor_split(pred,tuple(sizeList),dim=0)
            ysplit = torch.tensor_split(Y,tuple(sizeList),dim=0)
            batchLoss = torch.zeros(1, device=device)
            for (jetPred,jetY) in zip(predsplit,ysplit): 
                if jetY.shape[0]==1: #needed for jan26 ntuples but not later
                    continue
                epochValLoss+=PairwiseHingeLoss(jetPred,jetY, torch.tensor(lsize)).detach()
        epochValLosses.append(float(epochValLoss)/float(valDS.count))
        print("Epoch time: {:.2f} Training Loss: {:.2f} Validation Loss: {:.2f}".format(time.time()-epochStart,epochLosses[-1],epochValLosses[-1]))

    plt.plot(epochLosses,label='training')
    plt.plot(epochValLosses,label='validation')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig(f"plots/loss_gnn_{lsize}.png")

    #save model for later use
    torch.save(model, f"models/trained_gnn_{lsize}.pth")

    print("Saved model successfully")

    ###ONNX saving###
    neigh = NearestNeighbors(n_neighbors = 5) ##NEEDS PARSING FROM EDGE MATRIX TO EDGE ARRAY
    onx = to_onnx(neigh, dummy_input_knn)
    with open("knn.onnx","wb") as f:
        f.write(onx.SerializeToString())

    torch.onnx.export(model, dummy_input_gnn, "gnn.onnx", verbose=True, input_names=['my_input2'], output_names=['my_output2'])
    ######

if __name__=="__main__":
    main()
            
