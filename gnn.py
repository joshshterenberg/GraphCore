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
from sklearn.neighbors import kneighbors_graph

from mlp import Net, OctopiDataset

from sklearn.cluster import DBSCAN, KMeans, SpectralClustering

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR

import time
import glob



def read_root(filename,treename,branches):
    #read single root file to memory
    with uproot.open(filename) as f:
        tree = f[treename]
        output = tree.arrays(branches)
        return output


@torch.jit.script
def PairwiseHingeLoss(pred,y):
    #TODO: split attractive/repulsive losses so we can scale their relative contributions
    dists = torch.pdist(pred).flatten()
    ys = torch.pdist(y.to(torch.float).unsqueeze(0).T,0.0).flatten() #0-norm: 0 if same, 1 if different
    ys = -2*ys+1 #map 0,1 to -1,1
    return  torch.nn.functional.hinge_embedding_loss(dists,ys,margin=1.0)


class GCN(nn.Module):
    def __init__(self,d):         
        super().__init__()         
        #torch.manual_seed(1234)         
        self.d = d
        self.conv1 = geonn.GCNConv(self.d, 25)         
        self.conv2 = geonn.GCNConv(25, 25)
        self.conv3 = geonn.GCNConv(25, 25)
        self.conv4 = geonn.GCNConv(25, 25)
        self.conv5 = geonn.GCNConv(25, 25)
        self.classifier = nn.Linear(25, self.d)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)         
        h = h.relu()
        h = self.conv2(h, edge_index)         
        h = h.relu()         
        h = self.conv3(h, edge_index)         
        h = h.relu() #pooling here?
        h = self.conv4(h, edge_index)         
        h = h.relu() #pooling here?
        h = self.conv5(h, edge_index)         
        h = h.relu() #pooling here?
        out = self.classifier(h)         
        return out#, h


def main():


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    featureBranches = ["pixelU","pixelV","pixelEta","pixelPhi","pixelR","pixelZ","pixelCharge","pixelTrackerLayer"]
    trainDS = OctopiDataset(glob.glob("/eos/user/n/nihaubri/OctopiNtuples/QCDJan26/train/OctopiNtuples_1.root"),featureBranches=featureBranches,labelBranch="pixelSimTrackID",batchsize=20) #batches of 50 jets with ~100 pixels each
    print("training dataset has {} jets. Running {} batches".format(len(trainDS)*trainDS.batchsize,len(trainDS)))

    valDS = OctopiDataset(glob.glob("/eos/user/n/nihaubri/OctopiNtuples/QCDJan26/train/OctopiNtuples_11.root"),featureBranches=featureBranches,labelBranch="pixelSimTrackID",batchsize=500) #GPU can handle it (5GB VRAM usage), so seems fine for validation


    #directory_path = 'QCDJan26/'

    #load models
    mva = torch.load('models/trained_mlp.pth')
    mva.to(device).eval()
    model = GCN(d=3).to(device)

    opt = torch.optim.Adam(mva.parameters(),lr=.001)

    scaler = GradScaler()
    scheduler = StepLR(opt, step_size=3, gamma=0.5)

    epochLosses = []
    epochValLosses = []
   

    for epoch in range(10):
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
            Y=Y.to(device)
            opt.zero_grad()

            #mixed precision with torch.cuda.amp
            #added with function, modified backward and step.
            with autocast():

                l1 = mva(X) #mlp

                #potential crunch here
                edge_index = geonn.knn_graph(l1, k=4) #overkill, mod to use initial data
                #edge_index = kneighbors_graph(l1, 4)
                
                pred = model(l1, edge_index) #gnn

                predsplit = torch.tensor_split(pred,tuple(sizeList),dim=0)
                ysplit = torch.tensor_split(Y,tuple(sizeList),dim=0)
                batchLoss = torch.zeros(1,device=device)
                for j,(jetPred,jetY) in enumerate(zip(predsplit,ysplit)): #vectorize this somehow?
                    if jetY.shape[0]==1: #needed for jan26 ntuples but not later
                        continue
                    batchLoss+=PairwiseHingeLoss(jetPred,jetY)
                
                if i%50==epoch:
                    print("batch {} loss: {:.5f}".format(i,float(batchLoss.detach())))
                epochLoss+=batchLoss.detach()


            #batchLoss.backward()
            #opt.step()
            scaler.scale(batchLoss).backward()
            scaler.step(opt)
            scaler.update()


        epochLosses.append(float(epochLoss))
        scheduler.step()
        
        mva.eval()
        for i,(X,Y,sizeList) in enumerate(valDS):
            if i>len(valDS):
                i=0
                break

            X=X.to(device)
            Y=Y.to(device)

            l1 = mva(X) #mlp

            #potential crunch here
            edge_index = geonn.knn_graph(l1, k=4) #overkill, mod to use initial data
            #edge_index = kneighbors_graph(l1, 4)
            
            pred = model(l1, edge_index) #gnn

            predsplit = torch.tensor_split(pred,tuple(sizeList),dim=0)
            ysplit = torch.tensor_split(Y,tuple(sizeList),dim=0)
            for (jetPred,jetY) in zip(predsplit,ysplit): 
                if jetY.shape[0]==1: #needed for jan26 ntuples but not later
                    continue
                epochValLoss+=PairwiseHingeLoss(jetPred,jetY).detach()
        epochValLosses.append(float(epochValLoss))
        print("Epoch time: {:.2f} Training Loss: {:.2f} Validation Loss: {:.2f}".format(time.time()-epochStart,epochLosses[-1],epochValLosses[-1]))

    plt.plot(epochLosses,label='training')
    plt.plot(epochValLosses,label='validation')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig("loss_gnn.png")

    #save model for later use
    torch.save(mva, 'models/trained_gnn.pth')










    '''

    for filename in os.listdir(directory_path): ##next 3 lines modify for all files in folder PATH
        if os.path.isfile(os.path.join(directory_path, filename)) and filename.endswith('.root'):
            with uproot.open(os.path.join(directory_path, filename)) as f:
            #with uproot.open("GraphCoreNtuples.root") as f:

                tree = f['ntuples/tree']
                coords = tree.arrays()
                n=1
            
                #load mlp
                mva = torch.load('models/trained_mlp.pth')
                mva.eval()
        
                #define GNN
                model = GCN(d=3).to(device) #d must match mlp 
                
                opt = torch.optim.SGD(mva.parameters(),lr=.001,momentum=0.5)
                opt = torch.optim.Adam(mva.parameters(),lr=.001)
                lossfunc = losses.ContrastiveLoss()
               
                model.train() #training constitutes learning object condensation
        
                for epoch in range(500): ##S_EDIT change to 500
                    print("EPOCH {}".format(epoch)) 
                    for i in range(coords['caloJetPt'].to_numpy().shape[0]):
                        if i%2==0:
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
                            Y = Y.to(torch.float32).to(device)
            
                            #push data through model
                            with torch.no_grad():
                                latent = mva(X)
                            
                            #implement knn graph with max radius (arbitrarily 0.01 for now)
                            edge_index = geonn.knn_graph(latent, k=4)
                            #dists = (latent[edge_index[0]] - latent[edge_index[1]]).norm(dim=-1)
                            #edge_index = edge_index[:, dists < 0.01]
            
        
                        opt.zero_grad()
                        pred = model(latent.to(device), edge_index.to(device))
                        loss = lossfunc(pred,Y)
                        if i%100==0:
                            print("epoch {} loss: {:.5f}".format(epoch,loss))
                        loss.backward()
                        opt.step()
        
        
    #save model for later
    torch.save(model, "models/trained_gnn.pth")
    ''' 
                        
            
if __name__=="__main__":
    main()
            
