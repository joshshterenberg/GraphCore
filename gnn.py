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

    directory_path = 'QCDJan26/'

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
        
                        
            
if __name__=="__main__":
    main()
            
