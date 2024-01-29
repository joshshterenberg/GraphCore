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
    with uproot.open("GraphCoreNtuples.root") as f:
        tree = f['ntuples/tree']
        coords = tree.arrays()
        n=1
    
        #load mlp
        mva = torch.load('models/trained_mlp.pth')
        mva.eval()

        #define GNN
        model = GCN(d=3) #d must match mlp 
        
        opt = torch.optim.SGD(mva.parameters(),lr=.001,momentum=0.5)
        opt = torch.optim.Adam(mva.parameters(),lr=.001)
        lossfunc = losses.ContrastiveLoss()
       
        tt = input("Train? y/n: ")
        if tt=="y":
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
                        Y = Y.to(torch.float32)
    
                        #push data through model
                        with torch.no_grad():
                            latent = mva(X)
                        
                        #implement knn graph with max radius (arbitrarily 0.01 for now)
                        edge_index = geonn.knn_graph(latent, k=4)
                        #dists = (latent[edge_index[0]] - latent[edge_index[1]]).norm(dim=-1)
                        #edge_index = edge_index[:, dists < 0.01]
    

                    opt.zero_grad()
                    pred = model(latent, edge_index)
                    loss = lossfunc(pred,Y)
                    if i%100==0:
                        print("epoch {} loss: {:.5f}".format(epoch,loss))
                    loss.backward()
                    opt.step()


            #save model for later
            torch.save(model, "models/trained_gnn.pth")

        else:
            #load model 
        
            model = torch.load('models/trained_gnn.pth')
            model.eval()

                
        #test visualization, same as in mlp
        i=5
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

        pred = model(latent, edge_index)

        colors = np.zeros(len(simIDs))
        for i,uniqueID in enumerate(uniqueIDs):
            colors[simIDs==uniqueID]=i/(nUniqueIDs+1.0)
        

        fig=plt.figure(constrained_layout=True)
        ax=fig.add_subplot(121,projection='3d')
        
        #fig.tight_layout()

        ax.scatter(latent[:,0].detach().numpy(),latent[:,1].detach().numpy(),latent[:,2].detach().numpy(),c=colors,cmap="hsv")
        ax.set_xlabel("v1")
        ax.set_ylabel("v2")
        ax.set_zlabel("v3")
        
        fig.suptitle("Pixels in Jet by SimTrack",fontsize=18,weight='bold')
        ax.set_title("Learned Representation")

        ax2=fig.add_subplot(122,projection='3d')
        ax2.scatter(pred[:,0].detach().numpy(),pred[:,1].detach().numpy(),pred[:,2].detach().numpy(),c=colors,cmap="hsv")
        ax2.set_title("Post-Object Condensation Representation")
        #ax2=fig.add_subplot(122)
        #ax2.scatter(pred[:,0].detach().numpy(),pred[:,1].detach().numpy(),c=colors,cmap="hsv")
        ax2.set_xlabel("v1")
        ax2.set_ylabel("v2")
        ax2.set_zlabel("v3")

        x = input("Show sample graph? y/n")
        if x=="y": plt.show()
    
    
if __name__=="__main__":
    main()
    
