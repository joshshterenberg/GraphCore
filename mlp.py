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

from sklearn.cluster import DBSCAN, KMeans, SpectralClustering

class Net(nn.Module):
    def __init__(self,d):
        super(Net,self).__init__()
        self.d = d
        self.fc1 = nn.Linear(self.d,25)
        self.ac1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(25,25)
        self.ac2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(25,25)
        self.ac3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(25,25)
        self.ac4 = nn.LeakyReLU()
        self.fc5 = nn.Linear(25,25)
        self.ac5 = nn.LeakyReLU()
        self.fcLast = nn.Linear(25,3) #2nd dim must match gnn
        #self.double()
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.fc2(x)
        x = self.ac2(x)
        x = self.fc3(x)
        x = self.ac3(x)
        x = self.fc4(x)
        x = self.ac4(x)
        x = self.fc5(x)
        x = self.ac5(x)
        x = self.fcLast(x)
        return x 


def main():
    with uproot.open("GraphCoreNtuples.root") as f:
        tree=f['ntuples/tree']
        #plot one jet core
        n=1

        mva = Net(d=6) ## with xmod set, without should be 6
        
        opt = torch.optim.SGD(mva.parameters(),lr=.001,momentum=0.5)
        opt = torch.optim.Adam(mva.parameters(),lr=.001)
        lossfunc = losses.ContrastiveLoss()
        
        coords = tree.arrays()#,entry_start=n,entry_stop=n+1) 

        tt = input("Train? y/n: ")
        if tt=="y":
            mva.train()
            #train loop
            #pdb.set_trace()
            for epoch in range(500):
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

                        ##############
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

                        rvals = np.sqrt(xvals**2+yvals**2) #maybe calc with xmod,ymod?
                        thetavals = np.arctan2(yvals,xvals)

                        Xmod = torch.from_numpy(np.vstack([xmod, ymod, zmod, etavals, phivals, charges]).T)
                        Xmod = Xmod.to(torch.float32)
                        ###########


                    opt.zero_grad()
                    pred = mva(Xmod)
                    loss = lossfunc(pred,Y)
                    if i%100==0:
                        print("epoch {} loss: {:.5f}".format(epoch,loss))
                    loss.backward()
                    opt.step()


            #save model for later use
            torch.save(mva, 'models/trained_mlp.pth')
        else:
            #load model 
        
            mva = torch.load('models/trained_mlp.pth')
            mva.eval()


        #test visualization
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
        

        ##############
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

        rvals = np.sqrt(xvals**2+yvals**2) #maybe calc with xmod,ymod?
        thetavals = np.arctan2(yvals,xvals)

        Xmod = torch.from_numpy(np.vstack([xmod, ymod, zmod, rvals, thetavals, etavals, phivals, charges]).T)
        Xmod = Xmod.to(torch.float32)
        ##########

        pred = mva(Xmod)

        colors = np.zeros(len(simIDs))
        for i,uniqueID in enumerate(uniqueIDs):
            colors[simIDs==uniqueID]=i/(nUniqueIDs+1.0)
        

        fig=plt.figure(constrained_layout=True)
        ax=fig.add_subplot(121,projection='3d')
        
        #fig.tight_layout()

        ax.scatter(xmod,ymod,zmod,c=colors,cmap="hsv")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
        fig.suptitle("Pixels in Jet by SimTrack",fontsize=18,weight='bold')
        ax.set_title("Jet pT: {:.0f} GeV, $\eta$: {:.2f}, $\phi$: {:.2f}\n {} Pixels, {} SimTracks".format(jetPt,jetEta,jetPhi,len(xvals),nUniqueIDs),fontsize=18)

        ax2=fig.add_subplot(122,projection='3d')
        ax2.scatter(pred[:,0].detach().numpy(),pred[:,1].detach().numpy(),pred[:,2].detach().numpy(),c=colors,cmap="hsv")
        ax2.set_title("Learned Representation")
        #ax2=fig.add_subplot(122)
        #ax2.scatter(pred[:,0].detach().numpy(),pred[:,1].detach().numpy(),c=colors,cmap="hsv")
        ax2.set_xlabel("v1")
        ax2.set_ylabel("v2")
        ax2.set_zlabel("v3")

        x = input("Show sample graph? y/n")
        if x=="y": plt.show()


if __name__ == "__main__":
    main()
