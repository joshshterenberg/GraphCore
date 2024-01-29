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


def k_means_mod(X):
    kmeans = KMeans(n_clusters=4).fit(X[:,:3])
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    u_lables = np.unique(labels)
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

    Xmod = torch.from_numpy(np.vstack([xmod, ymod, zmod, etavals, phivals, charges]).T)
    Xmod = Xmod.to(torch.float32)
    return Xmod




def main():

    directory_path = 'QCDJan26/'

    for filename in os.listdir(directory_path): ##next 3 lines modify for all files in folder PATH
        if os.path.isfile(os.path.join(directory_path, filename)) and filename.endswith('.root'):
            with uproot.open(os.path.join(directory_path, filename)) as f:
            #with uproot.open("GraphCoreNtuples.root") as f:

                tree=f['ntuples/tree']
                #plot one jet core
                n=1

                mva = Net(d=6) ## with xmod set, without should be 6
                
                opt = torch.optim.SGD(mva.parameters(),lr=.001,momentum=0.5)
                opt = torch.optim.Adam(mva.parameters(),lr=.001)
                lossfunc = losses.ContrastiveLoss()
                
                coords = tree.arrays()#,entry_start=n,entry_stop=n+1) 

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

                            #try shifting each point given tracker layer in tree


                        opt.zero_grad()
                        pred = mva(X) #Xmod
                        loss = lossfunc(pred,Y)
                        if i%100==0:
                            print("epoch {} loss: {:.5f}".format(epoch,loss))
                        loss.backward()
                        opt.step()


    #save model for later use
    torch.save(mva, 'models/trained_mlp.pth')



if __name__ == "__main__":
    main()
