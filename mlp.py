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

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import time
import glob



def read_root(filename,treename,branches):
    #read single root file to memory
    with uproot.open(filename) as f:
        tree = f[treename]
        output = tree.arrays(branches)
        return output


class OctopiDataset(Dataset):
    def __init__(self,filelist,featureBranches,labelBranch,batchsize):
        import awkward as ak
        self.filelist = filelist
        self.featureBranches = featureBranches
        self.labelBranch = labelBranch
        self.batchsize = batchsize

        
        #read everything into memory '_'
        self.input_array = ak.Array([])
        #self.input_list = []
        print("reading files into memory")
        for f in filelist:
            print(f)
            self.input_array= ak.concatenate([self.input_array, read_root(f,'ntuples/tree',self.featureBranches+[self.labelBranch])]) #10G virt, 4G real

        self.count = int(ak.num(self.input_array,axis=0))
        print("done")

    def __len__(self):
        return int(self.count/self.batchsize)

    def __getitem__(self,idx):
        import awkward as ak
        item = self.input_array[idx:idx+self.batchsize]
        
        akflat = [ak.flatten(item[branch]).to_numpy() for branch in self.featureBranches]
        npstack = np.vstack(akflat,dtype=np.float32)
        X = torch.from_numpy(npstack)

        Y = torch.from_numpy(ak.flatten(item[self.labelBranch]).to_numpy())
        sizeList = torch.tensor(ak.count(item[self.labelBranch],axis=1)).cumsum(axis=0)[:-1]
        return X.T,Y,sizeList

@torch.jit.script
def PairwiseHingeLoss(pred,y):
    dists = torch.pdist(pred).flatten()
    ys = torch.pdist(y.to(torch.float).unsqueeze(0).T,0.0).flatten() #0-norm: 0 if same, 1 if different
    ys = 2*ys-1 #map 0,1 to -1,1
    return  torch.nn.functional.hinge_embedding_loss(dists,ys,margin=1.0)


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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    featureBranches = ["pixelU","pixelV","pixelEta","pixelPhi","pixelR","pixelZ","pixelCharge","pixelTrackerLayer"]
    trainDS = OctopiDataset(glob.glob("/eos/user/n/nihaubri/OctopiNtuples/QCDJan26/train/OctopiNtuples_1.root"),featureBranches=featureBranches,labelBranch="pixelSimTrackID",batchsize=20) #batches of 50 jets with ~100 pixels each
    print("training dataset has {} jets. Running {} batches".format(len(trainDS)*trainDS.batchsize,len(trainDS)))

    valDS = OctopiDataset(glob.glob("/eos/user/n/nihaubri/OctopiNtuples/QCDJan26/train/OctopiNtuples_11.root"),featureBranches=featureBranches,labelBranch="pixelSimTrackID",batchsize=500) #GPU can handle it (5GB VRAM usage), so seems fine for validation


    mva = Net(d=len(featureBranches)).to(device) ## with xmod set, without should be 6
    opt = torch.optim.Adam(mva.parameters(),lr=.001)

    lossVals = []
    valLossVals = []
    for epoch in range(10):
        mva.train()
        print("EPOCH {}".format(epoch)) 
        
        epochStart = time.time()
        
        for i,(X,Y,sizeList) in enumerate(trainDS):
            if i>len(trainDS):
                i=0
                break
            
            X=X.to(device)
            Y=Y.to(device)
            opt.zero_grad()

            pred = mva(X) #Xmod
            predsplit = torch.tensor_split(pred,tuple(sizeList),dim=0)
            ysplit = torch.tensor_split(Y,tuple(sizeList),dim=0)
            totLoss = torch.zeros(1,device=device)
            for j,(jetPred,jetY) in enumerate(zip(predsplit,ysplit)): #vectorize this somehow?
                if jetY.shape[0]==1: #needed for jan26 ntuples but not later
                    continue
                totLoss+=PairwiseHingeLoss(jetPred,jetY)
            
            if i%50==epoch:
                print("mini {} loss: {:.5f}".format(i,float(totLoss)))
                lossVals.append(float(totLoss))
            totLoss.backward()
            opt.step()
        
        print("Epoch time: {:.2f}".format(time.time()-epochStart))
        
        mva.eval()
        valLoss = torch.zeros(1,device=device)
        for i,(X,Y,sizeList) in enumerate(valDS):
            if i>len(valDS):
                i=0
                break

            X=X.to(device)
            Y=Y.to(device)
            opt.zero_grad()

            pred = mva(X) 

            predsplit = torch.tensor_split(pred,tuple(sizeList),dim=0)
            ysplit = torch.tensor_split(Y,tuple(sizeList),dim=0)
            for (jetPred,jetY) in zip(predsplit,ysplit): 
                if jetY.shape[0]==1: #needed for jan26 ntuples but not later
                    continue
                valLoss+=PairwiseHingeLoss(jetPred,jetY)
        valLossVals.append(float(valLoss))


    plt.plot(lossVals)
    plt.savefig("loss.png")
    plt.clf()
    plt.plot(valLossVals)
    plt.savefig("valloss.png")

    #save model for later use
    torch.save(mva, 'models/trained_mlp.pth')


if __name__ == "__main__":
    main()
