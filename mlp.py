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
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
import torch.onnx

import time
import glob


#####
if len(sys.argv) > 1:
    lsize = sys.argv[1] ###temp
else:
    lsize = 10
print(f"LSize: {lsize}")
lsize = float(lsize)
#####


class OctopiDataset(Dataset):

    def read_root(self, filename,treename,branches):
        #read single root file to memory
        with uproot.open(filename) as f:
            tree = f[treename]
            output = tree.arrays(branches)
            return output

    def __init__(self,filelist,featureBranches,labelBranches,batchsize):
        import awkward as ak
        self.filelist = filelist
        self.featureBranches = featureBranches
        self.labelBranches = labelBranches
        self.batchsize = batchsize

        
        #read everything into memory '_'
        self.input_array = ak.Array([])
        #self.input_list = []
        print("reading files into memory")
        for f in filelist:
            print(f)
            self.input_array= ak.concatenate([self.input_array, self.read_root(f,'ntuples/tree',self.featureBranches+self.labelBranches)]) #10G virt, 4G real

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

        akflat2 = [ak.flatten(item[branch]).to_numpy() for branch in self.labelBranches]
        npstack2 = np.vstack(akflat2,dtype=np.float32)
        Y = torch.from_numpy(npstack2)

        sizeList = torch.tensor(ak.count(item[self.labelBranches[0]],axis=1)).cumsum(axis=0)[:-1]
        return X.T,Y,sizeList



@torch.jit.script
def PairwiseHingeLoss(pred,y,a = torch.tensor(1.0)):
    dists = torch.pdist(pred).flatten()
    ys = torch.pdist(y.to(torch.float).unsqueeze(0).T,0.0).flatten() #0-norm: 0 if same, 1 if different
    hinge_part = torch.max(torch.tensor(0),a*(1 - dists)) #TODO: margin ('1'-x) is arbitrary and dependent on scale factor of patent space. Needs optimization.
    return torch.mean(torch.where(ys==0.0, dists, hinge_part))


class Net(nn.Module):
    def __init__(self,d):
        super(Net,self).__init__()
        self.d = d
        self.fc1 = nn.Linear(self.d,15)
        self.ac1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(15,15)
        self.ac2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(15,15)
        self.ac3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(15,15) #trying less layers
        self.ac4 = nn.LeakyReLU()
        self.fc5 = nn.Linear(15,15)
        self.ac5 = nn.LeakyReLU()
        self.fcLast = nn.Linear(15,8) #2nd dim must match gnn. attempting cast to 8d for no reason
    
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if (device == 'cpu'): 
        quit() #needs to run on gpu
    
    featureBranches = ["pixelU","pixelV","pixelEta","pixelPhi","pixelR","pixelZ","pixelCharge","pixelTrackerLayer"]
    labelBranches = ["pixelSimTrackID", "pixelSimTrackPt"]
    trainDS = OctopiDataset(glob.glob("/eos/user/n/nihaubri/OctopiNtuples/QCDMar7/OctopiNtuples_1.root"),featureBranches=featureBranches,labelBranches=labelBranches,batchsize=20) #batches of 50 jets with ~100 pixels each
    print("training dataset has {} jets. Running {} batches".format(len(trainDS)*trainDS.batchsize,len(trainDS)))

    valDS = OctopiDataset(glob.glob("/eos/user/n/nihaubri/OctopiNtuples/QCDMar7/OctopiNtuples_59.root"),featureBranches=featureBranches,labelBranches=labelBranches,batchsize=500) ###less for test


    mva = Net(d=len(featureBranches)).to(device) ## with xmod set, without should be 6
    opt = torch.optim.Adam(mva.parameters(),lr=0.001)

    scaler = GradScaler()
    scheduler = StepLR(opt, step_size=10, gamma=0.5)

    epochLosses = []
    epochValLosses = []
    for epoch in range(20): #was 20
        epochLoss = torch.zeros(1,device=device).detach()
        epochValLoss = torch.zeros(1,device=device).detach()

        mva.train()
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
                if i == 0: dummy_input = X #needed for ONNX
                pred = mva(X) #Xmod
                predsplit = torch.tensor_split(pred,tuple(sizeList),dim=0)
                ysplit = torch.tensor_split(Y,tuple(sizeList),dim=0)
                batchLoss = torch.zeros(1,device=device)
                for j,(jetPred,jetY) in enumerate(zip(predsplit,ysplit)): #vectorize this somehow?
                    if jetY.shape[0]==1: #needed for jan26 ntuples but not later
                        continue
                    batchLoss+=PairwiseHingeLoss(jetPred,jetY, torch.tensor(lsize)) #a=weight of hinge over MSE, trying 10X loss for hinge.

                if i%50==epoch:
                    print("batch {} loss: {:.5f}".format(i,float(batchLoss.detach())))
                epochLoss+=batchLoss.detach()

            
            #batchLoss.backward()
            #opt.step()
            scaler.scale(batchLoss).backward()
            scaler.step(opt)
            scaler.update()
        
        #normalize loss before append
        epochLosses.append(float(epochLoss)/float(trainDS.count))
        scheduler.step()
        
        mva.eval()
        for i,(X,Y,sizeList) in enumerate(valDS):
            if i>len(valDS):
                i=0
                break

            X=X.to(device)
            Y=Y[0].to(device)

            pred = mva(X) 

            predsplit = torch.tensor_split(pred,tuple(sizeList),dim=0)
            ysplit = torch.tensor_split(Y,tuple(sizeList),dim=0)
            batchLoss = torch.zeros(1, device=device)
            for (jetPred,jetY) in zip(predsplit,ysplit): # could use this w/ clustering, or run on less data
                if jetY.shape[0]==1: #needed for jan26 ntuples but not later
                    continue
                epochValLoss+=PairwiseHingeLoss(jetPred,jetY, torch.tensor(lsize)).detach()
        epochValLosses.append(float(epochValLoss)/float(valDS.count))
        print("Epoch time: {:.2f} Training Loss: {:.2f} Validation Loss: {:.2f}".format(time.time()-epochStart,epochLosses[-1],epochValLosses[-1]))


    plt.plot(epochLosses,label='training')
    plt.plot(epochValLosses,label='validation')
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig(f"plots/loss_{lsize}.png")

    print("Saved loss.png")

    #save model for later use
    torch.save(mva, f"models/trained_mlp_{lsize}.pth")

    print("Saved model successfully")

    ###ONNX save model###
    torch.onnx.export(mva, dummy_input, "mlp.onnx", verbose=True, input_names=['my_input'], output_names=['my_output'])
    print("Saved to ONNX")
    ######

if __name__ == "__main__":
    main()
