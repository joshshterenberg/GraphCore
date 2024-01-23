import uproot
import matplotlib.pyplot as plt
import sys
import pdb
import numpy as np
from itertools import cycle


with uproot.open("GraphCoreNtuples.root") as f:
    tree=f['ntuples/tree']
    #plot one jet core
    n=1
    #coords = tree.arrays(filter_name=["pixelX","pixelY","pixelZ","pixelSimTrackID"]+["caloJetPt","caloJetEta","caloJetPhi"])#,entry_start=n,entry_stop=n+1) 
    coords = tree.arrays()#,entry_start=n,entry_stop=n+1) 
    
    #fig=plt.figure(constrained_layout=True)
    #ax=fig.add_subplot(111,projection='3d')
    
    for i in range(coords['caloJetPt'].to_numpy().shape[0]):

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

        #pdb.set_trace()
        uniqueIDs = set(simIDs)
        nUniqueIDs = len(uniqueIDs)
        
        print("Pixels: {} nSimTracks: {} Unmatched pixels: {} Jet Eta: {:.1f}".format(len(xvals),nUniqueIDs,sum(simIDs==-99),jetEta))
        

        
        
        reply=input("plot? (y/n)")
        #reply="y"
        if reply=="q":
            break
        if reply=="y":
            #let's learn pytorch lol. Regress label from position
            #then try contrastive loss from metric learningggj
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from pytorch_metric_learning import distances, losses, miners, reducers, testers 

            X = torch.from_numpy(np.vstack([xvals,yvals,zvals,etavals,phivals,charges]).T)
            X = X.to(torch.float32)
            #Y = torch.from_numpy(simIDs).reshape(-1,1)
            Y = torch.from_numpy(simIDs)
            Y = Y.to(torch.float32)
            #Y = torch.clip(Y,max=100)

            class Net(nn.Module):
                def __init__(self,d):
                    super(Net,self).__init__()
                    self.d = d
                    self.fc1 = nn.Linear(self.d,50)
                    self.ac1 = nn.LeakyReLU()
                    self.fc2 = nn.Linear(50,50)
                    self.ac2 = nn.LeakyReLU()
                    self.fc3 = nn.Linear(50,50)
                    self.ac3 = nn.LeakyReLU()
                    self.fc4 = nn.Linear(50,3)
                    #self.double()
                
                def forward(self,x):
                    x = self.fc1(x)
                    x = self.ac1(x)
                    x = self.fc2(x)
                    x = self.ac2(x)
                    x = self.fc3(x)
                    x = self.ac3(x)
                    x = self.fc4(x)
                    return x

            mva = Net(d=X.shape[1])
            
            opt = torch.optim.SGD(mva.parameters(),lr=.001,momentum=0.5)
            opt = torch.optim.Adam(mva.parameters(),lr=.001)
            #lossfunc = nn.MSELoss()
            lossfunc = losses.ContrastiveLoss()

            #Y = torch.log(Y+100)
            #Y = torch.rand((Y.shape[0],1))
            
            mva.train()
            #train loop
            #pdb.set_trace()
            for epoch in range(2000):
                opt.zero_grad()
                pred = mva(X)
                loss = lossfunc(pred,Y)
                print("epoch {} loss: {:.5f}".format(epoch,loss))
                loss.backward()
                opt.step()



            colors = np.zeros(len(simIDs))
            for i,uniqueID in enumerate(uniqueIDs):
                colors[simIDs==uniqueID]=i/(nUniqueIDs+1.0)
            

            fig=plt.figure(constrained_layout=True)
            ax=fig.add_subplot(121,projection='3d')
            
            #fig.tight_layout()

            ax.scatter(xvals,yvals,zvals,c=colors,cmap="hsv")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            
            fig.suptitle("Pixels in Jet by SimTrack",fontsize=18,weight='bold')
            ax.set_title("Jet pT: {:.0f} GeV, $\eta$: {:.2f}, $\phi$: {:.2f}\n {} Pixels, {} SimTracks".format(jetPt,jetEta,jetPhi,len(xvals),nUniqueIDs),fontsize=18)

            ax2=fig.add_subplot(122,projection='3d')
            ax2.scatter(pred[:,0].detach().numpy(),pred[:,1].detach().numpy(),pred[:,2].detach().numpy(),c=colors,cmap="hsv")
            ax2.set_title("Embedded Space Representation",fontsize=18)

            ax2.set_xlabel("v1")
            ax2.set_ylabel("v2")
            ax2.set_zlabel("v3")

            plt.show()
