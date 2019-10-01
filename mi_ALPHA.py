import torch
import math
import torch.distributions as tdis
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, RandomSampler, BatchSampler, DataLoader

def InfoALPHA(X, Y, alpha=0.01 , batch_size=512, num_epochs=80, dev=torch.device("cuda"), model=None):
    # Move data to device
    
    X   =  X.to(dev)
    Y   =  Y.to(dev) + torch.randn_like(Y) * 1e-4
    Yq  =  Y[torch.randperm(Y.shape[0])]
    
    if not model:
        model1 = nn.Sequential(
             nn.Linear(X.shape[1]+Y.shape[1], 36),
             #nn.Dropout(p=0.1),
             nn.ReLU(),
             nn.Linear(36, 18),
             #nn.Dropout(p=0.1),
             nn.ReLU(),
             nn.Linear(18, 1),
        )
        model2 = nn.Sequential(
             nn.Linear(Yq.shape[1], 36),
             #nn.Dropout(p=0.1),
             nn.ReLU(),
             nn.Linear(36, 18),
            # nn.Dropout(p=0.1),
             nn.ReLU(),
             nn.Linear(18, 1),
        )
        
    model1 = model1.to(dev)
    model2 = model2.to(dev)
    
    opt1   = optim.Adam(model1.parameters(), lr=0.01)
    opt2   = optim.Adam(model2.parameters(), lr=0.01)
    
    td     = TensorDataset(X, Y, Yq)
    
    result = []
    
    for epoch in range(num_epochs):            
        for x, y, yq in DataLoader(td, batch_size, shuffle=True, drop_last=True):            
            opt1.zero_grad()
            opt2.zero_grad()
            
            top1    = model1(torch.cat([x, y], 1)).exp()
            top2    = model1(torch.cat([x, yq], 1)).exp()
            
            xiyj1   = torch.cat([x.repeat_interleave(batch_size,dim=0),y.repeat(batch_size,1)], 1) 
            xiyj2   = torch.cat([x.repeat_interleave(batch_size,dim=0),yq.repeat(batch_size,1)], 1) 
            
            bottom1 = (alpha * model1(xiyj1).exp().reshape(batch_size,batch_size).mean(dim=1) + (1 - alpha) * model2(y)).mean()
            bottom2 = (alpha * model1(xiyj2).exp().reshape(batch_size,batch_size).mean(dim=1) + (1 - alpha) * model2(yq)).mean()

            mean1   = (top1 / bottom1).log().mean()
            mean2   = (top2 / bottom2).mean()
            
            loss    = -(1 + mean1 - mean2)
            result.append(-loss.item())

            loss.backward(retain_graph=True)
            opt1.step()
            opt2.step()
            
    
    r = torch.mean(torch.tensor(result[-50:]))
    
    plt.plot(result,label="I_α")
    plt.title('I_α')
    plt.xlabel('Number of Epochs')
    plt.legend(loc='upper left')
    #plt.savefig("I_α fails at High Dimension")
    print(r)              
    return r
    