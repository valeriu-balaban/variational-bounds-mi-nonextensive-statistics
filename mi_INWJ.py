import torch
import math
import torch.distributions as tdis
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, RandomSampler, BatchSampler, DataLoader

def InfoNWJ(X, Y, batch_size=1024, num_epochs=150, dev=torch.device("cpu"), model=None):
    A = torch.tensor(batch_size).float().log()
    
    if not model:
        model = nn.Sequential(
            nn.Linear(X.shape[1]+Y.shape[1], 36),         
            nn.ReLU(),
            nn.Linear(36, 18),            
            nn.ReLU(),
            nn.Linear(18, 1),
        )
        
    X       = X.to(dev)
    Y       = Y.to(dev) + torch.randn_like(Y) * 1e-4
    model = model.to(dev)
    
    opt     = optim.Adam(model.parameters(), lr=0.0015)
    td       = TensorDataset(X, Y)
    
    result  = []  
    
    for epoch in range(num_epochs):            
        for x, y in DataLoader(td, batch_size, shuffle=True, drop_last=True):            
            opt.zero_grad()
            
            x1     = x[:batch_size//2,:] 
            y1     = y[:batch_size//2,:] 
            x2    = x[batch_size//2:,:]
            y2    = y[batch_size//2:,:]
            EXY  = model(torch.cat([x1, y1], 1)).flatten().mean()
            EY    = model(torch.cat([x1, y2], 1)).flatten().exp().mean() / math.e
            
            loss   = -(EXY-EY)
            
            result.append(-loss.item())
          
            loss.backward(retain_graph=True)
            opt.step()
    
    r = torch.mean(torch.tensor(result[-80:]))
    plt.plot(result,label="Inwj")
    plt.title('Inwj')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Mutual Infomation')
    plt.legend(loc='lower right')

    print(r)              
    return r