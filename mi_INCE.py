import torch
import math
import torch.distributions as tdis
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, RandomSampler, BatchSampler, DataLoader

def InfoNCE(X, Y, batch_size=512, num_epochs=50, dev=torch.device("cpu"), model=None):
    A = torch.tensor(batch_size).float().log()
    
    if not model:
        model = nn.Sequential(
            nn.Linear(X.shape[1]+Y.shape[1], 16),         
            nn.ReLU(),
            nn.Linear(16, 8),            
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        
    # Move data to device
    X       = X.to(dev)
    Y       = Y.to(dev) + torch.randn_like(Y) * 1e-4
    model = model.to(dev)
    
    opt   = optim.Adam(model.parameters(), lr=0.01)
    td     = TensorDataset(X, Y)
    
    result = []  
    
    for epoch in range(num_epochs):            
        for x, y in DataLoader(td, batch_size, shuffle=True, drop_last=True):            
            opt.zero_grad()
            
            top      = model(torch.cat([x, y], 1)).flatten()
            xiyj      = torch.cat([x.repeat_interleave(batch_size,dim=0),y.repeat(batch_size,1)], 1)    
            bottom = torch.logsumexp(model(xiyj).reshape(batch_size,batch_size), 1) - A
            
            loss     = -(top - bottom).mean()
            
            result.append(-loss.item())

            loss.backward(retain_graph=True)
            opt.step()
    
    r = torch.mean(torch.tensor(result[-50:]))
    plt.plot(result,label="Ince")
    plt.title('Ince')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Mutual Infomation')
    plt.legend(loc='lower right')

    print(r)              
    return r