import torch
import math
from time import time
import torch.distributions as tdis
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, RandomSampler, BatchSampler, DataLoader

def get_model(in_size):
    return nn.Sequential(
            nn.Linear(in_size, 36),
            #nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(36, 18),
            #nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(18, 1),
            nn.ReLU()
        )

def InfoNES(X, Y, q=2.4, q1=0.6, batch_size=512, num_epochs=300, dev=torch.device("cpu"), model=None,lrate=0.01):
    A = torch.tensor([0.0001] * batch_size).to(dev)
    B = torch.tensor([0.0001] * (batch_size * batch_size)).to(dev)
    
    if not model:
        model = get_model(X.shape[1]+Y.shape[1])
        
    # Move data to device
    X        = X.to(dev)
    Y        = Y.to(dev)
    Y      += torch.randn_like(Y) * 1e-4
    model = model.to(dev)
    
    opt     = optim.Adam(model.parameters(), lr=lrate)
    td       = TensorDataset(X, Y)
    
    result  = []
    epoch  = 0
    
    ref_time = time()
    
    while epoch < num_epochs:    
        if epoch == 50 and result[-1] < 0.001:            
            # Start from the beginning
            model = get_model(X.shape[1]+Y.shape[1]).to(dev)
            opt     = optim.Adam(model.parameters(), lr=lrate)
            epoch = 0
            print("Did not converge in 50 epochs")
            
        if epoch % 200 == 0 and epoch > 0:
            print("MI at", epoch, "-", result[-1], "elapsed", time() - ref_time, "seconds")
            ref_time = time()
            
        for x, y in DataLoader(td, batch_size, shuffle=True, drop_last=True):            
            opt.zero_grad()

            fxy         = model(torch.cat([x, y], 1)).flatten()
            topin      = torch.max(A,(1 + (1-q) * fxy))
            top         = torch.pow(topin, (1 / (1-q)))
            xiyj         = torch.cat([x.repeat_interleave(batch_size,dim=0),y.repeat(batch_size,1)], 1)    
            bottomin = torch.max(B,(1 + (1-q) * model(xiyj)).flatten())
            bottom   = torch.pow(bottomin, (1 / (1-q))).reshape(batch_size,batch_size).mean(dim=1)

            tb      = top/bottom
            loss   = -((torch.pow(tb, (1-q1)) - 1) / (1-q1)).mean()
            
            if math.isnan(loss.item()):
                break

            result.append(-loss.item())
            loss.backward(retain_graph=True)
            opt.step()
            
        epoch += 1
                
    r = torch.tensor(result[-50:]).mean()
    plt.plot(result,label="q-exp=2.4,q-log=0.6")
    plt.title('Qabe')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Mutual Infomation')
    plt.legend(loc='lower right')
    
    print(r)
    return r
