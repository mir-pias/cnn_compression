#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from scipy.fft import dct
import math


# In[2]:


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((32, 32)),
     transforms.Normalize((0.5,), (0.5,))])

batch_size = 4

trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='../data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


# In[3]:


class DCT_layer(nn.Module):
    def __init__(self,out_features: int):
        super(DCT_layer, self).__init__()
        
        self.out_features = out_features
        
        default_dtype = torch.get_default_dtype()
        self.fc = nn.Parameter(torch.arange((self.out_features), dtype=default_dtype).reshape(-1,1))      

    def dct_kernel(self,t): 
        return torch.cos(0.5 * np.pi * self.fc * (2 * t + 1) / self.out_features)
    
        
    def forward(self,x):        
        t = torch.arange(x.shape[-1]).reshape(1,-1)
        w = nn.Parameter(self.dct_kernel(t))
    
        y = F.linear(x,w)   
        return y


# In[4]:


class DCT_LeNet(nn.Module):

    def __init__(self):
        super(DCT_LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)  
        
#         self.fc1 = nn.Linear(120, 84)
        self.dct = DCT_layer(84)
        self.fc2 = nn.Linear(84, 10)
#         self.dct2 = DCT_layer(10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = F.relu(self.conv3(x))
        
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
#         x = F.relu(self.dct(x))
#         x = self.fc2(x)
        x = F.relu(self.dct(x))
        x = self.fc2(x)
#         x = self.dct2(x)
        return x


dct_net = DCT_LeNet()
print(dct_net)


# In[5]:


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)  
        
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = F.relu(self.conv3(x))
        
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


lenet = LeNet()
print(lenet)

# from torchviz import make_dot

# batch_size = 32
# in_features = 4096
# out_features = 4096

# x = torch.nn.Parameter(torch.randn(batch_size, in_features))
# dct_layer = DCT_layer(out_features)

# linear_layer = nn.Linear(in_features,out_features)
# y = dct_layer(x)

# make_dot(
#     y,
#     params=dict(dct_layer.named_parameters()),
#     show_saved=True
# ).render('../LinearDCT-dev_alexnet2', format='png')
# In[6]:


def train(dataloader,model,criterion,optimizer):

    train_loss = 0.0
    for X, y in dataloader:
        inputs, labels = X, y
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item()*inputs.size(0)
    train_loss = train_loss/len(dataloader)
    
    print(f'Training Loss: {train_loss:.8f}')


# In[7]:


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X, y
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# In[8]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(dct_net.parameters(), lr=0.001, momentum=0.9)

print("DCT_LeNet")
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(trainloader, dct_net, criterion, optimizer)
    test(testloader, dct_net, criterion)


# In[9]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lenet.parameters(), lr=0.001, momentum=0.9)

print("LeNet")
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(trainloader, lenet, criterion, optimizer)
    test(testloader, lenet, criterion)


# In[ ]:




