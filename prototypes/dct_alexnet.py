#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from scipy.fftpack import dct
import math
## cifar10 download problem solve
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# In[2]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[3]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# In[4]:


class DCT_layer(nn.Module):
    def __init__(self,out_features: int):
        super(DCT_layer, self).__init__()
        
        self.out_features = out_features
        
        default_dtype = torch.get_default_dtype()
        self.fc = nn.Parameter(torch.arange((self.out_features), dtype=default_dtype).reshape(-1,1))      

    def dct_kernel(self,t): 
        return torch.cos(0.5 * np.pi * self.fc * (2 * t + 1) / self.out_features)
    
        
    def forward(self,x):
#         print(x.shape)
        t = torch.arange(x.shape[-1]).reshape(1,-1).to(device)
        w = nn.Parameter(self.dct_kernel(t)) 
        
        y = F.linear(x,w)   
        return y


# In[5]:


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 10) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
#             nn.Linear(256 * 2 * 2, 4096),
            DCT_layer(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            DCT_layer(4096),
#             nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
#             DCT_layer(num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x
    
net = AlexNet(num_classes=10).to(device)
print(net)


# In[6]:


def train(dataloader,model,criterion,optimizer):

    train_loss = 0.0
    for X, y in dataloader:
        inputs, labels = X.to(device), y.to(device)
        
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
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# In[8]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("DCT_AlexNet")
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(trainloader, net, criterion, optimizer)
    test(testloader, net, criterion)


# In[ ]:





# In[ ]:




