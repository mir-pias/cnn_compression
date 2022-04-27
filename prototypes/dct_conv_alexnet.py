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
        
        self.fc.register_hook(lambda grad: grad / (torch.linalg.norm(grad) + 1e-8))

    def dct_kernel(self,t): 
        dct_m = np.sqrt(2/(self.out_features)) * torch.cos(0.5 * np.pi * self.fc * (2 * t + 1) / self.out_features)
        
        dct_m[0] = dct_m[0]/np.sqrt(2)
        
        return dct_m
    
        
    def forward(self,x):
#         print(x.shape)
        t = torch.arange(x.shape[-1]).reshape(1,-1).to(device)
        w = self.dct_kernel(t) 
        
        
        y = F.linear(x,w)   
        return y


# In[5]:


class DCT_conv_layer(nn.Module):
    def __init__(self,in_channels: int,out_channels: int, kernel_size, stride=1,padding=0):
        super(DCT_conv_layer, self).__init__()
        
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        default_dtype = torch.get_default_dtype()
        self.fc = nn.Parameter(torch.arange((self.kernel_size), dtype=default_dtype, device=device).reshape(-1,1))     
        
        self.weight = nn.Parameter(torch.empty((self.out_channels,self.in_channels,self.kernel_size,self.kernel_size), 
                                          dtype=default_dtype, device=device))
        
        self.reset_parameters()
        
        self.weight.register_hook(lambda grad: grad / (torch.linalg.norm(grad) + 1e-8))
        self.fc.register_hook(lambda grad: grad / (torch.linalg.norm(grad) + 1e-8))
        

    def dct_kernel(self,t): 
        dct_m = np.sqrt(2/(self.kernel_size)) * torch.cos(0.5 * np.pi * self.fc * (2 * t + 1) / self.kernel_size)
        
        dct_m[0] = dct_m[0]/np.sqrt(2)
        
        return dct_m.to(device)
    
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5), non)
        k = 1/(self.in_channels * (self.kernel_size)**2)
        nn.init.uniform_(self.weight, a= -math.sqrt(k), b = math.sqrt(k))
        
        
    def forward(self,x):
        
        t = torch.arange(self.kernel_size).reshape(1,-1).to(device)
        dct_m = self.dct_kernel(t) 
        
#         print(self.w.shape)

        w = self.weight @ dct_m   ## dct on uniform weights, weights as conv kernel 

#         print(x.shape)
#         print(w.shape)
#         print((x@w.T).shape)
        
        y = F.conv2d(x,w,stride = self.stride, padding = self.padding)   
        return y


# In[6]:


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 10) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            DCT_conv_layer(3, 64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),   
#             nn.Conv2d(64, 192, kernel_size=3, padding=1),
            DCT_conv_layer(64, 192, kernel_size=3, padding=1),
#             nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),    
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
            DCT_conv_layer(192, 384, kernel_size=3, padding=1),
#             nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
            DCT_conv_layer(384, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
            DCT_conv_layer(256, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
#             DCT_layer(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
#             DCT_layer(4096),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x
    
net = AlexNet(num_classes=10).to(device)
print(net)


# In[7]:


def train(dataloader,model,criterion,optimizer):
#     torch.autograd.detect_anomaly() 
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


# In[8]:


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


# In[9]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

print("DCT_conv_AlexNet")
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(trainloader, net, criterion, optimizer)
    test(testloader, net, criterion)


# In[ ]:





# In[ ]:




