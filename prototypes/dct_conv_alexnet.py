#!/usr/bin/env python
# coding: utf-8

if __name__ == '__main__':
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
    from scipy.fftpack import dct
    import math
    from DCT_layers import ConvDCT
    ## cifar10 download problem solve
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context



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


    device = "cuda" if torch.cuda.is_available() else "cpu"



    class AlexNet(nn.Module):

        def __init__(self, num_classes: int = 10) -> None:
            super(AlexNet, self).__init__()
            self.features = nn.Sequential(
    #             nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                # DCT_conv_layer(3, 64, kernel_size=3, stride=2, padding=1),
                ConvDCT(in_channels=3,out_channels=64,kernel_size=3,stride=2,padding=1),
    #             nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),   
    #             nn.Conv2d(64, 192, kernel_size=3, padding=1),
                # DCT_conv_layer(64, 192, kernel_size=3, padding=1),
                ConvDCT(in_channels=64,out_channels=192,kernel_size=3,padding=1),
    #             nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),    
    #             nn.Conv2d(192, 384, kernel_size=3, padding=1),
                # DCT_conv_layer(192, 384, kernel_size=3, padding=1),
                ConvDCT(in_channels=192,out_channels=384,kernel_size=3,padding=1),
    #             nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
    #             nn.Conv2d(384, 256, kernel_size=3, padding=1),
                # DCT_conv_layer(384, 256, kernel_size=3, padding=1),
                ConvDCT(in_channels=384,out_channels=256,kernel_size=3,padding=1),
    #             nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
    #             nn.Conv2d(256, 256, kernel_size=3, padding=1),
                # DCT_conv_layer(256, 256, kernel_size=3, padding=1),
                ConvDCT(in_channels=256,out_channels=256,kernel_size=3,padding=1),
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


    def train(dataloader,model,criterion,optimizer,file):
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
        file.write(f'Training Loss: {train_loss:.8f} \n')

    def test(dataloader, model, loss_fn,file):
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
        file.write(f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n \n")


    import torch.optim as optim


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    file = open('DCT_conv_alexnet_py_output.txt','w')

    print("DCT_conv_AlexNet..........new dct_conv")
    file.write("DCT_conv_AlexNet..........new dct_conv \n \n")
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        file.write(f"Epoch {t+1}\n-------------------------------\n")
        train(trainloader, net, criterion, optimizer,file)
        test(testloader, net, criterion,file)


    from prettytable import PrettyTable


    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        # print(table)
        # print(f"Total Trainable Params: {total_params}")
        return table,total_params

    tab, tot_params = count_parameters(net)
    
    file.write("\n \n" + tab.get_string() + "\n \n" )
    file.write('\ntotal no of trainable params: ' + str(tot_params) )

    print('total trainable params : ', tot_params )
    print(tab)
    file.close()
