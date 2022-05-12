    #!/usr/bin/env python
    # coding: utf-8
import sys  
sys.path.append('.')

if __name__ == '__main__':
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
    from models.alexnet_cifar10 import AlexNet
    import math
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


    def test(dataloader, model, loss_fn, file):
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
        file.write((f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n \n"))


    import torch.optim as optim
    # CUDA_LAUNCH_BLOCKING=1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    file = open('orig_alexnet_py_output.txt','w')


    print("AlexNet")
    file.write("AlexNet \n \n")
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




