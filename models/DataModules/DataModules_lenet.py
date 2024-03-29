import sys
sys.path.append('.')

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN
## cifar download problem solve
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class Cifar10DataModuleLenet(pl.LightningDataModule):
    def __init__(self, data_dir: str = "/ds/images/CIFAR", batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=False)
        CIFAR10(self.data_dir, train=False, download=False)

    def setup(self, stage= None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar10_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            
            val_size = int(len(cifar10_full)*0.1)
            train_size = int(len(cifar10_full) - val_size)

            self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.cifar10_predict = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.cifar10_predict, batch_size=self.batch_size)


class Cifar100DataModuleLenet(pl.LightningDataModule):
    def __init__(self, data_dir: str = "/ds/images/CIFAR", batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        CIFAR100(self.data_dir, train=True, download=False)
        CIFAR100(self.data_dir, train=False, download=False)

    def setup(self, stage= None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar100_full = CIFAR100(self.data_dir, train=True, transform=self.transform)
            
            val_size = int(len(cifar100_full)*0.1)
            train_size = int(len(cifar100_full) - val_size)

            self.cifar100_train, self.cifar100_val = random_split(cifar100_full, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar100_test = CIFAR100(self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.cifar100_predict = CIFAR100(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar100_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar100_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar100_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.cifar100_predict, batch_size=self.batch_size)


class MNISTDataModuleLenet(pl.LightningDataModule):
    def __init__(self, data_dir: str = "/ds/images", batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((32, 32)),transforms.Normalize((0.1307,), (0.3081,))])

        self.batch_size = batch_size

    def prepare_data(self):
        # print(self.data_dir)
        # download
        MNIST(self.data_dir, train=True, download=False)
        MNIST(self.data_dir, train=False, download=False)

    def setup(self, stage= None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            
            val_size = int(len(mnist_full)*0.1)
            train_size = int(len(mnist_full) - val_size)

            self.mnist_train, self.mnist_val = random_split(mnist_full, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)



class SVHNDataModuleLenet(pl.LightningDataModule):
    def __init__(self, data_dir: str = "/ds/images/SVHN/SVHN", batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        SVHN(self.data_dir, split = 'train', download=False)
        SVHN(self.data_dir, split = 'test', download=False)

    def setup(self, stage= None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            SVHN_full = SVHN(self.data_dir, split = 'train', transform=self.transform)
            
            val_size = int(len(SVHN_full)*0.1)
            train_size = int(len(SVHN_full) - val_size)

            self.SVHN_train, self.SVHN_val = random_split(SVHN_full, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.SVHN_test = SVHN(self.data_dir, split = 'test', transform=self.transform)

        if stage == "predict" or stage is None:
            self.SVHN_predict = SVHN(self.data_dir, split = 'test', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.SVHN_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.SVHN_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.SVHN_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.SVHN_predict, batch_size=self.batch_size)
