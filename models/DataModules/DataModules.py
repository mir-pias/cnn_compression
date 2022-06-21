import sys
sys.path.append('.')

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10, CIFAR100
## cifar download problem solve
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class Cifar10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "../data", batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage= None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar100_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            
            val_size = int(len(cifar100_full)*0.1)
            train_size = int(len(cifar100_full) - val_size)

            self.cifar100_train, self.cifar100_val = random_split(cifar100_full, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar100_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.cifar100_predict = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar100_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar100_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar100_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.cifar100_predict, batch_size=self.batch_size)


class Cifar100DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "../data", batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)

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