import imp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, AveragePrecision
from models.TransformLayers.DFT_layers import LinearDFT, Conv2dDFT
# from models.TransformLayers.conv2d_dft import Conv2dDFT
from utils.complex import Cardioid, ComplexMaxPool2d, complex_abs


class LeNetLinearDFT(pl.LightningModule):
        def __init__(self, num_classes:int =10, in_channels: int = 1):
            super(LeNetLinearDFT, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 6, 5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(6, 16, 5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(16, 120, 5),
                nn.ReLU(inplace=True),

            )
            self.classifier = nn.Sequential(
                LinearDFT(120, 84),
                Cardioid(),
                LinearDFT(84, num_classes),
            )
            
            self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
            self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

            # self.val_ap = AveragePrecision(num_classes=num_classes)
            # self.test_ap = AveragePrecision(num_classes=num_classes)


        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
            x = self.classifier(x)
            return complex_abs(x) ##  magnitude

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
            return optimizer

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            self.log('train loss', loss, on_step=False, on_epoch=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            val_loss = F.cross_entropy(y_hat, y)
            
            preds = torch.argmax(y_hat, dim=1)
            self.val_accuracy.update(preds, y)
            # self.val_ap.update(y_hat, y)

            self.log("val_loss", val_loss, prog_bar=True)
            self.log("val_acc", self.val_accuracy, prog_bar=True)
            # self.log('val_AP', self.val_ap,prog_bar=True)
            
            # return val_loss, self.val_accuracy
             
        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            test_loss = F.cross_entropy(y_hat, y)
            
            preds = torch.argmax(y_hat, dim=1)
            self.test_accuracy.update(preds, y)
            # self.test_ap.update(y_hat, y)

            self.log("test_loss", test_loss, prog_bar=True)
            self.log("test_acc", self.test_accuracy, prog_bar=True)
            # self.log('test_AP', self.test_ap,prog_bar=True)

            # return test_loss, self.test_accuracy

        def predict_step(self, batch, batch_idx):
            x, y = batch
            pred = self(x)
            return pred


class LeNetDFT(pl.LightningModule):
        def __init__(self, num_classes:int =10,in_channels: int = 1):
            super(LeNetDFT, self).__init__()
            self.features = nn.Sequential(
                Conv2dDFT(in_channels, 6, 5),
                Cardioid(),
                ComplexMaxPool2d(kernel_size=2),
                Conv2dDFT(6, 16, 5),
                Cardioid(),
                ComplexMaxPool2d(kernel_size=2),
                Conv2dDFT(16, 120, 5),
                Cardioid(),

            )
            self.classifier = nn.Sequential(
                LinearDFT(120, 84),
                Cardioid(),
                LinearDFT(84, num_classes),
            )
            
            self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
            self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

            # self.val_ap = AveragePrecision(num_classes=num_classes)
            # self.test_ap = AveragePrecision(num_classes=num_classes)


        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
            x = self.classifier(x)
            return complex_abs(x)    

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
            return optimizer

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            self.log('train loss', loss, on_step=False, on_epoch=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            val_loss = F.cross_entropy(y_hat, y)
            
            preds = torch.argmax(y_hat, dim=1)
            self.val_accuracy.update(preds, y)
            # self.val_ap.update(y_hat, y)

            self.log("val_loss", val_loss, prog_bar=True)
            self.log("val_acc", self.val_accuracy, prog_bar=True)
            # self.log('val_AP', self.val_ap,prog_bar=True)
            
            # return val_loss, self.val_accuracy
             
        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            test_loss = F.cross_entropy(y_hat, y)
            
            preds = torch.argmax(y_hat, dim=1)
            self.test_accuracy.update(preds, y)
            # self.test_ap.update(y_hat, y)

            self.log("test_loss", test_loss, prog_bar=True)
            self.log("test_acc", self.test_accuracy, prog_bar=True)
            # self.log('test_AP', self.test_ap,prog_bar=True)

            # return test_loss, self.test_accuracy

        def predict_step(self, batch, batch_idx):
            x, y = batch
            pred = self(x)
            return pred

class LeNetConvDFT(pl.LightningModule):
        def __init__(self, num_classes:int =10, in_channels: int = 1):
            super(LeNetConvDFT, self).__init__()
            self.features = nn.Sequential(
                Conv2dDFT(in_channels, 6, 5),
                Cardioid(),
                ComplexMaxPool2d(kernel_size=2),
                Conv2dDFT(6, 16, 5),
                Cardioid(),
                ComplexMaxPool2d(kernel_size=2),
                Conv2dDFT(16, 120, 5),
                Cardioid(),

            )
            self.classifier = nn.Sequential(
                nn.Linear(120, 84),
                nn.ReLU(inplace=True),
                nn.Linear(84, num_classes),
            )
            
            self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
            self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

            # self.val_ap = AveragePrecision(num_classes=num_classes)
            # self.test_ap = AveragePrecision(num_classes=num_classes)


        def forward(self, x):
            x = self.features(x)
            x = complex_abs(x + 1e-6) 
            x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
            x = self.classifier(x)
            return x   

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
            return optimizer

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            if torch.isnan(loss) == True:
                loss = torch.nan_to_num(loss)
            self.log('train loss', loss, on_step=False, on_epoch=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            val_loss = F.cross_entropy(y_hat, y)

            if torch.isnan(val_loss) == True:
                val_loss = torch.nan_to_num(val_loss)
            
            preds = torch.argmax(y_hat, dim=1)
            self.val_accuracy.update(preds, y)
            # self.val_ap.update(y_hat, y)

            self.log("val_loss", val_loss, prog_bar=True)
            self.log("val_acc", self.val_accuracy, prog_bar=True)
            # self.log('val_AP', self.val_ap,prog_bar=True)
            
            # return val_loss, self.val_accuracy
             
        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            test_loss = F.cross_entropy(y_hat, y)

            if torch.isnan(test_loss) == True:
                test_loss = torch.nan_to_num(test_loss)
            
            preds = torch.argmax(y_hat, dim=1)
            self.test_accuracy.update(preds, y)
            # self.test_ap.update(y_hat, y)

            self.log("test_loss", test_loss, prog_bar=True)
            self.log("test_acc", self.test_accuracy, prog_bar=True)
            # self.log('test_AP', self.test_ap,prog_bar=True)

            # return test_loss, self.test_accuracy
        def predict_step(self, batch, batch_idx):
            x, y = batch
            pred = self(x)
            return pred