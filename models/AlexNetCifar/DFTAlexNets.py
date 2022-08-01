import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.TransformLayers.DFT_layers import LinearDFT, Conv2dDFT
# from models.TransformLayers.conv2d_dft import Conv2dDFT
import math
import pytorch_lightning as pl
from torchmetrics import Accuracy
from utils.complex import Cardioid, ComplexMaxPool2d, complex_abs

class AlexNetLinearDFT(pl.LightningModule):

        def __init__(self, num_classes: int = 10,  in_channels: int = 3) -> None:
            super(AlexNetLinearDFT, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
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
                LinearDFT(256 * 2 * 2, 4096),
                Cardioid(),
                nn.Dropout(),
                LinearDFT(4096, 4096),
                Cardioid(),
                LinearDFT(4096, num_classes),
            )

            self.val_accuracy = Accuracy()
            self.test_accuracy = Accuracy()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = x.view(x.size(0), 256 * 2 * 2)
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

            self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
            self.log("val_acc", self.val_accuracy, prog_bar=True, on_epoch=True)
            
            # return val_loss, self.val_accuracy
             
        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            test_loss = F.cross_entropy(y_hat, y)
            
            preds = torch.argmax(y_hat, dim=1)
            self.test_accuracy.update(preds, y)

            self.log("test_loss", test_loss, prog_bar=True)
            self.log("test_acc", self.test_accuracy, prog_bar=True)

            # return test_loss, self.test_accuracy

        def predict_step(self, batch, batch_idx):
            x, y = batch
            pred = self(x)
            return pred



class AlexNetDFT(pl.LightningModule):

        def __init__(self, num_classes: int = 10,  in_channels: int = 3) -> None:
            super(AlexNetDFT, self).__init__()
            self.features = nn.Sequential(
                Conv2dDFT(in_channels, 64, kernel_size=3, stride=2, padding=1),
                Cardioid(),
                ComplexMaxPool2d(kernel_size=2),   
                Conv2dDFT(64, 192, kernel_size=3, padding=1),
                Cardioid(),
                ComplexMaxPool2d(kernel_size=2),    
                Conv2dDFT(192, 384, kernel_size=3, padding=1),
                Cardioid(),
                Conv2dDFT(384, 256, kernel_size=3, padding=1),
                Cardioid(),
                Conv2dDFT(256, 256, kernel_size=3, padding=1),
                Cardioid(),
                ComplexMaxPool2d(kernel_size=2),

            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                LinearDFT(256 * 2 * 2, 4096),
                Cardioid(),
                nn.Dropout(),
                LinearDFT(4096, 4096),
                Cardioid(),
                LinearDFT(4096, num_classes),
            )

            self.val_accuracy = Accuracy()
            self.test_accuracy = Accuracy()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = x.view(x.size(0), 256 * 2 * 2, 2)
            x = self.classifier(x)
            return complex_abs(x)  ## sum in the last dim to get correct shape output for loss function


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

            self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
            self.log("val_acc", self.val_accuracy, prog_bar=True, on_epoch=True)
            
            # return val_loss, self.val_accuracy
             
        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            test_loss = F.cross_entropy(y_hat, y)
            
            preds = torch.argmax(y_hat, dim=1)
            self.test_accuracy.update(preds, y)

            self.log("test_loss", test_loss, prog_bar=True)
            self.log("test_acc", self.test_accuracy, prog_bar=True)

            # return test_loss, self.test_accuracy

        def predict_step(self, batch, batch_idx):
            x, y = batch
            pred = self(x)
            return pred


class AlexNetConvDFT(pl.LightningModule):

        def __init__(self, num_classes: int = 10,  in_channels: int = 3) -> None:
            super(AlexNetConvDFT, self).__init__()
            self.features = nn.Sequential(
                Conv2dDFT(in_channels, 64, kernel_size=3, stride=2, padding=1),
                Cardioid(),
                ComplexMaxPool2d(kernel_size=2),   
                Conv2dDFT(64, 192, kernel_size=3, padding=1),
                Cardioid(),
                ComplexMaxPool2d(kernel_size=2),    
                Conv2dDFT(192, 384, kernel_size=3, padding=1),
                Cardioid(),
                Conv2dDFT(384, 256, kernel_size=3, padding=1),
                Cardioid(),
                Conv2dDFT(256, 256, kernel_size=3, padding=1),
                Cardioid(),
                ComplexMaxPool2d(kernel_size=2),

            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 2 * 2, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

            self.val_accuracy = Accuracy()
            self.test_accuracy = Accuracy()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = complex_abs(x)
            x = x.view(x.size(0), 256 * 2 * 2)
            x = self.classifier(x)
            return x  


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

            self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
            self.log("val_acc", self.val_accuracy, prog_bar=True, on_epoch=True)
            
            # return val_loss, self.val_accuracy
             
        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            test_loss = F.cross_entropy(y_hat, y)
            
            preds = torch.argmax(y_hat, dim=1)
            self.test_accuracy.update(preds, y)

            self.log("test_loss", test_loss, prog_bar=True)
            self.log("test_acc", self.test_accuracy, prog_bar=True)

            # return test_loss, self.test_accuracy

        def predict_step(self, batch, batch_idx):
            x, y = batch
            pred = self(x)
            return pred