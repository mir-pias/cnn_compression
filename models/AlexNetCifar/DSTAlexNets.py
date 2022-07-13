import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.TransformLayers.DST_layers import LinearDST
from models.TransformLayers.conv2d_dst import Conv2dDST
import pytorch_lightning as pl
from torchmetrics import Accuracy

class AlexNetLinearDST(pl.LightningModule):

        def __init__(self, num_classes: int = 10) -> None:
            super(AlexNetLinearDST, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
    #             DCT_conv_layer(3, 64, kernel_size=3, stride=2, padding=1),
    #             nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),   
                nn.Conv2d(64, 192, kernel_size=3, padding=1),
    #             DCT_conv_layer(64, 192, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),    
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
    #             DCT_conv_layer(192, 384, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
    #             DCT_conv_layer(384, 256, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
    #             DCT_conv_layer(256, 256, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),

            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
    #             nn.Linear(256 * 2 * 2, 4096),
                LinearDST(256 * 2 * 2, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                LinearDST(4096, 4096),
    #             nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                LinearDST(4096, num_classes),
    #          
            )

            self.val_accuracy = Accuracy()
            self.test_accuracy = Accuracy()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
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

            self.log("test_loss", test_loss, prog_bar=True, on_epoch=True)
            self.log("test_acc", self.test_accuracy, prog_bar=True, on_epoch=True)

            # return test_loss, self.test_accuracy

        def predict_step(self, batch, batch_idx):
            x, y = batch
            pred = self(x)
            return pred


class AlexNetConvDST(pl.LightningModule):

        def __init__(self, num_classes: int = 10) -> None:
            super(AlexNetConvDST, self).__init__()
            self.features = nn.Sequential(
    #             nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                # DCT_conv_layer(3, 64, kernel_size=3, stride=2, padding=1),
                Conv2dDST(in_channels=3,out_channels=64,kernel_size=3,stride=2,padding=1),
    #             nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),   
    #             nn.Conv2d(64, 192, kernel_size=3, padding=1),
                # DCT_conv_layer(64, 192, kernel_size=3, padding=1),
                Conv2dDST(in_channels=64,out_channels=192,kernel_size=3,padding=1),
    #             nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),    
    #             nn.Conv2d(192, 384, kernel_size=3, padding=1),
                # DCT_conv_layer(192, 384, kernel_size=3, padding=1),
                Conv2dDST(in_channels=192,out_channels=384,kernel_size=3,padding=1),
    #             nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
    #             nn.Conv2d(384, 256, kernel_size=3, padding=1),
                # DCT_conv_layer(384, 256, kernel_size=3, padding=1),
                Conv2dDST(in_channels=384,out_channels=256,kernel_size=3,padding=1),
    #             nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
    #             nn.Conv2d(256, 256, kernel_size=3, padding=1),
                # DCT_conv_layer(256, 256, kernel_size=3, padding=1),
                Conv2dDST(in_channels=256,out_channels=256,kernel_size=3,padding=1),
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

            self.val_accuracy = Accuracy()
            self.test_accuracy = Accuracy()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
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

            self.log("test_loss", test_loss, prog_bar=True, on_epoch=True)
            self.log("test_acc", self.test_accuracy, prog_bar=True, on_epoch=True)

            # return test_loss, self.test_accuracy

        def predict_step(self, batch, batch_idx):
            x, y = batch
            pred = self(x)
            return pred


class AlexNetDST(pl.LightningModule):

        def __init__(self, num_classes: int = 10) -> None:
            super(AlexNetDST, self).__init__()
            self.features = nn.Sequential(
    #             nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                Conv2dDST(3, 64, kernel_size=3, stride=2, padding=1),
    #             nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),   
    #             nn.Conv2d(64, 192, kernel_size=3, padding=1),
                Conv2dDST(64, 192, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),    
    #             nn.Conv2d(192, 384, kernel_size=3, padding=1),
                Conv2dDST(192, 384, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
    #             nn.Conv2d(384, 256, kernel_size=3, padding=1),
                Conv2dDST(384, 256, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
    #             nn.Conv2d(256, 256, kernel_size=3, padding=1),
                Conv2dDST(256, 256, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),

            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
    #             nn.Linear(256 * 2 * 2, 4096),
                LinearDST(256 * 2 * 2,4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                LinearDST(4096,4096),
    #             nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                LinearDST(4096, num_classes),
    #             DCT_layer(num_classes)
            )

            self.val_accuracy = Accuracy()
            self.test_accuracy = Accuracy()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
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

            self.log("test_loss", test_loss, prog_bar=True, on_epoch=True)
            self.log("test_acc", self.test_accuracy, prog_bar=True, on_epoch=True)

            # return test_loss, self.test_accuracy

        def predict_step(self, batch, batch_idx):
            x, y = batch
            pred = self(x)
            return pred