import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torchvision.models as models
from utils.replace_layers import replace_conv2d, replace_linear

class DenseNet121LinearDCT(pl.LightningModule):
        def __init__(self, num_classes: int=10):
            super(DenseNet121LinearDCT, self).__init__()
            
            model = models.densenet121(num_classes=num_classes)
            replace_linear(model,'model', kernel='DCT')

            self.features = model.features
            self.classifier = model.classifier

            self.val_accuracy = Accuracy()
            self.test_accuracy = Accuracy()

        def forward(self, x):
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
            # self.log('train loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            val_loss = F.cross_entropy(y_hat, y)
            
            preds = torch.argmax(y_hat, dim=1)
            self.val_accuracy.update(preds, y)

            self.log("val_loss", val_loss, prog_bar=True)
            self.log("val_acc", self.val_accuracy, prog_bar=True)
            
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

class DenseNet121ConvDCT(pl.LightningModule):
        def __init__(self, num_classes: int=10):
            super(DenseNet121ConvDCT, self).__init__()
            
            model = models.densenet121(num_classes=num_classes)
            replace_conv2d(model,'model', kernel='DCT')

            self.features = model.features
            self.classifier = model.classifier

            self.val_accuracy = Accuracy()
            self.test_accuracy = Accuracy()

        def forward(self, x):
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
            # self.log('train loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            val_loss = F.cross_entropy(y_hat, y)
            
            preds = torch.argmax(y_hat, dim=1)
            self.val_accuracy.update(preds, y)

            self.log("val_loss", val_loss, prog_bar=True)
            self.log("val_acc", self.val_accuracy, prog_bar=True)
            
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

class DenseNet121DCT(pl.LightningModule):
        def __init__(self, num_classes: int=10):
            super(DenseNet121LinearDCT, self).__init__()
            
            model = models.densenet121(num_classes=num_classes)
            replace_linear(model,'model', kernel='DCT')
            replace_conv2d(model,'model', kernel='DCT')

            self.features = model.features
            self.classifier = model.classifier

            self.val_accuracy = Accuracy()
            self.test_accuracy = Accuracy()

        def forward(self, x):
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
            # self.log('train loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            val_loss = F.cross_entropy(y_hat, y)
            
            preds = torch.argmax(y_hat, dim=1)
            self.val_accuracy.update(preds, y)

            self.log("val_loss", val_loss, prog_bar=True)
            self.log("val_acc", self.val_accuracy, prog_bar=True)
            
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