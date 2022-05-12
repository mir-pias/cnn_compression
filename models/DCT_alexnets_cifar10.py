import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.DCT_layers import DCT_conv_layer, LinearDCT, Conv2dDCT, DCT_linear_layer
import math

class DCT_fc_AlexNet(nn.Module):

        def __init__(self, num_classes: int = 10) -> None:
            super(DCT_fc_AlexNet, self).__init__()
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
                LinearDCT(256 * 2 * 2, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                LinearDCT(4096, 4096),
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
        

class DCT_conv_AlexNet(nn.Module):

        def __init__(self, num_classes: int = 10) -> None:
            super(DCT_conv_AlexNet, self).__init__()
            self.features = nn.Sequential(
    #             nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                # DCT_conv_layer(3, 64, kernel_size=3, stride=2, padding=1),
                Conv2dDCT(in_channels=3,out_channels=64,kernel_size=3,stride=2,padding=1),
    #             nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),   
    #             nn.Conv2d(64, 192, kernel_size=3, padding=1),
                # DCT_conv_layer(64, 192, kernel_size=3, padding=1),
                Conv2dDCT(in_channels=64,out_channels=192,kernel_size=3,padding=1),
    #             nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),    
    #             nn.Conv2d(192, 384, kernel_size=3, padding=1),
                # DCT_conv_layer(192, 384, kernel_size=3, padding=1),
                Conv2dDCT(in_channels=192,out_channels=384,kernel_size=3,padding=1),
    #             nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
    #             nn.Conv2d(384, 256, kernel_size=3, padding=1),
                # DCT_conv_layer(384, 256, kernel_size=3, padding=1),
                Conv2dDCT(in_channels=384,out_channels=256,kernel_size=3,padding=1),
    #             nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
    #             nn.Conv2d(256, 256, kernel_size=3, padding=1),
                # DCT_conv_layer(256, 256, kernel_size=3, padding=1),
                Conv2dDCT(in_channels=256,out_channels=256,kernel_size=3,padding=1),
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


class DCT_AlexNet(nn.Module):

        def __init__(self, num_classes: int = 10) -> None:
            super(DCT_AlexNet, self).__init__()
            self.features = nn.Sequential(
    #             nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                Conv2dDCT(3, 64, kernel_size=3, stride=2, padding=1),
    #             nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),   
    #             nn.Conv2d(64, 192, kernel_size=3, padding=1),
                Conv2dDCT(64, 192, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),    
    #             nn.Conv2d(192, 384, kernel_size=3, padding=1),
                Conv2dDCT(192, 384, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
    #             nn.Conv2d(384, 256, kernel_size=3, padding=1),
                Conv2dDCT(384, 256, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
    #             nn.Conv2d(256, 256, kernel_size=3, padding=1),
                Conv2dDCT(256, 256, kernel_size=3, padding=1),
    #             nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),

            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
    #             nn.Linear(256 * 2 * 2, 4096),
                LinearDCT(256 * 2 * 2,4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                LinearDCT(4096,4096),
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