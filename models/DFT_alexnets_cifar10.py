import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.DFT_layers import LinearDFT
import math

class AlexNetLinearDFT(nn.Module):

        def __init__(self, num_classes: int = 10) -> None:
            super(AlexNetLinearDFT, self).__init__()
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
                LinearDFT(256 * 2 * 2, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                LinearDFT(4096, 4096),
    #             nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                LinearDFT(4096, num_classes),
    #             DCT_layer(num_classes)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = x.view(x.size(0), 256 * 2 * 2)
            x = self.classifier(x)
            return x.mean(-1)  ## mean in the last dim to get correct shape output for loss function