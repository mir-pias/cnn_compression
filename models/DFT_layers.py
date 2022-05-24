import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
# noinspection PyProtectedMember
from torch.nn.modules.utils import _single

from math import pi as PI

from typing import (
    Tuple,
    Union,
    Optional
)

## dft_matrix = weights
class LinearDFT(nn.Module):
    def __init__(self,in_features: int,out_features: int, bias: bool = True ):
        super(LinearDFT, self).__init__()
        
        self.out_features = out_features
        self.in_features = in_features
        
        default_dtype = torch.get_default_dtype()
        self.fc = nn.Parameter(torch.arange((self.out_features), dtype=default_dtype).reshape(-1, 1, 1))     
        
        self.fc.register_hook(lambda grad: grad / (torch.linalg.norm(grad) + 1e-8))

        if bias:
            self.register_parameter(
                name='bias',
                param=torch.nn.Parameter(
                    torch.normal(
                        mean=0.0,
                        std=0.5,
                        size=(self.out_features,2)
                    )
                )
            )
        else:
            self.register_parameter(
                name='bias',
                param=None
            )

    def dft_kernel(self,t,fc): 

        dft_m = torch.cat((torch.cos((fc*t*2*PI)/self.out_features), - torch.sin((fc*t*2*PI)/self.out_features)), dim=-1)
        
        dft_m = (1/math.sqrt(self.out_features)) * dft_m
        
        return dft_m
    
    def materialize_weights(self, x):
        x_is_complex = x.shape[-1] == 2
        in_features = x.shape[-1 - int(x_is_complex)]

        t = torch.linspace(-1.0, 1.0, in_features, dtype=x.dtype, device=x.device).reshape(1, -1, 1)
        fc = self.fc

        weights = self.dft_kernel(t,fc)

        return weights, x_is_complex

    def forward(self,x):
        # print(x.shape)
        weights, x_is_complex = self.materialize_weights(x) 
        
        # print('dft weights: ', weights.shape)  
        if x_is_complex:
            y = torch.stack((
                    F.linear(x[..., 0], weights[..., 0]) - F.linear(x[..., 1], weights[..., 1]),
                    F.linear(x[..., 0], weights[..., 1]) + F.linear(x[..., 1], weights[..., 0])
                ), dim=-1)

        else:
            y = torch.stack((
                        F.linear(x, weights[..., 0]),
                        F.linear(x, weights[..., 1])
                    ), dim=-1)    

        if (self.bias is not None) and (self.bias.numel() == (self.out_features * 2)):
            y = y + self.bias

        return y

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )