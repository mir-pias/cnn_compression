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
    Optional,
    cast
)

## cwt_matrix = weights
class LinearCWT(nn.Module):
    def __init__(self,in_features: int,out_features: int, bias: bool = True ):
        super(LinearCWT, self).__init__()
        
        self.out_features = out_features
        self.in_features = in_features
        
        default_dtype = torch.get_default_dtype()
        self.a = nn.Parameter(torch.ones(self.out_features, dtype=default_dtype).reshape(-1,1)  ) 
        self.b = nn.Parameter(torch.zeros(self.out_features, dtype=default_dtype).reshape(-1,1))     
        
        self.a.register_hook(lambda grad: grad / (torch.linalg.norm(grad) + 1e-8))
        self.b.register_hook(lambda grad: grad / (torch.linalg.norm(grad) + 1e-8))

        if bias:
            self.register_parameter(
                name='bias',
                param=torch.nn.Parameter(
                    torch.normal(
                        mean=0.0,
                        std=0.5,
                        size=(self.out_features,)
                    )
                )
            )
        else:
            self.register_parameter(
                name='bias',
                param=None
            )

    def shannon_kernel(self,t): 
            
        psi = torch.sinc(0.5*t) * torch.cos(0.5*3*PI*t)
        
        return psi
    
    def materialize_weights(self,x):

        t = torch.linspace(-1.0, 1.0, x.shape[-1], device = x.device).reshape(1, -1)

        psi = self.shannon_kernel(t)
        
        w =  torch.rsqrt(torch.pow(2,self.a)) * psi * (t- (torch.pow(2,self.a))*self.b)/ (torch.pow(2,self.a))  

        return w
        
    def forward(self,x):
        
        weights = self.materialize_weights(x)  
         
        y = F.linear(x,weights, self.bias)   
        return y

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )