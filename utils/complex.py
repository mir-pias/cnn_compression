from turtle import forward
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.common_types import (_size_any_t, _size_1_t, _size_2_t, _size_3_t,
                            _ratio_3_t, _ratio_2_t, _size_any_opt_t, _size_2_opt_t, _size_3_opt_t)

from typing import List, Optional
from torch.nn.modules.utils import _single, _pair, _triple
from torch import Tensor

class Cardioid(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, z):
        # print(torch.cos(torch.atan2(z[...,1], z[...,0])).shape)
        # print(z.shape)
        return 0.5*(1 + torch.cos(torch.atan2(z[...,1], z[...,0])))*(torch.pow(z[...,0],2) + torch.pow(z[...,1],2))  ## squared magnitude of z

class ConcatenatedMaxPool2d(nn.Module):
    def __init__(
        self,kernel_size: _size_2_t, stride: Optional[_size_2_t] = None,
                 padding: _size_2_t = 0, dilation: _size_2_t = 1,
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
    
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self,input: Tensor) -> Tensor:
        
    
        y1 = F.max_pool2d(input[...,0], self.kernel_size, self.stride,
                            self.padding, self.dilation, ceil_mode=self.ceil_mode,
                            return_indices=self.return_indices)

        y2 = F.max_pool2d(input[...,1], self.kernel_size, self.stride,
                            self.padding, self.dilation, ceil_mode=self.ceil_mode,
                            return_indices=self.return_indices)

        return torch.stack((y1,y2), dim=-1)
        

class ConcatenatedReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,x):
        act = nn.ReLU()
        
        return torch.stack((act(x[...,0]),act(-x[...,1])), dim=-1)