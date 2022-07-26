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

## dst_matrix = weights
class LinearDST(nn.Module):
    def __init__(self,in_features: int,out_features: int, bias: bool = True ):
        super(LinearDST, self).__init__()
        
        self.out_features = out_features
        self.in_features = in_features
        
        default_dtype = torch.get_default_dtype()
        self.fc = nn.Parameter(torch.arange(self.out_features, dtype=default_dtype).reshape(-1,1))     
        
        self.fc.register_hook(lambda grad: grad / (torch.linalg.norm(grad) + 1e-8))

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

    def materialize_weights(self,x): 

        t: torch.Tensor = torch.arange(x.shape[-1], device = x.device).reshape(1,-1)

        norm: torch.Tensor = torch.rsqrt(
            torch.full_like(
                self.fc, 2 * self.in_features
            ) * (
                torch.eye(self.out_features, 1, device=x.device, dtype=x.dtype) + 1
            )
        )

        dst_m: torch.Tensor = 2 * norm * torch.sin(0.5 * PI * ((self.fc + 1 )/ self.in_features) * (2 * t + 1)) 

        return dst_m
    
        
    def forward(self,x):
        
        
        w = self.materialize_weights(x) 
         
        y = F.linear(x,w, self.bias)   
        return y

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


## 2D dst_matrix as conv kernel
class Conv2dDST(torch.nn.Module):

    fcc: torch.nn.Parameter  # central frequencies (output channels)
    fcl: torch.nn.Parameter  # central frequencies (2D convolutional kernel length)
    bias: torch.nn.Parameter

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Optional[Union[int, Tuple[int]]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super(Conv2dDST, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.groups = groups

        self.register_parameter(
            name='fcc',
            param=torch.nn.Parameter(
                torch.arange(
                    self.out_channels,
                    dtype=torch.get_default_dtype()
                ).reshape(-1, 1)
            )
        )
        self.register_parameter(
            name='fcl',
            param=torch.nn.Parameter(
                torch.arange(
                    self.kernel_size[0],
                    dtype=torch.get_default_dtype()
                ).reshape(-1, 1)
            )
        )

        if bias:
            self.register_parameter(
                name='bias',
                param=torch.nn.Parameter(
                    torch.normal(
                        mean=0.0,
                        std=0.5,
                        size=(self.out_channels,)
                    )
                )
            )
        else:
            self.register_parameter(
                name='bias',
                param=None
            )

        def norm(grad):
            return grad / torch.linalg.norm(grad, ord=None)

        self.fcc.register_hook(norm)
        self.fcl.register_hook(norm)

    def _materialize_weights(self, x: torch.Tensor) -> torch.Tensor:
        # in_features = x.shape[1]

        try:
            width = self.kernel_size[1]
        except IndexError:
            width = self.kernel_size[0]

        t_c = torch.arange(self.in_channels, dtype=x.dtype, device=x.device).reshape(1, -1)

        t_l = torch.arange(width, dtype=x.dtype, device=x.device).reshape(1, -1)

        norm_c = torch.rsqrt(
            torch.full_like(
                self.fcc, 2 * self.out_channels
            ) * (
                torch.eye(self.out_channels, 1, device=x.device, dtype=x.dtype) + 1
            )
        ) * torch.rsqrt(torch.full_like(
                self.fcc, 2))

        kc: torch.Tensor = 2 * norm_c * torch.sin(0.5 * PI * (self.fcc + 1) * (2 * t_c + 1) / self.out_channels)

        norm_l = torch.rsqrt(
            torch.full_like(
                self.fcl, 2 * self.kernel_size[0]
            ) * (
                torch.eye(self.kernel_size[0], 1, device=x.device, dtype=x.dtype) + 1
            )
        ) * torch.rsqrt(torch.full_like(
                self.fcl, 2))

        kl: torch.Tensor = 2 * norm_l * torch.sin(0.5 * PI * (self.fcl + 1) * (2 * t_l + 1) / self.kernel_size[0])

        w: torch.Tensor = kc.reshape(
            self.out_channels, -1, 1,1
        ) * kl.reshape(
            1,1, -1, width
        )

        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._materialize_weights(x)

        return F.conv2d(
            input=x,
            weight=w,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

    def extra_repr(self):
        s = ('in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)