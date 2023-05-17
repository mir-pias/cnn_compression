import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
# noinspection PyProtectedMember
from torch.nn.modules.utils import _single

from math import pi as PI

from torch.nn.modules.utils import _pair

from typing import (
    Tuple,
    Union,
    Optional
)

## dct_matrix = weights
class LinearDCT(nn.Module):
    def __init__(self,in_features: int,out_features: int, bias: bool = True ):
        super(LinearDCT, self).__init__()
        
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

        
        dct_m: torch.Tensor = 2 * norm * torch.cos(0.5 * PI * (self.fc / self.out_features) * (2 * t + 1))

        
        return dct_m.unsqueeze(1)
    
        
    def forward(self,x):
        
        w = self.materialize_weights(x) 

        dummy = torch.ones(x.shape[0], *((1,) * (x.dim() - 1)), device=x.device, dtype=x.dtype, requires_grad=False)

        y = F.bilinear(dummy, x, w, bias=self.bias)
          
        # y = F.linear(x,w, self.bias)   
        return y

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

## 2D dct_matrix as conv kernel
class Conv2dDCT(torch.nn.Module):

    fcc: torch.nn.Parameter    # central frequencies (output channels)
    fch: torch.nn.Parameter    # central frequencies (2D convolutional kernel height)
    fcw: torch.nn.Parameter    # central frequencies (2D convolutional kernel width)
    delta: torch.nn.Parameter  # weights of basis functions
    bias: torch.nn.Parameter

    def __init__(
        self,
        in_channels,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super(Conv2dDCT, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels    
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        fcc = torch.arange(
            self.out_channels,
            dtype=torch.get_default_dtype()
        ).reshape(-1, 1)
        self.register_parameter(name='fcc', param=torch.nn.Parameter(fcc))

        fch = torch.arange(
            self.kernel_size[0],
            dtype=torch.get_default_dtype()
        ).reshape(-1, 1)
        self.register_parameter(name='fch', param=torch.nn.Parameter(fch))

        fcw = torch.arange(
            self.kernel_size[1],
            dtype=torch.get_default_dtype()
        ).reshape(-1, 1)
        self.register_parameter(name='fcw', param=torch.nn.Parameter(fcw))

        num_filters = self.kernel_size[0] * self.kernel_size[1]

        # delta = torch.full((self.out_channels,num_filters), 1/num_filters)
        delta = torch.randn(self.out_channels, num_filters)

        self.register_parameter(name='delta', param=torch.nn.Parameter(delta))

        self.register_parameter(name='bias', param=torch.nn.Parameter(torch.zeros(self.out_channels)) if bias else None)

        def norm(grad):
            return grad / (torch.linalg.norm(grad, ord=None) + 1e-8)

        self.fcc.register_hook(norm)
        self.fch.register_hook(norm)
        self.fcw.register_hook(norm)
        self.delta.register_hook(norm)


    def materialize_weights(self, x: torch.Tensor) -> torch.Tensor:
        # in_channels: int = x.shape[1] // self.groups
        in_channels = self.in_channels // self.groups

        tc: torch.Tensor = torch.arange(
            in_channels,
            dtype=x.dtype,
            device=x.device
        ).reshape(
            1,  # feature dimension
            -1  # time dimension
        )
        th: torch.Tensor = torch.arange(
            self.kernel_size[0], dtype=x.dtype, device=x.device
        ).reshape(1, -1)
        tw: torch.Tensor = torch.arange(
            self.kernel_size[1], dtype=x.dtype, device=x.device
        ).reshape(1, -1)

        norm_c: torch.Tensor = torch.rsqrt(
            torch.full_like(
                self.fcc, 2 * in_channels
            ) * (
                torch.eye(self.out_channels, 1, device=x.device, dtype=x.dtype) + 1
            )
        )

        kc: torch.Tensor = 2 * norm_c * torch.cos(0.5 * PI * (self.fcc / self.out_channels) * (2 * tc + 1))

        norm_h: torch.Tensor = torch.rsqrt(
            torch.full_like(
                self.fch, 2 * self.kernel_size[0]
            ) * (
                torch.eye(self.kernel_size[0], 1, device=x.device, dtype=x.dtype) + 1
            )
        )
        kh: torch.Tensor = 2 * norm_h * torch.cos(0.5 * PI * (self.fch / self.kernel_size[0]) * (2 * th + 1))

        norm_w: torch.Tensor = torch.rsqrt(
            torch.full_like(
                self.fcw, 2 * self.kernel_size[1]
            ) * (
                torch.eye(self.kernel_size[1], 1, device=x.device, dtype=x.dtype) + 1
            )
        )
        kw: torch.Tensor = 2 * norm_w * torch.cos(0.5 * PI * (self.fcw / self.kernel_size[1]) * (2 * tw + 1))


        w: torch.Tensor = kc.reshape(
            self.out_channels, -1, 1, 1, 1, 1
        ) * kh.reshape(
            1,1, self.kernel_size[0], -1, 1, 1
        ) * kw.reshape(
            1, 1, 1, 1, -1, self.kernel_size[1]                               # N_out x N_in x kH x kH x kW x kW
        )

        
        w = w.reshape(self.out_channels, in_channels, -1, *self.kernel_size)  # N_out x N_in x kH x (kH * kW) x kW

        # weighted sum of basis functions
        w = torch.mean(w * self.delta.reshape(self.out_channels, 1, -1, 1, 1), dim=2)  # N_out x N_in x kH x kW
        # w = torch.mean(w,dim=2)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.materialize_weights(x)

        y = F.conv2d(
            input=x,
            weight=w,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

        return y
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
