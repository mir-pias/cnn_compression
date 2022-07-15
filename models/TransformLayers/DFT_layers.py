import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
# noinspection PyProtectedMember
from torch.nn.modules.utils import _pair

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

    def dft_kernel(self,t,fc,in_features): 

        norm = torch.rsqrt(
            torch.full_like(
                fc, in_features
            ) 
        )

        dft_m = norm * torch.cat((torch.cos((fc*t*2*PI)/in_features), - torch.sin((fc*t*2*PI)/in_features)), dim=-1) 
        
        # dft_m = dft_m / (math.sqrt(self.in_features)) ## normalize
        
        return dft_m
    
    def materialize_weights(self, x):
        x_is_complex = x.shape[-1] == 2
        in_features = x.shape[-1 - int(x_is_complex)]

        # print(x_is_complex)
        # print(in_features)
        t = torch.arange(in_features,dtype=x.dtype, device=x.device).reshape(1, -1, 1)
        # t = torch.linspace(-1.0, 1.0, in_features, dtype=x.dtype, device=x.device).reshape(1, -1, 1)
        fc = self.fc

        weights = self.dft_kernel(t,fc,in_features) 
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

class Conv2dDFT(torch.nn.Module):

    fcc: torch.nn.Parameter  # central frequencies (output channels)
    fch: torch.nn.Parameter  # central frequencies (2D convolutional kernel length)
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
        super(Conv2dDFT, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.register_parameter(
            name='fcc',
            param=torch.nn.Parameter(
                torch.arange(
                    self.out_channels,
                    dtype=torch.get_default_dtype()
                ).reshape(-1, 1, 1)
            )
        )
        self.register_parameter(
            name='fch',
            param=torch.nn.Parameter(
                torch.arange(
                    self.kernel_size[0],
                    dtype=torch.get_default_dtype()
                ).reshape(-1, 1, 1)
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
        self.fch.register_hook(norm)

    def _materialize_weights(self, x: torch.Tensor) -> torch.Tensor:
        # in_features = x.shape[1]
        

        x_is_complex = x.shape[-1] == 2

        tc = torch.arange(self.in_channels, dtype=x.dtype, device=x.device).reshape(1, -1, 1)

        tl = torch.arange(self.kernel_size[0], dtype=x.dtype, device=x.device).reshape(1, -1, 1)

        norm_c = torch.rsqrt(
            torch.full_like(
                self.fcc, self.in_channels
            ) * (
                torch.ones(self.in_channels, 1, device=x.device, dtype=x.dtype) 
            )
        )

        kc: torch.Tensor = norm_c * torch.cat((torch.cos((self.fcc*tc*2*PI)/self.out_channels), - torch.sin((self.fcc*tc*2*PI)/self.out_channels)), dim=-1)

        norm_l = torch.rsqrt(
            torch.full_like(
                self.fch, self.kernel_size[0]
            ) * (
                torch.ones(self.kernel_size[0], 1, device=x.device, dtype=x.dtype) 
            )
        )
        
        kl: torch.Tensor = norm_l * torch.cat((torch.cos((self.fch*tl*2*PI)/self.kernel_size[0]), - torch.sin((self.fch*tl*2*PI)/self.kernel_size[0])), dim=-1)


        w: torch.Tensor = kc.reshape(
        self.out_channels, -1,kc.shape[-1],1,1
        ) * kl.reshape(
        1,1,kl.shape[-1], -1, self.kernel_size[0]
        )

        return w.transpose(2,4), x_is_complex ## transpose to fix shape for F.conv2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, x_is_complex = self._materialize_weights(x)
        # print('w: ', w.shape)
        
        if x_is_complex:

            y0 = F.conv2d(input=x[..., 0],weight=weights[...,0],bias=self.bias,
                        stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups
                        ) 
            - F.conv2d(input=x[..., 1],weight=weights[...,1],bias=self.bias,
                        stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups
                        ) 

            y1 = F.conv2d(input=x[..., 0],weight=weights[...,1],bias=self.bias,
                        stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups
                        ) 
            + F.conv2d(input=x[..., 1],weight=weights[...,0],bias=self.bias,
                        stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups
                        ) 

            y = torch.stack((y0,y1), dim=-1)

        else:
            y0 = F.conv2d(input=x,weight=weights[...,0],bias=self.bias,
                        stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups
                        ) 
            y1 = F.conv2d(input=x,weight=weights[...,1],bias=self.bias,
                        stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups
                        ) 

            y = torch.stack((y0,y1), dim=-1)

        return y.sum(-1) ## no more complex outputs from this layer, maybe ok

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
