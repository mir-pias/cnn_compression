import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
# noinspection PyProtectedMember
from torch.nn.modules.utils import _single, _pair

from math import pi as PI

from typing import (
    Tuple,
    Union,
    Optional,
    cast
)

## cwt_matrix = weights
class LinearRealShannon(nn.Module):
    fc: torch.nn.Parameter
    bias: torch.nn.Parameter

    def __init__(self, in_features, out_features: int, bias: bool = True):
        super(LinearRealShannon, self).__init__()

        self.out_features = out_features
        self.in_features = in_features

        fc = torch.arange(
            self.out_features,
            dtype=torch.get_default_dtype()
        ).reshape(-1, 1)
        self.register_parameter(name='fc', param=torch.nn.Parameter(fc))

        # self.register_parameter(name='bias', param=torch.nn.Parameter(torch.zeros(self.out_features)) if bias else None)

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


        self.fc.register_hook(lambda grad: grad / (torch.linalg.norm(grad) + 1e-8))

    def materialize_weights(self, x: torch.Tensor) -> torch.Tensor:
        in_features = x.shape[-1]

        t = torch.arange(
            in_features,
            dtype=x.dtype,
            device=x.device
        ).reshape(
            1,  # feature dimension
            -1  # time dimension
        )

        # fb: torch.Tensor = torch.ones_like(self.fc)
        norm: torch.Tensor = torch.rsqrt(
            1.2 * torch.full_like(
                self.fc, in_features
            ) * (
                torch.eye(self.out_features, 1, device=x.device, dtype=x.dtype) + 1
            ) / PI
        )
        w: torch.Tensor = norm  * torch.sinc(
            (t - in_features // 2) / in_features
        ) * torch.cos(2 * PI * (self.fc / self.out_features) * t)

        return w.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.materialize_weights(x)
        dummy = torch.ones(x.shape[0], *((1,) * (x.dim() - 1)), device=x.device, dtype=x.dtype, requires_grad=False)

        y = F.bilinear(dummy, x, w, bias=self.bias)

        return y

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Conv2dRealShannon(torch.nn.Module):

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
        super(Conv2dRealShannon, self).__init__()

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
        in_channels: int = x.shape[1] // self.groups

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

        # fbc: torch.Tensor = torch.ones_like(self.fcc)
        # fbh: torch.Tensor = torch.ones_like(self.fch)
        # fbw: torch.Tensor = torch.ones_like(self.fcw)

        norm_c: torch.Tensor = torch.rsqrt(
            1.2 * torch.full_like(
                self.fcc, in_channels
            ) * (
                torch.eye(self.out_channels, 1, device=x.device, dtype=x.dtype) + 1
            ) / PI
        )
        kc: torch.Tensor = norm_c * torch.sinc(
             (tc - in_channels // 2) / in_channels
        ) * torch.cos(2 * PI * (self.fcc / self.out_channels) * tc)

        norm_h: torch.Tensor = torch.rsqrt(
            1.2 * torch.full_like(
                self.fch, self.kernel_size[0]
            ) * (
                torch.eye(self.kernel_size[0], 1, device=x.device, dtype=x.dtype) + 1
            ) / PI
        )
        kh: torch.Tensor = norm_h * torch.sinc(
             (th - in_channels // 2) / in_channels
        ) * torch.cos(2 * PI * (self.fch / self.kernel_size[0]) * th)

        norm_w: torch.Tensor = torch.rsqrt(
            1.2 * torch.full_like(
                self.fcw, self.kernel_size[1]
            ) * (
                torch.eye(self.kernel_size[1], 1, device=x.device, dtype=x.dtype) + 1
            ) / PI
        )
        kw: torch.Tensor = norm_w * torch.sinc(
             (tw - in_channels // 2) / in_channels
        ) * torch.cos(2 * PI * (self.fcw / self.kernel_size[1]) * tw)

        w: torch.Tensor = kc.reshape(
            self.out_channels, 1, 1, -1, 1, 1
        ) * kh.reshape(
            1, self.kernel_size[0], 1, 1, -1, 1
        ) * kw.reshape(
            1, 1, -1, 1, 1, self.kernel_size[1]                               # N_out x kH x kW x N_in x kH x kW
        )
        w = w.reshape(self.out_channels, -1, in_channels, *self.kernel_size)  # N_out x (kH * kW) x N_in x kH x kW
        # weighted sum of basis functions
        w = torch.mean(w * self.delta.reshape(self.out_channels, -1, 1, 1, 1), dim=1)  # N_out x N_in x kH x kW

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


class LinearShannon(nn.Module):
    def __init__(self,in_features: int,out_features: int, bias: bool = True ):
        super(LinearShannon, self).__init__()
        
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
    
    def materialize_weights(self, x):
        x_is_complex = x.shape[-1] == 2
        in_features = x.shape[-1 - int(x_is_complex)]

        t = torch.arange(in_features,dtype=x.dtype, device=x.device).reshape(1, -1, 1)

        norm: torch.Tensor = torch.rsqrt(
            1.2 * torch.full_like(
                self.fc,2 * in_features
            ) / PI
        )

        kernel = torch.cat((torch.cos(2*PI * (self.fc / self.out_features) * t), - torch.sin(2*PI * (self.fc / self.out_features) * t)), dim=-1)

        weights: torch.Tensor = norm  * torch.sinc(
            (t - in_features // 2) / in_features
        ) * kernel


        return weights.unsqueeze(1), x_is_complex

    def forward(self,x):
        weights, x_is_complex = self.materialize_weights(x) 

        dummy = torch.ones(x.shape[0], *((1,) * (x.dim() - 1)), device=x.device, dtype=x.dtype, requires_grad=False)
        
        if x_is_complex:
            dummy = dummy.squeeze(-1)
            y = torch.stack((
                    F.bilinear(dummy, x[...,0], weights[...,0]) - F.bilinear(dummy, x[...,1], weights[...,1]),
                    F.bilinear(dummy, x[...,0], weights[...,1]) + F.bilinear(dummy, x[...,1], weights[...,0])
                ), dim=-1)

        else:
            y = torch.stack((
                        F.bilinear(dummy, x, weights[...,0]),
                        F.bilinear(dummy, x, weights[...,1])
                    ), dim=-1)    

        if (self.bias is not None) and (self.bias.numel() == (self.out_features * 2)):
            y = y + self.bias

        return y

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class Conv2dShannon(torch.nn.Module):

    fcc: torch.nn.Parameter  # central frequencies (output channels)
    fch: torch.nn.Parameter  # central frequencies (2D convolutional kernel height)
    fcw: torch.nn.Parameter  # central frequencies (2D convolutional kernel width)
    bias: torch.nn.Parameter
    delta: torch.nn.Parameter  # weights of basis functions

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
        super(Conv2dShannon, self).__init__()
        
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
                ).reshape(-1, 1, 1)

        self.register_parameter(
            name='fcc',
            param= torch.nn.Parameter(fcc)
        )

        fch = torch.arange(
                    self.kernel_size[0],
                    dtype=torch.get_default_dtype()
                ).reshape(-1, 1, 1)

        self.register_parameter(
            name='fch',
            param= torch.nn.Parameter(fch)
        )

        fcw = torch.arange(
                    self.kernel_size[1],
                    dtype=torch.get_default_dtype()
                ).reshape(-1, 1, 1)

        self.register_parameter(
            name='fcw',
            param= torch.nn.Parameter(fcw)
        )

        self.register_parameter(name='bias', param=torch.nn.Parameter(torch.zeros(self.out_channels)) if bias else None)

        num_filters = self.kernel_size[0] * self.kernel_size[1]

        # delta = torch.full((self.out_channels,num_filters), 1/num_filters)
        delta = torch.randn(self.out_channels, num_filters)

        self.register_parameter(name='delta', param=torch.nn.Parameter(delta))

        def norm(grad):
            return grad / (torch.linalg.norm(grad, ord=None) + 1e-8)

        self.fcc.register_hook(norm)
        self.fch.register_hook(norm)
        self.fcw.register_hook(norm)
        self.delta.register_hook(norm)

    def materialize_weights(self, x: torch.Tensor) -> torch.Tensor:
        in_channels = x.shape[1]
    
        x_is_complex = x.shape[-1] == 2

        tc: torch.Tensor = torch.arange(in_channels, dtype=x.dtype, device=x.device).reshape(1, -1, 1)

        th: torch.Tensor = torch.arange(self.kernel_size[0], dtype=x.dtype, device=x.device).reshape(1, -1, 1)

        tw: torch.Tensor = torch.arange(self.kernel_size[1], dtype=x.dtype, device=x.device).reshape(1, -1, 1)

        norm_c: torch.Tensor = torch.rsqrt(
            1.2 * torch.full_like(
                self.fcc, 2 * in_channels
            ) / PI
        )
        win_c = torch.sinc((tc - in_channels // 2) / in_channels) ## window

        kernel_c = torch.cat((torch.cos(2*PI * (self.fcc / self.out_channels) * tc), - torch.sin(2*PI * (self.fcc / self.out_channels) * tc)), dim=-1)

        kc: torch.Tensor = norm_c * win_c * kernel_c

        norm_h: torch.Tensor = torch.rsqrt(
          1.2 * torch.full_like(
                self.fch, 2 * self.kernel_size[0]
            ) / PI
        ) 

        win_h = torch.sinc((th - self.kernel_size[0] // 2) / self.kernel_size[0]) ## window

        kernel_h = torch.cat((torch.cos(2*PI*(self.fch / self.kernel_size[0]) * th),- torch.sin(2*PI*(self.fch / self.kernel_size[0]) * th)), dim=-1)
        
        kh: torch.Tensor = norm_h * win_h * kernel_h

        norm_w: torch.Tensor = torch.rsqrt(
            1.2 * torch.full_like(
                self.fcw, 2 * self.kernel_size[1]
            ) / PI 
        ) 

        win_w = torch.sinc((tw - self.kernel_size[1] // 2) / self.kernel_size[1]) ## window

        kernel_w = torch.cat((torch.cos(2*PI*(self.fcw / self.kernel_size[1]) * tw),- torch.sin(2*PI*(self.fcw / self.kernel_size[1]) * tw)), dim=-1)
        
        kw: torch.Tensor = norm_w * win_w * kernel_w


        w: torch.Tensor =kc.reshape(
        self.out_channels, -1, kc.shape[-1], 1,1,1,1,1,1
        )*  kh.reshape(
        1,1, 1, self.kernel_size[0],-1,kh.shape[-1],1,1,1
        ) * kw.reshape(1,1,1,1 ,1,1, self.kernel_size[1],-1 ,kw.shape[-1])  # N_out x N_in x 2 x kH x kH x 2 x kW x kW x 2

        w = torch.sum(w,dim=(2,5))  # N_out x N_in x 2 x kH x kH  x kW x kW x 2
        ## mean doesn't train, sum in complex dims trains

        w = w.reshape(self.out_channels, self.in_channels , self.kernel_size[0], -1, self.kernel_size[1], w.shape[-1] )  
        # N_out x N_in x kH x (kH * kW) x kW x 2
        
        
        w = torch.mean(w * self.delta.reshape(self.out_channels, 1, 1, -1, 1, 1), dim=3)  # N_out x N_in x kH x kW x 2
        # w = w.mean(w,dim=3)

        return w, x_is_complex 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, x_is_complex = self.materialize_weights(x)
        
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
