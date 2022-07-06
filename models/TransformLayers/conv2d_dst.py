import torch
import torch.nn.functional as F

# noinspection PyProtectedMember
from torch.nn.modules.utils import _pair
import math
from math import pi as PI

from typing import (
    Tuple,
    Union,
    Optional
)


class Conv2dDST(torch.nn.Module):

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
        super(Conv2dDST, self).__init__()

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


    def _materialize_weights(self, x: torch.Tensor) -> torch.Tensor:
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

        kc: torch.Tensor = 2 * norm_c * torch.sin(0.5 * PI * ((self.fcc + 1 )/ self.out_channels) * (2 * tc + 1)) 

        norm_h: torch.Tensor = torch.rsqrt(
            torch.full_like(
                self.fch, 2 * self.kernel_size[0]
            ) * (
                torch.eye(self.kernel_size[0], 1, device=x.device, dtype=x.dtype) + 1
            )
        ) 
        kh: torch.Tensor = 2 * norm_h * torch.sin(0.5 * PI * ((self.fch +1) / self.kernel_size[0]) * (2 * th + 1)) 

        norm_w: torch.Tensor = torch.rsqrt(
            torch.full_like(
                self.fcw, 2 * self.kernel_size[1]
            ) * (
                torch.eye(self.kernel_size[1], 1, device=x.device, dtype=x.dtype) + 1
            )
        ) 
        kw: torch.Tensor = 2 * norm_w * torch.sin(0.5 * PI * ((self.fcw +1) / self.kernel_size[1]) * (2 * tw + 1)) 


        w: torch.Tensor = kc.reshape(
            self.out_channels, -1, 1, 1, 1, 1
        ) * kh.reshape(
            1,1, self.kernel_size[0], -1, 1, 1
        ) * kw.reshape(
            1, 1, 1, 1, -1, self.kernel_size[1]                               # N_out x N_in x kH x kH x kW x kW
        )
        
        w = w.reshape(self.out_channels, in_channels, -1, *self.kernel_size)  # N_out x N_in x kH x (kH * kW) x kW

        # weighted sum of basis functions
        # w = torch.mean(w * self.delta.reshape(self.out_channels, 1, -1, 1, 1), dim=2)  # N_out x N_in x kH x kW
        w = torch.mean(w,dim=2)
        return w 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._materialize_weights(x)

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
        s = 'out_channels={out_channels}, kernel_size={kernel_size}, stride={stride}'

        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'

        return s.format(**self.__dict__)
