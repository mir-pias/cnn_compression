# from turtle import forward
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.common_types import (_size_any_t, _size_1_t, _size_2_t, _size_3_t,
                            _ratio_3_t, _ratio_2_t, _size_any_opt_t, _size_2_opt_t, _size_3_opt_t)

from torch.nn.modules.utils import _single, _pair, _triple
from torch import Tensor

from typing import (
    Tuple,
    Union,
    Optional
)


def complex_abs(x: torch.Tensor) -> torch.Tensor:
    return (x[..., 0] ** 2 + x[..., 1] ** 2) ** 0.5


def complex_angle(x: torch.Tensor) -> torch.Tensor:
    x = torch.where(
        (x[..., :1] == 0) & (x[..., 1:] == 0),
        torch.stack((torch.ones_like(x[..., 0]), torch.zeros_like(x[..., 1])), -1),
        x
    )

    return torch.atan2(x[..., 1], x[..., 0])


class Cardioid(torch.nn.Module):

    @staticmethod
    def cardioid(x: torch.Tensor) -> torch.Tensor:
        scale = torch.cos(complex_angle(x))
        return 0.5 * (1.0 + torch.stack((scale,) * 2, dim=-1)) * x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cardioid(x)

class ComplexMaxPool2d(torch.nn.MaxPool2d):

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        return_indices: bool = False,
        ceil_mode: bool = False
    ):
        super(ComplexMaxPool2d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode
        )

    @staticmethod
    def max_pool2d(
        input: torch.Tensor,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        ceil_mode: bool = False,
        return_indices: bool = False,
    ):
        _, indices = F.max_pool2d(
            input=complex_abs(input),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=True
        )

        pooled = torch.stack(
            (
                input[..., 0].flatten(start_dim=2).gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices),
                input[..., 1].flatten(start_dim=2).gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)),
            dim=-1
        )

        return (pooled, indices) if return_indices else pooled

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.max_pool2d(
            input=x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices
        )
        

class ConcatenatedReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,x):
        act = nn.ReLU()
        
        return torch.stack((act(x[...,0]),act(-x[...,1])), dim=-1)


class ComplexBatchNorm2d(torch.nn.Module):

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ):
        super(ComplexBatchNorm2d, self).__init__()

        self.bn_real = torch.nn.BatchNorm2d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )
        self.bn_imag = torch.nn.BatchNorm2d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack((self.bn_real(x[..., 0]), self.bn_imag(x[..., 1])), dim=-1) / 2.0


class ComplexAdaptiveAvgPool2d(torch.nn.Module):

    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        self.output_size = output_size
        super().__init__()

    def forward(self, x:Tensor) -> Tensor:
        return torch.stack((F.adaptive_avg_pool2d(x[..., 0],self.output_size), F.adaptive_avg_pool2d(x[..., 1], self.output_size)), dim=-1)


class ComplexAvgPool2d(torch.nn.AvgPool2d):

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        
    ):
        super(ComplexAvgPool2d, self).__init__(
            kernel_size=kernel_size,
            stride=stride if (stride is not None) else kernel_size,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad = count_include_pad,
            divisor_override = divisor_override

        )
    @staticmethod
    def avgpool2d(
        input: torch.Tensor,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
    ):

        pooled_real = F.avg_pool2d(input[...,0], kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

        pooled_complex = F.avg_pool2d(input[...,1], kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

        pooled = torch.stack((pooled_real,pooled_complex), dim=-1)

        return pooled

    def forward(self, x: torch.Tensor):

        return self.avgpool2d(
            input=x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override
            )

