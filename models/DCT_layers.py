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

## dct_matrix = weights
class LinearDCT(nn.Module):
    def __init__(self,in_features: int,out_features: int, bias: bool = True ):
        super(LinearDCT, self).__init__()
        
        self.out_features = out_features
        self.in_features = in_features
        
        default_dtype = torch.get_default_dtype()
        self.fc = nn.Parameter(torch.arange((self.out_features), dtype=default_dtype).reshape(-1,1))     
        
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

    def dct_kernel(self,t): 
        dct_m = np.sqrt(2/(self.out_features)) * torch.cos(0.5 * np.pi * self.fc * (2 * t + 1) / self.out_features)
        
        dct_m[0] = dct_m[0]/np.sqrt(2)
        
        return dct_m
    
        
    def forward(self,x):
        # print(x.shape)
        
        t = torch.arange(x.shape[-1], device = x.device).reshape(1,-1)
        w = self.dct_kernel(t) 
        
        # print('dct_lin w: ', w.shape)  
        y = F.linear(x,w, self.bias)   
        return y

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

## 2D dct_matrix as conv kernel
class Conv2dDCT(torch.nn.Module):

    fcc: torch.nn.Parameter  # central frequencies (output channels)
    fcl: torch.nn.Parameter  # central frequencies (1D convolutional kernel length)
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
        super(Conv2dDCT, self).__init__()
        
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

        t_c = torch.arange(self.in_channels, dtype=x.dtype, device=x.device).reshape(1, -1)

        t_l = torch.arange(self.kernel_size[0], dtype=x.dtype, device=x.device).reshape(1, -1)

        norm_c = torch.rsqrt(
            torch.full_like(
                self.fcc, 2 * self.out_channels
            ) * (
                torch.eye(self.out_channels, 1, device=x.device, dtype=x.dtype) + 1
            )
        )
        # print('norm_c shape: ', norm_c.shape)

        kc: torch.Tensor = 2 * norm_c * torch.cos(0.5 * PI * self.fcc * (2 * t_c + 1) / self.out_channels)

        norm_l = torch.rsqrt(
            torch.full_like(
                self.fcl, 2 * self.kernel_size[0]
            ) * (
                torch.eye(self.kernel_size[0], 1, device=x.device, dtype=x.dtype) + 1
            )
        )
        # print('t_l shape: ', t_l.shape)
        # print('t_c shape: ', t_c.shape)
        # print('norm_l shape: ', norm_l.shape)
        kl: torch.Tensor = 2 * norm_l * torch.cos(0.5 * PI * self.fcl * (2 * t_l + 1) / self.kernel_size[0])

        # print('kc_reshape:', kc.reshape(
        #     self.out_channels, -1, 1,1
        # ).shape)
        # print('kl_reshape:', kl.reshape(
        #     1,1, -1, self.kernel_size[0]
        # ).shape)

        # print('kc_shape: ', kc.shape)
        # print('kl_shape: ', kl.shape)

        w: torch.Tensor = kc.reshape(
            self.out_channels, -1, 1,1
        ) * kl.reshape(
            1,1, -1, self.kernel_size[0]
        )

        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._materialize_weights(x)
        # print('w: ', w.shape)
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


## dct_matrix @ uniform weights, weights as conv kernel
class DCT_conv_layer(nn.Module):
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
        super(DCT_conv_layer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.groups = groups
        
        default_dtype = torch.get_default_dtype()
        self.fc = nn.Parameter(torch.arange(self.kernel_size[0], dtype=default_dtype).reshape(-1,1))     
        
        self.weight = nn.Parameter(torch.empty((self.out_channels,self.in_channels,self.kernel_size[0],self.kernel_size[0]), 
                                          dtype=default_dtype))

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
        
        self.reset_parameters()
        
        self.weight.register_hook(lambda grad: grad / (torch.linalg.norm(grad) + 1e-8))
        self.fc.register_hook(lambda grad: grad / (torch.linalg.norm(grad) + 1e-8))
        

    def dct_kernel(self,t): 
        dct_m = np.sqrt(2/(self.kernel_size[0])) * torch.cos(0.5 * np.pi * self.fc * (2 * t + 1) / self.kernel_size[0])
        
        dct_m[0] = dct_m[0]/np.sqrt(2)
        
        return dct_m
    
    def reset_parameters(self) -> None:
        ## torch.Conv2d docs
        k = 1/(self.in_channels * (self.kernel_size[0])**2)
        nn.init.uniform_(self.weight, a= -math.sqrt(k), b = math.sqrt(k))
        
        
    def forward(self,x):
        
        t = torch.arange(self.kernel_size[0], device=x.device).reshape(1,-1)
        dct_m = self.dct_kernel(t) 
        
#         print(self.w.shape)

        w = self.weight @ dct_m   ## dct on uniform weights, weights as conv kernel 


#         print(x.shape)
        # print('w ',w.shape)
        # print(dct_m.shape)
#         print((x@w.T).shape)
        
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


# dct on kaiming_uniform weights, doesn't train, needs a better look, possibly useless
class DCT_linear_layer(nn.Module):
    def __init__(self,in_features: int, out_features: int, bias: bool = True):
        super(DCT_linear_layer, self).__init__()
        
        self.out_features = out_features
        self.in_features = in_features
        
        default_dtype = torch.get_default_dtype()
        self.fc = nn.Parameter(torch.arange((self.out_features), dtype=default_dtype).reshape(-1,1))     

        self.weight = nn.Parameter(torch.empty((out_features, in_features),  dtype=default_dtype))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.fc.register_hook(lambda grad: grad / (torch.linalg.norm(grad) + 1e-8))
        self.weight.register_hook(lambda grad: grad / (torch.linalg.norm(grad) + 1e-8))


    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5), non)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def dct_kernel(self,t): 
        dct_m = np.sqrt(2/(self.out_features)) * torch.cos(0.5 * np.pi * self.fc * (2 * t + 1) / self.out_features)
        
        dct_m[0] = dct_m[0]/np.sqrt(2)
        
        return dct_m
    
        
    def forward(self,x):
#         print(x.shape)
        t = torch.arange(x.shape[-1], device = x.device).reshape(1,-1)
        dct_m = self.dct_kernel(t) 
        
        w = dct_m * self.weight ## dct on kaiming_uniform weights
        # print('LinearDCT w: ', w.shape)
        y = F.linear(x,w, self.bias)   
        return y

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )