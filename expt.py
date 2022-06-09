import sys

from numpy import dtype  
sys.path.append('.')

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from scipy.fftpack import dct, fft , ifft
import math
from misc.conv import Conv1dDCT
from models.TransformLayers.DCT_layers import Conv2dDCT, LinearDCT
from models.TransformLayers.DFT_layers import LinearDFT
from models.TransformLayers.LinearFBSP import LinearFBSP
from math import pi as PI

if __name__ == '__main__':
    

    ## dct matrix check

    # in_features, out_features = 3,3

    # t = torch.arange(in_features).reshape(1, -1)
    # # t = torch.linspace(-1.0, 1.0, in_features).reshape(1, -1, 1)
    # fc = torch.arange(out_features ).reshape(-1, 1)
    # print((t * fc).shape)
    
    # # w = torch.exp(torch.as_tensor((-2*np.pi*i)/N, dtype=torch.cfloat))

    # norm = torch.rsqrt(
    #         torch.full_like(
    #             fc, 2 * out_features
    #         ) * (
    #             torch.eye(out_features, 1, device=t.device, dtype=t.dtype) + 1
    #         )
    #     )

    # dct_m = 2 * norm * torch.cos(0.5 * PI * fc * (2 * t + 1) / out_features)

    # print(dct_m)

    # print(dct(np.eye(3), norm='ortho'))


    # dct conv check .....................................................................

    # in_features = 256 * 2 * 2
    # out_features = 4096
    # batch_size = 32

    # x2 = torch.nn.Parameter(torch.randn(batch_size,3,32,32))
    # dct_layer_conv1 = ConvDCT(in_channels=3,out_channels=64, kernel_size=3, stride=2,padding=1)

    # # dct_layer_conv2 = DCT_conv_layer(3,64,3,2,1)

    # y = dct_layer_conv1(x2)
    # print(y.shape)

    # # y2 = dct_layer_conv2(x2)
    # # print(y2.shape)

    # dct_layer_conv1_2 = ConvDCT(in_channels=64,out_channels=192,kernel_size=3,padding=1)

    # y1_2 = dct_layer_conv1_2(y)
    # print(y1_2.shape)

    # ...................................................................


    # DST matrix check..................................................................................

    # in_features, out_features = 512,512

    # t = torch.arange(in_features).reshape(1, -1)
    # fc = torch.arange(out_features ).reshape(-1, 1)
    # # print((t * fc).shape)

    # norm = torch.rsqrt(
    #         torch.full_like(
    #             fc, 2 * out_features
    #         ) * (
    #             torch.eye(out_features, 1) + 1
    #         )
    #     )
    # tmp = t * fc
    # dst_m = 2 * norm * np.sin(0.5 * np.pi * (fc + 1) * (2 * t + 1) / out_features)

    # dst_m[0] = dst_m[0]/np.sqrt(2)
    # print(dst_m.shape)

    
    # import matplotlib.pyplot as plt

    # plt.plot(dst_m[1])
    # plt.plot(dst_m[2])
    # plt.plot(dst_m[3])
    # plt.show()
    # .............................................................................


    # shape check .............................................

    # from torchviz import make_dot

    # batch_size = 32
    # in_features = 3
    # out_features = 64
    # kernel_size = 3


    # x = torch.nn.Parameter(torch.randn(batch_size, in_features, 32, 32))
    # x2 = torch.nn.Parameter(torch.randn(batch_size, 120))
    # conv_dct = Conv2dDCT(in_features,out_features, kernel_size, stride=2, padding=1)

    # lin_dct = LinearDFT(120,84)

    # y = conv_dct(x)
    # y2 = lin_dct(x2)

    # print(y.shape)
    # print(y2.shape)
    # make_dot(
    #     y2,
    #     params=dict(conv_dct.named_parameters()),
    #     show_saved=True
    # ).render('lin_DCT-dev_alexnet', format='png')

    # ............................................................

    # DFT matrix check ........................................................

    from scipy.linalg import dft

    in_features, out_features = 5,3

    t = torch.arange(in_features).reshape(1, -1, 1)
    # t = torch.linspace(-1.0, 1.0, in_features).reshape(1, -1, 1)
    fc = torch.arange(out_features ).reshape(-1, 1, 1)
    print((t * fc).shape)
    
    # w = torch.exp(torch.as_tensor((-2*np.pi*i)/N, dtype=torch.cfloat))

    norm = torch.rsqrt(
            torch.full_like(
                fc, in_features
            ) * (
                torch.ones(in_features, 1, device=t.device, dtype=t.dtype) 
            )
        )

    print(norm)

    dft_m = norm * torch.cat((torch.cos((fc*t*2*np.pi)/in_features), - torch.sin((fc*t*2*np.pi)/in_features)), dim=-1)

    # dft_m = dft_m / (math.sqrt(in_features)) ## normalize
    
    print(dft_m)

    print(dft(5,scale= 'sqrtn'))
    # print(dft_m[1,1])
    
    # print(dft_m[...,2])
    import matplotlib.pyplot as plt

    # plt.plot(dft_m[0,0])
    # plt.plot(dft_m[1,1])
    # plt.show()
    
    # print(dft(3))

    # print(dft(3) == W)

    # .............................................................................................................

    # LinearDFT check..............................................................................................

    in_features, out_features = 1024,4096
    batch_size = 32
    
    x = torch.randn(batch_size,out_features)

    # t = torch.linspace(-1.0, 1.0, in_features).reshape(1, -1, 1)
    # fc = torch.arange(out_features ).reshape(-1, 1, 1)

    # weights = torch.cat((torch.cos((fc*t*2*np.pi)/out_features), - torch.sin((fc*t*2*np.pi)/out_features)), dim=-1)

    # print(weights[..., 1].shape)

    # y = torch.stack((
    #                 F.linear(x, weights[..., 0]),
    #                 F.linear(x, weights[..., 1])
    #             ), dim=-1)  

    # print(y.shape)

    fbsp_layer = LinearFBSP(out_features=out_features)

    dft_layer = LinearDFT(in_features=in_features, out_features=out_features)
    # y = fbsp_layer(x)

    # print(y[1])

    y2 = dft_layer(x)

    print(y2.shape)

    dft_layer2 = LinearDFT(in_features=4096, out_features=4096)

    y22 = dft_layer2(y2)

    print(y22.shape)

    dft_layer3 = LinearDFT(4096,10)

    o = dft_layer3(y22)

    print(o.shape)
    
    # print((o[:,:,0]+o[:,:,1]).shape)

    # o2 = (o[:,:,0]*o[:,:,1])

    # o3 =(o[0,:,1]+o[1,:,1])
    print(o.mean(-1).shape)
    # print(o2.shape)
    # print(o3)
    # print(torch.stack((o2,o3)).shape)
    # print(ifft_x[:1])
    # plt.plot(x[0][0:10])
    # plt.show()
    # plt.plot(o2[:10].detach().numpy())
    # plt.plot(fft_x[:3])
    # plt.plot(ifft_x[:3])
    plt.show()
