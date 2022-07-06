import sys

from numpy import dtype  
sys.path.append('.')

import numpy as np
import torch
from torch import dropout, nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from scipy.fftpack import dct, fft , ifft
import math
from misc.conv import Conv1dDCT
from models.TransformLayers.DCT_layers import Conv2dDCT, LinearDCT
from models.TransformLayers.DFT_layers import LinearDFT, Conv2dDFT
from models.TransformLayers.DST_layers import LinearDST, Conv2dDST
from models.TransformLayers.LinearFBSP import LinearFBSP
from math import pi as PI
import matplotlib.pyplot as plt

if __name__ == '__main__':
    

    ## dct matrix check

    # in_channels, out_features = 3,3

    # t = torch.arange(in_channels).reshape(1, -1)
    # # t = torch.linspace(-1.0, 1.0, in_channels).reshape(1, -1, 1)
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

    # in_channels = 3
    # out_features = 64
    # batch_size = 32

    # x2 = torch.nn.Parameter(torch.randn(batch_size,3,32,32))
    # dct_layer_conv1 = Conv2dDCT(in_channels=3,out_channels=64, kernel_size=3, stride=2,padding=1)

    # y = dct_layer_conv1(x2)
    # print(y.shape)

    # dct_layer_conv1_2 = Conv2dDCT(in_channels=64,out_channels=192,kernel_size=(2,3),padding=1)

    # y1_2 = dct_layer_conv1_2(y)
    # print(y1_2.shape)
    # print('................................................................................')
    # in_channels = 3
    # out_channels = 4
    # kernel_size = (3,2)

    # fcc = torch.arange(out_channels ).reshape(-1, 1)

    # fcl = torch.arange(kernel_size[0]).reshape(-1, 1)

    # tc = torch.arange(in_channels).reshape(1, -1)

    # t_l = torch.arange(kernel_size[1]).reshape(1, -1)

    # norm_c = torch.rsqrt(
    #     torch.full_like(
    #         fcc, 2 * out_channels
    #     ) * (
    #         torch.eye(out_channels, 1) + 1
    #     )
    # )
    # # print((t_l*fcl).shape)

    # kc: torch.Tensor = 2 * norm_c * torch.cos(0.5 * PI * fcc * (2 * tc + 1) / out_channels)

    # norm_l = torch.rsqrt(
    #     torch.full_like(
    #         fcl, 2 * kernel_size[0]
    #     ) * (
    #         torch.eye(kernel_size[0], 1) + 1
    #     )
    # )
    # # print('t_l shape: ', t_l.shape)
    # # print('tc shape: ', tc.shape)
    # # print(norm_l)

    # kl: torch.Tensor = 2 * norm_l * torch.cos(0.5 * PI * fcl * (2 * t_l + 1) / kernel_size[0])

    # # print('kc_reshape:', kc.reshape(
    # #     out_channels, -1, 1,1
    # # ).shape)
    # # print('kl_reshape:', kl.reshape(
    # #     1,1, -1, kernel_size[0]
    # # ).shape)

    # print('kc_shape: ', kc.shape)
    # print('kl_shape: ', kl.shape)

    # w: torch.Tensor = kc.reshape(
    #     out_channels, -1, 1,1
    # ) * kl.reshape(
    #     1,1, -1, kernel_size[1]
    # )
    # print(kc.reshape(
    #     out_channels, -1, 1,1
    # ).shape)

    # print(kl.reshape(
    #     1,1, -1, kernel_size[1]
    # ).shape)

    # print(w.shape)
    
    # dct_1 = dct(np.eye(3,4),norm='ortho')

    # # print(dct_1.shape)

    # dct_2 = dct(np.eye(3,2),norm='ortho')

    # # print(dct_2.shape)

    # w2 = dct_1.reshape(
    #     out_channels, -1, 1
    # ) * dct_2.reshape(
    #     1, -1, kernel_size[1]
    # )

    # print(w2.shape)


    # ...................................................................


    # DST matrix check..................................................................................

    # in_channels, out_features = 512,512

    # t = torch.arange(in_channels).reshape(1, -1)
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
    # in_channels = 3
    # out_features = 64
    # kernel_size = 3


    # x = torch.nn.Parameter(torch.randn(batch_size, in_channels, 32, 32))
    # x2 = torch.nn.Parameter(torch.randn(batch_size, 120))
    # conv_dct2 = Conv2dDCT(in_channels,out_features, kernel_size, stride=2, padding=1)

    # lin_dct = LinearDFT(120,84)

    # y = conv_dct2(x)
    # y2 = lin_dct(x2)

    # print(y.shape)
    # print(y2.shape)
    # make_dot(
    #     y2,
    #     params=dict(conv_dct2.named_parameters()),
    #     show_saved=True
    # ).render('lin_DCT-dev_alexnet', format='png')

    # ............................................................

    # DFT matrix check ........................................................

    # from scipy.linalg import dft

    # in_channels, out_features = 5,3

    # t = torch.arange(in_channels).reshape(1, -1, 1)
    # # t = torch.linspace(-1.0, 1.0, in_channels).reshape(1, -1, 1)
    # fc = torch.arange(out_features ).reshape(-1, 1, 1)
    # print((t * fc).shape)
    
    # # w = torch.exp(torch.as_tensor((-2*np.pi*i)/N, dtype=torch.cfloat))

    # norm = torch.rsqrt(
    #         torch.full_like(
    #             fc, in_channels
    #         ) * (
    #             torch.ones(in_channels, 1, device=t.device, dtype=t.dtype) 
    #         )
    #     )

    # print(norm)

    # dft_m = norm * torch.cat((torch.cos((fc*t*2*np.pi)/in_channels), - torch.sin((fc*t*2*np.pi)/in_channels)), dim=-1)

    # # dft_m = dft_m / (math.sqrt(in_channels)) ## normalize
    
    # print(dft_m)

    # print(dft(5,scale= 'sqrtn'))
    # # print(dft_m[1,1])
    
    # # print(dft_m[...,2])
    # import matplotlib.pyplot as plt

    # # plt.plot(dft_m[0,0])
    # # plt.plot(dft_m[1,1])
    # # plt.show()
    
    # # print(dft(3))

    # # print(dft(3) == W)

    # # .............................................................................................................

    # # LinearDFT check..............................................................................................

    # in_features, out_features = 1024,4096
    # batch_size = 32
    
    # x = torch.randn(batch_size,in_features)

    # # t = torch.linspace(-1.0, 1.0, in_channels).reshape(1, -1, 1)
    # # fc = torch.arange(out_features ).reshape(-1, 1, 1)

    # # w = torch.cat((torch.cos((fc*t*2*np.pi)/out_features), - torch.sin((fc*t*2*np.pi)/out_features)), dim=-1)

    # # print(w[..., 1].shape)

    # # y = torch.stack((
    # #                 F.linear(x, w[..., 0]),
    # #                 F.linear(x, w[..., 1])
    # #             ), dim=-1)  

    # # print(y.shape)

    # fbsp_layer = LinearFBSP(out_features=out_features)

    # dft_layer = LinearDFT(in_features=in_features, out_features=out_features)
    # # y = fbsp_layer(x)

    # # print(y[1])

    # y2 = dft_layer(x)

    # print(y2.shape)

    # dft_layer2 = LinearDFT(in_features=4096, out_features=4096)

    # y22 = dft_layer2(y2)

    # print(y22.shape)

    # dft_layer3 = LinearDFT(4096,10)

    # o = dft_layer3(y22)

    # print(o[0])

    # print(o[0].sum(-1))
    
    # print('................................................................................')
    # # print((o[:,:,0]+o[:,:,1]).shape)

    # # o2 = (o[:,:,0]*o[:,:,1])

    # # o3 =(o[0,:,1]+o[1,:,1])
    # print(o.mean(-1).shape)
    # # print(o2.shape)
    # # print(o3)
    # # print(torch.stack((o2,o3)).shape)
    # # print(ifft_x[:1])
    # # plt.plot(x[0][0:10])
    # # plt.show()
    # # plt.plot(o2[:10].detach().numpy())
    # # plt.plot(fft_x[:3])
    # # plt.plot(ifft_x[:3])
    # plt.show()

    ## conv2dDFT check
    
    # in_channels = 3
    # out_channels = 64
    # kernel_size =(7,3)

    # fcc = torch.arange(out_channels ).reshape(-1,1, 1)

    # fcl = torch.arange(kernel_size[0]).reshape(-1,1, 1)

    # tc = torch.arange(in_channels).reshape(1, -1,1)

    # tl = torch.arange(kernel_size[1]).reshape(1, -1,1)

    # norm_c = torch.rsqrt(
    #         torch.full_like(
    #             fcc, in_channels
    #         ) * (
    #             torch.ones(in_channels, 1) 
    #         )
    #     )

    # kc: torch.Tensor = 2 * norm_c * torch.cat((torch.cos((fcc*tc*2*PI)/in_channels), - torch.sin((fcc*tc*2*PI)/in_channels)), dim=-1) 

    # norm_l = torch.rsqrt(
    #         torch.full_like(
    #             fcl, kernel_size[0]
    #         ) * (
    #             torch.ones(kernel_size[1], 1) 
    #         )
    #     )
    # # print('t_l shape: ', tl.shape)
    # # print('tc shape: ', tc.shape)
    # # print('fcl shape: ', fcl.shape)
    # # print(norm_l.shape)
    # # print((fcl*tl).shape)

    # kl: torch.Tensor = 2 * norm_l * torch.cat((torch.cos((fcl*tl*2*PI)/kernel_size[0]), - torch.sin((fcl*tl*2*PI)/kernel_size[0])), dim=-1)

    # # print(torch.cat((torch.cos((fcl*tl*2*PI)/kernel_size[0]), - torch.sin((fcl*tl*2*PI)/kernel_size[0])), dim=-1).shape)
    # # print('kc_reshape:', kc.reshape(
    # #     out_channels, -1, 1,1
    # # ).shape)
    # # print('kl_reshape:', kl.reshape(
    # #     1,1, -1, kernel_size[0]
    # # ).shape)

    # # print('kc_shape: ', kc.shape)
    # # print('kl_shape: ', kl.shape)


    # # print(kc.reshape(
    #     # out_channels, -1,kc.shape[-1],1,1
    # # ).shape)

    # # print(kl.reshape(
    # #     1,1, kl.shape[-1],-1,  kernel_size[1]
    # # ).shape)

    # w: torch.Tensor = kc.reshape(
    #     out_channels, -1,kc.shape[-1],1,1
    # ) * kl.reshape(
    #     1,1, kl.shape[-1], -1, kernel_size[0]
    # )

    # print(w.shape)
    # print(w.transpose(2,4).shape)

    # w = w.transpose(2,4)

    # from models.TransformLayers.DFT_layers import Conv2dDFT

    # batch_size = 32

    # x = torch.nn.Parameter(torch.randn(batch_size,3,32,32))
    
    # # print(x.shape)
    # y = torch.stack((
    #                 F.conv2d(x, w[..., 0],stride=2,padding=1) ,
    #                 F.conv2d(x, w[..., 1], stride=2,padding=1)), dim=-1)

    # # print(y.shape)


    # ConvDFT_1 = Conv2dDFT(in_channels,out_channels, kernel_size, stride=2, padding=1)
    # conv = nn.Conv2d(in_channels,out_channels, kernel_size, stride=2, padding=1)
    
    # y1 = ConvDFT_1(x)
    # # print(y1.shape)
    # # print(conv(x).shape)

    # # m = nn.MaxPool2d(kernel_size=2)
    # # print(m(y1).shape)

    # ConvDFT_2 = Conv2dDFT(64, 192, kernel_size=(3,2), padding=1)
    # conv2 = nn.Conv2d(64, 192, kernel_size=(3,2), padding=1)

    # # y2 = ConvDFT_2(m(y1))


    # print(y2.shape)
    # print(conv2(m(y1)).shape)
    ## .............................................................................................


    # import torchvision.models as models

    # model = models.DenseNet(drop_rate=0.2)
    # # print(model)

    # from utils.replace_layers import replace_conv2d, replace_linear

    # # replace_conv2d(model, 'model','DCT')
    # # replace_conv2d(model, 'model')
    
    # for name, module in model.features.named_children():
    #     if 'denseblock1' in name:
    #         replace_conv2d(module, 'module','DCT')
    #         print(module)

    # print(model)
    # mod = models.densenet121(num_classes=10)
    
    # replace_conv2d(mod,'mod',kernel='DFT')
    # replace_linear(mod,'mod','DFT')

    # print(mod.classifier)
    ##...............................................

    in_features, out_features = 1024,4096
    batch_size = 32

    from typing import cast

    def sinc(x: torch.Tensor) -> torch.Tensor:
        return torch.where(cast(torch.Tensor, x == 0), torch.ones_like(x), torch.sin(x) / x)
    
    x = torch.randn(batch_size,in_features)

    # print(x.shape[-1])
    t = torch.linspace(-1.0 , 1.0, x.shape[-1]).reshape(1, -1)
    a = torch.arange(1, out_features+1 ).reshape(-1, 1)
    b = torch.arange(out_features ).reshape(-1, 1)

    psi = torch.sinc(t/2) * torch.cos((3*PI*t)/2)
    w = torch.rsqrt(a) * psi * ((t-b)/a)

    # print(psi.shape)
    # print(w.shape)
    # plt.imshow((w.T @ w).detach().numpy())
    # plt.show()
    # # plt.plot(psi[0])
    # # plt.plot(psi[11])
    # # plt.show()
    
    # y = F.linear(x,w)
    # print(y.shape)

    # print((sinc(2*t) - sinc(t)).shape)
    
    # plt.plot((2*sinc(PI*2*t) - sinc(PI*t))[0])
    # plt.show()

    from models.TransformLayers.CWT_layers import LinearCWT

    lin_cwt = LinearCWT(in_features,out_features)

    # p = lin_cwt.materialize_weights(x)

    # print(p.shape)
    ##...........................................................................................................

    lin_dct = LinearDCT(in_features,out_features)

    lin_dft = LinearDFT(in_features,out_features)

    lin_dst = LinearDST(in_features,out_features)

    weights = lin_cwt.materialize_weights(x)
    
    check = torch.linalg.lstsq(weights,weights).solution

    # ch = torch.linalg.lstsq(weights[0][:,:,0],weights[0][:,:,0]).solution ## for dft

    # print('linear weights: ', weights.shape)

    # plt.plot(weights[2].detach().numpy())
    # plt.show()
    # print(torch.diagonal(check,0))

    # print(weights[0][0:3])
    # plt.imshow(check.detach().numpy())
    # plt.show()

    ##...............

    from models.TransformLayers.conv2d_dct import Conv2dDCT as dct2
    from models.TransformLayers.conv2d_dst import Conv2dDST as dst2

    x2 = torch.nn.Parameter(torch.randn(batch_size,3,32,32))
    
    conv_dct2 = dst2(3,64,(3,3),stride=2,padding=1)
    conv_dct = Conv2dDST(3,64,(3,3),stride=2,padding=1)
    

    conv_weight2 = conv_dct2._materialize_weights(x2)
    conv_weight = conv_dct._materialize_weights(x2)


    # plt.plot(conv_weight2[0][0][2].detach().numpy())
    # plt.show()

    print(conv_weight2.shape)
    print(conv_weight.shape)
    
    
    ortho2 = torch.linalg.lstsq(conv_weight2[1][1],conv_weight2[1][1]).solution
    ortho = torch.linalg.lstsq(conv_weight[1][1],conv_weight[1][1]).solution

    plt.imshow(ortho2.detach().numpy())
    plt.show()

    # print(ortho2)

    plt.imshow(ortho.detach().numpy())
    plt.show()

    # print(ortho)

    # sp_dct = dct(np.eye(3)) * dct(np.eye(3)) 

    # # print(weights[0:2][0:3])
    # # print(sp_dct.T @ sp_dct) 
    # # plt.imshow(sp_dct.T @ sp_dct)
    # # plt.show()

    # # plt.plot(sp_dct[2])
    # # plt.show()

    print(conv_weight2[1][1][0][:].T @ conv_weight2[1][1][-1][:])
    print(conv_weight[1][1][0][:].T @ conv_weight[1][1][-1][:])