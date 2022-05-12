
if __name__ == '__main__':
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
    from scipy.fftpack import dct
    import math
    from conv import Conv1dDCT
    from prototypes.DCT_layers import DCT_conv_layer, ConvDCT, LinearDCT, DCT_linear_layer
    ## cifar10 download problem solve
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    

    in_features = 256 * 2 * 2
    out_features = 4096
    batch_size = 32

    x = torch.nn.Parameter(torch.randn(3,64,64))

    x2 = torch.nn.Parameter(torch.randn(batch_size,3,32,32))
    dct_layer_conv1 = ConvDCT(in_channels=3,out_channels=64, kernel_size=3, stride=2,padding=1)

    # dct_layer_conv2 = DCT_conv_layer(3,64,3,2,1)

    y = dct_layer_conv1(x2)
    print(y.shape)

    # y2 = dct_layer_conv2(x2)
    # print(y2.shape)

    dct_layer_conv1_2 = ConvDCT(in_channels=64,out_channels=192,kernel_size=3,padding=1)

    y1_2 = dct_layer_conv1_2(y)
    print(y1_2.shape)

    # dct_layer_conv2_2 = DCT_conv_layer(64,192,kernel_size=3,padding=1)

    # y2_2 = dct_layer_conv2_2(y2)
    # print(y2_2.shape)

    # x = torch.nn.Parameter(torch.randn(batch_size,in_features))

    # linearDCT = LinearDCT(in_features=in_features, out_features=out_features)

    # dct_lin = DCT_linear_layer(out_features=out_features)

    # m = nn.Linear(in_features=in_features, out_features=out_features)

    # m_o = m(x)
    # # print(m_o.shape)
    # print('nn weight: ', m.weight.shape)
    
    # y = dct_lin(x)
    # print('dct_lin out: ', y.shape)

    # y2 = linearDCT(x)
    # print('linearDCT out: ',y2.shape)
