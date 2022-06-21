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
from models.TransformLayers.DFT_layers import LinearDFT, Conv2dDFT
from models.TransformLayers.DST_layers import LinearDST, Conv2dDST
from models.TransformLayers.LinearFBSP import LinearFBSP
from math import pi as PI
from models.AlexNet.DFTAlexNets import AlexNetLinearDFT, AlexNetDFT, AlexNetConvDFT
from models.AlexNet.AlexNet import AlexNet
from models.AlexNet.DCTAlexNets import AlexNetLinearDCT, AlexNetConvDCT, AlexNetDCT
from models.AlexNet.DSTAlexNets import AlexNetConvDST, AlexNetDST, AlexNetLinearDST
from models.DenseNet121.DenseNet121 import DenseNet121


def model_select_AlexNet(kernel, layers, num_classes):

    if kernel == None:
        return AlexNet(num_classes=num_classes), 'AlexNet'

    if kernel == 'DCT' or kernel == 'dct':
        if layers == 'all' or layers == 'All' or layers == None:
            return AlexNetDCT(num_classes=num_classes), 'AlexNetDCT'
        if layers == 'conv' or layers == 'Conv':
            return AlexNetConvDCT(num_classes=num_classes), 'AlexNetConvDCT'
        if layers == 'Linear' or layers == 'linear' or layers == 'fc' or layers =='FC':
            return AlexNetLinearDCT(num_classes=num_classes), 'AlexNetLinearDCT'

    if kernel == 'DST' or kernel == 'dst':
        if layers == 'all' or layers == 'All' or layers == None:
            return AlexNetDST(num_classes=num_classes), 'AlexNetDST'
        if layers == 'conv' or layers == 'Conv':
            return AlexNetConvDST(num_classes=num_classes), 'AlexNetConvDST'
        if layers == 'Linear' or layers == 'linear' or layers == 'fc' or layers == 'FC':
            return AlexNetLinearDST(num_classes=num_classes), 'AlexNetLinearDST'

    if kernel == 'DFT' or kernel == 'dft':
        if layers == 'all' or layers == 'All' or layers == None:
            return AlexNetDFT(num_classes=num_classes) , 'AlexNetDFT'
        if layers == 'conv' or layers == 'Conv':
            return AlexNetConvDFT(num_classes=num_classes), 'AlexNetConvDFT'
        if layers == 'Linear' or layers == 'linear' or layers == 'fc' or layers == 'FC':
            return AlexNetLinearDFT(num_classes=num_classes), 'AlexNetLinearDFT'

def replace_conv2d(module, name, kernel):

        # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
        for attr_str in dir(module):
            target_attr = getattr(module, attr_str)
            if type(target_attr) == torch.nn.Conv2d:
                # print('replaced: ', name, attr_str)
                if kernel == "DFT":
                    new_conv = Conv2dDFT(target_attr.in_channels, target_attr.out_channels, target_attr.kernel_size, target_attr.stride,
                                                target_attr.padding, target_attr.dilation, target_attr.groups, target_attr.bias)
                if kernel == "DCT":
                    new_conv = Conv2dDCT(target_attr.in_channels, target_attr.out_channels, target_attr.kernel_size, target_attr.stride,
                                                target_attr.padding, target_attr.dilation, target_attr.groups, target_attr.bias)
                if kernel == "DST":
                    new_conv = Conv2dDST(target_attr.in_channels, target_attr.out_channels, target_attr.kernel_size, target_attr.stride,
                                                target_attr.padding, target_attr.dilation, target_attr.groups, target_attr.bias)
                
                setattr(module, attr_str, new_conv)

        # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
        for name, immediate_child_module in module.named_children():
            replace_conv2d(immediate_child_module, name, kernel)

def replace_linear(module, name, kernel):

        # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
        for attr_str in dir(module):
            target_attr = getattr(module, attr_str)
            if type(target_attr) == torch.nn.Linear:
                # print('replaced: ', name, attr_str)
                if kernel == "DFT":
                    new_lin = LinearDFT(target_attr.in_features, target_attr.out_features )
                if kernel == "DCT":
                    new_lin = LinearDCT(target_attr.in_features, target_attr.out_features)
                if kernel == "DST":
                    new_lin = LinearDST(target_attr.in_features, target_attr.out_features)
                
                setattr(module, attr_str, new_lin)

        # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
        for name, immediate_child_module in module.named_children():
            replace_conv2d(immediate_child_module, name, kernel)


def model_select_DenseNet121(kernel, layers, num_classes):

    if kernel == None:
        return DenseNet121(num_classes=num_classes), 'DenseNet121'

    # if kernel == 'DCT' or kernel == 'dct':
    #     if layers == 'all' or layers == 'All' or layers == None:
    #         return DenseNet121DCT(), 'DenseNet121DCT'
    #     if layers == 'conv' or layers == 'Conv':
    #         return DenseNet121ConvDCT(), 'DenseNet121ConvDCT'
    #     if layers == 'Linear' or layers == 'linear' or layers == 'fc' or layers =='FC':
    #         return DenseNet121LinearDCT(), 'DenseNet121LinearDCT'

    # if kernel == 'DST' or kernel == 'dst':
    #     if layers == 'all' or layers == 'All' or layers == None:
    #         return DenseNet121DST(), 'DenseNet121DST'
    #     if layers == 'conv' or layers == 'Conv':
    #         return DenseNet121ConvDST(), 'DenseNet121ConvDST'
    #     if layers == 'Linear' or layers == 'linear' or layers == 'fc' or layers == 'FC':
    #         return DenseNet121LinearDST(), 'DenseNet121LinearDST'

    # if kernel == 'DFT' or kernel == 'dft':
    #     if layers == 'all' or layers == 'All' or layers == None:
    #         return DenseNet121DFT() , 'DenseNet121DFT'
    #     if layers == 'conv' or layers == 'Conv':
    #         return DenseNet121ConvDFT(), 'DenseNet121ConvDFT'
    #     if layers == 'Linear' or layers == 'linear' or layers == 'fc' or layers == 'FC':
    #         return DenseNet121LinearDFT(), 'DenseNet121LinearDFT'