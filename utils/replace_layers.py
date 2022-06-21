import torch
from misc.conv import Conv1dDCT
from models.TransformLayers.DCT_layers import Conv2dDCT, LinearDCT
from models.TransformLayers.DFT_layers import LinearDFT, Conv2dDFT
from models.TransformLayers.DST_layers import LinearDST, Conv2dDST

## https://discuss.pytorch.org/t/how-to-replace-a-layer-with-own-custom-variant/43586/7

def replace_conv2d(module, name, kernel):

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

        for name, immediate_child_module in module.named_children():
            replace_conv2d(immediate_child_module, name, kernel)

def replace_linear(module, name, kernel):

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

        for name, immediate_child_module in module.named_children():
            replace_linear(immediate_child_module, name, kernel)


