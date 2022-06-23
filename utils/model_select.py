from models.TransformLayers.LinearFBSP import LinearFBSP
from math import pi as PI
from models.AlexNetCifar.DFTAlexNets import AlexNetLinearDFT, AlexNetDFT, AlexNetConvDFT
from models.AlexNetCifar.AlexNet import AlexNet
from models.AlexNetCifar.DCTAlexNets import AlexNetLinearDCT, AlexNetConvDCT, AlexNetDCT
from models.AlexNetCifar.DSTAlexNets import AlexNetConvDST, AlexNetDST, AlexNetLinearDST
from models.DenseNet121.DenseNet121 import DenseNet121
from models.DenseNet121.DCTDenseNets121 import DenseNet121ConvDCT, DenseNet121DCT, DenseNet121LinearDCT


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


def model_select_DenseNet121(kernel, layers, num_classes):

    if kernel == None:
        return DenseNet121(num_classes=num_classes), 'DenseNet121'

    if kernel == 'DCT' or kernel == 'dct':
        if layers == 'all' or layers == 'All' or layers == None:
            return DenseNet121DCT(num_classes=num_classes), 'DenseNet121DCT'
        if layers == 'conv' or layers == 'Conv':
            return DenseNet121ConvDCT(num_classes=num_classes), 'DenseNet121ConvDCT'
        if layers == 'Linear' or layers == 'linear' or layers == 'fc' or layers =='FC':
            return DenseNet121LinearDCT(num_classes=num_classes), 'DenseNet121LinearDCT'

    # if kernel == 'DST' or kernel == 'dst':
    #     if layers == 'all' or layers == 'All' or layers == None:
    #         return DenseNet121DST(num_classes=num_classes), 'DenseNet121DST'
    #     if layers == 'conv' or layers == 'Conv':
    #         return DenseNet121ConvDST(num_classes=num_classes), 'DenseNet121ConvDST'
    #     if layers == 'Linear' or layers == 'linear' or layers == 'fc' or layers == 'FC':
    #         return DenseNet121LinearDST(num_classes=num_classes), 'DenseNet121LinearDST'

    # if kernel == 'DFT' or kernel == 'dft':
    #     if layers == 'all' or layers == 'All' or layers == None:
    #         return DenseNet121DFT(num_classes=num_classes) , 'DenseNet121DFT'
    #     if layers == 'conv' or layers == 'Conv':
    #         return DenseNet121ConvDFT(num_classes=num_classes), 'DenseNet121ConvDFT'
    #     if layers == 'Linear' or layers == 'linear' or layers == 'fc' or layers == 'FC':
    #         return DenseNet121LinearDFT(num_classes=num_classes), 'DenseNet121LinearDFT'