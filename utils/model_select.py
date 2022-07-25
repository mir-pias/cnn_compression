import imp
from models.TransformLayers.LinearFBSP import LinearFBSP
from models.AlexNetCifar.DFTAlexNets import AlexNetLinearDFT, AlexNetDFT, AlexNetConvDFT
from models.AlexNetCifar.AlexNet import AlexNet
from models.AlexNetCifar.DCTAlexNets import AlexNetLinearDCT, AlexNetConvDCT, AlexNetDCT
from models.AlexNetCifar.DSTAlexNets import AlexNetConvDST, AlexNetDST, AlexNetLinearDST
from models.DenseNetCifar.DenseNet import DenseNet
from models.DenseNetCifar.DCTDenseNets import DenseNetConvDCT, DenseNetDCT, DenseNetLinearDCT
from models.AlexNetCifar.CWTAlexNets import AlexNetLinearCWT
from models.LeNet.LeNet import LeNet
from models.LeNet.DFTLeNets import LeNetLinearDFT, LeNetConvDFT, LeNetDFT
from models.LeNet.DCTLeNets import LeNetLinearDCT, LeNetConvDCT, LeNetDCT
from models.LeNet.DSTLeNets import LeNetLinearDST, LeNetConvDST, LeNetDST
from models.LeNet.CWTLeNets import LeNetLinearCWT
from models.ResNet.ResNet import resnet18
from models.ResNet.DCTResNets.ResNetDCT import resnet18DCT
from models.ResNet.DCTResNets.ResNetConvDCT import resnet18ConvDCT
from models.ResNet.DCTResNets.ResNetLinearDCT import resnet18LinearDCT
from models.ResNet.DSTResNets.ResNetDST import resnet18DST
from models.ResNet.DSTResNets.ResNetConvDST import resnet18ConvDST
from models.ResNet.DSTResNets.ResNetLinearDST import resnet18LinearDST

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

    if kernel == 'CWT' or kernel == 'cwt':
        # if layers == 'all' or layers == 'All' or layers == None:
            # return AlexNetDFT(num_classes=num_classes) , 'AlexNetDFT'
        # if layers == 'conv' or layers == 'Conv':
            # return AlexNetConvDFT(num_classes=num_classes), 'AlexNetConvDFT'
        if layers == 'Linear' or layers == 'linear' or layers == 'fc' or layers == 'FC':
            return AlexNetLinearCWT(num_classes=num_classes), 'AlexNetLinearCWT'


def model_select_DenseNet(kernel, layers, num_classes):

    if kernel == None:
        return DenseNet(num_classes=num_classes), 'DenseNet'

    if kernel == 'DCT' or kernel == 'dct':
        if layers == 'all' or layers == 'All' or layers == None:
            return DenseNetDCT(num_classes=num_classes), 'DenseNetDCT'
        if layers == 'conv' or layers == 'Conv':
            return DenseNetConvDCT(num_classes=num_classes), 'DenseNetConvDCT'
        if layers == 'Linear' or layers == 'linear' or layers == 'fc' or layers =='FC':
            return DenseNetLinearDCT(num_classes=num_classes), 'DenseNetLinearDCT'

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


def model_select_LeNet(kernel, layers, num_classes):

    if kernel == None:
        return LeNet(num_classes=num_classes), 'LeNet'

    if kernel == 'DCT' or kernel == 'dct':
        if layers == 'all' or layers == 'All' or layers == None:
            return LeNetDCT(num_classes=num_classes), 'LeNetDCT'
        if layers == 'conv' or layers == 'Conv':
            return LeNetConvDCT(num_classes=num_classes), 'LeNetConvDCT'
        if layers == 'Linear' or layers == 'linear' or layers == 'fc' or layers =='FC':
            return LeNetLinearDCT(num_classes=num_classes), 'LeNetLinearDCT'

    if kernel == 'DST' or kernel == 'dst':
        if layers == 'all' or layers == 'All' or layers == None:
            return LeNetDST(num_classes=num_classes), 'LeNetDST'
        if layers == 'conv' or layers == 'Conv':
            return LeNetConvDST(num_classes=num_classes), 'LeNetConvDST'
        if layers == 'Linear' or layers == 'linear' or layers == 'fc' or layers == 'FC':
            return LeNetLinearDST(num_classes=num_classes), 'LeNetLinearDST'

    if kernel == 'DFT' or kernel == 'dft':
        if layers == 'all' or layers == 'All' or layers == None:
            return LeNetDFT(num_classes=num_classes) , 'LeNetDFT'
        if layers == 'conv' or layers == 'Conv':
            return LeNetConvDFT(num_classes=num_classes), 'LeNetConvDFT'
        if layers == 'Linear' or layers == 'linear' or layers == 'fc' or layers == 'FC':
            return LeNetLinearDFT(num_classes=num_classes), 'LeNetLinearDFT'

    if kernel == 'CWT' or kernel == 'cwt':
        # if layers == 'all' or layers == 'All' or layers == None:
            # return AlexNetDFT(num_classes=num_classes) , 'AlexNetDFT'
    #     # if layers == 'conv' or layers == 'Conv':
    #         # return AlexNetConvDFT(num_classes=num_classes), 'AlexNetConvDFT'
        if layers == 'Linear' or layers == 'linear' or layers == 'fc' or layers == 'FC':
            return LeNetLinearCWT(num_classes=num_classes), 'LeNetLinearCWT'


def model_select_ResNet18(kernel, layers, num_classes):

    if kernel == None:
        return resnet18(num_classes=num_classes), 'ResNet18'

    if kernel == 'DCT' or kernel == 'dct':
        if layers == 'all' or layers == 'All' or layers == None:
            return resnet18DCT(num_classes=num_classes), 'ResNet18DCT'
        if layers == 'conv' or layers == 'Conv':
            return resnet18ConvDCT(num_classes=num_classes), 'ResNet18ConvDCT'
        if layers == 'Linear' or layers == 'linear' or layers == 'fc' or layers =='FC':
            return resnet18LinearDCT(num_classes=num_classes), 'ResNet18LinearDCT'

    if kernel == 'DST' or kernel == 'dst':
        if layers == 'all' or layers == 'All' or layers == None:
            return resnet18DST(num_classes=num_classes), 'ResNet18DST'
        if layers == 'conv' or layers == 'Conv':
            return resnet18ConvDST(num_classes=num_classes), 'ResNet18ConvDST'
        if layers == 'Linear' or layers == 'linear' or layers == 'fc' or layers =='FC':
            return resnet18LinearDST(num_classes=num_classes), 'ResNet18LinearDST'


    # if kernel == 'DFT' or kernel == 'dft':
    #     if layers == 'all' or layers == 'All' or layers == None:
    #         return DenseNet121DFT(num_classes=num_classes) , 'DenseNet121DFT'
    #     if layers == 'conv' or layers == 'Conv':
    #         return DenseNet121ConvDFT(num_classes=num_classes), 'DenseNet121ConvDFT'
    #     if layers == 'Linear' or layers == 'linear' or layers == 'fc' or layers == 'FC':
    #         return DenseNet121LinearDFT(num_classes=num_classes), 'DenseNet121LinearDFT'
