from models.LeNet.LeNet import LeNet
from models.LeNet.DFTLeNets import LeNetLinearDFT, LeNetConvDFT, LeNetDFT
from models.LeNet.DCTLeNets import LeNetLinearDCT, LeNetConvDCT, LeNetDCT
from models.LeNet.DSTLeNets import LeNetLinearDST, LeNetConvDST, LeNetDST
from models.LeNet.CWTLeNets import LeNetLinearCWT
from models.AlexNetCifar.DFTAlexNets import AlexNetLinearDFT, AlexNetDFT, AlexNetConvDFT
from models.AlexNetCifar.AlexNet import AlexNet
from models.AlexNetCifar.DCTAlexNets import AlexNetLinearDCT, AlexNetConvDCT, AlexNetDCT
from models.AlexNetCifar.DSTAlexNets import AlexNetConvDST, AlexNetDST, AlexNetLinearDST
from models.AlexNetCifar.CWTAlexNets import AlexNetLinearCWT
from models.ResNet.ResNet import resnet18
from models.ResNet.DCTResNets.ResNetDCT import resnet18DCT
from models.ResNet.DCTResNets.ResNetConvDCT import resnet18ConvDCT
from models.ResNet.DCTResNets.ResNetLinearDCT import resnet18LinearDCT
from models.ResNet.DSTResNets.ResNetDST import resnet18DST
from models.ResNet.DSTResNets.ResNetConvDST import resnet18ConvDST
from models.ResNet.DSTResNets.ResNetLinearDST import resnet18LinearDST
from models.ResNet.DFTResNets.ResNetConvDFT import resnet18ConvDFT
from models.ResNet.DFTResNets.ResNetLinearDFT import resnet18LinearDFT
from models.ResNet.DFTResNets.ResNetDFT import resnet18DFT

class ModelSelect():
    def __init__(self) -> None:
        pass

    def _alexnet(self, kernel, layers, num_classes):
        if kernel.casefold() == 'dct':
            if layers.casefold() == 'all' or layers == None:
                return AlexNetDCT(num_classes=num_classes), 'AlexNetDCT'
            if layers.casefold() == 'conv':
                return AlexNetConvDCT(num_classes=num_classes), 'AlexNetConvDCT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return AlexNetLinearDCT(num_classes=num_classes), 'AlexNetLinearDCT'

        if kernel.casefold() == 'dst':
            if layers.casefold() == 'all' or layers == None:
                return AlexNetDST(num_classes=num_classes), 'AlexNetDST'
            if layers.casefold() == 'conv':
                return AlexNetConvDST(num_classes=num_classes), 'AlexNetConvDST'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return AlexNetLinearDST(num_classes=num_classes), 'AlexNetLinearDST'
        
        if kernel.casefold() == 'dft':
            if layers.casefold() == 'all' or layers == None:
                return AlexNetDFT(num_classes=num_classes), 'AlexNetDFT'
            if layers.casefold() == 'conv':
                return AlexNetConvDFT(num_classes=num_classes), 'AlexNetConvDFT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return AlexNetLinearDFT(num_classes=num_classes), 'AlexNetLinearDFT'

        if kernel.casefold() == 'cwt':
            # if layers.casefold() == 'all' or layers == None:
            #     return AlexNetCWT(num_classes=num_classes), 'AlexNetCWT'
            # if layers.casefold() == 'conv':
            #     return AlexNetConvCWT(num_classes=num_classes), 'AlexNetConvCWT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return AlexNetLinearCWT(num_classes=num_classes), 'AlexNetLinearCWT'

        if kernel == None:
            return AlexNet(num_classes=num_classes), 'AlexNet'

    def _lenet(self, kernel, layers, num_classes):
        if kernel.casefold() == 'dct':
            if layers.casefold() == 'all' or layers == None:
                return LeNetDCT(num_classes=num_classes), 'LeNetDCT'
            if layers.casefold() == 'conv':
                return LeNetConvDCT(num_classes=num_classes), 'LeNetConvDCT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return LeNetLinearDCT(num_classes=num_classes), 'LeNetLinearDCT'

        if kernel.casefold() == 'dst':
            if layers.casefold() == 'all' or layers == None:
                return LeNetDST(num_classes=num_classes), 'LeNetDST'
            if layers.casefold() == 'conv':
                return LeNetConvDST(num_classes=num_classes), 'LeNetConvDST'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return LeNetLinearDST(num_classes=num_classes), 'LeNetLinearDST'
        
        if kernel.casefold() == 'dft':
            if layers.casefold() == 'all' or layers == None:
                return LeNetDFT(num_classes=num_classes), 'LeNetDFT'
            if layers.casefold() == 'conv':
                return LeNetConvDFT(num_classes=num_classes), 'LeNetConvDFT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return LeNetLinearDFT(num_classes=num_classes), 'LeNetLinearDFT'

        if kernel.casefold() == 'cwt':
            # if layers.casefold() == 'all' or layers == None:
            #     return LeNetCWT(num_classes=num_classes), 'LeNetCWT'
            # if layers.casefold() == 'conv':
            #     return LeNetConvCWT(num_classes=num_classes), 'LeNetConvCWT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return LeNetLinearCWT(num_classes=num_classes), 'LeNetLinearCWT'

        if kernel == None:
            return LeNet(num_classes=num_classes), 'LeNet'

    def _resnet18(self, kernel, layers, num_classes):
        if kernel.casefold() == 'dct':
            if layers.casefold() == 'all' or layers == None:
                return resnet18DCT(num_classes=num_classes), 'ResNet18DCT'
            if layers.casefold() == 'conv':
                return resnet18ConvDCT(num_classes=num_classes), 'ResNet18ConvDCT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return resnet18LinearDCT(num_classes=num_classes), 'ResNet18LinearDCT'

        if kernel.casefold() == 'dst':
            if layers.casefold() == 'all' or layers == None:
                return resnet18DST(num_classes=num_classes), 'ResNet18DST'
            if layers.casefold() == 'conv':
                return resnet18ConvDST(num_classes=num_classes), 'ResNet18ConvDST'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return resnet18LinearDST(num_classes=num_classes), 'ResNet18LinearDST'
        
        if kernel.casefold() == 'dft':
            if layers.casefold() == 'all' or layers == None:
                return resnet18DFT(num_classes=num_classes), 'ResNet18DFT'
            if layers.casefold() == 'conv':
                return resnet18ConvDFT(num_classes=num_classes), 'ResNet18ConvDFT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return resnet18LinearDFT(num_classes=num_classes), 'ResNet18LinearDFT'

        # if kernel.casefold() == 'cwt':
            # if layers.casefold() == 'all' or layers == None:
            #     return resnet18CWT(num_classes=num_classes), 'ResNet18CWT'
            # if layers.casefold() == 'conv':
            #     return resnet18ConvCWT(num_classes=num_classes), 'ResNet18ConvCWT'
            # if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
            #     return resnet18LinearCWT(num_classes=num_classes), 'ResNet18LinearCWT'

        if kernel == None:
            return resnet18(num_classes=num_classes), 'ResNet18'

    def getModel(self,model,kernel,layers,num_classes):

        if model.casefold() == 'alexnet':
            return self._alexnet(kernel,layers,num_classes)

        if model.casefold() == 'lenet':
            return self._lenet(kernel,layers,num_classes)

        if model.casefold() == 'resnet18':
            return self._resnet18(kernel,layers,num_classes)