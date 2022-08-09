from models.LeNet.LeNet import LeNet
from models.LeNet.DFTLeNets import LeNetLinearDFT, LeNetConvDFT, LeNetDFT
from models.LeNet.DCTLeNets import LeNetLinearDCT, LeNetConvDCT, LeNetDCT
from models.LeNet.DSTLeNets import LeNetLinearDST, LeNetConvDST, LeNetDST
from models.LeNet.ShanLeNets import LeNetLinearShan, LeNetConvShan, LeNetShan
from models.AlexNet.DFTAlexNets import AlexNetLinearDFT, AlexNetDFT, AlexNetConvDFT
from models.AlexNet.AlexNet import AlexNet
from models.AlexNet.DCTAlexNets import AlexNetLinearDCT, AlexNetConvDCT, AlexNetDCT
from models.AlexNet.DSTAlexNets import AlexNetConvDST, AlexNetDST, AlexNetLinearDST
from models.AlexNet.ShanAlexNets import AlexNetLinearShan, AlexNetConvShan, AlexNetShan
from models.ResNet.ResNet import resnet18, resnet50
from models.ResNet.DCTResNets.ResNetDCT import resnet18DCT, resnet50DCT
from models.ResNet.DCTResNets.ResNetConvDCT import resnet18ConvDCT, resnet50ConvDCT
from models.ResNet.DCTResNets.ResNetLinearDCT import resnet18LinearDCT, resnet50LinearDCT
from models.ResNet.DSTResNets.ResNetDST import resnet18DST, resnet50DST
from models.ResNet.DSTResNets.ResNetConvDST import resnet18ConvDST, resnet50ConvDST
from models.ResNet.DSTResNets.ResNetLinearDST import resnet18LinearDST, resnet50LinearDST
from models.ResNet.DFTResNets.ResNetConvDFT import resnet18ConvDFT, resnet50ConvDFT
from models.ResNet.DFTResNets.ResNetLinearDFT import resnet18LinearDFT, resnet50LinearDFT
from models.ResNet.DFTResNets.ResNetDFT import resnet18DFT, resnet50DFT
from models.DenseNet.DenseNet import densenet121, densenet201
from models.DenseNet.DCTDenseNets.DenseNetConvDCT import densenet121ConvDCT, densenet201ConvDCT
from models.DenseNet.DCTDenseNets.DenseNetDCT import densenet121DCT, densenet201DCT
from models.DenseNet.DCTDenseNets.DenseNetLinearDCT import densenet121LinearDCT, densenet201LinearDCT
from models.DenseNet.DSTDenseNets.DenseNetConvDST import densenet121ConvDST, densenet201ConvDST
from models.DenseNet.DSTDenseNets.DenseNetDST import densenet121DST, densenet201DST
from models.DenseNet.DSTDenseNets.DenseNetLinearDST import densenet121LinearDST, densenet201LinearDST
from models.DenseNet.DFTDenseNets.DenseNetConvDFT import densenet121ConvDFT, densenet201ConvDFT
from models.DenseNet.DFTDenseNets.DenseNetLinearDFT import densenet121LinearDFT, densenet201LinearDFT
from models.DenseNet.DFTDenseNets.DenseNetDFT import densenet121DFT, densenet201DFT

class ModelSelect():
    def __init__(self) -> None:
        pass

    def _alexnet(self, kernel, layers, num_classes, in_channels):
        if kernel == None:
            return AlexNet(num_classes=num_classes,in_channels=in_channels), 'AlexNet'

        if kernel.casefold() == 'dct':
            if layers.casefold() == 'all' or layers == None:
                return AlexNetDCT(num_classes=num_classes,in_channels=in_channels), 'AlexNetDCT'
            if layers.casefold() == 'conv':
                return AlexNetConvDCT(num_classes=num_classes,in_channels=in_channels), 'AlexNetConvDCT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return AlexNetLinearDCT(num_classes=num_classes,in_channels=in_channels), 'AlexNetLinearDCT'

        if kernel.casefold() == 'dst':
            if layers.casefold() == 'all' or layers == None:
                return AlexNetDST(num_classes=num_classes,in_channels=in_channels), 'AlexNetDST'
            if layers.casefold() == 'conv':
                return AlexNetConvDST(num_classes=num_classes,in_channels=in_channels), 'AlexNetConvDST'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return AlexNetLinearDST(num_classes=num_classes,in_channels=in_channels), 'AlexNetLinearDST'
        
        if kernel.casefold() == 'dft':
            if layers.casefold() == 'all' or layers == None:
                return AlexNetDFT(num_classes=num_classes,in_channels=in_channels), 'AlexNetDFT'
            if layers.casefold() == 'conv':
                return AlexNetConvDFT(num_classes=num_classes,in_channels=in_channels), 'AlexNetConvDFT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return AlexNetLinearDFT(num_classes=num_classes,in_channels=in_channels), 'AlexNetLinearDFT'

        if kernel.casefold() == 'Shan':
            if layers.casefold() == 'all' or layers == None:
                return AlexNetShan(num_classes=num_classes,in_channels=in_channels), 'AlexNetShan'
            if layers.casefold() == 'conv':
                return AlexNetConvShan(num_classes=num_classes,in_channels=in_channels), 'AlexNetConvShan'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return AlexNetLinearShan(num_classes=num_classes,in_channels=in_channels), 'AlexNetLinearShan'

        

    def _lenet(self, kernel, layers, num_classes,in_channels):
        if kernel == None:
            return LeNet(num_classes=num_classes, in_channels=in_channels), 'LeNet'

        if kernel.casefold() == 'dct':
            if layers.casefold() == 'all' or layers == None:
                return LeNetDCT(num_classes=num_classes, in_channels=in_channels), 'LeNetDCT'
            if layers.casefold() == 'conv':
                return LeNetConvDCT(num_classes=num_classes,in_channels=in_channels), 'LeNetConvDCT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return LeNetLinearDCT(num_classes=num_classes,in_channels=in_channels), 'LeNetLinearDCT'

        if kernel.casefold() == 'dst':
            if layers.casefold() == 'all' or layers == None:
                return LeNetDST(num_classes=num_classes,in_channels=in_channels), 'LeNetDST'
            if layers.casefold() == 'conv':
                return LeNetConvDST(num_classes=num_classes,in_channels=in_channels), 'LeNetConvDST'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return LeNetLinearDST(num_classes=num_classes,in_channels=in_channels), 'LeNetLinearDST'
        
        if kernel.casefold() == 'dft':
            if layers.casefold() == 'all' or layers == None:
                return LeNetDFT(num_classes=num_classes,in_channels=in_channels), 'LeNetDFT'
            if layers.casefold() == 'conv':
                return LeNetConvDFT(num_classes=num_classes,in_channels=in_channels), 'LeNetConvDFT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return LeNetLinearDFT(num_classes=num_classes,in_channels=in_channels), 'LeNetLinearDFT'

        if kernel.casefold() == 'Shan':
            if layers.casefold() == 'all' or layers == None:
                return LeNetShan(num_classes=num_classes,in_channels=in_channels), 'LeNetShan'
            if layers.casefold() == 'conv':
                return LeNetConvShan(num_classes=num_classes,in_channels=in_channels), 'LeNetConvShan'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return LeNetLinearShan(num_classes=num_classes,in_channels=in_channels), 'LeNetLinearShan'

        

    def _resnet18(self, kernel, layers, num_classes, in_channels):
        if kernel == None:
            return resnet18(num_classes=num_classes, in_channels=in_channels), 'ResNet18'

        if kernel.casefold() == 'dct':
            if layers.casefold() == 'all' or layers == None:
                return resnet18DCT(num_classes=num_classes,in_channels=in_channels), 'ResNet18DCT'
            if layers.casefold() == 'conv':
                return resnet18ConvDCT(num_classes=num_classes, in_channels=in_channels), 'ResNet18ConvDCT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return resnet18LinearDCT(num_classes=num_classes, in_channels=in_channels), 'ResNet18LinearDCT'

        if kernel.casefold() == 'dst':
            if layers.casefold() == 'all' or layers == None:
                return resnet18DST(num_classes=num_classes, in_channels=in_channels), 'ResNet18DST'
            if layers.casefold() == 'conv':
                return resnet18ConvDST(num_classes=num_classes, in_channels=in_channels), 'ResNet18ConvDST'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return resnet18LinearDST(num_classes=num_classes, in_channels=in_channels), 'ResNet18LinearDST'
        
        if kernel.casefold() == 'dft':
            if layers.casefold() == 'all' or layers == None:
                return resnet18DFT(num_classes=num_classes, in_channels=in_channels), 'ResNet18DFT'
            if layers.casefold() == 'conv':
                return resnet18ConvDFT(num_classes=num_classes, in_channels=in_channels), 'ResNet18ConvDFT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return resnet18LinearDFT(num_classes=num_classes, in_channels=in_channels), 'ResNet18LinearDFT'

        # if kernel.casefold() == 'cwt':
            # if layers.casefold() == 'all' or layers == None:
            #     return resnet18CWT(num_classes=num_classes, in_channels=in_channels), 'ResNet18CWT'
            # if layers.casefold() == 'conv':
            #     return resnet18ConvCWT(num_classes=num_classes, in_channels=in_channels), 'ResNet18ConvCWT'
            # if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
            #     return resnet18LinearCWT(num_classes=num_classes, in_channels=in_channels), 'ResNet18LinearCWT'

    def _resnet50(self, kernel, layers, num_classes, in_channels):
        if kernel == None:
            return resnet50(num_classes=num_classes, in_channels=in_channels), 'ResNet50'

        if kernel.casefold() == 'dct':
            if layers.casefold() == 'all' or layers == None:
                return resnet50DCT(num_classes=num_classes,in_channels=in_channels), 'ResNet50DCT'
            if layers.casefold() == 'conv':
                return resnet50ConvDCT(num_classes=num_classes, in_channels=in_channels), 'ResNet50ConvDCT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return resnet50LinearDCT(num_classes=num_classes, in_channels=in_channels), 'ResNet50LinearDCT'

        if kernel.casefold() == 'dst':
            if layers.casefold() == 'all' or layers == None:
                return resnet50DST(num_classes=num_classes, in_channels=in_channels), 'ResNet50DST'
            if layers.casefold() == 'conv':
                return resnet50ConvDST(num_classes=num_classes, in_channels=in_channels), 'ResNet50ConvDST'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return resnet50LinearDST(num_classes=num_classes, in_channels=in_channels), 'ResNet50LinearDST'
        
        if kernel.casefold() == 'dft':
            if layers.casefold() == 'all' or layers == None:
                return resnet50DFT(num_classes=num_classes, in_channels=in_channels), 'ResNet50DFT'
            if layers.casefold() == 'conv':
                return resnet50ConvDFT(num_classes=num_classes, in_channels=in_channels), 'ResNet50ConvDFT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return resnet50LinearDFT(num_classes=num_classes, in_channels=in_channels), 'ResNet50LinearDFT'

        # if kernel.casefold() == 'cwt':
            # if layers.casefold() == 'all' or layers == None:
            #     return resnet50CWT(num_classes=num_classes, in_channels=in_channels), 'ResNet50CWT'
            # if layers.casefold() == 'conv':
            #     return resnet50ConvCWT(num_classes=num_classes, in_channels=in_channels), 'ResNet50ConvCWT'
            # if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
            #     return resnet50LinearCWT(num_classes=num_classes, in_channels=in_channels), 'ResNet50LinearCWT'

    def _densenet121(self, kernel, layers, num_classes, in_channels):
        if kernel == None:
            return densenet121(num_classes=num_classes, in_channels=in_channels), 'DenseNet121'

        if kernel.casefold() == 'dct':
            if layers.casefold() == 'all' or layers == None:
                return densenet121DCT(num_classes=num_classes,in_channels=in_channels), 'DenseNet121DCT'
            if layers.casefold() == 'conv':
                return densenet121ConvDCT(num_classes=num_classes, in_channels=in_channels), 'DenseNet121ConvDCT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return densenet121LinearDCT(num_classes=num_classes, in_channels=in_channels), 'DenseNet121LinearDCT'

        if kernel.casefold() == 'dst':
            if layers.casefold() == 'all' or layers == None:
                return densenet121DST(num_classes=num_classes,in_channels=in_channels), 'DenseNet121DST'
            if layers.casefold() == 'conv':
                return densenet121ConvDST(num_classes=num_classes, in_channels=in_channels), 'DenseNet121ConvDST'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return densenet121LinearDST(num_classes=num_classes, in_channels=in_channels), 'DenseNet121LinearDST'

        if kernel.casefold() == 'dft':
            if layers.casefold() == 'all' or layers == None:
                return densenet121DFT(num_classes=num_classes,in_channels=in_channels), 'DenseNet121DFT'
            if layers.casefold() == 'conv':
                return densenet121ConvDFT(num_classes=num_classes, in_channels=in_channels), 'DenseNet121ConvDFT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return densenet121LinearDFT(num_classes=num_classes, in_channels=in_channels), 'DenseNet121LinearDFT'

        
    def _densenet201(self, kernel, layers, num_classes, in_channels):
        if kernel == None:
            return densenet201(num_classes=num_classes, in_channels=in_channels), 'DenseNet201'

        if kernel.casefold() == 'dct':
            if layers.casefold() == 'all' or layers == None:
                return densenet201DCT(num_classes=num_classes,in_channels=in_channels), 'DenseNet201DCT'
            if layers.casefold() == 'conv':
                return densenet201ConvDCT(num_classes=num_classes, in_channels=in_channels), 'DenseNet201ConvDCT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return densenet201LinearDCT(num_classes=num_classes, in_channels=in_channels), 'DenseNet201LinearDCT'

        if kernel.casefold() == 'dst':
            if layers.casefold() == 'all' or layers == None:
                return densenet201DST(num_classes=num_classes,in_channels=in_channels), 'DenseNet201DST'
            if layers.casefold() == 'conv':
                return densenet201ConvDST(num_classes=num_classes, in_channels=in_channels), 'DenseNet201ConvDST'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return densenet201LinearDST(num_classes=num_classes, in_channels=in_channels), 'DenseNet201LinearDST'

        if kernel.casefold() == 'dft':
            if layers.casefold() == 'all' or layers == None:
                return densenet201DFT(num_classes=num_classes,in_channels=in_channels), 'DenseNet201DFT'
            if layers.casefold() == 'conv':
                return densenet201ConvDFT(num_classes=num_classes, in_channels=in_channels), 'DenseNet201ConvDFT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return densenet201LinearDFT(num_classes=num_classes, in_channels=in_channels), 'DenseNet201LinearDFT'

        
        

    def getModel(self,model,kernel:None,layers:None,num_classes:int,in_channels:int):

        if model.casefold() == 'alexnet':
            return self._alexnet(kernel,layers,num_classes, in_channels)

        if model.casefold() == 'lenet':
            return self._lenet(kernel,layers,num_classes, in_channels)

        if model.casefold() == 'resnet18':
            return self._resnet18(kernel,layers,num_classes, in_channels)

        if model.casefold() == 'resnet50':
            return self._resnet50(kernel,layers,num_classes, in_channels)

        if model.casefold() == 'densenet121':
            return self._densenet121(kernel,layers,num_classes, in_channels)

        if model.casefold() == 'densenet201':
            return self._densenet201(kernel,layers,num_classes, in_channels)
        