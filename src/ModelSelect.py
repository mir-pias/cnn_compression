from models.LeNet.LeNet import LeNet
from models.LeNet.DFTLeNets import LeNetLinearDFT, LeNetConvDFT, LeNetDFT
from models.LeNet.DCTLeNets import LeNetLinearDCT, LeNetConvDCT, LeNetDCT
from models.LeNet.DSTLeNets import LeNetLinearDST, LeNetConvDST, LeNetDST
from models.LeNet.ShanLeNets import LeNetLinearRealShannon, LeNetConvRealShannon, LeNetRealShannon, LeNetLinearShannon, LeNetConvShannon, LeNetShannon

from models.AlexNet.DFTAlexNets import AlexNetLinearDFT, AlexNetDFT, AlexNetConvDFT
from models.AlexNet.AlexNet import AlexNet
from models.AlexNet.DCTAlexNets import AlexNetLinearDCT, AlexNetConvDCT, AlexNetDCT
from models.AlexNet.DSTAlexNets import AlexNetConvDST, AlexNetDST, AlexNetLinearDST
from models.AlexNet.ShanAlexNets import AlexNetLinearRealShannon, AlexNetConvRealShannon, AlexNetRealShannon, AlexNetConvShannon, AlexNetLinearShannon, AlexNetShannon

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
from models.ResNet.ShanResNets.ResNetConvRealShannon import resnet18ConvRealShannon, resnet50ConvRealShannon
from models.ResNet.ShanResNets.ResNetLinearRealShannon import resnet18LinearRealShannon, resnet50LinearRealShannon
from models.ResNet.ShanResNets.ResNetRealShannon import resnet18RealShannon, resnet50RealShannon
from models.ResNet.ShanResNets.ResNetConvShannon import resnet18ConvShannon, resnet50ConvShannon
from models.ResNet.ShanResNets.ResNetLinearShannon import resnet18LinearShannon, resnet50LinearShannon
from models.ResNet.ShanResNets.ResNetShannon import resnet18Shannon, resnet50Shannon


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
from models.DenseNet.ShanDenseNets.DenseNetConvRealShannon import densenet121ConvRealShannon, densenet201ConvRealShannon
from models.DenseNet.ShanDenseNets.DenseNetLinearRealShannon import densenet121LinearRealShannon, densenet201LinearRealShannon
from models.DenseNet.ShanDenseNets.DenseNetRealShannon import densenet121RealShannon, densenet201RealShannon
from models.DenseNet.ShanDenseNets.DenseNetConvShannon import densenet121ConvShannon, densenet201ConvShannon
from models.DenseNet.ShanDenseNets.DenseNetLinearShannon import densenet121LinearShannon, densenet201LinearShannon
from models.DenseNet.ShanDenseNets.DenseNetShannon import densenet121Shannon, densenet201Shannon

class ModelSelect():
    def __init__(self) -> None:
        pass

    def _alexnet(self, kernel, layers, num_classes, in_channels):
        if kernel == None:
            return AlexNet(num_classes=num_classes,in_channels=in_channels), 'AlexNet'

        if kernel.casefold() == 'dct':
            if layers == None or layers.casefold() == 'all':
                return AlexNetDCT(num_classes=num_classes,in_channels=in_channels), 'AlexNetDCT'
            if layers.casefold() == 'conv':
                return AlexNetConvDCT(num_classes=num_classes,in_channels=in_channels), 'AlexNetConvDCT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return AlexNetLinearDCT(num_classes=num_classes,in_channels=in_channels), 'AlexNetLinearDCT'

        if kernel.casefold() == 'dst':
            if layers == None or layers.casefold() == 'all':
                return AlexNetDST(num_classes=num_classes,in_channels=in_channels), 'AlexNetDST'
            if layers.casefold() == 'conv':
                return AlexNetConvDST(num_classes=num_classes,in_channels=in_channels), 'AlexNetConvDST'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return AlexNetLinearDST(num_classes=num_classes,in_channels=in_channels), 'AlexNetLinearDST'
        
        if kernel.casefold() == 'dft':
            if layers == None or layers.casefold() == 'all':
                return AlexNetDFT(num_classes=num_classes,in_channels=in_channels), 'AlexNetDFT'
            if layers.casefold() == 'conv':
                return AlexNetConvDFT(num_classes=num_classes,in_channels=in_channels), 'AlexNetConvDFT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return AlexNetLinearDFT(num_classes=num_classes,in_channels=in_channels), 'AlexNetLinearDFT'

        if kernel.casefold() == 'realshannon':
            if layers == None or layers.casefold() == 'all' :
                return AlexNetRealShannon(num_classes=num_classes,in_channels=in_channels), 'AlexNetRealShannon'
            if layers.casefold() == 'conv':
                return AlexNetConvRealShannon(num_classes=num_classes,in_channels=in_channels), 'AlexNetConvRealShannon'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return AlexNetLinearRealShannon(num_classes=num_classes,in_channels=in_channels), 'AlexNetLinearRealShannon'

        if kernel.casefold() == 'shannon':
            if layers == None or layers.casefold() == 'all' :
                return AlexNetShannon(num_classes=num_classes,in_channels=in_channels), 'AlexNetShannon'
            if layers.casefold() == 'conv':
                return AlexNetConvShannon(num_classes=num_classes,in_channels=in_channels), 'AlexNetConvShannon'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return AlexNetLinearShannon(num_classes=num_classes,in_channels=in_channels), 'AlexNetLinearShannon'

        

    def _lenet(self, kernel, layers, num_classes,in_channels):
        if kernel == None:
            return LeNet(num_classes=num_classes, in_channels=in_channels), 'LeNet'

        if kernel.casefold() == 'dct':
            if layers == None or layers.casefold() == 'all':
                return LeNetDCT(num_classes=num_classes, in_channels=in_channels), 'LeNetDCT'
            if layers.casefold() == 'conv':
                return LeNetConvDCT(num_classes=num_classes,in_channels=in_channels), 'LeNetConvDCT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return LeNetLinearDCT(num_classes=num_classes,in_channels=in_channels), 'LeNetLinearDCT'

        if kernel.casefold() == 'dst':
            if layers == None or layers.casefold() == 'all':
                return LeNetDST(num_classes=num_classes,in_channels=in_channels), 'LeNetDST'
            if layers.casefold() == 'conv':
                return LeNetConvDST(num_classes=num_classes,in_channels=in_channels), 'LeNetConvDST'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return LeNetLinearDST(num_classes=num_classes,in_channels=in_channels), 'LeNetLinearDST'
        
        if kernel.casefold() == 'dft':
            if layers == None or layers.casefold() == 'all':
                return LeNetDFT(num_classes=num_classes,in_channels=in_channels), 'LeNetDFT'
            if layers.casefold() == 'conv':
                return LeNetConvDFT(num_classes=num_classes,in_channels=in_channels), 'LeNetConvDFT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return LeNetLinearDFT(num_classes=num_classes,in_channels=in_channels), 'LeNetLinearDFT'

        if kernel.casefold() == 'realshannon':
            if layers == None or layers.casefold() == 'all':
                return LeNetRealShannon(num_classes=num_classes,in_channels=in_channels), 'LeNetRealShannon'
            if layers.casefold() == 'conv':
                return LeNetConvRealShannon(num_classes=num_classes,in_channels=in_channels), 'LeNetConvRealShannon'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return LeNetLinearRealShannon(num_classes=num_classes,in_channels=in_channels), 'LeNetLinearRealShannon'

        if kernel.casefold() == 'shannon':
            if layers == None or layers.casefold() == 'all':
                return LeNetShannon(num_classes=num_classes,in_channels=in_channels), 'LeNetShannon'
            if layers.casefold() == 'conv':
                return LeNetConvShannon(num_classes=num_classes,in_channels=in_channels), 'LeNetConvShannon'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return LeNetLinearShannon(num_classes=num_classes,in_channels=in_channels), 'LeNetLinearShannon'

        

    def _resnet18(self, kernel, layers, num_classes, in_channels):
        if kernel == None:
            return resnet18(num_classes=num_classes, in_channels=in_channels), 'ResNet18'

        if kernel.casefold() == 'dct':
            if layers == None or layers.casefold() == 'all' :
                return resnet18DCT(num_classes=num_classes,in_channels=in_channels), 'ResNet18DCT'
            if layers.casefold() == 'conv':
                return resnet18ConvDCT(num_classes=num_classes, in_channels=in_channels), 'ResNet18ConvDCT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return resnet18LinearDCT(num_classes=num_classes, in_channels=in_channels), 'ResNet18LinearDCT'

        if kernel.casefold() == 'dst':
            if layers == None or layers.casefold() == 'all':
                return resnet18DST(num_classes=num_classes, in_channels=in_channels), 'ResNet18DST'
            if layers.casefold() == 'conv':
                return resnet18ConvDST(num_classes=num_classes, in_channels=in_channels), 'ResNet18ConvDST'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return resnet18LinearDST(num_classes=num_classes, in_channels=in_channels), 'ResNet18LinearDST'
        
        if kernel.casefold() == 'dft':
            if layers == None or layers.casefold() == 'all':
                return resnet18DFT(num_classes=num_classes, in_channels=in_channels), 'ResNet18DFT'
            if layers.casefold() == 'conv':
                return resnet18ConvDFT(num_classes=num_classes, in_channels=in_channels), 'ResNet18ConvDFT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return resnet18LinearDFT(num_classes=num_classes, in_channels=in_channels), 'ResNet18LinearDFT'

        if kernel.casefold() == 'realshannon':
            if layers == None or layers.casefold() == 'all':
                return resnet18RealShannon(num_classes=num_classes, in_channels=in_channels), 'ResNet18RealShannon'
            if layers.casefold() == 'conv':
                return resnet18ConvRealShannon(num_classes=num_classes, in_channels=in_channels), 'ResNet18ConvRealShannon'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return resnet18LinearRealShannon(num_classes=num_classes, in_channels=in_channels), 'ResNet18LinearRealShannon'

        if kernel.casefold() == 'shannon':
            if layers == None or layers.casefold() == 'all' :
                return resnet18Shannon(num_classes=num_classes,in_channels=in_channels), 'ResNet18Shannon'
            if layers.casefold() == 'conv':
                return resnet18ConvShannon(num_classes=num_classes,in_channels=in_channels), 'ResNet18ConvShannon'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return resnet18LinearShannon(num_classes=num_classes,in_channels=in_channels), 'ResNet18LinearShannon'

    def _resnet50(self, kernel, layers, num_classes, in_channels):
        if kernel == None:
            return resnet50(num_classes=num_classes, in_channels=in_channels), 'ResNet50'

        if kernel.casefold() == 'dct':
            if layers == None or layers.casefold() == 'all':
                return resnet50DCT(num_classes=num_classes,in_channels=in_channels), 'ResNet50DCT'
            if layers.casefold() == 'conv':
                return resnet50ConvDCT(num_classes=num_classes, in_channels=in_channels), 'ResNet50ConvDCT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return resnet50LinearDCT(num_classes=num_classes, in_channels=in_channels), 'ResNet50LinearDCT'

        if kernel.casefold() == 'dst':
            if layers == None or layers.casefold() == 'all':
                return resnet50DST(num_classes=num_classes, in_channels=in_channels), 'ResNet50DST'
            if layers.casefold() == 'conv':
                return resnet50ConvDST(num_classes=num_classes, in_channels=in_channels), 'ResNet50ConvDST'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return resnet50LinearDST(num_classes=num_classes, in_channels=in_channels), 'ResNet50LinearDST'
        
        if kernel.casefold() == 'dft':
            if layers == None or layers.casefold() == 'all':
                return resnet50DFT(num_classes=num_classes, in_channels=in_channels), 'ResNet50DFT'
            if layers.casefold() == 'conv':
                return resnet50ConvDFT(num_classes=num_classes, in_channels=in_channels), 'ResNet50ConvDFT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return resnet50LinearDFT(num_classes=num_classes, in_channels=in_channels), 'ResNet50LinearDFT'

        if kernel.casefold() == 'realshannon':
            if layers == None or layers.casefold() == 'all':
                return resnet50RealShannon(num_classes=num_classes, in_channels=in_channels), 'ResNet50RealShannon'
            if layers.casefold() == 'conv':
                return resnet50ConvRealShannon(num_classes=num_classes, in_channels=in_channels), 'ResNet50ConvRealShannon'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return resnet50LinearRealShannon(num_classes=num_classes, in_channels=in_channels), 'ResNet50LinearRealShannon'

        if kernel.casefold() == 'shannon':
            if layers == None or layers.casefold() == 'all':
                return resnet50Shannon(num_classes=num_classes,in_channels=in_channels), 'ResNet50Shannon'
            if layers.casefold() == 'conv':
                return resnet50ConvShannon(num_classes=num_classes,in_channels=in_channels), 'ResNet50ConvShannon'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return resnet50LinearShannon(num_classes=num_classes,in_channels=in_channels), 'ResNet50LinearShannon'

    def _densenet121(self, kernel, layers, num_classes, in_channels):
        if kernel == None:
            return densenet121(num_classes=num_classes, in_channels=in_channels), 'DenseNet121'

        if kernel.casefold() == 'dct':
            if layers == None or layers.casefold() == 'all' :
                return densenet121DCT(num_classes=num_classes,in_channels=in_channels), 'DenseNet121DCT'
            if layers.casefold() == 'conv':
                return densenet121ConvDCT(num_classes=num_classes, in_channels=in_channels), 'DenseNet121ConvDCT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return densenet121LinearDCT(num_classes=num_classes, in_channels=in_channels), 'DenseNet121LinearDCT'

        if kernel.casefold() == 'dst':
            if layers == None or layers.casefold() == 'all' :
                return densenet121DST(num_classes=num_classes,in_channels=in_channels), 'DenseNet121DST'
            if layers.casefold() == 'conv':
                return densenet121ConvDST(num_classes=num_classes, in_channels=in_channels), 'DenseNet121ConvDST'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return densenet121LinearDST(num_classes=num_classes, in_channels=in_channels), 'DenseNet121LinearDST'

        if kernel.casefold() == 'dft':
            if layers == None or layers.casefold() == 'all':
                return densenet121DFT(num_classes=num_classes,in_channels=in_channels), 'DenseNet121DFT'
            if layers.casefold() == 'conv':
                return densenet121ConvDFT(num_classes=num_classes, in_channels=in_channels), 'DenseNet121ConvDFT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return densenet121LinearDFT(num_classes=num_classes, in_channels=in_channels), 'DenseNet121LinearDFT'

        if kernel.casefold() == 'realshannon':
            if layers == None or layers.casefold() == 'all' :
                return densenet121RealShannon(num_classes=num_classes, in_channels=in_channels), 'DenseNet121RealShannon'
            if layers.casefold() == 'conv':
                return densenet121ConvRealShannon(num_classes=num_classes, in_channels=in_channels), 'DenseNet121ConvRealShannon'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return densenet121LinearRealShannon(num_classes=num_classes, in_channels=in_channels), 'DenseNet121LinearRealShannon'

        if kernel.casefold() == 'shannon':
            if layers == None or layers.casefold() == 'all':
                return densenet121Shannon(num_classes=num_classes,in_channels=in_channels), 'DenseNet121Shannon'
            if layers.casefold() == 'conv':
                return densenet121ConvShannon(num_classes=num_classes,in_channels=in_channels), 'DenseNet121ConvShannon'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return densenet121LinearShannon(num_classes=num_classes,in_channels=in_channels), 'DenseNet121LinearShannon'

        
    def _densenet201(self, kernel, layers, num_classes, in_channels):
        if kernel == None:
            return densenet201(num_classes=num_classes, in_channels=in_channels), 'DenseNet201'

        if kernel.casefold() == 'dct':
            if layers == None or layers.casefold() == 'all':
                return densenet201DCT(num_classes=num_classes,in_channels=in_channels), 'DenseNet201DCT'
            if layers.casefold() == 'conv':
                return densenet201ConvDCT(num_classes=num_classes, in_channels=in_channels), 'DenseNet201ConvDCT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return densenet201LinearDCT(num_classes=num_classes, in_channels=in_channels), 'DenseNet201LinearDCT'

        if kernel.casefold() == 'dst':
            if layers == None or layers.casefold() == 'all' :
                return densenet201DST(num_classes=num_classes,in_channels=in_channels), 'DenseNet201DST'
            if layers.casefold() == 'conv':
                return densenet201ConvDST(num_classes=num_classes, in_channels=in_channels), 'DenseNet201ConvDST'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return densenet201LinearDST(num_classes=num_classes, in_channels=in_channels), 'DenseNet201LinearDST'

        if kernel.casefold() == 'dft':
            if layers == None or layers.casefold() == 'all' :
                return densenet201DFT(num_classes=num_classes,in_channels=in_channels), 'DenseNet201DFT'
            if layers.casefold() == 'conv':
                return densenet201ConvDFT(num_classes=num_classes, in_channels=in_channels), 'DenseNet201ConvDFT'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return densenet201LinearDFT(num_classes=num_classes, in_channels=in_channels), 'DenseNet201LinearDFT'

        if kernel.casefold() == 'realshannon':
            if layers == None or layers.casefold() == 'all' :
                return densenet201RealShannon(num_classes=num_classes, in_channels=in_channels), 'DenseNet201RealShannon'
            if layers.casefold() == 'conv':
                return densenet201ConvRealShannon(num_classes=num_classes, in_channels=in_channels), 'DenseNet201ConvRealShannon'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return densenet201LinearRealShannon(num_classes=num_classes, in_channels=in_channels), 'DenseNet201LinearRealShannon'

        if kernel.casefold() == 'shannon':
            if layers == None or layers.casefold() == 'all' :
                return densenet201Shannon(num_classes=num_classes,in_channels=in_channels), 'DenseNet201Shannon'
            if layers.casefold() == 'conv':
                return densenet201ConvShannon(num_classes=num_classes,in_channels=in_channels), 'DenseNet201ConvShannon'
            if layers.casefold() == 'fc' or layers.casefold() == 'linear' or layers.casefold() == 'dense':
                return densenet201LinearShannon(num_classes=num_classes,in_channels=in_channels), 'DenseNet201LinearShannon'
        
        

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
        