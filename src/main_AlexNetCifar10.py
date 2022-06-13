from asyncio.log import logger
import sys
sys.path.append('.')

import torch

from models.AlexNetCifar10.DFTAlexNets import AlexNetLinearDFT, AlexNetDFT, AlexNetConvDFT
from models.AlexNetCifar10.AlexNet import AlexNet
from models.AlexNetCifar10.DCTAlexNets import AlexNetLinearDCT, AlexNetConvDCT, AlexNetDCT
from models.AlexNetCifar10.DSTAlexNets import AlexNetConvDST, AlexNetDST, AlexNetLinearDST
from models.DataModules.Cifar10DataModule import Cifar10DataModule

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from argparse import ArgumentParser

def model_select(kernel, layers):

    if kernel == None:
        return AlexNet(), 'AlexNet'

    if kernel == 'DCT' or kernel == 'dct':
        if layers == 'all' or layers == 'All' or layers == None:
            return AlexNetDCT(), 'AlexNetDCT'
        if layers == 'conv' or layers == 'Conv':
            return AlexNetConvDCT(), 'AlexNetConvDCT'
        if layers == 'Linear' or layers == 'linear' or layers == 'fc' or layers =='FC':
            return AlexNetLinearDCT(), 'AlexNetLinearDCT'

    if kernel == 'DST' or kernel == 'dst':
        if layers == 'all' or layers == 'All' or layers == None:
            return AlexNetDST(), 'AlexNetDST'
        if layers == 'conv' or layers == 'Conv':
            return AlexNetConvDST(), 'AlexNetConvDST'
        if layers == 'Linear' or layers == 'linear' or layers == 'fc' or layers == 'FC':
            return AlexNetLinearDST(), 'AlexNetLinearDST'

    if kernel == 'DFT' or kernel == 'dft':
        if layers == 'all' or layers == 'All' or layers == None:
            return AlexNetDFT() , 'AlexNetDFT'
        if layers == 'conv' or layers == 'Conv':
            return AlexNetConvDFT(), 'AlexNetConvDFT'
        if layers == 'Linear' or layers == 'linear' or layers == 'fc' or layers == 'FC':
            return AlexNetLinearDFT(), 'AlexNetLinearDFT'
    

def main(inputs):
    
    if inputs.rep:
        pl.seed_everything(87, workers=True) ## for reproduciblilty

    ## data load
    data = Cifar10DataModule()

    ## model init
    model , model_name = model_select(inputs.kernel, inputs.layers)
    print(model)


    if torch.cuda.is_available():
        if inputs.devices == None:
            devices = 1
        else:
            devices = inputs.devices
    else:
        devices = None

    ## train and test
    if inputs.rep:
        trainer_det = pl.Trainer(accelerator="auto",
                                devices=devices, 
                                max_epochs=inputs.max_epochs, callbacks=[TQDMProgressBar(refresh_rate=20)],
                                logger = CSVLogger("lightning_logs/cifar10/", name=model_name), deterministic=True) ## for reproduciblilty

        trainer_det.fit(model=model, datamodule=data)
            
        ## test
        trainer_det.test(model=model, datamodule=data)

    else:
        trainer = pl.Trainer(accelerator="auto",
                            devices=devices, 
                            max_epochs=inputs.max_epochs, callbacks=[TQDMProgressBar(refresh_rate=20)],
                            logger = CSVLogger("lightning_logs/cifar10/", name=model_name),)
        trainer.fit(model=model, datamodule=data)

            
        ## test
        trainer.test(model=model, datamodule=data)

    print(ModelSummary(model, max_depth=-1))

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("--kernel", default=None)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--max_epochs", default=5)
    parser.add_argument("--rep", default=False) ## reproducible flag
    args = parser.parse_args()
    
    main(args)