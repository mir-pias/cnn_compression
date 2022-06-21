from asyncio.log import logger
import sys
sys.path.append('.')

import torch

from models.DataModules.DataModules import Cifar10DataModule, Cifar100DataModule

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from argparse import ArgumentParser
from utils.utils import model_select_DenseNet121

def main(inputs):
    
    if inputs.rep:
        pl.seed_everything(87, workers=True) ## for reproduciblilty

    ## data load
    if inputs.dataset == 'Cifar10' or inputs.dataset == 'cifar10':
        data = Cifar10DataModule()
        num_classes = 10
        
    
    if inputs.dataset == 'Cifar100' or inputs.dataset == 'cifar100':
        data = Cifar100DataModule()
        num_classes = 100
        

    ## model init
    model , model_name = model_select_DenseNet121(inputs.kernel, inputs.layers, num_classes)
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
                                logger = CSVLogger(f"lightning_logs/{inputs.dataset}/", name=model_name), deterministic=True) ## for reproduciblilty

        trainer_det.fit(model=model, datamodule=data)
            
        ## test
        trainer_det.test(model=model, datamodule=data)

    else:
        trainer = pl.Trainer(accelerator="auto",
                            devices=devices, 
                            max_epochs=inputs.max_epochs, callbacks=[TQDMProgressBar(refresh_rate=20)],
                            logger = CSVLogger(f"lightning_logs/{inputs.dataset}/", name=model_name),)
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
    parser.add_argument('--dataset', default='Cifar10')
    args = parser.parse_args()
    
    main(args)