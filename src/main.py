import sys
sys.path.append('.')

import torch



from models.DataModules.DataModules import Cifar10DataModule, Cifar100DataModule, MNISTDataModule, SVHNDataModule
from models.DataModules.DataModules_lenet import Cifar10DataModuleLenet, Cifar100DataModuleLenet, MNISTDataModuleLenet, SVHNDataModuleLenet

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.utilities.model_summary import ModelSummary
from argparse import ArgumentParser
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from ModelSelect import ModelSelect
from pytorch_lightning.loggers import MLFlowLogger

# mlflow.set_tracking_uri("/netscratch/pias/mlruns")

def mlflowExpt(model: str):
    if 'alexnet' in model.casefold():
        return "AlexNets"
    
    if 'lenet' in model.casefold():
         return "LeNets"
    
    if 'resnet18' in model.casefold():
         return "ResNet18s"

    if 'resnet50' in model.casefold():
         return "ResNet50s"

    if 'densenet121' in model.casefold():
        return "DenseNet121s"
    
    if 'densenet201' in model.casefold():
        return "DenseNet201s"

    if 'resnet18Ablation' in model.casefold():
         return "ResNet18Ablation"

    if 'resnet50Ablation' in model.casefold():
         return "ResNet50Ablation"

def main(inputs):
    
    if inputs.rep:
        pl.seed_everything(87, workers=True) ## for reproduciblilty

    ## data load
    if inputs.dataset.casefold() == 'cifar10':
        if 'lenet' in inputs.model.casefold(): 
            data = Cifar10DataModuleLenet(batch_size=int(inputs.batch_size))
        else:
            data = Cifar10DataModule(batch_size=int(inputs.batch_size))
        num_classes = 10
        in_channels = 3

    if inputs.dataset.casefold() == 'cifar100':
        if 'lenet' in inputs.model.casefold(): 
            data = Cifar100DataModuleLenet(batch_size=int(inputs.batch_size))
        else:
            data = Cifar100DataModule(batch_size=int(inputs.batch_size))
        num_classes = 100
        in_channels = 3

    if inputs.dataset.casefold() == 'mnist':
        if 'lenet' in inputs.model.casefold(): 
            data = MNISTDataModuleLenet(batch_size=int(inputs.batch_size))
        else:
            data = MNISTDataModule(batch_size=int(inputs.batch_size))
        num_classes = 10
        in_channels = 1

    if inputs.dataset.casefold() == 'svhn':
        if 'lenet' in inputs.model.casefold(): 
            data = SVHNDataModuleLenet(batch_size=int(inputs.batch_size))
        else:
            data = SVHNDataModule(batch_size=int(inputs.batch_size))
        num_classes = 10
        in_channels = 3
    
    ## model init
    modelSelect = ModelSelect()

    model , model_name = modelSelect.getModel(inputs.model, inputs.kernel, inputs.layers, num_classes, in_channels)
    print(model_name)

    expt_id = inputs.dataset + "_" + mlflowExpt(inputs.model)

    if torch.cuda.is_available():
        devices = inputs.devices
        torch.cuda.amp.autocast(enabled=False)
    else:
        devices = None

    # try:
    #     expt_id = mlflow.create_experiment(expt_id,artifact_location='/netscratch/pias/mlruns')
    # except mlflow.exceptions.MlflowException:
    #     expt = mlflow.get_experiment_by_name(expt_id)
    #     expt_id = expt.experiment_id
    
    # mlflow.pytorch.autolog()

    run_name = model_name

    mlf_logger = MLFlowLogger(experiment_name=expt_id, run_name=run_name, 
                                save_dir = "/netscratch/pias/mlruns" ,artifact_location='/netscratch/pias/mlruns')

    ## train and test
    # with mlflow.start_run(experiment_id = expt_id, run_name=run_name ) as run:
    if inputs.rep:
        trainer_det = pl.Trainer(accelerator="auto",
                                devices=devices, 
                                max_epochs=int(inputs.max_epochs),deterministic=True, logger=mlf_logger,
                                callbacks=[TQDMProgressBar(refresh_rate=20)],
                                ) ## for reproduciblilty
        
        
        trainer_det.fit(model=model, datamodule=data)
            
        ## test
        trainer_det.test(model=model, datamodule=data)

    else:
        trainer = pl.Trainer(accelerator="auto",
                            devices=devices, 
                            max_epochs=int(inputs.max_epochs),
                            logger=mlf_logger, callbacks=[TQDMProgressBar(refresh_rate=20)],
                            strategy="ddp")

        
        trainer.fit(model=model, datamodule=data)

        ## test
        trainer.test(model=model, datamodule=data)

    # state_dict = model.state_dict()
    # mlflow.pytorch.log_state_dict(state_dict, artifact_path='/netscratch/pias/mlruns/{}/{}/artifacts/model'.format(mlf_logger.experiment_id,
    #                                 mlf_logger.run_id))


    print(ModelSummary(model, max_depth=-1))

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("--model", default='lenet')
    parser.add_argument("--kernel", default=None)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--devices", default=1)
    parser.add_argument("--max_epochs", default=5)
    parser.add_argument("--rep", default=False) ## reproducible flag
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--batch_size', default=32)
    args = parser.parse_args()
    
    main(args)