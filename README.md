# cnn_compression

usage: install requirements.txt, make necessary changes in DataModules.py and DataModules_lenet.py to read the datasets, run with the desired arguments. Models are built using lightning module and training runs are logged using MLFlow, kindly refer to MLFlow and pytorch-lightling documentations. 

https://lightning.ai/docs/pytorch/stable/common/lightning_module.html

https://lightning.ai/docs/pytorch/stable/common/trainer.html

https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.mlflow.html#module-lightning.pytorch.loggers.mlflow

run example 

python src/main.py --model=alexnet --kernel=dct --layers=linear --devices=4 --max_epochs=30 --dataset=mnist batch_size=64 nodes=1

arguments

--model: default=LeNet || alexnet, resnet18, resnet50, densenet121, densenet201

--kernel:  default=None || dct, dst, dft, realshannon, shannon

--layers:  default=None, || linear|fc|dense, conv, all (None is same as 'all')

--devices: default=1

--max_epochs: default=5

--rep: default=False ## reproducible flag, to compare changes in runs. 

--dataset: default=cifar10 || cifar10, cifar100, mnist, svhn

--batch_size: default=32

--nodes: default=1

implemented models: AlexNet, LeNet, ResNet18, ResNet50, DenseNet121, DenseNet201. 
datasets: CIFAR10, CIFAR100, MNIST, SVHN


Hyperparameters:

optimizer: SGDM, momentum = 0.9
learning_rate=1e-3
loss function: cross_entropy
train_val split: 90% train, 10% validation

Currently, changes in hyperparameters (except for train_val split, specified in DataModules.py and DataModules_lenet.py) have to be made for each model individually

