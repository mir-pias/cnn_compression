# cnn_compression
DEADLINE : 01.09.2022

run example 

python src/main.py --model=alexnet --kernel=dct --layers=linear 

options

--model: default=LeNet || alexnet, resnet18

--kernel:  default=None || dct, dst, dft, None

--layers:  default=None, || linear|fc|dense, conv, all (None is same as 'all')

--devices: default=None

--max_epochs: default=5

--rep: default=False ## reproducible flag, to compare changes in runs. 

--dataset: default=cifar10 || cifar10, cifar100, mnist

-- batch_size: default=32



implemented arch: AlexNet, LeNet, ResNet18. dataset: CIFAR10, CIFAR100, MNIST

to do: 

implement everything for wavelets


implement archs: ResNet, DenseNet, TBD


try more datasets, ImageNet in DFKI cluster + TBD


figure out MLFlow logs + START WRITING



thesis plan


To do:
1. High-level SciPy prototype (if case you need a reference implementation)
2. Purely NumPy-implemented prototype
3. Prototype based on PyTorch (make sure that things are differentiable)
4. Fully functional layer implemented in PyTorch

experimental pipeline:
1. Data loading
2. Data split (training/validation/testing)
3. Model training (with logs and snapshots)
4. Model evaluation

thesis plan for now:
1) Implement a DCT-compressed FC-layer. Evaluate it on some small datasets (e.g., CIFAR-10/100, Caltech-UCSD Birds 200, etc.) by using compressed layers instead of regular ones. For example, AlexNet is good for this kind of experiments, since the majority of the parameters (58.6M out of 60.9M) belongs to the FC-part. However, it's also needed to quantify the effect on other architectures (e.g., ResNet, DenseNet, etc.). Optionally, consider implementing the layer in the way that you could replace kernels, so you'd be able to evaluate different ones.

2) Implement a DCT-compressed Conv2d-layer. Run the same evaluation procedure, but now the difference will be that only Conv2d-layers are compressed, while FC-ones are kept regular. Same optional point about the kernel replacement. Note about the 2d-DCT: the 2d-case can be separated into two 1d-cases; in other words, you take the DCT along the X-axis first, then you apply the same kernel to the obtained result along the Y-axis.

3) Finally, in the chosen models, replace all Conv2d- and FC-layers with the compressed counterparts and evaluate the performance following the same procedure. This way, you'll come up with an in-depth evaluation of the effect of the proposed layer compression approach on model's performance and size.
