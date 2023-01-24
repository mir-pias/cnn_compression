#!/bin/bash
python src/main.py --model=LeNet --dataset='mnist' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dct' --layers='fc' --dataset='mnist' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dct' --layers='conv' --dataset='mnist' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dct' --layers='all' --dataset='mnist' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dst' --layers='fc' --dataset='mnist' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dst' --layers='conv' --dataset='mnist' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dst' --layers='all'  --dataset='mnist' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dft' --layers='fc' --dataset='mnist' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dft' --layers='conv' --dataset='mnist' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dft' --layers='all' --dataset='mnist' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='realshannon' --layers='fc' --dataset='mnist' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='realshannon' --layers='conv' --dataset='mnist' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='realshannon' --layers='all' --dataset='mnist' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='shannon' --layers='fc' --dataset='mnist' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='shannon' --layers='conv' --dataset='mnist' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='shannon' --layers='all' --dataset='mnist' --max_epochs=30 --batch_size=64


python src/main.py --model=LeNet --dataset='cifar10' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dct' --layers='fc' --dataset='cifar10' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dct' --layers='conv' --dataset='cifar10' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dct' --layers='all' --dataset='cifar10' --max_epochs=30 --batch_size=64
 
python src/main.py --model=LeNet --kernel='dst' --layers='fc' --dataset='cifar10' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dst' --layers='conv' --dataset='cifar10' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dst' --layers='all'  --dataset='cifar10' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dft' --layers='fc' --dataset='cifar10' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dft' --layers='conv' --dataset='cifar10' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dft' --layers='all' --dataset='cifar10' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='realshannon' --layers='fc' --dataset='cifar10' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='realshannon' --layers='conv' --dataset='cifar10' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='realshannon' --layers='all' --dataset='cifar10' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='shannon' --layers='fc' --dataset='cifar10' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='shannon' --layers='conv' --dataset='cifar10' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='shannon' --layers='all' --dataset='cifar10' --max_epochs=30 --batch_size=64


python src/main.py --model=LeNet --dataset='cifar100' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dct' --layers='fc' --dataset='cifar100' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dct' --layers='conv' --dataset='cifar100' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dct' --layers='all' --dataset='cifar100' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dst' --layers='fc' --dataset='cifar100' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dst' --layers='conv' --dataset='cifar100' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dst' --layers='all'  --dataset='cifar100' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dft' --layers='fc' --dataset='cifar100' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dft' --layers='conv' --dataset='cifar100' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='dft' --layers='all' --dataset='cifar100' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='realshannon' --layers='fc' --dataset='cifar100' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='realshannon' --layers='conv' --dataset='cifar100' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='realshannon' --layers='all' --dataset='cifar100' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='shannon' --layers='fc' --dataset='cifar100' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='shannon' --layers='conv' --dataset='cifar100' --max_epochs=30 --batch_size=64

python src/main.py --model=LeNet --kernel='shannon' --layers='all' --dataset='cifar100' --max_epochs=30 --batch_size=64

 