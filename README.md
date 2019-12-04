# Describe

Pruning neural network model via sketch.

# Pre-train Models

Additionally, we provide several  pre-trained models used in our experiments.

## CIFAR-10

| [VGG16](https://drive.google.com/open?id=1iqcLZyMTnciVLiKOHNaKbeXixK0KOzuX) | [ResNet56](https://drive.google.com/open?id=1pt-LgK3kI_4ViXIQWuOP0qmmQa3p2qW5) | [ResNet110]() | [DenseNet40]() | [GoogLeNet]() |

## ImageNet

| [VGG16](https://download.pytorch.org/models/vgg16_bn-6c64b313.pth) | 
|[ResNet18](https://download.pytorch.org/models/resnet18-5c106cde.pth) | [ResNet34](https://download.pytorch.org/models/resnet34-333f7ec4.pth) | [ResNet50](https://download.pytorch.org/models/resnet50-19c8e357.pth) | [ResNet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) | [ResNet152](https://download.pytorch.org/models/resnet152-b121ed2d.pth)|
|[GoogLeNet](https://download.pytorch.org/models/googlenet-1378be20.pth)|
|[DenseNet121](https://drive.google.com/open?id=1-ZZu8yGmh518F6621BvHwBZ7NV17wf-9)|[DenseNet161](https://drive.google.com/open?id=1lNWiyyeQKtsldO7iFNmQ11WLNUNH22Jr)|[DenseNet169](https://drive.google.com/open?id=10iScGCR4QY6ZkghATkEaa61-F8buW3fB)|[DenseNet201](https://drive.google.com/open?id=1DZytePACQJyXbgLX_KIUDJRHAerUo4OT)|

# Running Code

In this code, you can run our models on CIFAR-10 and ImageNet dataset. The code has been tested by Pytorch1.3 and CUDA10.0 on Ubuntu16.04.

## Single-shot Sketch

```shell
python sketch.py 
--data_set cifar10 
--data_path ../data/cifar10 
--sketch_model ./experiment/pretrain/resne56.pt 
--job_dir ./experiment/resnet56/sketch
--arch resnet 
--cfg resnet56 
--lr 0.01
--lr_decay_step 75 112
--num_epochs 150 
--gpus 0
--sketch_rate [0.5]*9+[0.6]*9+[0.4]*9
--start_conv 1
--weight_norm_method l2 
--filter_norm True
```
