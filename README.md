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

## Remarks

The number of pruning rates required for different networks is as follows:

|           | CIFAR-10 | ImageNet |
| :-------: | :------: | :------: |
|   VGG16   |    12    |    -     |
| ResNet56  |    27    |    -     |
| ResNet110 |    54    |    -     |
| GoogLeNet |    9     |    -     |
| DenseNet  |    -     |    -     |
| ResNet18  |    -     |    8     |
| ResNet34  |    -     |    16    |
| ResNet50  |    -     |    16    |
| ResNet101 |    -     |    33    |
| ResNet152 |    -     |    50    |



## Other Arguments

```shell
optional arguments:
  -h, --help            show this help message and exit
  --gpus GPUS [GPUS ...]
                        Select gpu_id to use. default:[0]
  --data_set DATA_SET   Select dataset to train. default:cifar10
  --data_path DATA_PATH
                        The dictionary where the input is stored.
                        default:/home/lishaojie/data/cifar10/
  --job_dir JOB_DIR     The directory where the summaries will be stored.
                        default:./experiments
  --reset               Reset the directory?
  --resume RESUME       Load the model from the specified checkpoint.
  --refine REFINE       Path to the model to be fine tuned.
  --arch ARCH           Architecture of model. default:vgg
  --cfg CFG             Detail architecuture of model. default:vgg16
  --num_epochs NUM_EPOCHS
                        The num of epochs to train. default:150
  --train_batch_size TRAIN_BATCH_SIZE
                        Batch size for training. default:128
  --eval_batch_size EVAL_BATCH_SIZE
                        Batch size for validation. default:100
  --momentum MOMENTUM   Momentum for MomentumOptimizer. default:0.9
  --lr LR               Learning rate for train. default:1e-2
  --lr_decay_step LR_DECAY_STEP [LR_DECAY_STEP ...]
                        the iterval of learn rate. default:50, 100
  --weight_decay WEIGHT_DECAY
                        The weight decay of loss. default:5e-4
  --start_conv START_CONV
                        The index of Conv to start sketch, index starts from
                        0. default:1
  --sketch_rate SKETCH_RATE
                        The rate of each sketch conv. default:None
  --sketch_model SKETCH_MODEL
                        Path to the model wait for sketch. default:None
  --sketch_bn SKETCH_BN
                        Whether the BN weights are sketched or not?
                        default:False
  --weight_norm_method WEIGHT_NORM_METHOD
                        Select the weight norm method. default:None
                        Optional:max,sum,l2,l1,l2_2,2max
  --filter_norm FILTER_NORM
                        Filter level normalization or not? default:False
  --sketch_lastconv SKETCH_LASTCONV
                        Is the last layer of convolution sketched?
                        default:True
  --random_rule RANDOM_RULE
                        Weight initialization criterion after random clipping.
                        default:default
                        optional:default,random_pretrain,l1_pretrain
  --test_only           Test only?

```

