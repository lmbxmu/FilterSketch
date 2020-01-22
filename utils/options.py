import argparse

parser = argparse.ArgumentParser(description='Filter Sketch')

parser.add_argument(
    '--gpus',
    type=int,
    nargs='+',
    default=[0],
    help='Select gpu_id to use. default:[0]',
)

parser.add_argument(
    '--data_set',
    type=str,
    default='cifar10',
    help='Select dataset to train. default:cifar10',
)

parser.add_argument(
    '--data_path',
    type=str,
    default='/home/lishaojie/data/cifar10/',
    help='The dictionary where the input is stored. default:/home/lishaojie/data/cifar10/',
)

parser.add_argument(
    '--job_dir',
    type=str,
    default='experiments/',
    help='The directory where the summaries will be stored. default:./experiments')

## Training
parser.add_argument(
    '--arch',
    type=str,
    default='resnet',
    help='Architecture of model. default:resnet')

parser.add_argument(
    '--cfg',
    type=str,
    default='resnet56',
    help='Detail architecuture of model. default:resnet56'
)

parser.add_argument(
    '--num_epochs',
    type=int,
    default=150,
    help='The num of epochs to train. default:150')

parser.add_argument(
    '--train_batch_size',
    type=int,
    default=128,
    help='Batch size for training. default:128')

parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=100,
    help='Batch size for validation. default:100')

parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='Momentum for MomentumOptimizer. default:0.9')

parser.add_argument(
    '--lr',
    type=float,
    default=1e-2,
    help='Learning rate for train. default:1e-2'
)

parser.add_argument(
    '--lr_decay_step',
    type=int,
    nargs='+',
    default=[50, 100],
    help='the iterval of learn rate. default:50, 100'
)

parser.add_argument(
    '--weight_decay',
    type=float,
    default=5e-4,
    help='The weight decay of loss. default:5e-4')

## Sketch
parser.add_argument(
    '--start_conv',
    type=int,
    default=1,
    help='The index of Conv to start sketch, index starts from 0. default:1'
)

parser.add_argument(
    '--sketch_rate',
    type=str,
    default=None,
    help='The proportion of each layer reserved after sketching convolution layer sketch. default:None'
)

parser.add_argument(
    '--sketch_model',
    type=str,
    default=None,
    help='Path to the model wait for sketch/test. default:None'
)

parser.add_argument(
    '--weight_norm_method',
    type=str,
    default=None,
    help='Select the weight norm method. default:None Optional:l2'
)

args = parser.parse_args()