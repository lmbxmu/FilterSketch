
import torch
import argparse
import get_flops
import utils.common as utils
from importlib import import_module

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument(
    '--input_image_size',
    type=int,
    default=32,
    help='The input_image_size')
parser.add_argument(
    '--arch',
    type=str,
    default='vgg16',
    choices=('vgg','resnet','densenet','googlenet'),
    help='The architecture to prune')
parser.add_argument(
    '--data_set',
    type=str,
    default='cifar10',)
parser.add_argument(
    '--cfg',
    type=str,
    default='resnet56'
)
parser.add_argument(
    '--sketch_rate',
    type=str,
    default=None,
    help='The num of cov to start prune')
args = parser.parse_args()

device = torch.device("cpu")

print('==> Building model..')
sketch_rate = utils.get_sketch_rate(args.sketch_rate)
if args.arch == 'vgg':
    model = import_module(f'model.{args.arch}').SketchVGG(sketch_rate, start_conv=1).to(device)
elif args.arch == 'resnet':
    if args.data_set == 'imagenet':
        model = import_module(f'model.{args.arch}_imagenet')\
                    .resnet(args.cfg, sketch_rate=sketch_rate, start_conv=1).to(device)
    else:
        model = import_module(f'model.{args.arch}')\
                    .resnet(args.cfg, sketch_rate=sketch_rate, start_conv=1).to(device)
elif args.arch == 'googlenet':
    model = import_module(f'model.{args.arch}').googlenet(sketch_rate).to(device)
elif args.arch == 'densenet':
    model = import_module(f'model.{args.arch}').densenet_cifar(sketch_rate).to(device)

if args.arch=='googlenet' or args.arch=='resnet_50':
    flops, params = get_flops.measure_model(model, device, 3, args.input_image_size, args.input_image_size, True)
else:
    flops, params= get_flops.measure_model(model, device, 3, args.input_image_size, args.input_image_size)

print('Params: %.2f'%(params))
print('Flops: %.2f'%(flops))

