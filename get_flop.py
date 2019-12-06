import torch
import argparse
import utils.common as utils
from importlib import import_module
from thop import profile

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
    orimodel = import_module(f'model.{args.arch}').VGG().to(device)
    model = import_module(f'model.{args.arch}').SketchVGG(sketch_rate, start_conv=1).to(device)
elif args.arch == 'resnet':
    if args.data_set == 'imagenet':
        orimodel = import_module(f'model.{args.arch}_imagenet')\
                    .resnet(args.cfg).to(device)
        model = import_module(f'model.{args.arch}_imagenet')\
                    .resnet(args.cfg, sketch_rate=sketch_rate, start_conv=1).to(device)
    else:
        orimodel = import_module(f'model.{args.arch}').resnet(args.cfg).to(device)
        model = import_module(f'model.{args.arch}')\
                    .resnet(args.cfg, sketch_rate=sketch_rate, start_conv=1).to(device)
elif args.arch == 'googlenet':
    orimodel = import_module(f'model.{args.arch}').googlenet().to(device)
    model = import_module(f'model.{args.arch}').googlenet(sketch_rate).to(device)
elif args.arch == 'densenet':
    orimodel = import_module(f'model.{args.arch}').densenet_cifar().to(device)
    model = import_module(f'model.{args.arch}').densenet_cifar(sketch_rate).to(device)
else:
    raise ('arch not exist!')

input = torch.randn(1, 3, args.input_image_size, args.input_image_size)

print('--------------UnPrune Model--------------')
oriflops, oriparams = profile(orimodel, inputs=(input, ))
print('Params: %.2f'%(oriparams))
print('FLOPS: %.2f'%(oriflops))

print('--------------Prune Model--------------')
flops, params = profile(model, inputs=(input, ))
print('Params: %.2f'%(params))
print('FLOPS: %.2f'%(flops))

print('--------------Compress Rate--------------')
print('Params Compress Rate: %d/%d (%.2f%%)' % (params, oriparams, 100. * params / oriparams))
print('FLOPS Compress Rate: %d/%d (%.2f%%)' % (flops, oriflops, 100. * flops / oriflops))


